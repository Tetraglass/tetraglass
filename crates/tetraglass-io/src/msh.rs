//! Pure-Rust Gmsh `.msh` (MSH 4.1 and MSH 2.2, ASCII and binary) reader (`M1.6`).
//!
//! Parses `$MeshFormat`, `$Nodes`, `$Elements`, `$NodeData`, and `$ElementData` from a byte
//! slice (`&[u8]`) and permutes higher-order element connectivity from Gmsh's node ordering
//! into VTK's canonical node ordering (`D-009`, `D-018`).
//!
//! Reference: <https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format> and
//! <https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering>.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use tetraglass_core::{CellType, ComponentCount, DataArray, UnstructuredGrid, try_u32_from_usize};

use crate::error::IoError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MshVersion {
    V22,
    V41,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByteOrder {
    LittleEndian,
    BigEndian,
}

/// Specification of a Gmsh element type and its node permutation into VTK ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GmshElementSpec {
    /// Target VTK-ordered [`CellType`] in `tetraglass-core`.
    pub cell_type: CellType,
    /// Permutation table `perm` of length `num_nodes`, where `perm[vtk_slot]` is the 0-based
    /// index of the corresponding node in the Gmsh element's node list.
    pub gmsh_index_for_vtk_slot: &'static [usize],
}

// Permutation tables from Gmsh node ordering to VTK canonical node ordering.
// Reference: https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering

const PERM_IDENTITY_1: [usize; 1] = [0];
const PERM_IDENTITY_2: [usize; 2] = [0, 1];
const PERM_IDENTITY_3: [usize; 3] = [0, 1, 2];
const PERM_IDENTITY_4: [usize; 4] = [0, 1, 2, 3];
const PERM_IDENTITY_5: [usize; 5] = [0, 1, 2, 3, 4];
const PERM_IDENTITY_6: [usize; 6] = [0, 1, 2, 3, 4, 5];
const PERM_IDENTITY_8: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
const PERM_IDENTITY_9: [usize; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];

// Gmsh 10-node quadratic tetrahedron (MSH_TET_10, type 11):
// Gmsh mid-edge nodes 4..10 are on edges (0,1), (1,2), (2,0), (0,3), (2,3), (1,3).
// VTK_QUADRATIC_TETRA (type 24) mid-edge nodes 4..10 are on edges (0,1), (1,2), (2,0), (0,3), (1,3), (2,3).
// Nodes 8 and 9 are swapped.
// Reference: https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
const PERM_TET_10: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8];

// Gmsh 20-node serendipity quadratic hexahedron (MSH_HEX_20, type 17):
// Gmsh mid-edges 8..20: 8=(0,1), 9=(0,3), 10=(0,4), 11=(1,2), 12=(1,5), 13=(2,3),
//                       14=(2,6), 15=(3,7), 16=(4,5), 17=(4,7), 18=(5,6), 19=(6,7).
// VTK_QUADRATIC_HEXAHEDRON (type 25) mid-edges 8..20:
//   bottom quad: (0,1)=8, (1,2)=11, (2,3)=13, (3,0)=9
//   top quad:    (4,5)=16, (5,6)=18, (6,7)=19, (7,4)=17
//   pillars:     (0,4)=10, (1,5)=12, (2,6)=14, (3,7)=15
// Reference: https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
const PERM_HEX_20: [usize; 20] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15,
];

// Gmsh 27-node triquadratic / Lagrange order-2 hexahedron (MSH_HEX_27, type 12):
// First 20 nodes match MSH_HEX_20; face centers in Gmsh are 20=(z=-1), 21=(y=-1), 22=(x=-1),
// 23=(x=+1), 24=(y=+1), 25=(z=+1), and volume center is 26.
// VTK orders face centers as (-x)=22, (+x)=23, (-y)=21, (+y)=24, (-z)=20, (+z)=25, volume=26.
// Reference: https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
const PERM_HEX_27: [usize; 27] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15, 22, 23, 21, 24, 20, 25,
    26,
];

// Gmsh 15-node serendipity quadratic wedge / prism (MSH_PRI_15, type 18):
// Gmsh mid-edges 6..15: 6=(0,1), 7=(0,2), 8=(0,3), 9=(1,2), 10=(1,4), 11=(2,5),
//                       12=(3,4), 13=(3,5), 14=(4,5).
// VTK_QUADRATIC_WEDGE (type 26) mid-edges 6..15:
//   bottom tri: (0,1)=6, (1,2)=9, (2,0)=7
//   top tri:    (3,4)=12, (4,5)=14, (5,3)=13
//   pillars:    (0,3)=8, (1,4)=10, (2,5)=11
// Reference: https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
const PERM_PRI_15: [usize; 15] = [0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11];

// Gmsh 18-node biquadratic / Lagrange order-2 wedge / prism (MSH_PRI_18, type 13):
// First 15 nodes match MSH_PRI_15; quad face centers in Gmsh are 15=(0,1,4,3), 16=(0,2,5,3),
// 17=(1,2,5,4). VTK orders quad face centers as (0,1,4,3)=15, (1,2,5,4)=17, (2,0,3,5)=16.
// Reference: https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
const PERM_PRI_18: [usize; 18] = [0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11, 15, 17, 16];

// Gmsh 13-node serendipity quadratic pyramid (MSH_PYR_13, type 19):
// Gmsh mid-edges 5..13: 5=(0,1), 6=(0,3), 7=(0,4), 8=(1,2), 9=(1,4), 10=(2,3), 11=(2,4), 12=(3,4).
// VTK_QUADRATIC_PYRAMID (type 27) mid-edges 5..13:
//   base quad: (0,1)=5, (1,2)=8, (2,3)=10, (3,0)=6
//   apex:      (0,4)=7, (1,4)=9, (2,4)=11, (3,4)=12
// Reference: https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
const PERM_PYR_13: [usize; 13] = [0, 1, 2, 3, 4, 5, 8, 10, 6, 7, 9, 11, 12];

// Gmsh 4-node cubic line (MSH_LIN_4, type 26):
// Corners 0, 1, interior nodes 2, 3 along (0 -> 1). Matches VTK_LAGRANGE_CURVE order 3.
// Reference: https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
const PERM_LIN_4: [usize; 4] = [0, 1, 2, 3];

// Gmsh 10-node cubic triangle (MSH_TRI_10, type 21):
// Corners 0,1,2; edge nodes 3,4 on (0,1), 5,6 on (1,2), 7,8 on (2,0); interior node 9.
// Matches VTK_LAGRANGE_TRIANGLE order 3.
// Reference: https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
const PERM_TRI_10: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/// Maps a Gmsh numeric element type to its [`CellType`] and VTK node permutation table.
#[must_use]
pub const fn gmsh_element_spec(gmsh_elm_type: i32) -> Option<GmshElementSpec> {
    match gmsh_elm_type {
        // Linear elements (order 1)
        15 => Some(GmshElementSpec {
            cell_type: CellType::Vertex,
            gmsh_index_for_vtk_slot: &PERM_IDENTITY_1,
        }),
        1 => Some(GmshElementSpec {
            cell_type: CellType::Line,
            gmsh_index_for_vtk_slot: &PERM_IDENTITY_2,
        }),
        2 => Some(GmshElementSpec {
            cell_type: CellType::Triangle,
            gmsh_index_for_vtk_slot: &PERM_IDENTITY_3,
        }),
        3 => Some(GmshElementSpec {
            cell_type: CellType::Quad,
            gmsh_index_for_vtk_slot: &PERM_IDENTITY_4,
        }),
        4 => Some(GmshElementSpec {
            cell_type: CellType::Tetra,
            gmsh_index_for_vtk_slot: &PERM_IDENTITY_4,
        }),
        5 => Some(GmshElementSpec {
            cell_type: CellType::Hexahedron,
            gmsh_index_for_vtk_slot: &PERM_IDENTITY_8,
        }),
        6 => Some(GmshElementSpec {
            cell_type: CellType::Wedge,
            gmsh_index_for_vtk_slot: &PERM_IDENTITY_6,
        }),
        7 => Some(GmshElementSpec {
            cell_type: CellType::Pyramid,
            gmsh_index_for_vtk_slot: &PERM_IDENTITY_5,
        }),

        // Quadratic serendipity and complete Lagrange order-2 elements
        8 => Some(GmshElementSpec {
            cell_type: CellType::QuadraticEdge,
            gmsh_index_for_vtk_slot: &PERM_IDENTITY_3,
        }),
        9 => Some(GmshElementSpec {
            cell_type: CellType::QuadraticTriangle,
            gmsh_index_for_vtk_slot: &PERM_IDENTITY_6,
        }),
        10 => Some(GmshElementSpec {
            cell_type: CellType::LagrangeQuadrilateral { order: 2 },
            gmsh_index_for_vtk_slot: &PERM_IDENTITY_9,
        }),
        11 => Some(GmshElementSpec {
            cell_type: CellType::QuadraticTetra,
            gmsh_index_for_vtk_slot: &PERM_TET_10,
        }),
        12 => Some(GmshElementSpec {
            cell_type: CellType::LagrangeHexahedron { order: 2 },
            gmsh_index_for_vtk_slot: &PERM_HEX_27,
        }),
        13 => Some(GmshElementSpec {
            cell_type: CellType::LagrangeWedge { order: 2 },
            gmsh_index_for_vtk_slot: &PERM_PRI_18,
        }),
        16 => Some(GmshElementSpec {
            cell_type: CellType::QuadraticQuad,
            gmsh_index_for_vtk_slot: &PERM_IDENTITY_8,
        }),
        17 => Some(GmshElementSpec {
            cell_type: CellType::QuadraticHexahedron,
            gmsh_index_for_vtk_slot: &PERM_HEX_20,
        }),
        18 => Some(GmshElementSpec {
            cell_type: CellType::QuadraticWedge,
            gmsh_index_for_vtk_slot: &PERM_PRI_15,
        }),
        19 => Some(GmshElementSpec {
            cell_type: CellType::QuadraticPyramid,
            gmsh_index_for_vtk_slot: &PERM_PYR_13,
        }),

        // Order-3 Lagrange curve and triangle
        26 => Some(GmshElementSpec {
            cell_type: CellType::LagrangeCurve { order: 3 },
            gmsh_index_for_vtk_slot: &PERM_LIN_4,
        }),
        21 => Some(GmshElementSpec {
            cell_type: CellType::LagrangeTriangle { order: 3 },
            gmsh_index_for_vtk_slot: &PERM_TRI_10,
        }),
        _ => None,
    }
}

/// Reads a Gmsh `.msh` file (version 4.1 or 2.2, ASCII or binary) from a native file path.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_msh_from_path(path: impl AsRef<Path>) -> Result<UnstructuredGrid, IoError> {
    let bytes = std::fs::read(path)?;
    read_msh(&bytes)
}

/// Parses a Gmsh `.msh` file (version 4.1 or 2.2, ASCII or binary) from a byte slice into an
/// [`UnstructuredGrid`].
pub fn read_msh(bytes: &[u8]) -> Result<UnstructuredGrid, IoError> {
    let mut reader = MshCursor::new(bytes);

    let mut format_info: Option<MeshFormatHeader> = None;
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut node_tag_to_idx: HashMap<u64, u32> = HashMap::new();
    let mut cell_types: Vec<CellType> = Vec::new();
    let mut offsets: Vec<u32> = Vec::new();
    let mut connectivity: Vec<u32> = Vec::new();
    let mut elem_tag_to_idx: HashMap<u64, usize> = HashMap::new();
    let mut point_data_arrays: Vec<(String, DataArray)> = Vec::new();
    let mut cell_data_arrays: Vec<(String, DataArray)> = Vec::new();

    let mut saw_nodes = false;
    let mut saw_elements = false;

    while let Some((line_offset, raw_line)) = reader.read_line_opt()? {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if !line.starts_with('$') {
            return Err(IoError::InvalidMshStructure {
                offset: line_offset as u64,
                reason: format!("expected section header starting with `$`, got `{line}`"),
            });
        }

        match line {
            "$MeshFormat" => {
                let hdr = parse_mesh_format(&mut reader)?;
                format_info = Some(hdr);
            }
            "$Nodes" => {
                let hdr = format_info.ok_or(IoError::InvalidMshStructure {
                    offset: line_offset as u64,
                    reason: "`$Nodes` appeared before `$MeshFormat`".to_string(),
                })?;
                parse_nodes(&mut reader, hdr, &mut points, &mut node_tag_to_idx)?;
                saw_nodes = true;
            }
            "$Elements" => {
                let hdr = format_info.ok_or(IoError::InvalidMshStructure {
                    offset: line_offset as u64,
                    reason: "`$Elements` appeared before `$MeshFormat`".to_string(),
                })?;
                if !saw_nodes {
                    return Err(IoError::InvalidMshStructure {
                        offset: line_offset as u64,
                        reason: "`$Elements` appeared before `$Nodes`".to_string(),
                    });
                }
                parse_elements(
                    &mut reader,
                    hdr,
                    &node_tag_to_idx,
                    &mut cell_types,
                    &mut offsets,
                    &mut connectivity,
                    &mut elem_tag_to_idx,
                )?;
                saw_elements = true;
            }
            "$NodeData" => {
                let hdr = format_info.ok_or(IoError::InvalidMshStructure {
                    offset: line_offset as u64,
                    reason: "`$NodeData` appeared before `$MeshFormat`".to_string(),
                })?;
                if !saw_nodes {
                    return Err(IoError::InvalidMshStructure {
                        offset: line_offset as u64,
                        reason: "`$NodeData` appeared before `$Nodes`".to_string(),
                    });
                }
                let (name, array) =
                    parse_model_data(&mut reader, hdr, "$EndNodeData", points.len(), |tag| {
                        node_tag_to_idx.get(&tag).map(|&idx| idx as usize)
                    })?;
                if !point_data_arrays.iter().any(|(n, _)| n == &name) {
                    point_data_arrays.push((name, array));
                }
            }
            "$ElementData" => {
                let hdr = format_info.ok_or(IoError::InvalidMshStructure {
                    offset: line_offset as u64,
                    reason: "`$ElementData` appeared before `$MeshFormat`".to_string(),
                })?;
                if !saw_elements {
                    return Err(IoError::InvalidMshStructure {
                        offset: line_offset as u64,
                        reason: "`$ElementData` appeared before `$Elements`".to_string(),
                    });
                }
                let (name, array) = parse_model_data(
                    &mut reader,
                    hdr,
                    "$EndElementData",
                    cell_types.len(),
                    |tag| elem_tag_to_idx.get(&tag).copied(),
                )?;
                if !cell_data_arrays.iter().any(|(n, _)| n == &name) {
                    cell_data_arrays.push((name, array));
                }
            }
            other_section => {
                let section_name = &other_section[1..];
                let end_marker = format!("$End{section_name}");
                skip_until_end_marker(&mut reader, &end_marker, line_offset)?;
            }
        }
    }

    if format_info.is_none() {
        return Err(IoError::InvalidMshStructure {
            offset: 0,
            reason: "missing `$MeshFormat` section".to_string(),
        });
    }
    if !saw_nodes {
        return Err(IoError::InvalidMshStructure {
            offset: reader.pos as u64,
            reason: "missing `$Nodes` section".to_string(),
        });
    }
    if !saw_elements {
        return Err(IoError::InvalidMshStructure {
            offset: reader.pos as u64,
            reason: "missing `$Elements` section".to_string(),
        });
    }

    let mut grid = UnstructuredGrid::new(points, cell_types, offsets, connectivity, None)?;
    for (name, arr) in point_data_arrays {
        grid.insert_point_data(name, arr)?;
    }
    for (name, arr) in cell_data_arrays {
        grid.insert_cell_data(name, arr)?;
    }
    Ok(grid)
}

#[derive(Debug, Clone, Copy)]
struct MeshFormatHeader {
    version: MshVersion,
    is_binary: bool,
    size_t_bytes: usize,
    byte_order: ByteOrder,
}

fn parse_mesh_format(reader: &mut MshCursor<'_>) -> Result<MeshFormatHeader, IoError> {
    let (offset, line) = reader.read_line_required("$MeshFormat")?;
    let mut parts = line.split_whitespace();
    let ver_str = parts.next().ok_or_else(|| IoError::InvalidMshStructure {
        offset: offset as u64,
        reason: "missing version in `$MeshFormat`".to_string(),
    })?;
    let file_type_str = parts.next().ok_or_else(|| IoError::InvalidMshStructure {
        offset: offset as u64,
        reason: "missing file-type in `$MeshFormat`".to_string(),
    })?;
    let data_size_str = parts.next().ok_or_else(|| IoError::InvalidMshStructure {
        offset: offset as u64,
        reason: "missing data-size in `$MeshFormat`".to_string(),
    })?;

    let version = if ver_str == "4.1" || ver_str == "4.0" || ver_str == "4" {
        MshVersion::V41
    } else if ver_str == "2.2" || ver_str == "2.1" || ver_str == "2.0" || ver_str == "2" {
        MshVersion::V22
    } else {
        return Err(IoError::UnsupportedFeature {
            offset: offset as u64,
            feature: format!("Gmsh MSH version `{ver_str}` (supported: 2.2 and 4.1)"),
        });
    };

    let file_type: i32 = file_type_str.parse().map_err(|_| IoError::InvalidNumber {
        array: "$MeshFormat".to_string(),
        text: file_type_str.to_string(),
        offset: offset as u64,
    })?;
    let is_binary = match file_type {
        0 => false,
        1 => true,
        _ => {
            return Err(IoError::UnsupportedFeature {
                offset: offset as u64,
                feature: format!("Gmsh file-type `{file_type}` (expected 0 or 1)"),
            });
        }
    };

    let size_t_bytes: usize = data_size_str.parse().map_err(|_| IoError::InvalidNumber {
        array: "$MeshFormat".to_string(),
        text: data_size_str.to_string(),
        offset: offset as u64,
    })?;
    if size_t_bytes != 8 && size_t_bytes != 4 {
        return Err(IoError::UnsupportedFeature {
            offset: offset as u64,
            feature: format!("Gmsh data-size `{size_t_bytes}` (expected 4 or 8)"),
        });
    }

    let mut byte_order = ByteOrder::LittleEndian;
    if is_binary {
        let endian_offset = reader.pos;
        let raw_one = reader.read_exact(4, "$MeshFormat endian check")?;
        let arr: [u8; 4] = [raw_one[0], raw_one[1], raw_one[2], raw_one[3]];
        if i32::from_le_bytes(arr) == 1 {
            byte_order = ByteOrder::LittleEndian;
        } else if i32::from_be_bytes(arr) == 1 {
            byte_order = ByteOrder::BigEndian;
        } else {
            return Err(IoError::InvalidMshStructure {
                offset: endian_offset as u64,
                reason: format!("invalid binary endianness marker {arr:?} in `$MeshFormat`"),
            });
        }
        reader.skip_optional_line_ending();
    }

    reader.expect_end_marker("$EndMeshFormat")?;

    Ok(MeshFormatHeader {
        version,
        is_binary,
        size_t_bytes,
        byte_order,
    })
}

fn parse_nodes(
    reader: &mut MshCursor<'_>,
    hdr: MeshFormatHeader,
    points: &mut Vec<[f64; 3]>,
    node_tag_to_idx: &mut HashMap<u64, u32>,
) -> Result<(), IoError> {
    match hdr.version {
        MshVersion::V22 => parse_nodes_v22(reader, hdr, points, node_tag_to_idx)?,
        MshVersion::V41 => parse_nodes_v41(reader, hdr, points, node_tag_to_idx)?,
    }
    Ok(())
}

fn parse_nodes_v22(
    reader: &mut MshCursor<'_>,
    hdr: MeshFormatHeader,
    points: &mut Vec<[f64; 3]>,
    node_tag_to_idx: &mut HashMap<u64, u32>,
) -> Result<(), IoError> {
    let (offset, count_line) = reader.read_line_required("$Nodes")?;
    let num_nodes = parse_usize(count_line.trim(), "$Nodes", offset)?;
    if num_nodes > reader.remaining() / 4 {
        return Err(IoError::InvalidMshStructure {
            offset: offset as u64,
            reason: format!("`$Nodes` count {num_nodes} exceeds remaining file size"),
        });
    }

    points.reserve(num_nodes);
    node_tag_to_idx.reserve(num_nodes);

    if hdr.is_binary {
        for _ in 0..num_nodes {
            let rec_offset = reader.pos;
            let tag_i32 = reader.read_i32(hdr.byte_order, "$Nodes")?;
            if tag_i32 <= 0 {
                return Err(IoError::InvalidMshStructure {
                    offset: rec_offset as u64,
                    reason: format!("non-positive node tag {tag_i32} in `$Nodes`"),
                });
            }
            let x = reader.read_f64(hdr.byte_order, "$Nodes")?;
            let y = reader.read_f64(hdr.byte_order, "$Nodes")?;
            let z = reader.read_f64(hdr.byte_order, "$Nodes")?;
            let idx = try_u32_from_usize(points.len())?;
            if node_tag_to_idx.insert(tag_i32 as u64, idx).is_some() {
                return Err(IoError::InvalidMshStructure {
                    offset: rec_offset as u64,
                    reason: format!("duplicate node tag {tag_i32} in `$Nodes`"),
                });
            }
            points.push([x, y, z]);
        }
        reader.skip_optional_line_ending();
    } else {
        for _ in 0..num_nodes {
            let (line_off, line) = reader.read_line_required("$Nodes")?;
            let mut it = line.split_whitespace();
            let tag_str = it.next().ok_or_else(|| IoError::InvalidMshStructure {
                offset: line_off as u64,
                reason: "missing node tag in `$Nodes`".to_string(),
            })?;
            let tag = parse_u64(tag_str, "$Nodes", line_off)?;
            let x = parse_next_f64(&mut it, "$Nodes", line_off)?;
            let y = parse_next_f64(&mut it, "$Nodes", line_off)?;
            let z = parse_next_f64(&mut it, "$Nodes", line_off)?;
            let idx = try_u32_from_usize(points.len())?;
            if node_tag_to_idx.insert(tag, idx).is_some() {
                return Err(IoError::InvalidMshStructure {
                    offset: line_off as u64,
                    reason: format!("duplicate node tag {tag} in `$Nodes`"),
                });
            }
            points.push([x, y, z]);
        }
    }

    reader.expect_end_marker("$EndNodes")
}

fn parse_nodes_v41(
    reader: &mut MshCursor<'_>,
    hdr: MeshFormatHeader,
    points: &mut Vec<[f64; 3]>,
    node_tag_to_idx: &mut HashMap<u64, u32>,
) -> Result<(), IoError> {
    if hdr.is_binary {
        let hdr_offset = reader.pos;
        let num_entity_blocks = reader.read_size_t(hdr, "$Nodes")?;
        let total_nodes = reader.read_size_t(hdr, "$Nodes")?;
        let _min_tag = reader.read_size_t(hdr, "$Nodes")?;
        let _max_tag = reader.read_size_t(hdr, "$Nodes")?;

        if total_nodes > reader.remaining() / 24 {
            return Err(IoError::InvalidMshStructure {
                offset: hdr_offset as u64,
                reason: format!("`$Nodes` total_nodes {total_nodes} exceeds remaining file size"),
            });
        }
        points.reserve(total_nodes);
        node_tag_to_idx.reserve(total_nodes);

        let mut read_nodes = 0usize;
        for _ in 0..num_entity_blocks {
            let blk_offset = reader.pos;
            let entity_dim = reader.read_i32(hdr.byte_order, "$Nodes")?;
            let _entity_tag = reader.read_i32(hdr.byte_order, "$Nodes")?;
            let parametric = reader.read_i32(hdr.byte_order, "$Nodes")?;
            let num_in_block = reader.read_size_t(hdr, "$Nodes")?;

            if !(0..=3).contains(&entity_dim) || (parametric != 0 && parametric != 1) {
                return Err(IoError::InvalidMshStructure {
                    offset: blk_offset as u64,
                    reason: format!(
                        "invalid `$Nodes` block header (entity_dim={entity_dim}, parametric={parametric})"
                    ),
                });
            }
            if num_in_block > reader.remaining() / 24 {
                return Err(IoError::InvalidMshStructure {
                    offset: blk_offset as u64,
                    reason: format!(
                        "`$Nodes` block count {num_in_block} exceeds remaining file size"
                    ),
                });
            }

            let mut block_tags = Vec::with_capacity(num_in_block);
            for _ in 0..num_in_block {
                let tag = reader.read_size_t(hdr, "$Nodes")? as u64;
                block_tags.push(tag);
            }

            let extra_coords = if parametric != 0 {
                entity_dim as usize
            } else {
                0
            };
            for tag in block_tags {
                let x = reader.read_f64(hdr.byte_order, "$Nodes")?;
                let y = reader.read_f64(hdr.byte_order, "$Nodes")?;
                let z = reader.read_f64(hdr.byte_order, "$Nodes")?;
                for _ in 0..extra_coords {
                    let _ = reader.read_f64(hdr.byte_order, "$Nodes")?;
                }
                let idx = try_u32_from_usize(points.len())?;
                if node_tag_to_idx.insert(tag, idx).is_some() {
                    return Err(IoError::InvalidMshStructure {
                        offset: blk_offset as u64,
                        reason: format!("duplicate node tag {tag} in `$Nodes`"),
                    });
                }
                points.push([x, y, z]);
            }
            read_nodes =
                read_nodes
                    .checked_add(num_in_block)
                    .ok_or(IoError::InvalidMshStructure {
                        offset: blk_offset as u64,
                        reason: "node count overflow in `$Nodes`".to_string(),
                    })?;
        }
        if read_nodes != total_nodes {
            return Err(IoError::InvalidMshStructure {
                offset: hdr_offset as u64,
                reason: format!(
                    "`$Nodes` declared {total_nodes} nodes, but blocks had {read_nodes}"
                ),
            });
        }
        reader.skip_optional_line_ending();
    } else {
        let mut tok = AsciiSectionTokens::new(reader, "$Nodes", "$EndNodes");
        let hdr_offset = tok.pos();
        let num_entity_blocks = tok.next_usize()?;
        let total_nodes = tok.next_usize()?;
        let _min_tag = tok.next_usize()?;
        let _max_tag = tok.next_usize()?;

        if total_nodes > tok.remaining_bytes() / 4 {
            return Err(IoError::InvalidMshStructure {
                offset: hdr_offset as u64,
                reason: format!("`$Nodes` total_nodes {total_nodes} exceeds remaining file size"),
            });
        }
        points.reserve(total_nodes);
        node_tag_to_idx.reserve(total_nodes);

        let mut read_nodes = 0usize;
        for _ in 0..num_entity_blocks {
            let blk_offset = tok.pos();
            let entity_dim = tok.next_i32()?;
            let _entity_tag = tok.next_i32()?;
            let parametric = tok.next_i32()?;
            let num_in_block = tok.next_usize()?;

            if !(0..=3).contains(&entity_dim) || (parametric != 0 && parametric != 1) {
                return Err(IoError::InvalidMshStructure {
                    offset: blk_offset as u64,
                    reason: format!(
                        "invalid `$Nodes` block header (entity_dim={entity_dim}, parametric={parametric})"
                    ),
                });
            }
            if num_in_block > tok.remaining_bytes() / 4 {
                return Err(IoError::InvalidMshStructure {
                    offset: blk_offset as u64,
                    reason: format!(
                        "`$Nodes` block count {num_in_block} exceeds remaining file size"
                    ),
                });
            }

            let mut block_tags = Vec::with_capacity(num_in_block);
            for _ in 0..num_in_block {
                block_tags.push(tok.next_u64()?);
            }

            let extra_coords = if parametric != 0 {
                entity_dim as usize
            } else {
                0
            };
            for tag in block_tags {
                let x = tok.next_f64()?;
                let y = tok.next_f64()?;
                let z = tok.next_f64()?;
                for _ in 0..extra_coords {
                    let _ = tok.next_f64()?;
                }
                let idx = try_u32_from_usize(points.len())?;
                if node_tag_to_idx.insert(tag, idx).is_some() {
                    return Err(IoError::InvalidMshStructure {
                        offset: blk_offset as u64,
                        reason: format!("duplicate node tag {tag} in `$Nodes`"),
                    });
                }
                points.push([x, y, z]);
            }
            read_nodes =
                read_nodes
                    .checked_add(num_in_block)
                    .ok_or(IoError::InvalidMshStructure {
                        offset: blk_offset as u64,
                        reason: "node count overflow in `$Nodes`".to_string(),
                    })?;
        }
        if read_nodes != total_nodes {
            return Err(IoError::InvalidMshStructure {
                offset: hdr_offset as u64,
                reason: format!(
                    "`$Nodes` declared {total_nodes} nodes, but blocks had {read_nodes}"
                ),
            });
        }
    }

    reader.expect_end_marker("$EndNodes")
}

fn parse_elements(
    reader: &mut MshCursor<'_>,
    hdr: MeshFormatHeader,
    node_tag_to_idx: &HashMap<u64, u32>,
    cell_types: &mut Vec<CellType>,
    offsets: &mut Vec<u32>,
    connectivity: &mut Vec<u32>,
    elem_tag_to_idx: &mut HashMap<u64, usize>,
) -> Result<(), IoError> {
    match hdr.version {
        MshVersion::V22 => parse_elements_v22(
            reader,
            hdr,
            node_tag_to_idx,
            cell_types,
            offsets,
            connectivity,
            elem_tag_to_idx,
        )?,
        MshVersion::V41 => parse_elements_v41(
            reader,
            hdr,
            node_tag_to_idx,
            cell_types,
            offsets,
            connectivity,
            elem_tag_to_idx,
        )?,
    }
    Ok(())
}

fn parse_elements_v22(
    reader: &mut MshCursor<'_>,
    hdr: MeshFormatHeader,
    node_tag_to_idx: &HashMap<u64, u32>,
    cell_types: &mut Vec<CellType>,
    offsets: &mut Vec<u32>,
    connectivity: &mut Vec<u32>,
    elem_tag_to_idx: &mut HashMap<u64, usize>,
) -> Result<(), IoError> {
    let (offset, count_line) = reader.read_line_required("$Elements")?;
    let num_elements = parse_usize(count_line.trim(), "$Elements", offset)?;
    if num_elements > reader.remaining() / 4 {
        return Err(IoError::InvalidMshStructure {
            offset: offset as u64,
            reason: format!("`$Elements` count {num_elements} exceeds remaining file size"),
        });
    }

    cell_types.reserve(num_elements);
    offsets.reserve(num_elements);
    elem_tag_to_idx.reserve(num_elements);

    if hdr.is_binary {
        let mut read_elems = 0usize;
        let mut gmsh_nodes = Vec::new();
        while read_elems < num_elements {
            let sub_offset = reader.pos;
            let elm_type = reader.read_i32(hdr.byte_order, "$Elements")?;
            let num_in_block_i32 = reader.read_i32(hdr.byte_order, "$Elements")?;
            let num_tags_i32 = reader.read_i32(hdr.byte_order, "$Elements")?;
            if num_in_block_i32 <= 0 || num_tags_i32 < 0 {
                return Err(IoError::InvalidMshStructure {
                    offset: sub_offset as u64,
                    reason: format!(
                        "invalid binary `$Elements` sub-block header (num_in_block={num_in_block_i32}, num_tags={num_tags_i32})"
                    ),
                });
            }
            let num_in_block = num_in_block_i32 as usize;
            let num_tags = num_tags_i32 as usize;
            let spec = gmsh_element_spec(elm_type).ok_or_else(|| IoError::UnsupportedFeature {
                offset: sub_offset as u64,
                feature: format!("Gmsh element type {elm_type}"),
            })?;
            let num_nodes = spec.gmsh_index_for_vtk_slot.len();

            for _ in 0..num_in_block {
                let elem_offset = reader.pos;
                let elm_number = reader.read_i32(hdr.byte_order, "$Elements")?;
                if elm_number <= 0 {
                    return Err(IoError::InvalidMshStructure {
                        offset: elem_offset as u64,
                        reason: format!("non-positive element tag {elm_number} in `$Elements`"),
                    });
                }
                for _ in 0..num_tags {
                    let _ = reader.read_i32(hdr.byte_order, "$Elements")?;
                }
                gmsh_nodes.clear();
                for _ in 0..num_nodes {
                    let n_tag = reader.read_i32(hdr.byte_order, "$Elements")?;
                    if n_tag <= 0 {
                        return Err(IoError::InvalidMshStructure {
                            offset: elem_offset as u64,
                            reason: format!("non-positive node tag {n_tag} in `$Elements`"),
                        });
                    }
                    gmsh_nodes.push(n_tag as u64);
                }
                append_permuted_element(
                    elem_offset,
                    elm_number as u64,
                    spec,
                    &gmsh_nodes,
                    node_tag_to_idx,
                    cell_types,
                    offsets,
                    connectivity,
                    elem_tag_to_idx,
                )?;
            }
            read_elems =
                read_elems
                    .checked_add(num_in_block)
                    .ok_or(IoError::InvalidMshStructure {
                        offset: sub_offset as u64,
                        reason: "element count overflow in `$Elements`".to_string(),
                    })?;
        }
        if read_elems != num_elements {
            return Err(IoError::InvalidMshStructure {
                offset: offset as u64,
                reason: format!(
                    "`$Elements` declared {num_elements} elements, but binary blocks had {read_elems}"
                ),
            });
        }
        reader.skip_optional_line_ending();
    } else {
        let mut gmsh_nodes = Vec::new();
        for _ in 0..num_elements {
            let (line_off, line) = reader.read_line_required("$Elements")?;
            let mut it = line.split_whitespace();
            let elm_number = parse_next_u64(&mut it, "$Elements", line_off)?;
            let elm_type = parse_next_i32(&mut it, "$Elements", line_off)?;
            let num_tags = parse_next_usize(&mut it, "$Elements", line_off)?;
            for _ in 0..num_tags {
                let _ = parse_next_i32(&mut it, "$Elements", line_off)?;
            }
            let spec = gmsh_element_spec(elm_type).ok_or_else(|| IoError::UnsupportedFeature {
                offset: line_off as u64,
                feature: format!("Gmsh element type {elm_type}"),
            })?;
            let num_nodes = spec.gmsh_index_for_vtk_slot.len();
            gmsh_nodes.clear();
            for _ in 0..num_nodes {
                gmsh_nodes.push(parse_next_u64(&mut it, "$Elements", line_off)?);
            }
            append_permuted_element(
                line_off,
                elm_number,
                spec,
                &gmsh_nodes,
                node_tag_to_idx,
                cell_types,
                offsets,
                connectivity,
                elem_tag_to_idx,
            )?;
        }
    }

    reader.expect_end_marker("$EndElements")
}

fn parse_elements_v41(
    reader: &mut MshCursor<'_>,
    hdr: MeshFormatHeader,
    node_tag_to_idx: &HashMap<u64, u32>,
    cell_types: &mut Vec<CellType>,
    offsets: &mut Vec<u32>,
    connectivity: &mut Vec<u32>,
    elem_tag_to_idx: &mut HashMap<u64, usize>,
) -> Result<(), IoError> {
    if hdr.is_binary {
        let hdr_offset = reader.pos;
        let num_entity_blocks = reader.read_size_t(hdr, "$Elements")?;
        let total_elements = reader.read_size_t(hdr, "$Elements")?;
        let _min_tag = reader.read_size_t(hdr, "$Elements")?;
        let _max_tag = reader.read_size_t(hdr, "$Elements")?;

        if total_elements > reader.remaining() / 8 {
            return Err(IoError::InvalidMshStructure {
                offset: hdr_offset as u64,
                reason: format!(
                    "`$Elements` total_elements {total_elements} exceeds remaining file size"
                ),
            });
        }
        cell_types.reserve(total_elements);
        offsets.reserve(total_elements);
        elem_tag_to_idx.reserve(total_elements);

        let mut read_elems = 0usize;
        let mut gmsh_nodes = Vec::new();
        for _ in 0..num_entity_blocks {
            let blk_offset = reader.pos;
            let _entity_dim = reader.read_i32(hdr.byte_order, "$Elements")?;
            let _entity_tag = reader.read_i32(hdr.byte_order, "$Elements")?;
            let elm_type = reader.read_i32(hdr.byte_order, "$Elements")?;
            let num_in_block = reader.read_size_t(hdr, "$Elements")?;

            let spec = gmsh_element_spec(elm_type).ok_or_else(|| IoError::UnsupportedFeature {
                offset: blk_offset as u64,
                feature: format!("Gmsh element type {elm_type}"),
            })?;
            let num_nodes = spec.gmsh_index_for_vtk_slot.len();

            if num_in_block > reader.remaining() / ((1 + num_nodes) * hdr.size_t_bytes) {
                return Err(IoError::InvalidMshStructure {
                    offset: blk_offset as u64,
                    reason: format!(
                        "`$Elements` block count {num_in_block} exceeds remaining file size"
                    ),
                });
            }

            for _ in 0..num_in_block {
                let elem_offset = reader.pos;
                let elm_tag = reader.read_size_t(hdr, "$Elements")? as u64;
                gmsh_nodes.clear();
                for _ in 0..num_nodes {
                    gmsh_nodes.push(reader.read_size_t(hdr, "$Elements")? as u64);
                }
                append_permuted_element(
                    elem_offset,
                    elm_tag,
                    spec,
                    &gmsh_nodes,
                    node_tag_to_idx,
                    cell_types,
                    offsets,
                    connectivity,
                    elem_tag_to_idx,
                )?;
            }
            read_elems =
                read_elems
                    .checked_add(num_in_block)
                    .ok_or(IoError::InvalidMshStructure {
                        offset: blk_offset as u64,
                        reason: "element count overflow in `$Elements`".to_string(),
                    })?;
        }
        if read_elems != total_elements {
            return Err(IoError::InvalidMshStructure {
                offset: hdr_offset as u64,
                reason: format!(
                    "`$Elements` declared {total_elements} elements, but blocks had {read_elems}"
                ),
            });
        }
        reader.skip_optional_line_ending();
    } else {
        let mut tok = AsciiSectionTokens::new(reader, "$Elements", "$EndElements");
        let hdr_offset = tok.pos();
        let num_entity_blocks = tok.next_usize()?;
        let total_elements = tok.next_usize()?;
        let _min_tag = tok.next_usize()?;
        let _max_tag = tok.next_usize()?;

        if total_elements > tok.remaining_bytes() / 4 {
            return Err(IoError::InvalidMshStructure {
                offset: hdr_offset as u64,
                reason: format!(
                    "`$Elements` total_elements {total_elements} exceeds remaining file size"
                ),
            });
        }
        cell_types.reserve(total_elements);
        offsets.reserve(total_elements);
        elem_tag_to_idx.reserve(total_elements);

        let mut read_elems = 0usize;
        let mut gmsh_nodes = Vec::new();
        for _ in 0..num_entity_blocks {
            let blk_offset = tok.pos();
            let _entity_dim = tok.next_i32()?;
            let _entity_tag = tok.next_i32()?;
            let elm_type = tok.next_i32()?;
            let num_in_block = tok.next_usize()?;

            let spec = gmsh_element_spec(elm_type).ok_or_else(|| IoError::UnsupportedFeature {
                offset: blk_offset as u64,
                feature: format!("Gmsh element type {elm_type}"),
            })?;
            let num_nodes = spec.gmsh_index_for_vtk_slot.len();

            if num_in_block > tok.remaining_bytes() / 4 {
                return Err(IoError::InvalidMshStructure {
                    offset: blk_offset as u64,
                    reason: format!(
                        "`$Elements` block count {num_in_block} exceeds remaining file size"
                    ),
                });
            }

            for _ in 0..num_in_block {
                let elem_offset = tok.pos();
                let elm_tag = tok.next_u64()?;
                gmsh_nodes.clear();
                for _ in 0..num_nodes {
                    gmsh_nodes.push(tok.next_u64()?);
                }
                append_permuted_element(
                    elem_offset,
                    elm_tag,
                    spec,
                    &gmsh_nodes,
                    node_tag_to_idx,
                    cell_types,
                    offsets,
                    connectivity,
                    elem_tag_to_idx,
                )?;
            }
            read_elems =
                read_elems
                    .checked_add(num_in_block)
                    .ok_or(IoError::InvalidMshStructure {
                        offset: blk_offset as u64,
                        reason: "element count overflow in `$Elements`".to_string(),
                    })?;
        }
        if read_elems != total_elements {
            return Err(IoError::InvalidMshStructure {
                offset: hdr_offset as u64,
                reason: format!(
                    "`$Elements` declared {total_elements} elements, but blocks had {read_elems}"
                ),
            });
        }
    }

    reader.expect_end_marker("$EndElements")
}

#[allow(clippy::too_many_arguments)] // Collects output vectors of `$Elements` in place.
fn append_permuted_element(
    offset: usize,
    elm_tag: u64,
    spec: GmshElementSpec,
    gmsh_nodes: &[u64],
    node_tag_to_idx: &HashMap<u64, u32>,
    cell_types: &mut Vec<CellType>,
    offsets: &mut Vec<u32>,
    connectivity: &mut Vec<u32>,
    elem_tag_to_idx: &mut HashMap<u64, usize>,
) -> Result<(), IoError> {
    let cell_idx = cell_types.len();
    if elem_tag_to_idx.insert(elm_tag, cell_idx).is_some() {
        return Err(IoError::InvalidMshStructure {
            offset: offset as u64,
            reason: format!("duplicate element tag {elm_tag} in `$Elements`"),
        });
    }
    for &gmsh_local_idx in spec.gmsh_index_for_vtk_slot {
        let node_tag = gmsh_nodes[gmsh_local_idx];
        let &pt_idx =
            node_tag_to_idx
                .get(&node_tag)
                .ok_or_else(|| IoError::InvalidMshStructure {
                    offset: offset as u64,
                    reason: format!(
                        "element {elm_tag} references unknown node tag {node_tag} in `$Elements`"
                    ),
                })?;
        connectivity.push(pt_idx);
    }
    cell_types.push(spec.cell_type);
    offsets.push(try_u32_from_usize(connectivity.len())?);
    Ok(())
}

fn parse_model_data<F>(
    reader: &mut MshCursor<'_>,
    hdr: MeshFormatHeader,
    end_marker: &str,
    target_tuples: usize,
    resolve_tag: F,
) -> Result<(String, DataArray), IoError>
where
    F: Fn(u64) -> Option<usize>,
{
    let section_name = if end_marker == "$EndNodeData" {
        "$NodeData"
    } else {
        "$ElementData"
    };

    let (off_s, num_str_line) = reader.read_line_required(section_name)?;
    let num_string_tags = parse_usize(num_str_line.trim(), section_name, off_s)?;
    if num_string_tags > 32 {
        return Err(IoError::InvalidMshStructure {
            offset: off_s as u64,
            reason: format!("excessive string tag count {num_string_tags} in `{section_name}`"),
        });
    }
    let mut field_name = String::new();
    for i in 0..num_string_tags {
        let (_, s_line) = reader.read_line_required(section_name)?;
        if i == 0 {
            let trimmed = s_line.trim().trim_matches('"');
            field_name = if trimmed.is_empty() {
                "unnamed".to_string()
            } else {
                trimmed.to_string()
            };
        }
    }
    if field_name.is_empty() {
        field_name = "unnamed".to_string();
    }

    let (off_r, num_real_line) = reader.read_line_required(section_name)?;
    let num_real_tags = parse_usize(num_real_line.trim(), section_name, off_r)?;
    if num_real_tags > 32 {
        return Err(IoError::InvalidMshStructure {
            offset: off_r as u64,
            reason: format!("excessive real tag count {num_real_tags} in `{section_name}`"),
        });
    }
    for _ in 0..num_real_tags {
        let (r_off, r_line) = reader.read_line_required(section_name)?;
        let _ = parse_f64(r_line.trim(), section_name, r_off)?;
    }

    let (off_i, num_int_line) = reader.read_line_required(section_name)?;
    let num_int_tags = parse_usize(num_int_line.trim(), section_name, off_i)?;
    if !(3..=32).contains(&num_int_tags) {
        return Err(IoError::InvalidMshStructure {
            offset: off_i as u64,
            reason: format!(
                "expected at least 3 integer tags in `{section_name}`, got {num_int_tags}"
            ),
        });
    }
    let mut int_tags = Vec::with_capacity(num_int_tags);
    for _ in 0..num_int_tags {
        let (i_off, i_line) = reader.read_line_required(section_name)?;
        int_tags.push(parse_i64(i_line.trim(), section_name, i_off)?);
    }

    let num_comp_i64 = int_tags[1];
    let num_entries_i64 = int_tags[2];
    if num_comp_i64 <= 0 || num_entries_i64 < 0 {
        return Err(IoError::InvalidMshStructure {
            offset: off_i as u64,
            reason: format!(
                "invalid `{section_name}` dimensions (components={num_comp_i64}, entries={num_entries_i64})"
            ),
        });
    }
    let num_comp = num_comp_i64 as usize;
    let num_entries = num_entries_i64 as usize;
    let comp_count = ComponentCount::try_from(num_comp)?;

    let total_values = target_tuples
        .checked_mul(num_comp)
        .ok_or(IoError::InvalidMshStructure {
            offset: off_i as u64,
            reason: format!("value count overflow in `{section_name}`"),
        })?;
    let mut values = vec![0.0f64; total_values];

    if hdr.is_binary {
        let bytes_per_entry = 4usize
            .checked_add(
                num_comp
                    .checked_mul(8)
                    .ok_or(IoError::InvalidMshStructure {
                        offset: reader.pos as u64,
                        reason: "record size overflow".to_string(),
                    })?,
            )
            .ok_or(IoError::InvalidMshStructure {
                offset: reader.pos as u64,
                reason: "record size overflow".to_string(),
            })?;
        if num_entries > reader.remaining() / bytes_per_entry {
            return Err(IoError::InvalidMshStructure {
                offset: reader.pos as u64,
                reason: format!(
                    "`{section_name}` entry count {num_entries} exceeds remaining file size"
                ),
            });
        }

        for _ in 0..num_entries {
            let rec_off = reader.pos;
            let tag_i32 = reader.read_i32(hdr.byte_order, &field_name)?;
            if tag_i32 <= 0 {
                return Err(IoError::InvalidMshStructure {
                    offset: rec_off as u64,
                    reason: format!(
                        "non-positive tag {tag_i32} in `{section_name}` `{field_name}`"
                    ),
                });
            }
            let tuple_idx =
                resolve_tag(tag_i32 as u64).ok_or_else(|| IoError::InvalidMshStructure {
                    offset: rec_off as u64,
                    reason: format!(
                        "`{section_name}` `{field_name}` references unknown tag {tag_i32}"
                    ),
                })?;
            let base = tuple_idx * num_comp;
            for c in 0..num_comp {
                values[base + c] = reader.read_f64(hdr.byte_order, &field_name)?;
            }
        }
        reader.skip_optional_line_ending();
    } else {
        if num_entries > reader.remaining() / 2 {
            return Err(IoError::InvalidMshStructure {
                offset: reader.pos as u64,
                reason: format!(
                    "`{section_name}` entry count {num_entries} exceeds remaining file size"
                ),
            });
        }
        let mut tok = AsciiSectionTokens::new(reader, &field_name, end_marker);
        for _ in 0..num_entries {
            let rec_off = tok.pos();
            let tag = tok.next_u64()?;
            let tuple_idx = resolve_tag(tag).ok_or_else(|| IoError::InvalidMshStructure {
                offset: rec_off as u64,
                reason: format!("`{section_name}` `{field_name}` references unknown tag {tag}"),
            })?;
            let base = tuple_idx * num_comp;
            for c in 0..num_comp {
                values[base + c] = tok.next_f64()?;
            }
        }
    }

    reader.expect_end_marker(end_marker)?;
    let array = DataArray::from_f64(comp_count, values)?;
    Ok((field_name, array))
}

fn skip_until_end_marker(
    reader: &mut MshCursor<'_>,
    end_marker: &str,
    start_offset: usize,
) -> Result<(), IoError> {
    let needle = end_marker.as_bytes();
    let haystack = &reader.bytes[reader.pos..];
    if let Some(rel_pos) = haystack.windows(needle.len()).position(|w| w == needle) {
        reader.pos += rel_pos + needle.len();
        reader.skip_optional_line_ending();
        Ok(())
    } else {
        Err(IoError::InvalidMshStructure {
            offset: start_offset as u64,
            reason: format!("unterminated section: missing `{end_marker}`"),
        })
    }
}

struct MshCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> MshCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    const fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn read_line_opt(&mut self) -> Result<Option<(usize, &'a str)>, IoError> {
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let start = self.pos;
        let rel_nl = self.bytes[start..].iter().position(|&b| b == b'\n');
        let line_end = match rel_nl {
            Some(off) => {
                self.pos = start + off + 1;
                start + off
            }
            None => {
                self.pos = self.bytes.len();
                self.bytes.len()
            }
        };
        let slice = &self.bytes[start..line_end];
        let slice = slice.strip_suffix(b"\r").unwrap_or(slice);
        let text = std::str::from_utf8(slice).map_err(|e| IoError::InvalidMshStructure {
            offset: start as u64,
            reason: format!("invalid UTF-8 on line: {e}"),
        })?;
        Ok(Some((start, text)))
    }

    fn read_line_required(&mut self, context: &str) -> Result<(usize, &'a str), IoError> {
        self.read_line_opt()?
            .ok_or_else(|| IoError::InvalidMshStructure {
                offset: self.pos as u64,
                reason: format!("unexpected end of file while reading `{context}`"),
            })
    }

    fn expect_end_marker(&mut self, expected: &str) -> Result<(), IoError> {
        loop {
            let (offset, line) = self.read_line_required(expected)?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed == expected {
                return Ok(());
            }
            return Err(IoError::InvalidMshStructure {
                offset: offset as u64,
                reason: format!("expected `{expected}`, got `{trimmed}`"),
            });
        }
    }

    fn skip_optional_line_ending(&mut self) {
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'\r' {
            self.pos += 1;
        }
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'\n' {
            self.pos += 1;
        }
    }

    fn read_exact(&mut self, n: usize, context: &str) -> Result<&'a [u8], IoError> {
        let end = self.pos.checked_add(n).ok_or_else(|| IoError::BinaryData {
            array: context.to_string(),
            offset: self.pos as u64,
            reason: "byte offset overflow".to_string(),
        })?;
        if end > self.bytes.len() {
            return Err(IoError::BinaryData {
                array: context.to_string(),
                offset: self.pos as u64,
                reason: format!(
                    "unexpected EOF reading {n} binary bytes (only {} remaining)",
                    self.remaining()
                ),
            });
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_i32(&mut self, order: ByteOrder, context: &str) -> Result<i32, IoError> {
        let b = self.read_exact(4, context)?;
        let arr = [b[0], b[1], b[2], b[3]];
        Ok(match order {
            ByteOrder::LittleEndian => i32::from_le_bytes(arr),
            ByteOrder::BigEndian => i32::from_be_bytes(arr),
        })
    }

    fn read_f64(&mut self, order: ByteOrder, context: &str) -> Result<f64, IoError> {
        let b = self.read_exact(8, context)?;
        let arr = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
        Ok(match order {
            ByteOrder::LittleEndian => f64::from_le_bytes(arr),
            ByteOrder::BigEndian => f64::from_be_bytes(arr),
        })
    }

    fn read_size_t(&mut self, hdr: MeshFormatHeader, context: &str) -> Result<usize, IoError> {
        let offset = self.pos;
        if hdr.size_t_bytes == 8 {
            let b = self.read_exact(8, context)?;
            let arr = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
            let val = match hdr.byte_order {
                ByteOrder::LittleEndian => u64::from_le_bytes(arr),
                ByteOrder::BigEndian => u64::from_be_bytes(arr),
            };
            usize::try_from(val).map_err(|_| IoError::InvalidMshStructure {
                offset: offset as u64,
                reason: format!("`size_t` value {val} overflows platform `usize`"),
            })
        } else {
            let b = self.read_exact(4, context)?;
            let arr = [b[0], b[1], b[2], b[3]];
            let val = match hdr.byte_order {
                ByteOrder::LittleEndian => u32::from_le_bytes(arr),
                ByteOrder::BigEndian => u32::from_be_bytes(arr),
            };
            Ok(val as usize)
        }
    }
}

struct AsciiSectionTokens<'r, 'a> {
    reader: &'r mut MshCursor<'a>,
    section: &'r str,
    end_marker: &'r str,
    current_line: &'a str,
    current_line_offset: usize,
}

impl<'r, 'a> AsciiSectionTokens<'r, 'a> {
    fn new(reader: &'r mut MshCursor<'a>, section: &'r str, end_marker: &'r str) -> Self {
        Self {
            reader,
            section,
            end_marker,
            current_line: "",
            current_line_offset: 0,
        }
    }

    const fn pos(&self) -> usize {
        self.reader.pos
    }

    const fn remaining_bytes(&self) -> usize {
        self.reader.remaining()
    }

    fn next_token(&mut self) -> Result<(usize, &'a str), IoError> {
        loop {
            let trimmed = self.current_line.trim_start();
            if !trimmed.is_empty() {
                let tok_end = trimmed
                    .find(|c: char| c.is_ascii_whitespace())
                    .unwrap_or(trimmed.len());
                let tok = &trimmed[..tok_end];
                self.current_line = &trimmed[tok_end..];
                return Ok((self.current_line_offset, tok));
            }
            let (line_off, line) = self.reader.read_line_required(self.section)?;
            if line.trim() == self.end_marker {
                return Err(IoError::InvalidMshStructure {
                    offset: line_off as u64,
                    reason: format!(
                        "unexpected `{}` before finishing `{}`",
                        self.end_marker, self.section
                    ),
                });
            }
            self.current_line = line;
            self.current_line_offset = line_off;
        }
    }

    fn next_usize(&mut self) -> Result<usize, IoError> {
        let (off, tok) = self.next_token()?;
        parse_usize(tok, self.section, off)
    }

    fn next_u64(&mut self) -> Result<u64, IoError> {
        let (off, tok) = self.next_token()?;
        parse_u64(tok, self.section, off)
    }

    fn next_i32(&mut self) -> Result<i32, IoError> {
        let (off, tok) = self.next_token()?;
        parse_i32(tok, self.section, off)
    }

    fn next_f64(&mut self) -> Result<f64, IoError> {
        let (off, tok) = self.next_token()?;
        parse_f64(tok, self.section, off)
    }
}

fn parse_usize(text: &str, array: &str, offset: usize) -> Result<usize, IoError> {
    text.parse::<usize>().map_err(|_| IoError::InvalidNumber {
        array: array.to_string(),
        text: text.to_string(),
        offset: offset as u64,
    })
}

fn parse_u64(text: &str, array: &str, offset: usize) -> Result<u64, IoError> {
    let val = text.parse::<u64>().map_err(|_| IoError::InvalidNumber {
        array: array.to_string(),
        text: text.to_string(),
        offset: offset as u64,
    })?;
    if val == 0 {
        return Err(IoError::InvalidMshStructure {
            offset: offset as u64,
            reason: format!("zero tag in `{array}` (Gmsh tags are 1-based)"),
        });
    }
    Ok(val)
}

fn parse_i32(text: &str, array: &str, offset: usize) -> Result<i32, IoError> {
    text.parse::<i32>().map_err(|_| IoError::InvalidNumber {
        array: array.to_string(),
        text: text.to_string(),
        offset: offset as u64,
    })
}

fn parse_i64(text: &str, array: &str, offset: usize) -> Result<i64, IoError> {
    text.parse::<i64>().map_err(|_| IoError::InvalidNumber {
        array: array.to_string(),
        text: text.to_string(),
        offset: offset as u64,
    })
}

fn parse_f64(text: &str, array: &str, offset: usize) -> Result<f64, IoError> {
    let v = text.parse::<f64>().map_err(|_| IoError::InvalidNumber {
        array: array.to_string(),
        text: text.to_string(),
        offset: offset as u64,
    })?;
    if !v.is_finite() {
        return Err(IoError::InvalidNumber {
            array: array.to_string(),
            text: text.to_string(),
            offset: offset as u64,
        });
    }
    Ok(v)
}

fn parse_next_u64<'a>(
    it: &mut impl Iterator<Item = &'a str>,
    array: &str,
    offset: usize,
) -> Result<u64, IoError> {
    let tok = it.next().ok_or_else(|| IoError::InvalidMshStructure {
        offset: offset as u64,
        reason: format!("missing integer token on line in `{array}`"),
    })?;
    parse_u64(tok, array, offset)
}

fn parse_next_usize<'a>(
    it: &mut impl Iterator<Item = &'a str>,
    array: &str,
    offset: usize,
) -> Result<usize, IoError> {
    let tok = it.next().ok_or_else(|| IoError::InvalidMshStructure {
        offset: offset as u64,
        reason: format!("missing count token on line in `{array}`"),
    })?;
    parse_usize(tok, array, offset)
}

fn parse_next_i32<'a>(
    it: &mut impl Iterator<Item = &'a str>,
    array: &str,
    offset: usize,
) -> Result<i32, IoError> {
    let tok = it.next().ok_or_else(|| IoError::InvalidMshStructure {
        offset: offset as u64,
        reason: format!("missing integer token on line in `{array}`"),
    })?;
    parse_i32(tok, array, offset)
}

fn parse_next_f64<'a>(
    it: &mut impl Iterator<Item = &'a str>,
    array: &str,
    offset: usize,
) -> Result<f64, IoError> {
    let tok = it.next().ok_or_else(|| IoError::InvalidMshStructure {
        offset: offset as u64,
        reason: format!("missing float token on line in `{array}`"),
    })?;
    parse_f64(tok, array, offset)
}

//! Legacy VTK DataFile (`.vtk`, `DATASET UNSTRUCTURED_GRID`) reader (`M5.1`).
//!
//! Supports Legacy VTK DataFile versions 1.0 through 5.1 in both `ASCII` and big-endian
//! `BINARY` encodings (`D-008`, `D-009`, `D-018`):
//!
//! - **Geometry:** `POINTS <n> <dataType>`, classic VTK 2.0–4.2 `CELLS <n> <size>` (`[k, v0, ...]`),
//!   VTK 5.0–5.1 `CELLS <n+1> <size>` (`OFFSETS <type>` + `CONNECTIVITY <type>`), and
//!   `CELL_TYPES <n>`.
//! - **Cell types:** linear (`1, 3, 5, 9, 10, 12, 13, 14`), quadratic (`21..=27`), arbitrary-order
//!   Lagrange (`68..=73`), and polyhedra (`42`, both inline `CELLS` face streams and VTK 5.x
//!   `FACES` / `FACE_LOCATIONS` blocks).
//! - **Attributes:** `POINT_DATA <n>` and `CELL_DATA <n>` with `SCALARS`, `VECTORS`, `NORMALS`,
//!   `TENSORS`, `TENSORS6`, `TEXTURE_COORDINATES`, and `FIELD` data tables, plus `METADATA`
//!   (`INFORMATION`) blocks.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;

use tetraglass_core::{
    CellType, ComponentCount, DataArray, PolyhedronFaces, StepLoader, TimeSeries, UnstructuredGrid,
    try_u32_from_i64, try_u32_from_usize,
};

use crate::error::IoError;
use crate::vtu_codec::{
    ByteOrder, VtuDataType, convert_vtu23_polyhedra_to_stream, decode_bytes_data_array,
    decode_bytes_f64, decode_bytes_u32_indices,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VtkEncoding {
    Ascii,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveAttributeSection {
    None,
    PointData(usize),
    CellData(usize),
}

#[derive(Debug, Clone)]
enum ParsedCells {
    /// Classic VTK 2.0–4.2 `CELLS num_cells total_ints`: each cell is a slice `[v_0, ..., v_{k-1}]`.
    Classic {
        cell_slices: Vec<Vec<u32>>,
        offset: u64,
    },
    /// VTK 5.0–5.1 `CELLS (num_cells + 1) num_conn` with `OFFSETS` and `CONNECTIVITY`.
    OffsetsConnectivity {
        /// `num_cells` end-offsets (after stripping the leading `0`).
        offsets: Vec<u32>,
        connectivity: Vec<u32>,
        offset: u64,
    },
}

/// Reads a legacy VTK DataFile (`.vtk`, `DATASET UNSTRUCTURED_GRID`) from a byte slice.
///
/// Works on both `ASCII` and big-endian `BINARY` legacy `.vtk` files (versions 1.0–5.1)
/// and requires no filesystem access (`D-008`).
pub fn read_vtk(bytes: &[u8]) -> Result<UnstructuredGrid, IoError> {
    let (grid, _) = read_vtk_with_time(bytes)?;
    Ok(grid)
}

/// Wraps a legacy `.vtk` byte buffer in a single-step lazy [`TimeSeries<IoError>`].
///
/// If the `.vtk` file contains a dataset-level `FIELD` array named `TIME` or `TimeValue`
/// with a single finite scalar, that value is used as the step time; otherwise `0.0` is used.
pub fn read_vtk_series(bytes: impl Into<Arc<[u8]>>) -> Result<TimeSeries<IoError>, IoError> {
    let shared: Arc<[u8]> = bytes.into();
    // Validate the header upfront so format errors surface on series creation, matching `.msh`.
    let time_val = peek_vtk_header_and_time(&shared)?;
    let loader: StepLoader<IoError> = Arc::new(move || read_vtk(&shared));
    Ok(TimeSeries::from_boxed_steps(vec![(time_val, loader)])?)
}

/// Reads a legacy `.vtk` file from a filesystem path (available on non-wasm32 targets only).
#[cfg(not(target_arch = "wasm32"))]
pub fn read_vtk_from_path(path: impl AsRef<Path>) -> Result<UnstructuredGrid, IoError> {
    let bytes = std::fs::read(path)?;
    read_vtk(&bytes)
}

fn peek_vtk_header_and_time(bytes: &[u8]) -> Result<f64, IoError> {
    let mut cursor = VtkCursor::new(bytes);
    let encoding = cursor.parse_header()?;
    if cursor.peek_keyword_ignore_case("FIELD") {
        let _ = cursor.next_keyword_token()?;
        let _field_name = cursor.read_string_token("FIELD::name")?;
        let num_arrays = cursor.read_usize_token("FIELD::numArrays")?;
        for _ in 0..num_arrays {
            cursor.skip_optional_metadata()?;
            let (arr_offset, arr_name) =
                cursor
                    .next_keyword_token()?
                    .ok_or_else(|| IoError::InvalidVtuStructure {
                        offset: cursor.offset(),
                        reason: "unexpected EOF inside FIELD block".to_string(),
                    })?;
            let ncomp = cursor.read_usize_token(&arr_name)?;
            let ntuples = cursor.read_usize_token(&arr_name)?;
            let dtype_str = cursor.read_string_token(&arr_name)?;
            let dtype = parse_legacy_data_type(&dtype_str, cursor.offset())?;
            cursor.consume_line_end_for_data(encoding)?;
            let n_vals =
                ncomp
                    .checked_mul(ntuples)
                    .ok_or_else(|| IoError::InvalidVtuStructure {
                        offset: arr_offset,
                        reason: format!("FIELD array '{arr_name}' size overflow"),
                    })?;
            let vals = cursor.read_f64_values(&arr_name, dtype, n_vals, encoding, arr_offset)?;
            if (arr_name.eq_ignore_ascii_case("TIME") || arr_name.eq_ignore_ascii_case("TimeValue"))
                && vals.len() == 1
                && vals[0].is_finite()
            {
                return Ok(vals[0]);
            }
            cursor.skip_optional_metadata()?;
        }
    }
    Ok(0.0)
}

fn read_vtk_with_time(bytes: &[u8]) -> Result<(UnstructuredGrid, Option<f64>), IoError> {
    let mut cursor = VtkCursor::new(bytes);
    let encoding = cursor.parse_header()?;

    let mut points: Option<Vec<[f64; 3]>> = None;
    let mut parsed_cells: Option<ParsedCells> = None;
    let mut raw_cell_types: Option<(Vec<u8>, u64)> = None;
    let mut vtu23_faces: Option<(Vec<u32>, Vec<u32>, u64)> = None;
    let mut vtu23_face_locs: Option<(Vec<u32>, Vec<u32>)> = None;

    let mut point_data_arrays: Vec<(String, DataArray)> = Vec::new();
    let mut cell_data_arrays: Vec<(String, DataArray)> = Vec::new();
    let mut global_time: Option<f64> = None;
    let mut attr_section = ActiveAttributeSection::None;

    while let Some((kw_offset, keyword)) = cursor.next_keyword_token()? {
        let kw_upper = keyword.to_ascii_uppercase();
        match kw_upper.as_str() {
            "METADATA" => {
                cursor.skip_metadata_block()?;
            }
            "POINTS" => {
                let n_pts = cursor.read_usize_token("POINTS")?;
                let dtype_str = cursor.read_string_token("POINTS::dataType")?;
                let dtype = parse_legacy_data_type(&dtype_str, cursor.offset())?;
                cursor.consume_line_end_for_data(encoding)?;

                let n_vals = n_pts
                    .checked_mul(3)
                    .ok_or_else(|| IoError::InvalidVtuStructure {
                        offset: kw_offset,
                        reason: format!("POINTS count {n_pts} * 3 overflows usize"),
                    })?;
                let flat = cursor.read_f64_values("POINTS", dtype, n_vals, encoding, kw_offset)?;
                points = Some(flat.as_chunks::<3>().0.to_vec());
            }
            "CELLS" => {
                let n1 = cursor.read_usize_token("CELLS")?;
                let n2 = cursor.read_usize_token("CELLS")?;
                cursor.consume_line_end_for_data(encoding)?;

                if cursor.peek_is_offsets_keyword() {
                    // VTK 5.0 / 5.1 layout:
                    // CELLS <num_cells + 1> <num_conn>
                    // OFFSETS <dtype>
                    // <n1 offsets>
                    // CONNECTIVITY <dtype>
                    // <n2 indices>
                    let (_, off_kw) = cursor.next_keyword_token()?.ok_or_else(|| {
                        IoError::InvalidVtuStructure {
                            offset: cursor.offset(),
                            reason: "unexpected EOF reading OFFSETS in CELLS".to_string(),
                        }
                    })?;
                    debug_assert!(off_kw.eq_ignore_ascii_case("OFFSETS"));
                    let off_dtype_str = cursor.read_string_token("CELLS::OFFSETS")?;
                    let off_dtype = parse_legacy_data_type(&off_dtype_str, cursor.offset())?;
                    cursor.consume_line_end_for_data(encoding)?;
                    let raw_offsets = cursor.read_u32_values(
                        "CELLS::OFFSETS",
                        off_dtype,
                        n1,
                        encoding,
                        kw_offset,
                    )?;
                    cursor.skip_optional_metadata()?;

                    let (conn_pos, conn_kw) = cursor.next_keyword_token()?.ok_or_else(|| {
                        IoError::InvalidVtuStructure {
                            offset: cursor.offset(),
                            reason: "missing CONNECTIVITY block after OFFSETS in CELLS".to_string(),
                        }
                    })?;
                    if !conn_kw.eq_ignore_ascii_case("CONNECTIVITY") {
                        return Err(IoError::InvalidVtuStructure {
                            offset: conn_pos,
                            reason: format!(
                                "expected CONNECTIVITY after OFFSETS in CELLS, found '{conn_kw}'"
                            ),
                        });
                    }
                    let conn_dtype_str = cursor.read_string_token("CELLS::CONNECTIVITY")?;
                    let conn_dtype = parse_legacy_data_type(&conn_dtype_str, cursor.offset())?;
                    cursor.consume_line_end_for_data(encoding)?;
                    let connectivity = cursor.read_u32_values(
                        "CELLS::CONNECTIVITY",
                        conn_dtype,
                        n2,
                        encoding,
                        conn_pos,
                    )?;
                    cursor.skip_optional_metadata()?;

                    let offsets = if raw_offsets.is_empty() {
                        Vec::new()
                    } else {
                        if raw_offsets[0] != 0 {
                            return Err(IoError::InvalidVtuStructure {
                                offset: kw_offset,
                                reason: format!(
                                    "VTK 5.x CELLS OFFSETS must start with 0, got {}",
                                    raw_offsets[0]
                                ),
                            });
                        }
                        raw_offsets[1..].to_vec()
                    };

                    parsed_cells = Some(ParsedCells::OffsetsConnectivity {
                        offsets,
                        connectivity,
                        offset: kw_offset,
                    });
                } else {
                    // Classic VTK 2.0–4.2 layout:
                    // CELLS <num_cells> <total_ints>
                    // Each cell is [k, v_0, v_1, ..., v_{k-1}].
                    let raw_ints = cursor.read_u32_values(
                        "CELLS",
                        VtuDataType::Int32,
                        n2,
                        encoding,
                        kw_offset,
                    )?;
                    let mut cell_slices = Vec::with_capacity(n1);
                    let mut idx = 0usize;
                    for cell_i in 0..n1 {
                        if idx >= raw_ints.len() {
                            return Err(IoError::InvalidVtuStructure {
                                offset: kw_offset,
                                reason: format!(
                                    "CELLS ended early at cell {cell_i} of {n1} (index {idx} of {n2})"
                                ),
                            });
                        }
                        let k = raw_ints[idx] as usize;
                        idx += 1;
                        let end =
                            idx.checked_add(k)
                                .ok_or_else(|| IoError::InvalidVtuStructure {
                                    offset: kw_offset,
                                    reason: "cell vertex count overflow in CELLS".to_string(),
                                })?;
                        if end > raw_ints.len() {
                            return Err(IoError::InvalidVtuStructure {
                                offset: kw_offset,
                                reason: format!(
                                    "cell {cell_i} declares {k} entries, exceeding CELLS size {n2}"
                                ),
                            });
                        }
                        cell_slices.push(raw_ints[idx..end].to_vec());
                        idx = end;
                    }
                    if idx != raw_ints.len() {
                        return Err(IoError::InvalidVtuStructure {
                            offset: kw_offset,
                            reason: format!(
                                "CELLS declared {n2} integers for {n1} cells, but cells consumed {idx}"
                            ),
                        });
                    }
                    parsed_cells = Some(ParsedCells::Classic {
                        cell_slices,
                        offset: kw_offset,
                    });
                }
            }
            "FACES" => {
                // VTK 5.x polyhedron face connectivity block:
                // FACES <num_faces + 1> <num_face_conn>
                // OFFSETS <dtype> ... CONNECTIVITY <dtype> ...
                let (face_offs, face_conn) =
                    cursor.read_offsets_connectivity_pair("FACES", encoding, kw_offset)?;
                vtu23_faces = Some((face_conn, face_offs, kw_offset));
            }
            "FACE_LOCATIONS" => {
                let (poly_offs, poly_to_faces) =
                    cursor.read_offsets_connectivity_pair("FACE_LOCATIONS", encoding, kw_offset)?;
                vtu23_face_locs = Some((poly_to_faces, poly_offs));
            }
            "CELL_TYPES" => {
                let n_cells = cursor.read_usize_token("CELL_TYPES")?;
                cursor.consume_line_end_for_data(encoding)?;
                let raw_u32 = cursor.read_u32_values(
                    "CELL_TYPES",
                    VtuDataType::Int32,
                    n_cells,
                    encoding,
                    kw_offset,
                )?;
                let mut types_u8 = Vec::with_capacity(n_cells);
                for v in raw_u32 {
                    let b = u8::try_from(v).map_err(|_| IoError::InvalidVtuStructure {
                        offset: kw_offset,
                        reason: format!("CELL_TYPES value {v} overflows u8"),
                    })?;
                    types_u8.push(b);
                }
                raw_cell_types = Some((types_u8, kw_offset));
            }
            "POINT_DATA" => {
                let n = cursor.read_usize_token("POINT_DATA")?;
                attr_section = ActiveAttributeSection::PointData(n);
            }
            "CELL_DATA" => {
                let n = cursor.read_usize_token("CELL_DATA")?;
                attr_section = ActiveAttributeSection::CellData(n);
            }
            "SCALARS" => {
                let expected_tuples = require_attr_tuples(attr_section, "SCALARS", kw_offset)?;
                let name = cursor.read_string_token("SCALARS::name")?;
                let dtype_str = cursor.read_string_token("SCALARS::dataType")?;
                let dtype = parse_legacy_data_type(&dtype_str, cursor.offset())?;
                let ncomp = cursor.read_optional_usize_on_line()?.unwrap_or(1);
                cursor.consume_line_end_for_data(encoding)?;

                // Optional `LOOKUP_TABLE <table_name>` line.
                if cursor.peek_is_lookup_table_keyword() {
                    let _ = cursor.next_keyword_token()?;
                    let _table_name = cursor.read_string_token("LOOKUP_TABLE")?;
                    cursor.consume_line_end_for_data(encoding)?;
                }

                let comp = ComponentCount::try_from(ncomp)?;
                let arr = cursor.read_data_array(
                    &name,
                    dtype,
                    comp,
                    expected_tuples,
                    encoding,
                    kw_offset,
                )?;
                cursor.skip_optional_metadata()?;
                push_attribute_array(
                    attr_section,
                    name,
                    arr,
                    &mut point_data_arrays,
                    &mut cell_data_arrays,
                );
            }
            "VECTORS" | "NORMALS" => {
                let expected_tuples = require_attr_tuples(attr_section, &kw_upper, kw_offset)?;
                let name = cursor.read_string_token("VECTORS::name")?;
                let dtype_str = cursor.read_string_token("VECTORS::dataType")?;
                let dtype = parse_legacy_data_type(&dtype_str, cursor.offset())?;
                cursor.consume_line_end_for_data(encoding)?;

                let arr = cursor.read_data_array(
                    &name,
                    dtype,
                    ComponentCount::Vector3,
                    expected_tuples,
                    encoding,
                    kw_offset,
                )?;
                cursor.skip_optional_metadata()?;
                push_attribute_array(
                    attr_section,
                    name,
                    arr,
                    &mut point_data_arrays,
                    &mut cell_data_arrays,
                );
            }
            "TENSORS" => {
                let expected_tuples = require_attr_tuples(attr_section, "TENSORS", kw_offset)?;
                let name = cursor.read_string_token("TENSORS::name")?;
                let dtype_str = cursor.read_string_token("TENSORS::dataType")?;
                let dtype = parse_legacy_data_type(&dtype_str, cursor.offset())?;
                cursor.consume_line_end_for_data(encoding)?;

                let arr = cursor.read_data_array(
                    &name,
                    dtype,
                    ComponentCount::Tensor9,
                    expected_tuples,
                    encoding,
                    kw_offset,
                )?;
                cursor.skip_optional_metadata()?;
                push_attribute_array(
                    attr_section,
                    name,
                    arr,
                    &mut point_data_arrays,
                    &mut cell_data_arrays,
                );
            }
            "TENSORS6" => {
                let expected_tuples = require_attr_tuples(attr_section, "TENSORS6", kw_offset)?;
                let name = cursor.read_string_token("TENSORS6::name")?;
                let dtype_str = cursor.read_string_token("TENSORS6::dataType")?;
                let dtype = parse_legacy_data_type(&dtype_str, cursor.offset())?;
                cursor.consume_line_end_for_data(encoding)?;

                let arr = cursor.read_data_array(
                    &name,
                    dtype,
                    ComponentCount::SymmetricTensor6,
                    expected_tuples,
                    encoding,
                    kw_offset,
                )?;
                cursor.skip_optional_metadata()?;
                push_attribute_array(
                    attr_section,
                    name,
                    arr,
                    &mut point_data_arrays,
                    &mut cell_data_arrays,
                );
            }
            "TEXTURE_COORDINATES" => {
                let expected_tuples =
                    require_attr_tuples(attr_section, "TEXTURE_COORDINATES", kw_offset)?;
                let name = cursor.read_string_token("TEXTURE_COORDINATES::name")?;
                let dim = cursor.read_usize_token("TEXTURE_COORDINATES::dim")?;
                let dtype_str = cursor.read_string_token("TEXTURE_COORDINATES::dataType")?;
                let dtype = parse_legacy_data_type(&dtype_str, cursor.offset())?;
                cursor.consume_line_end_for_data(encoding)?;
                let n_vals = expected_tuples.checked_mul(dim).ok_or_else(|| {
                    IoError::InvalidVtuStructure {
                        offset: kw_offset,
                        reason: "TEXTURE_COORDINATES value count overflow".to_string(),
                    }
                })?;
                let _ = cursor.read_f64_values(&name, dtype, n_vals, encoding, kw_offset)?;
                cursor.skip_optional_metadata()?;
            }
            "LOOKUP_TABLE" => {
                let name = cursor.read_string_token("LOOKUP_TABLE::name")?;
                let size = cursor.read_usize_token("LOOKUP_TABLE::size")?;
                cursor.consume_line_end_for_data(encoding)?;
                match encoding {
                    VtkEncoding::Ascii => {
                        let n_floats =
                            size.checked_mul(4)
                                .ok_or_else(|| IoError::InvalidVtuStructure {
                                    offset: kw_offset,
                                    reason: "LOOKUP_TABLE size overflow".to_string(),
                                })?;
                        let _ = cursor.read_f64_values(
                            &name,
                            VtuDataType::Float32,
                            n_floats,
                            encoding,
                            kw_offset,
                        )?;
                    }
                    VtkEncoding::Binary => {
                        let n_bytes =
                            size.checked_mul(4)
                                .ok_or_else(|| IoError::InvalidVtuStructure {
                                    offset: kw_offset,
                                    reason: "LOOKUP_TABLE size overflow".to_string(),
                                })?;
                        let _ = cursor.read_exact_bytes(&name, n_bytes, kw_offset)?;
                    }
                }
            }
            "FIELD" => {
                let _field_name = cursor.read_string_token("FIELD::name")?;
                let num_arrays = cursor.read_usize_token("FIELD::numArrays")?;
                for _ in 0..num_arrays {
                    cursor.skip_optional_metadata()?;
                    let (arr_offset, arr_name) = cursor.next_keyword_token()?.ok_or_else(|| {
                        IoError::InvalidVtuStructure {
                            offset: cursor.offset(),
                            reason: "unexpected EOF inside FIELD block".to_string(),
                        }
                    })?;
                    let ncomp = cursor.read_usize_token(&arr_name)?;
                    let ntuples = cursor.read_usize_token(&arr_name)?;
                    let dtype_str = cursor.read_string_token(&arr_name)?;
                    let dtype = parse_legacy_data_type(&dtype_str, cursor.offset())?;
                    cursor.consume_line_end_for_data(encoding)?;

                    match attr_section {
                        ActiveAttributeSection::None => {
                            let n_vals = ncomp.checked_mul(ntuples).ok_or_else(|| {
                                IoError::InvalidVtuStructure {
                                    offset: arr_offset,
                                    reason: format!("FIELD array '{arr_name}' size overflow"),
                                }
                            })?;
                            let vals = cursor
                                .read_f64_values(&arr_name, dtype, n_vals, encoding, arr_offset)?;
                            if (arr_name.eq_ignore_ascii_case("TIME")
                                || arr_name.eq_ignore_ascii_case("TimeValue"))
                                && vals.len() == 1
                                && vals[0].is_finite()
                            {
                                global_time = Some(vals[0]);
                            }
                        }
                        ActiveAttributeSection::PointData(expected)
                        | ActiveAttributeSection::CellData(expected) => {
                            if ntuples != expected {
                                return Err(IoError::InvalidVtuStructure {
                                    offset: arr_offset,
                                    reason: format!(
                                        "FIELD array '{arr_name}' has {ntuples} tuples, expected {expected}"
                                    ),
                                });
                            }
                            let comp = ComponentCount::try_from(ncomp)?;
                            let arr = cursor.read_data_array(
                                &arr_name, dtype, comp, ntuples, encoding, arr_offset,
                            )?;
                            push_attribute_array(
                                attr_section,
                                arr_name,
                                arr,
                                &mut point_data_arrays,
                                &mut cell_data_arrays,
                            );
                        }
                    }
                    cursor.skip_optional_metadata()?;
                }
            }
            other => {
                return Err(IoError::InvalidVtuStructure {
                    offset: kw_offset,
                    reason: format!("unexpected or unsupported keyword '{other}' in .vtk file"),
                });
            }
        }
    }

    let points = points.ok_or_else(|| IoError::InvalidVtuStructure {
        offset: bytes.len() as u64,
        reason: "missing POINTS section in .vtk file".to_string(),
    })?;

    let BuiltCells {
        cell_types,
        offsets,
        connectivity,
        polyhedron_faces,
    } = build_cells_and_polyhedra(
        parsed_cells,
        raw_cell_types,
        vtu23_faces,
        vtu23_face_locs,
        bytes.len() as u64,
    )?;

    let mut grid =
        UnstructuredGrid::new(points, cell_types, offsets, connectivity, polyhedron_faces)?;

    for (name, arr) in point_data_arrays {
        grid.insert_point_data(name, arr)?;
    }
    for (name, arr) in cell_data_arrays {
        grid.insert_cell_data(name, arr)?;
    }

    Ok((grid, global_time))
}

struct BuiltCells {
    cell_types: Vec<CellType>,
    offsets: Vec<u32>,
    connectivity: Vec<u32>,
    polyhedron_faces: Option<PolyhedronFaces>,
}

fn build_cells_and_polyhedra(
    parsed_cells: Option<ParsedCells>,
    raw_cell_types: Option<(Vec<u8>, u64)>,
    vtu23_faces: Option<(Vec<u32>, Vec<u32>, u64)>,
    vtu23_face_locs: Option<(Vec<u32>, Vec<u32>)>,
    eof_offset: u64,
) -> Result<BuiltCells, IoError> {
    let Some(parsed_cells) = parsed_cells else {
        return Err(IoError::InvalidVtuStructure {
            offset: eof_offset,
            reason: "missing CELLS section in .vtk UNSTRUCTURED_GRID".to_string(),
        });
    };

    let (raw_types, types_offset) = raw_cell_types.ok_or_else(|| IoError::InvalidVtuStructure {
        offset: eof_offset,
        reason: "missing CELL_TYPES section in .vtk file".to_string(),
    })?;

    // Extract per-cell raw u32 slices from either Classic or OffsetsConnectivity layout.
    let (raw_cell_slices, cells_offset): (Vec<Vec<u32>>, u64) = match parsed_cells {
        ParsedCells::Classic {
            cell_slices,
            offset,
        } => (cell_slices, offset),
        ParsedCells::OffsetsConnectivity {
            offsets,
            connectivity,
            offset,
        } => {
            if offsets.len() != raw_types.len() {
                return Err(IoError::InvalidVtuStructure {
                    offset,
                    reason: format!(
                        "CELLS has {} cells, but CELL_TYPES has {}",
                        offsets.len(),
                        raw_types.len()
                    ),
                });
            }
            let mut slices = Vec::with_capacity(offsets.len());
            let mut prev = 0usize;
            for (cell_idx, &end_u32) in offsets.iter().enumerate() {
                let end = end_u32 as usize;
                if end < prev {
                    return Err(tetraglass_core::CoreError::NonMonotonicOffsets {
                        cell_index: cell_idx,
                        prev: prev as u32,
                        current: end_u32,
                    }
                    .into());
                }
                if end > connectivity.len() {
                    return Err(tetraglass_core::CoreError::OffsetOutOfBounds {
                        cell_index: cell_idx,
                        offset: end_u32,
                        connectivity_len: connectivity.len(),
                    }
                    .into());
                }
                slices.push(connectivity[prev..end].to_vec());
                prev = end;
            }
            if prev != connectivity.len() {
                return Err(tetraglass_core::CoreError::OffsetOutOfBounds {
                    cell_index: offsets.len().saturating_sub(1),
                    offset: prev as u32,
                    connectivity_len: connectivity.len(),
                }
                .into());
            }
            (slices, offset)
        }
    };

    if raw_cell_slices.len() != raw_types.len() {
        return Err(IoError::InvalidVtuStructure {
            offset: types_offset,
            reason: format!(
                "CELLS has {} cells, but CELL_TYPES has {}",
                raw_cell_slices.len(),
                raw_types.len()
            ),
        });
    }

    let has_polyhedron = raw_types.contains(&42);
    let has_separate_faces = vtu23_faces.is_some() && vtu23_face_locs.is_some();

    let mut cell_types = Vec::with_capacity(raw_types.len());
    let mut offsets = Vec::with_capacity(raw_types.len());
    let mut connectivity = Vec::new();
    let mut inline_poly_face_offsets = Vec::new();
    let mut inline_poly_faces = Vec::new();

    for (cell_idx, (&vtk_id, raw_slice)) in raw_types.iter().zip(raw_cell_slices.iter()).enumerate()
    {
        if vtk_id == 42 && !has_separate_faces {
            // In Legacy VTK (both 4.2 and 5.1), when no separate FACES section is present,
            // a VTK_POLYHEDRON cell's entry in CELLS is its face stream:
            // `[num_faces, f0_len, v00, ..., f1_len, v10, ...]`.
            let unique_verts =
                extract_unique_vertices_from_poly_stream(raw_slice, cell_idx, cells_offset)?;
            let ct = CellType::from_vtk_type_and_node_count(vtk_id, unique_verts.len())?;
            cell_types.push(ct);
            connectivity.extend_from_slice(&unique_verts);
            offsets.push(try_u32_from_usize(connectivity.len())?);

            inline_poly_faces.extend_from_slice(raw_slice);
            inline_poly_face_offsets.push(try_u32_from_usize(inline_poly_faces.len())?);
        } else {
            let ct = CellType::from_vtk_type_and_node_count(vtk_id, raw_slice.len())?;
            cell_types.push(ct);
            connectivity.extend_from_slice(raw_slice);
            offsets.push(try_u32_from_usize(connectivity.len())?);

            if has_polyhedron && !has_separate_faces {
                inline_poly_face_offsets.push(try_u32_from_usize(inline_poly_faces.len())?);
            }
        }
    }

    let polyhedron_faces =
        if let (Some((face_conn, face_offs, face_pos)), Some((poly_to_faces, poly_offs))) =
            (vtu23_faces, vtu23_face_locs)
        {
            let (face_offsets, faces) = convert_vtu23_polyhedra_to_stream(
                &raw_types,
                &face_conn,
                &face_offs,
                &poly_to_faces,
                &poly_offs,
                face_pos,
            )?;
            Some(PolyhedronFaces::new(face_offsets, faces))
        } else if has_polyhedron {
            Some(PolyhedronFaces::new(
                inline_poly_face_offsets,
                inline_poly_faces,
            ))
        } else {
            None
        };

    Ok(BuiltCells {
        cell_types,
        offsets,
        connectivity,
        polyhedron_faces,
    })
}

fn extract_unique_vertices_from_poly_stream(
    stream: &[u32],
    cell_index: usize,
    offset: u64,
) -> Result<Vec<u32>, IoError> {
    let Some((&num_faces, mut rest)) = stream.split_first() else {
        return Err(IoError::InvalidVtuStructure {
            offset,
            reason: format!("polyhedron cell {cell_index} has empty face stream in CELLS"),
        });
    };
    if num_faces < 4 {
        return Err(tetraglass_core::CoreError::InvalidPolyhedronFaceStream {
            cell_index,
            reason: "polyhedron cell must have at least 4 faces",
        }
        .into());
    }

    let mut unique = Vec::new();
    for _ in 0..num_faces {
        let Some((&npts, after_len)) = rest.split_first() else {
            return Err(tetraglass_core::CoreError::InvalidPolyhedronFaceStream {
                cell_index,
                reason: "truncated face vertex count in stream",
            }
            .into());
        };
        if npts < 3 {
            return Err(tetraglass_core::CoreError::InvalidPolyhedronFaceStream {
                cell_index,
                reason: "polygon face must have at least 3 vertices",
            }
            .into());
        }
        let npts_usize = npts as usize;
        if after_len.len() < npts_usize {
            return Err(tetraglass_core::CoreError::InvalidPolyhedronFaceStream {
                cell_index,
                reason: "truncated face vertex indices in stream",
            }
            .into());
        }
        for &vid in &after_len[..npts_usize] {
            if !unique.contains(&vid) {
                unique.push(vid);
            }
        }
        rest = &after_len[npts_usize..];
    }

    if !rest.is_empty() {
        return Err(tetraglass_core::CoreError::InvalidPolyhedronFaceStream {
            cell_index,
            reason: "trailing data in polyhedron cell stream",
        }
        .into());
    }
    unique.sort_unstable();
    Ok(unique)
}

fn require_attr_tuples(
    section: ActiveAttributeSection,
    keyword: &str,
    offset: u64,
) -> Result<usize, IoError> {
    match section {
        ActiveAttributeSection::PointData(n) | ActiveAttributeSection::CellData(n) => Ok(n),
        ActiveAttributeSection::None => Err(IoError::InvalidVtuStructure {
            offset,
            reason: format!("{keyword} appeared before POINT_DATA or CELL_DATA"),
        }),
    }
}

fn push_attribute_array(
    section: ActiveAttributeSection,
    name: String,
    arr: DataArray,
    point_data: &mut Vec<(String, DataArray)>,
    cell_data: &mut Vec<(String, DataArray)>,
) {
    match section {
        ActiveAttributeSection::PointData(_) => point_data.push((name, arr)),
        ActiveAttributeSection::CellData(_) => cell_data.push((name, arr)),
        ActiveAttributeSection::None => {}
    }
}

fn parse_legacy_data_type(raw: &str, offset: u64) -> Result<VtuDataType, IoError> {
    match raw.to_ascii_lowercase().as_str() {
        "float" | "vtktypefloat32" => Ok(VtuDataType::Float32),
        "double" | "vtktypefloat64" => Ok(VtuDataType::Float64),
        "char" | "signed_char" | "vtktypeint8" => Ok(VtuDataType::Int8),
        "unsigned_char" | "vtktypeuint8" => Ok(VtuDataType::UInt8),
        "short" | "vtktypeint16" => Ok(VtuDataType::Int16),
        "unsigned_short" | "vtktypeuint16" => Ok(VtuDataType::UInt16),
        "int" | "vtktypeint32" => Ok(VtuDataType::Int32),
        "unsigned_int" | "vtktypeuint32" => Ok(VtuDataType::UInt32),
        "long" | "long_long" | "__int64" | "vtktypeint64" | "vtkidtype" => Ok(VtuDataType::Int64),
        "unsigned_long" | "unsigned_long_long" | "unsigned___int64" | "vtktypeuint64" => {
            Ok(VtuDataType::UInt64)
        }
        other => Err(IoError::UnsupportedFeature {
            offset,
            feature: format!("legacy VTK data type '{other}'"),
        }),
    }
}

struct VtkCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> VtkCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    const fn offset(&self) -> u64 {
        self.pos as u64
    }

    fn parse_header(&mut self) -> Result<VtkEncoding, IoError> {
        // 1. `# vtk DataFile Version X.Y`
        let mut found_version_line = false;
        while let Some((line_off, line)) = self.read_ascii_line()? {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("# vtk DataFile Version") {
                let ver = rest.trim();
                if ver.is_empty() {
                    return Err(IoError::InvalidVtuStructure {
                        offset: line_off,
                        reason: "missing version number in '# vtk DataFile Version' header"
                            .to_string(),
                    });
                }
                found_version_line = true;
                break;
            }
            return Err(IoError::InvalidVtuStructure {
                offset: line_off,
                reason: "expected '# vtk DataFile Version X.Y' on first line of .vtk file"
                    .to_string(),
            });
        }

        if !found_version_line {
            return Err(IoError::InvalidVtuStructure {
                offset: 0,
                reason: "empty .vtk file (missing '# vtk DataFile Version' header)".to_string(),
            });
        }

        // 2. Title line (single line, may be empty).
        let _title = self
            .read_ascii_line()?
            .ok_or_else(|| IoError::InvalidVtuStructure {
                offset: self.offset(),
                reason: "truncated .vtk header (missing title line)".to_string(),
            })?;

        // 3. Encoding line (`ASCII` or `BINARY`).
        let (enc_off, enc_line) = self.read_next_nonempty_noncomment_line()?.ok_or_else(|| {
            IoError::InvalidVtuStructure {
                offset: self.offset(),
                reason: "truncated .vtk header (missing ASCII/BINARY encoding)".to_string(),
            }
        })?;
        let encoding = match enc_line.trim().to_ascii_uppercase().as_str() {
            "ASCII" => VtkEncoding::Ascii,
            "BINARY" => VtkEncoding::Binary,
            other => {
                return Err(IoError::UnsupportedFeature {
                    offset: enc_off,
                    feature: format!("legacy VTK encoding '{other}' (expected ASCII or BINARY)"),
                });
            }
        };

        // 4. `DATASET UNSTRUCTURED_GRID`
        let (ds_off, ds_line) = self.read_next_nonempty_noncomment_line()?.ok_or_else(|| {
            IoError::InvalidVtuStructure {
                offset: self.offset(),
                reason: "truncated .vtk header (missing DATASET line)".to_string(),
            }
        })?;
        let mut ds_parts = ds_line.split_whitespace();
        let Some(ds_kw) = ds_parts.next() else {
            return Err(IoError::InvalidVtuStructure {
                offset: ds_off,
                reason: "missing DATASET keyword".to_string(),
            });
        };
        if !ds_kw.eq_ignore_ascii_case("DATASET") {
            return Err(IoError::InvalidVtuStructure {
                offset: ds_off,
                reason: format!("expected DATASET line, found '{ds_kw}'"),
            });
        }
        let Some(ds_type) = ds_parts.next() else {
            return Err(IoError::InvalidVtuStructure {
                offset: ds_off,
                reason: "DATASET line is missing dataset type".to_string(),
            });
        };
        if !ds_type.eq_ignore_ascii_case("UNSTRUCTURED_GRID") {
            return Err(IoError::UnsupportedFeature {
                offset: ds_off,
                feature: format!(
                    "legacy VTK DATASET {ds_type} (only UNSTRUCTURED_GRID is supported)"
                ),
            });
        }

        Ok(encoding)
    }

    fn read_ascii_line(&mut self) -> Result<Option<(u64, String)>, IoError> {
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let start = self.pos;
        while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
            self.pos += 1;
        }
        let line_end = self.pos;
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'\n' {
            self.pos += 1;
        }
        let slice = &self.bytes[start..line_end];
        let trimmed = slice.strip_suffix(b"\r").unwrap_or(slice);
        let text = std::str::from_utf8(trimmed).map_err(|e| IoError::InvalidVtuStructure {
            offset: start as u64,
            reason: format!("invalid UTF-8 in .vtk header/line: {e}"),
        })?;
        Ok(Some((start as u64, text.to_string())))
    }

    fn read_next_nonempty_noncomment_line(&mut self) -> Result<Option<(u64, String)>, IoError> {
        while let Some((off, line)) = self.read_ascii_line()? {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            return Ok(Some((off, line)));
        }
        Ok(None)
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos < self.bytes.len() && self.bytes[self.pos] == b'#' {
                while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn next_keyword_token(&mut self) -> Result<Option<(u64, String)>, IoError> {
        self.skip_whitespace_and_comments();
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let start = self.pos;
        while self.pos < self.bytes.len() && !self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        let token = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|e| {
            IoError::InvalidVtuStructure {
                offset: start as u64,
                reason: format!("invalid UTF-8 in .vtk token: {e}"),
            }
        })?;
        Ok(Some((start as u64, token.to_string())))
    }

    fn read_string_token(&mut self, context: &str) -> Result<String, IoError> {
        let (_, tok) = self
            .next_keyword_token()?
            .ok_or_else(|| IoError::InvalidVtuStructure {
                offset: self.offset(),
                reason: format!("unexpected EOF while reading {context}"),
            })?;
        Ok(tok)
    }

    fn read_usize_token(&mut self, context: &str) -> Result<usize, IoError> {
        let (off, tok) =
            self.next_keyword_token()?
                .ok_or_else(|| IoError::InvalidVtuStructure {
                    offset: self.offset(),
                    reason: format!("unexpected EOF while reading {context}"),
                })?;
        tok.parse::<usize>().map_err(|_| IoError::InvalidNumber {
            array: context.to_string(),
            text: tok,
            offset: off,
        })
    }

    fn read_optional_usize_on_line(&mut self) -> Result<Option<usize>, IoError> {
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos] == b' ' || self.bytes[self.pos] == b'\t')
        {
            self.pos += 1;
        }
        if self.pos >= self.bytes.len()
            || self.bytes[self.pos] == b'\n'
            || self.bytes[self.pos] == b'\r'
        {
            return Ok(None);
        }
        let n = self.read_usize_token("SCALARS::numComp")?;
        Ok(Some(n))
    }

    /// Consumes horizontal spaces/tabs/`\r` and the single terminating `\n` of an ASCII header line
    /// before reading a binary or ASCII data payload.
    fn consume_line_end_for_data(&mut self, encoding: VtkEncoding) -> Result<(), IoError> {
        match encoding {
            VtkEncoding::Ascii => Ok(()),
            VtkEncoding::Binary => {
                while self.pos < self.bytes.len()
                    && (self.bytes[self.pos] == b' '
                        || self.bytes[self.pos] == b'\t'
                        || self.bytes[self.pos] == b'\r')
                {
                    self.pos += 1;
                }
                if self.pos >= self.bytes.len() || self.bytes[self.pos] != b'\n' {
                    return Err(IoError::InvalidVtuStructure {
                        offset: self.offset(),
                        reason: "expected newline after header line before binary payload"
                            .to_string(),
                    });
                }
                self.pos += 1;
                Ok(())
            }
        }
    }

    fn peek_keyword_ignore_case(&self, target: &str) -> bool {
        let mut p = self.pos;
        while p < self.bytes.len() && self.bytes[p].is_ascii_whitespace() {
            p += 1;
        }
        let end = p.saturating_add(target.len());
        if end > self.bytes.len() {
            return false;
        }
        if !self.bytes[p..end].eq_ignore_ascii_case(target.as_bytes()) {
            return false;
        }
        end == self.bytes.len() || self.bytes[end].is_ascii_whitespace()
    }

    fn peek_is_offsets_keyword(&self) -> bool {
        let mut p = self.pos;
        while p < self.bytes.len() && self.bytes[p].is_ascii_whitespace() {
            p += 1;
        }
        let kw = b"OFFSETS";
        let end = p.saturating_add(kw.len());
        if end >= self.bytes.len() || !self.bytes[p..end].eq_ignore_ascii_case(kw) {
            return false;
        }
        if self.bytes[end] != b' ' && self.bytes[end] != b'\t' {
            return false;
        }
        let mut q = end;
        while q < self.bytes.len() && (self.bytes[q] == b' ' || self.bytes[q] == b'\t') {
            q += 1;
        }
        let dtype_start = q;
        while q < self.bytes.len() && !self.bytes[q].is_ascii_whitespace() {
            q += 1;
        }
        if dtype_start == q {
            return false;
        }
        std::str::from_utf8(&self.bytes[dtype_start..q])
            .ok()
            .and_then(|s| parse_legacy_data_type(s, 0).ok())
            .is_some()
    }

    fn peek_is_lookup_table_keyword(&self) -> bool {
        self.peek_keyword_ignore_case("LOOKUP_TABLE")
    }

    fn skip_optional_metadata(&mut self) -> Result<(), IoError> {
        if self.peek_keyword_ignore_case("METADATA") {
            let _ = self.next_keyword_token()?;
            self.skip_metadata_block()?;
        }
        Ok(())
    }

    fn skip_metadata_block(&mut self) -> Result<(), IoError> {
        let meta_offset = self.offset();
        // Consume the remainder of the `METADATA` header line up to and including `\n`.
        while self.pos < self.bytes.len()
            && (self.bytes[self.pos] == b' '
                || self.bytes[self.pos] == b'\t'
                || self.bytes[self.pos] == b'\r')
        {
            self.pos += 1;
        }
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'\n' {
            self.pos += 1;
        }

        // `METADATA` must be followed by `INFORMATION <n>` and terminated by a blank line.
        let (info_off, info_line) =
            self.read_ascii_line()?
                .ok_or_else(|| IoError::InvalidVtuStructure {
                    offset: meta_offset,
                    reason: "truncated METADATA block (missing INFORMATION line)".to_string(),
                })?;
        let mut parts = info_line.split_whitespace();
        let Some(info_kw) = parts.next() else {
            return Err(IoError::InvalidVtuStructure {
                offset: info_off,
                reason: "empty line after METADATA (expected INFORMATION <n>)".to_string(),
            });
        };
        if !info_kw.eq_ignore_ascii_case("INFORMATION") {
            return Err(IoError::InvalidVtuStructure {
                offset: info_off,
                reason: format!("expected INFORMATION after METADATA, found '{info_kw}'"),
            });
        }
        let Some(n_str) = parts.next() else {
            return Err(IoError::InvalidVtuStructure {
                offset: info_off,
                reason: "INFORMATION line in METADATA is missing count".to_string(),
            });
        };
        let n_keys = n_str.parse::<usize>().map_err(|_| IoError::InvalidNumber {
            array: "METADATA::INFORMATION".to_string(),
            text: n_str.to_string(),
            offset: info_off,
        })?;

        let mut lines_read = 0usize;
        let mut saw_blank_terminator = false;
        while let Some((_, line)) = self.read_ascii_line()? {
            if line.trim().is_empty() {
                saw_blank_terminator = true;
                break;
            }
            lines_read += 1;
        }
        if n_keys > 0 && (lines_read < n_keys * 2 || !saw_blank_terminator) {
            return Err(IoError::InvalidVtuStructure {
                offset: meta_offset,
                reason: "truncated METADATA INFORMATION block".to_string(),
            });
        }
        Ok(())
    }

    fn read_offsets_connectivity_pair(
        &mut self,
        section_name: &str,
        encoding: VtkEncoding,
        section_offset: u64,
    ) -> Result<(Vec<u32>, Vec<u32>), IoError> {
        let n_offs = self.read_usize_token(section_name)?;
        let n_conn = self.read_usize_token(section_name)?;
        self.consume_line_end_for_data(encoding)?;

        let (off_pos, off_kw) =
            self.next_keyword_token()?
                .ok_or_else(|| IoError::InvalidVtuStructure {
                    offset: self.offset(),
                    reason: format!("unexpected EOF expecting OFFSETS in {section_name}"),
                })?;
        if !off_kw.eq_ignore_ascii_case("OFFSETS") {
            return Err(IoError::InvalidVtuStructure {
                offset: off_pos,
                reason: format!("expected OFFSETS in {section_name}, found '{off_kw}'"),
            });
        }
        let off_dtype_str = self.read_string_token(section_name)?;
        let off_dtype = parse_legacy_data_type(&off_dtype_str, self.offset())?;
        self.consume_line_end_for_data(encoding)?;
        let raw_offs =
            self.read_u32_values(section_name, off_dtype, n_offs, encoding, section_offset)?;
        self.skip_optional_metadata()?;

        let (conn_pos, conn_kw) =
            self.next_keyword_token()?
                .ok_or_else(|| IoError::InvalidVtuStructure {
                    offset: self.offset(),
                    reason: format!("unexpected EOF expecting CONNECTIVITY in {section_name}"),
                })?;
        if !conn_kw.eq_ignore_ascii_case("CONNECTIVITY") {
            return Err(IoError::InvalidVtuStructure {
                offset: conn_pos,
                reason: format!("expected CONNECTIVITY in {section_name}, found '{conn_kw}'"),
            });
        }
        let conn_dtype_str = self.read_string_token(section_name)?;
        let conn_dtype = parse_legacy_data_type(&conn_dtype_str, self.offset())?;
        self.consume_line_end_for_data(encoding)?;
        let conn = self.read_u32_values(section_name, conn_dtype, n_conn, encoding, conn_pos)?;
        self.skip_optional_metadata()?;

        let end_offsets = if raw_offs.is_empty() {
            Vec::new()
        } else {
            if raw_offs[0] != 0 {
                return Err(IoError::InvalidVtuStructure {
                    offset: section_offset,
                    reason: format!(
                        "{section_name} OFFSETS must start with 0, got {}",
                        raw_offs[0]
                    ),
                });
            }
            raw_offs[1..].to_vec()
        };

        Ok((end_offsets, conn))
    }

    fn read_exact_bytes(
        &mut self,
        array_name: &str,
        num_bytes: usize,
        header_offset: u64,
    ) -> Result<&'a [u8], IoError> {
        let end = self
            .pos
            .checked_add(num_bytes)
            .ok_or_else(|| IoError::BinaryData {
                array: array_name.to_string(),
                offset: header_offset,
                reason: "binary array byte length overflow".to_string(),
            })?;
        if end > self.bytes.len() {
            return Err(IoError::BinaryData {
                array: array_name.to_string(),
                offset: header_offset,
                reason: format!(
                    "truncated binary payload for '{array_name}': needed {num_bytes} bytes at offset {}, only {} available",
                    self.pos,
                    self.bytes.len().saturating_sub(self.pos)
                ),
            });
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_f64_values(
        &mut self,
        array_name: &str,
        dtype: VtuDataType,
        count: usize,
        encoding: VtkEncoding,
        header_offset: u64,
    ) -> Result<Vec<f64>, IoError> {
        match encoding {
            VtkEncoding::Ascii => {
                let mut out = Vec::with_capacity(count);
                for _ in 0..count {
                    let (tok_off, tok) =
                        self.next_keyword_token()?
                            .ok_or_else(|| IoError::InvalidVtuStructure {
                                offset: header_offset,
                                reason: format!(
                                    "truncated ASCII array '{array_name}': expected {count} values, got {}",
                                    out.len()
                                ),
                            })?;
                    let v = tok.parse::<f64>().map_err(|_| IoError::InvalidNumber {
                        array: array_name.to_string(),
                        text: tok.clone(),
                        offset: tok_off,
                    })?;
                    if !v.is_finite() {
                        return Err(IoError::InvalidNumber {
                            array: array_name.to_string(),
                            text: tok,
                            offset: tok_off,
                        });
                    }
                    out.push(v);
                }
                Ok(out)
            }
            VtkEncoding::Binary => {
                let num_bytes =
                    count
                        .checked_mul(dtype.byte_width())
                        .ok_or_else(|| IoError::BinaryData {
                            array: array_name.to_string(),
                            offset: header_offset,
                            reason: "binary byte size overflow".to_string(),
                        })?;
                let raw = self.read_exact_bytes(array_name, num_bytes, header_offset)?;
                decode_bytes_f64(array_name, raw, dtype, ByteOrder::BigEndian, header_offset)
            }
        }
    }

    fn read_u32_values(
        &mut self,
        array_name: &str,
        dtype: VtuDataType,
        count: usize,
        encoding: VtkEncoding,
        header_offset: u64,
    ) -> Result<Vec<u32>, IoError> {
        match encoding {
            VtkEncoding::Ascii => {
                let mut out = Vec::with_capacity(count);
                for _ in 0..count {
                    let (tok_off, tok) =
                        self.next_keyword_token()?
                            .ok_or_else(|| IoError::InvalidVtuStructure {
                                offset: header_offset,
                                reason: format!(
                                    "truncated ASCII index array '{array_name}': expected {count} values, got {}",
                                    out.len()
                                ),
                            })?;
                    let v_i64 = tok.parse::<i64>().map_err(|_| IoError::InvalidNumber {
                        array: array_name.to_string(),
                        text: tok,
                        offset: tok_off,
                    })?;
                    out.push(try_u32_from_i64(v_i64)?);
                }
                Ok(out)
            }
            VtkEncoding::Binary => {
                let num_bytes =
                    count
                        .checked_mul(dtype.byte_width())
                        .ok_or_else(|| IoError::BinaryData {
                            array: array_name.to_string(),
                            offset: header_offset,
                            reason: "binary index byte size overflow".to_string(),
                        })?;
                let raw = self.read_exact_bytes(array_name, num_bytes, header_offset)?;
                decode_bytes_u32_indices(
                    array_name,
                    raw,
                    dtype,
                    ByteOrder::BigEndian,
                    header_offset,
                )
            }
        }
    }

    fn read_data_array(
        &mut self,
        array_name: &str,
        dtype: VtuDataType,
        comp: ComponentCount,
        num_tuples: usize,
        encoding: VtkEncoding,
        header_offset: u64,
    ) -> Result<DataArray, IoError> {
        let count = num_tuples.checked_mul(comp.as_usize()).ok_or_else(|| {
            IoError::InvalidVtuStructure {
                offset: header_offset,
                reason: format!("array '{array_name}' tuple * component count overflow"),
            }
        })?;

        match encoding {
            VtkEncoding::Ascii => match dtype {
                VtuDataType::Float32 => {
                    let mut v = Vec::with_capacity(count);
                    for _ in 0..count {
                        let (off, tok) = self.next_ascii_element(array_name, count, v.len())?;
                        let val = tok.parse::<f32>().map_err(|_| IoError::InvalidNumber {
                            array: array_name.to_string(),
                            text: tok.clone(),
                            offset: off,
                        })?;
                        if !val.is_finite() {
                            return Err(IoError::InvalidNumber {
                                array: array_name.to_string(),
                                text: tok,
                                offset: off,
                            });
                        }
                        v.push(val);
                    }
                    Ok(DataArray::from_f32(comp, v)?)
                }
                VtuDataType::Float64 => {
                    let v = self.read_f64_values(
                        array_name,
                        VtuDataType::Float64,
                        count,
                        encoding,
                        header_offset,
                    )?;
                    Ok(DataArray::from_f64(comp, v)?)
                }
                VtuDataType::Int8 | VtuDataType::Int16 | VtuDataType::Int32 => {
                    let mut v = Vec::with_capacity(count);
                    for _ in 0..count {
                        let (off, tok) = self.next_ascii_element(array_name, count, v.len())?;
                        let val = tok.parse::<i32>().map_err(|_| IoError::InvalidNumber {
                            array: array_name.to_string(),
                            text: tok,
                            offset: off,
                        })?;
                        v.push(val);
                    }
                    Ok(DataArray::from_i32(comp, v)?)
                }
                VtuDataType::UInt8 => {
                    let mut v = Vec::with_capacity(count);
                    for _ in 0..count {
                        let (off, tok) = self.next_ascii_element(array_name, count, v.len())?;
                        let val = tok.parse::<u8>().map_err(|_| IoError::InvalidNumber {
                            array: array_name.to_string(),
                            text: tok,
                            offset: off,
                        })?;
                        v.push(val);
                    }
                    Ok(DataArray::from_u8(comp, v)?)
                }
                VtuDataType::UInt16
                | VtuDataType::UInt32
                | VtuDataType::Int64
                | VtuDataType::UInt64 => {
                    let mut v = Vec::with_capacity(count);
                    for _ in 0..count {
                        let (off, tok) = self.next_ascii_element(array_name, count, v.len())?;
                        let val = if dtype == VtuDataType::UInt64 {
                            let u = tok.parse::<u64>().map_err(|_| IoError::InvalidNumber {
                                array: array_name.to_string(),
                                text: tok.clone(),
                                offset: off,
                            })?;
                            i64::try_from(u).map_err(|_| IoError::InvalidNumber {
                                array: array_name.to_string(),
                                text: tok,
                                offset: off,
                            })?
                        } else {
                            tok.parse::<i64>().map_err(|_| IoError::InvalidNumber {
                                array: array_name.to_string(),
                                text: tok,
                                offset: off,
                            })?
                        };
                        v.push(val);
                    }
                    Ok(DataArray::from_i64(comp, v)?)
                }
            },
            VtkEncoding::Binary => {
                let num_bytes =
                    count
                        .checked_mul(dtype.byte_width())
                        .ok_or_else(|| IoError::BinaryData {
                            array: array_name.to_string(),
                            offset: header_offset,
                            reason: "binary DataArray byte size overflow".to_string(),
                        })?;
                let raw = self.read_exact_bytes(array_name, num_bytes, header_offset)?;
                decode_bytes_data_array(
                    array_name,
                    raw,
                    dtype,
                    comp,
                    ByteOrder::BigEndian,
                    header_offset,
                )
            }
        }
    }

    fn next_ascii_element(
        &mut self,
        array_name: &str,
        expected: usize,
        got: usize,
    ) -> Result<(u64, String), IoError> {
        self.next_keyword_token()?
            .ok_or_else(|| IoError::InvalidVtuStructure {
                offset: self.offset(),
                reason: format!(
                    "truncated ASCII array '{array_name}': expected {expected} values, got {got}"
                ),
            })
    }
}

//! Parallel VTK XML UnstructuredGrid (`.pvtu`) parser and piece merger (`M6.3`).
//!
//! A `.pvtu` file is an XML metadata manifest (`<VTKFile type="PUnstructuredGrid">`) that declares
//! global point/cell array schemas (`<PPointData>`, `<PCellData>`, `<PPoints>`) and references one
//! or more partitioned `.vtu` piece files (`<Piece Source="part_0.vtu"/>`):
//!
//! ```xml
//! <?xml version="1.0"?>
//! <VTKFile type="PUnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt32">
//!   <PUnstructuredGrid GhostLevel="1">
//!     <PPointData>
//!       <PDataArray type="Float64" Name="linear" NumberOfComponents="1"/>
//!     </PPointData>
//!     <PCellData>
//!       <PDataArray type="Int32" Name="cell_id" NumberOfComponents="1"/>
//!     </PCellData>
//!     <PPoints>
//!       <PDataArray type="Float64" Name="Points" NumberOfComponents="3"/>
//!     </PPoints>
//!     <Piece Source="hex_partitioned_0.vtu"/>
//!     <Piece Source="hex_partitioned_1.vtu"/>
//!   </PUnstructuredGrid>
//! </VTKFile>
//! ```
//!
//! [`read_pvtu_with_loader`] (for byte buffers / `wasm32-unknown-unknown`, `D-008`) and
//! [`read_pvtu_from_path`] (for native paths) parse the `.pvtu` manifest, read each referenced
//! `.vtu` piece, discard ghost cells (`vtkGhostType != 0`), weld coincident interface vertices
//! across piece boundaries when their point-array values agree, and return a unified
//! [`UnstructuredGrid`].

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;

use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use tetraglass_core::{
    ArrayBuffer, CellType, ComponentCount, DataArray, PolyhedronFaces, ScalarType, StepLoader,
    TimeSeries, UnstructuredGrid,
};

use crate::error::IoError;
use crate::read_vtu;

const GHOST_ARRAY_NAME: &str = "vtkGhostType";
const INTERFACE_WELD_QUANT_SCALE: f64 = 1.0e8;
const INTERFACE_WELD_DIST2_TOL: f64 = 1.0e-20;
const INTERFACE_VALUE_TOL: f64 = 1.0e-11;

/// Metadata for a `<PDataArray>` entry inside `<PPointData>` or `<PCellData>` in a `.pvtu` file (`M6.3`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvtuDataArrayMeta {
    /// Array name (`Name` attribute).
    pub name: String,
    /// VTK scalar type string (`type` attribute, e.g. `"Float64"`, `"Int32"`, `"UInt8"`).
    pub scalar_type: String,
    /// Number of scalar components per tuple (`NumberOfComponents` attribute, default `1`).
    pub number_of_components: usize,
}

/// A single `<Piece Source="..."/>` reference inside `<PUnstructuredGrid>` (`M6.3`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvtuPiece {
    /// Relative path to the piece `.vtu` file (`Source` attribute).
    pub source: String,
}

/// Parsed metadata manifest of a Parallel VTK XML `.pvtu` file (`M6.3`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvtuManifest {
    /// Number of ghost cell layers declared on `<PUnstructuredGrid GhostLevel="...">` (default `0`).
    pub ghost_level: u32,
    /// Global point array declarations from `<PPointData>`.
    pub point_data: Vec<PvtuDataArrayMeta>,
    /// Global cell array declarations from `<PCellData>`.
    pub cell_data: Vec<PvtuDataArrayMeta>,
    /// Ordered list of `<Piece Source="..."/>` references.
    pub pieces: Vec<PvtuPiece>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PvtuSection {
    Root,
    PPointData,
    PCellData,
    PPoints,
}

/// Parses the XML metadata manifest and `<Piece Source="..."/>` list from `.pvtu` bytes (`M6.3`).
pub fn parse_pvtu_manifest(bytes: &[u8]) -> Result<PvtuManifest, IoError> {
    let mut reader = Reader::from_reader(bytes);
    reader.config_mut().trim_text(true);

    let mut buf = Vec::new();
    let mut saw_vtk_file = false;
    let mut closed_vtk_file = false;
    let mut in_pgrid = false;
    let mut saw_pgrid = false;
    let mut closed_pgrid = false;
    let mut section = PvtuSection::Root;

    let mut ghost_level = 0u32;
    let mut point_data = Vec::new();
    let mut cell_data = Vec::new();
    let mut pieces = Vec::new();

    loop {
        let pos = reader.buffer_position();
        let event = reader
            .read_event_into(&mut buf)
            .map_err(|e| IoError::XmlSyntax {
                offset: pos,
                message: e.to_string(),
            })?;

        match event {
            Event::Eof => break,
            Event::Start(ref e) => match e.name().as_ref() {
                b"VTKFile" => {
                    saw_vtk_file = true;
                    validate_pvtu_vtk_file_attrs(e, pos)?;
                }
                b"PUnstructuredGrid" => {
                    if !saw_vtk_file {
                        return Err(IoError::InvalidVtuStructure {
                            offset: pos,
                            reason: "<PUnstructuredGrid> appeared outside <VTKFile>".to_string(),
                        });
                    }
                    ghost_level = parse_pgrid_ghost_level(e, pos)?;
                    in_pgrid = true;
                    saw_pgrid = true;
                }
                b"PPointData" => {
                    ensure_in_pgrid(in_pgrid, "PPointData", pos)?;
                    section = PvtuSection::PPointData;
                }
                b"PCellData" => {
                    ensure_in_pgrid(in_pgrid, "PCellData", pos)?;
                    section = PvtuSection::PCellData;
                }
                b"PPoints" => {
                    ensure_in_pgrid(in_pgrid, "PPoints", pos)?;
                    section = PvtuSection::PPoints;
                }
                b"PDataArray" => {
                    ensure_in_pgrid(in_pgrid, "PDataArray", pos)?;
                    match section {
                        PvtuSection::PPointData => {
                            if let Some(meta) = parse_pdata_array_attrs(e, pos, true)? {
                                point_data.push(meta);
                            }
                        }
                        PvtuSection::PCellData => {
                            if let Some(meta) = parse_pdata_array_attrs(e, pos, true)? {
                                cell_data.push(meta);
                            }
                        }
                        PvtuSection::PPoints => {
                            let _ = parse_pdata_array_attrs(e, pos, false)?;
                        }
                        PvtuSection::Root => {}
                    }
                }
                b"Piece" => {
                    ensure_in_pgrid(in_pgrid, "Piece", pos)?;
                    pieces.push(parse_piece_source(e, pos)?);
                }
                _ => {}
            },
            Event::Empty(ref e) => match e.name().as_ref() {
                b"PUnstructuredGrid" => {
                    if !saw_vtk_file {
                        return Err(IoError::InvalidVtuStructure {
                            offset: pos,
                            reason: "<PUnstructuredGrid/> appeared outside <VTKFile>".to_string(),
                        });
                    }
                    saw_pgrid = true;
                    closed_pgrid = true;
                }
                b"PPointData" | b"PCellData" | b"PPoints" => {
                    ensure_in_pgrid(in_pgrid, "PPointData/PCellData/PPoints", pos)?;
                }
                b"PDataArray" => {
                    ensure_in_pgrid(in_pgrid, "PDataArray", pos)?;
                    match section {
                        PvtuSection::PPointData => {
                            if let Some(meta) = parse_pdata_array_attrs(e, pos, true)? {
                                point_data.push(meta);
                            }
                        }
                        PvtuSection::PCellData => {
                            if let Some(meta) = parse_pdata_array_attrs(e, pos, true)? {
                                cell_data.push(meta);
                            }
                        }
                        PvtuSection::PPoints => {
                            let _ = parse_pdata_array_attrs(e, pos, false)?;
                        }
                        PvtuSection::Root => {}
                    }
                }
                b"Piece" => {
                    ensure_in_pgrid(in_pgrid, "Piece", pos)?;
                    pieces.push(parse_piece_source(e, pos)?);
                }
                _ => {}
            },
            Event::End(ref e) => match e.name().as_ref() {
                b"PPointData" | b"PCellData" | b"PPoints" => {
                    section = PvtuSection::Root;
                }
                b"PUnstructuredGrid" => {
                    in_pgrid = false;
                    closed_pgrid = true;
                }
                b"VTKFile" => {
                    closed_vtk_file = true;
                }
                _ => {}
            },
            _ => {}
        }
        buf.clear();
    }

    if !saw_vtk_file || !closed_vtk_file {
        return Err(IoError::InvalidVtuStructure {
            offset: reader.buffer_position(),
            reason: "missing or unclosed <VTKFile> root element in .pvtu".to_string(),
        });
    }
    if !saw_pgrid || !closed_pgrid {
        return Err(IoError::InvalidVtuStructure {
            offset: reader.buffer_position(),
            reason: "missing or unclosed <PUnstructuredGrid> element in .pvtu".to_string(),
        });
    }
    if pieces.is_empty() {
        return Err(IoError::InvalidVtuStructure {
            offset: reader.buffer_position(),
            reason: "<PUnstructuredGrid> in .pvtu contains no <Piece Source=\"...\"/> entries"
                .to_string(),
        });
    }

    Ok(PvtuManifest {
        ghost_level,
        point_data,
        cell_data,
        pieces,
    })
}

/// Reads and merges a partitioned `.pvtu` mesh from its XML bytes and a piece-fetching callback
/// `load_piece_bytes` (`M6.3`).
///
/// - Filters out ghost cells (`vtkGhostType != 0`).
/// - Strips unreferenced ghost-only vertices.
/// - Welds shared partition-interface vertices across pieces when both their 3D coordinates and
///   their point-array values agree.
/// - Works identically on native targets and `wasm32-unknown-unknown` (`D-008`).
pub fn read_pvtu_with_loader<F>(
    pvtu_bytes: &[u8],
    mut load_piece_bytes: F,
) -> Result<UnstructuredGrid, IoError>
where
    F: FnMut(&str) -> Result<Vec<u8>, IoError>,
{
    let manifest = parse_pvtu_manifest(pvtu_bytes)?;
    let mut piece_grids = Vec::with_capacity(manifest.pieces.len());
    for piece in &manifest.pieces {
        let piece_bytes = load_piece_bytes(&piece.source)?;
        let grid = read_vtu(&piece_bytes)?;
        piece_grids.push(grid);
    }
    merge_pvtu_pieces(&manifest, &piece_grids)
}

/// Wraps a `.pvtu` manifest and a piece-loading callback in a single-step lazy [`TimeSeries<IoError>`] (`M6.3`).
pub fn read_pvtu_series_with_loader<F>(
    pvtu_bytes: impl Into<Arc<[u8]>>,
    load_piece_bytes: F,
) -> Result<TimeSeries<IoError>, IoError>
where
    F: Fn(&str) -> Result<Vec<u8>, IoError> + Send + Sync + 'static,
{
    let shared_pvtu: Arc<[u8]> = pvtu_bytes.into();
    // Validate the manifest upfront so malformed `.pvtu` files fail before step evaluation.
    let _ = parse_pvtu_manifest(&shared_pvtu)?;
    let shared_loader = Arc::new(load_piece_bytes);
    let step_loader: StepLoader<IoError> = Arc::new(move || {
        let loader_ref = Arc::clone(&shared_loader);
        read_pvtu_with_loader(&shared_pvtu, move |rel_piece| loader_ref(rel_piece))
    });
    Ok(TimeSeries::from_boxed_steps(vec![(0.0, step_loader)])?)
}

/// Reads and merges a Parallel VTK XML `.pvtu` file from a native filesystem path (`M6.3`).
#[cfg(not(target_arch = "wasm32"))]
pub fn read_pvtu_from_path(path: impl AsRef<Path>) -> Result<UnstructuredGrid, IoError> {
    let pvtu_path = path.as_ref();
    let pvtu_bytes = std::fs::read(pvtu_path)?;
    let base_dir: PathBuf = pvtu_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    read_pvtu_with_loader(&pvtu_bytes, move |rel_piece| {
        let rel = Path::new(rel_piece);
        let piece_path = if rel.is_absolute() || base_dir.as_os_str().is_empty() {
            rel.to_path_buf()
        } else {
            base_dir.join(rel)
        };
        Ok(std::fs::read(piece_path)?)
    })
}

/// Reads a Parallel VTK XML `.pvtu` file from a native filesystem path as a lazy [`TimeSeries<IoError>`] (`M6.3`).
#[cfg(not(target_arch = "wasm32"))]
pub fn read_pvtu_series_from_path(path: impl AsRef<Path>) -> Result<TimeSeries<IoError>, IoError> {
    let pvtu_path = path.as_ref();
    let pvtu_bytes = std::fs::read(pvtu_path)?;
    let base_dir: PathBuf = pvtu_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    read_pvtu_series_with_loader(pvtu_bytes, move |rel_piece| {
        let rel = Path::new(rel_piece);
        let piece_path = if rel.is_absolute() || base_dir.as_os_str().is_empty() {
            rel.to_path_buf()
        } else {
            base_dir.join(rel)
        };
        Ok(std::fs::read(piece_path)?)
    })
}

#[derive(Debug, Clone)]
enum ArrayAccumulator {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
}

impl ArrayAccumulator {
    fn new(scalar_type: ScalarType) -> Self {
        match scalar_type {
            ScalarType::F32 => Self::F32(Vec::new()),
            ScalarType::F64 => Self::F64(Vec::new()),
            ScalarType::I32 => Self::I32(Vec::new()),
            ScalarType::I64 => Self::I64(Vec::new()),
            ScalarType::U8 => Self::U8(Vec::new()),
        }
    }

    fn promote_to_f64(&mut self) {
        if matches!(self, Self::F64(_)) {
            return;
        }
        let converted: Vec<f64> = match self {
            Self::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Self::F64(_) => unreachable!(),
            Self::I32(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Self::I64(v) => v.iter().map(|&x| x as f64).collect(),
            Self::U8(v) => v.iter().map(|&x| f64::from(x)).collect(),
        };
        *self = Self::F64(converted);
    }

    fn push_tuple_from(&mut self, src: &ArrayBuffer, tuple_idx: usize, ncomp: usize) {
        if self.scalar_type() != src.scalar_type() {
            self.promote_to_f64();
        }
        let start = tuple_idx * ncomp;
        let end = start + ncomp;
        match (self, src) {
            (Self::F32(dst), ArrayBuffer::F32(s)) => dst.extend_from_slice(&s[start..end]),
            (Self::F64(dst), ArrayBuffer::F64(s)) => dst.extend_from_slice(&s[start..end]),
            (Self::I32(dst), ArrayBuffer::I32(s)) => dst.extend_from_slice(&s[start..end]),
            (Self::I64(dst), ArrayBuffer::I64(s)) => dst.extend_from_slice(&s[start..end]),
            (Self::U8(dst), ArrayBuffer::U8(s)) => dst.extend_from_slice(&s[start..end]),
            (Self::F64(dst), other) => {
                for idx in start..end {
                    dst.push(other.get_as_f64(idx).unwrap_or(0.0));
                }
            }
            _ => unreachable!("non-matching scalar types are promoted to F64 first"),
        }
    }

    const fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::I32(_) => ScalarType::I32,
            Self::I64(_) => ScalarType::I64,
            Self::U8(_) => ScalarType::U8,
        }
    }

    fn get_as_f64(&self, flat_idx: usize) -> Option<f64> {
        match self {
            Self::F32(v) => v.get(flat_idx).map(|&x| f64::from(x)),
            Self::F64(v) => v.get(flat_idx).copied(),
            Self::I32(v) => v.get(flat_idx).map(|&x| f64::from(x)),
            Self::I64(v) => v.get(flat_idx).map(|&x| x as f64),
            Self::U8(v) => v.get(flat_idx).map(|&x| f64::from(x)),
        }
    }

    fn into_data_array(self, components: ComponentCount) -> Result<DataArray, IoError> {
        let buffer = match self {
            Self::F32(v) => ArrayBuffer::F32(v.into()),
            Self::F64(v) => ArrayBuffer::F64(v.into()),
            Self::I32(v) => ArrayBuffer::I32(v.into()),
            Self::I64(v) => ArrayBuffer::I64(v.into()),
            Self::U8(v) => ArrayBuffer::U8(v.into()),
        };
        Ok(DataArray::new(components, buffer)?)
    }
}

fn merge_pvtu_pieces(
    manifest: &PvtuManifest,
    pieces: &[UnstructuredGrid],
) -> Result<UnstructuredGrid, IoError> {
    let first_non_empty = pieces
        .iter()
        .find(|g| g.num_points() > 0 || g.num_cells() > 0)
        .or_else(|| pieces.first())
        .ok_or_else(|| IoError::InvalidVtuStructure {
            offset: 0,
            reason: "no pieces loaded from .pvtu".to_string(),
        })?;

    // Determine ordered point and cell array schemas (excluding `vtkGhostType`).
    let mut point_array_specs: Vec<(String, ComponentCount, ArrayAccumulator)> = Vec::new();
    for (name, arr) in first_non_empty.point_data().iter() {
        if name.eq_ignore_ascii_case(GHOST_ARRAY_NAME) {
            continue;
        }
        point_array_specs.push((
            name.to_string(),
            arr.components(),
            ArrayAccumulator::new(arr.scalar_type()),
        ));
    }

    let mut cell_array_specs: Vec<(String, ComponentCount, ArrayAccumulator)> = Vec::new();
    for (name, arr) in first_non_empty.cell_data().iter() {
        if name.eq_ignore_ascii_case(GHOST_ARRAY_NAME) {
            continue;
        }
        cell_array_specs.push((
            name.to_string(),
            arr.components(),
            ArrayAccumulator::new(arr.scalar_type()),
        ));
    }

    let mut merged_points: Vec<[f64; 3]> = Vec::new();
    let mut merged_cell_types: Vec<CellType> = Vec::new();
    let mut merged_offsets: Vec<u32> = Vec::new();
    let mut merged_connectivity: Vec<u32> = Vec::new();
    let mut merged_poly_faces: Vec<u32> = Vec::new();
    let mut merged_poly_offsets: Vec<u32> = Vec::new();
    let mut any_polyhedron = false;

    // Spatial hash `(qx, qy, qz) -> Vec<(merged_point_index, origin_piece_index)>`
    // for welding shared interface vertices across distinct pieces.
    let mut spatial_buckets: HashMap<[i64; 3], Vec<(u32, usize)>> = HashMap::new();

    for (piece_idx, piece) in pieces.iter().enumerate() {
        let piece_name = manifest
            .pieces
            .get(piece_idx)
            .map_or("<piece>", |p| p.source.as_str());

        // Validate that this piece carries every expected point and cell array.
        if piece.num_points() > 0 {
            for (arr_name, expected_comp, _) in &point_array_specs {
                let Some(arr) = piece.point_data().get(arr_name) else {
                    return Err(IoError::InvalidVtuStructure {
                        offset: 0,
                        reason: format!(
                            "piece `{piece_name}` is missing point DataArray `{arr_name}`"
                        ),
                    });
                };
                if arr.components() != *expected_comp {
                    return Err(IoError::InvalidVtuStructure {
                        offset: 0,
                        reason: format!(
                            "piece `{piece_name}` point DataArray `{arr_name}` has {} components, expected {}",
                            arr.components().as_usize(),
                            expected_comp.as_usize()
                        ),
                    });
                }
            }
        }
        if piece.num_cells() > 0 {
            for (arr_name, expected_comp, _) in &cell_array_specs {
                let Some(arr) = piece.cell_data().get(arr_name) else {
                    return Err(IoError::InvalidVtuStructure {
                        offset: 0,
                        reason: format!(
                            "piece `{piece_name}` is missing cell DataArray `{arr_name}`"
                        ),
                    });
                };
                if arr.components() != *expected_comp {
                    return Err(IoError::InvalidVtuStructure {
                        offset: 0,
                        reason: format!(
                            "piece `{piece_name}` cell DataArray `{arr_name}` has {} components, expected {}",
                            arr.components().as_usize(),
                            expected_comp.as_usize()
                        ),
                    });
                }
            }
        }

        let cell_ghost_arr = find_ghost_array(piece.cell_data());
        let point_ghost_arr = find_ghost_array(piece.point_data());

        // 1. Identify non-ghost cells and mark all local points referenced by those kept cells.
        let mut kept_cells: Vec<usize> = Vec::with_capacity(piece.num_cells());
        let mut point_used = vec![false; piece.num_points()];

        if piece.num_cells() == 0 {
            for (pt_idx, used) in point_used.iter_mut().enumerate() {
                if !is_ghost_tuple(point_ghost_arr, pt_idx) {
                    *used = true;
                }
            }
        } else {
            for cell_idx in 0..piece.num_cells() {
                if is_ghost_tuple(cell_ghost_arr, cell_idx) {
                    continue;
                }
                kept_cells.push(cell_idx);
                if let Some(conn) = piece.cell_connectivity(cell_idx) {
                    for &pt_u32 in conn {
                        if let Some(slot) = point_used.get_mut(pt_u32 as usize) {
                            *slot = true;
                        }
                    }
                }
                if let Some(face_iter) = piece.polyhedron_cell_faces(cell_idx) {
                    for face in face_iter {
                        for &pt_u32 in face {
                            if let Some(slot) = point_used.get_mut(pt_u32 as usize) {
                                *slot = true;
                            }
                        }
                    }
                }
            }
        }

        // 2. Map used local points to merged point indices, welding across piece boundaries.
        let mut local_to_merged: Vec<u32> = vec![u32::MAX; piece.num_points()];
        for (local_pt, &used) in point_used.iter().enumerate() {
            if !used {
                continue;
            }
            let pos = piece.points()[local_pt];
            let qkey = quantize_coord(pos);

            let mut welded_idx: Option<u32> = None;
            if let Some(candidates) = spatial_buckets.get(&qkey) {
                for &(cand_merged_u32, origin_piece) in candidates {
                    if origin_piece == piece_idx {
                        continue;
                    }
                    let cand_merged = cand_merged_u32 as usize;
                    let cand_pos = merged_points[cand_merged];
                    let dx = pos[0] - cand_pos[0];
                    let dy = pos[1] - cand_pos[1];
                    let dz = pos[2] - cand_pos[2];
                    if dx * dx + dy * dy + dz * dz <= INTERFACE_WELD_DIST2_TOL
                        && point_tuples_match(&point_array_specs, cand_merged, piece, local_pt)
                    {
                        welded_idx = Some(cand_merged_u32);
                        break;
                    }
                }
            }

            let merged_idx = if let Some(existing) = welded_idx {
                existing
            } else {
                let new_idx = u32::try_from(merged_points.len()).map_err(|_| {
                    tetraglass_core::CoreError::IndexOverflow {
                        value: merged_points.len() as u64,
                    }
                })?;
                merged_points.push(pos);
                for (arr_name, comp, accum) in &mut point_array_specs {
                    let src_arr = piece
                        .point_data()
                        .get(arr_name)
                        .expect("validated point array");
                    accum.push_tuple_from(src_arr.buffer(), local_pt, comp.as_usize());
                }
                spatial_buckets
                    .entry(qkey)
                    .or_default()
                    .push((new_idx, piece_idx));
                new_idx
            };
            local_to_merged[local_pt] = merged_idx;
        }

        // 3. Append kept cells and cell array tuples.
        for &cell_idx in &kept_cells {
            let ct = piece.cell_types()[cell_idx];
            let conn = piece
                .cell_connectivity(cell_idx)
                .expect("valid cell connectivity");
            for &local_pt in conn {
                let mapped = local_to_merged[local_pt as usize];
                merged_connectivity.push(mapped);
            }
            let end_offset = u32::try_from(merged_connectivity.len()).map_err(|_| {
                tetraglass_core::CoreError::IndexOverflow {
                    value: merged_connectivity.len() as u64,
                }
            })?;
            merged_cell_types.push(ct);
            merged_offsets.push(end_offset);

            if ct == CellType::Polyhedron {
                any_polyhedron = true;
                let face_iter = piece.polyhedron_cell_faces(cell_idx).ok_or_else(|| {
                    IoError::InvalidVtuStructure {
                        offset: 0,
                        reason: format!(
                            "polyhedron cell {cell_idx} in piece `{piece_name}` has no face stream"
                        ),
                    }
                })?;
                let faces_vec: Vec<&[u32]> = face_iter.collect();
                merged_poly_faces.push(faces_vec.len() as u32);
                for face in faces_vec {
                    merged_poly_faces.push(face.len() as u32);
                    for &local_pt in face {
                        merged_poly_faces.push(local_to_merged[local_pt as usize]);
                    }
                }
                let poly_end = u32::try_from(merged_poly_faces.len()).map_err(|_| {
                    tetraglass_core::CoreError::IndexOverflow {
                        value: merged_poly_faces.len() as u64,
                    }
                })?;
                merged_poly_offsets.push(poly_end);
            } else if any_polyhedron {
                let poly_end = merged_poly_offsets.last().copied().unwrap_or(0);
                merged_poly_offsets.push(poly_end);
            }

            for (arr_name, comp, accum) in &mut cell_array_specs {
                let src_arr = piece
                    .cell_data()
                    .get(arr_name)
                    .expect("validated cell array");
                accum.push_tuple_from(src_arr.buffer(), cell_idx, comp.as_usize());
            }
        }
    }

    // Backfill leading 0s in `merged_poly_offsets` if an earlier cell was not a polyhedron
    // before the first polyhedron cell appeared.
    let polyhedron_faces = if any_polyhedron {
        if merged_poly_offsets.len() < merged_cell_types.len() {
            let missing = merged_cell_types.len() - merged_poly_offsets.len();
            let mut full_offsets = vec![0u32; missing];
            full_offsets.extend_from_slice(&merged_poly_offsets);
            merged_poly_offsets = full_offsets;
        }
        Some(PolyhedronFaces::new(merged_poly_offsets, merged_poly_faces))
    } else {
        None
    };

    let mut grid = UnstructuredGrid::new(
        merged_points,
        merged_cell_types,
        merged_offsets,
        merged_connectivity,
        polyhedron_faces,
    )?;

    for (name, comp, accum) in point_array_specs {
        grid.insert_point_data(name, accum.into_data_array(comp)?)?;
    }
    for (name, comp, accum) in cell_array_specs {
        grid.insert_cell_data(name, accum.into_data_array(comp)?)?;
    }

    Ok(grid)
}

fn find_ghost_array(attrs: &tetraglass_core::Attributes) -> Option<&DataArray> {
    attrs
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(GHOST_ARRAY_NAME))
        .map(|(_, arr)| arr)
}

fn is_ghost_tuple(ghost_arr: Option<&DataArray>, tuple_idx: usize) -> bool {
    let Some(arr) = ghost_arr else {
        return false;
    };
    let ncomp = arr.components().as_usize();
    let base = tuple_idx * ncomp;
    arr.buffer().get_as_f64(base).is_some_and(|v| v.abs() > 0.5)
}

fn quantize_coord(p: [f64; 3]) -> [i64; 3] {
    [
        (p[0] * INTERFACE_WELD_QUANT_SCALE).round() as i64,
        (p[1] * INTERFACE_WELD_QUANT_SCALE).round() as i64,
        (p[2] * INTERFACE_WELD_QUANT_SCALE).round() as i64,
    ]
}

fn point_tuples_match(
    point_array_specs: &[(String, ComponentCount, ArrayAccumulator)],
    merged_pt_idx: usize,
    piece: &UnstructuredGrid,
    local_pt_idx: usize,
) -> bool {
    for (name, comp, accum) in point_array_specs {
        let Some(piece_arr) = piece.point_data().get(name) else {
            return false;
        };
        let ncomp = comp.as_usize();
        let merged_base = merged_pt_idx * ncomp;
        let local_base = local_pt_idx * ncomp;
        for c in 0..ncomp {
            let Some(v_merged) = accum.get_as_f64(merged_base + c) else {
                return false;
            };
            let Some(v_local) = piece_arr.buffer().get_as_f64(local_base + c) else {
                return false;
            };
            if (v_merged - v_local).abs() > INTERFACE_VALUE_TOL {
                return false;
            }
        }
    }
    true
}

fn ensure_in_pgrid(in_pgrid: bool, tag: &str, pos: u64) -> Result<(), IoError> {
    if !in_pgrid {
        return Err(IoError::InvalidVtuStructure {
            offset: pos,
            reason: format!("<{tag}> appeared outside <PUnstructuredGrid> in .pvtu"),
        });
    }
    Ok(())
}

fn validate_pvtu_vtk_file_attrs(e: &BytesStart<'_>, pos: u64) -> Result<(), IoError> {
    let mut file_type = None;
    for attr in e.attributes() {
        let attr = attr.map_err(|err| IoError::XmlSyntax {
            offset: pos,
            message: err.to_string(),
        })?;
        let val = std::str::from_utf8(&attr.value).map_err(|err| IoError::XmlSyntax {
            offset: pos,
            message: err.to_string(),
        })?;
        if attr.key.as_ref() == b"type" {
            file_type = Some(val.trim().to_string());
        }
    }
    match file_type.as_deref() {
        Some("PUnstructuredGrid") => Ok(()),
        Some(other) => Err(IoError::InvalidVtuStructure {
            offset: pos,
            reason: format!("expected VTKFile type=\"PUnstructuredGrid\", got \"{other}\""),
        }),
        None => Err(IoError::InvalidVtuStructure {
            offset: pos,
            reason: "VTKFile in .pvtu is missing required 'type' attribute".to_string(),
        }),
    }
}

fn parse_pgrid_ghost_level(e: &BytesStart<'_>, pos: u64) -> Result<u32, IoError> {
    let mut ghost_level = 0u32;
    for attr in e.attributes() {
        let attr = attr.map_err(|err| IoError::XmlSyntax {
            offset: pos,
            message: err.to_string(),
        })?;
        let val = std::str::from_utf8(&attr.value).map_err(|err| IoError::XmlSyntax {
            offset: pos,
            message: err.to_string(),
        })?;
        if attr.key.as_ref() == b"GhostLevel" {
            ghost_level = val
                .trim()
                .parse::<u32>()
                .map_err(|_| IoError::InvalidNumber {
                    array: "PUnstructuredGrid::GhostLevel".to_string(),
                    text: val.to_string(),
                    offset: pos,
                })?;
        }
    }
    Ok(ghost_level)
}

fn parse_pdata_array_attrs(
    e: &BytesStart<'_>,
    pos: u64,
    require_name: bool,
) -> Result<Option<PvtuDataArrayMeta>, IoError> {
    let mut name: Option<String> = None;
    let mut scalar_type: Option<String> = None;
    let mut number_of_components = 1usize;

    for attr in e.attributes() {
        let attr = attr.map_err(|err| IoError::XmlSyntax {
            offset: pos,
            message: err.to_string(),
        })?;
        let val = std::str::from_utf8(&attr.value).map_err(|err| IoError::XmlSyntax {
            offset: pos,
            message: err.to_string(),
        })?;
        match attr.key.as_ref() {
            b"Name" => {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    name = Some(trimmed.to_string());
                }
            }
            b"type" => {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    scalar_type = Some(trimmed.to_string());
                }
            }
            b"NumberOfComponents" => {
                number_of_components =
                    val.trim()
                        .parse::<usize>()
                        .map_err(|_| IoError::InvalidNumber {
                            array: "PDataArray::NumberOfComponents".to_string(),
                            text: val.to_string(),
                            offset: pos,
                        })?;
                if number_of_components == 0 {
                    return Err(IoError::InvalidNumber {
                        array: "PDataArray::NumberOfComponents".to_string(),
                        text: val.to_string(),
                        offset: pos,
                    });
                }
            }
            _ => {}
        }
    }

    let scalar_type = scalar_type.ok_or_else(|| IoError::InvalidVtuStructure {
        offset: pos,
        reason: "<PDataArray> in .pvtu is missing required 'type' attribute".to_string(),
    })?;

    if require_name {
        let name = name.ok_or_else(|| IoError::InvalidVtuStructure {
            offset: pos,
            reason: "<PDataArray> in .pvtu is missing required 'Name' attribute".to_string(),
        })?;
        if name.eq_ignore_ascii_case(GHOST_ARRAY_NAME) {
            return Ok(None);
        }
        Ok(Some(PvtuDataArrayMeta {
            name,
            scalar_type,
            number_of_components,
        }))
    } else {
        Ok(None)
    }
}

fn parse_piece_source(e: &BytesStart<'_>, pos: u64) -> Result<PvtuPiece, IoError> {
    let mut source: Option<String> = None;
    for attr in e.attributes() {
        let attr = attr.map_err(|err| IoError::XmlSyntax {
            offset: pos,
            message: err.to_string(),
        })?;
        let val = std::str::from_utf8(&attr.value).map_err(|err| IoError::XmlSyntax {
            offset: pos,
            message: err.to_string(),
        })?;
        if attr.key.as_ref() == b"Source" {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                source = Some(trimmed.to_string());
            }
        }
    }
    let source = source.ok_or_else(|| IoError::InvalidVtuStructure {
        offset: pos,
        reason: "<Piece> in .pvtu is missing required non-empty 'Source' attribute".to_string(),
    })?;
    Ok(PvtuPiece { source })
}

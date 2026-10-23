//! VTK XML `.vtu` (`UnstructuredGrid`) reader and writer.

use std::io::Write;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use tetraglass_core::{
    ArrayBuffer, CellType, ComponentCount, DataArray, PolyhedronFaces, ScalarType, UnstructuredGrid,
};

use crate::error::IoError;
use crate::vtu_codec::{
    AppendedEncoding, ByteOrder, Compressor, HeaderType, VtuConfig, VtuDataType,
    convert_vtu23_polyhedra_to_stream, decode_appended_slice, decode_bytes_data_array,
    decode_bytes_f64, decode_bytes_u32_indices, decode_inline_binary, parse_ascii_data_array,
    parse_ascii_f64, parse_ascii_u8, parse_ascii_u32_indices,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    Root,
    PointData,
    CellData,
    Points,
    Cells,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DataFormat {
    Ascii,
    Binary,
    Appended { offset: usize },
}

#[derive(Debug, Clone)]
struct RawDataArray {
    name: String,
    dtype: VtuDataType,
    components: usize,
    format: DataFormat,
    inline_text: String,
    xml_offset: u64,
}

impl RawDataArray {
    fn decode_raw_bytes(
        &self,
        cfg: VtuConfig,
        appended: Option<(AppendedEncoding, &[u8])>,
    ) -> Result<Vec<u8>, IoError> {
        match self.format {
            DataFormat::Ascii => Err(IoError::InvalidVtuStructure {
                offset: self.xml_offset,
                reason: "decode_raw_bytes called on ASCII DataArray".to_string(),
            }),
            DataFormat::Binary => {
                decode_inline_binary(&self.name, &self.inline_text, self.xml_offset, cfg)
            }
            DataFormat::Appended { offset } => {
                let (encoding, payload) = appended.ok_or_else(|| IoError::InvalidVtuStructure {
                    offset: self.xml_offset,
                    reason: format!(
                        "DataArray '{}' has format=\"appended\" but <AppendedData> is missing",
                        self.name
                    ),
                })?;
                decode_appended_slice(&self.name, payload, offset, self.xml_offset, encoding, cfg)
            }
        }
    }

    fn into_f64_vec(
        self,
        cfg: VtuConfig,
        appended: Option<(AppendedEncoding, &[u8])>,
    ) -> Result<Vec<f64>, IoError> {
        if self.format == DataFormat::Ascii {
            parse_ascii_f64(&self.name, &self.inline_text, self.xml_offset)
        } else {
            let bytes = self.decode_raw_bytes(cfg, appended)?;
            decode_bytes_f64(
                &self.name,
                &bytes,
                self.dtype,
                cfg.byte_order,
                self.xml_offset,
            )
        }
    }

    fn into_u32_indices(
        self,
        cfg: VtuConfig,
        appended: Option<(AppendedEncoding, &[u8])>,
    ) -> Result<Vec<u32>, IoError> {
        if self.format == DataFormat::Ascii {
            parse_ascii_u32_indices(&self.name, &self.inline_text, self.xml_offset)
        } else {
            let bytes = self.decode_raw_bytes(cfg, appended)?;
            decode_bytes_u32_indices(
                &self.name,
                &bytes,
                self.dtype,
                cfg.byte_order,
                self.xml_offset,
            )
        }
    }

    fn into_u8_vec(
        self,
        cfg: VtuConfig,
        appended: Option<(AppendedEncoding, &[u8])>,
    ) -> Result<Vec<u8>, IoError> {
        if self.format == DataFormat::Ascii {
            parse_ascii_u8(&self.name, &self.inline_text, self.xml_offset)
        } else {
            let bytes = self.decode_raw_bytes(cfg, appended)?;
            if self.dtype != VtuDataType::UInt8 && self.dtype != VtuDataType::Int8 {
                let idxs = decode_bytes_u32_indices(
                    &self.name,
                    &bytes,
                    self.dtype,
                    cfg.byte_order,
                    self.xml_offset,
                )?;
                idxs.into_iter()
                    .map(|v| {
                        u8::try_from(v).map_err(|_| IoError::BinaryData {
                            array: self.name.clone(),
                            offset: self.xml_offset,
                            reason: format!("cell type id {v} overflows u8"),
                        })
                    })
                    .collect()
            } else {
                Ok(bytes)
            }
        }
    }

    fn into_data_array(
        self,
        cfg: VtuConfig,
        appended: Option<(AppendedEncoding, &[u8])>,
    ) -> Result<(String, DataArray), IoError> {
        let comp = ComponentCount::try_from(self.components)?;
        let arr = if self.format == DataFormat::Ascii {
            parse_ascii_data_array(
                &self.name,
                self.dtype,
                comp,
                &self.inline_text,
                self.xml_offset,
            )?
        } else {
            let bytes = self.decode_raw_bytes(cfg, appended)?;
            decode_bytes_data_array(
                &self.name,
                &bytes,
                self.dtype,
                comp,
                cfg.byte_order,
                self.xml_offset,
            )?
        };
        Ok((self.name, arr))
    }
}

/// Reads a VTK XML `.vtu` (`UnstructuredGrid`) mesh from a byte slice.
///
/// Supports `ascii`, `binary` (inline base64), and `appended` (`base64` and `raw`) data formats,
/// both uncompressed and compressed with `vtkZLibDataCompressor`.
pub fn read_vtu(bytes: &[u8]) -> Result<UnstructuredGrid, IoError> {
    let appended = extract_appended_data(bytes)?;

    let mut reader = Reader::from_reader(bytes);
    reader.config_mut().trim_text(true);

    let mut buf = Vec::new();
    let mut section = Section::Root;

    let mut saw_vtk_file = false;
    let mut closed_vtk_file = false;
    let mut saw_piece = false;
    let mut expected_points = 0usize;
    let mut expected_cells = 0usize;

    let mut cfg = VtuConfig {
        header_type: HeaderType::UInt32,
        byte_order: ByteOrder::LittleEndian,
        compressor: Compressor::None,
    };

    let mut current_array: Option<RawDataArray> = None;
    let mut inside_info_key_depth = 0usize;

    let mut points_array: Option<RawDataArray> = None;
    let mut cell_arrays: Vec<RawDataArray> = Vec::new();
    let mut point_data_arrays: Vec<RawDataArray> = Vec::new();
    let mut cell_data_arrays: Vec<RawDataArray> = Vec::new();

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
            Event::Start(ref e) => {
                let tag = e.name();
                match tag.as_ref() {
                    b"VTKFile" => {
                        saw_vtk_file = true;
                        parse_vtk_file_attrs(e, pos, &mut cfg)?;
                    }
                    b"Piece" => {
                        if saw_piece {
                            return Err(IoError::UnsupportedFeature {
                                offset: pos,
                                feature: "multiple <Piece> elements in a single .vtu".to_string(),
                            });
                        }
                        saw_piece = true;
                        let (np, nc) = parse_piece_attrs(e, pos)?;
                        expected_points = np;
                        expected_cells = nc;
                    }
                    b"PointData" => section = Section::PointData,
                    b"CellData" => section = Section::CellData,
                    b"Points" => section = Section::Points,
                    b"Cells" => section = Section::Cells,
                    b"DataArray" => {
                        current_array = Some(parse_data_array_start(e, pos)?);
                        inside_info_key_depth = 0;
                    }
                    b"InformationKey" | b"Value" if current_array.is_some() => {
                        inside_info_key_depth += 1;
                    }
                    b"AppendedData" => {
                        verify_closing_tags_after_appended(bytes, pos as usize)?;
                        closed_vtk_file = true;
                        break;
                    }
                    _ => {}
                }
            }
            Event::Empty(ref e) => {
                let tag = e.name();
                match tag.as_ref() {
                    b"Piece" => {
                        if saw_piece {
                            return Err(IoError::UnsupportedFeature {
                                offset: pos,
                                feature: "multiple <Piece> elements in a single .vtu".to_string(),
                            });
                        }
                        saw_piece = true;
                        let (np, nc) = parse_piece_attrs(e, pos)?;
                        expected_points = np;
                        expected_cells = nc;
                    }
                    b"DataArray" => {
                        let arr = parse_data_array_start(e, pos)?;
                        store_finished_array(
                            section,
                            arr,
                            pos,
                            &mut points_array,
                            &mut cell_arrays,
                            &mut point_data_arrays,
                            &mut cell_data_arrays,
                        )?;
                    }
                    _ => {}
                }
            }
            Event::Text(ref t) => {
                if inside_info_key_depth == 0
                    && let Some(ref mut arr) = current_array
                {
                    let text = std::str::from_utf8(t.as_ref()).map_err(|e| IoError::XmlSyntax {
                        offset: pos,
                        message: e.to_string(),
                    })?;
                    if !arr.inline_text.is_empty() {
                        arr.inline_text.push(' ');
                    }
                    arr.inline_text.push_str(text);
                }
            }
            Event::End(ref e) => {
                let tag = e.name();
                match tag.as_ref() {
                    b"InformationKey" | b"Value" if inside_info_key_depth > 0 => {
                        inside_info_key_depth -= 1;
                    }
                    b"DataArray" => {
                        if let Some(arr) = current_array.take() {
                            store_finished_array(
                                section,
                                arr,
                                pos,
                                &mut points_array,
                                &mut cell_arrays,
                                &mut point_data_arrays,
                                &mut cell_data_arrays,
                            )?;
                        }
                    }
                    b"PointData" | b"CellData" | b"Points" | b"Cells" => {
                        section = Section::Root;
                    }
                    b"VTKFile" => {
                        closed_vtk_file = true;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        buf.clear();
    }

    if !saw_vtk_file || !closed_vtk_file || !saw_piece {
        return Err(IoError::InvalidVtuStructure {
            offset: bytes.len() as u64,
            reason: "incomplete or truncated .vtu XML document".to_string(),
        });
    }

    let app_ref = appended.as_ref().map(|(enc, data)| (*enc, data.as_slice()));

    // 1. Points
    let points = if expected_points == 0 {
        Vec::new()
    } else {
        let pts_raw = points_array.ok_or_else(|| IoError::InvalidVtuStructure {
            offset: 0,
            reason: "missing <Points><DataArray>".to_string(),
        })?;
        if pts_raw.components != 3 {
            return Err(IoError::InvalidVtuStructure {
                offset: pts_raw.xml_offset,
                reason: format!(
                    "Points DataArray must have NumberOfComponents=\"3\", got {}",
                    pts_raw.components
                ),
            });
        }
        let pts_offset = pts_raw.xml_offset;
        let flat = pts_raw.into_f64_vec(cfg, app_ref)?;
        if flat.len() != expected_points * 3 {
            return Err(IoError::InvalidVtuStructure {
                offset: pts_offset,
                reason: format!(
                    "Points DataArray has {} values, expected {} (3 * {expected_points})",
                    flat.len(),
                    expected_points * 3
                ),
            });
        }
        flat.as_chunks::<3>().0.to_vec()
    };

    // 2. Cells
    let mut conn_raw = None;
    let mut offsets_raw = None;
    let mut types_raw = None;
    let mut faces_raw = None;
    let mut faceoffsets_raw = None;
    let mut face_conn_raw = None;
    let mut face_offs_raw = None;
    let mut poly_to_faces_raw = None;
    let mut poly_offs_raw = None;

    for arr in cell_arrays {
        match arr.name.as_str() {
            "connectivity" => conn_raw = Some(arr),
            "offsets" => offsets_raw = Some(arr),
            "types" => types_raw = Some(arr),
            "faces" => faces_raw = Some(arr),
            "faceoffsets" => faceoffsets_raw = Some(arr),
            "face_connectivity" => face_conn_raw = Some(arr),
            "face_offsets" => face_offs_raw = Some(arr),
            "polyhedron_to_faces" => poly_to_faces_raw = Some(arr),
            "polyhedron_offsets" => poly_offs_raw = Some(arr),
            _ => {}
        }
    }

    let (cell_types, offsets, connectivity, polyhedron_faces) = if expected_cells == 0 {
        (Vec::new(), Vec::new(), Vec::new(), None)
    } else {
        let conn_arr = conn_raw.ok_or_else(|| IoError::InvalidVtuStructure {
            offset: 0,
            reason: "missing <Cells> DataArray 'connectivity'".to_string(),
        })?;
        let offs_arr = offsets_raw.ok_or_else(|| IoError::InvalidVtuStructure {
            offset: 0,
            reason: "missing <Cells> DataArray 'offsets'".to_string(),
        })?;
        let types_arr = types_raw.ok_or_else(|| IoError::InvalidVtuStructure {
            offset: 0,
            reason: "missing <Cells> DataArray 'types'".to_string(),
        })?;
        let types_xml_offset = types_arr.xml_offset;

        let connectivity = conn_arr.into_u32_indices(cfg, app_ref)?;
        let offsets = offs_arr.into_u32_indices(cfg, app_ref)?;
        let raw_types = types_arr.into_u8_vec(cfg, app_ref)?;

        if offsets.len() != expected_cells || raw_types.len() != expected_cells {
            return Err(IoError::InvalidVtuStructure {
                offset: types_xml_offset,
                reason: format!(
                    "Piece declares NumberOfCells=\"{expected_cells}\", but offsets has {} and types has {}",
                    offsets.len(),
                    raw_types.len()
                ),
            });
        }

        let mut cell_types = Vec::with_capacity(expected_cells);
        let mut prev_off = 0usize;
        for (cell_idx, (&vtk_id, &end_u32)) in raw_types.iter().zip(offsets.iter()).enumerate() {
            let end = end_u32 as usize;
            if end < prev_off {
                return Err(tetraglass_core::CoreError::NonMonotonicOffsets {
                    cell_index: cell_idx,
                    prev: prev_off as u32,
                    current: end_u32,
                }
                .into());
            }
            let node_count = end - prev_off;
            cell_types.push(CellType::from_vtk_type_and_node_count(vtk_id, node_count)?);
            prev_off = end;
        }

        let poly_faces = if let (Some(f_arr), Some(fo_arr)) = (faces_raw, faceoffsets_raw) {
            let faces = f_arr.into_u32_indices(cfg, app_ref)?;
            let face_offsets = fo_arr.into_u32_indices(cfg, app_ref)?;
            Some(PolyhedronFaces::new(face_offsets, faces))
        } else if let (Some(fc), Some(fo), Some(ptf), Some(po)) = (
            face_conn_raw,
            face_offs_raw,
            poly_to_faces_raw,
            poly_offs_raw,
        ) {
            let xml_off = fc.xml_offset;
            let face_conn = fc.into_u32_indices(cfg, app_ref)?;
            let face_offs = fo.into_u32_indices(cfg, app_ref)?;
            let poly_to_faces = ptf.into_u32_indices(cfg, app_ref)?;
            let poly_offs = po.into_u32_indices(cfg, app_ref)?;
            let (face_offsets, faces) = convert_vtu23_polyhedra_to_stream(
                &raw_types,
                &face_conn,
                &face_offs,
                &poly_to_faces,
                &poly_offs,
                xml_off,
            )?;
            Some(PolyhedronFaces::new(face_offsets, faces))
        } else {
            None
        };

        (cell_types, offsets, connectivity, poly_faces)
    };

    let mut grid =
        UnstructuredGrid::new(points, cell_types, offsets, connectivity, polyhedron_faces)?;

    for raw_arr in point_data_arrays {
        let (name, data_arr) = raw_arr.into_data_array(cfg, app_ref)?;
        grid.insert_point_data(name, data_arr)?;
    }

    for raw_arr in cell_data_arrays {
        let (name, data_arr) = raw_arr.into_data_array(cfg, app_ref)?;
        grid.insert_cell_data(name, data_arr)?;
    }

    Ok(grid)
}

/// Reads a `.vtu` file from a filesystem path (available on non-wasm32 targets only).
#[cfg(not(target_arch = "wasm32"))]
pub fn read_vtu_from_path(path: impl AsRef<Path>) -> Result<UnstructuredGrid, IoError> {
    let bytes = std::fs::read(path)?;
    read_vtu(&bytes)
}

/// Writes an [`UnstructuredGrid`] as an ASCII VTK XML `.vtu` stream.
pub fn write_vtu(grid: &UnstructuredGrid, mut writer: impl Write) -> Result<(), IoError> {
    writeln!(writer, "<?xml version=\"1.0\"?>")?;
    writeln!(
        writer,
        "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\">"
    )?;
    writeln!(writer, "  <UnstructuredGrid>")?;
    writeln!(
        writer,
        "    <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">",
        grid.num_points(),
        grid.num_cells()
    )?;

    writeln!(writer, "      <PointData>")?;
    for (name, arr) in grid.point_data().iter() {
        write_data_array(&mut writer, name, arr)?;
    }
    writeln!(writer, "      </PointData>")?;

    writeln!(writer, "      <CellData>")?;
    for (name, arr) in grid.cell_data().iter() {
        write_data_array(&mut writer, name, arr)?;
    }
    writeln!(writer, "      </CellData>")?;

    writeln!(writer, "      <Points>")?;
    writeln!(
        writer,
        "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">"
    )?;
    write!(writer, "         ")?;
    for &[x, y, z] in grid.points() {
        write!(writer, " {x} {y} {z}")?;
    }
    writeln!(writer)?;
    writeln!(writer, "        </DataArray>")?;
    writeln!(writer, "      </Points>")?;

    writeln!(writer, "      <Cells>")?;
    write_u32_slice(&mut writer, "Int64", "connectivity", grid.connectivity())?;
    write_u32_slice(&mut writer, "Int64", "offsets", grid.offsets())?;

    writeln!(
        writer,
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
    )?;
    write!(writer, "         ")?;
    for &ct in grid.cell_types() {
        write!(writer, " {}", ct.vtk_type_id())?;
    }
    writeln!(writer)?;
    writeln!(writer, "        </DataArray>")?;

    if let Some(pf) = grid.polyhedron_faces() {
        write_u32_slice(&mut writer, "Int64", "faces", pf.faces())?;
        write_u32_slice(&mut writer, "Int64", "faceoffsets", pf.face_offsets())?;
    }
    writeln!(writer, "      </Cells>")?;

    writeln!(writer, "    </Piece>")?;
    writeln!(writer, "  </UnstructuredGrid>")?;
    writeln!(writer, "</VTKFile>")?;
    Ok(())
}

/// Serializes an [`UnstructuredGrid`] into an in-memory `.vtu` byte vector.
pub fn write_vtu_to_bytes(grid: &UnstructuredGrid) -> Result<Vec<u8>, IoError> {
    let mut buf = Vec::new();
    write_vtu(grid, &mut buf)?;
    Ok(buf)
}

fn write_u32_slice(
    writer: &mut impl Write,
    vtk_type: &str,
    name: &str,
    values: &[u32],
) -> Result<(), IoError> {
    writeln!(
        writer,
        "        <DataArray type=\"{vtk_type}\" Name=\"{name}\" format=\"ascii\">"
    )?;
    write!(writer, "         ")?;
    for &v in values {
        write!(writer, " {v}")?;
    }
    writeln!(writer)?;
    writeln!(writer, "        </DataArray>")?;
    Ok(())
}

fn write_data_array(writer: &mut impl Write, name: &str, arr: &DataArray) -> Result<(), IoError> {
    let vtk_type = match arr.scalar_type() {
        ScalarType::F32 => "Float32",
        ScalarType::F64 => "Float64",
        ScalarType::I32 => "Int32",
        ScalarType::I64 => "Int64",
        ScalarType::U8 => "UInt8",
    };
    let ncomp = arr.components().as_u8();
    if ncomp == 1 {
        writeln!(
            writer,
            "        <DataArray type=\"{vtk_type}\" Name=\"{name}\" format=\"ascii\">"
        )?;
    } else {
        writeln!(
            writer,
            "        <DataArray type=\"{vtk_type}\" Name=\"{name}\" NumberOfComponents=\"{ncomp}\" format=\"ascii\">"
        )?;
    }
    write!(writer, "         ")?;
    match arr.buffer() {
        ArrayBuffer::F32(b) => {
            for &v in b.iter() {
                write!(writer, " {v}")?;
            }
        }
        ArrayBuffer::F64(b) => {
            for &v in b.iter() {
                write!(writer, " {v}")?;
            }
        }
        ArrayBuffer::I32(b) => {
            for &v in b.iter() {
                write!(writer, " {v}")?;
            }
        }
        ArrayBuffer::I64(b) => {
            for &v in b.iter() {
                write!(writer, " {v}")?;
            }
        }
        ArrayBuffer::U8(b) => {
            for &v in b.iter() {
                write!(writer, " {v}")?;
            }
        }
    }
    writeln!(writer)?;
    writeln!(writer, "        </DataArray>")?;
    Ok(())
}

fn parse_vtk_file_attrs(e: &BytesStart<'_>, pos: u64, cfg: &mut VtuConfig) -> Result<(), IoError> {
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
        match attr.key.as_ref() {
            b"type" => file_type = Some(val.to_string()),
            b"byte_order" => match val {
                "LittleEndian" => cfg.byte_order = ByteOrder::LittleEndian,
                "BigEndian" => cfg.byte_order = ByteOrder::BigEndian,
                other => {
                    return Err(IoError::UnsupportedFeature {
                        offset: pos,
                        feature: format!("byte_order '{other}'"),
                    });
                }
            },
            b"header_type" => match val {
                "UInt32" => cfg.header_type = HeaderType::UInt32,
                "UInt64" => cfg.header_type = HeaderType::UInt64,
                other => {
                    return Err(IoError::UnsupportedFeature {
                        offset: pos,
                        feature: format!("header_type '{other}'"),
                    });
                }
            },
            b"compressor" => match val {
                "" => cfg.compressor = Compressor::None,
                "vtkZLibDataCompressor" => cfg.compressor = Compressor::ZLib,
                other => {
                    return Err(IoError::UnsupportedFeature {
                        offset: pos,
                        feature: format!("compressor '{other}'"),
                    });
                }
            },
            _ => {}
        }
    }
    match file_type.as_deref() {
        Some("UnstructuredGrid") => Ok(()),
        Some(other) => Err(IoError::InvalidVtuStructure {
            offset: pos,
            reason: format!("expected VTKFile type=\"UnstructuredGrid\", got \"{other}\""),
        }),
        None => Err(IoError::InvalidVtuStructure {
            offset: pos,
            reason: "VTKFile is missing required 'type' attribute".to_string(),
        }),
    }
}

fn parse_piece_attrs(e: &BytesStart<'_>, pos: u64) -> Result<(usize, usize), IoError> {
    let mut num_points = None;
    let mut num_cells = None;
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
            b"NumberOfPoints" => {
                num_points =
                    Some(
                        val.trim()
                            .parse::<usize>()
                            .map_err(|_| IoError::InvalidNumber {
                                array: "Piece::NumberOfPoints".to_string(),
                                text: val.to_string(),
                                offset: pos,
                            })?,
                    );
            }
            b"NumberOfCells" => {
                num_cells =
                    Some(
                        val.trim()
                            .parse::<usize>()
                            .map_err(|_| IoError::InvalidNumber {
                                array: "Piece::NumberOfCells".to_string(),
                                text: val.to_string(),
                                offset: pos,
                            })?,
                    );
            }
            _ => {}
        }
    }
    let np = num_points.ok_or_else(|| IoError::InvalidVtuStructure {
        offset: pos,
        reason: "Piece missing NumberOfPoints".to_string(),
    })?;
    let nc = num_cells.ok_or_else(|| IoError::InvalidVtuStructure {
        offset: pos,
        reason: "Piece missing NumberOfCells".to_string(),
    })?;
    Ok((np, nc))
}

fn parse_data_array_start(e: &BytesStart<'_>, pos: u64) -> Result<RawDataArray, IoError> {
    let mut name = String::new();
    let mut dtype = None;
    let mut components = 1usize;
    let mut format_str = "ascii".to_string();
    let mut offset_attr = None;

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
            b"Name" => name = val.to_string(),
            b"type" => dtype = Some(VtuDataType::parse(val.trim(), pos)?),
            b"NumberOfComponents" => {
                components = val
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| IoError::InvalidNumber {
                        array: name.clone(),
                        text: val.to_string(),
                        offset: pos,
                    })?;
            }
            b"format" => format_str = val.trim().to_string(),
            b"offset" => {
                offset_attr =
                    Some(
                        val.trim()
                            .parse::<usize>()
                            .map_err(|_| IoError::InvalidNumber {
                                array: name.clone(),
                                text: val.to_string(),
                                offset: pos,
                            })?,
                    );
            }
            _ => {}
        }
    }

    let dtype = dtype.ok_or_else(|| IoError::InvalidVtuStructure {
        offset: pos,
        reason: format!("DataArray '{name}' is missing 'type' attribute"),
    })?;

    let format = match format_str.as_str() {
        "ascii" => DataFormat::Ascii,
        "binary" => DataFormat::Binary,
        "appended" => {
            let offset = offset_attr.ok_or_else(|| IoError::InvalidVtuStructure {
                offset: pos,
                reason: format!(
                    "DataArray '{name}' has format=\"appended\" but no offset attribute"
                ),
            })?;
            DataFormat::Appended { offset }
        }
        other => {
            return Err(IoError::UnsupportedFeature {
                offset: pos,
                feature: format!("DataArray format '{other}'"),
            });
        }
    };

    Ok(RawDataArray {
        name,
        dtype,
        components,
        format,
        inline_text: String::new(),
        xml_offset: pos,
    })
}

fn store_finished_array(
    section: Section,
    arr: RawDataArray,
    pos: u64,
    points_array: &mut Option<RawDataArray>,
    cell_arrays: &mut Vec<RawDataArray>,
    point_data_arrays: &mut Vec<RawDataArray>,
    cell_data_arrays: &mut Vec<RawDataArray>,
) -> Result<(), IoError> {
    match section {
        Section::Points => {
            if points_array.is_some() {
                return Err(IoError::InvalidVtuStructure {
                    offset: pos,
                    reason: "multiple DataArrays inside <Points>".to_string(),
                });
            }
            *points_array = Some(arr);
        }
        Section::Cells => cell_arrays.push(arr),
        Section::PointData => point_data_arrays.push(arr),
        Section::CellData => cell_data_arrays.push(arr),
        Section::Root => {}
    }
    Ok(())
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn extract_appended_data(bytes: &[u8]) -> Result<Option<(AppendedEncoding, Vec<u8>)>, IoError> {
    let Some(start_tag_pos) = find_subslice(bytes, b"<AppendedData") else {
        return Ok(None);
    };
    let after_start = &bytes[start_tag_pos..];
    let gt_rel = after_start.iter().position(|&b| b == b'>').ok_or_else(|| {
        IoError::InvalidVtuStructure {
            offset: start_tag_pos as u64,
            reason: "unclosed <AppendedData> start tag".to_string(),
        }
    })?;
    let header_tag =
        std::str::from_utf8(&after_start[..=gt_rel]).map_err(|e| IoError::XmlSyntax {
            offset: start_tag_pos as u64,
            message: e.to_string(),
        })?;
    let encoding = if header_tag.contains("encoding=\"raw\"") {
        AppendedEncoding::Raw
    } else if header_tag.contains("encoding=\"base64\"") || !header_tag.contains("encoding=") {
        AppendedEncoding::Base64
    } else {
        return Err(IoError::UnsupportedFeature {
            offset: start_tag_pos as u64,
            feature: format!("AppendedData tag '{header_tag}'"),
        });
    };

    let body_start = start_tag_pos + gt_rel + 1;
    let body_and_tail = &bytes[body_start..];
    let end_rel = find_subslice(body_and_tail, b"</AppendedData>").ok_or_else(|| {
        IoError::InvalidVtuStructure {
            offset: body_start as u64,
            reason: "missing </AppendedData> closing tag".to_string(),
        }
    })?;
    let body = &body_and_tail[..end_rel];
    let underscore_rel = body
        .iter()
        .position(|&b| !b.is_ascii_whitespace())
        .ok_or_else(|| IoError::InvalidVtuStructure {
            offset: body_start as u64,
            reason: "empty <AppendedData> section (missing '_' prefix)".to_string(),
        })?;
    if body[underscore_rel] != b'_' {
        return Err(IoError::InvalidVtuStructure {
            offset: (body_start + underscore_rel) as u64,
            reason: "AppendedData payload does not start with '_'".to_string(),
        });
    }
    Ok(Some((encoding, body[underscore_rel + 1..].to_vec())))
}

fn verify_closing_tags_after_appended(bytes: &[u8], pos: usize) -> Result<(), IoError> {
    let tail = &bytes[pos.min(bytes.len())..];
    let Some(end_app) = find_subslice(tail, b"</AppendedData>") else {
        return Err(IoError::InvalidVtuStructure {
            offset: pos as u64,
            reason: "missing </AppendedData>".to_string(),
        });
    };
    let after_app = &tail[end_app + b"</AppendedData>".len()..];
    if find_subslice(after_app, b"</VTKFile>").is_none() {
        return Err(IoError::InvalidVtuStructure {
            offset: (pos + end_app) as u64,
            reason: "missing </VTKFile> closing tag after </AppendedData>".to_string(),
        });
    }
    Ok(())
}

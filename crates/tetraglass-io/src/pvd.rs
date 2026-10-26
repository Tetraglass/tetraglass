//! ParaView XML data collection (`.pvd`) parser and lazy time-series reader (`M1.7`).
//!
//! A `.pvd` file is an XML manifest listing time steps and relative dataset paths:
//!
//! ```xml
//! <?xml version="1.0"?>
//! <VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
//!   <Collection>
//!     <DataSet timestep="0.0" group="" part="0" file="series_0.vtu"/>
//!     <DataSet timestep="0.5" group="" part="0" file="series_1.vtu"/>
//!   </Collection>
//! </VTKFile>
//! ```
//!
//! [`read_pvd_series_with_loader`] (for byte buffers / WASM) and [`read_pvd_series_from_path`]
//! (for native paths) return a [`TimeSeries<IoError>`] that parses only the requested step's
//! mesh file on demand.

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;

use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};
use tetraglass_core::{StepLoader, TimeSeries, UnstructuredGrid};

use crate::error::IoError;
use crate::{
    MeshFormat, detect_format_from_bytes, detect_format_from_extension, read_msh, read_vtk,
    read_vtu,
};

/// A single `<DataSet>` entry inside a ParaView `.pvd` `<Collection>`.
#[derive(Debug, Clone, PartialEq)]
pub struct PvdEntry {
    /// Simulation time value (`timestep` attribute).
    pub timestep: f64,
    /// Optional group identifier (`group` attribute, empty string by default).
    pub group: String,
    /// Part index (`part` attribute, `0` by default).
    pub part: u32,
    /// Relative path to the step's mesh file (`file` attribute).
    pub file: String,
}

/// Parses the `<DataSet>` entries from a ParaView `.pvd` XML byte buffer.
pub fn parse_pvd_entries(bytes: &[u8]) -> Result<Vec<PvdEntry>, IoError> {
    let mut reader = Reader::from_reader(bytes);
    reader.config_mut().trim_text(true);

    let mut buf = Vec::new();
    let mut saw_vtk_file = false;
    let mut closed_vtk_file = false;
    let mut in_collection = false;
    let mut saw_collection = false;
    let mut closed_collection = false;
    let mut entries = Vec::new();

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
                    validate_pvd_vtk_file_attrs(e, pos)?;
                }
                b"Collection" => {
                    if !saw_vtk_file {
                        return Err(IoError::InvalidPvdStructure {
                            offset: pos,
                            reason: "<Collection> appeared outside <VTKFile>".to_string(),
                        });
                    }
                    in_collection = true;
                    saw_collection = true;
                }
                b"DataSet" => {
                    if !in_collection {
                        return Err(IoError::InvalidPvdStructure {
                            offset: pos,
                            reason: "<DataSet> appeared outside <Collection>".to_string(),
                        });
                    }
                    entries.push(parse_dataset_attrs(e, pos)?);
                }
                _ => {}
            },
            Event::Empty(ref e) => match e.name().as_ref() {
                b"DataSet" => {
                    if !in_collection {
                        return Err(IoError::InvalidPvdStructure {
                            offset: pos,
                            reason: "<DataSet/> appeared outside <Collection>".to_string(),
                        });
                    }
                    entries.push(parse_dataset_attrs(e, pos)?);
                }
                b"Collection" => {
                    if !saw_vtk_file {
                        return Err(IoError::InvalidPvdStructure {
                            offset: pos,
                            reason: "<Collection/> appeared outside <VTKFile>".to_string(),
                        });
                    }
                    saw_collection = true;
                    closed_collection = true;
                }
                _ => {}
            },
            Event::End(ref e) => match e.name().as_ref() {
                b"Collection" => {
                    in_collection = false;
                    closed_collection = true;
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
        return Err(IoError::InvalidPvdStructure {
            offset: reader.buffer_position(),
            reason: "missing or unclosed <VTKFile> root element in .pvd".to_string(),
        });
    }
    if !saw_collection || !closed_collection {
        return Err(IoError::InvalidPvdStructure {
            offset: reader.buffer_position(),
            reason: "missing or unclosed <Collection> element in .pvd".to_string(),
        });
    }
    if entries.is_empty() {
        return Err(IoError::InvalidPvdStructure {
            offset: reader.buffer_position(),
            reason: "<Collection> in .pvd contains no <DataSet> entries".to_string(),
        });
    }

    Ok(entries)
}

/// Builds a lazy [`TimeSeries<IoError>`] from `.pvd` XML bytes and a byte-fetching callback
/// `load_step_bytes` that receives each step's relative `file` path only when that step is
/// requested.
///
/// This function is pure-Rust and works on both native targets and `wasm32-unknown-unknown` (`D-008`).
pub fn read_pvd_series_with_loader<F>(
    pvd_bytes: &[u8],
    load_step_bytes: F,
) -> Result<TimeSeries<IoError>, IoError>
where
    F: Fn(&str) -> Result<Vec<u8>, IoError> + Send + Sync + 'static,
{
    let entries = parse_pvd_entries(pvd_bytes)?;
    let shared_loader = Arc::new(load_step_bytes);

    let mut steps: Vec<(f64, StepLoader<IoError>)> = Vec::with_capacity(entries.len());
    for entry in entries {
        let file = entry.file;
        let loader_cb = Arc::clone(&shared_loader);
        let step_loader: StepLoader<IoError> = Arc::new(move || {
            let step_bytes = loader_cb(&file)?;
            parse_step_mesh_bytes(&file, &step_bytes)
        });
        steps.push((entry.timestep, step_loader));
    }

    Ok(TimeSeries::from_boxed_steps(steps)?)
}

/// Reads a ParaView `.pvd` collection from a native file path as a lazy [`TimeSeries<IoError>`].
///
/// Only the `.pvd` XML manifest is read immediately; individual `.vtu` / `.msh` step files
/// are read and parsed on demand when [`TimeSeries::step`] is called.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_pvd_series_from_path(path: impl AsRef<Path>) -> Result<TimeSeries<IoError>, IoError> {
    let pvd_path = path.as_ref();
    let pvd_bytes = std::fs::read(pvd_path)?;
    let base_dir: PathBuf = pvd_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    read_pvd_series_with_loader(&pvd_bytes, move |rel_file| {
        let rel = Path::new(rel_file);
        let step_path = if rel.is_absolute() || base_dir.as_os_str().is_empty() {
            rel.to_path_buf()
        } else {
            base_dir.join(rel)
        };
        Ok(std::fs::read(step_path)?)
    })
}

/// Reads the first time step of a ParaView `.pvd` collection from a native file path.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_pvd_from_path(path: impl AsRef<Path>) -> Result<UnstructuredGrid, IoError> {
    let series = read_pvd_series_from_path(path)?;
    let first = series.step(0)?;
    Ok((*first).clone())
}

fn parse_step_mesh_bytes(file_name: &str, bytes: &[u8]) -> Result<UnstructuredGrid, IoError> {
    let ext = file_name
        .rsplit_once('.')
        .map(|(_, e)| e)
        .unwrap_or_default();
    let format = detect_format_from_extension(ext)
        .or_else(|| detect_format_from_bytes(bytes))
        .unwrap_or(MeshFormat::Vtu);

    match format {
        MeshFormat::Vtu => read_vtu(bytes),
        MeshFormat::VtkLegacy => read_vtk(bytes),
        MeshFormat::Gmsh => read_msh(bytes),
        other => Err(IoError::InvalidPvdStructure {
            offset: 0,
            reason: format!(
                "unsupported step file format `{other:?}` referenced by .pvd (`{file_name}`)"
            ),
        }),
    }
}

fn validate_pvd_vtk_file_attrs(e: &BytesStart<'_>, pos: u64) -> Result<(), IoError> {
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
        Some("Collection") => Ok(()),
        Some(other) => Err(IoError::InvalidPvdStructure {
            offset: pos,
            reason: format!("expected VTKFile type=\"Collection\", got \"{other}\""),
        }),
        None => Err(IoError::InvalidPvdStructure {
            offset: pos,
            reason: "VTKFile in .pvd is missing required 'type' attribute".to_string(),
        }),
    }
}

fn parse_dataset_attrs(e: &BytesStart<'_>, pos: u64) -> Result<PvdEntry, IoError> {
    let mut timestep: Option<f64> = None;
    let mut group = String::new();
    let mut part = 0u32;
    let mut file: Option<String> = None;

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
            b"timestep" => {
                let t = val
                    .trim()
                    .parse::<f64>()
                    .map_err(|_| IoError::InvalidNumber {
                        array: "DataSet::timestep".to_string(),
                        text: val.to_string(),
                        offset: pos,
                    })?;
                if !t.is_finite() {
                    return Err(IoError::InvalidNumber {
                        array: "DataSet::timestep".to_string(),
                        text: val.to_string(),
                        offset: pos,
                    });
                }
                timestep = Some(t);
            }
            b"group" => group = val.to_string(),
            b"part" => {
                part = val
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| IoError::InvalidNumber {
                        array: "DataSet::part".to_string(),
                        text: val.to_string(),
                        offset: pos,
                    })?;
                if part > 0 {
                    return Err(IoError::UnsupportedFeature {
                        offset: pos,
                        feature: format!("multi-part .pvd DataSet (part=\"{part}\")"),
                    });
                }
            }
            b"file" => {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    file = Some(trimmed.to_string());
                }
            }
            _ => {}
        }
    }

    let timestep = timestep.ok_or_else(|| IoError::InvalidPvdStructure {
        offset: pos,
        reason: "<DataSet> in .pvd is missing required 'timestep' attribute".to_string(),
    })?;
    let file = file.ok_or_else(|| IoError::InvalidPvdStructure {
        offset: pos,
        reason: "<DataSet> in .pvd is missing required non-empty 'file' attribute".to_string(),
    })?;

    Ok(PvdEntry {
        timestep,
        group,
        part,
        file,
    })
}

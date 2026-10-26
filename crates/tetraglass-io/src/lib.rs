//! Readers and writers for VTK and Gmsh files, working on byte buffers.
//!
//! `tetraglass-io` depends only on `tetraglass-core` and pure-Rust parsing/compression crates,
//! so the same readers run natively and on `wasm32-unknown-unknown` (`D-008`, `D-018`).
//!
//! # Supported formats
//!
//! - **VTK XML UnstructuredGrid (`.vtu`):** [`read_vtu`], [`write_vtu`], [`write_vtu_to_bytes`]
//!   (and [`read_vtu_from_path`] on native targets).
//!   - Encodings: `ascii`, `binary` (inline base64), and `appended` (`base64` and `raw`).
//!   - Compression: uncompressed and `vtkZLibDataCompressor` (`UInt32` and `UInt64` headers,
//!     `LittleEndian` and `BigEndian`).
//!   - Cell types: linear (`1, 3, 5, 9, 10, 12, 13, 14`), quadratic (`21..=27`), arbitrary-order
//!     Lagrange (`68..=73`), and polyhedra (`42`, supporting both classic `faces`/`faceoffsets`
//!     and VTK 9.4+ `face_connectivity`/`face_offsets`/`polyhedron_to_faces`/`polyhedron_offsets`).
//! - **Legacy VTK DataFile (`.vtk`):** [`read_vtk`], [`read_vtk_series`] (and
//!   [`read_vtk_from_path`] on native targets).
//!   - Versions: Legacy VTK `1.0` through `5.1` (`DATASET UNSTRUCTURED_GRID`), both `ASCII`
//!     and big-endian `BINARY` encodings.
//!   - Cell layouts: classic VTK 2.0–4.2 `CELLS n size` and VTK 5.0–5.1 `OFFSETS` + `CONNECTIVITY`.
//!   - Cell types: linear (`1, 3, 5, 9, 10, 12, 13, 14`), quadratic (`21..=27`), arbitrary-order
//!     Lagrange (`68..=73`), and polyhedra (`42`, inline `CELLS` face streams and `FACES`/`FACE_LOCATIONS`).
//!   - Attributes: `POINT_DATA` and `CELL_DATA` (`SCALARS`, `VECTORS`, `NORMALS`, `TENSORS`,
//!     `TENSORS6`, `TEXTURE_COORDINATES`, and `FIELD` data tables).
//! - **ParaView XML data collection (`.pvd`):** [`parse_pvd_entries`],
//!   [`read_pvd_series_with_loader`] (and [`read_pvd_series_from_path`], [`read_pvd_from_path`]
//!   on native targets) producing a lazy [`tetraglass_core::TimeSeries`].
//! - **Gmsh MSH (`.msh`):** [`read_msh`], [`read_msh_series`] (and [`read_msh_from_path`],
//!   [`read_msh_series_from_path`] on native targets).
//!   - Versions: MSH `4.1` and `2.2`, both `ascii` (`0`) and `binary` (`1`, little- and big-endian).
//!   - Post-processing data: `$NodeData` and `$ElementData` mapped to `point_data` and `cell_data`
//!     (with multi-step `$NodeData` / `$ElementData` loaded lazily per step in [`read_msh_series`]).
//!   - Cell types & node ordering: linear (`1..=7, 15`), quadratic serendipity (`8, 9, 11, 16..=19`),
//!     complete quadratic (`10, 12, 13`), and cubic (`21, 26`) elements permuted into VTK canonical
//!     node ordering via [`gmsh_element_spec`].
//!
//! # Examples
//!
//! Round-tripping an [`UnstructuredGrid`](tetraglass_core::UnstructuredGrid) through in-memory
//! `.vtu` bytes (works identically on native and `wasm32-unknown-unknown`):
//!
//! ```
//! use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
//! use tetraglass_io::{read_vtu, write_vtu_to_bytes};
//!
//! let grid = SyntheticGridSpec::new(2, 2, 2, SyntheticCellLayout::Hexahedra)
//!     .generate()
//!     .expect("valid synthetic grid");
//! let vtu_bytes = write_vtu_to_bytes(&grid)?;
//! let loaded = read_vtu(&vtu_bytes)?;
//!
//! assert_eq!(loaded.num_points(), grid.num_points());
//! assert_eq!(loaded.num_cells(), grid.num_cells());
//! assert!(loaded.point_data().get("linear").is_some());
//! # Ok::<(), tetraglass_io::IoError>(())
//! ```

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;

use tetraglass_core::{StepLoader, TimeSeries};

mod error;
mod msh;
mod pvd;
mod vtk;
mod vtu;
mod vtu_codec;

pub use error::IoError;
pub use msh::{GmshElementSpec, gmsh_element_spec, read_msh, read_msh_series};
#[cfg(not(target_arch = "wasm32"))]
pub use msh::{read_msh_from_path, read_msh_series_from_path};
pub use pvd::{PvdEntry, parse_pvd_entries, read_pvd_series_with_loader};
#[cfg(not(target_arch = "wasm32"))]
pub use pvd::{read_pvd_from_path, read_pvd_series_from_path};
#[cfg(not(target_arch = "wasm32"))]
pub use vtk::read_vtk_from_path;
pub use vtk::{read_vtk, read_vtk_series};
#[cfg(not(target_arch = "wasm32"))]
pub use vtu::read_vtu_from_path;
pub use vtu::{read_vtu, write_vtu, write_vtu_to_bytes};

/// Wraps a `.vtu` byte buffer in a single-step lazy [`TimeSeries<IoError>`] at `t = 0.0`.
pub fn read_vtu_series(bytes: impl Into<Arc<[u8]>>) -> Result<TimeSeries<IoError>, IoError> {
    let shared: Arc<[u8]> = bytes.into();
    let loader: StepLoader<IoError> = Arc::new(move || read_vtu(&shared));
    Ok(TimeSeries::from_boxed_steps(vec![(0.0, loader)])?)
}

/// Opens a `.vtu`, `.vtk`, `.pvd`, or `.msh` file from a native path as a lazy [`TimeSeries<IoError>`].
#[cfg(not(target_arch = "wasm32"))]
pub fn read_series_from_path(
    path: impl AsRef<Path>,
    explicit_format: Option<MeshFormat>,
) -> Result<(MeshFormat, TimeSeries<IoError>), IoError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let format = explicit_format
        .or_else(|| {
            path.extension()
                .and_then(|e| e.to_str())
                .and_then(detect_format_from_extension)
        })
        .or_else(|| detect_format_from_bytes(&bytes))
        .ok_or_else(|| IoError::InvalidVtuStructure {
            offset: 0,
            reason: format!("cannot determine mesh format for `{}`", path.display()),
        })?;

    match format {
        MeshFormat::Vtu => Ok((format, read_vtu_series(bytes)?)),
        MeshFormat::VtkLegacy => Ok((format, read_vtk_series(bytes)?)),
        MeshFormat::Gmsh => Ok((format, read_msh_series(bytes)?)),
        MeshFormat::Pvd => {
            let base_dir = path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            let series = read_pvd_series_with_loader(&bytes, move |rel_file| {
                let rel = Path::new(rel_file);
                let step_path = if rel.is_absolute() || base_dir.as_os_str().is_empty() {
                    rel.to_path_buf()
                } else {
                    base_dir.join(rel)
                };
                Ok(std::fs::read(step_path)?)
            })?;
            Ok((format, series))
        }
    }
}

/// Mesh file formats recognized by `tetraglass-io`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeshFormat {
    /// VTK XML UnstructuredGrid (`.vtu`).
    Vtu,
    /// ParaView XML data collection (`.pvd`).
    Pvd,
    /// Legacy VTK file (`.vtk`).
    VtkLegacy,
    /// Gmsh MSH 2.2 or 4.1 file (`.msh`).
    Gmsh,
}

/// Detects the [`MeshFormat`] from a file extension (`vtu`, `pvd`, `vtk`, `msh`, case-insensitive).
#[must_use]
pub fn detect_format_from_extension(ext: &str) -> Option<MeshFormat> {
    if ext.eq_ignore_ascii_case("vtu") {
        Some(MeshFormat::Vtu)
    } else if ext.eq_ignore_ascii_case("pvd") {
        Some(MeshFormat::Pvd)
    } else if ext.eq_ignore_ascii_case("vtk") {
        Some(MeshFormat::VtkLegacy)
    } else if ext.eq_ignore_ascii_case("msh") {
        Some(MeshFormat::Gmsh)
    } else {
        None
    }
}

/// Detects the [`MeshFormat`] by inspecting the leading bytes of a file buffer.
#[must_use]
pub fn detect_format_from_bytes(bytes: &[u8]) -> Option<MeshFormat> {
    let prefix = &bytes[..bytes.len().min(512)];
    if prefix
        .windows(b"$MeshFormat".len())
        .any(|w| w == b"$MeshFormat")
    {
        return Some(MeshFormat::Gmsh);
    }
    if prefix
        .windows(b"# vtk DataFile".len())
        .any(|w| w == b"# vtk DataFile")
    {
        return Some(MeshFormat::VtkLegacy);
    }
    if prefix.windows(b"<VTKFile".len()).any(|w| w == b"<VTKFile") {
        if prefix
            .windows(b"UnstructuredGrid".len())
            .any(|w| w == b"UnstructuredGrid")
        {
            return Some(MeshFormat::Vtu);
        }
        if prefix
            .windows(b"Collection".len())
            .any(|w| w == b"Collection")
        {
            return Some(MeshFormat::Pvd);
        }
    }
    None
}

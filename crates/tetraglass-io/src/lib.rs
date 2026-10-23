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
//! - **Gmsh MSH (`.msh`):** [`read_msh`] (and [`read_msh_from_path`] on native targets).
//!   - Versions: MSH `4.1` and `2.2`, both `ascii` (`0`) and `binary` (`1`, little- and big-endian).
//!   - Post-processing data: `$NodeData` and `$ElementData` mapped to `point_data` and `cell_data`.
//!   - Cell types & node ordering: linear (`1..=7, 15`), quadratic serendipity (`8, 9, 11, 16..=19`),
//!     complete quadratic (`10, 12, 13`), and cubic (`21, 26`) elements permuted into VTK canonical
//!     node ordering via [`gmsh_element_spec`].

mod error;
mod msh;
mod vtu;
mod vtu_codec;

pub use error::IoError;
#[cfg(not(target_arch = "wasm32"))]
pub use msh::read_msh_from_path;
pub use msh::{GmshElementSpec, gmsh_element_spec, read_msh};
#[cfg(not(target_arch = "wasm32"))]
pub use vtu::read_vtu_from_path;
pub use vtu::{read_vtu, write_vtu, write_vtu_to_bytes};

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

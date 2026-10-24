//! Linearization, surface extraction and visualization filters on tetrahedral views.
//!
//! `tetraglass-filters` operates purely on `tetraglass-core` types and compiles for
//! `wasm32-unknown-unknown` with no GPU or filesystem dependencies (`D-008`).
//!
//! See `docs/design-plan.md` in the repository for how this crate fits the architecture.

mod basis;
mod clip;
mod contour;
mod linearize;
mod marching_tets;
mod slice;
mod surface;
mod threshold;

pub use clip::{ClipKeep, ClipMode, ClipParams, clip, clip_grid, clip_with_chunk_size};
pub use contour::{
    ContourComponent, ContourParams, contour, contour_grid, contour_with_chunk_size,
};
pub use linearize::{DEFAULT_SUBDIVISIONS, LinearizeParams, linearize, linearize_default};
pub use marching_tets::{
    DEFAULT_TET_CHUNK_SIZE, EdgeKey, clip_crinkle_tets_with_chunk_size, clip_tets_with_chunk_size,
    extract_isosurface, extract_isosurfaces_with_chunk_size,
    filter_tets_by_source_cell_with_chunk_size,
};
pub use slice::{SliceParams, slice, slice_grid, slice_with_chunk_size};
pub use surface::{
    FilterError, extract_external_surface, extract_external_surface_with_subdivisions,
    extract_tet_view_surface,
};
pub use tetraglass_core::{SurfaceMesh, TetView};
pub use threshold::{
    ThresholdAssociation, ThresholdComponent, ThresholdParams, threshold, threshold_grid,
    threshold_with_chunk_size,
};

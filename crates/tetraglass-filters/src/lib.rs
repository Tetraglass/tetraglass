//! Linearization, surface extraction and visualization filters on tetrahedral views.
//!
//! `tetraglass-filters` operates purely on `tetraglass-core` types and compiles for
//! `wasm32-unknown-unknown` with no GPU or filesystem dependencies (`D-008`).
//!
//! See `docs/design-plan.md` in the repository for how this crate fits the architecture.
//!
//! # Examples
//!
//! Linearizing a hexahedral mesh into a [`TetView`] and running [`slice()`], [`contour`],
//! [`clip`], and [`threshold`]:
//!
//! ```
//! use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
//! use tetraglass_filters::{
//!     ClipKeep, ClipMode, ClipParams, ContourParams, SliceParams, TetLocator, ThresholdParams,
//!     clip, contour, extract_tet_view_surface, linearize_default, slice, threshold,
//! };
//!
//! let grid = SyntheticGridSpec::new(4, 4, 4, SyntheticCellLayout::Hexahedra).generate()?;
//! let view = linearize_default(&grid)?;
//! assert_eq!(view.num_tets(), grid.num_cells() * 6);
//!
//! // BVH cell locator for 3D point location and field interpolation
//! let locator = TetLocator::new(&view);
//! let hit = locator.locate([0.25, 0.5, 0.75]).expect("point inside domain");
//! assert!(hit.source_cell_id < grid.num_cells() as u32);
//!
//! // Plane slice across the linearized tetrahedral view
//! let cut_surface = slice(&view, &SliceParams::new([0.0, 0.0, 1.0]))?;
//! assert!(cut_surface.num_triangles() > 0);
//!
//! // Isosurface contour on the "linear" scalar field
//! let iso_surface = contour(&view, &ContourParams::new("linear", vec![3.0]))?;
//! assert!(iso_surface.num_triangles() > 0);
//!
//! // Exact half-space volume clip followed by boundary surface extraction
//! let clipped = clip(&view, &ClipParams::new([1.0, 0.0, 0.0]).keep(ClipKeep::Back).mode(ClipMode::Exact))?;
//! let clipped_surface = extract_tet_view_surface(&clipped)?;
//! assert!(clipped_surface.num_triangles() > 0);
//!
//! // Whole-cell scalar threshold
//! let threshed = threshold(&view, &ThresholdParams::new("linear", Some(0.0), Some(3.5)))?;
//! assert!(threshed.num_tets() > 0 && threshed.num_tets() < view.num_tets());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod basis;
mod clip;
mod contour;
mod linearize;
mod locator;
mod marching_tets;
mod slice;
mod surface;
mod threshold;

pub use clip::{ClipKeep, ClipMode, ClipParams, clip, clip_grid, clip_with_chunk_size};
pub use contour::{
    ContourComponent, ContourParams, contour, contour_grid, contour_with_chunk_size,
};
pub use linearize::{DEFAULT_SUBDIVISIONS, LinearizeParams, linearize, linearize_default};
pub use locator::{
    BvhNode, DEFAULT_LOCATOR_CHUNK_SIZE, DEFAULT_LOCATOR_TOLERANCE, DEFAULT_MAX_LEAF_TETS,
    TetLocation, TetLocator, tet_barycentric_coordinates,
};
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

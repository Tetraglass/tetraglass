//! Linearization, surface extraction and visualization filters on tetrahedral views.
//!
//! `tetraglass-filters` operates purely on `tetraglass-core` types and compiles for
//! `wasm32-unknown-unknown` with no GPU or filesystem dependencies (`D-008`).
//!
//! See `docs/design-plan.md` in the repository for how this crate fits the architecture.

mod basis;
mod linearize;
mod surface;

pub use linearize::{DEFAULT_SUBDIVISIONS, LinearizeParams, linearize, linearize_default};
pub use surface::{
    FilterError, extract_external_surface, extract_external_surface_with_subdivisions,
};
pub use tetraglass_core::{SurfaceMesh, TetView};

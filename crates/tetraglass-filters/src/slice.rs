//! Plane slice filter on [`TetView`] (`M2.3a`, `D-005`, `D-006`).
//!
//! Computes the signed distance of every point in the linearized tetrahedral view to a plane
//! `(origin, normal)` and extracts the `0.0` isosurface via the shared marching-tetrahedra kernel.

use tetraglass_core::{SurfaceMesh, TetView, UnstructuredGrid};

use crate::linearize::{LinearizeParams, linearize};
use crate::marching_tets::{DEFAULT_TET_CHUNK_SIZE, extract_isosurfaces_with_chunk_size};
use crate::surface::FilterError;

/// Parameters for the plane [`slice()`] filter (`docs/scene-format.md`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliceParams {
    /// Point on the slice plane in world coordinates. When `None`, defaults to the center of the
    /// input [`TetView`]'s bounding box (`[0.0, 0.0, 0.0]` if empty).
    pub origin: Option<[f64; 3]>,
    /// Outward normal vector of the slice plane (must be finite and non-zero).
    pub normal: [f64; 3],
}

impl SliceParams {
    /// Creates [`SliceParams`] with `normal` and default `origin = None` (center of dataset bounds).
    #[must_use]
    pub const fn new(normal: [f64; 3]) -> Self {
        Self {
            origin: None,
            normal,
        }
    }

    /// Creates [`SliceParams`] with an explicit `origin` and `normal`.
    #[must_use]
    pub const fn with_origin(origin: [f64; 3], normal: [f64; 3]) -> Self {
        Self {
            origin: Some(origin),
            normal,
        }
    }

    /// Sets an explicit plane `origin`.
    #[must_use]
    pub const fn origin(mut self, origin: [f64; 3]) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Validates `self` and returns `(resolved_origin, unit_normal)`.
    pub fn resolve(&self, view: &TetView) -> Result<([f64; 3], [f64; 3]), FilterError> {
        let [nx, ny, nz] = self.normal;
        if !nx.is_finite() || !ny.is_finite() || !nz.is_finite() {
            return Err(FilterError::InvalidPlaneNormal {
                normal: self.normal,
            });
        }
        let len_sq = nx * nx + ny * ny + nz * nz;
        if len_sq <= 1e-30 {
            return Err(FilterError::InvalidPlaneNormal {
                normal: self.normal,
            });
        }
        let inv_len = 1.0 / len_sq.sqrt();
        let unit_normal = [nx * inv_len, ny * inv_len, nz * inv_len];

        let resolved_origin = match self.origin {
            Some(orig) => {
                if !orig[0].is_finite() || !orig[1].is_finite() || !orig[2].is_finite() {
                    return Err(FilterError::InvalidPlaneOrigin { origin: orig });
                }
                orig
            }
            None => view.bounds().map_or([0.0, 0.0, 0.0], |b| b.center()),
        };

        Ok((resolved_origin, unit_normal))
    }
}

/// Cuts a linearized [`TetView`] with a plane (`params`), returning a watertight [`SurfaceMesh`]
/// with all point arrays linearly interpolated along cut edges and all cell arrays mapped back
/// to the original [`UnstructuredGrid`] cell indices (`D-005`, `D-006`).
///
/// # Examples
///
/// ```
/// use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
/// use tetraglass_filters::{SliceParams, linearize_default, slice};
///
/// let grid = SyntheticGridSpec::new(4, 4, 4, SyntheticCellLayout::Hexahedra)
///     .generate()
///     .unwrap();
/// let view = linearize_default(&grid).unwrap();
///
/// // Slice the unit cube [0, 1]^3 at its center with normal +Z; the cross-section has area 1.0.
/// let cut = slice(&view, &SliceParams::new([0.0, 0.0, 1.0])).unwrap();
/// assert!((cut.total_area() - 1.0).abs() < 1e-12);
/// ```
pub fn slice(view: &TetView, params: &SliceParams) -> Result<SurfaceMesh, FilterError> {
    slice_with_chunk_size(view, params, DEFAULT_TET_CHUNK_SIZE)
}

/// Cuts a linearized [`TetView`] with a plane (`params`) using an explicit parallel `chunk_size`
/// (number of tetrahedra per chunk).
pub fn slice_with_chunk_size(
    view: &TetView,
    params: &SliceParams,
    chunk_size: usize,
) -> Result<SurfaceMesh, FilterError> {
    let (origin, unit_normal) = params.resolve(view)?;
    let scalars: Vec<f64> = view
        .points()
        .iter()
        .map(|&p| {
            (p[0] - origin[0]) * unit_normal[0]
                + (p[1] - origin[1]) * unit_normal[1]
                + (p[2] - origin[2]) * unit_normal[2]
        })
        .collect();

    extract_isosurfaces_with_chunk_size(view, &scalars, &[0.0], chunk_size)
}

/// Convenience helper that linearizes `grid` with `linearize_params` and slices the resulting [`TetView`].
pub fn slice_grid(
    grid: &UnstructuredGrid,
    params: &SliceParams,
    linearize_params: LinearizeParams,
) -> Result<SurfaceMesh, FilterError> {
    let view = linearize(grid, &linearize_params)?;
    slice(&view, params)
}

//! Plane clip filter on [`TetView`] (`M2.5`, `D-005`, `D-006`).
//!
//! Computes the signed distance of every point in the linearized tetrahedral view to a plane
//! `(origin, normal)` and clips the volume to the `"front"` (`>= 0`) or `"back"` (`<= 0`)
//! half-space in either `"exact"` mode (conformal marching-tetrahedra prism decomposition) or
//! `"crinkle"` mode (keeping whole original source cells).

use tetraglass_core::{TetView, UnstructuredGrid};

use crate::linearize::{LinearizeParams, linearize};
use crate::marching_tets::{
    DEFAULT_TET_CHUNK_SIZE, clip_crinkle_tets_with_chunk_size, clip_tets_with_chunk_size,
};
use crate::surface::FilterError;

/// Which half-space to keep relative to the clip plane normal (`docs/scene-format.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ClipKeep {
    /// Keep the half-space in the direction of the plane normal (`(p - origin) · normal >= 0`).
    #[default]
    Front,
    /// Keep the half-space opposite to the plane normal (`(p - origin) · normal <= 0`).
    Back,
}

/// Clipping algorithm mode (`docs/scene-format.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ClipMode {
    /// Cut tetrahedra along the clip plane via the shared marching-tetrahedra kernel (`D-006`).
    #[default]
    Exact,
    /// Keep whole original source cells (`tet_source_cells`) that overlap the kept half-space.
    Crinkle,
}

/// Parameters for the plane [`clip`] filter (`docs/scene-format.md`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipParams {
    /// Point on the clip plane in world coordinates. When `None`, defaults to the center of the
    /// input [`TetView`]'s bounding box (`[0.0, 0.0, 0.0]` if empty).
    pub origin: Option<[f64; 3]>,
    /// Outward normal vector of the clip plane (must be finite and non-zero).
    pub normal: [f64; 3],
    /// Which half-space to keep relative to `normal` (default: [`ClipKeep::Front`]).
    pub keep: ClipKeep,
    /// Clipping mode: [`ClipMode::Exact`] (default) or [`ClipMode::Crinkle`].
    pub mode: ClipMode,
}

impl ClipParams {
    /// Creates [`ClipParams`] with `normal`, `origin = None` (center of dataset bounds),
    /// `keep = ClipKeep::Front`, and `mode = ClipMode::Exact`.
    #[must_use]
    pub const fn new(normal: [f64; 3]) -> Self {
        Self {
            origin: None,
            normal,
            keep: ClipKeep::Front,
            mode: ClipMode::Exact,
        }
    }

    /// Creates [`ClipParams`] with an explicit `origin` and `normal`.
    #[must_use]
    pub const fn with_origin(origin: [f64; 3], normal: [f64; 3]) -> Self {
        Self {
            origin: Some(origin),
            normal,
            keep: ClipKeep::Front,
            mode: ClipMode::Exact,
        }
    }

    /// Sets an explicit plane `origin`.
    #[must_use]
    pub const fn origin(mut self, origin: [f64; 3]) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Sets which half-space to keep (`Front` or `Back`).
    #[must_use]
    pub const fn keep(mut self, keep: ClipKeep) -> Self {
        self.keep = keep;
        self
    }

    /// Sets the clipping mode (`Exact` or `Crinkle`).
    #[must_use]
    pub const fn mode(mut self, mode: ClipMode) -> Self {
        self.mode = mode;
        self
    }

    /// Validates `self` and returns `(resolved_origin, effective_unit_normal)` oriented toward
    /// the kept half-space.
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
        let sign = match self.keep {
            ClipKeep::Front => 1.0,
            ClipKeep::Back => -1.0,
        };
        let inv_len = sign / len_sq.sqrt();
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

/// Clips a linearized [`TetView`] with a plane (`params`), returning a conformal [`TetView`]
/// with all point arrays linearly interpolated along cut edges and all cell arrays and
/// `tet_source_cells` mapped back to the originating [`UnstructuredGrid`] cells (`D-005`, `D-006`).
///
/// # Examples
///
/// ```
/// use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
/// use tetraglass_filters::{ClipParams, clip, linearize_default};
///
/// let grid = SyntheticGridSpec::new(4, 4, 4, SyntheticCellLayout::Hexahedra)
///     .generate()
///     .unwrap();
/// let view = linearize_default(&grid).unwrap();
///
/// // Clip the unit cube [0, 1]^3 at its center with normal +X; the kept half has volume 0.5.
/// let clipped = clip(&view, &ClipParams::new([1.0, 0.0, 0.0])).unwrap();
/// assert!((clipped.total_volume() - 0.5).abs() < 1e-12);
/// ```
pub fn clip(view: &TetView, params: &ClipParams) -> Result<TetView, FilterError> {
    clip_with_chunk_size(view, params, DEFAULT_TET_CHUNK_SIZE)
}

/// Clips a linearized [`TetView`] with a plane (`params`) using an explicit parallel `chunk_size`
/// (number of tetrahedra per chunk).
pub fn clip_with_chunk_size(
    view: &TetView,
    params: &ClipParams,
    chunk_size: usize,
) -> Result<TetView, FilterError> {
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

    match params.mode {
        ClipMode::Exact => clip_tets_with_chunk_size(view, &scalars, 0.0, chunk_size),
        ClipMode::Crinkle => clip_crinkle_tets_with_chunk_size(view, &scalars, 0.0, chunk_size),
    }
}

/// Convenience helper that linearizes `grid` with `linearize_params`, clips the resulting
/// [`TetView`], and converts the clipped volume to an [`UnstructuredGrid`].
pub fn clip_grid(
    grid: &UnstructuredGrid,
    params: &ClipParams,
    linearize_params: LinearizeParams,
) -> Result<UnstructuredGrid, FilterError> {
    let view = linearize(grid, &linearize_params)?;
    let clipped = clip(&view, params)?;
    Ok(clipped.to_unstructured_grid()?)
}

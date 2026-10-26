//! `warp_by_vector` coordinate displacement filter (`M6.4`, `D-005`, `D-014`).
//!
//! Displaces the 3D vertex coordinates of a linearized [`TetView`] (or an [`UnstructuredGrid`])
//! by $\mathbf{p}'_i = \mathbf{p}_i + s \cdot \mathbf{v}_i$ in `f64` precision, where $s$ is a
//! finite `scale_factor` and $\mathbf{v}_i$ is a 3-component point vector attribute array
//! (`ComponentCount::Vector3`). All tetrahedron connectivity, originating cell provenance, and
//! point/cell [`tetraglass_core::Attributes`] are preserved zero-copy via `Arc`.

use tetraglass_core::{ComponentCount, DataArray, TetView, UnstructuredGrid};

use crate::linearize::{LinearizeParams, linearize};
use crate::surface::FilterError;

/// Default number of vertices processed per parallel worker chunk in [`warp_by_vector`].
pub const DEFAULT_WARP_CHUNK_SIZE: usize = 4096;

/// Parameters for the [`warp_by_vector`] filter (`M6.4`).
#[derive(Debug, Clone, PartialEq)]
pub struct WarpByVectorParams {
    /// Name of the 3-component (`ComponentCount::Vector3`) point attribute array used as displacement.
    pub vector_field: String,
    /// Multiplicative displacement scale factor $s$ in $\mathbf{p}'_i = \mathbf{p}_i + s \cdot \mathbf{v}_i$ (default `1.0`).
    pub scale_factor: f64,
}

impl WarpByVectorParams {
    /// Creates [`WarpByVectorParams`] for `vector_field` with the default `scale_factor` of `1.0`.
    #[must_use]
    pub fn new(vector_field: impl Into<String>) -> Self {
        Self {
            vector_field: vector_field.into(),
            scale_factor: 1.0,
        }
    }

    /// Creates [`WarpByVectorParams`] for `vector_field` with an explicit `scale_factor`.
    #[must_use]
    pub fn with_scale(vector_field: impl Into<String>, scale_factor: f64) -> Self {
        Self {
            vector_field: vector_field.into(),
            scale_factor,
        }
    }

    /// Sets the multiplicative displacement `scale_factor`.
    #[must_use]
    pub const fn scale(mut self, scale_factor: f64) -> Self {
        self.scale_factor = scale_factor;
        self
    }

    /// Validates that `scale_factor` is finite.
    pub fn validate(&self) -> Result<(), FilterError> {
        if !self.scale_factor.is_finite() {
            return Err(FilterError::InvalidWarpScaleFactor {
                scale_factor: self.scale_factor,
            });
        }
        Ok(())
    }
}

/// Displaces the vertex coordinates of `view` by `params.scale_factor * vector_field` in `f64` (`M6.4`, `D-005`).
///
/// # Examples
///
/// ```
/// use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
/// use tetraglass_filters::{WarpByVectorParams, linearize_default, warp_by_vector};
///
/// let grid = SyntheticGridSpec::new(2, 2, 2, SyntheticCellLayout::Hexahedra).generate()?;
/// let view = linearize_default(&grid)?;
///
/// // Displace vertices by 0.25 * velocity (where velocity = [-y, x, 0] on the synthetic grid)
/// let warped = warp_by_vector(&view, &WarpByVectorParams::with_scale("velocity", 0.25))?;
/// assert_eq!(warped.num_points(), view.num_points());
/// assert_eq!(warped.num_tets(), view.num_tets());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn warp_by_vector(view: &TetView, params: &WarpByVectorParams) -> Result<TetView, FilterError> {
    warp_by_vector_with_chunk_size(view, params, DEFAULT_WARP_CHUNK_SIZE)
}

/// Displaces the vertex coordinates of `view` using an explicit parallel vertex `chunk_size`.
///
/// Output coordinates are bit-for-bit identical across all `chunk_size` values and thread counts.
pub fn warp_by_vector_with_chunk_size(
    view: &TetView,
    params: &WarpByVectorParams,
    chunk_size: usize,
) -> Result<TetView, FilterError> {
    params.validate()?;
    let vec_arr = view.point_data().get(&params.vector_field).ok_or_else(|| {
        FilterError::MissingPointArray {
            name: params.vector_field.clone(),
        }
    })?;
    let warped_points = displace_points(
        view.points(),
        vec_arr,
        &params.vector_field,
        params.scale_factor,
        chunk_size,
    )?;

    Ok(TetView::new(
        warped_points,
        view.tets_arc(),
        view.tet_source_cells_arc(),
        view.point_data().clone(),
        view.cell_data().clone(),
    )?)
}

/// Linearizes `grid` with `linearize_params` and applies [`warp_by_vector`] (`M6.4`, `D-005`).
pub fn warp_by_vector_grid(
    grid: &UnstructuredGrid,
    params: &WarpByVectorParams,
    linearize_params: &LinearizeParams,
) -> Result<TetView, FilterError> {
    let view = linearize(grid, linearize_params)?;
    warp_by_vector(&view, params)
}

/// Displaces the vertex coordinates of an [`UnstructuredGrid`] by `params.scale_factor * vector_field`,
/// preserving all original cell types (including higher-order Lagrange and polyhedral cells) and
/// point/cell attributes (`M6.4`).
pub fn warp_unstructured_grid(
    grid: &UnstructuredGrid,
    params: &WarpByVectorParams,
) -> Result<UnstructuredGrid, FilterError> {
    params.validate()?;
    let vec_arr = grid.point_data().get(&params.vector_field).ok_or_else(|| {
        FilterError::MissingPointArray {
            name: params.vector_field.clone(),
        }
    })?;
    let warped_points = displace_points(
        grid.points(),
        vec_arr,
        &params.vector_field,
        params.scale_factor,
        DEFAULT_WARP_CHUNK_SIZE,
    )?;

    let mut out = UnstructuredGrid::new(
        warped_points,
        grid.cell_types().to_vec(),
        grid.offsets().to_vec(),
        grid.connectivity().to_vec(),
        grid.polyhedron_faces().cloned(),
    )?;
    for (name, arr) in grid.point_data().iter() {
        out.insert_point_data(name, arr.clone())?;
    }
    for (name, arr) in grid.cell_data().iter() {
        out.insert_cell_data(name, arr.clone())?;
    }
    Ok(out)
}

fn displace_points(
    points: &[[f64; 3]],
    vec_arr: &DataArray,
    array_name: &str,
    scale_factor: f64,
    chunk_size: usize,
) -> Result<Vec<[f64; 3]>, FilterError> {
    if vec_arr.components() != ComponentCount::Vector3 {
        return Err(FilterError::NonVectorArray {
            array: array_name.to_string(),
            num_components: vec_arr.components().as_usize(),
        });
    }

    let num_points = points.len();
    let eff_chunk = chunk_size.max(1);
    let mut out = vec![[0.0; 3]; num_points];

    #[cfg(not(target_arch = "wasm32"))]
    if num_points > eff_chunk {
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (chunk_idx, out_slice) in out.chunks_mut(eff_chunk).enumerate() {
                let start = chunk_idx * eff_chunk;
                let end = start + out_slice.len();
                let in_slice = &points[start..end];
                handles.push(s.spawn(move || {
                    displace_point_range(in_slice, vec_arr, start, scale_factor, out_slice);
                }));
            }
            for h in handles {
                let _ = h.join();
            }
        });
        return Ok(out);
    }

    displace_point_range(points, vec_arr, 0, scale_factor, &mut out);
    Ok(out)
}

fn displace_point_range(
    in_points: &[[f64; 3]],
    vec_arr: &DataArray,
    start_idx: usize,
    scale_factor: f64,
    out_points: &mut [[f64; 3]],
) {
    let buf = vec_arr.buffer();
    for (local_i, (&[px, py, pz], out_pt)) in
        in_points.iter().zip(out_points.iter_mut()).enumerate()
    {
        let base = (start_idx + local_i) * 3;
        let vx = buf.get_as_f64(base).unwrap_or(0.0);
        let vy = buf.get_as_f64(base + 1).unwrap_or(0.0);
        let vz = buf.get_as_f64(base + 2).unwrap_or(0.0);
        *out_pt = [
            px + scale_factor * vx,
            py + scale_factor * vy,
            pz + scale_factor * vz,
        ];
    }
}

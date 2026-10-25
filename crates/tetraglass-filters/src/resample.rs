//! Regular 3D grid resampling filter (`M4.4a`) over linearized tetrahedral views (`D-005`).
//!
//! [`resample`] samples a [`TetView`] onto a uniform 3D lattice of `dimensions = [nx, ny, nz]`
//! points spanning the dataset's [`BoundingBox`] (or an explicit custom bounding box), using
//! [`TetLocator::locate_with_hint`] along each X scanline to exploit spatial coherence.
//!
//! All point and cell [`DataArray`]s on the input [`TetView`] are interpolated into
//! [`RegularGrid::point_data`], and [`RegularGrid::inside_mask`] records `1` for grid points
//! that lie inside the tetrahedral domain and `0` for points outside (such as exterior regions
//! of clipped or non-convex meshes).

use tetraglass_core::{
    ArrayBuffer, Attributes, BoundingBox, DataArray, RegularGrid, TetView, UnstructuredGrid,
    try_u32_from_usize,
};

use crate::linearize::linearize_default;
use crate::locator::{TetLocation, TetLocator};
use crate::surface::FilterError;

/// Default number of Y-Z scanline rows processed per parallel worker chunk on native targets.
pub const DEFAULT_RESAMPLE_CHUNK_SIZE: usize = 64;

/// Configuration parameters for the [`resample`] filter (`M4.4a`).
#[derive(Debug, Clone, PartialEq)]
pub struct ResampleParams {
    /// Number of sample points `[nx, ny, nz]` along the X, Y, and Z axes (each `>= 2`).
    pub dimensions: [usize; 3],
    /// Optional explicit axis-aligned bounding box to sample. Defaults to `view.bounds()`.
    pub bounds: Option<BoundingBox>,
}

impl ResampleParams {
    /// Creates [`ResampleParams`] with `dimensions = [nx, ny, nz]` over the input dataset's bounds.
    #[must_use]
    pub const fn new(dimensions: [usize; 3]) -> Self {
        Self {
            dimensions,
            bounds: None,
        }
    }

    /// Overrides the axis-aligned [`BoundingBox`] sampled by the regular grid.
    #[must_use]
    pub const fn with_bounds(mut self, bounds: BoundingBox) -> Self {
        self.bounds = Some(bounds);
        self
    }
}

/// Resamples `view` onto a [`RegularGrid`] according to `params`, building a [`TetLocator`] on demand.
///
/// # Examples
///
/// ```
/// use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
/// use tetraglass_filters::{ResampleParams, linearize_default, resample};
///
/// let grid = SyntheticGridSpec::new(4, 4, 4, SyntheticCellLayout::Hexahedra).generate()?;
/// let view = linearize_default(&grid)?;
///
/// let volume = resample(&view, &ResampleParams::new([8, 8, 8]))?;
/// assert_eq!(volume.dimensions(), [8, 8, 8]);
/// assert_eq!(volume.num_inside_points(), 512);
/// assert!(volume.point_data().get("linear").is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn resample(view: &TetView, params: &ResampleParams) -> Result<RegularGrid, FilterError> {
    let locator = TetLocator::new(view);
    resample_with_locator(view, &locator, params)
}

/// Resamples `view` onto a [`RegularGrid`] reusing a pre-built [`TetLocator`].
pub fn resample_with_locator(
    view: &TetView,
    locator: &TetLocator,
    params: &ResampleParams,
) -> Result<RegularGrid, FilterError> {
    resample_with_chunk_size(view, locator, params, DEFAULT_RESAMPLE_CHUNK_SIZE)
}

/// Linearizes `grid` with default subdivision settings and resamples it onto a [`RegularGrid`].
pub fn resample_grid(
    grid: &UnstructuredGrid,
    params: &ResampleParams,
) -> Result<RegularGrid, FilterError> {
    let view = linearize_default(grid)?;
    resample(&view, params)
}

/// Resamples `view` onto a [`RegularGrid`] using an explicit parallel scanline `chunk_size`.
///
/// Output [`RegularGrid::inside_mask`] and [`RegularGrid::point_data`] are bit-for-bit
/// deterministic across all `chunk_size` values and CPU thread counts.
pub fn resample_with_chunk_size(
    view: &TetView,
    locator: &TetLocator,
    params: &ResampleParams,
    chunk_size: usize,
) -> Result<RegularGrid, FilterError> {
    let [nx, ny, nz] = params.dimensions;
    if nx < 2 || ny < 2 || nz < 2 {
        return Err(FilterError::InvalidResampleDimensions {
            dimensions: params.dimensions,
            reason: "each dimension [nx, ny, nz] must be at least 2",
        });
    }

    let num_points = nx.checked_mul(ny).and_then(|xy| xy.checked_mul(nz)).ok_or(
        FilterError::InvalidResampleDimensions {
            dimensions: params.dimensions,
            reason: "total grid point count overflows usize",
        },
    )?;
    let _ = try_u32_from_usize(num_points)?;

    let bounds = params
        .bounds
        .or_else(|| view.bounds())
        .unwrap_or(BoundingBox {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        });
    if !bounds.min.iter().all(|v| v.is_finite())
        || !bounds.max.iter().all(|v| v.is_finite())
        || bounds.min[0] > bounds.max[0]
        || bounds.min[1] > bounds.max[1]
        || bounds.min[2] > bounds.max[2]
    {
        return Err(FilterError::InvalidResampleDimensions {
            dimensions: params.dimensions,
            reason: "resample bounding box must have finite min <= max coordinates",
        });
    }

    let num_rows = ny * nz;
    let row_chunk_size = chunk_size.max(1);
    let points_per_row_chunk = row_chunk_size * nx;

    let mut locations: Vec<Option<TetLocation>> = vec![None; num_points];

    #[cfg(not(target_arch = "wasm32"))]
    if num_rows > row_chunk_size && view.num_tets() > 0 {
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (chunk_idx, out_slice) in locations.chunks_mut(points_per_row_chunk).enumerate() {
                let start_row = chunk_idx * row_chunk_size;
                handles.push(s.spawn(move || {
                    locate_row_range(
                        view,
                        locator,
                        params.dimensions,
                        bounds,
                        start_row,
                        out_slice,
                    );
                }));
            }
            for handle in handles {
                let _ = handle.join();
            }
        });
    } else if view.num_tets() > 0 {
        locate_row_range(view, locator, params.dimensions, bounds, 0, &mut locations);
    }

    #[cfg(target_arch = "wasm32")]
    if view.num_tets() > 0 {
        locate_row_range(view, locator, params.dimensions, bounds, 0, &mut locations);
    }

    let inside_mask: Vec<u8> = locations
        .iter()
        .map(|loc| u8::from(loc.is_some()))
        .collect();
    let first_inside_loc = locations.iter().flatten().copied().next();

    let mut point_data = Attributes::new(num_points);

    for (name, arr) in view.point_data().iter() {
        let sampled = interpolate_point_array_on_grid(view, arr, &locations, first_inside_loc)?;
        point_data.insert(name.to_string(), sampled)?;
    }

    for (name, arr) in view.cell_data().iter() {
        if point_data.get(name).is_some() {
            continue;
        }
        let sampled = sample_cell_array_on_grid(arr, &locations, first_inside_loc)?;
        point_data.insert(name.to_string(), sampled)?;
    }

    RegularGrid::new(params.dimensions, bounds, inside_mask, point_data).map_err(FilterError::from)
}

fn locate_row_range(
    view: &TetView,
    locator: &TetLocator,
    dimensions: [usize; 3],
    bounds: BoundingBox,
    start_row: usize,
    out_slice: &mut [Option<TetLocation>],
) {
    let [nx, ny, nz] = dimensions;
    let num_rows_in_slice = out_slice.len() / nx;
    let inv_nx = 1.0 / ((nx - 1) as f64);
    let inv_ny = 1.0 / ((ny - 1) as f64);
    let inv_nz = 1.0 / ((nz - 1) as f64);
    let dx = bounds.max[0] - bounds.min[0];
    let dy = bounds.max[1] - bounds.min[1];
    let dz = bounds.max[2] - bounds.min[2];

    let _ = view;
    for local_row in 0..num_rows_in_slice {
        let row_idx = start_row + local_row;
        let iy = row_idx % ny;
        let iz = row_idx / ny;
        let y = bounds.min[1] + (iy as f64) * inv_ny * dy;
        let z = bounds.min[2] + (iz as f64) * inv_nz * dz;

        let row_out = &mut out_slice[local_row * nx..(local_row + 1) * nx];
        let mut hint: Option<usize> = None;
        for (ix, slot) in row_out.iter_mut().enumerate() {
            let x = bounds.min[0] + (ix as f64) * inv_nx * dx;
            let hit = locator.locate_with_hint([x, y, z], hint);
            hint = hit.map(|loc| loc.tet_index);
            *slot = hit;
        }
    }
}

fn interpolate_point_array_on_grid(
    view: &TetView,
    arr: &DataArray,
    locations: &[Option<TetLocation>],
    first_inside_loc: Option<TetLocation>,
) -> Result<DataArray, FilterError> {
    let comps = arr.components().as_usize();
    let mut fallback = vec![0.0; comps];
    if let Some(loc) = first_inside_loc {
        let _ = loc.interpolate_point_tuple(view, arr, &mut fallback);
    }

    let mut values = Vec::with_capacity(locations.len() * comps);
    let mut tuple = vec![0.0; comps];
    for loc_opt in locations {
        if let Some(loc) = loc_opt
            && loc.interpolate_point_tuple(view, arr, &mut tuple)
        {
            values.extend_from_slice(&tuple);
        } else {
            values.extend_from_slice(&fallback);
        }
    }

    DataArray::new(arr.components(), ArrayBuffer::F64(values.into())).map_err(FilterError::from)
}

fn sample_cell_array_on_grid(
    arr: &DataArray,
    locations: &[Option<TetLocation>],
    first_inside_loc: Option<TetLocation>,
) -> Result<DataArray, FilterError> {
    let comps = arr.components().as_usize();
    let mut fallback = vec![0.0; comps];
    if let Some(loc) = first_inside_loc {
        for (c, slot) in fallback.iter_mut().enumerate() {
            *slot = loc.sample_cell_scalar(arr, c).unwrap_or(0.0);
        }
    }

    let mut values = Vec::with_capacity(locations.len() * comps);
    for loc_opt in locations {
        if let Some(loc) = loc_opt {
            for (c, &fb) in fallback.iter().enumerate() {
                values.push(loc.sample_cell_scalar(arr, c).unwrap_or(fb));
            }
        } else {
            values.extend_from_slice(&fallback);
        }
    }

    DataArray::new(arr.components(), ArrayBuffer::F64(values.into())).map_err(FilterError::from)
}

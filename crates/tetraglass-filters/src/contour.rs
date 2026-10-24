//! Multi-isovalue `contour` filter on [`TetView`] (`M2.4`, `D-005`, `D-006`).
//!
//! Extracts one or more isosurfaces $f(\mathbf{x}) = c_k$ from a continuous point attribute array
//! (scalar, vector/tensor component, or Euclidean magnitude) using the shared marching-tetrahedra
//! kernel ([`extract_isosurfaces_with_chunk_size`]). All point and cell [`tetraglass_core::Attributes`]
//! are carried and linearly interpolated onto the output [`SurfaceMesh`].

use tetraglass_core::{DataArray, SurfaceMesh, TetView, UnstructuredGrid};

use crate::linearize::{LinearizeParams, linearize};
use crate::marching_tets::{DEFAULT_TET_CHUNK_SIZE, extract_isosurfaces_with_chunk_size};
use crate::surface::FilterError;

/// Selects which scalar value per vertex is extracted from a [`DataArray`] for contouring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContourComponent {
    /// Zero-based component index (`0` for scalar arrays, `0..3` for 3-vectors, etc.).
    Component(usize),
    /// Euclidean norm $\sqrt{\sum_c v_c^2}$ across all components of each tuple.
    Magnitude,
}

impl Default for ContourComponent {
    fn default() -> Self {
        Self::Component(0)
    }
}

/// Parameters for the multi-isovalue [`contour`] filter (`M2.4`).
#[derive(Debug, Clone, PartialEq)]
pub struct ContourParams {
    /// Name of the point attribute array in [`TetView::point_data`].
    pub array_name: String,
    /// Component or magnitude selector on the point array (default: `Component(0)`).
    pub component: ContourComponent,
    /// Non-empty list of finite isovalues to extract.
    pub values: Vec<f64>,
}

impl ContourParams {
    /// Creates [`ContourParams`] for component `0` of point array `array_name` at `values`.
    #[must_use]
    pub fn new(array_name: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            array_name: array_name.into(),
            component: ContourComponent::Component(0),
            values,
        }
    }

    /// Creates [`ContourParams`] for an explicit [`ContourComponent`] of `array_name` at `values`.
    #[must_use]
    pub fn with_component(
        array_name: impl Into<String>,
        component: ContourComponent,
        values: Vec<f64>,
    ) -> Self {
        Self {
            array_name: array_name.into(),
            component,
            values,
        }
    }

    /// Creates [`ContourParams`] for the Euclidean magnitude of `array_name` at `values`.
    #[must_use]
    pub fn magnitude(array_name: impl Into<String>, values: Vec<f64>) -> Self {
        Self::with_component(array_name, ContourComponent::Magnitude, values)
    }
}

/// Extracts one or more isosurfaces from a linearized [`TetView`] (`M2.4`, `D-005`, `D-006`).
///
/// # Examples
///
/// ```
/// use tetraglass_core::{CellType, ComponentCount, DataArray, UnstructuredGrid};
/// use tetraglass_filters::{ContourParams, contour, linearize_default};
///
/// let points = vec![
///     [0.0, 0.0, 0.0],
///     [1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0],
/// ];
/// let mut grid = UnstructuredGrid::new(
///     points,
///     vec![CellType::Tetra],
///     vec![4],
///     vec![0, 1, 2, 3],
///     None,
/// )
/// .unwrap();
/// grid.insert_point_data(
///     "z_field",
///     DataArray::from_f64(ComponentCount::Scalar, vec![0.0, 0.0, 0.0, 1.0]).unwrap(),
/// )
/// .unwrap();
///
/// let view = linearize_default(&grid).unwrap();
/// let iso = contour(&view, &ContourParams::new("z_field", vec![0.5])).unwrap();
/// assert_eq!(iso.num_triangles(), 1);
/// assert!((iso.total_area() - 0.125).abs() < 1e-12);
/// ```
pub fn contour(view: &TetView, params: &ContourParams) -> Result<SurfaceMesh, FilterError> {
    contour_with_chunk_size(view, params, DEFAULT_TET_CHUNK_SIZE)
}

/// Extracts one or more isosurfaces from a linearized [`TetView`] using an explicit parallel
/// `chunk_size` (number of tetrahedra per chunk).
pub fn contour_with_chunk_size(
    view: &TetView,
    params: &ContourParams,
    chunk_size: usize,
) -> Result<SurfaceMesh, FilterError> {
    if params.values.is_empty() {
        return Err(FilterError::EmptyContourValues);
    }
    for &val in &params.values {
        if !val.is_finite() {
            return Err(FilterError::InvalidContourValue { value: val });
        }
    }

    let array = view.point_data().get(&params.array_name).ok_or_else(|| {
        FilterError::MissingPointArray {
            name: params.array_name.clone(),
        }
    })?;

    let scalars = extract_scalar_field(array, &params.array_name, params.component)?;
    extract_isosurfaces_with_chunk_size(view, &scalars, &params.values, chunk_size)
}

/// Convenience helper that linearizes `grid` with `linearize_params` and contours the resulting [`TetView`].
pub fn contour_grid(
    grid: &UnstructuredGrid,
    params: &ContourParams,
    linearize_params: LinearizeParams,
) -> Result<SurfaceMesh, FilterError> {
    let view = linearize(grid, &linearize_params)?;
    contour(&view, params)
}

fn extract_scalar_field(
    array: &DataArray,
    array_name: &str,
    component: ContourComponent,
) -> Result<Vec<f64>, FilterError> {
    let ncomp = array.components().as_usize();
    let num_tuples = array.num_tuples();
    let mut out = Vec::with_capacity(num_tuples);

    match component {
        ContourComponent::Component(comp_idx) => {
            if comp_idx >= ncomp {
                return Err(FilterError::InvalidComponentIndex {
                    array: array_name.to_string(),
                    component: comp_idx,
                    num_components: ncomp,
                });
            }
            for tuple_idx in 0..num_tuples {
                out.push(
                    array
                        .value_as_f64(tuple_idx, comp_idx)
                        .expect("valid tuple and component index"),
                );
            }
        }
        ContourComponent::Magnitude => {
            for tuple_idx in 0..num_tuples {
                let mut sum_sq = 0.0;
                for c in 0..ncomp {
                    let v = array
                        .value_as_f64(tuple_idx, c)
                        .expect("valid tuple and component index");
                    sum_sq += v * v;
                }
                out.push(sum_sq.sqrt());
            }
        }
    }

    Ok(out)
}

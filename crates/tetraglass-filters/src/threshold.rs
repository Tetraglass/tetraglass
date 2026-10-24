//! Volume `threshold` filter on [`TetView`] (`M2.6`, `D-005`).
//!
//! Keeps whole original source cells (`tet_source_cells`) whose point or cell attribute values
//! fall within `[min, max]` (supporting two-sided and one-sided bounds), compacting the kept
//! tetrahedra and vertices while preserving all point and cell [`tetraglass_core::Attributes`]
//! and `tet_source_cells` provenance.

use tetraglass_core::{DataArray, TetView, UnstructuredGrid};

use crate::linearize::{LinearizeParams, linearize};
use crate::marching_tets::{DEFAULT_TET_CHUNK_SIZE, filter_tets_by_source_cell_with_chunk_size};
use crate::surface::FilterError;

/// Attribute table association for [`ThresholdParams`] (`docs/scene-format.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ThresholdAssociation {
    /// Point data wins when the array name exists in both point and cell data; otherwise falls back to cell data.
    #[default]
    Auto,
    /// Force lookup in [`TetView::point_data`].
    Point,
    /// Force lookup in [`TetView::cell_data`].
    Cell,
}

/// Selects which scalar value per tuple is extracted from a [`DataArray`] for thresholding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThresholdComponent {
    /// Zero-based component index (`0` for scalar arrays, `0..3` for 3-vectors, etc.).
    Component(usize),
    /// Euclidean norm $\sqrt{\sum_c v_c^2}$ across all components of each tuple.
    Magnitude,
}

impl Default for ThresholdComponent {
    fn default() -> Self {
        Self::Component(0)
    }
}

/// Parameters for the [`threshold`] filter (`M2.6`, `docs/scene-format.md`).
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdParams {
    /// Name of the point or cell attribute array to threshold.
    pub array_name: String,
    /// Attribute table association (`Auto` default: point data wins when present in both).
    pub association: ThresholdAssociation,
    /// Component or magnitude selector on the attribute array (default: `Component(0)`).
    pub component: ThresholdComponent,
    /// Inclusive lower bound (`None` for one-sided upper threshold `<= max`).
    pub min: Option<f64>,
    /// Inclusive upper bound (`None` for one-sided lower threshold `>= min`).
    pub max: Option<f64>,
}

impl ThresholdParams {
    /// Creates [`ThresholdParams`] for `array_name` with optional inclusive `[min, max]` bounds.
    #[must_use]
    pub fn new(array_name: impl Into<String>, min: Option<f64>, max: Option<f64>) -> Self {
        Self {
            array_name: array_name.into(),
            association: ThresholdAssociation::Auto,
            component: ThresholdComponent::Component(0),
            min,
            max,
        }
    }

    /// Creates [`ThresholdParams`] for a two-sided inclusive range `[min, max]`.
    #[must_use]
    pub fn range(array_name: impl Into<String>, min: f64, max: f64) -> Self {
        Self::new(array_name, Some(min), Some(max))
    }

    /// Creates [`ThresholdParams`] for a one-sided lower bound `value >= min`.
    #[must_use]
    pub fn at_least(array_name: impl Into<String>, min: f64) -> Self {
        Self::new(array_name, Some(min), None)
    }

    /// Creates [`ThresholdParams`] for a one-sided upper bound `value <= max`.
    #[must_use]
    pub fn at_most(array_name: impl Into<String>, max: f64) -> Self {
        Self::new(array_name, None, Some(max))
    }

    /// Sets the [`ThresholdAssociation`] (`Auto`, `Point`, or `Cell`).
    #[must_use]
    pub const fn with_association(mut self, association: ThresholdAssociation) -> Self {
        self.association = association;
        self
    }

    /// Sets the [`ThresholdComponent`] (`Component(idx)` or `Magnitude`).
    #[must_use]
    pub const fn with_component(mut self, component: ThresholdComponent) -> Self {
        self.component = component;
        self
    }

    /// Validates that at least one bound (`min` or `max`) is present, finite, and satisfies `min <= max`.
    pub fn validate_bounds(&self) -> Result<(), FilterError> {
        if self.min.is_none() && self.max.is_none() {
            return Err(FilterError::InvalidThresholdBounds {
                min: None,
                max: None,
            });
        }
        if self.min.is_some_and(|v| !v.is_finite()) || self.max.is_some_and(|v| !v.is_finite()) {
            return Err(FilterError::InvalidThresholdBounds {
                min: self.min,
                max: self.max,
            });
        }
        if let (Some(lo), Some(hi)) = (self.min, self.max)
            && lo > hi
        {
            return Err(FilterError::InvalidThresholdBounds {
                min: self.min,
                max: self.max,
            });
        }
        Ok(())
    }

    #[inline]
    fn contains(&self, value: f64) -> bool {
        self.min.is_none_or(|lo| value >= lo) && self.max.is_none_or(|hi| value <= hi)
    }
}

/// Filters a linearized [`TetView`] by keeping whole original source cells (`tet_source_cells`)
/// whose attribute values fall within `[params.min, params.max]` (`M2.6`, `D-005`).
///
/// - When thresholding by a **cell** array, an original cell is kept if its cell value is in range.
/// - When thresholding by a **point** array, an original cell is kept if all of its vertices in
///   the linearized view have values in range (matching VTK's `AllScalars` cell-threshold rule).
///
/// # Examples
///
/// ```
/// use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
/// use tetraglass_filters::{ThresholdParams, linearize_default, threshold};
///
/// let grid = SyntheticGridSpec::new(2, 2, 2, SyntheticCellLayout::Hexahedra)
///     .generate()
///     .unwrap();
/// let view = linearize_default(&grid).unwrap();
///
/// // Keep cells 0..=3 out of the 8 hexahedra (`cell_id` in [0, 3]); kept volume is 0.5 of 1.0.
/// let kept = threshold(&view, &ThresholdParams::range("cell_id", 0.0, 3.0)).unwrap();
/// assert!((kept.total_volume() - 0.5).abs() < 1e-12);
/// ```
pub fn threshold(view: &TetView, params: &ThresholdParams) -> Result<TetView, FilterError> {
    threshold_with_chunk_size(view, params, DEFAULT_TET_CHUNK_SIZE)
}

/// Filters a linearized [`TetView`] by `[params.min, params.max]` using an explicit parallel
/// `chunk_size` (number of tetrahedra per chunk).
pub fn threshold_with_chunk_size(
    view: &TetView,
    params: &ThresholdParams,
    chunk_size: usize,
) -> Result<TetView, FilterError> {
    params.validate_bounds()?;

    let (is_point_array, array) = match params.association {
        ThresholdAssociation::Point => {
            let arr = view.point_data().get(&params.array_name).ok_or_else(|| {
                FilterError::MissingPointArray {
                    name: params.array_name.clone(),
                }
            })?;
            (true, arr)
        }
        ThresholdAssociation::Cell => {
            let arr = view.cell_data().get(&params.array_name).ok_or_else(|| {
                FilterError::MissingCellArray {
                    name: params.array_name.clone(),
                }
            })?;
            (false, arr)
        }
        ThresholdAssociation::Auto => {
            if let Some(arr) = view.point_data().get(&params.array_name) {
                (true, arr)
            } else if let Some(arr) = view.cell_data().get(&params.array_name) {
                (false, arr)
            } else {
                return Err(FilterError::MissingAttributeArray {
                    name: params.array_name.clone(),
                });
            }
        }
    };

    let scalars = extract_scalar_field(array, &params.array_name, params.component)?;
    let tets = view.tets();
    let tet_source_cells = view.tet_source_cells();

    if tets.is_empty() {
        return filter_tets_by_source_cell_with_chunk_size(view, &[], chunk_size);
    }

    let max_src = tet_source_cells.iter().copied().max().unwrap_or(0) as usize;
    let mut seen_cell = vec![false; max_src + 1];
    let mut keep_source_cell = vec![true; max_src + 1];

    if is_point_array {
        for (tet_idx, &[v0, v1, v2, v3]) in tets.iter().enumerate() {
            let src = tet_source_cells[tet_idx] as usize;
            seen_cell[src] = true;
            if !params.contains(scalars[v0 as usize])
                || !params.contains(scalars[v1 as usize])
                || !params.contains(scalars[v2 as usize])
                || !params.contains(scalars[v3 as usize])
            {
                keep_source_cell[src] = false;
            }
        }
    } else {
        for (tet_idx, &src_u32) in tet_source_cells.iter().enumerate() {
            let src = src_u32 as usize;
            seen_cell[src] = true;
            if !params.contains(scalars[tet_idx]) {
                keep_source_cell[src] = false;
            }
        }
    }

    for (keep, &seen) in keep_source_cell.iter_mut().zip(seen_cell.iter()) {
        *keep &= seen;
    }

    filter_tets_by_source_cell_with_chunk_size(view, &keep_source_cell, chunk_size)
}

/// Convenience helper that linearizes `grid` with `linearize_params`, thresholds the resulting
/// [`TetView`], and converts the kept volume to an [`UnstructuredGrid`].
pub fn threshold_grid(
    grid: &UnstructuredGrid,
    params: &ThresholdParams,
    linearize_params: LinearizeParams,
) -> Result<UnstructuredGrid, FilterError> {
    let view = linearize(grid, &linearize_params)?;
    let filtered = threshold(&view, params)?;
    Ok(filtered.to_unstructured_grid()?)
}

fn extract_scalar_field(
    array: &DataArray,
    array_name: &str,
    component: ThresholdComponent,
) -> Result<Vec<f64>, FilterError> {
    let ncomp = array.components().as_usize();
    let num_tuples = array.num_tuples();
    let mut out = Vec::with_capacity(num_tuples);

    match component {
        ThresholdComponent::Component(comp_idx) => {
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
        ThresholdComponent::Magnitude => {
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

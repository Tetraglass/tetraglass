//! Error types and checked index conversions for `tetraglass-core`.

use crate::cell::CellType;

/// Errors returned when constructing or validating core grid and array types.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CoreError {
    /// A 64-bit or `usize` index exceeded `u32::MAX` (see decision D-013).
    #[error("index {value} exceeds u32::MAX")]
    IndexOverflow {
        /// The out-of-range value.
        value: u64,
    },

    /// A signed index from file data was negative.
    #[error("index {value} cannot be negative")]
    NegativeIndex {
        /// The negative value.
        value: i64,
    },

    /// `cell_types.len()` and `offsets.len()` did not match.
    #[error("cell_types length ({cell_types}) does not match offsets length ({offsets})")]
    CellCountMismatch {
        /// Number of entries in `cell_types`.
        cell_types: usize,
        /// Number of entries in `offsets`.
        offsets: usize,
    },

    /// End-offsets were not strictly monotonic.
    #[error("cell {cell_index} end-offset {current} is smaller than previous end-offset {prev}")]
    NonMonotonicOffsets {
        /// Index of the offending cell.
        cell_index: usize,
        /// End-offset of the preceding cell.
        prev: u32,
        /// End-offset of the current cell.
        current: u32,
    },

    /// The final cell end-offset did not equal `connectivity.len()`.
    #[error(
        "cell {cell_index} end-offset {offset} does not match connectivity length {connectivity_len}"
    )]
    OffsetOutOfBounds {
        /// Index of the offending cell.
        cell_index: usize,
        /// Recorded end-offset.
        offset: u32,
        /// Total length of the flat connectivity array.
        connectivity_len: usize,
    },

    /// A fixed-size or Lagrange cell had the wrong number of nodes in `connectivity`.
    #[error("cell {cell_index} of type {cell_type:?} expected {expected} nodes, got {actual}")]
    InvalidCellNodeCount {
        /// Index of the offending cell.
        cell_index: usize,
        /// Cell type of the offending cell.
        cell_type: CellType,
        /// Expected number of nodes for `cell_type`.
        expected: usize,
        /// Actual number of nodes in the connectivity slice.
        actual: usize,
    },

    /// A polyhedral cell had fewer than 4 unique vertices.
    #[error("polyhedron cell {cell_index} requires at least 4 nodes, got {actual}")]
    PolyhedronTooFewNodes {
        /// Index of the offending cell.
        cell_index: usize,
        /// Actual number of nodes in the cell connectivity slice.
        actual: usize,
    },

    /// A connectivity or face-stream entry referenced a point index `>= num_points`.
    #[error("cell {cell_index} references point {point_index}, but grid has {num_points} points")]
    PointIndexOutOfBounds {
        /// Index of the offending cell.
        cell_index: usize,
        /// Out-of-bounds point index.
        point_index: u32,
        /// Total number of points in the grid.
        num_points: usize,
    },

    /// A grid contained a `CellType::Polyhedron` cell without a `PolyhedronFaces` stream.
    #[error("cell {cell_index} is a polyhedron, but no polyhedron face stream was provided")]
    MissingPolyhedronFaces {
        /// Index of the polyhedral cell.
        cell_index: usize,
    },

    /// `PolyhedronFaces::face_offsets.len()` did not equal `num_cells`.
    #[error(
        "polyhedron face_offsets length ({face_offsets}) does not match cell count ({num_cells})"
    )]
    PolyhedronFaceOffsetsMismatch {
        /// Length of `face_offsets`.
        face_offsets: usize,
        /// Total number of cells in the grid.
        num_cells: usize,
    },

    /// A polyhedron face stream was malformed (truncated, non-monotonic offsets, or < 3 vertices on a face).
    #[error("invalid polyhedron face stream at cell {cell_index}: {reason}")]
    InvalidPolyhedronFaceStream {
        /// Index of the offending cell.
        cell_index: usize,
        /// Static description of what was malformed.
        reason: &'static str,
    },

    /// Unknown or unsupported VTK cell type ID.
    #[error("unsupported VTK cell type id {vtk_type}")]
    UnsupportedVtkCellType {
        /// Raw VTK cell type identifier.
        vtk_type: u8,
    },

    /// A Lagrange cell type had a node count that does not correspond to any polynomial order >= 1.
    #[error("invalid node count {node_count} for VTK Lagrange cell type {vtk_type}")]
    InvalidLagrangeNodeCount {
        /// Raw VTK cell type identifier.
        vtk_type: u8,
        /// Node count encountered for the cell.
        node_count: usize,
    },

    /// An array component count was not one of `{1, 3, 6, 9}`.
    #[error("unsupported component count {count}; expected 1, 3, 6, or 9")]
    InvalidComponentCount {
        /// Requested component count.
        count: usize,
    },

    /// An array buffer length was not a multiple of its component count.
    #[error("array buffer length {len} is not divisible by component count {components}")]
    ArrayLengthMismatch {
        /// Total number of scalar values in the buffer.
        len: usize,
        /// Component count per tuple.
        components: u8,
    },

    /// An attribute array's tuple count did not match the grid's point or cell count.
    #[error("attribute '{name}' has {actual} tuples, expected {expected}")]
    AttributeTupleCountMismatch {
        /// Attribute name.
        name: String,
        /// Expected number of tuples (points or cells).
        expected: usize,
        /// Actual number of tuples in the provided `DataArray`.
        actual: usize,
    },

    /// A [`crate::TimeSeries`] was constructed with zero steps.
    #[error("time series must contain at least one time step")]
    EmptyTimeSeries,

    /// A time step in a [`crate::TimeSeries`] had a non-finite (`NaN` or infinite) time value.
    #[error("time step {step_index} has a non-finite time value")]
    InvalidTimeValue {
        /// Zero-based index of the offending time step.
        step_index: usize,
    },

    /// A requested time step index exceeded the number of steps in the [`crate::TimeSeries`].
    #[error("time step index {step_index} is out of bounds for series of length {num_steps}")]
    TimeStepOutOfBounds {
        /// Requested zero-based step index.
        step_index: usize,
        /// Total number of steps in the series.
        num_steps: usize,
    },
}

/// Converts a `u64` index to `u32`, returning [`CoreError::IndexOverflow`] if it exceeds `u32::MAX`.
///
/// ```
/// use tetraglass_core::{CoreError, try_u32_from_u64};
///
/// assert_eq!(try_u32_from_u64(42), Ok(42));
/// assert_eq!(
///     try_u32_from_u64(u64::from(u32::MAX) + 1),
///     Err(CoreError::IndexOverflow { value: u64::from(u32::MAX) + 1 })
/// );
/// ```
#[inline]
pub fn try_u32_from_u64(value: u64) -> Result<u32, CoreError> {
    u32::try_from(value).map_err(|_| CoreError::IndexOverflow { value })
}

/// Converts a `usize` index to `u32`, returning [`CoreError::IndexOverflow`] if it exceeds `u32::MAX`.
#[inline]
pub fn try_u32_from_usize(value: usize) -> Result<u32, CoreError> {
    u32::try_from(value).map_err(|_| CoreError::IndexOverflow {
        value: value as u64,
    })
}

/// Converts a signed `i64` index (common in VTK `Int64` connectivity arrays) to `u32`.
///
/// Returns [`CoreError::NegativeIndex`] if `value < 0` and [`CoreError::IndexOverflow`] if
/// `value > i64::from(u32::MAX)`.
///
/// ```
/// use tetraglass_core::{CoreError, try_u32_from_i64};
///
/// assert_eq!(try_u32_from_i64(7), Ok(7));
/// assert_eq!(try_u32_from_i64(-1), Err(CoreError::NegativeIndex { value: -1 }));
/// ```
#[inline]
pub fn try_u32_from_i64(value: i64) -> Result<u32, CoreError> {
    if value < 0 {
        return Err(CoreError::NegativeIndex { value });
    }
    try_u32_from_u64(value as u64)
}

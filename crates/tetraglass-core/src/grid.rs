//! `UnstructuredGrid` and `BoundingBox` types.

use std::sync::Arc;

use crate::array::{Attributes, DataArray};
use crate::cell::{CellType, PolyhedronFaceIter, PolyhedronFaces};
use crate::error::CoreError;

/// Axis-aligned 3D bounding box in `f64` dataset coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// Minimum corner `[x_min, y_min, z_min]`.
    pub min: [f64; 3],
    /// Maximum corner `[x_max, y_max, z_max]`.
    pub max: [f64; 3],
}

impl BoundingBox {
    /// Computes the bounding box of `points`, returning `None` if `points` is empty.
    #[must_use]
    pub fn from_points(points: &[[f64; 3]]) -> Option<Self> {
        let first = *points.first()?;
        let mut min = first;
        let mut max = first;
        for &[x, y, z] in &points[1..] {
            if x < min[0] {
                min[0] = x;
            }
            if y < min[1] {
                min[1] = y;
            }
            if z < min[2] {
                min[2] = z;
            }
            if x > max[0] {
                max[0] = x;
            }
            if y > max[1] {
                max[1] = y;
            }
            if z > max[2] {
                max[2] = z;
            }
        }
        Some(Self { min, max })
    }

    /// Returns the center `0.5 * (self.min + self.max)` used for relative `f32` GPU coordinates (D-014).
    #[must_use]
    pub fn center(&self) -> [f64; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Returns the side lengths `[dx, dy, dz]`.
    #[must_use]
    pub fn extents(&self) -> [f64; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Returns the Euclidean length of the bounding-box diagonal.
    #[must_use]
    pub fn diagonal(&self) -> f64 {
        let [dx, dy, dz] = self.extents();
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Core unstructured mesh data structure.
///
/// Mirrors VTK's `UnstructuredGrid` layout with `f64` CPU coordinates (D-014), `u32` connectivity
/// and offsets (D-013), native storage for higher-order Lagrange and polyhedral cells (D-005),
/// and `Arc`-backed point and cell [`Attributes`].
///
/// ```
/// use tetraglass_core::{CellType, ComponentCount, DataArray, UnstructuredGrid};
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
/// .expect("valid single tetrahedron");
///
/// grid.insert_point_data(
///     "linear",
///     DataArray::from_f64(ComponentCount::Scalar, vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
/// )
/// .unwrap();
///
/// assert_eq!(grid.num_points(), 4);
/// assert_eq!(grid.num_cells(), 1);
/// assert_eq!(grid.cell_connectivity(0), Some([0, 1, 2, 3].as_slice()));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct UnstructuredGrid {
    points: Arc<[[f64; 3]]>,
    bounds: Option<BoundingBox>,
    cell_types: Vec<CellType>,
    offsets: Vec<u32>,
    connectivity: Vec<u32>,
    polyhedron_faces: Option<PolyhedronFaces>,
    point_data: Attributes,
    cell_data: Attributes,
}

impl UnstructuredGrid {
    /// Constructs and validates an [`UnstructuredGrid`].
    ///
    /// Validation checks:
    /// - `points.len()` fits in `u32` (`D-013`).
    /// - `cell_types.len() == offsets.len()`.
    /// - `offsets` is monotonically increasing and the final end-offset equals `connectivity.len()`.
    /// - Each cell's node count matches [`CellType::expected_node_count`] (or is `>= 4` for [`CellType::Polyhedron`]).
    /// - Every node index in `connectivity` is `< points.len()`.
    /// - If any cell is [`CellType::Polyhedron`], `polyhedron_faces` is present and valid.
    pub fn new(
        points: impl Into<Arc<[[f64; 3]]>>,
        cell_types: Vec<CellType>,
        offsets: Vec<u32>,
        connectivity: Vec<u32>,
        polyhedron_faces: Option<PolyhedronFaces>,
    ) -> Result<Self, CoreError> {
        let points: Arc<[[f64; 3]]> = points.into();
        let num_points = points.len();
        if u32::try_from(num_points).is_err() {
            return Err(CoreError::IndexOverflow {
                value: num_points as u64,
            });
        }
        if u32::try_from(connectivity.len()).is_err() {
            return Err(CoreError::IndexOverflow {
                value: connectivity.len() as u64,
            });
        }

        if cell_types.len() != offsets.len() {
            return Err(CoreError::CellCountMismatch {
                cell_types: cell_types.len(),
                offsets: offsets.len(),
            });
        }

        let mut prev_offset = 0u32;
        let mut has_polyhedron = false;

        for (cell_index, (&cell_type, &end_offset)) in
            cell_types.iter().zip(offsets.iter()).enumerate()
        {
            if end_offset < prev_offset {
                return Err(CoreError::NonMonotonicOffsets {
                    cell_index,
                    prev: prev_offset,
                    current: end_offset,
                });
            }
            let start = prev_offset as usize;
            let end = end_offset as usize;
            if end > connectivity.len() {
                return Err(CoreError::OffsetOutOfBounds {
                    cell_index,
                    offset: end_offset,
                    connectivity_len: connectivity.len(),
                });
            }
            let actual_nodes = end - start;
            match cell_type.expected_node_count() {
                Some(expected) if expected == 0 || actual_nodes != expected => {
                    return Err(CoreError::InvalidCellNodeCount {
                        cell_index,
                        cell_type,
                        expected,
                        actual: actual_nodes,
                    });
                }
                Some(_) => {}
                None => {
                    has_polyhedron = true;
                    if actual_nodes < 4 {
                        return Err(CoreError::PolyhedronTooFewNodes {
                            cell_index,
                            actual: actual_nodes,
                        });
                    }
                }
            }

            for &pt_idx in &connectivity[start..end] {
                if (pt_idx as usize) >= num_points {
                    return Err(CoreError::PointIndexOutOfBounds {
                        cell_index,
                        point_index: pt_idx,
                        num_points,
                    });
                }
            }

            prev_offset = end_offset;
        }

        if prev_offset as usize != connectivity.len() {
            return Err(CoreError::OffsetOutOfBounds {
                cell_index: cell_types.len().saturating_sub(1),
                offset: prev_offset,
                connectivity_len: connectivity.len(),
            });
        }

        if let Some(ref pf) = polyhedron_faces {
            pf.validate(&cell_types, num_points)?;
        } else if has_polyhedron {
            let first_poly = cell_types
                .iter()
                .position(|&ct| ct == CellType::Polyhedron)
                .unwrap_or(0);
            return Err(CoreError::MissingPolyhedronFaces {
                cell_index: first_poly,
            });
        }

        let bounds = BoundingBox::from_points(&points);
        let num_cells = cell_types.len();

        Ok(Self {
            points,
            bounds,
            cell_types,
            offsets,
            connectivity,
            polyhedron_faces,
            point_data: Attributes::new(num_points),
            cell_data: Attributes::new(num_cells),
        })
    }

    /// Returns the number of points in the grid.
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Returns the number of cells in the grid.
    #[must_use]
    pub fn num_cells(&self) -> usize {
        self.cell_types.len()
    }

    /// Returns the 3D point coordinates slice.
    #[must_use]
    pub fn points(&self) -> &[[f64; 3]] {
        &self.points
    }

    /// Returns a cheap clone of the reference-counted point buffer.
    #[must_use]
    pub fn points_arc(&self) -> Arc<[[f64; 3]]> {
        Arc::clone(&self.points)
    }

    /// Returns the cached [`BoundingBox`] of the grid, or `None` if `num_points() == 0`.
    #[must_use]
    pub const fn bounds(&self) -> Option<BoundingBox> {
        self.bounds
    }

    /// Returns the per-cell [`CellType`] slice.
    #[must_use]
    pub fn cell_types(&self) -> &[CellType] {
        &self.cell_types
    }

    /// Returns the per-cell end-offsets into [`UnstructuredGrid::connectivity`].
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Returns the flat cell connectivity array (`u32` point indices).
    #[must_use]
    pub fn connectivity(&self) -> &[u32] {
        &self.connectivity
    }

    /// Returns the connectivity slice (point indices in VTK node order) for `cell_index`.
    #[must_use]
    pub fn cell_connectivity(&self, cell_index: usize) -> Option<&[u32]> {
        let end = *self.offsets.get(cell_index)? as usize;
        let start = if cell_index == 0 {
            0
        } else {
            self.offsets[cell_index - 1] as usize
        };
        self.connectivity.get(start..end)
    }

    /// Returns the polyhedron face streams, if present.
    #[must_use]
    pub const fn polyhedron_faces(&self) -> Option<&PolyhedronFaces> {
        self.polyhedron_faces.as_ref()
    }

    /// Returns an iterator over the polygon faces (`&[u32]`) of polyhedral cell `cell_index`.
    #[must_use]
    pub fn polyhedron_cell_faces(&self, cell_index: usize) -> Option<PolyhedronFaceIter<'_>> {
        if *self.cell_types.get(cell_index)? != CellType::Polyhedron {
            return None;
        }
        self.polyhedron_faces.as_ref()?.cell_faces(cell_index)
    }

    /// Returns the point attribute table.
    #[must_use]
    pub const fn point_data(&self) -> &Attributes {
        &self.point_data
    }

    /// Returns a mutable reference to the point attribute table.
    #[must_use]
    pub fn point_data_mut(&mut self) -> &mut Attributes {
        &mut self.point_data
    }

    /// Inserts a named point [`DataArray`], checking that `array.num_tuples() == self.num_points()`.
    pub fn insert_point_data(
        &mut self,
        name: impl Into<String>,
        array: DataArray,
    ) -> Result<(), CoreError> {
        self.point_data.insert(name, array)
    }

    /// Returns the cell attribute table.
    #[must_use]
    pub const fn cell_data(&self) -> &Attributes {
        &self.cell_data
    }

    /// Returns a mutable reference to the cell attribute table.
    #[must_use]
    pub fn cell_data_mut(&mut self) -> &mut Attributes {
        &mut self.cell_data
    }

    /// Inserts a named cell [`DataArray`], checking that `array.num_tuples() == self.num_cells()`.
    pub fn insert_cell_data(
        &mut self,
        name: impl Into<String>,
        array: DataArray,
    ) -> Result<(), CoreError> {
        self.cell_data.insert(name, array)
    }
}

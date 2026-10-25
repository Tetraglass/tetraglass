//! Linearized tetrahedral mesh view (`TetView`) consumed by volume filters and cell locators (`D-005`).

use std::sync::Arc;

use crate::array::{Attributes, DataArray};
use crate::cell::CellType;
use crate::error::CoreError;
use crate::grid::{BoundingBox, UnstructuredGrid};

/// Computes the signed volume $\frac{1}{6} ((\mathbf{p}_1 - \mathbf{p}_0) \times (\mathbf{p}_2 - \mathbf{p}_0)) \cdot (\mathbf{p}_3 - \mathbf{p}_0)$
/// of a tetrahedron with vertex coordinates `p0`, `p1`, `p2`, `p3`.
#[must_use]
pub fn signed_tet_volume(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let ux = p1[0] - p0[0];
    let uy = p1[1] - p0[1];
    let uz = p1[2] - p0[2];

    let vx = p2[0] - p0[0];
    let vy = p2[1] - p0[1];
    let vz = p2[2] - p0[2];

    let wx = p3[0] - p0[0];
    let wy = p3[1] - p0[1];
    let wz = p3[2] - p0[2];

    let cx = uy * vz - uz * vy;
    let cy = uz * vx - ux * vz;
    let cz = ux * vy - uy * vx;

    (cx * wx + cy * wy + cz * wz) / 6.0
}

/// Linearized tetrahedral view of an [`UnstructuredGrid`], with a map from every output
/// tetrahedron back to its originating source cell index (`D-005`).
///
/// Volume and surface-generating filters (`slice`, `clip`, `contour`, streamlines) consume
/// `TetView` so each algorithm is implemented once on linear 4-node tetrahedra while
/// picking and cell attributes still refer to the original mesh cells.
///
/// ```
/// use tetraglass_core::{Attributes, TetView};
///
/// let points = vec![
///     [0.0, 0.0, 0.0],
///     [1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0],
/// ];
/// let view = TetView::new(
///     points,
///     vec![[0, 1, 2, 3]],
///     vec![0],
///     Attributes::new(4),
///     Attributes::new(1),
/// )
/// .expect("valid single-tet view");
///
/// assert_eq!(view.num_tets(), 1);
/// assert!((view.total_volume() - 1.0 / 6.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TetView {
    points: Arc<[[f64; 3]]>,
    bounds: Option<BoundingBox>,
    tets: Arc<[[u32; 4]]>,
    tet_source_cells: Arc<[u32]>,
    point_data: Attributes,
    cell_data: Attributes,
}

impl TetView {
    /// Constructs and validates a [`TetView`].
    ///
    /// - `points.len()` and `tets.len() * 4` must fit in `u32` (`D-013`).
    /// - `tets.len()` must equal `tet_source_cells.len()`.
    /// - Every vertex index in `tets` must be `< points.len()`.
    /// - `point_data.expected_tuples()` must equal `points.len()`.
    /// - `cell_data.expected_tuples()` must equal `tets.len()`.
    pub fn new(
        points: impl Into<Arc<[[f64; 3]]>>,
        tets: impl Into<Arc<[[u32; 4]]>>,
        tet_source_cells: impl Into<Arc<[u32]>>,
        point_data: Attributes,
        cell_data: Attributes,
    ) -> Result<Self, CoreError> {
        let points: Arc<[[f64; 3]]> = points.into();
        let tets: Arc<[[u32; 4]]> = tets.into();
        let tet_source_cells: Arc<[u32]> = tet_source_cells.into();
        let num_points = points.len();
        if u32::try_from(num_points).is_err() {
            return Err(CoreError::IndexOverflow {
                value: num_points as u64,
            });
        }
        let conn_len = tets.len().saturating_mul(4);
        if u32::try_from(conn_len).is_err() {
            return Err(CoreError::IndexOverflow {
                value: conn_len as u64,
            });
        }
        if tets.len() != tet_source_cells.len() {
            return Err(CoreError::CellCountMismatch {
                cell_types: tets.len(),
                offsets: tet_source_cells.len(),
            });
        }
        for (tet_idx, tet) in tets.iter().enumerate() {
            for &pt_idx in tet {
                if (pt_idx as usize) >= num_points {
                    return Err(CoreError::PointIndexOutOfBounds {
                        cell_index: tet_idx,
                        point_index: pt_idx,
                        num_points,
                    });
                }
            }
        }
        if point_data.expected_tuples() != num_points {
            return Err(CoreError::AttributeTupleCountMismatch {
                name: "<point_data>".to_string(),
                expected: num_points,
                actual: point_data.expected_tuples(),
            });
        }
        if cell_data.expected_tuples() != tets.len() {
            return Err(CoreError::AttributeTupleCountMismatch {
                name: "<cell_data>".to_string(),
                expected: tets.len(),
                actual: cell_data.expected_tuples(),
            });
        }

        let bounds = BoundingBox::from_points(&points);

        Ok(Self {
            points,
            bounds,
            tets,
            tet_source_cells,
            point_data,
            cell_data,
        })
    }

    /// Returns the number of points in the linearized view.
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Returns the number of tetrahedra in the linearized view.
    #[must_use]
    pub fn num_tets(&self) -> usize {
        self.tets.len()
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

    /// Returns the cached [`BoundingBox`] of the points, or `None` if `num_points() == 0`.
    #[must_use]
    pub const fn bounds(&self) -> Option<BoundingBox> {
        self.bounds
    }

    /// Returns the tetrahedron connectivity slice (`[v0, v1, v2, v3]` point indices).
    #[must_use]
    pub fn tets(&self) -> &[[u32; 4]] {
        &self.tets
    }

    /// Returns a cheap clone of the reference-counted tetrahedron connectivity buffer.
    #[must_use]
    pub fn tets_arc(&self) -> Arc<[[u32; 4]]> {
        Arc::clone(&self.tets)
    }

    /// Returns the originating source cell index in the input [`UnstructuredGrid`] for each tetrahedron.
    #[must_use]
    pub fn tet_source_cells(&self) -> &[u32] {
        &self.tet_source_cells
    }

    /// Returns a cheap clone of the reference-counted originating source cell index buffer.
    #[must_use]
    pub fn tet_source_cells_arc(&self) -> Arc<[u32]> {
        Arc::clone(&self.tet_source_cells)
    }

    /// Returns the point [`Attributes`].
    #[must_use]
    pub const fn point_data(&self) -> &Attributes {
        &self.point_data
    }

    /// Returns the per-tetrahedron cell [`Attributes`].
    #[must_use]
    pub const fn cell_data(&self) -> &Attributes {
        &self.cell_data
    }

    /// Inserts a named point [`DataArray`].
    pub fn insert_point_data(
        &mut self,
        name: impl Into<String>,
        array: DataArray,
    ) -> Result<(), CoreError> {
        self.point_data.insert(name, array)
    }

    /// Inserts a named per-tetrahedron cell [`DataArray`].
    pub fn insert_cell_data(
        &mut self,
        name: impl Into<String>,
        array: DataArray,
    ) -> Result<(), CoreError> {
        self.cell_data.insert(name, array)
    }

    /// Returns the signed volume of tetrahedron `tet_index` in `f64`.
    #[must_use]
    pub fn tet_signed_volume(&self, tet_index: usize) -> Option<f64> {
        let &[i0, i1, i2, i3] = self.tets.get(tet_index)?;
        Some(signed_tet_volume(
            self.points[i0 as usize],
            self.points[i1 as usize],
            self.points[i2 as usize],
            self.points[i3 as usize],
        ))
    }

    /// Returns the non-negative geometric volume of tetrahedron `tet_index` in `f64`.
    #[must_use]
    pub fn tet_volume(&self, tet_index: usize) -> Option<f64> {
        self.tet_signed_volume(tet_index).map(f64::abs)
    }

    /// Computes the total geometric volume of all tetrahedra in `f64`.
    #[must_use]
    pub fn total_volume(&self) -> f64 {
        let mut vol = 0.0;
        for &[i0, i1, i2, i3] in self.tets.iter() {
            vol += signed_tet_volume(
                self.points[i0 as usize],
                self.points[i1 as usize],
                self.points[i2 as usize],
                self.points[i3 as usize],
            )
            .abs();
        }
        vol
    }

    /// Converts this [`TetView`] into an [`UnstructuredGrid`] of [`CellType::Tetra`] cells.
    pub fn to_unstructured_grid(&self) -> Result<UnstructuredGrid, CoreError> {
        let num_tets = self.tets.len();
        let cell_types = vec![CellType::Tetra; num_tets];
        let mut offsets = Vec::with_capacity(num_tets);
        let mut connectivity = Vec::with_capacity(num_tets * 4);
        for (idx, &[v0, v1, v2, v3]) in self.tets.iter().enumerate() {
            connectivity.extend_from_slice(&[v0, v1, v2, v3]);
            offsets.push(((idx + 1) * 4) as u32);
        }
        let mut grid = UnstructuredGrid::new(
            Arc::clone(&self.points),
            cell_types,
            offsets,
            connectivity,
            None,
        )?;
        for (name, arr) in self.point_data.iter() {
            grid.insert_point_data(name, arr.clone())?;
        }
        for (name, arr) in self.cell_data.iter() {
            grid.insert_cell_data(name, arr.clone())?;
        }
        Ok(grid)
    }
}

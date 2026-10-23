//! Triangulated surface mesh (`SurfaceMesh`) produced by surface extraction, slicing, and contouring.

use std::sync::Arc;

use crate::array::{Attributes, DataArray};
use crate::cell::CellType;
use crate::error::CoreError;
use crate::grid::{BoundingBox, UnstructuredGrid};

/// A triangulated 3D surface mesh with point attributes, per-triangle cell attributes,
/// and a map back to source cell indices in the originating [`UnstructuredGrid`].
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceMesh {
    points: Arc<[[f64; 3]]>,
    bounds: Option<BoundingBox>,
    triangles: Vec<[u32; 3]>,
    triangle_source_cells: Vec<u32>,
    num_faces: usize,
    face_source_cells: Vec<u32>,
    point_data: Attributes,
    cell_data: Attributes,
}

impl SurfaceMesh {
    /// Constructs and validates a [`SurfaceMesh`].
    ///
    /// - `triangles.len()` must equal `triangle_source_cells.len()`.
    /// - `num_faces` must equal `face_source_cells.len()`.
    /// - Every vertex index in `triangles` must be `< points.len()`.
    /// - `point_data.expected_tuples()` must equal `points.len()`.
    /// - `cell_data.expected_tuples()` must equal `triangles.len()`.
    pub fn new(
        points: impl Into<Arc<[[f64; 3]]>>,
        triangles: Vec<[u32; 3]>,
        triangle_source_cells: Vec<u32>,
        face_source_cells: Vec<u32>,
        point_data: Attributes,
        cell_data: Attributes,
    ) -> Result<Self, CoreError> {
        let points: Arc<[[f64; 3]]> = points.into();
        let num_points = points.len();
        if u32::try_from(num_points).is_err() {
            return Err(CoreError::IndexOverflow {
                value: num_points as u64,
            });
        }
        if triangles.len() != triangle_source_cells.len() {
            return Err(CoreError::CellCountMismatch {
                cell_types: triangles.len(),
                offsets: triangle_source_cells.len(),
            });
        }
        for (tri_idx, tri) in triangles.iter().enumerate() {
            for &pt_idx in tri {
                if (pt_idx as usize) >= num_points {
                    return Err(CoreError::PointIndexOutOfBounds {
                        cell_index: tri_idx,
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
        if cell_data.expected_tuples() != triangles.len() {
            return Err(CoreError::AttributeTupleCountMismatch {
                name: "<cell_data>".to_string(),
                expected: triangles.len(),
                actual: cell_data.expected_tuples(),
            });
        }

        let bounds = BoundingBox::from_points(&points);
        let num_faces = face_source_cells.len();

        Ok(Self {
            points,
            bounds,
            triangles,
            triangle_source_cells,
            num_faces,
            face_source_cells,
            point_data,
            cell_data,
        })
    }

    /// Returns the number of points in the surface mesh.
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Returns the number of triangles in the surface mesh.
    #[must_use]
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Returns the number of logical polygon faces before fan-triangulation of quads/polygons.
    #[must_use]
    pub const fn num_faces(&self) -> usize {
        self.num_faces
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

    /// Returns the cached [`BoundingBox`] of the surface points.
    #[must_use]
    pub const fn bounds(&self) -> Option<BoundingBox> {
        self.bounds
    }

    /// Returns the triangle connectivity slice (`[v0, v1, v2]` point indices).
    #[must_use]
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    /// Returns the source cell index in the originating [`UnstructuredGrid`] for each triangle.
    #[must_use]
    pub fn triangle_source_cells(&self) -> &[u32] {
        &self.triangle_source_cells
    }

    /// Returns the source cell index in the originating [`UnstructuredGrid`] for each logical polygon face.
    #[must_use]
    pub fn face_source_cells(&self) -> &[u32] {
        &self.face_source_cells
    }

    /// Returns the point [`Attributes`].
    #[must_use]
    pub const fn point_data(&self) -> &Attributes {
        &self.point_data
    }

    /// Returns the per-triangle cell [`Attributes`].
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

    /// Inserts a named per-triangle cell [`DataArray`].
    pub fn insert_cell_data(
        &mut self,
        name: impl Into<String>,
        array: DataArray,
    ) -> Result<(), CoreError> {
        self.cell_data.insert(name, array)
    }

    /// Computes the total geometric surface area of all triangles in `f64`.
    #[must_use]
    pub fn total_area(&self) -> f64 {
        let mut area = 0.0;
        for &[i0, i1, i2] in &self.triangles {
            let p0 = self.points[i0 as usize];
            let p1 = self.points[i1 as usize];
            let p2 = self.points[i2 as usize];
            let ux = p1[0] - p0[0];
            let uy = p1[1] - p0[1];
            let uz = p1[2] - p0[2];
            let vx = p2[0] - p0[0];
            let vy = p2[1] - p0[1];
            let vz = p2[2] - p0[2];
            let cx = uy * vz - uz * vy;
            let cy = uz * vx - ux * vz;
            let cz = ux * vy - uy * vx;
            area += 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();
        }
        area
    }

    /// Converts this [`SurfaceMesh`] into an [`UnstructuredGrid`] of [`CellType::Triangle`] cells
    /// (useful for exporting surface filter outputs to `.vtu`).
    pub fn to_unstructured_grid(&self) -> Result<UnstructuredGrid, CoreError> {
        let num_tris = self.triangles.len();
        let cell_types = vec![CellType::Triangle; num_tris];
        let mut offsets = Vec::with_capacity(num_tris);
        let mut connectivity = Vec::with_capacity(num_tris * 3);
        for (idx, &[v0, v1, v2]) in self.triangles.iter().enumerate() {
            connectivity.extend_from_slice(&[v0, v1, v2]);
            offsets.push(((idx + 1) * 3) as u32);
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

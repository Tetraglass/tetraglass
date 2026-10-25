//! Regular 3D grid container (`RegularGrid`) for resampled volume fields (`M4.4`).
//!
//! [`RegularGrid`] stores a uniform 3D lattice of `dimensions = [nx, ny, nz]` sample points
//! spanning an axis-aligned [`BoundingBox`], an `inside_mask` (`1` for points inside the
//! source unstructured mesh domain, `0` outside), and interpolated [`Attributes`] in
//! `x`-fastest, `y`-middle, `z`-slowest order (`idx = ix + nx * (iy + ny * iz)`).

use std::sync::Arc;

use crate::array::{ArrayBuffer, Attributes, DataArray};
use crate::error::{CoreError, try_u32_from_usize};
use crate::grid::BoundingBox;
use crate::surface::SurfaceMesh;

/// Uniform 3D regular grid produced by resampling an unstructured volume (`M4.4`).
///
/// Sample points are ordered with `x` varying fastest, `y` second, and `z` slowest:
/// `linear_index(ix, iy, iz) = ix + nx * (iy + ny * iz)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RegularGrid {
    dimensions: [usize; 3],
    bounds: BoundingBox,
    inside_mask: Arc<[u8]>,
    point_data: Attributes,
}

impl RegularGrid {
    /// Constructs a validated [`RegularGrid`].
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`CoreError::InvalidRegularGridDimensions`] if any dimension in `dimensions` is `< 2`
    ///   or if `nx * ny * nz` overflows `u32::MAX` (`D-013`).
    /// - [`CoreError::InsideMaskLengthMismatch`] if `inside_mask.len() != nx * ny * nz`.
    /// - [`CoreError::AttributeTupleCountMismatch`] if `point_data.num_tuples() != nx * ny * nz`.
    pub fn new(
        dimensions: [usize; 3],
        bounds: BoundingBox,
        inside_mask: impl Into<Arc<[u8]>>,
        point_data: Attributes,
    ) -> Result<Self, CoreError> {
        let [nx, ny, nz] = dimensions;
        if nx < 2 || ny < 2 || nz < 2 {
            return Err(CoreError::InvalidRegularGridDimensions {
                dimensions,
                reason: "each dimension [nx, ny, nz] must be at least 2",
            });
        }
        let num_points = nx.checked_mul(ny).and_then(|xy| xy.checked_mul(nz)).ok_or(
            CoreError::InvalidRegularGridDimensions {
                dimensions,
                reason: "total grid point count overflows usize",
            },
        )?;
        let _ = try_u32_from_usize(num_points)?;

        let inside_mask: Arc<[u8]> = inside_mask.into();
        if inside_mask.len() != num_points {
            return Err(CoreError::InsideMaskLengthMismatch {
                actual: inside_mask.len(),
                expected: num_points,
            });
        }

        if point_data.expected_tuples() != num_points {
            if let Some((first_name, first_arr)) = point_data.iter().next() {
                return Err(CoreError::AttributeTupleCountMismatch {
                    name: first_name.to_string(),
                    expected: num_points,
                    actual: first_arr.num_tuples(),
                });
            }
            return Err(CoreError::AttributeTupleCountMismatch {
                name: "<point_data>".to_string(),
                expected: num_points,
                actual: point_data.expected_tuples(),
            });
        }

        Ok(Self {
            dimensions,
            bounds,
            inside_mask,
            point_data,
        })
    }

    /// Returns the `[nx, ny, nz]` sample counts along the X, Y, and Z axes.
    #[inline]
    #[must_use]
    pub const fn dimensions(&self) -> [usize; 3] {
        self.dimensions
    }

    /// Returns the total number of grid sample points (`nx * ny * nz`).
    #[inline]
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.inside_mask.len()
    }

    /// Returns the number of grid sample points that lie inside the source mesh domain (`inside_mask[i] != 0`).
    #[must_use]
    pub fn num_inside_points(&self) -> usize {
        self.inside_mask.iter().filter(|&&m| m != 0).count()
    }

    /// Returns the world-space axis-aligned [`BoundingBox`] spanned by this regular grid.
    #[inline]
    #[must_use]
    pub const fn bounds(&self) -> BoundingBox {
        self.bounds
    }

    /// Returns the world-space grid spacing `[dx, dy, dz]` between adjacent sample points.
    #[inline]
    #[must_use]
    pub fn spacing(&self) -> [f64; 3] {
        let [nx, ny, nz] = self.dimensions;
        let ext = self.bounds.extents();
        [
            ext[0] / ((nx - 1) as f64),
            ext[1] / ((ny - 1) as f64),
            ext[2] / ((nz - 1) as f64),
        ]
    }

    /// Converts 3D grid indices `(ix, iy, iz)` to the flat linear index `ix + nx * (iy + ny * iz)`.
    ///
    /// Returns `None` if `ix >= nx`, `iy >= ny`, or `iz >= nz`.
    #[inline]
    #[must_use]
    pub const fn linear_index(&self, ix: usize, iy: usize, iz: usize) -> Option<usize> {
        let [nx, ny, nz] = self.dimensions;
        if ix >= nx || iy >= ny || iz >= nz {
            return None;
        }
        Some(ix + nx * (iy + ny * iz))
    }

    /// Converts a flat linear index `0..num_points()` back into `[ix, iy, iz]`.
    #[inline]
    #[must_use]
    pub const fn grid_coords(&self, linear_idx: usize) -> Option<[usize; 3]> {
        let [nx, ny, nz] = self.dimensions;
        let total = nx * ny * nz;
        if linear_idx >= total {
            return None;
        }
        let ix = linear_idx % nx;
        let rem = linear_idx / nx;
        let iy = rem % ny;
        let iz = rem / ny;
        Some([ix, iy, iz])
    }

    /// Computes the world-space `[x, y, z]` position of grid sample `(ix, iy, iz)` in `f64`.
    #[inline]
    #[must_use]
    pub fn point_position(&self, ix: usize, iy: usize, iz: usize) -> Option<[f64; 3]> {
        let [nx, ny, nz] = self.dimensions;
        if ix >= nx || iy >= ny || iz >= nz {
            return None;
        }
        let tx = (ix as f64) / ((nx - 1) as f64);
        let ty = (iy as f64) / ((ny - 1) as f64);
        let tz = (iz as f64) / ((nz - 1) as f64);
        Some([
            self.bounds.min[0] + tx * (self.bounds.max[0] - self.bounds.min[0]),
            self.bounds.min[1] + ty * (self.bounds.max[1] - self.bounds.min[1]),
            self.bounds.min[2] + tz * (self.bounds.max[2] - self.bounds.min[2]),
        ])
    }

    /// Returns the domain occupancy mask slice (`1` = inside source mesh, `0` = outside).
    #[inline]
    #[must_use]
    pub fn inside_mask(&self) -> &[u8] {
        &self.inside_mask
    }

    /// Returns the shared `Arc<[u8]>` domain occupancy mask.
    #[inline]
    #[must_use]
    pub const fn inside_mask_arc(&self) -> &Arc<[u8]> {
        &self.inside_mask
    }

    /// Returns `true` if `linear_idx` is in bounds and lies inside the source mesh domain.
    #[inline]
    #[must_use]
    pub fn is_inside(&self, linear_idx: usize) -> bool {
        self.inside_mask.get(linear_idx).copied().unwrap_or(0) != 0
    }

    /// Returns the point-associated [`Attributes`] sampled on this grid.
    #[inline]
    #[must_use]
    pub const fn point_data(&self) -> &Attributes {
        &self.point_data
    }

    /// Returns a mutable reference to the point-associated [`Attributes`] on this grid.
    #[inline]
    pub fn point_data_mut(&mut self) -> &mut Attributes {
        &mut self.point_data
    }

    /// Builds an 8-vertex, 12-triangle axis-aligned bounding-box [`SurfaceMesh`] with 12 wireframe
    /// edges and the 8 corner values of every `point_data` array.
    pub fn to_bounding_surface(&self) -> Result<SurfaceMesh, CoreError> {
        let min = self.bounds.min;
        let max = self.bounds.max;
        let points = vec![
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [max[0], max[1], min[2]],
            [min[0], max[1], min[2]],
            [min[0], min[1], max[2]],
            [max[0], min[1], max[2]],
            [max[0], max[1], max[2]],
            [min[0], max[1], max[2]],
        ];
        let triangles = vec![
            // -Z
            [0, 2, 1],
            [0, 3, 2],
            // +Z
            [4, 5, 6],
            [4, 6, 7],
            // -Y
            [0, 1, 5],
            [0, 5, 4],
            // +Y
            [3, 7, 6],
            [3, 6, 2],
            // -X
            [0, 4, 7],
            [0, 7, 3],
            // +X
            [1, 2, 6],
            [1, 6, 5],
        ];
        let wireframe_edges = vec![
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ];

        let [nx, ny, nz] = self.dimensions;
        let corner_indices = [
            self.linear_index(0, 0, 0).unwrap_or(0),
            self.linear_index(nx - 1, 0, 0).unwrap_or(0),
            self.linear_index(nx - 1, ny - 1, 0).unwrap_or(0),
            self.linear_index(0, ny - 1, 0).unwrap_or(0),
            self.linear_index(0, 0, nz - 1).unwrap_or(0),
            self.linear_index(nx - 1, 0, nz - 1).unwrap_or(0),
            self.linear_index(nx - 1, ny - 1, nz - 1).unwrap_or(0),
            self.linear_index(0, ny - 1, nz - 1).unwrap_or(0),
        ];

        let mut corner_point_data = Attributes::new(8);
        for (name, arr) in self.point_data.iter() {
            let comps = arr.components().as_usize();
            let mut vals = Vec::with_capacity(8 * comps);
            for &ci in &corner_indices {
                for c in 0..comps {
                    vals.push(arr.value_as_f64(ci, c).unwrap_or(0.0));
                }
            }
            let gathered = DataArray::new(arr.components(), ArrayBuffer::F64(vals.into()))?;
            corner_point_data.insert(name.to_string(), gathered)?;
        }

        SurfaceMesh::new(
            points,
            triangles,
            vec![0; 12],
            vec![0; 6],
            corner_point_data,
            Attributes::new(12),
        )?
        .with_wireframe_edges(wireframe_edges)
    }
}

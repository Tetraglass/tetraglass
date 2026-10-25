//! 3D polyline collection (`PolylineSet`) produced by streamline integration (`M4.2a`),
//! with conversion to cylindrical tube [`SurfaceMesh`]es and [`UnstructuredGrid`]s.

use std::sync::Arc;

use crate::array::{ArrayBuffer, Attributes, DataArray};
use crate::cell::CellType;
use crate::error::{CoreError, try_u32_from_usize};
use crate::grid::{BoundingBox, UnstructuredGrid};
use crate::surface::SurfaceMesh;

/// Default number of circumferential sides when sweeping a [`PolylineSet`] into a 3D tube
/// [`SurfaceMesh`] via [`PolylineSet::to_tube_surface`].
pub const DEFAULT_TUBE_SIDES: usize = 12;

/// A collection of 3D polylines (such as streamlines) with per-point and per-polyline
/// [`Attributes`] and a map from every point back to its originating cell in the source
/// [`UnstructuredGrid`].
///
/// The points of polyline `i` are stored contiguously in `points()[range]`, where
/// `range = self.polyline_point_range(i)`.
///
/// # Examples
///
/// ```
/// use tetraglass_core::{Attributes, ComponentCount, DataArray, PolylineSet};
///
/// let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
/// let mut point_data = Attributes::new(3);
/// point_data.insert(
///     "arc_length",
///     DataArray::from_f64(ComponentCount::Scalar, vec![0.0, 1.0, 2.0])?,
/// )?;
///
/// let polylines = PolylineSet::new(
///     points,
///     vec![3],
///     vec![0, 0, 1],
///     vec![0],
///     point_data,
///     Attributes::new(1),
/// )?;
/// assert_eq!(polylines.num_polylines(), 1);
/// assert_eq!(polylines.num_segments(), 2);
/// assert!((polylines.total_length() - 2.0).abs() < 1e-12);
///
/// let tube = polylines.to_tube_surface(0.05, 8)?;
/// assert!(tube.num_triangles() > 0);
/// # Ok::<(), tetraglass_core::CoreError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PolylineSet {
    points: Arc<[[f64; 3]]>,
    bounds: Option<BoundingBox>,
    offsets: Vec<u32>,
    point_source_cells: Vec<u32>,
    polyline_seed_ids: Vec<u32>,
    point_data: Attributes,
    cell_data: Attributes,
}

impl PolylineSet {
    /// Constructs and validates a [`PolylineSet`].
    ///
    /// - `points.len()` must fit in `u32` (`D-013`).
    /// - `point_source_cells.len()` must equal `points.len()`.
    /// - `offsets.len()` must equal `polyline_seed_ids.len()`.
    /// - Each polyline in `offsets` must contain at least 2 points, and the final offset must
    ///   equal `points.len()`.
    /// - `point_data.expected_tuples()` must equal `points.len()`.
    /// - `cell_data.expected_tuples()` must equal `offsets.len()`.
    pub fn new(
        points: impl Into<Arc<[[f64; 3]]>>,
        offsets: Vec<u32>,
        point_source_cells: Vec<u32>,
        polyline_seed_ids: Vec<u32>,
        point_data: Attributes,
        cell_data: Attributes,
    ) -> Result<Self, CoreError> {
        let points: Arc<[[f64; 3]]> = points.into();
        let num_points = points.len();
        let _ = try_u32_from_usize(num_points)?;

        if point_source_cells.len() != num_points {
            return Err(CoreError::CellCountMismatch {
                cell_types: num_points,
                offsets: point_source_cells.len(),
            });
        }
        if offsets.len() != polyline_seed_ids.len() {
            return Err(CoreError::CellCountMismatch {
                cell_types: offsets.len(),
                offsets: polyline_seed_ids.len(),
            });
        }

        let mut prev = 0u32;
        for (line_idx, &end) in offsets.iter().enumerate() {
            if end <= prev {
                return Err(CoreError::NonMonotonicOffsets {
                    cell_index: line_idx,
                    prev,
                    current: end,
                });
            }
            let count = (end - prev) as usize;
            if count < 2 {
                return Err(CoreError::InvalidCellNodeCount {
                    cell_index: line_idx,
                    cell_type: CellType::Line,
                    expected: 2,
                    actual: count,
                });
            }
            prev = end;
        }
        if prev as usize != num_points {
            return Err(CoreError::OffsetOutOfBounds {
                cell_index: offsets.len().saturating_sub(1),
                offset: prev,
                connectivity_len: num_points,
            });
        }

        if point_data.expected_tuples() != num_points {
            return Err(CoreError::AttributeTupleCountMismatch {
                name: "<point_data>".to_string(),
                expected: num_points,
                actual: point_data.expected_tuples(),
            });
        }
        if cell_data.expected_tuples() != offsets.len() {
            return Err(CoreError::AttributeTupleCountMismatch {
                name: "<cell_data>".to_string(),
                expected: offsets.len(),
                actual: cell_data.expected_tuples(),
            });
        }

        let bounds = BoundingBox::from_points(&points);

        Ok(Self {
            points,
            bounds,
            offsets,
            point_source_cells,
            polyline_seed_ids,
            point_data,
            cell_data,
        })
    }

    /// Returns the total number of points across all polylines.
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Returns the number of polylines in the set.
    #[must_use]
    pub fn num_polylines(&self) -> usize {
        self.offsets.len()
    }

    /// Returns the total number of 2-point line segments across all polylines.
    #[must_use]
    pub fn num_segments(&self) -> usize {
        self.points.len().saturating_sub(self.offsets.len())
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

    /// Returns the cached [`BoundingBox`] of the polyline points, or `None` if empty.
    #[must_use]
    pub const fn bounds(&self) -> Option<BoundingBox> {
        self.bounds
    }

    /// Returns the cumulative end-offsets in `points()` for each polyline.
    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Returns the originating source cell index in the input [`UnstructuredGrid`] for each point.
    #[must_use]
    pub fn point_source_cells(&self) -> &[u32] {
        &self.point_source_cells
    }

    /// Returns the originating seed index (`0..num_seeds`) for each polyline.
    #[must_use]
    pub fn polyline_seed_ids(&self) -> &[u32] {
        &self.polyline_seed_ids
    }

    /// Returns the per-point [`Attributes`].
    #[must_use]
    pub const fn point_data(&self) -> &Attributes {
        &self.point_data
    }

    /// Returns the per-polyline [`Attributes`].
    #[must_use]
    pub const fn cell_data(&self) -> &Attributes {
        &self.cell_data
    }

    /// Inserts a named per-point [`DataArray`].
    pub fn insert_point_data(
        &mut self,
        name: impl Into<String>,
        array: DataArray,
    ) -> Result<(), CoreError> {
        self.point_data.insert(name, array)
    }

    /// Inserts a named per-polyline [`DataArray`].
    pub fn insert_cell_data(
        &mut self,
        name: impl Into<String>,
        array: DataArray,
    ) -> Result<(), CoreError> {
        self.cell_data.insert(name, array)
    }

    /// Returns the `start..end` index range into `self.points()` for polyline `polyline_index`.
    #[must_use]
    pub fn polyline_point_range(&self, polyline_index: usize) -> Option<std::ops::Range<usize>> {
        let end = *self.offsets.get(polyline_index)? as usize;
        let start = if polyline_index == 0 {
            0
        } else {
            self.offsets[polyline_index - 1] as usize
        };
        Some(start..end)
    }

    /// Returns the 3D vertex slice for polyline `polyline_index`.
    #[must_use]
    pub fn polyline_points(&self, polyline_index: usize) -> Option<&[[f64; 3]]> {
        let range = self.polyline_point_range(polyline_index)?;
        Some(&self.points[range])
    }

    /// Computes the geometric arc length of polyline `polyline_index` in `f64`.
    #[must_use]
    pub fn polyline_length(&self, polyline_index: usize) -> Option<f64> {
        let pts = self.polyline_points(polyline_index)?;
        let mut len = 0.0;
        for pair in pts.windows(2) {
            len += dist3(pair[0], pair[1]);
        }
        Some(len)
    }

    /// Computes the total geometric arc length of all polylines in `f64`.
    #[must_use]
    pub fn total_length(&self) -> f64 {
        let mut total = 0.0;
        for i in 0..self.num_polylines() {
            if let Some(len) = self.polyline_length(i) {
                total += len;
            }
        }
        total
    }

    /// Converts this [`PolylineSet`] into a line-only [`SurfaceMesh`] (zero triangles, with
    /// [`SurfaceMesh::wireframe_edges`] populated from the consecutive segments of each polyline).
    pub fn to_line_surface(&self) -> Result<SurfaceMesh, CoreError> {
        let mut edges = Vec::with_capacity(self.num_segments());
        for i in 0..self.num_polylines() {
            if let Some(range) = self.polyline_point_range(i) {
                for v in range.start..range.end.saturating_sub(1) {
                    edges.push([v as u32, (v + 1) as u32]);
                }
            }
        }
        SurfaceMesh::new(
            Arc::clone(&self.points),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            self.point_data.clone(),
            Attributes::new(0),
        )?
        .with_wireframe_edges(edges)
    }

    /// Sweeps every polyline in this [`PolylineSet`] into a 3D cylindrical tube [`SurfaceMesh`]
    /// of radius `radius` (`> 0.0`) with `sides` circumferential facets (`clamped to >= 3`).
    ///
    /// Uses a smooth rotation-minimizing (parallel-transport) frame along each polyline so tubes
    /// do not twist around inflection points, replicates all `point_data` attributes onto the
    /// ring vertices at each station, maps `point_source_cells` and `cell_data` onto the generated
    /// triangles, and populates [`SurfaceMesh::wireframe_edges`] with one longitudinal edge along
    /// each polyline segment.
    pub fn to_tube_surface(&self, radius: f64, sides: usize) -> Result<SurfaceMesh, CoreError> {
        let sides = sides.max(3);
        let radius = if radius.is_finite() && radius > 0.0 {
            radius
        } else {
            1e-3
        };

        let num_pts = self.points.len();
        let num_segs = self.num_segments();
        if num_pts == 0 || num_segs == 0 {
            return SurfaceMesh::new(
                Vec::<[f64; 3]>::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Attributes::new(0),
                Attributes::new(0),
            );
        }

        let total_tube_verts = num_pts.saturating_mul(sides);
        let _ = try_u32_from_usize(total_tube_verts)?;
        let total_tris = num_segs.saturating_mul(sides * 2);

        let mut tube_points = Vec::with_capacity(total_tube_verts);
        let mut triangles = Vec::with_capacity(total_tris);
        let mut triangle_source_cells = Vec::with_capacity(total_tris);
        let mut face_source_cells = Vec::with_capacity(num_segs.saturating_mul(sides));
        let mut triangle_polyline_indices = Vec::with_capacity(total_tris);
        let mut wireframe_edges = Vec::with_capacity(num_segs);

        // Precompute unit circle offsets `(cos(theta), sin(theta))`
        let mut circle = Vec::with_capacity(sides);
        for s in 0..sides {
            let theta = (s as f64) * std::f64::consts::TAU / (sides as f64);
            circle.push((theta.cos(), theta.sin()));
        }

        for line_idx in 0..self.num_polylines() {
            let Some(range) = self.polyline_point_range(line_idx) else {
                continue;
            };
            let pts = &self.points[range.clone()];
            let n = pts.len();
            if n < 2 {
                continue;
            }

            // Compute vertex tangents and parallel-transport normal/binormal frames:
            let mut tangents = Vec::with_capacity(n);
            for k in 0..n {
                let dir = if k == 0 {
                    sub3(pts[1], pts[0])
                } else if k + 1 == n {
                    sub3(pts[n - 1], pts[n - 2])
                } else {
                    sub3(pts[k + 1], pts[k - 1])
                };
                tangents.push(normalize3_or_default(dir, [1.0, 0.0, 0.0]));
            }

            let mut normal = initial_orthogonal_unit(tangents[0]);
            for k in 0..n {
                let t = tangents[k];
                if k > 0 {
                    // Parallel transport: project previous normal orthogonal to current tangent `t`
                    let dot_tn = dot3(normal, t);
                    let projected = [
                        normal[0] - dot_tn * t[0],
                        normal[1] - dot_tn * t[1],
                        normal[2] - dot_tn * t[2],
                    ];
                    normal = normalize3_or_default(projected, initial_orthogonal_unit(t));
                }
                let binormal = cross3(t, normal);
                let p = pts[k];
                for &(c, s) in &circle {
                    let offset = [
                        radius * (c * normal[0] + s * binormal[0]),
                        radius * (c * normal[1] + s * binormal[1]),
                        radius * (c * normal[2] + s * binormal[2]),
                    ];
                    tube_points.push([p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]]);
                }
            }

            for k in 0..(n - 1) {
                let src_pt_idx = range.start + k;
                let next_pt_idx = src_pt_idx + 1;
                let ring0_base = (src_pt_idx * sides) as u32;
                let ring1_base = (next_pt_idx * sides) as u32;
                let src_cell = self.point_source_cells[src_pt_idx];

                // Track one longitudinal wireframe edge per streamline segment
                wireframe_edges.push([ring0_base, ring1_base]);

                for s in 0..sides {
                    let s_next = (s + 1) % sides;
                    let v00 = ring0_base + s as u32;
                    let v01 = ring0_base + s_next as u32;
                    let v10 = ring1_base + s as u32;
                    let v11 = ring1_base + s_next as u32;

                    triangles.push([v00, v10, v11]);
                    triangles.push([v00, v11, v01]);
                    triangle_source_cells.push(src_cell);
                    triangle_source_cells.push(src_cell);
                    triangle_polyline_indices.push(line_idx);
                    triangle_polyline_indices.push(line_idx);
                    face_source_cells.push(src_cell);
                }
            }
        }

        // Replicate point attributes `sides` times per polyline vertex
        let mut tube_point_data = Attributes::new(total_tube_verts);
        for (name, arr) in self.point_data.iter() {
            let replicated = replicate_point_array(arr, sides)?;
            tube_point_data.insert(name, replicated)?;
        }

        // Map per-polyline cell attributes to per-triangle cell attributes
        let mut tube_cell_data = Attributes::new(triangles.len());
        for (name, arr) in self.cell_data.iter() {
            let mapped = gather_array_by_indices(arr, &triangle_polyline_indices)?;
            tube_cell_data.insert(name, mapped)?;
        }

        SurfaceMesh::new(
            tube_points,
            triangles,
            triangle_source_cells,
            face_source_cells,
            tube_point_data,
            tube_cell_data,
        )?
        .with_wireframe_edges(wireframe_edges)
    }

    /// Converts this [`PolylineSet`] into an [`UnstructuredGrid`] of 2-node [`CellType::Line`]
    /// segments, preserving all point attributes.
    pub fn to_unstructured_grid(&self) -> Result<UnstructuredGrid, CoreError> {
        let num_segs = self.num_segments();
        let cell_types = vec![CellType::Line; num_segs];
        let mut offsets = Vec::with_capacity(num_segs);
        let mut connectivity = Vec::with_capacity(num_segs * 2);
        let mut seg_polyline_indices = Vec::with_capacity(num_segs);

        for line_idx in 0..self.num_polylines() {
            if let Some(range) = self.polyline_point_range(line_idx) {
                for v in range.start..range.end.saturating_sub(1) {
                    connectivity.push(v as u32);
                    connectivity.push((v + 1) as u32);
                    offsets.push(connectivity.len() as u32);
                    seg_polyline_indices.push(line_idx);
                }
            }
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
            let mapped = gather_array_by_indices(arr, &seg_polyline_indices)?;
            grid.insert_cell_data(name, mapped)?;
        }
        Ok(grid)
    }
}

fn replicate_point_array(array: &DataArray, repeat: usize) -> Result<DataArray, CoreError> {
    let ncomp = array.components().as_usize();
    let num_tuples = array.num_tuples();
    let out_len = num_tuples.saturating_mul(repeat).saturating_mul(ncomp);

    let buffer = match array.buffer() {
        ArrayBuffer::F32(src) => {
            let mut out = Vec::with_capacity(out_len);
            for tuple in src.chunks_exact(ncomp) {
                for _ in 0..repeat {
                    out.extend_from_slice(tuple);
                }
            }
            ArrayBuffer::F32(out.into())
        }
        ArrayBuffer::F64(src) => {
            let mut out = Vec::with_capacity(out_len);
            for tuple in src.chunks_exact(ncomp) {
                for _ in 0..repeat {
                    out.extend_from_slice(tuple);
                }
            }
            ArrayBuffer::F64(out.into())
        }
        ArrayBuffer::I32(src) => {
            let mut out = Vec::with_capacity(out_len);
            for tuple in src.chunks_exact(ncomp) {
                for _ in 0..repeat {
                    out.extend_from_slice(tuple);
                }
            }
            ArrayBuffer::I32(out.into())
        }
        ArrayBuffer::I64(src) => {
            let mut out = Vec::with_capacity(out_len);
            for tuple in src.chunks_exact(ncomp) {
                for _ in 0..repeat {
                    out.extend_from_slice(tuple);
                }
            }
            ArrayBuffer::I64(out.into())
        }
        ArrayBuffer::U8(src) => {
            let mut out = Vec::with_capacity(out_len);
            for tuple in src.chunks_exact(ncomp) {
                for _ in 0..repeat {
                    out.extend_from_slice(tuple);
                }
            }
            ArrayBuffer::U8(out.into())
        }
    };

    DataArray::new(array.components(), buffer)
}

fn gather_array_by_indices(array: &DataArray, indices: &[usize]) -> Result<DataArray, CoreError> {
    let ncomp = array.components().as_usize();
    let out_len = indices.len().saturating_mul(ncomp);

    let buffer = match array.buffer() {
        ArrayBuffer::F32(src) => {
            let mut out = Vec::with_capacity(out_len);
            for &idx in indices {
                let start = idx * ncomp;
                out.extend_from_slice(&src[start..start + ncomp]);
            }
            ArrayBuffer::F32(out.into())
        }
        ArrayBuffer::F64(src) => {
            let mut out = Vec::with_capacity(out_len);
            for &idx in indices {
                let start = idx * ncomp;
                out.extend_from_slice(&src[start..start + ncomp]);
            }
            ArrayBuffer::F64(out.into())
        }
        ArrayBuffer::I32(src) => {
            let mut out = Vec::with_capacity(out_len);
            for &idx in indices {
                let start = idx * ncomp;
                out.extend_from_slice(&src[start..start + ncomp]);
            }
            ArrayBuffer::I32(out.into())
        }
        ArrayBuffer::I64(src) => {
            let mut out = Vec::with_capacity(out_len);
            for &idx in indices {
                let start = idx * ncomp;
                out.extend_from_slice(&src[start..start + ncomp]);
            }
            ArrayBuffer::I64(out.into())
        }
        ArrayBuffer::U8(src) => {
            let mut out = Vec::with_capacity(out_len);
            for &idx in indices {
                let start = idx * ncomp;
                out.extend_from_slice(&src[start..start + ncomp]);
            }
            ArrayBuffer::U8(out.into())
        }
    };

    DataArray::new(array.components(), buffer)
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub3(a, b);
    dot3(d, d).sqrt()
}

#[inline]
fn normalize3_or_default(v: [f64; 3], default: [f64; 3]) -> [f64; 3] {
    let len = dot3(v, v).sqrt();
    if len.is_finite() && len > 1e-14 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        default
    }
}

#[inline]
fn initial_orthogonal_unit(tangent: [f64; 3]) -> [f64; 3] {
    let axis = if tangent[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let n = cross3(tangent, axis);
    normalize3_or_default(n, [0.0, 0.0, 1.0])
}

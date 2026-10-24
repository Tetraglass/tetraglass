//! Linearization of [`UnstructuredGrid`] cells into a conformal [`TetView`] (`M2.1a`, `D-005`).
//!
//! - **Linear 3D cells (`Tetra`, `Hexahedron`, `Wedge`, `Pyramid`):** decomposed into `1`, `6`, `3`,
//!   or `2` tetrahedra with zero added vertices using the Dompierre et al. minimum-global-vertex
//!   diagonal rule so every shared quadrilateral face between adjacent cells is split along the
//!   exact same diagonal.
//! - **Polyhedra (`Polyhedron`):** decomposed by inserting a cell centroid point (with equal-weight
//!   averaged point attributes, preserving linear fields exactly), fan-triangulating each polygon
//!   face from its minimum global vertex index, and joining each face triangle to the centroid.
//! - **Higher-order 3D cells (`Quadratic*`, `Lagrange*`):** use corner-node decomposition in `M2.1a`
//!   as a stopgap until reference-space basis subdivision in `M2.1b`.

use tetraglass_core::{
    ArrayBuffer, Attributes, CellType, CoreError, DataArray, TetView, UnstructuredGrid,
    signed_tet_volume, try_u32_from_usize,
};

use crate::surface::{FilterError, map_cell_array};

/// Default reference-space subdivision level for higher-order cells (`2` sub-intervals per edge).
pub const DEFAULT_SUBDIVISIONS: u8 = 2;

/// Parameters controlling [`linearize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LinearizeParams {
    /// Number of sub-intervals per reference edge when subdividing higher-order cells (`>= 1`).
    pub subdivisions: u8,
}

impl Default for LinearizeParams {
    fn default() -> Self {
        Self {
            subdivisions: DEFAULT_SUBDIVISIONS,
        }
    }
}

impl LinearizeParams {
    /// Creates [`LinearizeParams`] with the requested `subdivisions` level (`>= 1`).
    #[must_use]
    pub const fn new(subdivisions: u8) -> Self {
        Self { subdivisions }
    }
}

/// Linearizes an [`UnstructuredGrid`] with [`LinearizeParams::default`].
///
/// ```
/// use tetraglass_core::{CellType, UnstructuredGrid};
/// use tetraglass_filters::linearize_default;
///
/// let points = vec![
///     [0.0, 0.0, 0.0],
///     [1.0, 0.0, 0.0],
///     [1.0, 1.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0],
///     [1.0, 0.0, 1.0],
///     [1.0, 1.0, 1.0],
///     [0.0, 1.0, 1.0],
/// ];
/// let grid = UnstructuredGrid::new(
///     points,
///     vec![CellType::Hexahedron],
///     vec![8],
///     vec![0, 1, 2, 3, 4, 5, 6, 7],
///     None,
/// )
/// .unwrap();
///
/// let tet_view = linearize_default(&grid).unwrap();
/// assert_eq!(tet_view.num_tets(), 6);
/// assert!((tet_view.total_volume() - 1.0).abs() < 1e-12);
/// ```
pub fn linearize_default(grid: &UnstructuredGrid) -> Result<TetView, FilterError> {
    linearize(grid, &LinearizeParams::default())
}

/// Decomposes all 3D cells of `grid` into a conformal [`TetView`] with a source-cell map (`D-005`).
///
/// - When no new vertices are synthesized (e.g. meshes of linear 3D cells), `points` and
///   `point_data` are shared directly with `grid` via `Arc` without copying.
/// - When polyhedral cells add centroid vertices, point attributes are interpolated at each
///   centroid by averaging the cell's vertex values (which reproduces any linear field
///   $f(\mathbf{x}) = \mathbf{a} \cdot \mathbf{x} + b$ to machine precision).
/// - All cell attributes in `grid.cell_data()` are mapped to output tetrahedra through
///   [`TetView::tet_source_cells`].
pub fn linearize(
    grid: &UnstructuredGrid,
    params: &LinearizeParams,
) -> Result<TetView, FilterError> {
    if params.subdivisions == 0 {
        return Err(FilterError::InvalidSubdivisionLevel {
            subdivisions: params.subdivisions,
        });
    }

    let base_points = grid.points();
    let base_num_points = base_points.len();
    let num_cells = grid.num_cells();

    let mut extra_points: Vec<[f64; 3]> = Vec::new();
    // Each entry in `centroid_sources` stores the slice range in `grid.connectivity()`
    // whose vertices are averaged equally to interpolate point attributes at that centroid.
    let mut centroid_conn_ranges: Vec<(usize, usize)> = Vec::new();

    let mut tets: Vec<[u32; 4]> = Vec::with_capacity(num_cells * 2);
    let mut tet_source_cells: Vec<u32> = Vec::with_capacity(num_cells * 2);

    let point_at = |idx: u32, extra: &[[f64; 3]]| -> [f64; 3] {
        let u = idx as usize;
        if u < base_num_points {
            base_points[u]
        } else {
            extra[u - base_num_points]
        }
    };

    let mut emit_tet = |raw: [u32; 4], source_cell: u32, extra: &[[f64; 3]]| {
        let p0 = point_at(raw[0], extra);
        let p1 = point_at(raw[1], extra);
        let p2 = point_at(raw[2], extra);
        let p3 = point_at(raw[3], extra);
        let oriented = if signed_tet_volume(p0, p1, p2, p3) < 0.0 {
            [raw[0], raw[2], raw[1], raw[3]]
        } else {
            raw
        };
        tets.push(oriented);
        tet_source_cells.push(source_cell);
    };

    for cell_idx in 0..num_cells {
        let source_cell = try_u32_from_usize(cell_idx)?;
        let cell_type = grid.cell_types()[cell_idx];
        let conn = grid
            .cell_connectivity(cell_idx)
            .expect("validated grid cell_connectivity");

        match cell_type {
            // 0D, 1D, and 2D cells have zero 3D volume and produce no tetrahedra.
            CellType::Vertex
            | CellType::Line
            | CellType::QuadraticEdge
            | CellType::LagrangeCurve { .. }
            | CellType::Triangle
            | CellType::QuadraticTriangle
            | CellType::LagrangeTriangle { .. }
            | CellType::Quad
            | CellType::QuadraticQuad
            | CellType::LagrangeQuadrilateral { .. } => {}

            // 3D Tetrahedron (and higher-order stopgap until M2.1b).
            CellType::Tetra | CellType::QuadraticTetra | CellType::LagrangeTetrahedron { .. } => {
                emit_tet(
                    [conn[0], conn[1], conn[2], conn[3]],
                    source_cell,
                    &extra_points,
                );
            }

            // 3D Pyramid (and higher-order stopgap until M2.1b): 2 tets split along the
            // base quad diagonal incident to the minimum global vertex index.
            CellType::Pyramid | CellType::QuadraticPyramid => {
                let [v0, v1, v2, v3, v4] = [conn[0], conn[1], conn[2], conn[3], conn[4]];
                if v0.min(v2) < v1.min(v3) {
                    emit_tet([v0, v1, v2, v4], source_cell, &extra_points);
                    emit_tet([v0, v2, v3, v4], source_cell, &extra_points);
                } else {
                    emit_tet([v1, v2, v3, v4], source_cell, &extra_points);
                    emit_tet([v1, v3, v0, v4], source_cell, &extra_points);
                }
            }

            // 3D Wedge / triangular prism (and higher-order stopgap until M2.1b):
            // 3 tets using Dompierre et al. face-consistent subdivision.
            CellType::Wedge | CellType::QuadraticWedge | CellType::LagrangeWedge { .. } => {
                let v = [conn[0], conn[1], conn[2], conn[3], conn[4], conn[5]];
                for tet in decompose_wedge_face_consistent(v) {
                    emit_tet(tet, source_cell, &extra_points);
                }
            }

            // 3D Hexahedron (and higher-order stopgap until M2.1b):
            // 6 tets using Dompierre et al. face-consistent subdivision.
            CellType::Hexahedron
            | CellType::QuadraticHexahedron
            | CellType::LagrangeHexahedron { .. } => {
                let v = [
                    conn[0], conn[1], conn[2], conn[3], conn[4], conn[5], conn[6], conn[7],
                ];
                for tet in decompose_hex_face_consistent(v) {
                    emit_tet(tet, source_cell, &extra_points);
                }
            }

            // 3D Polyhedron: centroid point + minimum-vertex fan triangulation of each face.
            CellType::Polyhedron => {
                let inv_n = 1.0 / (conn.len() as f64);
                let mut cx = 0.0;
                let mut cy = 0.0;
                let mut cz = 0.0;
                for &pt_idx in conn {
                    let p = base_points[pt_idx as usize];
                    cx += p[0];
                    cy += p[1];
                    cz += p[2];
                }
                let centroid_pos = [cx * inv_n, cy * inv_n, cz * inv_n];
                let centroid_idx = try_u32_from_usize(base_num_points + extra_points.len())?;
                extra_points.push(centroid_pos);

                let end = grid.offsets()[cell_idx] as usize;
                let start = if cell_idx == 0 {
                    0
                } else {
                    grid.offsets()[cell_idx - 1] as usize
                };
                centroid_conn_ranges.push((start, end));

                if let Some(face_iter) = grid.polyhedron_cell_faces(cell_idx) {
                    for face_verts in face_iter {
                        if face_verts.len() < 3 {
                            continue;
                        }
                        let min_pos = face_verts
                            .iter()
                            .enumerate()
                            .min_by_key(|&(_, &vid)| vid)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let v_min = face_verts[min_pos];
                        let n = face_verts.len();
                        for offset in 1..(n - 1) {
                            let v_a = face_verts[(min_pos + offset) % n];
                            let v_b = face_verts[(min_pos + offset + 1) % n];
                            emit_tet([v_min, v_a, v_b, centroid_idx], source_cell, &extra_points);
                        }
                    }
                }
            }
        }
    }

    let (points_arc, point_data) = if extra_points.is_empty() {
        (grid.points_arc(), grid.point_data().clone())
    } else {
        let total_points = base_num_points + extra_points.len();
        let mut all_points = Vec::with_capacity(total_points);
        all_points.extend_from_slice(base_points);
        all_points.extend_from_slice(&extra_points);

        let mut pd = Attributes::new(total_points);
        let full_conn = grid.connectivity();
        for (name, arr) in grid.point_data().iter() {
            let extended =
                extend_point_array_with_centroids(arr, full_conn, &centroid_conn_ranges)?;
            pd.insert(name, extended)?;
        }
        (all_points.into(), pd)
    };

    let mut cell_data = Attributes::new(tets.len());
    for (name, arr) in grid.cell_data().iter() {
        let mapped = map_cell_array(arr, &tet_source_cells)?;
        cell_data.insert(name, mapped)?;
    }

    let view = TetView::new(points_arc, tets, tet_source_cells, point_data, cell_data)?;
    Ok(view)
}

/// Decomposes a 6-node linear wedge `v = [v0, v1, v2, v3, v4, v5]` into 3 tetrahedra
/// such that all 3 quadrilateral faces are split along the diagonal incident to that
/// face's minimum global vertex ID.
fn decompose_wedge_face_consistent(v: [u32; 6]) -> [[u32; 4]; 3] {
    let min_local = v
        .iter()
        .enumerate()
        .min_by_key(|&(_, &vid)| vid)
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let [i0, i1, i2, i3, i4, i5] = match min_local {
        0 => [v[0], v[1], v[2], v[3], v[4], v[5]],
        1 => [v[1], v[2], v[0], v[4], v[5], v[3]],
        2 => [v[2], v[0], v[1], v[5], v[3], v[4]],
        3 => [v[3], v[5], v[4], v[0], v[2], v[1]],
        4 => [v[4], v[3], v[5], v[1], v[0], v[2]],
        _ => [v[5], v[4], v[3], v[2], v[1], v[0]],
    };

    if i1.min(i5) < i2.min(i4) {
        [[i0, i1, i2, i5], [i0, i1, i5, i4], [i0, i4, i5, i3]]
    } else {
        [[i0, i1, i2, i4], [i0, i4, i2, i5], [i0, i4, i5, i3]]
    }
}

/// Decomposes an 8-node linear hexahedron `v = [v0..v7]` into 6 tetrahedra such that
/// all 6 quadrilateral faces are split along the diagonal incident to that face's
/// minimum global vertex ID.
fn decompose_hex_face_consistent(v: [u32; 8]) -> [[u32; 4]; 6] {
    let min_local = v
        .iter()
        .enumerate()
        .min_by_key(|&(_, &vid)| vid)
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let [i0, i1, i2, i3, i4, i5, i6, i7] = match min_local {
        0 => [v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]],
        1 => [v[1], v[2], v[3], v[0], v[5], v[6], v[7], v[4]],
        2 => [v[2], v[3], v[0], v[1], v[6], v[7], v[4], v[5]],
        3 => [v[3], v[0], v[1], v[2], v[7], v[4], v[5], v[6]],
        4 => [v[4], v[7], v[6], v[5], v[0], v[3], v[2], v[1]],
        5 => [v[5], v[4], v[7], v[6], v[1], v[0], v[3], v[2]],
        6 => [v[6], v[5], v[4], v[7], v[2], v[1], v[0], v[3]],
        _ => [v[7], v[6], v[5], v[4], v[3], v[2], v[1], v[0]],
    };

    // Opposite face F1 = (i1, i2, i6, i5)
    let (t0, t1) = if i1.min(i6) < i2.min(i5) {
        ([i0, i1, i2, i6], [i0, i1, i6, i5])
    } else {
        ([i0, i1, i2, i5], [i0, i2, i6, i5])
    };

    // Opposite face F2 = (i2, i3, i7, i6)
    let (t2, t3) = if i3.min(i6) < i2.min(i7) {
        ([i0, i2, i3, i6], [i0, i3, i7, i6])
    } else {
        ([i0, i2, i3, i7], [i0, i2, i7, i6])
    };

    // Opposite face F3 = (i4, i5, i6, i7)
    let (t4, t5) = if i4.min(i6) < i5.min(i7) {
        ([i0, i4, i5, i6], [i0, i4, i6, i7])
    } else {
        ([i0, i4, i5, i7], [i0, i5, i6, i7])
    };

    [t0, t1, t2, t3, t4, t5]
}

fn extend_point_array_with_centroids(
    source: &DataArray,
    full_conn: &[u32],
    centroid_conn_ranges: &[(usize, usize)],
) -> Result<DataArray, CoreError> {
    let comp = source.components();
    let ncomp = comp.as_usize();
    let extra_tuples = centroid_conn_ranges.len();

    match source.buffer() {
        ArrayBuffer::F32(buf) => {
            let mut out = Vec::with_capacity(buf.len() + extra_tuples * ncomp);
            out.extend_from_slice(buf);
            for &(start, end) in centroid_conn_ranges {
                let verts = &full_conn[start..end];
                let inv_n = 1.0 / (verts.len() as f64);
                for c in 0..ncomp {
                    let mut sum = 0.0_f64;
                    for &pt in verts {
                        sum += f64::from(buf[(pt as usize) * ncomp + c]);
                    }
                    out.push((sum * inv_n) as f32);
                }
            }
            DataArray::from_f32(comp, out)
        }
        ArrayBuffer::F64(buf) => {
            let mut out = Vec::with_capacity(buf.len() + extra_tuples * ncomp);
            out.extend_from_slice(buf);
            for &(start, end) in centroid_conn_ranges {
                let verts = &full_conn[start..end];
                let inv_n = 1.0 / (verts.len() as f64);
                for c in 0..ncomp {
                    let mut sum = 0.0_f64;
                    for &pt in verts {
                        sum += buf[(pt as usize) * ncomp + c];
                    }
                    out.push(sum * inv_n);
                }
            }
            DataArray::from_f64(comp, out)
        }
        ArrayBuffer::I32(buf) => {
            let mut out = Vec::with_capacity(buf.len() + extra_tuples * ncomp);
            out.extend_from_slice(buf);
            for &(start, end) in centroid_conn_ranges {
                let verts = &full_conn[start..end];
                let inv_n = 1.0 / (verts.len() as f64);
                for c in 0..ncomp {
                    let mut sum = 0.0_f64;
                    for &pt in verts {
                        sum += f64::from(buf[(pt as usize) * ncomp + c]);
                    }
                    out.push((sum * inv_n).round() as i32);
                }
            }
            DataArray::from_i32(comp, out)
        }
        ArrayBuffer::I64(buf) => {
            let mut out = Vec::with_capacity(buf.len() + extra_tuples * ncomp);
            out.extend_from_slice(buf);
            for &(start, end) in centroid_conn_ranges {
                let verts = &full_conn[start..end];
                let inv_n = 1.0 / (verts.len() as f64);
                for c in 0..ncomp {
                    let mut sum = 0.0_f64;
                    for &pt in verts {
                        sum += buf[(pt as usize) * ncomp + c] as f64;
                    }
                    out.push((sum * inv_n).round() as i64);
                }
            }
            DataArray::from_i64(comp, out)
        }
        ArrayBuffer::U8(buf) => {
            let mut out = Vec::with_capacity(buf.len() + extra_tuples * ncomp);
            out.extend_from_slice(buf);
            for &(start, end) in centroid_conn_ranges {
                let verts = &full_conn[start..end];
                let inv_n = 1.0 / (verts.len() as f64);
                for c in 0..ncomp {
                    let mut sum = 0.0_f64;
                    for &pt in verts {
                        sum += f64::from(buf[(pt as usize) * ncomp + c]);
                    }
                    out.push((sum * inv_n).round().clamp(0.0, 255.0) as u8);
                }
            }
            DataArray::from_u8(comp, out)
        }
    }
}

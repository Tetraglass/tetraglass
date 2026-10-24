//! External boundary surface extraction via face hashing (`M0.6`).

use std::collections::HashMap;

use tetraglass_core::{
    ArrayBuffer, Attributes, CellType, CoreError, DataArray, SurfaceMesh, UnstructuredGrid,
    try_u32_from_usize,
};

/// Errors returned by filters in `tetraglass-filters`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FilterError {
    /// Underlying core grid or array error.
    #[error(transparent)]
    Core(#[from] CoreError),

    /// Subdivision level for higher-order cell linearization must be at least 1.
    #[error("linearize subdivision level must be >= 1, got {subdivisions}")]
    InvalidSubdivisionLevel {
        /// Requested subdivision level.
        subdivisions: u8,
    },
}

/// Canonical order-independent face key used to match shared faces between adjacent cells.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum FaceKey {
    Tri([u32; 3]),
    Quad([u32; 4]),
    Poly(Vec<u32>),
}

impl FaceKey {
    #[inline]
    fn from_tri(mut v: [u32; 3]) -> Self {
        v.sort_unstable();
        Self::Tri(v)
    }

    #[inline]
    fn from_quad(mut v: [u32; 4]) -> Self {
        v.sort_unstable();
        Self::Quad(v)
    }

    fn from_slice(verts: &[u32]) -> Self {
        match verts {
            &[a, b, c] => Self::from_tri([a, b, c]),
            &[a, b, c, d] => Self::from_quad([a, b, c, d]),
            other => {
                let mut sorted = other.to_vec();
                sorted.sort_unstable();
                Self::Poly(sorted)
            }
        }
    }
}

#[derive(Debug, Clone)]
enum OrientedFace {
    Tri([u32; 3]),
    Quad([u32; 4]),
    Poly(Vec<u32>),
}

#[derive(Debug, Clone)]
struct CandidateFace {
    source_cell: u32,
    face: OrientedFace,
    owner_count: u32,
}

/// Extracts the external boundary surface of an [`UnstructuredGrid`] by face hashing,
/// keeping faces that have exactly one owner cell.
///
/// - **Linear 3D cells (`Tetra`, `Hexahedron`, `Wedge`, `Pyramid`):** hashes all boundary faces
///   with outward-pointing vertex winding.
/// - **Higher-order 3D cells (`Quadratic*`, `Lagrange*`):** uses the corner nodes (`0..4`, `0..8`,
///   `0..6`, `0..5` in VTK node ordering) as a stopgap until curved surface tessellation in `M2.2`.
/// - **Polyhedra (`Polyhedron`):** hashes each polygon face from the cell's face stream.
/// - **2D surface cells (`Triangle`, `Quad`, and their quadratic/Lagrange counterparts):** emitted
///   directly as single-owner surface faces.
/// - **Attributes:** carries all point arrays via cheap `Arc` clones and maps all cell arrays
///   to surface triangles through `triangle_source_cells`.
///
/// ```
/// use tetraglass_core::{CellType, UnstructuredGrid};
/// use tetraglass_filters::extract_external_surface;
///
/// let points = vec![
///     [0.0, 0.0, 0.0],
///     [1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0],
/// ];
/// let grid = UnstructuredGrid::new(
///     points,
///     vec![CellType::Tetra],
///     vec![4],
///     vec![0, 1, 2, 3],
///     None,
/// )
/// .unwrap();
///
/// let surface = extract_external_surface(&grid).unwrap();
/// assert_eq!(surface.num_faces(), 4);
/// assert_eq!(surface.num_triangles(), 4);
/// ```
pub fn extract_external_surface(grid: &UnstructuredGrid) -> Result<SurfaceMesh, FilterError> {
    let num_cells = grid.num_cells();
    let mut candidates: Vec<CandidateFace> = Vec::with_capacity(num_cells * 4);
    let mut first_seen: HashMap<FaceKey, usize> = HashMap::with_capacity(num_cells * 4);

    let mut register_face = |source_cell: u32, key: FaceKey, face: OrientedFace| {
        if let Some(&idx) = first_seen.get(&key) {
            candidates[idx].owner_count += 1;
        } else {
            let idx = candidates.len();
            candidates.push(CandidateFace {
                source_cell,
                face,
                owner_count: 1,
            });
            first_seen.insert(key, idx);
        }
    };

    for cell_idx in 0..num_cells {
        let source_cell = try_u32_from_usize(cell_idx)?;
        let cell_type = grid.cell_types()[cell_idx];
        let conn = grid
            .cell_connectivity(cell_idx)
            .expect("validated grid cell_connectivity");

        match cell_type {
            // 0D / 1D cells have no 2D surface faces.
            CellType::Vertex
            | CellType::Line
            | CellType::QuadraticEdge
            | CellType::LagrangeCurve { .. } => {}

            // 2D triangle family (corners 0, 1, 2).
            CellType::Triangle
            | CellType::QuadraticTriangle
            | CellType::LagrangeTriangle { .. } => {
                let tri = [conn[0], conn[1], conn[2]];
                register_face(source_cell, FaceKey::from_tri(tri), OrientedFace::Tri(tri));
            }

            // 2D quad family (corners 0, 1, 2, 3).
            CellType::Quad | CellType::QuadraticQuad | CellType::LagrangeQuadrilateral { .. } => {
                let quad = [conn[0], conn[1], conn[2], conn[3]];
                register_face(
                    source_cell,
                    FaceKey::from_quad(quad),
                    OrientedFace::Quad(quad),
                );
            }

            // 3D tetrahedron family (corners 0, 1, 2, 3).
            CellType::Tetra | CellType::QuadraticTetra | CellType::LagrangeTetrahedron { .. } => {
                let [c0, c1, c2, c3] = [conn[0], conn[1], conn[2], conn[3]];
                for tri in [[c0, c1, c3], [c1, c2, c3], [c2, c0, c3], [c0, c2, c1]] {
                    register_face(source_cell, FaceKey::from_tri(tri), OrientedFace::Tri(tri));
                }
            }

            // 3D hexahedron family (corners 0..8).
            CellType::Hexahedron
            | CellType::QuadraticHexahedron
            | CellType::LagrangeHexahedron { .. } => {
                let [c0, c1, c2, c3, c4, c5, c6, c7] = [
                    conn[0], conn[1], conn[2], conn[3], conn[4], conn[5], conn[6], conn[7],
                ];
                for quad in [
                    [c0, c3, c2, c1], // -z
                    [c4, c5, c6, c7], // +z
                    [c0, c1, c5, c4], // -y
                    [c2, c3, c7, c6], // +y
                    [c0, c4, c7, c3], // -x
                    [c1, c2, c6, c5], // +x
                ] {
                    register_face(
                        source_cell,
                        FaceKey::from_quad(quad),
                        OrientedFace::Quad(quad),
                    );
                }
            }

            // 3D wedge / triangular prism family (corners 0..6).
            CellType::Wedge | CellType::QuadraticWedge | CellType::LagrangeWedge { .. } => {
                let [c0, c1, c2, c3, c4, c5] =
                    [conn[0], conn[1], conn[2], conn[3], conn[4], conn[5]];
                for tri in [[c0, c2, c1], [c3, c4, c5]] {
                    register_face(source_cell, FaceKey::from_tri(tri), OrientedFace::Tri(tri));
                }
                for quad in [[c0, c1, c4, c3], [c1, c2, c5, c4], [c2, c0, c3, c5]] {
                    register_face(
                        source_cell,
                        FaceKey::from_quad(quad),
                        OrientedFace::Quad(quad),
                    );
                }
            }

            // 3D pyramid family (corners 0..5).
            CellType::Pyramid | CellType::QuadraticPyramid => {
                let [c0, c1, c2, c3, c4] = [conn[0], conn[1], conn[2], conn[3], conn[4]];
                let base = [c0, c3, c2, c1];
                register_face(
                    source_cell,
                    FaceKey::from_quad(base),
                    OrientedFace::Quad(base),
                );
                for tri in [[c0, c1, c4], [c1, c2, c4], [c2, c3, c4], [c3, c0, c4]] {
                    register_face(source_cell, FaceKey::from_tri(tri), OrientedFace::Tri(tri));
                }
            }

            // 3D general polyhedron: iterate face stream polygons.
            CellType::Polyhedron => {
                if let Some(face_iter) = grid.polyhedron_cell_faces(cell_idx) {
                    for face_verts in face_iter {
                        let key = FaceKey::from_slice(face_verts);
                        let oriented = match face_verts {
                            &[a, b, c] => OrientedFace::Tri([a, b, c]),
                            &[a, b, c, d] => OrientedFace::Quad([a, b, c, d]),
                            other => OrientedFace::Poly(other.to_vec()),
                        };
                        register_face(source_cell, key, oriented);
                    }
                }
            }
        }
    }

    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut triangle_source_cells: Vec<u32> = Vec::new();
    let mut face_source_cells: Vec<u32> = Vec::new();

    for cand in candidates {
        if cand.owner_count != 1 {
            continue;
        }
        face_source_cells.push(cand.source_cell);
        match cand.face {
            OrientedFace::Tri(tri) => {
                triangles.push(tri);
                triangle_source_cells.push(cand.source_cell);
            }
            OrientedFace::Quad([v0, v1, v2, v3]) => {
                triangles.push([v0, v1, v2]);
                triangle_source_cells.push(cand.source_cell);
                triangles.push([v0, v2, v3]);
                triangle_source_cells.push(cand.source_cell);
            }
            OrientedFace::Poly(verts) => {
                if let Some((&v0, rest)) = verts.split_first() {
                    for win in rest.windows(2) {
                        triangles.push([v0, win[0], win[1]]);
                        triangle_source_cells.push(cand.source_cell);
                    }
                }
            }
        }
    }

    let point_data = grid.point_data().clone();
    let mut cell_data = Attributes::new(triangles.len());
    for (name, arr) in grid.cell_data().iter() {
        let mapped = map_cell_array(arr, &triangle_source_cells)?;
        cell_data.insert(name, mapped)?;
    }

    let surface = SurfaceMesh::new(
        grid.points_arc(),
        triangles,
        triangle_source_cells,
        face_source_cells,
        point_data,
        cell_data,
    )?;
    Ok(surface)
}

pub(crate) fn map_cell_array(
    source: &DataArray,
    source_cells: &[u32],
) -> Result<DataArray, CoreError> {
    let comp = source.components();
    let ncomp = comp.as_usize();
    let out_len = source_cells.len() * ncomp;

    match source.buffer() {
        ArrayBuffer::F32(buf) => {
            let mut out = Vec::with_capacity(out_len);
            for &cell_id in source_cells {
                let start = (cell_id as usize) * ncomp;
                out.extend_from_slice(&buf[start..start + ncomp]);
            }
            DataArray::from_f32(comp, out)
        }
        ArrayBuffer::F64(buf) => {
            let mut out = Vec::with_capacity(out_len);
            for &cell_id in source_cells {
                let start = (cell_id as usize) * ncomp;
                out.extend_from_slice(&buf[start..start + ncomp]);
            }
            DataArray::from_f64(comp, out)
        }
        ArrayBuffer::I32(buf) => {
            let mut out = Vec::with_capacity(out_len);
            for &cell_id in source_cells {
                let start = (cell_id as usize) * ncomp;
                out.extend_from_slice(&buf[start..start + ncomp]);
            }
            DataArray::from_i32(comp, out)
        }
        ArrayBuffer::I64(buf) => {
            let mut out = Vec::with_capacity(out_len);
            for &cell_id in source_cells {
                let start = (cell_id as usize) * ncomp;
                out.extend_from_slice(&buf[start..start + ncomp]);
            }
            DataArray::from_i64(comp, out)
        }
        ArrayBuffer::U8(buf) => {
            let mut out = Vec::with_capacity(out_len);
            for &cell_id in source_cells {
                let start = (cell_id as usize) * ncomp;
                out.extend_from_slice(&buf[start..start + ncomp]);
            }
            DataArray::from_u8(comp, out)
        }
    }
}

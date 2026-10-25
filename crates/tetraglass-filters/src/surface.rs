//! External boundary surface extraction via face hashing (`M0.6`) and curved higher-order
//! face tessellation with original-element wireframe edges (`M2.2`).

use std::collections::HashMap;
use std::sync::Arc;

use tetraglass_core::{
    ArrayBuffer, Attributes, CellType, CoreError, DataArray, SurfaceMesh, TetView,
    UnstructuredGrid, try_u32_from_usize,
};

use crate::basis::{corner_reference_rst, evaluate_cell_basis};
use crate::linearize::DEFAULT_SUBDIVISIONS;

/// Errors returned by filters in `tetraglass-filters`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
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

    /// Plane normal vector is zero or non-finite.
    #[error("plane normal must be finite and non-zero, got {normal:?}")]
    InvalidPlaneNormal {
        /// Provided plane normal `[nx, ny, nz]`.
        normal: [f64; 3],
    },

    /// Plane origin contains non-finite coordinates.
    #[error("plane origin must be finite, got {origin:?}")]
    InvalidPlaneOrigin {
        /// Provided plane origin `[ox, oy, oz]`.
        origin: [f64; 3],
    },

    /// Requested point attribute array was not found on the dataset.
    #[error("point array `{name}` was not found on the dataset")]
    MissingPointArray {
        /// Name of the missing point array.
        name: String,
    },

    /// Requested component index is out of bounds for the point array.
    #[error(
        "component index {component} is out of bounds for {num_components}-component array `{array}`"
    )]
    InvalidComponentIndex {
        /// Name of the point array.
        array: String,
        /// Requested zero-based component index.
        component: usize,
        /// Number of components in the array.
        num_components: usize,
    },

    /// Contour filter requires at least one isovalue.
    #[error("contour values list must not be empty")]
    EmptyContourValues,

    /// Contour isovalue is non-finite (`NaN` or `±∞`).
    #[error("contour isovalue must be finite, got {value}")]
    InvalidContourValue {
        /// Non-finite isovalue.
        value: f64,
    },

    /// Requested cell attribute array was not found on the dataset.
    #[error("cell array `{name}` was not found on the dataset")]
    MissingCellArray {
        /// Name of the missing cell array.
        name: String,
    },

    /// Requested point or cell attribute array was not found on the dataset.
    #[error("attribute array `{name}` was not found on the dataset")]
    MissingAttributeArray {
        /// Name of the missing attribute array.
        name: String,
    },

    /// Threshold bounds are invalid (both `min` and `max` omitted, non-finite, or `min > max`).
    #[error("invalid threshold bounds: min = {min:?}, max = {max:?}")]
    InvalidThresholdBounds {
        /// Lower bound (`None` if omitted).
        min: Option<f64>,
        /// Upper bound (`None` if omitted).
        max: Option<f64>,
    },

    /// Requested vector array does not have 3 components (`ComponentCount::Vector3`).
    #[error("vector array `{array}` must have 3 components, got {num_components}")]
    NonVectorArray {
        /// Name of the attribute array.
        array: String,
        /// Actual number of components in the array.
        num_components: usize,
    },

    /// Streamline seed specification is invalid (empty, zero count, non-finite coordinates, or zero plane normal).
    #[error("invalid streamline seeds: {reason}")]
    InvalidStreamlineSeeds {
        /// Human-readable description of why the seed specification is invalid.
        reason: &'static str,
    },

    /// Streamline integration parameters are invalid (non-positive `max_length`, `step_size`, or `max_steps == 0`).
    #[error("invalid streamline parameters: {reason}")]
    InvalidStreamlineParams {
        /// Human-readable description of why the parameters are invalid.
        reason: &'static str,
    },

    /// Glyph filter parameters are invalid (zero `every` stride or non-positive scale factor).
    #[error("invalid glyph parameters: {reason}")]
    InvalidGlyphParams {
        /// Human-readable description of why the parameters are invalid.
        reason: &'static str,
    },
}

#[inline]
const fn sort3(mut a: u32, mut b: u32, mut c: u32) -> (u32, u64) {
    if a > b {
        let t = a;
        a = b;
        b = t;
    }
    if b > c {
        let t = b;
        b = c;
        c = t;
        if a > b {
            let t2 = a;
            a = b;
            b = t2;
        }
    }
    (a, ((b as u64) << 32) | (c as u64))
}

#[inline]
const fn sort4(mut a: u32, mut b: u32, mut c: u32, mut d: u32) -> (u32, [u32; 3]) {
    if a > b {
        let t = a;
        a = b;
        b = t;
    }
    if c > d {
        let t = c;
        c = d;
        d = t;
    }
    if a > c {
        let t = a;
        a = c;
        c = t;
    }
    if b > d {
        let t = b;
        b = d;
        d = t;
    }
    if b > c {
        let t = b;
        b = c;
        c = t;
    }
    (a, [b, c, d])
}

#[derive(Debug, Clone, Copy)]
enum FaceMatchKey {
    Tri(u64),
    Quad([u32; 3]),
    Poly,
}

#[derive(Debug, Clone, Copy)]
enum OrientedFace {
    Tri {
        verts: [u32; 3],
        corner_slots: [u8; 3],
    },
    Quad {
        verts: [u32; 4],
        corner_slots: [u8; 4],
    },
    Poly(u32),
}

#[derive(Debug, Clone, Copy)]
struct CandidateFace {
    match_key: FaceMatchKey,
    next: u32,
    source_cell: u32,
    face: OrientedFace,
    owner_count: u32,
}

struct FaceRegistry {
    candidates: Vec<CandidateFace>,
    tri_heads: Vec<u32>,
    quad_heads: Vec<u32>,
    poly_faces: Vec<Vec<u32>>,
    poly_first_seen: HashMap<Vec<u32>, usize>,
}

impl FaceRegistry {
    fn new(num_points: usize, num_cells: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(num_cells * 3),
            tri_heads: vec![u32::MAX; num_points],
            quad_heads: Vec::new(),
            poly_faces: Vec::new(),
            poly_first_seen: HashMap::new(),
        }
    }

    #[inline]
    fn register_tri(&mut self, source_cell: u32, verts: [u32; 3], face: OrientedFace) {
        let (min_v, rest_packed) = sort3(verts[0], verts[1], verts[2]);
        let head_slot = &mut self.tri_heads[min_v as usize];
        let mut cur = *head_slot;
        while cur != u32::MAX {
            let cand = &mut self.candidates[cur as usize];
            if let FaceMatchKey::Tri(r) = cand.match_key
                && r == rest_packed
            {
                cand.owner_count += 1;
                return;
            }
            cur = cand.next;
        }
        let idx = self.candidates.len() as u32;
        self.candidates.push(CandidateFace {
            match_key: FaceMatchKey::Tri(rest_packed),
            next: *head_slot,
            source_cell,
            face,
            owner_count: 1,
        });
        *head_slot = idx;
    }

    #[inline]
    fn register_quad(
        &mut self,
        num_points: usize,
        source_cell: u32,
        verts: [u32; 4],
        face: OrientedFace,
    ) {
        if self.quad_heads.is_empty() {
            self.quad_heads.resize(num_points, u32::MAX);
        }
        let (min_v, rest) = sort4(verts[0], verts[1], verts[2], verts[3]);
        let head_slot = &mut self.quad_heads[min_v as usize];
        let mut cur = *head_slot;
        while cur != u32::MAX {
            let cand = &mut self.candidates[cur as usize];
            if let FaceMatchKey::Quad(r) = cand.match_key
                && r == rest
            {
                cand.owner_count += 1;
                return;
            }
            cur = cand.next;
        }
        let idx = self.candidates.len() as u32;
        self.candidates.push(CandidateFace {
            match_key: FaceMatchKey::Quad(rest),
            next: *head_slot,
            source_cell,
            face,
            owner_count: 1,
        });
        *head_slot = idx;
    }

    fn register_poly(&mut self, num_points: usize, source_cell: u32, face_verts: &[u32]) {
        let poly_idx = self.poly_faces.len() as u32;
        self.poly_faces.push(face_verts.to_vec());
        let face = OrientedFace::Poly(poly_idx);
        match face_verts {
            &[a, b, c] => self.register_tri(source_cell, [a, b, c], face),
            &[a, b, c, d] => self.register_quad(num_points, source_cell, [a, b, c, d], face),
            other => {
                let mut sorted = other.to_vec();
                sorted.sort_unstable();
                if let Some(&idx) = self.poly_first_seen.get(&sorted) {
                    self.candidates[idx].owner_count += 1;
                } else {
                    let idx = self.candidates.len();
                    self.candidates.push(CandidateFace {
                        match_key: FaceMatchKey::Poly,
                        next: u32::MAX,
                        source_cell,
                        face,
                        owner_count: 1,
                    });
                    self.poly_first_seen.insert(sorted, idx);
                }
            }
        }
    }
}

/// Extracts the external boundary surface of an [`UnstructuredGrid`] using the default
/// higher-order face subdivision level ([`DEFAULT_SUBDIVISIONS`] = `2`).
///
/// See [`extract_external_surface_with_subdivisions`] for details on curved higher-order
/// face tessellation and original-element wireframe edges.
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
/// assert_eq!(surface.num_wireframe_edges(), 6);
/// ```
pub fn extract_external_surface(grid: &UnstructuredGrid) -> Result<SurfaceMesh, FilterError> {
    extract_external_surface_with_subdivisions(grid, DEFAULT_SUBDIVISIONS)
}

/// Extracts the external boundary surface of an [`UnstructuredGrid`] by face hashing,
/// keeping faces that have exactly one owner cell and tessellating higher-order faces
/// at `subdivisions` (`>= 1`).
///
/// - **Linear 3D cells (`Tetra`, `Hexahedron`, `Wedge`, `Pyramid`):** hashes all boundary faces
///   with outward-pointing vertex winding.
/// - **Higher-order 2D/3D cells (`Quadratic*`, `Lagrange*`):** when `subdivisions >= 2`,
///   each external triangle face is tessellated into `subdivisions^2` sub-triangles and each
///   external quad face into `2 * subdivisions^2` sub-triangles using the element's Lagrange
///   or serendipity basis functions.
/// - **Original element wireframe edges (`M2.2`):** [`SurfaceMesh::wireframe_edges`] stores
///   only the outer boundary edges of the original mesh faces (segmented into `subdivisions`
///   segments along curved higher-order edges), excluding quad/polygon fan diagonals and
///   internal higher-order face subdivision edges.
/// - **Polyhedra (`Polyhedron`):** hashes each polygon face from the cell's face stream.
/// - **Attributes:** carries point arrays (zero-copy via `Arc` when no new subdivision points
///   are needed, or interpolated via basis functions) and maps cell arrays to surface triangles
///   through `triangle_source_cells`.
pub fn extract_external_surface_with_subdivisions(
    grid: &UnstructuredGrid,
    subdivisions: u8,
) -> Result<SurfaceMesh, FilterError> {
    if subdivisions == 0 {
        return Err(FilterError::InvalidSubdivisionLevel { subdivisions });
    }

    let num_cells = grid.num_cells();
    let num_points = grid.num_points();
    let mut registry = FaceRegistry::new(num_points, num_cells);

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
                let verts = [conn[0], conn[1], conn[2]];
                registry.register_tri(
                    source_cell,
                    verts,
                    OrientedFace::Tri {
                        verts,
                        corner_slots: [0, 1, 2],
                    },
                );
            }

            // 2D quad family (corners 0, 1, 2, 3).
            CellType::Quad | CellType::QuadraticQuad | CellType::LagrangeQuadrilateral { .. } => {
                let verts = [conn[0], conn[1], conn[2], conn[3]];
                registry.register_quad(
                    num_points,
                    source_cell,
                    verts,
                    OrientedFace::Quad {
                        verts,
                        corner_slots: [0, 1, 2, 3],
                    },
                );
            }

            // 3D tetrahedron family (corners 0, 1, 2, 3).
            CellType::Tetra | CellType::QuadraticTetra | CellType::LagrangeTetrahedron { .. } => {
                for corner_slots in [[0u8, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]] {
                    let verts = [
                        conn[corner_slots[0] as usize],
                        conn[corner_slots[1] as usize],
                        conn[corner_slots[2] as usize],
                    ];
                    registry.register_tri(
                        source_cell,
                        verts,
                        OrientedFace::Tri {
                            verts,
                            corner_slots,
                        },
                    );
                }
            }

            // 3D hexahedron family (corners 0..8).
            CellType::Hexahedron
            | CellType::QuadraticHexahedron
            | CellType::LagrangeHexahedron { .. } => {
                for corner_slots in [
                    [0u8, 3, 2, 1], // -z
                    [4, 5, 6, 7],   // +z
                    [0, 1, 5, 4],   // -y
                    [2, 3, 7, 6],   // +y
                    [0, 4, 7, 3],   // -x
                    [1, 2, 6, 5],   // +x
                ] {
                    let verts = [
                        conn[corner_slots[0] as usize],
                        conn[corner_slots[1] as usize],
                        conn[corner_slots[2] as usize],
                        conn[corner_slots[3] as usize],
                    ];
                    registry.register_quad(
                        num_points,
                        source_cell,
                        verts,
                        OrientedFace::Quad {
                            verts,
                            corner_slots,
                        },
                    );
                }
            }

            // 3D wedge / triangular prism family (corners 0..6).
            CellType::Wedge | CellType::QuadraticWedge | CellType::LagrangeWedge { .. } => {
                for corner_slots in [[0u8, 2, 1], [3, 4, 5]] {
                    let verts = [
                        conn[corner_slots[0] as usize],
                        conn[corner_slots[1] as usize],
                        conn[corner_slots[2] as usize],
                    ];
                    registry.register_tri(
                        source_cell,
                        verts,
                        OrientedFace::Tri {
                            verts,
                            corner_slots,
                        },
                    );
                }
                for corner_slots in [[0u8, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]] {
                    let verts = [
                        conn[corner_slots[0] as usize],
                        conn[corner_slots[1] as usize],
                        conn[corner_slots[2] as usize],
                        conn[corner_slots[3] as usize],
                    ];
                    registry.register_quad(
                        num_points,
                        source_cell,
                        verts,
                        OrientedFace::Quad {
                            verts,
                            corner_slots,
                        },
                    );
                }
            }

            // 3D pyramid family (corners 0..5).
            CellType::Pyramid | CellType::QuadraticPyramid => {
                let base_slots = [0u8, 3, 2, 1];
                let base_verts = [conn[0], conn[3], conn[2], conn[1]];
                registry.register_quad(
                    num_points,
                    source_cell,
                    base_verts,
                    OrientedFace::Quad {
                        verts: base_verts,
                        corner_slots: base_slots,
                    },
                );
                for corner_slots in [[0u8, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]] {
                    let verts = [
                        conn[corner_slots[0] as usize],
                        conn[corner_slots[1] as usize],
                        conn[corner_slots[2] as usize],
                    ];
                    registry.register_tri(
                        source_cell,
                        verts,
                        OrientedFace::Tri {
                            verts,
                            corner_slots,
                        },
                    );
                }
            }

            // 3D general polyhedron: iterate face stream polygons.
            CellType::Polyhedron => {
                if let Some(face_iter) = grid.polyhedron_cell_faces(cell_idx) {
                    for face_verts in face_iter {
                        registry.register_poly(num_points, source_cell, face_verts);
                    }
                }
            }
        }
    }

    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut triangle_source_cells: Vec<u32> = Vec::new();
    let mut face_source_cells: Vec<u32> = Vec::new();
    let mut wireframe_edges: Vec<[u32; 2]> = Vec::new();

    let mut subdiv_ctx = if subdivisions >= 2 {
        Some(SurfaceSubdivContext::new(grid))
    } else {
        None
    };

    for cand in registry.candidates {
        if cand.owner_count != 1 {
            continue;
        }
        face_source_cells.push(cand.source_cell);
        let cell_idx = cand.source_cell as usize;
        let cell_type = grid.cell_types()[cell_idx];
        let is_higher_order = is_higher_order_face_cell(cell_type) && subdivisions >= 2;

        match cand.face {
            OrientedFace::Tri {
                verts: [v0, v1, v2],
                corner_slots,
            } => {
                if is_higher_order {
                    let ctx = subdiv_ctx.as_mut().expect("subdiv_ctx initialized");
                    let conn = grid
                        .cell_connectivity(cell_idx)
                        .expect("validated cell connectivity");
                    ctx.tessellate_tri_face(
                        cand.source_cell,
                        cell_type,
                        conn,
                        corner_slots,
                        subdivisions,
                        &mut triangles,
                        &mut triangle_source_cells,
                        &mut wireframe_edges,
                    )?;
                } else {
                    triangles.push([v0, v1, v2]);
                    triangle_source_cells.push(cand.source_cell);
                    push_canonical_edge(&mut wireframe_edges, v0, v1);
                    push_canonical_edge(&mut wireframe_edges, v1, v2);
                    push_canonical_edge(&mut wireframe_edges, v2, v0);
                }
            }
            OrientedFace::Quad {
                verts: [v0, v1, v2, v3],
                corner_slots,
            } => {
                if is_higher_order {
                    let ctx = subdiv_ctx.as_mut().expect("subdiv_ctx initialized");
                    let conn = grid
                        .cell_connectivity(cell_idx)
                        .expect("validated cell connectivity");
                    ctx.tessellate_quad_face(
                        cand.source_cell,
                        cell_type,
                        conn,
                        corner_slots,
                        subdivisions,
                        &mut triangles,
                        &mut triangle_source_cells,
                        &mut wireframe_edges,
                    )?;
                } else {
                    triangles.push([v0, v1, v2]);
                    triangle_source_cells.push(cand.source_cell);
                    triangles.push([v0, v2, v3]);
                    triangle_source_cells.push(cand.source_cell);
                    push_canonical_edge(&mut wireframe_edges, v0, v1);
                    push_canonical_edge(&mut wireframe_edges, v1, v2);
                    push_canonical_edge(&mut wireframe_edges, v2, v3);
                    push_canonical_edge(&mut wireframe_edges, v3, v0);
                }
            }
            OrientedFace::Poly(poly_idx) => {
                let verts = &registry.poly_faces[poly_idx as usize];
                if let Some((&v0, rest)) = verts.split_first() {
                    for win in rest.windows(2) {
                        triangles.push([v0, win[0], win[1]]);
                        triangle_source_cells.push(cand.source_cell);
                    }
                }
                let n = verts.len();
                for k in 0..n {
                    push_canonical_edge(&mut wireframe_edges, verts[k], verts[(k + 1) % n]);
                }
            }
        }
    }

    wireframe_edges.sort_unstable();
    wireframe_edges.dedup();

    let (points, point_data) = match subdiv_ctx {
        Some(ctx) if ctx.has_new_points() => ctx.finish_points_and_attributes()?,
        _ => (grid.points_arc(), grid.point_data().clone()),
    };

    let mut cell_data = Attributes::new(triangles.len());
    for (name, arr) in grid.cell_data().iter() {
        let mapped = map_cell_array(arr, &triangle_source_cells)?;
        cell_data.insert(name, mapped)?;
    }

    let surface = SurfaceMesh::new(
        points,
        triangles,
        triangle_source_cells,
        face_source_cells,
        point_data,
        cell_data,
    )?
    .with_wireframe_edges(wireframe_edges)?;
    Ok(surface)
}

#[derive(Debug, Clone, Copy)]
struct TetCandidateFace {
    rest_packed: u64,
    next: u32,
    source_cell: u32,
    tet_idx: u32,
    verts: [u32; 3],
    owner_count: u32,
}

#[derive(Debug, Clone, Copy)]
enum TetBoundaryEdgeState {
    Once { src_cell: u32, normal: [f64; 3] },
    TwiceSameCellCoplanar,
    Keep,
}

#[derive(Debug, Clone, Copy)]
struct TetBoundaryEdgeEntry {
    edge: [u32; 2],
    next: u32,
    state: TetBoundaryEdgeState,
}

fn triangle_unit_normal(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> [f64; 3] {
    let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let nx = u[1] * v[2] - u[2] * v[1];
    let ny = u[2] * v[0] - u[0] * v[2];
    let nz = u[0] * v[1] - u[1] * v[0];
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-24 {
        [nx / len, ny / len, nz / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Extracts the external boundary surface of a [`TetView`] (`M2.5`), preserving the
/// originating source cell index (`tet_source_cells`) on every boundary triangle and
/// omitting internal coplanar same-cell decomposition diagonals from [`SurfaceMesh::wireframe_edges`].
pub fn extract_tet_view_surface(view: &TetView) -> Result<SurfaceMesh, FilterError> {
    let tets = view.tets();
    let tet_source_cells = view.tet_source_cells();
    let num_points = view.num_points();
    let mut candidates: Vec<TetCandidateFace> = Vec::with_capacity(tets.len() * 2);
    let mut tri_heads: Vec<u32> = vec![u32::MAX; num_points];

    for (tet_idx_usize, &conn) in tets.iter().enumerate() {
        let tet_idx = try_u32_from_usize(tet_idx_usize)?;
        let source_cell = tet_source_cells[tet_idx_usize];
        for [s0, s1, s2] in [[0usize, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]] {
            let verts = [conn[s0], conn[s1], conn[s2]];
            let (min_v, rest_packed) = sort3(verts[0], verts[1], verts[2]);
            let head_slot = &mut tri_heads[min_v as usize];
            let mut cur = *head_slot;
            let mut found = false;
            while cur != u32::MAX {
                let cand = &mut candidates[cur as usize];
                if cand.rest_packed == rest_packed {
                    cand.owner_count += 1;
                    found = true;
                    break;
                }
                cur = cand.next;
            }
            if !found {
                let idx = candidates.len() as u32;
                candidates.push(TetCandidateFace {
                    rest_packed,
                    next: *head_slot,
                    source_cell,
                    tet_idx,
                    verts,
                    owner_count: 1,
                });
                *head_slot = idx;
            }
        }
    }

    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut triangle_source_cells: Vec<u32> = Vec::new();
    let mut triangle_tet_indices: Vec<u32> = Vec::new();
    let mut face_source_cells: Vec<u32> = Vec::new();
    let mut edge_heads = tri_heads;
    edge_heads.fill(u32::MAX);
    let mut edge_entries: Vec<TetBoundaryEdgeEntry> = Vec::new();
    let points = view.points();

    for cand in candidates {
        if cand.owner_count != 1 {
            continue;
        }
        let [v0, v1, v2] = cand.verts;
        triangles.push([v0, v1, v2]);
        triangle_source_cells.push(cand.source_cell);
        triangle_tet_indices.push(cand.tet_idx);
        face_source_cells.push(cand.source_cell);

        let normal = triangle_unit_normal(
            points[v0 as usize],
            points[v1 as usize],
            points[v2 as usize],
        );
        for [a, b] in [[v0, v1], [v1, v2], [v2, v0]] {
            if a == b {
                continue;
            }
            let edge = if a < b { [a, b] } else { [b, a] };
            let head_slot = &mut edge_heads[edge[0] as usize];
            let mut cur = *head_slot;
            let mut found = false;
            while cur != u32::MAX {
                let entry = &mut edge_entries[cur as usize];
                if entry.edge[1] == edge[1] {
                    entry.state = match entry.state {
                        TetBoundaryEdgeState::Once {
                            src_cell,
                            normal: prev_n,
                        } => {
                            let dot = prev_n[0] * normal[0]
                                + prev_n[1] * normal[1]
                                + prev_n[2] * normal[2];
                            if src_cell == cand.source_cell && dot > 1.0 - 1e-6 {
                                TetBoundaryEdgeState::TwiceSameCellCoplanar
                            } else {
                                TetBoundaryEdgeState::Keep
                            }
                        }
                        TetBoundaryEdgeState::TwiceSameCellCoplanar
                        | TetBoundaryEdgeState::Keep => TetBoundaryEdgeState::Keep,
                    };
                    found = true;
                    break;
                }
                cur = entry.next;
            }
            if !found {
                let idx = edge_entries.len() as u32;
                edge_entries.push(TetBoundaryEdgeEntry {
                    edge,
                    next: *head_slot,
                    state: TetBoundaryEdgeState::Once {
                        src_cell: cand.source_cell,
                        normal,
                    },
                });
                *head_slot = idx;
            }
        }
    }

    let mut wireframe_edges: Vec<[u32; 2]> = edge_entries
        .into_iter()
        .filter_map(|entry| match entry.state {
            TetBoundaryEdgeState::Once { .. } | TetBoundaryEdgeState::Keep => Some(entry.edge),
            TetBoundaryEdgeState::TwiceSameCellCoplanar => None,
        })
        .collect();
    wireframe_edges.sort_unstable();

    let mut cell_data = Attributes::new(triangles.len());
    for (name, arr) in view.cell_data().iter() {
        let mapped = map_cell_array(arr, &triangle_tet_indices)?;
        cell_data.insert(name, mapped)?;
    }

    let surface = SurfaceMesh::new(
        view.points_arc(),
        triangles,
        triangle_source_cells,
        face_source_cells,
        view.point_data().clone(),
        cell_data,
    )?
    .with_wireframe_edges(wireframe_edges)?;
    Ok(surface)
}

#[inline]
pub(crate) fn push_canonical_edge(edges: &mut Vec<[u32; 2]>, a: u32, b: u32) {
    if a != b {
        edges.push(if a < b { [a, b] } else { [b, a] });
    }
}

const fn is_higher_order_face_cell(cell_type: CellType) -> bool {
    match cell_type {
        CellType::QuadraticTriangle
        | CellType::QuadraticQuad
        | CellType::QuadraticTetra
        | CellType::QuadraticHexahedron
        | CellType::QuadraticWedge
        | CellType::QuadraticPyramid => true,
        CellType::LagrangeTriangle { order }
        | CellType::LagrangeQuadrilateral { order }
        | CellType::LagrangeTetrahedron { order }
        | CellType::LagrangeHexahedron { order }
        | CellType::LagrangeWedge { order } => order >= 2,
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PositionKey([i64; 3]);

struct SurfaceSubdivContext<'a> {
    grid: &'a UnstructuredGrid,
    quant_scale: f64,
    new_points: Vec<[f64; 3]>,
    boundary_dedup: HashMap<PositionKey, u32>,
    new_point_attrs: Vec<(String, SurfaceAttrAccumulator<'a>)>,
    weights_scratch: Vec<f64>,
}

enum SurfaceAttrAccumulator<'a> {
    F32 {
        comp: tetraglass_core::ComponentCount,
        src: &'a [f32],
        new_vals: Vec<f32>,
    },
    F64 {
        comp: tetraglass_core::ComponentCount,
        src: &'a [f64],
        new_vals: Vec<f64>,
    },
    I32 {
        comp: tetraglass_core::ComponentCount,
        src: &'a [i32],
        new_vals: Vec<i32>,
    },
    I64 {
        comp: tetraglass_core::ComponentCount,
        src: &'a [i64],
        new_vals: Vec<i64>,
    },
    U8 {
        comp: tetraglass_core::ComponentCount,
        src: &'a [u8],
        new_vals: Vec<u8>,
    },
}

impl<'a> SurfaceAttrAccumulator<'a> {
    fn from_source(source: &'a DataArray) -> Self {
        let comp = source.components();
        match source.buffer() {
            ArrayBuffer::F32(src) => Self::F32 {
                comp,
                src,
                new_vals: Vec::new(),
            },
            ArrayBuffer::F64(src) => Self::F64 {
                comp,
                src,
                new_vals: Vec::new(),
            },
            ArrayBuffer::I32(src) => Self::I32 {
                comp,
                src,
                new_vals: Vec::new(),
            },
            ArrayBuffer::I64(src) => Self::I64 {
                comp,
                src,
                new_vals: Vec::new(),
            },
            ArrayBuffer::U8(src) => Self::U8 {
                comp,
                src,
                new_vals: Vec::new(),
            },
        }
    }

    fn push_interpolated(&mut self, cell_nodes: &[u32], weights: &[f64]) {
        match self {
            Self::F32 {
                comp,
                src,
                new_vals,
            } => {
                let ncomp = comp.as_usize();
                for c in 0..ncomp {
                    let mut sum = 0.0f64;
                    for (&node_id, &w) in cell_nodes.iter().zip(weights.iter()) {
                        sum += w * f64::from(src[(node_id as usize) * ncomp + c]);
                    }
                    new_vals.push(sum as f32);
                }
            }
            Self::F64 {
                comp,
                src,
                new_vals,
            } => {
                let ncomp = comp.as_usize();
                for c in 0..ncomp {
                    let mut sum = 0.0f64;
                    for (&node_id, &w) in cell_nodes.iter().zip(weights.iter()) {
                        sum += w * src[(node_id as usize) * ncomp + c];
                    }
                    new_vals.push(sum);
                }
            }
            Self::I32 {
                comp,
                src,
                new_vals,
            } => {
                let ncomp = comp.as_usize();
                for c in 0..ncomp {
                    let mut sum = 0.0f64;
                    for (&node_id, &w) in cell_nodes.iter().zip(weights.iter()) {
                        sum += w * f64::from(src[(node_id as usize) * ncomp + c]);
                    }
                    new_vals.push(sum.round() as i32);
                }
            }
            Self::I64 {
                comp,
                src,
                new_vals,
            } => {
                let ncomp = comp.as_usize();
                for c in 0..ncomp {
                    let mut sum = 0.0f64;
                    for (&node_id, &w) in cell_nodes.iter().zip(weights.iter()) {
                        sum += w * (src[(node_id as usize) * ncomp + c] as f64);
                    }
                    new_vals.push(sum.round() as i64);
                }
            }
            Self::U8 {
                comp,
                src,
                new_vals,
            } => {
                let ncomp = comp.as_usize();
                for c in 0..ncomp {
                    let mut sum = 0.0f64;
                    for (&node_id, &w) in cell_nodes.iter().zip(weights.iter()) {
                        sum += w * f64::from(src[(node_id as usize) * ncomp + c]);
                    }
                    new_vals.push(sum.round().clamp(0.0, 255.0) as u8);
                }
            }
        }
    }

    fn finish(self) -> Result<DataArray, CoreError> {
        match self {
            Self::F32 {
                comp,
                src,
                new_vals,
            } => {
                let mut out = Vec::with_capacity(src.len() + new_vals.len());
                out.extend_from_slice(src);
                out.extend_from_slice(&new_vals);
                DataArray::from_f32(comp, out)
            }
            Self::F64 {
                comp,
                src,
                new_vals,
            } => {
                let mut out = Vec::with_capacity(src.len() + new_vals.len());
                out.extend_from_slice(src);
                out.extend_from_slice(&new_vals);
                DataArray::from_f64(comp, out)
            }
            Self::I32 {
                comp,
                src,
                new_vals,
            } => {
                let mut out = Vec::with_capacity(src.len() + new_vals.len());
                out.extend_from_slice(src);
                out.extend_from_slice(&new_vals);
                DataArray::from_i32(comp, out)
            }
            Self::I64 {
                comp,
                src,
                new_vals,
            } => {
                let mut out = Vec::with_capacity(src.len() + new_vals.len());
                out.extend_from_slice(src);
                out.extend_from_slice(&new_vals);
                DataArray::from_i64(comp, out)
            }
            Self::U8 {
                comp,
                src,
                new_vals,
            } => {
                let mut out = Vec::with_capacity(src.len() + new_vals.len());
                out.extend_from_slice(src);
                out.extend_from_slice(&new_vals);
                DataArray::from_u8(comp, out)
            }
        }
    }
}

type PointBufferAndAttributes = (Arc<[[f64; 3]]>, Attributes);

impl<'a> SurfaceSubdivContext<'a> {
    fn new(grid: &'a UnstructuredGrid) -> Self {
        let diag = grid.bounds().map_or(1.0, |b| b.diagonal()).max(1e-12);
        let quant_scale = 1e9 / diag;
        let new_point_attrs = grid
            .point_data()
            .iter()
            .map(|(name, arr)| (name.to_string(), SurfaceAttrAccumulator::from_source(arr)))
            .collect();
        Self {
            grid,
            quant_scale,
            new_points: Vec::new(),
            boundary_dedup: HashMap::new(),
            new_point_attrs,
            weights_scratch: Vec::new(),
        }
    }

    fn has_new_points(&self) -> bool {
        !self.new_points.is_empty()
    }

    fn finish_points_and_attributes(self) -> Result<PointBufferAndAttributes, CoreError> {
        let base_pts = self.grid.points();
        let total_pts = base_pts.len() + self.new_points.len();
        let mut points = Vec::with_capacity(total_pts);
        points.extend_from_slice(base_pts);
        points.extend_from_slice(&self.new_points);

        let mut point_data = Attributes::new(total_pts);
        for (name, acc) in self.new_point_attrs {
            point_data.insert(name, acc.finish()?)?;
        }
        Ok((Arc::from(points), point_data))
    }

    fn evaluate_face_point(
        &mut self,
        cell_type: CellType,
        conn: &[u32],
        rst: [f64; 3],
        on_face_perimeter: bool,
    ) -> Result<u32, CoreError> {
        self.weights_scratch.resize(conn.len(), 0.0);
        evaluate_cell_basis(cell_type, rst, &mut self.weights_scratch);

        // Fast path: if this subdivision lattice point coincides with an existing VTK node of the
        // cell (e.g. corner nodes for all S, mid-edge nodes of Quadratic* at S = 2, or Lagrange
        // nodes when S == order), reuse its original mesh vertex index directly.
        if let Some(slot) = self
            .weights_scratch
            .iter()
            .position(|&w| (w - 1.0).abs() < 1e-11)
            && self
                .weights_scratch
                .iter()
                .enumerate()
                .all(|(idx, &w)| idx == slot || w.abs() < 1e-11)
        {
            return Ok(conn[slot]);
        }

        let src_points = self.grid.points();
        let mut pos = [0.0f64; 3];
        for (&node_id, &w) in conn.iter().zip(self.weights_scratch.iter()) {
            let p = src_points[node_id as usize];
            pos[0] += w * p[0];
            pos[1] += w * p[1];
            pos[2] += w * p[2];
        }

        if on_face_perimeter {
            let key = PositionKey([
                (pos[0] * self.quant_scale).round() as i64,
                (pos[1] * self.quant_scale).round() as i64,
                (pos[2] * self.quant_scale).round() as i64,
            ]);
            if let Some(&existing) = self.boundary_dedup.get(&key) {
                return Ok(existing);
            }
            let new_idx = try_u32_from_usize(src_points.len() + self.new_points.len())?;
            self.new_points.push(pos);
            for (_, acc) in &mut self.new_point_attrs {
                acc.push_interpolated(conn, &self.weights_scratch);
            }
            self.boundary_dedup.insert(key, new_idx);
            Ok(new_idx)
        } else {
            let new_idx = try_u32_from_usize(src_points.len() + self.new_points.len())?;
            self.new_points.push(pos);
            for (_, acc) in &mut self.new_point_attrs {
                acc.push_interpolated(conn, &self.weights_scratch);
            }
            Ok(new_idx)
        }
    }

    // Helper Takes 8 explicit arguments for clarity rather than an intermediate struct.
    #[allow(clippy::too_many_arguments)]
    fn tessellate_tri_face(
        &mut self,
        source_cell: u32,
        cell_type: CellType,
        conn: &[u32],
        corner_slots: [u8; 3],
        subdivisions: u8,
        triangles: &mut Vec<[u32; 3]>,
        triangle_source_cells: &mut Vec<u32>,
        wireframe_edges: &mut Vec<[u32; 2]>,
    ) -> Result<(), CoreError> {
        let s = subdivisions as usize;
        let sf = f64::from(subdivisions);
        let rst0 = corner_reference_rst(cell_type, corner_slots[0]);
        let rst1 = corner_reference_rst(cell_type, corner_slots[1]);
        let rst2 = corner_reference_rst(cell_type, corner_slots[2]);

        let mut rows: Vec<Vec<u32>> = Vec::with_capacity(s + 1);
        for j in 0..=s {
            let mut row = Vec::with_capacity(s + 1 - j);
            for i in 0..=(s - j) {
                let u = (i as f64) / sf;
                let v = (j as f64) / sf;
                let w = 1.0 - u - v;
                let rst = if cell_type == CellType::QuadraticPyramid && corner_slots[2] == 4 {
                    let denom = w + u;
                    let r = if denom > 1e-12 {
                        (w * rst0[0] + u * rst1[0]) / denom
                    } else {
                        0.5
                    };
                    let s_coord = if denom > 1e-12 {
                        (w * rst0[1] + u * rst1[1]) / denom
                    } else {
                        0.5
                    };
                    [r, s_coord, v]
                } else {
                    [
                        w * rst0[0] + u * rst1[0] + v * rst2[0],
                        w * rst0[1] + u * rst1[1] + v * rst2[1],
                        w * rst0[2] + u * rst1[2] + v * rst2[2],
                    ]
                };
                let on_perimeter = i == 0 || j == 0 || i + j == s;
                let vid = self.evaluate_face_point(cell_type, conn, rst, on_perimeter)?;
                row.push(vid);
            }
            rows.push(row);
        }

        for j in 0..s {
            for i in 0..(s - j) {
                let v00 = rows[j][i];
                let v10 = rows[j][i + 1];
                let v01 = rows[j + 1][i];
                triangles.push([v00, v10, v01]);
                triangle_source_cells.push(source_cell);

                if i + j + 1 < s {
                    let v11 = rows[j + 1][i + 1];
                    triangles.push([v10, v11, v01]);
                    triangle_source_cells.push(source_cell);
                }
            }
        }

        // Emit only the 3 outer perimeter edges of the original triangle face (each split into S segments).
        for m in 0..s {
            push_canonical_edge(wireframe_edges, rows[0][m], rows[0][m + 1]);
            push_canonical_edge(wireframe_edges, rows[m][s - m], rows[m + 1][s - m - 1]);
            push_canonical_edge(wireframe_edges, rows[s - m][0], rows[s - m - 1][0]);
        }

        Ok(())
    }

    // Helper takes 8 explicit arguments for clarity rather than an intermediate struct.
    #[allow(clippy::too_many_arguments)]
    fn tessellate_quad_face(
        &mut self,
        source_cell: u32,
        cell_type: CellType,
        conn: &[u32],
        corner_slots: [u8; 4],
        subdivisions: u8,
        triangles: &mut Vec<[u32; 3]>,
        triangle_source_cells: &mut Vec<u32>,
        wireframe_edges: &mut Vec<[u32; 2]>,
    ) -> Result<(), CoreError> {
        let s = subdivisions as usize;
        let sf = f64::from(subdivisions);
        let rst0 = corner_reference_rst(cell_type, corner_slots[0]);
        let rst1 = corner_reference_rst(cell_type, corner_slots[1]);
        let rst2 = corner_reference_rst(cell_type, corner_slots[2]);
        let rst3 = corner_reference_rst(cell_type, corner_slots[3]);

        let stride = s + 1;
        let mut grid_vids = Vec::with_capacity(stride * stride);
        for j in 0..=s {
            let v = (j as f64) / sf;
            for i in 0..=s {
                let u = (i as f64) / sf;
                let w00 = (1.0 - u) * (1.0 - v);
                let w10 = u * (1.0 - v);
                let w11 = u * v;
                let w01 = (1.0 - u) * v;
                let rst = [
                    w00 * rst0[0] + w10 * rst1[0] + w11 * rst2[0] + w01 * rst3[0],
                    w00 * rst0[1] + w10 * rst1[1] + w11 * rst2[1] + w01 * rst3[1],
                    w00 * rst0[2] + w10 * rst1[2] + w11 * rst2[2] + w01 * rst3[2],
                ];
                let on_perimeter = i == 0 || i == s || j == 0 || j == s;
                let vid = self.evaluate_face_point(cell_type, conn, rst, on_perimeter)?;
                grid_vids.push(vid);
            }
        }

        let vid_at = |i: usize, j: usize| -> u32 { grid_vids[j * stride + i] };

        for j in 0..s {
            for i in 0..s {
                let v00 = vid_at(i, j);
                let v10 = vid_at(i + 1, j);
                let v11 = vid_at(i + 1, j + 1);
                let v01 = vid_at(i, j + 1);
                triangles.push([v00, v10, v11]);
                triangle_source_cells.push(source_cell);
                triangles.push([v00, v11, v01]);
                triangle_source_cells.push(source_cell);
            }
        }

        // Emit only the 4 outer perimeter edges of the original quad face (each split into S segments).
        for m in 0..s {
            push_canonical_edge(wireframe_edges, vid_at(m, 0), vid_at(m + 1, 0));
            push_canonical_edge(wireframe_edges, vid_at(s, m), vid_at(s, m + 1));
            push_canonical_edge(wireframe_edges, vid_at(s - m, s), vid_at(s - m - 1, s));
            push_canonical_edge(wireframe_edges, vid_at(0, s - m), vid_at(0, s - m - 1));
        }

        Ok(())
    }
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

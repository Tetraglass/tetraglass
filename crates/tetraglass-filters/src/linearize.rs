//! Linearization of [`UnstructuredGrid`] cells into a conformal [`TetView`] (`M2.1a`, `M2.1b`, `D-005`).
//!
//! - **Linear 3D cells (`Tetra`, `Hexahedron`, `Wedge`, `Pyramid`):** decomposed into `1`, `6`, `3`,
//!   or `2` tetrahedra with zero added vertices using the Dompierre et al. minimum-global-vertex
//!   diagonal rule so every shared quadrilateral face between adjacent cells is split along the
//!   exact same diagonal.
//! - **Higher-order 3D cells (`Quadratic*`, `Lagrange*`):** subdivided in reference space into
//!   `subdivisions` (`S >= 1`) intervals per edge; 3D positions and point attributes are evaluated
//!   from the cell's Lagrange or serendipity basis functions, existing cell nodes are reused when
//!   a sample point coincides with a node, and shared boundary sample points are deduplicated so
//!   shared faces remain watertight across adjacent cells.
//! - **Polyhedra (`Polyhedron`):** decomposed by inserting a cell centroid point (with equal-weight
//!   averaged point attributes, preserving linear fields exactly), fan-triangulating each polygon
//!   face from its minimum global vertex index, and joining each face triangle to the centroid.

use std::collections::HashMap;

use tetraglass_core::{
    ArrayBuffer, Attributes, CellType, CoreError, DataArray, TetView, UnstructuredGrid,
    signed_tet_volume, try_u32_from_usize,
};

use crate::basis::evaluate_cell_basis;
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

/// Interpolation record for a synthesized vertex (either a polyhedron centroid or a
/// higher-order basis sample point).
#[derive(Debug, Clone, Copy)]
struct SynthPointInterp {
    /// Start index in `grid.connectivity()`.
    conn_start: usize,
    /// End index in `grid.connectivity()`.
    conn_end: usize,
    /// Start index in `interp_weights` (`None` means equal weights `1 / (conn_end - conn_start)`).
    weight_start: Option<usize>,
}

/// Decomposes all 3D cells of `grid` into a conformal [`TetView`] with a source-cell map (`D-005`).
///
/// - When no new vertices are synthesized (e.g. meshes of linear 3D cells, or higher-order cells
///   whose subdivision lattice points all coincide with existing mesh nodes), `points` and
///   `point_data` are shared directly with `grid` via `Arc` without copying.
/// - When higher-order or polyhedral cells synthesize new vertices, positions and point attributes
///   are evaluated via the element's basis weights (reproducing any linear field
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
    let subdiv = params.subdivisions;

    let inv_quant_step = grid
        .bounds()
        .map(|b| 1e10 / b.diagonal().max(1e-12))
        .unwrap_or(1e10);
    let mut subdiv_ctx = SubdivContext {
        base_points,
        inv_quant_step,
        weights_buf: Vec::new(),
        extra_points: Vec::new(),
        synth_interps: Vec::new(),
        interp_weights: Vec::new(),
        synth_point_map: HashMap::new(),
    };

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

    let emit_tet = |raw: [u32; 4],
                    source_cell: u32,
                    extra: &[[f64; 3]],
                    tets_out: &mut Vec<[u32; 4]>,
                    src_out: &mut Vec<u32>| {
        let p0 = point_at(raw[0], extra);
        let p1 = point_at(raw[1], extra);
        let p2 = point_at(raw[2], extra);
        let p3 = point_at(raw[3], extra);
        let oriented = if signed_tet_volume(p0, p1, p2, p3) < 0.0 {
            [raw[0], raw[2], raw[1], raw[3]]
        } else {
            raw
        };
        tets_out.push(oriented);
        src_out.push(source_cell);
    };

    for cell_idx in 0..num_cells {
        let source_cell = try_u32_from_usize(cell_idx)?;
        let cell_type = grid.cell_types()[cell_idx];
        let conn_end = grid.offsets()[cell_idx] as usize;
        let conn_start = if cell_idx == 0 {
            0
        } else {
            grid.offsets()[cell_idx - 1] as usize
        };
        let conn = &grid.connectivity()[conn_start..conn_end];

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

            // Linear 3D Tetrahedron: 1 tet.
            CellType::Tetra => {
                emit_tet(
                    [conn[0], conn[1], conn[2], conn[3]],
                    source_cell,
                    &subdiv_ctx.extra_points,
                    &mut tets,
                    &mut tet_source_cells,
                );
            }

            // Linear 3D Pyramid: 2 tets split along the minimum-vertex base diagonal.
            CellType::Pyramid => {
                let [v0, v1, v2, v3, v4] = [conn[0], conn[1], conn[2], conn[3], conn[4]];
                for tet in decompose_pyramid_face_consistent([v0, v1, v2, v3, v4]) {
                    emit_tet(
                        tet,
                        source_cell,
                        &subdiv_ctx.extra_points,
                        &mut tets,
                        &mut tet_source_cells,
                    );
                }
            }

            // Linear 3D Wedge: 3 tets using Dompierre et al. face-consistent subdivision.
            CellType::Wedge => {
                let v = [conn[0], conn[1], conn[2], conn[3], conn[4], conn[5]];
                for tet in decompose_wedge_face_consistent(v) {
                    emit_tet(
                        tet,
                        source_cell,
                        &subdiv_ctx.extra_points,
                        &mut tets,
                        &mut tet_source_cells,
                    );
                }
            }

            // Linear 3D Hexahedron: 6 tets using Dompierre et al. face-consistent subdivision.
            CellType::Hexahedron => {
                let v = [
                    conn[0], conn[1], conn[2], conn[3], conn[4], conn[5], conn[6], conn[7],
                ];
                for tet in decompose_hex_face_consistent(v) {
                    emit_tet(
                        tet,
                        source_cell,
                        &subdiv_ctx.extra_points,
                        &mut tets,
                        &mut tet_source_cells,
                    );
                }
            }

            // Higher-order 3D Tetrahedron (`QuadraticTetra`, `LagrangeTetrahedron`).
            CellType::QuadraticTetra | CellType::LagrangeTetrahedron { .. } => {
                if subdiv == 1 {
                    emit_tet(
                        [conn[0], conn[1], conn[2], conn[3]],
                        source_cell,
                        &subdiv_ctx.extra_points,
                        &mut tets,
                        &mut tet_source_cells,
                    );
                } else {
                    let s = usize::from(subdiv);
                    let sf = f64::from(subdiv);
                    let stride = s + 1;
                    let mut grid_vid = vec![0u32; stride * stride * stride];
                    let idx3 = |i: usize, j: usize, k: usize| (k * stride + j) * stride + i;

                    for k in 0..=s {
                        for j in 0..=(s - k) {
                            for i in 0..=(s - k - j) {
                                let rst = [i as f64 / sf, j as f64 / sf, k as f64 / sf];
                                let vid = subdiv_ctx.resolve_sample_vertex(
                                    cell_type, rst, conn, conn_start, conn_end,
                                )?;
                                grid_vid[idx3(i, j, k)] = vid;
                            }
                        }
                    }

                    // Freudenthal / Bey decomposition of the level-S tetrahedral lattice into S^3 tets.
                    for k in 0..s {
                        for j in 0..(s - k) {
                            for i in 0..(s - k - j) {
                                let sum = i + j + k;
                                // 1. Upright sub-tet (always present for i + j + k < S)
                                let p000 = grid_vid[idx3(i, j, k)];
                                let p100 = grid_vid[idx3(i + 1, j, k)];
                                let p010 = grid_vid[idx3(i, j + 1, k)];
                                let p001 = grid_vid[idx3(i, j, k + 1)];
                                emit_tet(
                                    [p000, p100, p010, p001],
                                    source_cell,
                                    &subdiv_ctx.extra_points,
                                    &mut tets,
                                    &mut tet_source_cells,
                                );

                                // 2. Octahedron -> 4 sub-tets along interior diagonal (p001, p110)
                                if sum + 2 <= s {
                                    let p011 = grid_vid[idx3(i, j + 1, k + 1)];
                                    let p101 = grid_vid[idx3(i + 1, j, k + 1)];
                                    let p110 = grid_vid[idx3(i + 1, j + 1, k)];
                                    for oct_tet in [
                                        [p001, p110, p100, p010],
                                        [p001, p110, p010, p011],
                                        [p001, p110, p011, p101],
                                        [p001, p110, p101, p100],
                                    ] {
                                        emit_tet(
                                            oct_tet,
                                            source_cell,
                                            &subdiv_ctx.extra_points,
                                            &mut tets,
                                            &mut tet_source_cells,
                                        );
                                    }
                                }

                                // 3. Inverted sub-tet
                                if sum + 3 <= s {
                                    let p011 = grid_vid[idx3(i, j + 1, k + 1)];
                                    let p101 = grid_vid[idx3(i + 1, j, k + 1)];
                                    let p110 = grid_vid[idx3(i + 1, j + 1, k)];
                                    let p111 = grid_vid[idx3(i + 1, j + 1, k + 1)];
                                    emit_tet(
                                        [p110, p101, p011, p111],
                                        source_cell,
                                        &subdiv_ctx.extra_points,
                                        &mut tets,
                                        &mut tet_source_cells,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Higher-order 3D Hexahedron (`QuadraticHexahedron`, `LagrangeHexahedron`).
            CellType::QuadraticHexahedron | CellType::LagrangeHexahedron { .. } => {
                if subdiv == 1 {
                    let v = [
                        conn[0], conn[1], conn[2], conn[3], conn[4], conn[5], conn[6], conn[7],
                    ];
                    for tet in decompose_hex_face_consistent(v) {
                        emit_tet(
                            tet,
                            source_cell,
                            &subdiv_ctx.extra_points,
                            &mut tets,
                            &mut tet_source_cells,
                        );
                    }
                } else {
                    let s = usize::from(subdiv);
                    let sf = f64::from(subdiv);
                    let stride = s + 1;
                    let mut grid_vid = vec![0u32; stride * stride * stride];
                    let idx3 = |i: usize, j: usize, k: usize| (k * stride + j) * stride + i;

                    for k in 0..=s {
                        for j in 0..=s {
                            for i in 0..=s {
                                let rst = [i as f64 / sf, j as f64 / sf, k as f64 / sf];
                                let vid = subdiv_ctx.resolve_sample_vertex(
                                    cell_type, rst, conn, conn_start, conn_end,
                                )?;
                                grid_vid[idx3(i, j, k)] = vid;
                            }
                        }
                    }

                    for k in 0..s {
                        for j in 0..s {
                            for i in 0..s {
                                let sub_hex = [
                                    grid_vid[idx3(i, j, k)],
                                    grid_vid[idx3(i + 1, j, k)],
                                    grid_vid[idx3(i + 1, j + 1, k)],
                                    grid_vid[idx3(i, j + 1, k)],
                                    grid_vid[idx3(i, j, k + 1)],
                                    grid_vid[idx3(i + 1, j, k + 1)],
                                    grid_vid[idx3(i + 1, j + 1, k + 1)],
                                    grid_vid[idx3(i, j + 1, k + 1)],
                                ];
                                for tet in decompose_hex_face_consistent(sub_hex) {
                                    emit_tet(
                                        tet,
                                        source_cell,
                                        &subdiv_ctx.extra_points,
                                        &mut tets,
                                        &mut tet_source_cells,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Higher-order 3D Wedge (`QuadraticWedge`, `LagrangeWedge`).
            CellType::QuadraticWedge | CellType::LagrangeWedge { .. } => {
                if subdiv == 1 {
                    let v = [conn[0], conn[1], conn[2], conn[3], conn[4], conn[5]];
                    for tet in decompose_wedge_face_consistent(v) {
                        emit_tet(
                            tet,
                            source_cell,
                            &subdiv_ctx.extra_points,
                            &mut tets,
                            &mut tet_source_cells,
                        );
                    }
                } else {
                    let s = usize::from(subdiv);
                    let sf = f64::from(subdiv);
                    let stride = s + 1;
                    let mut grid_vid = vec![0u32; stride * stride * stride];
                    let idx3 = |i: usize, j: usize, k: usize| (k * stride + j) * stride + i;

                    for k in 0..=s {
                        for j in 0..=s {
                            for i in 0..=(s - j) {
                                let rst = [i as f64 / sf, j as f64 / sf, k as f64 / sf];
                                let vid = subdiv_ctx.resolve_sample_vertex(
                                    cell_type, rst, conn, conn_start, conn_end,
                                )?;
                                grid_vid[idx3(i, j, k)] = vid;
                            }
                        }
                    }

                    for k in 0..s {
                        for j in 0..s {
                            for i in 0..(s - j) {
                                // Upright sub-wedge
                                let upright = [
                                    grid_vid[idx3(i, j, k)],
                                    grid_vid[idx3(i + 1, j, k)],
                                    grid_vid[idx3(i, j + 1, k)],
                                    grid_vid[idx3(i, j, k + 1)],
                                    grid_vid[idx3(i + 1, j, k + 1)],
                                    grid_vid[idx3(i, j + 1, k + 1)],
                                ];
                                for tet in decompose_wedge_face_consistent(upright) {
                                    emit_tet(
                                        tet,
                                        source_cell,
                                        &subdiv_ctx.extra_points,
                                        &mut tets,
                                        &mut tet_source_cells,
                                    );
                                }

                                // Inverted sub-wedge (when i + j + 2 <= S)
                                if i + j + 2 <= s {
                                    let inverted = [
                                        grid_vid[idx3(i + 1, j, k)],
                                        grid_vid[idx3(i + 1, j + 1, k)],
                                        grid_vid[idx3(i, j + 1, k)],
                                        grid_vid[idx3(i + 1, j, k + 1)],
                                        grid_vid[idx3(i + 1, j + 1, k + 1)],
                                        grid_vid[idx3(i, j + 1, k + 1)],
                                    ];
                                    for tet in decompose_wedge_face_consistent(inverted) {
                                        emit_tet(
                                            tet,
                                            source_cell,
                                            &subdiv_ctx.extra_points,
                                            &mut tets,
                                            &mut tet_source_cells,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Higher-order 3D Pyramid (`QuadraticPyramid`).
            CellType::QuadraticPyramid => {
                if subdiv == 1 {
                    let v = [conn[0], conn[1], conn[2], conn[3], conn[4]];
                    for tet in decompose_pyramid_face_consistent(v) {
                        emit_tet(
                            tet,
                            source_cell,
                            &subdiv_ctx.extra_points,
                            &mut tets,
                            &mut tet_source_cells,
                        );
                    }
                } else {
                    let s = usize::from(subdiv);
                    let sf = f64::from(subdiv);
                    let stride = s + 1;
                    let mut grid_vid = vec![0u32; stride * stride * stride];
                    let idx3 = |i: usize, j: usize, k: usize| (k * stride + j) * stride + i;

                    for k in 0..s {
                        for j in 0..=s {
                            for i in 0..=s {
                                let rst = [i as f64 / sf, j as f64 / sf, k as f64 / sf];
                                let vid = subdiv_ctx.resolve_sample_vertex(
                                    cell_type, rst, conn, conn_start, conn_end,
                                )?;
                                grid_vid[idx3(i, j, k)] = vid;
                            }
                        }
                    }

                    // Frustum layers k = 0..(S - 1) -> S^2 sub-hexahedra per layer
                    for k in 0..(s - 1) {
                        for j in 0..s {
                            for i in 0..s {
                                let sub_hex = [
                                    grid_vid[idx3(i, j, k)],
                                    grid_vid[idx3(i + 1, j, k)],
                                    grid_vid[idx3(i + 1, j + 1, k)],
                                    grid_vid[idx3(i, j + 1, k)],
                                    grid_vid[idx3(i, j, k + 1)],
                                    grid_vid[idx3(i + 1, j, k + 1)],
                                    grid_vid[idx3(i + 1, j + 1, k + 1)],
                                    grid_vid[idx3(i, j + 1, k + 1)],
                                ];
                                for tet in decompose_hex_face_consistent(sub_hex) {
                                    emit_tet(
                                        tet,
                                        source_cell,
                                        &subdiv_ctx.extra_points,
                                        &mut tets,
                                        &mut tet_source_cells,
                                    );
                                }
                            }
                        }
                    }

                    // Top cap layer k = S - 1 joined to apex conn[4] -> S^2 sub-pyramids
                    let apex = conn[4];
                    let k_top = s - 1;
                    for j in 0..s {
                        for i in 0..s {
                            let sub_pyr = [
                                grid_vid[idx3(i, j, k_top)],
                                grid_vid[idx3(i + 1, j, k_top)],
                                grid_vid[idx3(i + 1, j + 1, k_top)],
                                grid_vid[idx3(i, j + 1, k_top)],
                                apex,
                            ];
                            for tet in decompose_pyramid_face_consistent(sub_pyr) {
                                emit_tet(
                                    tet,
                                    source_cell,
                                    &subdiv_ctx.extra_points,
                                    &mut tets,
                                    &mut tet_source_cells,
                                );
                            }
                        }
                    }
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
                let centroid_idx =
                    try_u32_from_usize(base_num_points + subdiv_ctx.extra_points.len())?;
                subdiv_ctx.extra_points.push(centroid_pos);
                subdiv_ctx.synth_interps.push(SynthPointInterp {
                    conn_start,
                    conn_end,
                    weight_start: None,
                });

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
                            emit_tet(
                                [v_min, v_a, v_b, centroid_idx],
                                source_cell,
                                &subdiv_ctx.extra_points,
                                &mut tets,
                                &mut tet_source_cells,
                            );
                        }
                    }
                }
            }
        }
    }

    let (points_arc, point_data) = if subdiv_ctx.extra_points.is_empty() {
        (grid.points_arc(), grid.point_data().clone())
    } else {
        let total_points = base_num_points + subdiv_ctx.extra_points.len();
        let mut all_points = Vec::with_capacity(total_points);
        all_points.extend_from_slice(base_points);
        all_points.extend_from_slice(&subdiv_ctx.extra_points);

        let mut pd = Attributes::new(total_points);
        let full_conn = grid.connectivity();
        for (name, arr) in grid.point_data().iter() {
            let extended = extend_point_array_with_synth_points(
                arr,
                full_conn,
                &subdiv_ctx.synth_interps,
                &subdiv_ctx.interp_weights,
            )?;
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

struct SubdivContext<'a> {
    base_points: &'a [[f64; 3]],
    inv_quant_step: f64,
    weights_buf: Vec<f64>,
    extra_points: Vec<[f64; 3]>,
    synth_interps: Vec<SynthPointInterp>,
    interp_weights: Vec<f64>,
    synth_point_map: HashMap<[i64; 3], u32>,
}

impl SubdivContext<'_> {
    fn resolve_sample_vertex(
        &mut self,
        cell_type: CellType,
        rst: [f64; 3],
        conn: &[u32],
        conn_start: usize,
        conn_end: usize,
    ) -> Result<u32, CoreError> {
        self.weights_buf.resize(conn.len(), 0.0);
        evaluate_cell_basis(cell_type, rst, &mut self.weights_buf);

        // Fast path: if this sample point coincides with one of the cell's own nodes
        // (weight >= 1 - 1e-11), reuse that existing global vertex index directly.
        for (slot, &w) in self.weights_buf.iter().enumerate() {
            if (w - 1.0).abs() < 1e-11 {
                let mut others_zero = true;
                for (other_slot, &ow) in self.weights_buf.iter().enumerate() {
                    if other_slot != slot && ow.abs() > 1e-11 {
                        others_zero = false;
                        break;
                    }
                }
                if others_zero {
                    return Ok(conn[slot]);
                }
            }
        }

        let mut pos = [0.0_f64; 3];
        for (&pt_idx, &w) in conn.iter().zip(self.weights_buf.iter()) {
            let p = self.base_points[pt_idx as usize];
            pos[0] += w * p[0];
            pos[1] += w * p[1];
            pos[2] += w * p[2];
        }

        let key = [
            (pos[0] * self.inv_quant_step).round() as i64,
            (pos[1] * self.inv_quant_step).round() as i64,
            (pos[2] * self.inv_quant_step).round() as i64,
        ];
        if let Some(&existing_vid) = self.synth_point_map.get(&key) {
            return Ok(existing_vid);
        }

        let new_vid = try_u32_from_usize(self.base_points.len() + self.extra_points.len())?;
        self.extra_points.push(pos);
        let w_start = self.interp_weights.len();
        self.interp_weights.extend_from_slice(&self.weights_buf);
        self.synth_interps.push(SynthPointInterp {
            conn_start,
            conn_end,
            weight_start: Some(w_start),
        });
        self.synth_point_map.insert(key, new_vid);
        Ok(new_vid)
    }
}

/// Decomposes a 5-node pyramid `[v0, v1, v2, v3, v4]` into 2 tetrahedra split along
/// the base quad diagonal incident to the minimum global vertex ID of `{v0, v1, v2, v3}`.
fn decompose_pyramid_face_consistent(v: [u32; 5]) -> [[u32; 4]; 2] {
    let [v0, v1, v2, v3, v4] = v;
    if v0.min(v2) < v1.min(v3) {
        [[v0, v1, v2, v4], [v0, v2, v3, v4]]
    } else {
        [[v1, v2, v3, v4], [v1, v3, v0, v4]]
    }
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

fn extend_point_array_with_synth_points(
    source: &DataArray,
    full_conn: &[u32],
    synth_interps: &[SynthPointInterp],
    interp_weights: &[f64],
) -> Result<DataArray, CoreError> {
    let comp = source.components();
    let ncomp = comp.as_usize();
    let extra_tuples = synth_interps.len();

    let eval_component = |interp: &SynthPointInterp, c: usize, get_val: &dyn Fn(usize) -> f64| {
        let verts = &full_conn[interp.conn_start..interp.conn_end];
        if let Some(w_start) = interp.weight_start {
            let weights = &interp_weights[w_start..w_start + verts.len()];
            let mut sum = 0.0_f64;
            for (&pt, &w) in verts.iter().zip(weights.iter()) {
                sum += w * get_val((pt as usize) * ncomp + c);
            }
            sum
        } else {
            let inv_n = 1.0 / (verts.len() as f64);
            let mut sum = 0.0_f64;
            for &pt in verts {
                sum += get_val((pt as usize) * ncomp + c);
            }
            sum * inv_n
        }
    };

    match source.buffer() {
        ArrayBuffer::F32(buf) => {
            let mut out = Vec::with_capacity(buf.len() + extra_tuples * ncomp);
            out.extend_from_slice(buf);
            for interp in synth_interps {
                for c in 0..ncomp {
                    let v = eval_component(interp, c, &|idx| f64::from(buf[idx]));
                    out.push(v as f32);
                }
            }
            DataArray::from_f32(comp, out)
        }
        ArrayBuffer::F64(buf) => {
            let mut out = Vec::with_capacity(buf.len() + extra_tuples * ncomp);
            out.extend_from_slice(buf);
            for interp in synth_interps {
                for c in 0..ncomp {
                    let v = eval_component(interp, c, &|idx| buf[idx]);
                    out.push(v);
                }
            }
            DataArray::from_f64(comp, out)
        }
        ArrayBuffer::I32(buf) => {
            let mut out = Vec::with_capacity(buf.len() + extra_tuples * ncomp);
            out.extend_from_slice(buf);
            for interp in synth_interps {
                for c in 0..ncomp {
                    let v = eval_component(interp, c, &|idx| f64::from(buf[idx]));
                    out.push(v.round() as i32);
                }
            }
            DataArray::from_i32(comp, out)
        }
        ArrayBuffer::I64(buf) => {
            let mut out = Vec::with_capacity(buf.len() + extra_tuples * ncomp);
            out.extend_from_slice(buf);
            for interp in synth_interps {
                for c in 0..ncomp {
                    let v = eval_component(interp, c, &|idx| buf[idx] as f64);
                    out.push(v.round() as i64);
                }
            }
            DataArray::from_i64(comp, out)
        }
        ArrayBuffer::U8(buf) => {
            let mut out = Vec::with_capacity(buf.len() + extra_tuples * ncomp);
            out.extend_from_slice(buf);
            for interp in synth_interps {
                for c in 0..ncomp {
                    let v = eval_component(interp, c, &|idx| f64::from(buf[idx]));
                    out.push(v.round().clamp(0.0, 255.0) as u8);
                }
            }
            DataArray::from_u8(comp, out)
        }
    }
}

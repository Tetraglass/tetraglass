//! Shared marching-tetrahedra isosurface kernel (`M2.3a`, `D-005`, `D-006`).
//!
//! Consumes only [`TetView`] and powers `slice`, `contour`, and `clip` surface extraction.
//!
//! # Kernel invariants (`.agents/skills/add-filter/SKILL.md`)
//!
//! 1. **Edge interpolation:** `t = ((iso - a) / (b - a)).clamp(0.0, 1.0)` along canonicalized
//!    edge endpoints `(p_min, p_max)`.
//! 2. **Tie-breaking rule:** A vertex value `val >= iso` is classified as *above* (`1`), and
//!    `val < iso` as *below* (`0`). When an intersection lands exactly at an endpoint (`t == 0.0`
//!    or `t == 1.0`), its deduplication key collapses to the vertex key `EdgeKey(v, v)` so
//!    planes passing through vertices or faces never produce cracks or duplicate triangles.
//! 3. **Edge deduplication:** Output vertices are deduplicated by `(iso_idx, EdgeKey(min(p, q), max(p, q)))`
//!    so output surfaces are watertight and point attributes are interpolated once per cut edge.
//! 4. **Deterministic chunked execution:** Each chunk of tetrahedra builds its own local edge map
//!    and triangle list; merging in chunk order produces bit-for-bit identical output across any
//!    chunk size and thread count.

use std::collections::HashMap;
use std::sync::Arc;

use tetraglass_core::{
    ArrayBuffer, Attributes, ComponentCount, CoreError, DataArray, SurfaceMesh, TetView,
    signed_tet_volume, try_u32_from_usize,
};

use crate::surface::{FilterError, map_cell_array, push_canonical_edge};

/// Default number of tetrahedra processed per parallel chunk.
pub const DEFAULT_TET_CHUNK_SIZE: usize = 8192;

/// Canonical edge key `(min(p, q), max(p, q))` of `TetView` point indices.
/// When an intersection lands exactly on a vertex `v`, `EdgeKey(v, v)` is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeKey(pub u32, pub u32);

impl EdgeKey {
    /// Constructs a canonical [`EdgeKey`] with `self.0 <= self.1`.
    #[inline]
    #[must_use]
    pub const fn new(a: u32, b: u32) -> Self {
        if a <= b { Self(a, b) } else { Self(b, a) }
    }
}

/// The 6 local edges `(v_a, v_b)` of a tetrahedron `[v0, v1, v2, v3]`.
const TET_EDGES: [[usize; 2]; 6] = [
    [0, 1], // 0: E01
    [1, 2], // 1: E12
    [2, 0], // 2: E20
    [0, 3], // 3: E03
    [1, 3], // 4: E13
    [2, 3], // 5: E23
];

/// Polygon cut case for a single tetrahedron mask `0..16`.
#[derive(Debug, Clone, Copy)]
enum TetCutCase {
    Empty,
    Tri([u8; 3]),
    Quad([u8; 4]),
}

/// Marching-tetrahedra lookup table indexed by `mask = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)`,
/// where `b_k = 1` iff `val[v_k] >= iso`.
///
/// For positively oriented tetrahedra (`(p1 - p0) x (p2 - p0) . (p3 - p0) > 0`), all emitted
/// triangles have normals pointing toward the `>= iso` side ($+\nabla \text{val}$).
const MARCHING_TET_TABLE: [TetCutCase; 16] = [
    TetCutCase::Empty,              // 0b0000 (0): none above
    TetCutCase::Tri([0, 3, 2]),     // 0b0001 (1): v0 above
    TetCutCase::Tri([0, 1, 4]),     // 0b0010 (2): v1 above
    TetCutCase::Quad([2, 1, 4, 3]), // 0b0011 (3): v0, v1 above
    TetCutCase::Tri([1, 2, 5]),     // 0b0100 (4): v2 above
    TetCutCase::Quad([0, 3, 5, 1]), // 0b0101 (5): v0, v2 above
    TetCutCase::Quad([0, 2, 5, 4]), // 0b0110 (6): v1, v2 above
    TetCutCase::Tri([3, 5, 4]),     // 0b0111 (7): v0, v1, v2 above (v3 below)
    TetCutCase::Tri([3, 4, 5]),     // 0b1000 (8): v3 above
    TetCutCase::Quad([0, 4, 5, 2]), // 0b1001 (9): v0, v3 above
    TetCutCase::Quad([0, 1, 5, 3]), // 0b1010 (10): v1, v3 above
    TetCutCase::Tri([1, 5, 2]),     // 0b1011 (11): v0, v1, v3 above (v2 below)
    TetCutCase::Quad([2, 3, 4, 1]), // 0b1100 (12): v2, v3 above
    TetCutCase::Tri([0, 4, 1]),     // 0b1101 (13): v0, v2, v3 above (v1 below)
    TetCutCase::Tri([0, 2, 3]),     // 0b1110 (14): v1, v2, v3 above (v0 below)
    TetCutCase::Empty,              // 0b1111 (15): all above
];

/// Extracts a single isosurface `val == iso` from `view` using the default chunk size.
pub fn extract_isosurface(
    view: &TetView,
    scalars: &[f64],
    iso: f64,
) -> Result<SurfaceMesh, FilterError> {
    extract_isosurfaces_with_chunk_size(view, scalars, &[iso], DEFAULT_TET_CHUNK_SIZE)
}

/// Extracts one or more isosurfaces `val == isovalues[k]` from `view` using an explicit
/// `chunk_size` (number of tetrahedra per chunk).
///
/// Output vertices, triangles, wireframe edges, and interpolated attributes are deterministic
/// and independent of `chunk_size` and thread count.
pub fn extract_isosurfaces_with_chunk_size(
    view: &TetView,
    scalars: &[f64],
    isovalues: &[f64],
    chunk_size: usize,
) -> Result<SurfaceMesh, FilterError> {
    assert_eq!(
        scalars.len(),
        view.num_points(),
        "scalars length must match view.num_points()"
    );
    let chunk_size = chunk_size.max(1);
    let tets = view.tets();
    let tet_source_cells = view.tet_source_cells();

    if tets.is_empty() || isovalues.is_empty() {
        let empty_points: Arc<[[f64; 3]]> = Arc::from(Vec::<[f64; 3]>::new());
        let point_data = empty_attributes_like(view.point_data());
        let cell_data = empty_attributes_like(view.cell_data());
        return Ok(
            SurfaceMesh::new(empty_points, vec![], vec![], vec![], point_data, cell_data)?
                .with_wireframe_edges(vec![])?,
        );
    }

    let mut s_min = f64::INFINITY;
    let mut s_max = f64::NEG_INFINITY;
    for &s in scalars {
        if s < s_min {
            s_min = s;
        }
        if s > s_max {
            s_max = s;
        }
    }
    let scalar_span = if s_max > s_min && (s_max - s_min).is_finite() {
        (s_max - s_min).max(1.0)
    } else {
        1.0
    };
    let iso_tol = 1e-12 * scalar_span;

    let num_tets = tets.len();
    let num_chunks = num_tets.div_ceil(chunk_size);
    let chunk_outputs = run_chunks(
        tets,
        tet_source_cells,
        scalars,
        isovalues,
        iso_tol,
        chunk_size,
        num_chunks,
    );

    // Deterministic merge in chunk order `0..num_chunks`.
    let points_src = view.points();
    let mut global_edge_map: HashMap<(u32, EdgeKey), u32> = HashMap::new();
    let mut out_points: Vec<[f64; 3]> = Vec::new();
    let mut out_edges: Vec<(EdgeKey, f64)> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut triangle_source_cells: Vec<u32> = Vec::new();
    let mut triangle_tet_indices: Vec<u32> = Vec::new();
    let mut face_source_cells: Vec<u32> = Vec::new();
    let mut wireframe_edges: Vec<[u32; 2]> = Vec::new();

    for chunk in chunk_outputs {
        let mut local_to_global = Vec::with_capacity(chunk.local_edges.len());
        for (iso_idx, edge_key, t) in chunk.local_edges {
            if let Some(&existing_vid) = global_edge_map.get(&(iso_idx, edge_key)) {
                local_to_global.push(existing_vid);
            } else {
                let vid = try_u32_from_usize(out_points.len())?;
                let p0 = points_src[edge_key.0 as usize];
                let p1 = points_src[edge_key.1 as usize];
                let pos = if edge_key.0 == edge_key.1 || t == 0.0 {
                    p0
                } else if t == 1.0 {
                    p1
                } else {
                    let omt = 1.0 - t;
                    [
                        omt * p0[0] + t * p1[0],
                        omt * p0[1] + t * p1[1],
                        omt * p0[2] + t * p1[2],
                    ]
                };
                out_points.push(pos);
                out_edges.push((edge_key, t));
                global_edge_map.insert((iso_idx, edge_key), vid);
                local_to_global.push(vid);
            }
        }

        for (([l0, l1, l2], src_cell), tet_idx) in chunk
            .local_triangles
            .into_iter()
            .zip(chunk.local_tri_source_cells)
            .zip(chunk.local_tri_tet_indices)
        {
            let g0 = local_to_global[l0 as usize];
            let g1 = local_to_global[l1 as usize];
            let g2 = local_to_global[l2 as usize];
            triangles.push([g0, g1, g2]);
            triangle_source_cells.push(src_cell);
            triangle_tet_indices.push(tet_idx);
        }

        face_source_cells.extend(chunk.local_face_source_cells);

        for [la, lb] in chunk.local_wireframe_edges {
            let ga = local_to_global[la as usize];
            let gb = local_to_global[lb as usize];
            push_canonical_edge(&mut wireframe_edges, ga, gb);
        }
    }

    wireframe_edges.sort_unstable();
    wireframe_edges.dedup();

    let mut point_data = Attributes::new(out_points.len());
    for (name, arr) in view.point_data().iter() {
        let interp = interpolate_edge_array(arr, &out_edges)?;
        point_data.insert(name, interp)?;
    }

    let mut cell_data = Attributes::new(triangles.len());
    for (name, arr) in view.cell_data().iter() {
        let mapped = map_cell_array(arr, &triangle_tet_indices)?;
        cell_data.insert(name, mapped)?;
    }

    let surface = SurfaceMesh::new(
        out_points,
        triangles,
        triangle_source_cells,
        face_source_cells,
        point_data,
        cell_data,
    )?
    .with_wireframe_edges(wireframe_edges)?;
    Ok(surface)
}

struct ChunkOutput {
    local_edges: Vec<(u32, EdgeKey, f64)>,
    local_triangles: Vec<[u32; 3]>,
    local_tri_source_cells: Vec<u32>,
    local_tri_tet_indices: Vec<u32>,
    local_face_source_cells: Vec<u32>,
    local_wireframe_edges: Vec<[u32; 2]>,
}

fn process_tet_chunk(
    tet_offset: u32,
    tets: &[[u32; 4]],
    tet_source_cells: &[u32],
    scalars: &[f64],
    isovalues: &[f64],
    iso_tol: f64,
) -> ChunkOutput {
    let mut local_edge_map: HashMap<(u32, EdgeKey), u32> = HashMap::new();
    let mut local_edges: Vec<(u32, EdgeKey, f64)> = Vec::new();
    let mut local_triangles: Vec<[u32; 3]> = Vec::new();
    let mut local_tri_source_cells: Vec<u32> = Vec::new();
    let mut local_tri_tet_indices: Vec<u32> = Vec::new();
    let mut local_face_source_cells: Vec<u32> = Vec::new();
    let mut local_wireframe_edges: Vec<[u32; 2]> = Vec::new();

    let snap_scalar =
        |s: f64, iso: f64| -> f64 { if (s - iso).abs() <= iso_tol { iso } else { s } };

    let mut get_or_insert_edge = |iso_idx: u32, p: u32, q: u32, iso: f64| -> u32 {
        let raw_key = EdgeKey::new(p, q);
        let a = snap_scalar(scalars[raw_key.0 as usize], iso);
        let b = snap_scalar(scalars[raw_key.1 as usize], iso);
        let denom = b - a;
        let t_raw = if denom.abs() > 0.0 {
            ((iso - a) / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // Collapse endpoint hits (`a == iso`, `b == iso`, or `t_raw` within `1e-12` of 0/1) to a
        // vertex key `EdgeKey(v, v)` so all edges incident to an on-isosurface vertex share a
        // single deduplicated vertex id without creating zero-area slivers.
        let (canonical_key, t) = if a == iso || t_raw <= 1e-12 {
            (EdgeKey(raw_key.0, raw_key.0), 0.0)
        } else if b == iso || t_raw >= 1.0 - 1e-12 {
            (EdgeKey(raw_key.1, raw_key.1), 0.0)
        } else {
            (raw_key, t_raw)
        };

        if let Some(&lid) = local_edge_map.get(&(iso_idx, canonical_key)) {
            lid
        } else {
            let lid = local_edges.len() as u32;
            local_edges.push((iso_idx, canonical_key, t));
            local_edge_map.insert((iso_idx, canonical_key), lid);
            lid
        }
    };

    for (local_tet_idx, &[v0, v1, v2, v3]) in tets.iter().enumerate() {
        let tet_verts = [v0, v1, v2, v3];
        let src_cell = tet_source_cells[local_tet_idx];
        let global_tet_idx = tet_offset + local_tet_idx as u32;

        for (iso_idx_usize, &iso) in isovalues.iter().enumerate() {
            let iso_idx = iso_idx_usize as u32;
            let s0 = snap_scalar(scalars[v0 as usize], iso);
            let s1 = snap_scalar(scalars[v1 as usize], iso);
            let s2 = snap_scalar(scalars[v2 as usize], iso);
            let s3 = snap_scalar(scalars[v3 as usize], iso);

            let b0 = u8::from(s0 >= iso);
            let b1 = u8::from(s1 >= iso);
            let b2 = u8::from(s2 >= iso);
            let b3 = u8::from(s3 >= iso);
            let mask = (b0 | (b1 << 1) | (b2 << 2) | (b3 << 3)) as usize;

            match MARCHING_TET_TABLE[mask] {
                TetCutCase::Empty => {}
                TetCutCase::Tri([e0, e1, e2]) => {
                    let [a0, b0_idx] = TET_EDGES[e0 as usize];
                    let [a1, b1_idx] = TET_EDGES[e1 as usize];
                    let [a2, b2_idx] = TET_EDGES[e2 as usize];
                    let i0 = get_or_insert_edge(iso_idx, tet_verts[a0], tet_verts[b0_idx], iso);
                    let i1 = get_or_insert_edge(iso_idx, tet_verts[a1], tet_verts[b1_idx], iso);
                    let i2 = get_or_insert_edge(iso_idx, tet_verts[a2], tet_verts[b2_idx], iso);

                    if i0 != i1 && i1 != i2 && i2 != i0 {
                        local_triangles.push([i0, i1, i2]);
                        local_tri_source_cells.push(src_cell);
                        local_tri_tet_indices.push(global_tet_idx);
                        local_face_source_cells.push(src_cell);
                        local_wireframe_edges.push([i0, i1]);
                        local_wireframe_edges.push([i1, i2]);
                        local_wireframe_edges.push([i2, i0]);
                    }
                }
                TetCutCase::Quad([e0, e1, e2, e3]) => {
                    let [a0, b0_idx] = TET_EDGES[e0 as usize];
                    let [a1, b1_idx] = TET_EDGES[e1 as usize];
                    let [a2, b2_idx] = TET_EDGES[e2 as usize];
                    let [a3, b3_idx] = TET_EDGES[e3 as usize];
                    let i0 = get_or_insert_edge(iso_idx, tet_verts[a0], tet_verts[b0_idx], iso);
                    let i1 = get_or_insert_edge(iso_idx, tet_verts[a1], tet_verts[b1_idx], iso);
                    let i2 = get_or_insert_edge(iso_idx, tet_verts[a2], tet_verts[b2_idx], iso);
                    let i3 = get_or_insert_edge(iso_idx, tet_verts[a3], tet_verts[b3_idx], iso);

                    let mut emitted_any = false;
                    if i0 != i1 && i1 != i2 && i2 != i0 {
                        local_triangles.push([i0, i1, i2]);
                        local_tri_source_cells.push(src_cell);
                        local_tri_tet_indices.push(global_tet_idx);
                        emitted_any = true;
                    }
                    if i0 != i2 && i2 != i3 && i3 != i0 {
                        local_triangles.push([i0, i2, i3]);
                        local_tri_source_cells.push(src_cell);
                        local_tri_tet_indices.push(global_tet_idx);
                        emitted_any = true;
                    }
                    if emitted_any {
                        local_face_source_cells.push(src_cell);
                        // Record the 4 perimeter edges of the quad cross-section on the tet faces
                        // (excluding the internal quad-triangulation diagonal `[i0, i2]`).
                        for [u, v] in [[i0, i1], [i1, i2], [i2, i3], [i3, i0]] {
                            if u != v {
                                local_wireframe_edges.push([u, v]);
                            }
                        }
                    }
                }
            }
        }
    }

    ChunkOutput {
        local_edges,
        local_triangles,
        local_tri_source_cells,
        local_tri_tet_indices,
        local_face_source_cells,
        local_wireframe_edges,
    }
}

fn run_chunks(
    tets: &[[u32; 4]],
    tet_source_cells: &[u32],
    scalars: &[f64],
    isovalues: &[f64],
    iso_tol: f64,
    chunk_size: usize,
    num_chunks: usize,
) -> Vec<ChunkOutput> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if num_chunks > 1 {
            let max_threads = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
                .min(num_chunks);
            if max_threads > 1 {
                let mut results: Vec<Option<ChunkOutput>> = (0..num_chunks).map(|_| None).collect();
                let chunks_per_worker = num_chunks.div_ceil(max_threads);
                std::thread::scope(|s| {
                    for (worker_idx, out_slice) in results.chunks_mut(chunks_per_worker).enumerate()
                    {
                        let start_chunk = worker_idx * chunks_per_worker;
                        s.spawn(move || {
                            for (offset, slot) in out_slice.iter_mut().enumerate() {
                                let chunk_idx = start_chunk + offset;
                                let start = chunk_idx * chunk_size;
                                let end = (start + chunk_size).min(tets.len());
                                *slot = Some(process_tet_chunk(
                                    start as u32,
                                    &tets[start..end],
                                    &tet_source_cells[start..end],
                                    scalars,
                                    isovalues,
                                    iso_tol,
                                ));
                            }
                        });
                    }
                });
                return results
                    .into_iter()
                    .map(|opt| opt.expect("chunk completed"))
                    .collect();
            }
        }
    }

    (0..num_chunks)
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(tets.len());
            process_tet_chunk(
                start as u32,
                &tets[start..end],
                &tet_source_cells[start..end],
                scalars,
                isovalues,
                iso_tol,
            )
        })
        .collect()
}

fn empty_attributes_like(source: &Attributes) -> Attributes {
    let mut out = Attributes::new(0);
    for (name, arr) in source.iter() {
        let comp = arr.components();
        let empty = match arr.buffer() {
            ArrayBuffer::F32(_) => DataArray::from_f32(comp, vec![]),
            ArrayBuffer::F64(_) => DataArray::from_f64(comp, vec![]),
            ArrayBuffer::I32(_) => DataArray::from_i32(comp, vec![]),
            ArrayBuffer::I64(_) => DataArray::from_i64(comp, vec![]),
            ArrayBuffer::U8(_) => DataArray::from_u8(comp, vec![]),
        }
        .expect("empty DataArray");
        let _ = out.insert(name, empty);
    }
    out
}

fn empty_tet_view_like(view: &TetView) -> Result<TetView, FilterError> {
    let empty_points: Arc<[[f64; 3]]> = Arc::from(Vec::<[f64; 3]>::new());
    let point_data = empty_attributes_like(view.point_data());
    let cell_data = empty_attributes_like(view.cell_data());
    Ok(TetView::new(
        empty_points,
        vec![],
        vec![],
        point_data,
        cell_data,
    )?)
}

fn compute_iso_tolerance(scalars: &[f64]) -> f64 {
    let mut s_min = f64::INFINITY;
    let mut s_max = f64::NEG_INFINITY;
    for &s in scalars {
        if s < s_min {
            s_min = s;
        }
        if s > s_max {
            s_max = s;
        }
    }
    let scalar_span = if s_max > s_min && (s_max - s_min).is_finite() {
        (s_max - s_min).max(1.0)
    } else {
        1.0
    };
    1e-12 * scalar_span
}

/// Clips `view` to the half-space `scalars >= iso` using exact conformal marching-tetrahedra
/// prism decomposition (`M2.5`, `D-005`, `D-006`).
///
/// Output vertices, tetrahedra, `tet_source_cells`, and interpolated attributes are deterministic
/// and independent of `chunk_size` and thread count.
pub fn clip_tets_with_chunk_size(
    view: &TetView,
    scalars: &[f64],
    iso: f64,
    chunk_size: usize,
) -> Result<TetView, FilterError> {
    assert_eq!(
        scalars.len(),
        view.num_points(),
        "scalars length must match view.num_points()"
    );
    let chunk_size = chunk_size.max(1);
    let tets = view.tets();
    let tet_source_cells = view.tet_source_cells();

    if tets.is_empty() {
        return empty_tet_view_like(view);
    }

    let iso_tol = compute_iso_tolerance(scalars);
    let num_tets = tets.len();
    let num_chunks = num_tets.div_ceil(chunk_size);
    let chunk_outputs = run_exact_clip_chunks(
        tets,
        tet_source_cells,
        scalars,
        iso,
        iso_tol,
        chunk_size,
        num_chunks,
    );

    merge_clip_chunks(view, chunk_outputs)
}

/// Clips `view` to the half-space `scalars >= iso` in crinkle mode (`M2.5`, `D-005`), keeping
/// whole original source cells (`tet_source_cells`) that overlap the kept half-space.
pub fn clip_crinkle_tets_with_chunk_size(
    view: &TetView,
    scalars: &[f64],
    iso: f64,
    chunk_size: usize,
) -> Result<TetView, FilterError> {
    assert_eq!(
        scalars.len(),
        view.num_points(),
        "scalars length must match view.num_points()"
    );
    let chunk_size = chunk_size.max(1);
    let tets = view.tets();
    let tet_source_cells = view.tet_source_cells();

    if tets.is_empty() {
        return empty_tet_view_like(view);
    }

    let iso_tol = compute_iso_tolerance(scalars);
    let snap_scalar = |s: f64| -> f64 { if (s - iso).abs() <= iso_tol { iso } else { s } };

    let max_src = tet_source_cells.iter().copied().max().unwrap_or(0) as usize;
    let mut keep_source_cell = vec![false; max_src + 1];
    for (tet_idx, &[v0, v1, v2, v3]) in tets.iter().enumerate() {
        let s0 = snap_scalar(scalars[v0 as usize]);
        let s1 = snap_scalar(scalars[v1 as usize]);
        let s2 = snap_scalar(scalars[v2 as usize]);
        let s3 = snap_scalar(scalars[v3 as usize]);
        let overlaps = (s0 > iso || s1 > iso || s2 > iso || s3 > iso)
            || (s0 >= iso && s1 >= iso && s2 >= iso && s3 >= iso);
        if overlaps {
            keep_source_cell[tet_source_cells[tet_idx] as usize] = true;
        }
    }

    filter_tets_by_source_cell_with_chunk_size(view, &keep_source_cell, chunk_size)
}

/// Extracts and compacts all tetrahedra of `view` whose originating source cell index `src_cell`
/// satisfies `keep_source_cell[src_cell as usize] == true`, using deterministic parallel chunks (`M2.5`, `M2.6`).
pub fn filter_tets_by_source_cell_with_chunk_size(
    view: &TetView,
    keep_source_cell: &[bool],
    chunk_size: usize,
) -> Result<TetView, FilterError> {
    let chunk_size = chunk_size.max(1);
    let tets = view.tets();
    let tet_source_cells = view.tet_source_cells();

    if tets.is_empty() {
        return empty_tet_view_like(view);
    }

    let num_tets = tets.len();
    let num_chunks = num_tets.div_ceil(chunk_size);
    let chunk_outputs = run_crinkle_clip_chunks(
        tets,
        tet_source_cells,
        keep_source_cell,
        chunk_size,
        num_chunks,
    );

    merge_clip_chunks(view, chunk_outputs)
}

struct ClipChunkOutput {
    local_edges: Vec<(EdgeKey, f64)>,
    local_tets: Vec<[u32; 4]>,
    local_tet_source_cells: Vec<u32>,
    local_tet_indices: Vec<u32>,
}

fn merge_clip_chunks(
    view: &TetView,
    chunk_outputs: Vec<ClipChunkOutput>,
) -> Result<TetView, FilterError> {
    let points_src = view.points();
    let diag = view.bounds().map_or(1.0, |b| b.diagonal()).max(1e-12);
    let vol_tol = 1e-18 * diag * diag * diag;

    let mut global_edge_map: HashMap<EdgeKey, u32> = HashMap::new();
    let mut out_points: Vec<[f64; 3]> = Vec::new();
    let mut out_edges: Vec<(EdgeKey, f64)> = Vec::new();
    let mut out_tets: Vec<[u32; 4]> = Vec::new();
    let mut out_tet_source_cells: Vec<u32> = Vec::new();
    let mut out_tet_indices: Vec<u32> = Vec::new();

    for chunk in chunk_outputs {
        let mut local_to_global = Vec::with_capacity(chunk.local_edges.len());
        for (edge_key, t) in chunk.local_edges {
            if let Some(&existing_vid) = global_edge_map.get(&edge_key) {
                local_to_global.push(existing_vid);
            } else {
                let vid = try_u32_from_usize(out_points.len())?;
                let p0 = points_src[edge_key.0 as usize];
                let p1 = points_src[edge_key.1 as usize];
                let pos = if edge_key.0 == edge_key.1 || t == 0.0 {
                    p0
                } else if t == 1.0 {
                    p1
                } else {
                    let omt = 1.0 - t;
                    [
                        omt * p0[0] + t * p1[0],
                        omt * p0[1] + t * p1[1],
                        omt * p0[2] + t * p1[2],
                    ]
                };
                out_points.push(pos);
                out_edges.push((edge_key, t));
                global_edge_map.insert(edge_key, vid);
                local_to_global.push(vid);
            }
        }

        for (([l0, l1, l2, l3], src_cell), tet_idx) in chunk
            .local_tets
            .into_iter()
            .zip(chunk.local_tet_source_cells)
            .zip(chunk.local_tet_indices)
        {
            let g0 = local_to_global[l0 as usize];
            let g1 = local_to_global[l1 as usize];
            let g2 = local_to_global[l2 as usize];
            let g3 = local_to_global[l3 as usize];
            if g0 == g1 || g0 == g2 || g0 == g3 || g1 == g2 || g1 == g3 || g2 == g3 {
                continue;
            }
            let vol = signed_tet_volume(
                out_points[g0 as usize],
                out_points[g1 as usize],
                out_points[g2 as usize],
                out_points[g3 as usize],
            );
            if vol.abs() <= vol_tol {
                continue;
            }
            let oriented = if vol < 0.0 {
                [g0, g2, g1, g3]
            } else {
                [g0, g1, g2, g3]
            };
            out_tets.push(oriented);
            out_tet_source_cells.push(src_cell);
            out_tet_indices.push(tet_idx);
        }
    }

    if out_tets.is_empty() {
        return empty_tet_view_like(view);
    }

    let mut point_data = Attributes::new(out_points.len());
    for (name, arr) in view.point_data().iter() {
        let interp = interpolate_edge_array(arr, &out_edges)?;
        point_data.insert(name, interp)?;
    }

    let mut cell_data = Attributes::new(out_tets.len());
    for (name, arr) in view.cell_data().iter() {
        let mapped = map_cell_array(arr, &out_tet_indices)?;
        cell_data.insert(name, mapped)?;
    }

    Ok(TetView::new(
        out_points,
        out_tets,
        out_tet_source_cells,
        point_data,
        cell_data,
    )?)
}

fn process_exact_clip_chunk(
    tet_offset: u32,
    tets: &[[u32; 4]],
    tet_source_cells: &[u32],
    scalars: &[f64],
    iso: f64,
    iso_tol: f64,
) -> ClipChunkOutput {
    let mut local_edge_map: HashMap<EdgeKey, u32> = HashMap::new();
    let mut local_edges: Vec<(EdgeKey, f64)> = Vec::new();
    let mut local_tets: Vec<[u32; 4]> = Vec::new();
    let mut local_tet_source_cells: Vec<u32> = Vec::new();
    let mut local_tet_indices: Vec<u32> = Vec::new();

    let snap_scalar = |s: f64| -> f64 { if (s - iso).abs() <= iso_tol { iso } else { s } };

    let mut get_or_insert_edge = |p: u32, q: u32| -> u32 {
        let (canonical_key, t) = if p == q {
            (EdgeKey(p, p), 0.0)
        } else {
            let raw_key = EdgeKey::new(p, q);
            let a = snap_scalar(scalars[raw_key.0 as usize]);
            let b = snap_scalar(scalars[raw_key.1 as usize]);
            let denom = b - a;
            let t_raw = if denom.abs() > 0.0 {
                ((iso - a) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            if a == iso || t_raw <= 1e-12 {
                (EdgeKey(raw_key.0, raw_key.0), 0.0)
            } else if b == iso || t_raw >= 1.0 - 1e-12 {
                (EdgeKey(raw_key.1, raw_key.1), 0.0)
            } else {
                (raw_key, t_raw)
            }
        };

        if let Some(&lid) = local_edge_map.get(&canonical_key) {
            lid
        } else {
            let lid = local_edges.len() as u32;
            local_edges.push((canonical_key, t));
            local_edge_map.insert(canonical_key, lid);
            lid
        }
    };

    for (local_tet_idx, &[v0, v1, v2, v3]) in tets.iter().enumerate() {
        let src_cell = tet_source_cells[local_tet_idx];
        let global_tet_idx = tet_offset + local_tet_idx as u32;

        let s0 = snap_scalar(scalars[v0 as usize]);
        let s1 = snap_scalar(scalars[v1 as usize]);
        let s2 = snap_scalar(scalars[v2 as usize]);
        let s3 = snap_scalar(scalars[v3 as usize]);

        // Skip tetrahedra that lie entirely in `s <= iso` with at least one vertex strictly below `iso`.
        if s0 <= iso
            && s1 <= iso
            && s2 <= iso
            && s3 <= iso
            && (s0 < iso || s1 < iso || s2 < iso || s3 < iso)
        {
            continue;
        }

        let mut above = [0u32; 4];
        let mut num_above = 0usize;
        let mut below = [0u32; 4];
        let mut num_below = 0usize;

        for (v, s) in [(v0, s0), (v1, s1), (v2, s2), (v3, s3)] {
            if s >= iso {
                above[num_above] = v;
                num_above += 1;
            } else {
                below[num_below] = v;
                num_below += 1;
            }
        }
        above[..num_above].sort_unstable();
        below[..num_below].sort_unstable();

        match num_above {
            0 => {}
            4 => {
                let i0 = get_or_insert_edge(v0, v0);
                let i1 = get_or_insert_edge(v1, v1);
                let i2 = get_or_insert_edge(v2, v2);
                let i3 = get_or_insert_edge(v3, v3);
                local_tets.push([i0, i1, i2, i3]);
                local_tet_source_cells.push(src_cell);
                local_tet_indices.push(global_tet_idx);
            }
            1 => {
                let u0 = above[0];
                let d0 = below[0];
                let d1 = below[1];
                let d2 = below[2];
                let iu0 = get_or_insert_edge(u0, u0);
                let ie0 = get_or_insert_edge(u0, d0);
                let ie1 = get_or_insert_edge(u0, d1);
                let ie2 = get_or_insert_edge(u0, d2);
                local_tets.push([iu0, ie0, ie1, ie2]);
                local_tet_source_cells.push(src_cell);
                local_tet_indices.push(global_tet_idx);
            }
            2 => {
                // Conformal 3-tet prism decomposition for `u0 < u1` and `d0 < d1`.
                // Splits each shared quad face `(u0, u1, e(u1, d), e(u0, d))` along `(u0, e(u1, d))`.
                let u0 = above[0];
                let u1 = above[1];
                let d0 = below[0];
                let d1 = below[1];
                let iu0 = get_or_insert_edge(u0, u0);
                let iu1 = get_or_insert_edge(u1, u1);
                let ie00 = get_or_insert_edge(u0, d0);
                let ie01 = get_or_insert_edge(u0, d1);
                let ie10 = get_or_insert_edge(u1, d0);
                let ie11 = get_or_insert_edge(u1, d1);
                for sub_tet in [
                    [iu0, iu1, ie10, ie11],
                    [iu0, ie00, ie10, ie11],
                    [iu0, ie11, ie01, ie00],
                ] {
                    local_tets.push(sub_tet);
                    local_tet_source_cells.push(src_cell);
                    local_tet_indices.push(global_tet_idx);
                }
            }
            3 => {
                // Conformal 3-tet prism decomposition for `u0 < u1 < u2` and `d0`.
                // Splits each shared quad face `(u_a, u_b, w_b, w_a)` (`u_a < u_b`) along `(u_a, w_b)`.
                let u0 = above[0];
                let u1 = above[1];
                let u2 = above[2];
                let d0 = below[0];
                let iu0 = get_or_insert_edge(u0, u0);
                let iu1 = get_or_insert_edge(u1, u1);
                let iu2 = get_or_insert_edge(u2, u2);
                let iw0 = get_or_insert_edge(u0, d0);
                let iw1 = get_or_insert_edge(u1, d0);
                let iw2 = get_or_insert_edge(u2, d0);
                for sub_tet in [
                    [iu0, iu1, iu2, iw2],
                    [iu0, iu1, iw2, iw1],
                    [iu0, iw1, iw2, iw0],
                ] {
                    local_tets.push(sub_tet);
                    local_tet_source_cells.push(src_cell);
                    local_tet_indices.push(global_tet_idx);
                }
            }
            _ => unreachable!(),
        }
    }

    ClipChunkOutput {
        local_edges,
        local_tets,
        local_tet_source_cells,
        local_tet_indices,
    }
}

fn process_crinkle_clip_chunk(
    tet_offset: u32,
    tets: &[[u32; 4]],
    tet_source_cells: &[u32],
    keep_source_cell: &[bool],
) -> ClipChunkOutput {
    let mut local_edge_map: HashMap<EdgeKey, u32> = HashMap::new();
    let mut local_edges: Vec<(EdgeKey, f64)> = Vec::new();
    let mut local_tets: Vec<[u32; 4]> = Vec::new();
    let mut local_tet_source_cells: Vec<u32> = Vec::new();
    let mut local_tet_indices: Vec<u32> = Vec::new();

    let mut get_or_insert_vertex = |v: u32| -> u32 {
        let key = EdgeKey(v, v);
        if let Some(&lid) = local_edge_map.get(&key) {
            lid
        } else {
            let lid = local_edges.len() as u32;
            local_edges.push((key, 0.0));
            local_edge_map.insert(key, lid);
            lid
        }
    };

    for (local_tet_idx, &[v0, v1, v2, v3]) in tets.iter().enumerate() {
        let src_cell = tet_source_cells[local_tet_idx];
        if !keep_source_cell[src_cell as usize] {
            continue;
        }
        let global_tet_idx = tet_offset + local_tet_idx as u32;
        let i0 = get_or_insert_vertex(v0);
        let i1 = get_or_insert_vertex(v1);
        let i2 = get_or_insert_vertex(v2);
        let i3 = get_or_insert_vertex(v3);
        local_tets.push([i0, i1, i2, i3]);
        local_tet_source_cells.push(src_cell);
        local_tet_indices.push(global_tet_idx);
    }

    ClipChunkOutput {
        local_edges,
        local_tets,
        local_tet_source_cells,
        local_tet_indices,
    }
}

fn run_exact_clip_chunks(
    tets: &[[u32; 4]],
    tet_source_cells: &[u32],
    scalars: &[f64],
    iso: f64,
    iso_tol: f64,
    chunk_size: usize,
    num_chunks: usize,
) -> Vec<ClipChunkOutput> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if num_chunks > 1 {
            let max_threads = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
                .min(num_chunks);
            if max_threads > 1 {
                let mut results: Vec<Option<ClipChunkOutput>> =
                    (0..num_chunks).map(|_| None).collect();
                let chunks_per_worker = num_chunks.div_ceil(max_threads);
                std::thread::scope(|s| {
                    for (worker_idx, out_slice) in results.chunks_mut(chunks_per_worker).enumerate()
                    {
                        let start_chunk = worker_idx * chunks_per_worker;
                        s.spawn(move || {
                            for (offset, slot) in out_slice.iter_mut().enumerate() {
                                let chunk_idx = start_chunk + offset;
                                let start = chunk_idx * chunk_size;
                                let end = (start + chunk_size).min(tets.len());
                                *slot = Some(process_exact_clip_chunk(
                                    start as u32,
                                    &tets[start..end],
                                    &tet_source_cells[start..end],
                                    scalars,
                                    iso,
                                    iso_tol,
                                ));
                            }
                        });
                    }
                });
                return results
                    .into_iter()
                    .map(|opt| opt.expect("chunk completed"))
                    .collect();
            }
        }
    }

    (0..num_chunks)
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(tets.len());
            process_exact_clip_chunk(
                start as u32,
                &tets[start..end],
                &tet_source_cells[start..end],
                scalars,
                iso,
                iso_tol,
            )
        })
        .collect()
}

fn run_crinkle_clip_chunks(
    tets: &[[u32; 4]],
    tet_source_cells: &[u32],
    keep_source_cell: &[bool],
    chunk_size: usize,
    num_chunks: usize,
) -> Vec<ClipChunkOutput> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if num_chunks > 1 {
            let max_threads = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
                .min(num_chunks);
            if max_threads > 1 {
                let mut results: Vec<Option<ClipChunkOutput>> =
                    (0..num_chunks).map(|_| None).collect();
                let chunks_per_worker = num_chunks.div_ceil(max_threads);
                std::thread::scope(|s| {
                    for (worker_idx, out_slice) in results.chunks_mut(chunks_per_worker).enumerate()
                    {
                        let start_chunk = worker_idx * chunks_per_worker;
                        s.spawn(move || {
                            for (offset, slot) in out_slice.iter_mut().enumerate() {
                                let chunk_idx = start_chunk + offset;
                                let start = chunk_idx * chunk_size;
                                let end = (start + chunk_size).min(tets.len());
                                *slot = Some(process_crinkle_clip_chunk(
                                    start as u32,
                                    &tets[start..end],
                                    &tet_source_cells[start..end],
                                    keep_source_cell,
                                ));
                            }
                        });
                    }
                });
                return results
                    .into_iter()
                    .map(|opt| opt.expect("chunk completed"))
                    .collect();
            }
        }
    }

    (0..num_chunks)
        .map(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(tets.len());
            process_crinkle_clip_chunk(
                start as u32,
                &tets[start..end],
                &tet_source_cells[start..end],
                keep_source_cell,
            )
        })
        .collect()
}

fn interpolate_edge_array(
    source: &DataArray,
    edges: &[(EdgeKey, f64)],
) -> Result<DataArray, CoreError> {
    let comp: ComponentCount = source.components();
    let ncomp = comp.as_usize();
    let out_len = edges.len() * ncomp;

    match source.buffer() {
        ArrayBuffer::F32(buf) => {
            let mut out = Vec::with_capacity(out_len);
            for &(EdgeKey(i0, i1), t) in edges {
                let s0 = (i0 as usize) * ncomp;
                if i0 == i1 || t == 0.0 {
                    out.extend_from_slice(&buf[s0..s0 + ncomp]);
                } else if t == 1.0 {
                    let s1 = (i1 as usize) * ncomp;
                    out.extend_from_slice(&buf[s1..s1 + ncomp]);
                } else {
                    let s1 = (i1 as usize) * ncomp;
                    let omt = 1.0 - t;
                    for c in 0..ncomp {
                        let v0 = f64::from(buf[s0 + c]);
                        let v1 = f64::from(buf[s1 + c]);
                        out.push((omt * v0 + t * v1) as f32);
                    }
                }
            }
            DataArray::from_f32(comp, out)
        }
        ArrayBuffer::F64(buf) => {
            let mut out = Vec::with_capacity(out_len);
            for &(EdgeKey(i0, i1), t) in edges {
                let s0 = (i0 as usize) * ncomp;
                if i0 == i1 || t == 0.0 {
                    out.extend_from_slice(&buf[s0..s0 + ncomp]);
                } else if t == 1.0 {
                    let s1 = (i1 as usize) * ncomp;
                    out.extend_from_slice(&buf[s1..s1 + ncomp]);
                } else {
                    let s1 = (i1 as usize) * ncomp;
                    let omt = 1.0 - t;
                    for c in 0..ncomp {
                        let v0 = buf[s0 + c];
                        let v1 = buf[s1 + c];
                        out.push(omt * v0 + t * v1);
                    }
                }
            }
            DataArray::from_f64(comp, out)
        }
        ArrayBuffer::I32(buf) => {
            let mut out = Vec::with_capacity(out_len);
            for &(EdgeKey(i0, i1), t) in edges {
                let s0 = (i0 as usize) * ncomp;
                if i0 == i1 || t == 0.0 {
                    out.extend_from_slice(&buf[s0..s0 + ncomp]);
                } else if t == 1.0 {
                    let s1 = (i1 as usize) * ncomp;
                    out.extend_from_slice(&buf[s1..s1 + ncomp]);
                } else {
                    let s1 = (i1 as usize) * ncomp;
                    let omt = 1.0 - t;
                    for c in 0..ncomp {
                        let v0 = f64::from(buf[s0 + c]);
                        let v1 = f64::from(buf[s1 + c]);
                        out.push((omt * v0 + t * v1).round() as i32);
                    }
                }
            }
            DataArray::from_i32(comp, out)
        }
        ArrayBuffer::I64(buf) => {
            let mut out = Vec::with_capacity(out_len);
            for &(EdgeKey(i0, i1), t) in edges {
                let s0 = (i0 as usize) * ncomp;
                if i0 == i1 || t == 0.0 {
                    out.extend_from_slice(&buf[s0..s0 + ncomp]);
                } else if t == 1.0 {
                    let s1 = (i1 as usize) * ncomp;
                    out.extend_from_slice(&buf[s1..s1 + ncomp]);
                } else {
                    let s1 = (i1 as usize) * ncomp;
                    let omt = 1.0 - t;
                    for c in 0..ncomp {
                        let v0 = buf[s0 + c] as f64;
                        let v1 = buf[s1 + c] as f64;
                        out.push((omt * v0 + t * v1).round() as i64);
                    }
                }
            }
            DataArray::from_i64(comp, out)
        }
        ArrayBuffer::U8(buf) => {
            let mut out = Vec::with_capacity(out_len);
            for &(EdgeKey(i0, i1), t) in edges {
                let s0 = (i0 as usize) * ncomp;
                if i0 == i1 || t == 0.0 {
                    out.extend_from_slice(&buf[s0..s0 + ncomp]);
                } else if t == 1.0 {
                    let s1 = (i1 as usize) * ncomp;
                    out.extend_from_slice(&buf[s1..s1 + ncomp]);
                } else {
                    let s1 = (i1 as usize) * ncomp;
                    let omt = 1.0 - t;
                    for c in 0..ncomp {
                        let v0 = f64::from(buf[s0 + c]);
                        let v1 = f64::from(buf[s1 + c]);
                        out.push((omt * v0 + t * v1).round().clamp(0.0, 255.0) as u8);
                    }
                }
            }
            DataArray::from_u8(comp, out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_fourteen_active_marching_tet_cases_orient_normals_toward_positive_gradient() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let view = TetView::new(
            points.clone(),
            vec![[0, 1, 2, 3]],
            vec![0],
            Attributes::new(4),
            Attributes::new(1),
        )
        .unwrap();

        for mask in 1u8..15u8 {
            let scalars = [
                if (mask & 1) != 0 { 1.0 } else { -1.0 },
                if (mask & 2) != 0 { 1.0 } else { -1.0 },
                if (mask & 4) != 0 { 1.0 } else { -1.0 },
                if (mask & 8) != 0 { 1.0 } else { -1.0 },
            ];
            // Gradient of affine scalar field s(x, y, z) = s0 + (s1 - s0)x + (s2 - s0)y + (s3 - s0)z
            let grad = [
                scalars[1] - scalars[0],
                scalars[2] - scalars[0],
                scalars[3] - scalars[0],
            ];
            let surf = extract_isosurface(&view, &scalars, 0.0).unwrap();
            assert!(
                surf.num_triangles() == 1 || surf.num_triangles() == 2,
                "mask {mask:#06b} produced {} triangles",
                surf.num_triangles()
            );
            let pts = surf.points();
            for &[i0, i1, i2] in surf.triangles() {
                let p0 = pts[i0 as usize];
                let p1 = pts[i1 as usize];
                let p2 = pts[i2 as usize];
                let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
                let n = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                let dot = n[0] * grad[0] + n[1] * grad[1] + n[2] * grad[2];
                assert!(
                    dot > 1e-6,
                    "mask {mask:#06b} triangle normal {n:?} does not point along grad {grad:?} (dot = {dot})"
                );
            }
        }
    }
}

//! Bounding volume hierarchy (BVH) cell locator over linearized tetrahedra (`M4.1`).
//!
//! [`TetLocator`] accelerates 3D point-in-tetrahedron queries (`O(log N)` average time, or
//! `O(1)` when stepping through adjacent cells with [`TetLocator::locate_with_hint`]) on a
//! [`TetView`] (`D-005`). Downstream flow and volume filters (`streamlines`, `glyphs`, and
//! regular-grid `resampling`) use [`TetLocator`] and [`TetLocation`] to locate query points,
//! track original source cell provenance, and interpolate point or cell [`DataArray`]s in
//! `f64` precision (`D-014`).

use std::sync::Arc;

use tetraglass_core::{BoundingBox, ComponentCount, DataArray, TetView};

/// Default barycentric coordinate tolerance for boundary point-in-tetrahedron tests.
///
/// A query point with barycentric weights `[w0, w1, w2, w3]` is considered inside a
/// tetrahedron when `min(w0, w1, w2, w3) >= -tolerance`.
pub const DEFAULT_LOCATOR_TOLERANCE: f64 = 1e-9;

/// Default maximum number of tetrahedra stored in a single BVH leaf node.
pub const DEFAULT_MAX_LEAF_TETS: usize = 8;

/// Default batch chunk size for parallel [`TetLocator::locate_many`] queries.
pub const DEFAULT_LOCATOR_CHUNK_SIZE: usize = 4096;

/// Minimum primitive count before splitting BVH construction across threads on native targets.
#[cfg(not(target_arch = "wasm32"))]
const PARALLEL_BUILD_THRESHOLD: usize = 16_384;

/// Strictly interior barycentric threshold above which a non-overlapping tetrahedron uniquely
/// contains the query point without ambiguity on shared faces, edges, or vertices.
const STRICT_INTERIOR_EPS: f64 = 1e-12;

/// Tie-breaking threshold when comparing two boundary candidate tetrahedra.
const BOUNDARY_TIE_EPS: f64 = 1e-14;

/// Computes the 4 barycentric coordinates `[w0, w1, w2, w3]` of `point` with respect to a
/// tetrahedron with vertices `p0`, `p1`, `p2`, `p3`.
///
/// Returns `None` if any vertex or query coordinate is non-finite, or if the tetrahedron is
/// degenerate (zero or near-zero volume).
///
/// The returned weights satisfy `w0 + w1 + w2 + w3 = 1.0` and
/// `w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3 = point`. A point lies inside or on the boundary of
/// the tetrahedron when every `w_i >= 0.0` (up to floating-point tolerance).
///
/// # Examples
///
/// ```
/// use tetraglass_filters::tet_barycentric_coordinates;
///
/// let p0 = [0.0, 0.0, 0.0];
/// let p1 = [1.0, 0.0, 0.0];
/// let p2 = [0.0, 1.0, 0.0];
/// let p3 = [0.0, 0.0, 1.0];
///
/// let bary = tet_barycentric_coordinates(p0, p1, p2, p3, [0.25, 0.25, 0.25]).unwrap();
/// for w in bary {
///     assert!((w - 0.25).abs() < 1e-15);
/// }
/// ```
#[must_use]
pub fn tet_barycentric_coordinates(
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
    point: [f64; 3],
) -> Option<[f64; 4]> {
    if !is_finite_vec3(p0)
        || !is_finite_vec3(p1)
        || !is_finite_vec3(p2)
        || !is_finite_vec3(p3)
        || !is_finite_vec3(point)
    {
        return None;
    }

    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let e3 = [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]];

    let e1_sq = dot3(e1, e1);
    let e2_sq = dot3(e2, e2);
    let e3_sq = dot3(e3, e3);
    let scale_sq = e1_sq.max(e2_sq).max(e3_sq);
    if scale_sq == 0.0 || !scale_sq.is_finite() {
        return None;
    }

    let c23 = cross3(e2, e3);
    let det = dot3(e1, c23);
    if !det.is_finite() || det * det <= 1e-60 * scale_sq * scale_sq * scale_sq {
        return None;
    }

    let c31 = cross3(e3, e1);
    let c12 = cross3(e1, e2);
    let d = [point[0] - p0[0], point[1] - p0[1], point[2] - p0[2]];

    let inv_det = 1.0 / det;
    let w1 = dot3(c23, d) * inv_det;
    let w2 = dot3(c31, d) * inv_det;
    let w3 = dot3(c12, d) * inv_det;
    let w0 = 1.0 - w1 - w2 - w3;

    if w0.is_finite() && w1.is_finite() && w2.is_finite() && w3.is_finite() {
        Some([w0, w1, w2, w3])
    } else {
        None
    }
}

/// Result of locating a 3D point inside a [`TetView`] via [`TetLocator`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetLocation {
    /// Index of the containing tetrahedron in [`TetView::tets`].
    pub tet_index: usize,
    /// Originating source cell index in the input `UnstructuredGrid` (`D-005`).
    pub source_cell_id: u32,
    /// Barycentric weights `[w0, w1, w2, w3]` corresponding to the four vertices of
    /// `view.tets()[tet_index]`, summing to `1.0`.
    pub barycentric: [f64; 4],
}

impl TetLocation {
    /// Returns the minimum of the four barycentric weights (`min(w0, w1, w2, w3)`).
    #[must_use]
    #[inline]
    pub fn min_barycentric(&self) -> f64 {
        min4(self.barycentric)
    }

    /// Returns barycentric weights clamped to `[0.0, 1.0]` and normalized to sum to `1.0`.
    #[must_use]
    pub fn clamped_barycentric(&self) -> [f64; 4] {
        let c0 = self.barycentric[0].max(0.0);
        let c1 = self.barycentric[1].max(0.0);
        let c2 = self.barycentric[2].max(0.0);
        let c3 = self.barycentric[3].max(0.0);
        let sum = c0 + c1 + c2 + c3;
        if sum > 0.0 {
            let inv = 1.0 / sum;
            [c0 * inv, c1 * inv, c2 * inv, c3 * inv]
        } else {
            [0.25, 0.25, 0.25, 0.25]
        }
    }

    /// Reconstructs the 3D world position from the tetrahedron's vertices and `self.barycentric`.
    #[must_use]
    pub fn interpolate_position(&self, view: &TetView) -> Option<[f64; 3]> {
        let &[v0, v1, v2, v3] = view.tets().get(self.tet_index)?;
        let pts = view.points();
        let p0 = *pts.get(v0 as usize)?;
        let p1 = *pts.get(v1 as usize)?;
        let p2 = *pts.get(v2 as usize)?;
        let p3 = *pts.get(v3 as usize)?;
        let [w0, w1, w2, w3] = self.barycentric;
        Some([
            w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
            w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1],
            w0 * p0[2] + w1 * p1[2] + w2 * p2[2] + w3 * p3[2],
        ])
    }

    /// Interpolates a single component (`component`) of a point [`DataArray`] at this location.
    #[must_use]
    pub fn interpolate_point_scalar(
        &self,
        view: &TetView,
        array: &DataArray,
        component: usize,
    ) -> Option<f64> {
        let &[v0, v1, v2, v3] = view.tets().get(self.tet_index)?;
        let s0 = array.value_as_f64(v0 as usize, component)?;
        let s1 = array.value_as_f64(v1 as usize, component)?;
        let s2 = array.value_as_f64(v2 as usize, component)?;
        let s3 = array.value_as_f64(v3 as usize, component)?;
        let [w0, w1, w2, w3] = self.barycentric;
        Some(w0 * s0 + w1 * s1 + w2 * s2 + w3 * s3)
    }

    /// Interpolates a 3-component (`ComponentCount::Vector3`) point [`DataArray`] at this location.
    #[must_use]
    pub fn interpolate_point_vector3(&self, view: &TetView, array: &DataArray) -> Option<[f64; 3]> {
        if array.components() != ComponentCount::Vector3 {
            return None;
        }
        let &[v0, v1, v2, v3] = view.tets().get(self.tet_index)?;
        let [w0, w1, w2, w3] = self.barycentric;
        let mut out = [0.0; 3];
        for (c, slot) in out.iter_mut().enumerate() {
            let a0 = array.value_as_f64(v0 as usize, c)?;
            let a1 = array.value_as_f64(v1 as usize, c)?;
            let a2 = array.value_as_f64(v2 as usize, c)?;
            let a3 = array.value_as_f64(v3 as usize, c)?;
            *slot = w0 * a0 + w1 * a1 + w2 * a2 + w3 * a3;
        }
        Some(out)
    }

    /// Interpolates all components of a point [`DataArray`] at this location and returns the
    /// Euclidean magnitude $\|\sum_{k=0}^3 w_k \mathbf{v}_k\|_2$.
    #[must_use]
    pub fn interpolate_point_magnitude(&self, view: &TetView, array: &DataArray) -> Option<f64> {
        let &[v0, v1, v2, v3] = view.tets().get(self.tet_index)?;
        let ncomp = array.components().as_usize();
        let [w0, w1, w2, w3] = self.barycentric;
        let mut sum_sq = 0.0;
        for c in 0..ncomp {
            let a0 = array.value_as_f64(v0 as usize, c)?;
            let a1 = array.value_as_f64(v1 as usize, c)?;
            let a2 = array.value_as_f64(v2 as usize, c)?;
            let a3 = array.value_as_f64(v3 as usize, c)?;
            let val = w0 * a0 + w1 * a1 + w2 * a2 + w3 * a3;
            sum_sq += val * val;
        }
        Some(sum_sq.sqrt())
    }

    /// Interpolates all components of a point [`DataArray`] tuple into `out`.
    ///
    /// Returns `false` if `out.len() != array.components().as_usize()` or if `self.tet_index` is
    /// out of bounds for `view`.
    pub fn interpolate_point_tuple(
        &self,
        view: &TetView,
        array: &DataArray,
        out: &mut [f64],
    ) -> bool {
        let ncomp = array.components().as_usize();
        if out.len() != ncomp {
            return false;
        }
        let Some(&[v0, v1, v2, v3]) = view.tets().get(self.tet_index) else {
            return false;
        };
        let [w0, w1, w2, w3] = self.barycentric;
        for (c, slot) in out.iter_mut().enumerate() {
            let (Some(a0), Some(a1), Some(a2), Some(a3)) = (
                array.value_as_f64(v0 as usize, c),
                array.value_as_f64(v1 as usize, c),
                array.value_as_f64(v2 as usize, c),
                array.value_as_f64(v3 as usize, c),
            ) else {
                return false;
            };
            *slot = w0 * a0 + w1 * a1 + w2 * a2 + w3 * a3;
        }
        true
    }

    /// Reads a per-tetrahedron cell [`DataArray`] component at `self.tet_index`.
    #[must_use]
    pub fn sample_cell_scalar(&self, array: &DataArray, component: usize) -> Option<f64> {
        array.value_as_f64(self.tet_index, component)
    }

    /// Reads a 3-component (`ComponentCount::Vector3`) per-tetrahedron cell [`DataArray`] at
    /// `self.tet_index`.
    #[must_use]
    pub fn sample_cell_vector3(&self, array: &DataArray) -> Option<[f64; 3]> {
        if array.components() != ComponentCount::Vector3 {
            return None;
        }
        Some([
            array.value_as_f64(self.tet_index, 0)?,
            array.value_as_f64(self.tet_index, 1)?,
            array.value_as_f64(self.tet_index, 2)?,
        ])
    }
}

/// A node in the flat bounding volume hierarchy (BVH) of a [`TetLocator`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BvhNode {
    /// Axis-aligned minimum corner `[xmin, ymin, zmin]` enclosing all tetrahedra in this subtree.
    pub min: [f64; 3],
    /// Axis-aligned maximum corner `[xmax, ymax, zmax]` enclosing all tetrahedra in this subtree.
    pub max: [f64; 3],
    /// For an internal node (`!is_leaf`), index of the left child node in [`TetLocator::nodes`].
    /// For a leaf node (`is_leaf`), starting offset in [`TetLocator::leaf_tets`].
    pub left_or_first: u32,
    /// For an internal node (`!is_leaf`), index of the right child node in [`TetLocator::nodes`].
    /// For a leaf node (`is_leaf`), number of tetrahedra in this leaf (`>= 1`).
    pub right_or_count: u32,
    /// `true` if this node is a leaf node referencing [`TetLocator::leaf_tets`].
    pub is_leaf: bool,
}

#[derive(Debug, Clone, Copy)]
struct TetPrimitive {
    tet_index: u32,
    min: [f64; 3],
    max: [f64; 3],
    centroid: [f64; 3],
}

/// Bounding volume hierarchy (BVH) over the tetrahedra of a [`TetView`] for fast 3D point location.
///
/// # Examples
///
/// ```
/// use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
/// use tetraglass_filters::{TetLocator, linearize_default};
///
/// let grid = SyntheticGridSpec::new(4, 4, 4, SyntheticCellLayout::Hexahedra).generate()?;
/// let view = linearize_default(&grid)?;
/// let locator = TetLocator::new(&view);
///
/// let hit = locator.locate([0.3, 0.4, 0.5]).expect("point inside [0, 1]^3");
/// let pos = hit.interpolate_position(&view).unwrap();
/// assert!((pos[0] - 0.3).abs() < 1e-12);
/// assert!((pos[1] - 0.4).abs() < 1e-12);
/// assert!((pos[2] - 0.5).abs() < 1e-12);
///
/// // Linear scalar field `1*x + 2*y + 3*z` is reproduced to machine precision:
/// let linear_arr = view.point_data().get("linear").unwrap();
/// let val = hit.interpolate_point_scalar(&view, linear_arr, 0).unwrap();
/// assert!((val - (0.3 + 2.0 * 0.4 + 3.0 * 0.5)).abs() < 1e-12);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct TetLocator {
    points: Arc<[[f64; 3]]>,
    tets: Arc<[[u32; 4]]>,
    tet_source_cells: Arc<[u32]>,
    nodes: Vec<BvhNode>,
    leaf_tets: Vec<u32>,
    tolerance: f64,
}

impl TetLocator {
    /// Builds a [`TetLocator`] over `view` using [`DEFAULT_MAX_LEAF_TETS`] (`8`) and
    /// [`DEFAULT_LOCATOR_TOLERANCE`] (`1e-9`).
    #[must_use]
    pub fn new(view: &TetView) -> Self {
        Self::with_max_leaf_tets(view, DEFAULT_MAX_LEAF_TETS)
    }

    /// Builds a [`TetLocator`] over `view` with a custom maximum number of tetrahedra per leaf
    /// (`max_leaf_tets` is clamped to `>= 1`).
    #[must_use]
    pub fn with_max_leaf_tets(view: &TetView, max_leaf_tets: usize) -> Self {
        let max_leaf_tets = max_leaf_tets.max(1);
        let points = view.points_arc();
        let tets = view.tets_arc();
        let tet_source_cells = view.tet_source_cells_arc();
        let num_tets = tets.len();

        if num_tets == 0 {
            return Self {
                points,
                tets,
                tet_source_cells,
                nodes: Vec::new(),
                leaf_tets: Vec::new(),
                tolerance: DEFAULT_LOCATOR_TOLERANCE,
            };
        }

        let mut prims = Vec::with_capacity(num_tets);
        for (tet_idx, &[v0, v1, v2, v3]) in tets.iter().enumerate() {
            let p0 = points[v0 as usize];
            let p1 = points[v1 as usize];
            let p2 = points[v2 as usize];
            let p3 = points[v3 as usize];
            let min = [
                p0[0].min(p1[0]).min(p2[0]).min(p3[0]),
                p0[1].min(p1[1]).min(p2[1]).min(p3[1]),
                p0[2].min(p1[2]).min(p2[2]).min(p3[2]),
            ];
            let max = [
                p0[0].max(p1[0]).max(p2[0]).max(p3[0]),
                p0[1].max(p1[1]).max(p2[1]).max(p3[1]),
                p0[2].max(p1[2]).max(p2[2]).max(p3[2]),
            ];
            let centroid = [
                0.25 * (p0[0] + p1[0] + p2[0] + p3[0]),
                0.25 * (p0[1] + p1[1] + p2[1] + p3[1]),
                0.25 * (p0[2] + p1[2] + p2[2] + p3[2]),
            ];
            prims.push(TetPrimitive {
                tet_index: tet_idx as u32,
                min,
                max,
                centroid,
            });
        }

        let total_nodes = subtree_node_count(num_tets, max_leaf_tets);
        let mut nodes = vec![BvhNode::default(); total_nodes];
        build_bvh_subtree(0, 0, max_leaf_tets, &mut nodes, &mut prims);

        let leaf_tets = prims.into_iter().map(|p| p.tet_index).collect();

        Self {
            points,
            tets,
            tet_source_cells,
            nodes,
            leaf_tets,
            tolerance: DEFAULT_LOCATOR_TOLERANCE,
        }
    }

    /// Sets the barycentric coordinate tolerance (`>= 0.0`) used for boundary containment checks.
    #[must_use]
    pub const fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = if tolerance >= 0.0 && tolerance.is_finite() {
            tolerance
        } else {
            0.0
        };
        self
    }

    /// Returns the configured barycentric coordinate tolerance.
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Returns the number of tetrahedra indexed by this locator.
    #[must_use]
    pub fn num_tets(&self) -> usize {
        self.tets.len()
    }

    /// Returns the number of BVH nodes.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the flat slice of BVH nodes (`nodes()[0]` is the root when non-empty).
    #[must_use]
    pub fn nodes(&self) -> &[BvhNode] {
        &self.nodes
    }

    /// Returns the flat leaf tetrahedron index array referenced by leaf [`BvhNode`]s.
    #[must_use]
    pub fn leaf_tets(&self) -> &[u32] {
        &self.leaf_tets
    }

    /// Returns the root axis-aligned [`BoundingBox`] of all tetrahedra, or `None` if empty.
    #[must_use]
    pub fn bounds(&self) -> Option<BoundingBox> {
        let root = self.nodes.first()?;
        Some(BoundingBox {
            min: root.min,
            max: root.max,
        })
    }

    /// Computes the barycentric coordinates `[w0, w1, w2, w3]` of `point` in tetrahedron
    /// `tet_index`, or `None` if `tet_index` is out of bounds or the tetrahedron is degenerate.
    #[must_use]
    pub fn tet_barycentric(&self, tet_index: usize, point: [f64; 3]) -> Option<[f64; 4]> {
        let &[v0, v1, v2, v3] = self.tets.get(tet_index)?;
        tet_barycentric_coordinates(
            self.points[v0 as usize],
            self.points[v1 as usize],
            self.points[v2 as usize],
            self.points[v3 as usize],
            point,
        )
    }

    /// Locates the tetrahedron containing `point` within `self.tolerance()`.
    #[must_use]
    pub fn locate(&self, point: [f64; 3]) -> Option<TetLocation> {
        self.locate_with_hint(point, None)
    }

    /// Locates the tetrahedron containing `point`, testing `hint_tet_index` in $O(1)$ before
    /// falling back to BVH traversal.
    ///
    /// Streamline integrators pass the previous step's `tet_index` as `hint_tet_index` so
    /// substeps that remain inside the same tetrahedron bypass the BVH completely.
    #[must_use]
    pub fn locate_with_hint(
        &self,
        point: [f64; 3],
        hint_tet_index: Option<usize>,
    ) -> Option<TetLocation> {
        if self.nodes.is_empty() || !is_finite_vec3(point) {
            return None;
        }

        let tol = self.tolerance;
        let interior_eps = STRICT_INTERIOR_EPS.min(tol * 0.01);
        let mut best: Option<TetLocation> = None;

        if let Some(hint_idx) = hint_tet_index
            && let Some(bary) = self.tet_barycentric(hint_idx, point)
        {
            let min_w = min4(bary);
            if min_w >= interior_eps {
                return Some(TetLocation {
                    tet_index: hint_idx,
                    source_cell_id: self.tet_source_cells[hint_idx],
                    barycentric: bary,
                });
            }
            if min_w >= -tol {
                best = Some(TetLocation {
                    tet_index: hint_idx,
                    source_cell_id: self.tet_source_cells[hint_idx],
                    barycentric: bary,
                });
            }
        }

        // A balanced median-split BVH over <= 2^32 primitives has depth <= 32, so a 64-entry
        // stack is sufficient for any valid TetView (`D-013`).
        let mut stack = [0u32; 64];
        let mut stack_len = 1usize;

        while stack_len > 0 {
            stack_len -= 1;
            let node_idx = stack[stack_len] as usize;
            let node = &self.nodes[node_idx];

            if !point_in_expanded_aabb(point, node.min, node.max, tol) {
                continue;
            }

            if node.is_leaf {
                let start = node.left_or_first as usize;
                let end = start + node.right_or_count as usize;
                for &tet_u32 in &self.leaf_tets[start..end] {
                    let tet_idx = tet_u32 as usize;
                    let Some(bary) = self.tet_barycentric(tet_idx, point) else {
                        continue;
                    };
                    let min_w = min4(bary);
                    if min_w < -tol {
                        continue;
                    }
                    if min_w >= interior_eps {
                        return Some(TetLocation {
                            tet_index: tet_idx,
                            source_cell_id: self.tet_source_cells[tet_idx],
                            barycentric: bary,
                        });
                    }
                    let candidate = TetLocation {
                        tet_index: tet_idx,
                        source_cell_id: self.tet_source_cells[tet_idx],
                        barycentric: bary,
                    };
                    best = Some(match best {
                        None => candidate,
                        Some(prev) => select_better_boundary_candidate(prev, candidate),
                    });
                }
            } else {
                // Push right child first so left child is popped first (pre-order traversal).
                stack[stack_len] = node.right_or_count;
                stack[stack_len + 1] = node.left_or_first;
                stack_len += 2;
            }
        }

        best
    }

    /// Locates a batch of query `points` in deterministic order using
    /// [`DEFAULT_LOCATOR_CHUNK_SIZE`].
    #[must_use]
    pub fn locate_many(&self, points: &[[f64; 3]]) -> Vec<Option<TetLocation>> {
        self.locate_many_with_chunk_size(points, DEFAULT_LOCATOR_CHUNK_SIZE)
    }

    /// Locates a batch of query `points` in deterministic chunk order, executing chunks in
    /// parallel on native targets when `points.len() > chunk_size`.
    #[must_use]
    pub fn locate_many_with_chunk_size(
        &self,
        points: &[[f64; 3]],
        chunk_size: usize,
    ) -> Vec<Option<TetLocation>> {
        let chunk_size = chunk_size.max(1);
        if points.is_empty() {
            return Vec::new();
        }

        #[cfg(not(target_arch = "wasm32"))]
        if points.len() > chunk_size {
            let mut results = vec![None; points.len()];
            std::thread::scope(|s| {
                let mut handles = Vec::new();
                for (pt_chunk, out_chunk) in points
                    .chunks(chunk_size)
                    .zip(results.chunks_mut(chunk_size))
                {
                    handles.push(s.spawn(move || {
                        for (pt, slot) in pt_chunk.iter().zip(out_chunk.iter_mut()) {
                            *slot = self.locate(*pt);
                        }
                    }));
                }
                for handle in handles {
                    let _ = handle.join();
                }
            });
            return results;
        }

        points.iter().map(|&pt| self.locate(pt)).collect()
    }
}

#[inline]
const fn subtree_node_count(count: usize, max_leaf_tets: usize) -> usize {
    if count <= max_leaf_tets {
        1
    } else {
        let mid = count / 2;
        1 + subtree_node_count(mid, max_leaf_tets) + subtree_node_count(count - mid, max_leaf_tets)
    }
}

fn build_bvh_subtree(
    node_base_idx: u32,
    prim_base_idx: u32,
    max_leaf_tets: usize,
    nodes_slice: &mut [BvhNode],
    prims_slice: &mut [TetPrimitive],
) {
    let mut bounds_min = [f64::INFINITY; 3];
    let mut bounds_max = [f64::NEG_INFINITY; 3];
    let mut centroid_min = [f64::INFINITY; 3];
    let mut centroid_max = [f64::NEG_INFINITY; 3];

    for prim in prims_slice.iter() {
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(prim.min[axis]);
            bounds_max[axis] = bounds_max[axis].max(prim.max[axis]);
            centroid_min[axis] = centroid_min[axis].min(prim.centroid[axis]);
            centroid_max[axis] = centroid_max[axis].max(prim.centroid[axis]);
        }
    }

    let count = prims_slice.len();
    if count <= max_leaf_tets {
        // Canonicalize leaf tetrahedron order by ascending `tet_index` so the BVH is
        // bit-for-bit identical regardless of intermediate `select_nth_unstable_by` permutations.
        prims_slice.sort_unstable_by_key(|p| p.tet_index);
        nodes_slice[0] = BvhNode {
            min: bounds_min,
            max: bounds_max,
            left_or_first: prim_base_idx,
            right_or_count: count as u32,
            is_leaf: true,
        };
        return;
    }

    // Choose split axis with the largest centroid extent (ties break toward X, then Y, then Z).
    let ext_x = centroid_max[0] - centroid_min[0];
    let ext_y = centroid_max[1] - centroid_min[1];
    let ext_z = centroid_max[2] - centroid_min[2];
    let split_axis = if ext_x >= ext_y && ext_x >= ext_z {
        0
    } else if ext_y >= ext_z {
        1
    } else {
        2
    };

    let mid = count / 2;
    prims_slice.select_nth_unstable_by(mid, |a, b| {
        a.centroid[split_axis]
            .total_cmp(&b.centroid[split_axis])
            .then_with(|| a.tet_index.cmp(&b.tet_index))
    });

    let left_nodes = subtree_node_count(mid, max_leaf_tets);
    let left_node_idx = node_base_idx + 1;
    let right_node_idx = node_base_idx + 1 + left_nodes as u32;

    nodes_slice[0] = BvhNode {
        min: bounds_min,
        max: bounds_max,
        left_or_first: left_node_idx,
        right_or_count: right_node_idx,
        is_leaf: false,
    };

    let (_, children_nodes) = nodes_slice.split_at_mut(1);
    let (left_nodes_slice, right_nodes_slice) = children_nodes.split_at_mut(left_nodes);
    let (left_prims, right_prims) = prims_slice.split_at_mut(mid);
    let right_prim_base = prim_base_idx + mid as u32;

    #[cfg(not(target_arch = "wasm32"))]
    if count >= PARALLEL_BUILD_THRESHOLD {
        std::thread::scope(|s| {
            let left_handle = s.spawn(move || {
                build_bvh_subtree(
                    left_node_idx,
                    prim_base_idx,
                    max_leaf_tets,
                    left_nodes_slice,
                    left_prims,
                );
            });
            build_bvh_subtree(
                right_node_idx,
                right_prim_base,
                max_leaf_tets,
                right_nodes_slice,
                right_prims,
            );
            let _ = left_handle.join();
        });
        return;
    }

    build_bvh_subtree(
        left_node_idx,
        prim_base_idx,
        max_leaf_tets,
        left_nodes_slice,
        left_prims,
    );
    build_bvh_subtree(
        right_node_idx,
        right_prim_base,
        max_leaf_tets,
        right_nodes_slice,
        right_prims,
    );
}

#[inline]
fn select_better_boundary_candidate(a: TetLocation, b: TetLocation) -> TetLocation {
    let min_a = a.min_barycentric();
    let min_b = b.min_barycentric();
    if (min_a - min_b).abs() <= BOUNDARY_TIE_EPS {
        if a.tet_index <= b.tet_index { a } else { b }
    } else if min_b > min_a {
        b
    } else {
        a
    }
}

#[inline]
fn point_in_expanded_aabb(point: [f64; 3], min: [f64; 3], max: [f64; 3], tol: f64) -> bool {
    let span = ((max[0] - min[0]) + (max[1] - min[1]) + (max[2] - min[2])).max(1.0);
    let pad = tol * span;
    point[0] >= min[0] - pad
        && point[0] <= max[0] + pad
        && point[1] >= min[1] - pad
        && point[1] <= max[1] + pad
        && point[2] >= min[2] - pad
        && point[2] <= max[2] + pad
}

#[inline]
const fn is_finite_vec3(v: [f64; 3]) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

#[inline]
const fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
const fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
const fn min4(v: [f64; 4]) -> f64 {
    v[0].min(v[1]).min(v[2]).min(v[3])
}

#[cfg(test)]
mod tests {
    use tetraglass_core::{Attributes, ComponentCount, DataArray, TetView};

    use super::*;

    fn single_tet_view() -> TetView {
        let points = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ];
        let mut point_data = Attributes::new(4);
        point_data
            .insert(
                "linear",
                DataArray::from_f64(ComponentCount::Scalar, vec![1.0, 3.0, 7.0, 13.0]).unwrap(),
            )
            .unwrap();
        let mut cell_data = Attributes::new(1);
        cell_data
            .insert(
                "cell_id",
                DataArray::from_i32(ComponentCount::Scalar, vec![42]).unwrap(),
            )
            .unwrap();

        TetView::new(points, vec![[0, 1, 2, 3]], vec![42], point_data, cell_data).unwrap()
    }

    #[test]
    fn barycentric_coordinates_handle_vertices_centroid_and_degeneracy() {
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [0.0, 1.0, 0.0];
        let p3 = [0.0, 0.0, 1.0];

        assert_eq!(
            tet_barycentric_coordinates(p0, p1, p2, p3, p0),
            Some([1.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            tet_barycentric_coordinates(p0, p1, p2, p3, p1),
            Some([0.0, 1.0, 0.0, 0.0])
        );
        assert_eq!(
            tet_barycentric_coordinates(p0, p1, p2, p3, p2),
            Some([0.0, 0.0, 1.0, 0.0])
        );
        assert_eq!(
            tet_barycentric_coordinates(p0, p1, p2, p3, p3),
            Some([0.0, 0.0, 0.0, 1.0])
        );

        // Coplanar / degenerate tetrahedron returns None:
        assert_eq!(
            tet_barycentric_coordinates(p0, p1, p2, [0.5, 0.5, 0.0], [0.1, 0.1, 0.0]),
            None
        );

        // Non-finite query returns None:
        assert_eq!(
            tet_barycentric_coordinates(p0, p1, p2, p3, [f64::NAN, 0.0, 0.0]),
            None
        );
    }

    #[test]
    fn single_tet_locator_interpolates_positions_and_fields() {
        let view = single_tet_view();
        let locator = TetLocator::new(&view);
        assert_eq!(locator.num_tets(), 1);
        assert_eq!(locator.num_nodes(), 1);

        let query = [0.2, 0.6, 0.8];
        let loc = locator.locate(query).expect("point is inside single tet");
        assert_eq!(loc.tet_index, 0);
        assert_eq!(loc.source_cell_id, 42);
        assert_eq!(loc.interpolate_position(&view), Some(query));

        // Linear field f(x, y, z) = 1 + x + 2*y + 3*z:
        // f(0,0,0)=1, f(2,0,0)=3, f(0,3,0)=7, f(0,0,4)=13
        let linear_arr = view.point_data().get("linear").unwrap();
        let val = loc.interpolate_point_scalar(&view, linear_arr, 0).unwrap();
        let expected = 1.0 + query[0] + 2.0 * query[1] + 3.0 * query[2];
        assert!((val - expected).abs() < 1e-12);

        let cell_arr = view.cell_data().get("cell_id").unwrap();
        assert_eq!(loc.sample_cell_scalar(cell_arr, 0), Some(42.0));

        // Outside point returns None:
        assert!(locator.locate([2.0, 2.0, 2.0]).is_none());
        assert!(locator.locate([-0.01, 0.1, 0.1]).is_none());
    }
}

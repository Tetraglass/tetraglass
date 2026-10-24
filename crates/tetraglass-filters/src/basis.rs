//! Reference-element nodal coordinates and basis evaluation for quadratic serendipity
//! and arbitrary-order Lagrange 3D cells (`M2.1b`, `D-005`, `D-009`).
//!
//! All node orderings follow VTK's canonical conventions (`VTK_QUADRATIC_*` and
//! `VTK_LAGRANGE_*`). Reference coordinates `(r, s, t)` live in `[0, 1]^3`:
//! - **Tetrahedra:** `r >= 0, s >= 0, t >= 0, r + s + t <= 1`
//! - **Hexahedra:** `(r, s, t) in [0, 1]^3`
//! - **Wedges:** `r >= 0, s >= 0, r + s <= 1` and `t in [0, 1]`
//! - **Pyramids:** `(r, s) in [0, 1]^2` (normalized cross-section coordinates) and `t in [0, 1]`
//!   (where physical reference coordinates are `((1 - t) * r, (1 - t) * s, t)`).

use tetraglass_core::CellType;

/// Evaluates the 1D equispaced Lagrange basis polynomial $L_m^{(p)}(u)$ on $u \in [0, 1]$
/// for node $m \in 0..=p$ at $\xi_m = m / p$.
#[must_use]
pub fn lagrange_1d_basis(order: u8, node: u8, u: f64) -> f64 {
    if order == 0 {
        return 1.0;
    }
    let p = f64::from(order);
    let xm = f64::from(node) / p;
    let mut prod = 1.0;
    for n in 0..=order {
        if n != node {
            let xn = f64::from(n) / p;
            prod *= (u - xn) / (xm - xn);
        }
    }
    prod
}

/// Evaluates the Silvester simplex basis factor $\psi_m^{(p)}(\lambda) = \prod_{q=0}^{m-1} \frac{p \lambda - q}{q + 1}$.
#[must_use]
pub fn simplex_basis_term(order: u8, index: u8, lambda: f64) -> f64 {
    if index == 0 {
        return 1.0;
    }
    let p_lambda = f64::from(order) * lambda;
    let mut prod = 1.0;
    for q in 0..index {
        let qf = f64::from(q);
        prod *= (p_lambda - qf) / (qf + 1.0);
    }
    prod
}

/// Returns the `(i, j, k)` integer lattice coordinates (`0..=order`) for each VTK node slot
/// of a [`CellType::LagrangeHexahedron`] of degree `order`.
#[must_use]
pub fn lagrange_hex_lattice_nodes(order: u8) -> Vec<[u8; 3]> {
    let p = order;
    let n = (p as usize) + 1;
    let mut nodes = Vec::with_capacity(n * n * n);

    // 1. 8 corners
    nodes.extend_from_slice(&[
        [0, 0, 0],
        [p, 0, 0],
        [p, p, 0],
        [0, p, 0],
        [0, 0, p],
        [p, 0, p],
        [p, p, p],
        [0, p, p],
    ]);

    // 2. 12 edges (matched to VTK `vtkLagrangeHexahedron::PointIndexFromIJK`)
    for m in 1..p {
        nodes.push([m, 0, 0]); // edge 0: (0 -> 1)
    }
    for m in 1..p {
        nodes.push([p, m, 0]); // edge 1: (1 -> 2)
    }
    for m in 1..p {
        nodes.push([m, p, 0]); // edge 2: (3 -> 2)
    }
    for m in 1..p {
        nodes.push([0, m, 0]); // edge 3: (0 -> 3)
    }
    for m in 1..p {
        nodes.push([m, 0, p]); // edge 4: (4 -> 5)
    }
    for m in 1..p {
        nodes.push([p, m, p]); // edge 5: (5 -> 6)
    }
    for m in 1..p {
        nodes.push([m, p, p]); // edge 6: (7 -> 6)
    }
    for m in 1..p {
        nodes.push([0, m, p]); // edge 7: (4 -> 7)
    }
    for m in 1..p {
        nodes.push([0, 0, m]); // edge 8: (0 -> 4)
    }
    for m in 1..p {
        nodes.push([p, 0, m]); // edge 9: (1 -> 5)
    }
    for m in 1..p {
        nodes.push([p, p, m]); // edge 10: (2 -> 6)
    }
    for m in 1..p {
        nodes.push([0, p, m]); // edge 11: (3 -> 7)
    }

    // 3. 6 faces: (-x), (+x), (-y), (+y), (-z), (+z)
    for k in 1..p {
        for j in 1..p {
            nodes.push([0, j, k]); // -x
        }
    }
    for k in 1..p {
        for j in 1..p {
            nodes.push([p, j, k]); // +x
        }
    }
    for k in 1..p {
        for i in 1..p {
            nodes.push([i, 0, k]); // -y
        }
    }
    for k in 1..p {
        for i in 1..p {
            nodes.push([i, p, k]); // +y
        }
    }
    for j in 1..p {
        for i in 1..p {
            nodes.push([i, j, 0]); // -z
        }
    }
    for j in 1..p {
        for i in 1..p {
            nodes.push([i, j, p]); // +z
        }
    }

    // 4. Volume interior
    for k in 1..p {
        for j in 1..p {
            for i in 1..p {
                nodes.push([i, j, k]);
            }
        }
    }

    nodes
}

/// Returns the `(i, j, k, l)` barycentric integer lattice coordinates (`i + j + k + l == order`)
/// for each VTK node slot of a [`CellType::LagrangeTetrahedron`] of degree `order`, where
/// `i` corresponds to $r = \lambda_1$, `j` to $s = \lambda_2$, `k` to $t = \lambda_3$, and
/// `l = order - i - j - k` to $\lambda_0 = 1 - r - s - t$.
#[must_use]
pub fn lagrange_tet_lattice_nodes(order: u8) -> Vec<[u8; 4]> {
    let p = order;
    let n = (p as usize) + 1;
    let total = n * (n + 1) * (n + 2) / 6;
    let mut nodes = Vec::with_capacity(total);

    // 1. 4 corners
    nodes.extend_from_slice(&[[0, 0, 0, p], [p, 0, 0, 0], [0, p, 0, 0], [0, 0, p, 0]]);

    // 2. 6 edges: (0,1), (1,2), (2,0), (0,3), (1,3), (2,3)
    for m in 1..p {
        nodes.push([m, 0, 0, p - m]); // (0 -> 1)
    }
    for m in 1..p {
        nodes.push([p - m, m, 0, 0]); // (1 -> 2)
    }
    for m in 1..p {
        nodes.push([0, p - m, 0, m]); // (2 -> 0)
    }
    for m in 1..p {
        nodes.push([0, 0, m, p - m]); // (0 -> 3)
    }
    for m in 1..p {
        nodes.push([p - m, 0, m, 0]); // (1 -> 3)
    }
    for m in 1..p {
        nodes.push([0, p - m, m, 0]); // (2 -> 3)
    }

    // 3. 4 faces: (0,1,3) [j=0], (1,2,3) [l=0], (2,0,3) [i=0], (0,2,1) [k=0]
    for k in 1..p {
        for i in 1..(p - k) {
            nodes.push([i, 0, k, p - i - k]);
        }
    }
    for k in 1..p {
        for j in 1..(p - k) {
            let i = p - j - k;
            nodes.push([i, j, k, 0]);
        }
    }
    for k in 1..p {
        for j in 1..(p - k) {
            nodes.push([0, j, k, p - j - k]);
        }
    }
    for j in 1..p {
        for i in 1..(p - j) {
            nodes.push([i, j, 0, p - i - j]);
        }
    }

    // 4. Volume interior (i >= 1, j >= 1, k >= 1, l >= 1)
    for k in 1..p {
        for j in 1..(p - k) {
            for i in 1..(p - k - j) {
                let l = p - i - j - k;
                nodes.push([i, j, k, l]);
            }
        }
    }

    nodes
}

/// Returns the `(i, j, l2, k)` lattice coordinates (`i + j + l2 == order` on the triangle,
/// `k in 0..=order` along the prism axis) for each VTK node slot of a [`CellType::LagrangeWedge`].
#[must_use]
pub fn lagrange_wedge_lattice_nodes(order: u8) -> Vec<[u8; 4]> {
    let p = order;
    let n = (p as usize) + 1;
    let total = n * (n + 1) / 2 * n;
    let mut nodes = Vec::with_capacity(total);

    // 1. 6 corners
    nodes.extend_from_slice(&[
        [0, 0, p, 0],
        [p, 0, 0, 0],
        [0, p, 0, 0],
        [0, 0, p, p],
        [p, 0, 0, p],
        [0, p, 0, p],
    ]);

    // 2. 9 edges: bottom tri (0,1), (1,2), (2,0); top tri (3,4), (4,5), (5,3); pillars (0,3), (1,4), (2,5)
    for m in 1..p {
        nodes.push([m, 0, p - m, 0]); // (0 -> 1)
    }
    for m in 1..p {
        nodes.push([p - m, m, 0, 0]); // (1 -> 2)
    }
    for m in 1..p {
        nodes.push([0, p - m, m, 0]); // (2 -> 0)
    }
    for m in 1..p {
        nodes.push([m, 0, p - m, p]); // (3 -> 4)
    }
    for m in 1..p {
        nodes.push([p - m, m, 0, p]); // (4 -> 5)
    }
    for m in 1..p {
        nodes.push([0, p - m, m, p]); // (5 -> 3)
    }
    for k in 1..p {
        nodes.push([0, 0, p, k]); // (0 -> 3)
    }
    for k in 1..p {
        nodes.push([p, 0, 0, k]); // (1 -> 4)
    }
    for k in 1..p {
        nodes.push([0, p, 0, k]); // (2 -> 5)
    }

    // 3. Faces: bottom triangle (k = 0), top triangle (k = p), 3 quad faces (0,1,4,3), (1,2,5,4), (2,0,3,5)
    for j in 1..p {
        for i in 1..(p - j) {
            nodes.push([i, j, p - i - j, 0]);
        }
    }
    for j in 1..p {
        for i in 1..(p - j) {
            nodes.push([i, j, p - i - j, p]);
        }
    }
    for k in 1..p {
        for m in 1..p {
            nodes.push([m, 0, p - m, k]); // quad (0,1,4,3)
        }
    }
    for k in 1..p {
        for m in 1..p {
            nodes.push([p - m, m, 0, k]); // quad (1,2,5,4)
        }
    }
    for k in 1..p {
        for m in 1..p {
            nodes.push([0, p - m, m, k]); // quad (2,0,3,5)
        }
    }

    // 4. Volume interior
    for k in 1..p {
        for j in 1..p {
            for i in 1..(p - j) {
                nodes.push([i, j, p - i - j, k]);
            }
        }
    }

    nodes
}

/// Returns the `(i, j, l0)` barycentric integer lattice coordinates (`i + j + l0 == order`)
/// for each VTK node slot of a [`CellType::LagrangeTriangle`] of degree `order`.
#[must_use]
pub fn lagrange_tri_lattice_nodes(order: u8) -> Vec<[u8; 3]> {
    let p = order;
    let n = (p as usize) + 1;
    let mut nodes = Vec::with_capacity(n * (n + 1) / 2);

    // 1. 3 corners
    nodes.extend_from_slice(&[[0, 0, p], [p, 0, 0], [0, p, 0]]);

    // 2. 3 edges: (0 -> 1), (1 -> 2), (2 -> 0)
    for m in 1..p {
        nodes.push([m, 0, p - m]);
    }
    for m in 1..p {
        nodes.push([p - m, m, 0]);
    }
    for m in 1..p {
        nodes.push([0, p - m, m]);
    }

    // 3. Face interior
    for j in 1..p {
        for i in 1..(p - j) {
            nodes.push([i, j, p - i - j]);
        }
    }

    nodes
}

/// Returns the `(i, j)` integer lattice coordinates (`0..=order`) for each VTK node slot
/// of a [`CellType::LagrangeQuadrilateral`] of degree `order`.
#[must_use]
pub fn lagrange_quad_lattice_nodes(order: u8) -> Vec<[u8; 2]> {
    let p = order;
    let n = (p as usize) + 1;
    let mut nodes = Vec::with_capacity(n * n);

    // 1. 4 corners
    nodes.extend_from_slice(&[[0, 0], [p, 0], [p, p], [0, p]]);

    // 2. 4 edges: (0 -> 1), (1 -> 2), (3 -> 2), (0 -> 3)
    for m in 1..p {
        nodes.push([m, 0]);
    }
    for m in 1..p {
        nodes.push([p, m]);
    }
    for m in 1..p {
        nodes.push([m, p]);
    }
    for m in 1..p {
        nodes.push([0, m]);
    }

    // 3. Face interior
    for j in 1..p {
        for i in 1..p {
            nodes.push([i, j]);
        }
    }

    nodes
}

/// Returns the reference `[r, s, t]` coordinate of local corner slot `corner_slot` on `cell_type`.
#[must_use]
pub const fn corner_reference_rst(cell_type: CellType, corner_slot: u8) -> [f64; 3] {
    match cell_type {
        CellType::Triangle | CellType::QuadraticTriangle | CellType::LagrangeTriangle { .. } => {
            match corner_slot {
                0 => [0.0, 0.0, 0.0],
                1 => [1.0, 0.0, 0.0],
                _ => [0.0, 1.0, 0.0],
            }
        }
        CellType::Quad | CellType::QuadraticQuad | CellType::LagrangeQuadrilateral { .. } => {
            match corner_slot {
                0 => [0.0, 0.0, 0.0],
                1 => [1.0, 0.0, 0.0],
                2 => [1.0, 1.0, 0.0],
                _ => [0.0, 1.0, 0.0],
            }
        }
        CellType::Tetra | CellType::QuadraticTetra | CellType::LagrangeTetrahedron { .. } => {
            match corner_slot {
                0 => [0.0, 0.0, 0.0],
                1 => [1.0, 0.0, 0.0],
                2 => [0.0, 1.0, 0.0],
                _ => [0.0, 0.0, 1.0],
            }
        }
        CellType::Hexahedron
        | CellType::QuadraticHexahedron
        | CellType::LagrangeHexahedron { .. } => match corner_slot {
            0 => [0.0, 0.0, 0.0],
            1 => [1.0, 0.0, 0.0],
            2 => [1.0, 1.0, 0.0],
            3 => [0.0, 1.0, 0.0],
            4 => [0.0, 0.0, 1.0],
            5 => [1.0, 0.0, 1.0],
            6 => [1.0, 1.0, 1.0],
            _ => [0.0, 1.0, 1.0],
        },
        CellType::Wedge | CellType::QuadraticWedge | CellType::LagrangeWedge { .. } => {
            match corner_slot {
                0 => [0.0, 0.0, 0.0],
                1 => [1.0, 0.0, 0.0],
                2 => [0.0, 1.0, 0.0],
                3 => [0.0, 0.0, 1.0],
                4 => [1.0, 0.0, 1.0],
                _ => [0.0, 1.0, 1.0],
            }
        }
        CellType::Pyramid | CellType::QuadraticPyramid => match corner_slot {
            0 => [0.0, 0.0, 0.0],
            1 => [1.0, 0.0, 0.0],
            2 => [1.0, 1.0, 0.0],
            3 => [0.0, 1.0, 0.0],
            _ => [0.5, 0.5, 1.0],
        },
        _ => [0.0, 0.0, 0.0],
    }
}

/// Evaluates the nodal shape functions $N_0(r, s, t), \dots, N_{n-1}(r, s, t)$ for a 2D or 3D
/// higher-order cell at reference coordinates `rst = [r, s, t]` into `weights_out`.
pub fn evaluate_cell_basis(cell_type: CellType, rst: [f64; 3], weights_out: &mut [f64]) {
    let [r, s, t] = rst;
    match cell_type {
        CellType::QuadraticTriangle => {
            eval_lagrange_tri(2, &lagrange_tri_lattice_nodes(2), r, s, weights_out);
        }
        CellType::LagrangeTriangle { order } => {
            let nodes = lagrange_tri_lattice_nodes(order);
            eval_lagrange_tri(order, &nodes, r, s, weights_out);
        }
        CellType::QuadraticQuad => {
            eval_quadratic_quad(r, s, weights_out);
        }
        CellType::LagrangeQuadrilateral { order } => {
            let nodes = lagrange_quad_lattice_nodes(order);
            eval_lagrange_quad(order, &nodes, r, s, weights_out);
        }
        CellType::QuadraticTetra => {
            eval_lagrange_tet(2, &lagrange_tet_lattice_nodes(2), r, s, t, weights_out);
        }
        CellType::LagrangeTetrahedron { order } => {
            let nodes = lagrange_tet_lattice_nodes(order);
            eval_lagrange_tet(order, &nodes, r, s, t, weights_out);
        }
        CellType::QuadraticHexahedron => {
            eval_quadratic_hex(r, s, t, weights_out);
        }
        CellType::LagrangeHexahedron { order } => {
            let nodes = lagrange_hex_lattice_nodes(order);
            eval_lagrange_hex(order, &nodes, r, s, t, weights_out);
        }
        CellType::QuadraticWedge => {
            eval_quadratic_wedge(r, s, t, weights_out);
        }
        CellType::LagrangeWedge { order } => {
            let nodes = lagrange_wedge_lattice_nodes(order);
            eval_lagrange_wedge(order, &nodes, r, s, t, weights_out);
        }
        CellType::QuadraticPyramid => {
            eval_quadratic_pyramid(r, s, t, weights_out);
        }
        _ => {}
    }
}

/// Evaluates a 2D Lagrange triangle of degree `order` with precomputed `lattice_nodes`.
pub fn eval_lagrange_tri(
    order: u8,
    lattice_nodes: &[[u8; 3]],
    r: f64,
    s: f64,
    weights_out: &mut [f64],
) {
    let l0 = 1.0 - r - s;
    for (slot, &[i, j, l]) in lattice_nodes.iter().enumerate() {
        weights_out[slot] = simplex_basis_term(order, i, r)
            * simplex_basis_term(order, j, s)
            * simplex_basis_term(order, l, l0);
    }
}

/// Evaluates a 2D Lagrange quadrilateral of degree `order` with precomputed `lattice_nodes`.
pub fn eval_lagrange_quad(
    order: u8,
    lattice_nodes: &[[u8; 2]],
    r: f64,
    s: f64,
    weights_out: &mut [f64],
) {
    let n = (order as usize) + 1;
    let mut lr = vec![0.0; n];
    let mut ls = vec![0.0; n];
    for m in 0..=order {
        let idx = m as usize;
        lr[idx] = lagrange_1d_basis(order, m, r);
        ls[idx] = lagrange_1d_basis(order, m, s);
    }
    for (slot, &[i, j]) in lattice_nodes.iter().enumerate() {
        weights_out[slot] = lr[i as usize] * ls[j as usize];
    }
}

/// Evaluates the 8-node serendipity quadratic quadrilateral (`VTK_QUADRATIC_QUAD`, type 23).
pub fn eval_quadratic_quad(r: f64, s: f64, weights_out: &mut [f64]) {
    let xi = 2.0 * r - 1.0;
    let eta = 2.0 * s - 1.0;
    let corner_signs: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    for (i, &[sx, sy]) in corner_signs.iter().enumerate() {
        weights_out[i] = 0.25 * (1.0 + sx * xi) * (1.0 + sy * eta) * (sx * xi + sy * eta - 1.0);
    }
    let one_minus_xi2 = 1.0 - xi * xi;
    let one_minus_eta2 = 1.0 - eta * eta;
    weights_out[4] = 0.5 * one_minus_xi2 * (1.0 - eta);
    weights_out[5] = 0.5 * (1.0 + xi) * one_minus_eta2;
    weights_out[6] = 0.5 * one_minus_xi2 * (1.0 + eta);
    weights_out[7] = 0.5 * (1.0 - xi) * one_minus_eta2;
}

/// Evaluates a Lagrange tetrahedron of degree `order` with precomputed `lattice_nodes`.
pub fn eval_lagrange_tet(
    order: u8,
    lattice_nodes: &[[u8; 4]],
    r: f64,
    s: f64,
    t: f64,
    weights_out: &mut [f64],
) {
    let l0 = 1.0 - r - s - t;
    for (slot, &[i, j, k, l]) in lattice_nodes.iter().enumerate() {
        weights_out[slot] = simplex_basis_term(order, i, r)
            * simplex_basis_term(order, j, s)
            * simplex_basis_term(order, k, t)
            * simplex_basis_term(order, l, l0);
    }
}

/// Evaluates a Lagrange hexahedron of degree `order` with precomputed `lattice_nodes`.
pub fn eval_lagrange_hex(
    order: u8,
    lattice_nodes: &[[u8; 3]],
    r: f64,
    s: f64,
    t: f64,
    weights_out: &mut [f64],
) {
    let n = (order as usize) + 1;
    let mut lr = vec![0.0; n];
    let mut ls = vec![0.0; n];
    let mut lt = vec![0.0; n];
    for m in 0..=order {
        let idx = m as usize;
        lr[idx] = lagrange_1d_basis(order, m, r);
        ls[idx] = lagrange_1d_basis(order, m, s);
        lt[idx] = lagrange_1d_basis(order, m, t);
    }
    for (slot, &[i, j, k]) in lattice_nodes.iter().enumerate() {
        weights_out[slot] = lr[i as usize] * ls[j as usize] * lt[k as usize];
    }
}

/// Evaluates a Lagrange wedge of degree `order` with precomputed `lattice_nodes`.
pub fn eval_lagrange_wedge(
    order: u8,
    lattice_nodes: &[[u8; 4]],
    r: f64,
    s: f64,
    t: f64,
    weights_out: &mut [f64],
) {
    let l2 = 1.0 - r - s;
    let n = (order as usize) + 1;
    let mut lt = vec![0.0; n];
    for m in 0..=order {
        lt[m as usize] = lagrange_1d_basis(order, m, t);
    }
    for (slot, &[i, j, l, k]) in lattice_nodes.iter().enumerate() {
        weights_out[slot] = simplex_basis_term(order, i, r)
            * simplex_basis_term(order, j, s)
            * simplex_basis_term(order, l, l2)
            * lt[k as usize];
    }
}

/// Evaluates the 20-node serendipity quadratic hexahedron (`VTK_QUADRATIC_HEXAHEDRON`, type 25).
pub fn eval_quadratic_hex(r: f64, s: f64, t: f64, weights_out: &mut [f64]) {
    let xi = 2.0 * r - 1.0;
    let eta = 2.0 * s - 1.0;
    let zeta = 2.0 * t - 1.0;

    let corner_signs: [[f64; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    for (i, &[sx, sy, sz]) in corner_signs.iter().enumerate() {
        weights_out[i] = 0.125
            * (1.0 + sx * xi)
            * (1.0 + sy * eta)
            * (1.0 + sz * zeta)
            * (sx * xi + sy * eta + sz * zeta - 2.0);
    }

    let one_minus_xi2 = 1.0 - xi * xi;
    let one_minus_eta2 = 1.0 - eta * eta;
    let one_minus_zeta2 = 1.0 - zeta * zeta;

    // Bottom quad mid-edges: 8=(0,1), 9=(1,2), 10=(2,3), 11=(3,0)
    weights_out[8] = 0.25 * one_minus_xi2 * (1.0 - eta) * (1.0 - zeta);
    weights_out[9] = 0.25 * (1.0 + xi) * one_minus_eta2 * (1.0 - zeta);
    weights_out[10] = 0.25 * one_minus_xi2 * (1.0 + eta) * (1.0 - zeta);
    weights_out[11] = 0.25 * (1.0 - xi) * one_minus_eta2 * (1.0 - zeta);

    // Top quad mid-edges: 12=(4,5), 13=(5,6), 14=(6,7), 15=(7,4)
    weights_out[12] = 0.25 * one_minus_xi2 * (1.0 - eta) * (1.0 + zeta);
    weights_out[13] = 0.25 * (1.0 + xi) * one_minus_eta2 * (1.0 + zeta);
    weights_out[14] = 0.25 * one_minus_xi2 * (1.0 + eta) * (1.0 + zeta);
    weights_out[15] = 0.25 * (1.0 - xi) * one_minus_eta2 * (1.0 + zeta);

    // Vertical pillar mid-edges: 16=(0,4), 17=(1,5), 18=(2,6), 19=(3,7)
    weights_out[16] = 0.25 * (1.0 - xi) * (1.0 - eta) * one_minus_zeta2;
    weights_out[17] = 0.25 * (1.0 + xi) * (1.0 - eta) * one_minus_zeta2;
    weights_out[18] = 0.25 * (1.0 + xi) * (1.0 + eta) * one_minus_zeta2;
    weights_out[19] = 0.25 * (1.0 - xi) * (1.0 + eta) * one_minus_zeta2;
}

/// Evaluates the 15-node serendipity quadratic wedge (`VTK_QUADRATIC_WEDGE`, type 26).
pub fn eval_quadratic_wedge(r: f64, s: f64, t: f64, weights_out: &mut [f64]) {
    let lam = [1.0 - r - s, r, s];
    let zeta = 2.0 * t - 1.0;

    // Corners 0..3 (zeta_i = -1) and 3..6 (zeta_i = +1)
    for m in 0..3 {
        let lm = lam[m];
        weights_out[m] = 0.5 * lm * (1.0 - zeta) * (2.0 * lm - 2.0 - zeta);
        weights_out[3 + m] = 0.5 * lm * (1.0 + zeta) * (2.0 * lm - 2.0 + zeta);
    }

    // Bottom triangle mid-edges: 6=(0,1), 7=(1,2), 8=(2,0)
    weights_out[6] = 2.0 * lam[0] * lam[1] * (1.0 - zeta);
    weights_out[7] = 2.0 * lam[1] * lam[2] * (1.0 - zeta);
    weights_out[8] = 2.0 * lam[2] * lam[0] * (1.0 - zeta);

    // Top triangle mid-edges: 9=(3,4), 10=(4,5), 11=(5,3)
    weights_out[9] = 2.0 * lam[0] * lam[1] * (1.0 + zeta);
    weights_out[10] = 2.0 * lam[1] * lam[2] * (1.0 + zeta);
    weights_out[11] = 2.0 * lam[2] * lam[0] * (1.0 + zeta);

    // Pillar mid-edges: 12=(0,3), 13=(1,4), 14=(2,5)
    let one_minus_zeta2 = 1.0 - zeta * zeta;
    weights_out[12] = lam[0] * one_minus_zeta2;
    weights_out[13] = lam[1] * one_minus_zeta2;
    weights_out[14] = lam[2] * one_minus_zeta2;
}

/// Evaluates the 13-node serendipity quadratic pyramid (`VTK_QUADRATIC_PYRAMID`, type 27),
/// parameterized by normalized horizontal coordinates `(r, s) in [0, 1]^2` and height `t in [0, 1]`.
pub fn eval_quadratic_pyramid(r: f64, s: f64, t: f64, weights_out: &mut [f64]) {
    let xi = 2.0 * r - 1.0;
    let eta = 2.0 * s - 1.0;
    let one_minus_t = 1.0 - t;

    let q0 = 0.25 * (1.0 - xi) * (1.0 - eta);
    let q1 = 0.25 * (1.0 + xi) * (1.0 - eta);
    let q2 = 0.25 * (1.0 + xi) * (1.0 + eta);
    let q3 = 0.25 * (1.0 - xi) * (1.0 + eta);

    let l0 = one_minus_t * q0;
    let l1 = one_minus_t * q1;
    let l2 = one_minus_t * q2;
    let l3 = one_minus_t * q3;
    let l4 = t;

    let one_minus_xi2 = 1.0 - xi * xi;
    let one_minus_eta2 = 1.0 - eta * eta;

    // Base quad mid-edges: 5=(0,1), 6=(1,2), 7=(2,3), 8=(3,0)
    let e5 = 0.5 * one_minus_xi2 * (1.0 - eta) * one_minus_t;
    let e6 = 0.5 * (1.0 + xi) * one_minus_eta2 * one_minus_t;
    let e7 = 0.5 * one_minus_xi2 * (1.0 + eta) * one_minus_t;
    let e8 = 0.5 * (1.0 - xi) * one_minus_eta2 * one_minus_t;

    // Apex mid-edges: 9=(0,4), 10=(1,4), 11=(2,4), 12=(3,4)
    let four_t_omt = 4.0 * t * one_minus_t;
    let e9 = four_t_omt * q0;
    let e10 = four_t_omt * q1;
    let e11 = four_t_omt * q2;
    let e12 = four_t_omt * q3;

    weights_out[0] = l0 - 0.5 * (e5 + e8 + e9);
    weights_out[1] = l1 - 0.5 * (e5 + e6 + e10);
    weights_out[2] = l2 - 0.5 * (e6 + e7 + e11);
    weights_out[3] = l3 - 0.5 * (e7 + e8 + e12);
    weights_out[4] = l4 - 0.5 * (e9 + e10 + e11 + e12);
    weights_out[5] = e5;
    weights_out[6] = e6;
    weights_out[7] = e7;
    weights_out[8] = e8;
    weights_out[9] = e9;
    weights_out[10] = e10;
    weights_out[11] = e11;
    weights_out[12] = e12;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_nodes_for_cell(cell_type: CellType) -> Vec<[f64; 3]> {
        match cell_type {
            CellType::QuadraticTetra => lagrange_tet_lattice_nodes(2)
                .into_iter()
                .map(|[i, j, k, _]| [f64::from(i) / 2.0, f64::from(j) / 2.0, f64::from(k) / 2.0])
                .collect(),
            CellType::LagrangeTetrahedron { order } => {
                let p = f64::from(order);
                lagrange_tet_lattice_nodes(order)
                    .into_iter()
                    .map(|[i, j, k, _]| [f64::from(i) / p, f64::from(j) / p, f64::from(k) / p])
                    .collect()
            }
            CellType::QuadraticHexahedron => vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [0.5, 1.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 1.0],
                [1.0, 0.5, 1.0],
                [0.5, 1.0, 1.0],
                [0.0, 0.5, 1.0],
                [0.0, 0.0, 0.5],
                [1.0, 0.0, 0.5],
                [1.0, 1.0, 0.5],
                [0.0, 1.0, 0.5],
            ],
            CellType::LagrangeHexahedron { order } => {
                let p = f64::from(order);
                lagrange_hex_lattice_nodes(order)
                    .into_iter()
                    .map(|[i, j, k]| [f64::from(i) / p, f64::from(j) / p, f64::from(k) / p])
                    .collect()
            }
            CellType::QuadraticWedge => vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 1.0],
                [0.5, 0.5, 1.0],
                [0.0, 0.5, 1.0],
                [0.0, 0.0, 0.5],
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
            ],
            CellType::LagrangeWedge { order } => {
                let p = f64::from(order);
                lagrange_wedge_lattice_nodes(order)
                    .into_iter()
                    .map(|[i, j, _, k]| [f64::from(i) / p, f64::from(j) / p, f64::from(k) / p])
                    .collect()
            }
            CellType::QuadraticPyramid => vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [0.5, 1.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [1.0, 0.0, 0.5],
                [1.0, 1.0, 0.5],
                [0.0, 1.0, 0.5],
            ],
            other => panic!("unexpected cell type {other:?}"),
        }
    }

    #[test]
    fn all_higher_order_bases_satisfy_kronecker_delta_partition_of_unity_and_linear_reproduction() {
        let cell_types = [
            CellType::QuadraticTetra,
            CellType::QuadraticHexahedron,
            CellType::QuadraticWedge,
            CellType::QuadraticPyramid,
            CellType::LagrangeTetrahedron { order: 1 },
            CellType::LagrangeTetrahedron { order: 2 },
            CellType::LagrangeTetrahedron { order: 3 },
            CellType::LagrangeHexahedron { order: 1 },
            CellType::LagrangeHexahedron { order: 2 },
            CellType::LagrangeHexahedron { order: 3 },
            CellType::LagrangeWedge { order: 1 },
            CellType::LagrangeWedge { order: 2 },
            CellType::LagrangeWedge { order: 3 },
        ];

        for ct in cell_types {
            let ref_nodes = reference_nodes_for_cell(ct);
            let n = ct.expected_node_count().unwrap();
            assert_eq!(ref_nodes.len(), n, "{ct:?} node count");

            let mut weights = vec![0.0; n];

            // 1. Kronecker delta at every reference node
            for (b, &rst) in ref_nodes.iter().enumerate() {
                evaluate_cell_basis(ct, rst, &mut weights);
                for (a, &w) in weights.iter().enumerate() {
                    let expected = if a == b { 1.0 } else { 0.0 };
                    assert!(
                        (w - expected).abs() < 1e-12,
                        "{ct:?} Kronecker delta failed at node {b} slot {a}: got {w}, expected {expected}"
                    );
                }
            }

            // 2. Partition of unity and exact linear reproduction at interior test points
            let sample_points = [
                [0.15, 0.25, 0.35],
                [0.2, 0.1, 0.4],
                [0.33, 0.22, 0.11],
                [0.05, 0.45, 0.25],
            ];
            // Physical node coordinates on the element (accounting for pyramid tapering `(1 - t)`)
            let phys_nodes: Vec<[f64; 3]> = ref_nodes
                .iter()
                .map(|&[r, s, t]| {
                    if ct == CellType::QuadraticPyramid {
                        [(1.0 - t) * r, (1.0 - t) * s, t]
                    } else {
                        [r, s, t]
                    }
                })
                .collect();

            for rst in sample_points {
                evaluate_cell_basis(ct, rst, &mut weights);
                let sum_w: f64 = weights.iter().sum();
                assert!(
                    (sum_w - 1.0).abs() < 1e-12,
                    "{ct:?} partition of unity failed at {rst:?}: sum = {sum_w}"
                );

                let expected_phys = if ct == CellType::QuadraticPyramid {
                    [(1.0 - rst[2]) * rst[0], (1.0 - rst[2]) * rst[1], rst[2]]
                } else {
                    rst
                };
                let expected_field =
                    1.5 + 2.0 * expected_phys[0] - 3.5 * expected_phys[1] + 4.25 * expected_phys[2];

                let mut interp_field = 0.0;
                for (a, &w) in weights.iter().enumerate() {
                    let p = phys_nodes[a];
                    let node_val = 1.5 + 2.0 * p[0] - 3.5 * p[1] + 4.25 * p[2];
                    interp_field += w * node_val;
                }
                assert!(
                    (interp_field - expected_field).abs() < 1e-12,
                    "{ct:?} linear reproduction failed at {rst:?}: {interp_field} != {expected_field}"
                );
            }
        }
    }
}

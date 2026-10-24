//! Integration tests for the marching-tetrahedra kernel and plane `slice` filter (`M2.3a`).
#![cfg(not(target_arch = "wasm32"))]

use std::collections::HashMap;
use std::path::PathBuf;

use tetraglass_core::{SurfaceMesh, SyntheticCellLayout, SyntheticGridSpec, UnstructuredGrid};
use tetraglass_filters::{
    FilterError, LinearizeParams, SliceParams, linearize_default, slice, slice_grid,
    slice_with_chunk_size,
};
use tetraglass_io::read_vtu_from_path;

fn vtk_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ directory")
        .parent()
        .expect("workspace root")
        .join("testdata")
        .join("vtk")
}

fn assert_slice_invariants(
    grid: &UnstructuredGrid,
    cut: &SurfaceMesh,
    origin: [f64; 3],
    unit_normal: [f64; 3],
) {
    let pts = cut.points();

    // 1. Every vertex must lie on the slice plane within floating-point tolerance.
    for (idx, &p) in pts.iter().enumerate() {
        let dist = (p[0] - origin[0]) * unit_normal[0]
            + (p[1] - origin[1]) * unit_normal[1]
            + (p[2] - origin[2]) * unit_normal[2];
        assert!(
            dist.abs() < 1e-11,
            "slice vertex {idx} ({p:?}) is at signed distance {dist:e} from plane"
        );
    }

    // 2. Every triangle must have positive area and normal aligned with +unit_normal,
    // and no undirected triangle edge may be shared by more than 2 triangles (no duplicate/overlapping tris).
    let mut edge_counts: HashMap<(u32, u32), u32> = HashMap::new();
    for (tri_idx, &[i0, i1, i2]) in cut.triangles().iter().enumerate() {
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
        let dot = n[0] * unit_normal[0] + n[1] * unit_normal[1] + n[2] * unit_normal[2];
        assert!(
            dot > 1e-14,
            "triangle {tri_idx} normal {n:?} is not aligned with plane normal {unit_normal:?} (dot = {dot:e})"
        );

        for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
            let key = if a <= b { (a, b) } else { (b, a) };
            *edge_counts.entry(key).or_insert(0) += 1;
        }
    }
    for (edge, count) in edge_counts {
        assert!(
            count <= 2,
            "slice edge {edge:?} has {count} incident triangles (non-manifold / duplicate triangle)"
        );
    }

    // 3. Linear scalar field `linear = x + 2*y + 3*z` and vector field `velocity = [-y, x, 0]`
    // must be reproduced accurately at every slice vertex.
    if let Some(linear_arr) = cut.point_data().get("linear") {
        assert_eq!(linear_arr.num_tuples(), cut.num_points());
        for (idx, &[x, y, z]) in pts.iter().enumerate() {
            let expected = x + 2.0 * y + 3.0 * z;
            let actual = linear_arr.value_as_f64(idx, 0).unwrap();
            assert!(
                (actual - expected).abs() < 1e-11,
                "point {idx} `linear` mismatch: {actual} vs {expected}"
            );
        }
    }
    if let Some(vel_arr) = cut.point_data().get("velocity") {
        assert_eq!(vel_arr.num_tuples(), cut.num_points());
        for (idx, &[x, y, _z]) in pts.iter().enumerate() {
            let expected = [-y, x, 0.0];
            for (c, &exp_c) in expected.iter().enumerate() {
                let actual = vel_arr.value_as_f64(idx, c).unwrap();
                assert!(
                    (actual - exp_c).abs() < 1e-5,
                    "point {idx} `velocity[{c}]` mismatch: {actual} vs {exp_c}"
                );
            }
        }
    }

    // 4. Cell attribute `cell_id` must match `triangle_source_cells()`.
    if let Some(cell_id_arr) = cut.cell_data().get("cell_id") {
        assert_eq!(cell_id_arr.num_tuples(), cut.num_triangles());
        for (tri_idx, &src_cell) in cut.triangle_source_cells().iter().enumerate() {
            assert!(
                (src_cell as usize) < grid.num_cells(),
                "out-of-range source cell {src_cell}"
            );
            assert_eq!(
                cell_id_arr.value_as_f64(tri_idx, 0),
                Some(f64::from(src_cell))
            );
        }
    }
}

#[test]
fn slice_covers_all_cell_families_and_analytic_cross_section_areas() {
    let dir = vtk_dir();

    // 1. Linear tetrahedra on [0, 1]^3: axis-aligned slices have area 1.0; diagonal x - y = 0 has area sqrt(2).
    let tet_lin = read_vtu_from_path(dir.join("tet_linear.vtu")).unwrap();
    let tet_lin_view = linearize_default(&tet_lin).unwrap();
    for normal in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        let params = SliceParams::with_origin([0.37, 0.42, 0.61], normal);
        let cut = slice(&tet_lin_view, &params).unwrap();
        assert!((cut.total_area() - 1.0).abs() < 1e-12);
        assert_slice_invariants(&tet_lin, &cut, [0.37, 0.42, 0.61], normal);
    }

    // 2. Linear hexahedra (2x2x2 on [0, 2]^3): cross-section area is 2 * 2 = 4.0.
    let hex_lin = read_vtu_from_path(dir.join("hex_linear.vtu")).unwrap();
    let hex_cut = slice_grid(
        &hex_lin,
        &SliceParams::with_origin([0.65, 0.75, 0.85], [0.0, 0.0, 1.0]),
        LinearizeParams::default(),
    )
    .unwrap();
    assert!((hex_cut.total_area() - 4.0).abs() < 1e-12);
    assert_slice_invariants(&hex_lin, &hex_cut, [0.65, 0.75, 0.85], [0.0, 0.0, 1.0]);

    // 3. Mixed linear cells (Hexahedron + Wedge + Pyramid + Tetra):
    // Slice at z = 0.5 cuts the hex ([0,1]x[0,1]) and the wedge ([1,2]x[0,1], area 0.5) -> total area 1.5.
    let mixed = read_vtu_from_path(dir.join("mixed_linear.vtu")).unwrap();
    let mixed_cut_z05 = slice_grid(
        &mixed,
        &SliceParams::with_origin([0.5, 0.5, 0.5], [0.0, 0.0, 1.0]),
        LinearizeParams::default(),
    )
    .unwrap();
    assert!((mixed_cut_z05.total_area() - 1.5).abs() < 1e-12);
    assert_slice_invariants(&mixed, &mixed_cut_z05, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0]);

    // Slice at z = 1.25 cuts the pyramid and tetrahedron above z = 1.0.
    let mixed_cut_z125 = slice_grid(
        &mixed,
        &SliceParams::with_origin([0.5, 0.5, 1.25], [0.0, 0.0, 1.0]),
        LinearizeParams::default(),
    )
    .unwrap();
    assert!(mixed_cut_z125.num_triangles() > 0);
    assert_slice_invariants(&mixed, &mixed_cut_z125, [0.5, 0.5, 1.25], [0.0, 0.0, 1.0]);

    // 4. Quadratic tetrahedra on [0, 1]^3: default center slice has area 1.0.
    let tet_quad = read_vtu_from_path(dir.join("tet_quadratic.vtu")).unwrap();
    let tet_quad_cut = slice_grid(
        &tet_quad,
        &SliceParams::new([0.0, 1.0, 0.0]),
        LinearizeParams::default(),
    )
    .unwrap();
    assert!((tet_quad_cut.total_area() - 1.0).abs() < 1e-12);
    assert_slice_invariants(&tet_quad, &tet_quad_cut, [0.5, 0.5, 0.5], [0.0, 1.0, 0.0]);

    // 5. Third-order Lagrange hexahedra on [0, 2] x [0, 1]^2:
    // Slice normal +Z at center [1.0, 0.5, 0.5] has area 2.0; normal +X at x = 0.5 has area 1.0.
    let hex_lag = read_vtu_from_path(dir.join("hex_lagrange_p3.vtu")).unwrap();
    let hex_lag_cut_z = slice_grid(
        &hex_lag,
        &SliceParams::new([0.0, 0.0, 1.0]),
        LinearizeParams::default(),
    )
    .unwrap();
    assert!((hex_lag_cut_z.total_area() - 2.0).abs() < 1e-12);
    assert_slice_invariants(&hex_lag, &hex_lag_cut_z, [1.0, 0.5, 0.5], [0.0, 0.0, 1.0]);

    // 6. Polyhedra mesh: slice at z = 0.5 cuts both polyhedra.
    let poly = read_vtu_from_path(dir.join("polyhedra.vtu")).unwrap();
    let poly_cut = slice_grid(
        &poly,
        &SliceParams::with_origin([0.5, 0.5, 0.5], [0.0, 0.0, 1.0]),
        LinearizeParams::default(),
    )
    .unwrap();
    assert!(poly_cut.total_area() > 1.5);
    assert_slice_invariants(&poly, &poly_cut, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0]);
}

#[test]
fn degenerate_planes_through_vertices_and_faces_produce_watertight_non_duplicate_slices() {
    let dir = vtk_dir();

    // 1. Plane z = 1.0 passes directly through the 9 mid-plane vertices and 4 shared internal
    // hex faces of the 2x2x2 `hex_linear.vtu` mesh on [0, 2]^3.
    // The tie-breaking rule (`val >= iso` is above) must emit the z = 1.0 cross-section once
    // with exact area 4.0 and no duplicate triangles.
    let hex_lin = read_vtu_from_path(dir.join("hex_linear.vtu")).unwrap();
    let cut_midplane = slice_grid(
        &hex_lin,
        &SliceParams::with_origin([1.0, 1.0, 1.0], [0.0, 0.0, 1.0]),
        LinearizeParams::default(),
    )
    .unwrap();
    assert!((cut_midplane.total_area() - 4.0).abs() < 1e-12);
    assert_slice_invariants(&hex_lin, &cut_midplane, [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]);

    // 2. Diagonal plane x - y = 0 through the unit cube `tet_linear.vtu` passes directly through
    // 4 cube corners and the shared diagonal faces of the Kuhn tetrahedra; area must be sqrt(2).
    let tet_lin = read_vtu_from_path(dir.join("tet_linear.vtu")).unwrap();
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let cut_diag = slice_grid(
        &tet_lin,
        &SliceParams::with_origin([0.5, 0.5, 0.5], [inv_sqrt2, -inv_sqrt2, 0.0]),
        LinearizeParams::default(),
    )
    .unwrap();
    assert!((cut_diag.total_area() - 2.0_f64.sqrt()).abs() < 1e-12);
    assert_slice_invariants(
        &tet_lin,
        &cut_diag,
        [0.5, 0.5, 0.5],
        [inv_sqrt2, -inv_sqrt2, 0.0],
    );
}

#[test]
fn plane_outside_bounds_produces_empty_surface_and_invalid_params_return_errors() {
    let dir = vtk_dir();
    let tet_lin = read_vtu_from_path(dir.join("tet_linear.vtu")).unwrap();
    let view = linearize_default(&tet_lin).unwrap();

    // Plane at z = 5.0 misses [0, 1]^3 completely.
    let miss = slice(
        &view,
        &SliceParams::with_origin([0.5, 0.5, 5.0], [0.0, 0.0, 1.0]),
    )
    .unwrap();
    assert_eq!(miss.num_points(), 0);
    assert_eq!(miss.num_triangles(), 0);
    assert_eq!(miss.num_faces(), 0);
    assert_eq!(miss.num_wireframe_edges(), 0);
    assert_eq!(miss.total_area(), 0.0);
    assert!(miss.point_data().get("linear").is_some());
    assert!(miss.cell_data().get("cell_id").is_some());

    // Zero or non-finite normal is rejected.
    assert_eq!(
        slice(&view, &SliceParams::new([0.0, 0.0, 0.0])),
        Err(FilterError::InvalidPlaneNormal {
            normal: [0.0, 0.0, 0.0]
        })
    );
    assert!(matches!(
        slice(&view, &SliceParams::new([f64::NAN, 0.0, 1.0])),
        Err(FilterError::InvalidPlaneNormal { .. })
    ));
    assert!(matches!(
        slice(
            &view,
            &SliceParams::with_origin([f64::INFINITY, 0.0, 0.0], [0.0, 0.0, 1.0])
        ),
        Err(FilterError::InvalidPlaneOrigin { .. })
    ));
}

#[test]
fn slice_is_bit_for_bit_deterministic_across_chunk_sizes_and_threads() {
    let grid = SyntheticGridSpec::new(10, 8, 6, SyntheticCellLayout::Hexahedra)
        .generate()
        .unwrap();
    let view = linearize_default(&grid).unwrap();
    let params = SliceParams::with_origin([0.5, 0.5, 0.5], [0.3, -0.5, 0.8]);

    let single_chunk = slice_with_chunk_size(&view, &params, usize::MAX).unwrap();
    let chunk_64 = slice_with_chunk_size(&view, &params, 64).unwrap();
    let chunk_7 = slice_with_chunk_size(&view, &params, 7).unwrap();
    let chunk_1 = slice_with_chunk_size(&view, &params, 1).unwrap();

    assert!(single_chunk.num_triangles() > 100);
    assert_eq!(single_chunk, chunk_64);
    assert_eq!(single_chunk, chunk_7);
    assert_eq!(single_chunk, chunk_1);
}

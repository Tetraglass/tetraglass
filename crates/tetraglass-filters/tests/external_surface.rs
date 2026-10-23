//! Integration tests for `extract_external_surface` across linear, higher-order, and polyhedral meshes.
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;
use std::sync::Arc;

use tetraglass_core::{ArrayBuffer, UnstructuredGrid};
use tetraglass_filters::extract_external_surface;
use tetraglass_io::{read_vtu, read_vtu_from_path, write_vtu_to_bytes};

fn vtk_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ directory")
        .parent()
        .expect("workspace root")
        .join("testdata")
        .join("vtk")
}

#[test]
fn hex_linear_has_analytic_external_face_count_and_area() {
    let grid = read_vtu_from_path(vtk_dir().join("hex_linear.vtu")).unwrap();
    let surface = extract_external_surface(&grid).unwrap();

    // 2x2x2 block of 8 hexes on [0, 2]^3:
    // 8 * 6 = 48 cell quad faces, 12 shared internal quad faces (24 cell faces),
    // leaving 24 external quad faces = 48 triangles, with total area 6 * (2 * 2) = 24.0.
    assert_eq!(surface.num_faces(), 24);
    assert_eq!(surface.num_triangles(), 48);
    assert!((surface.total_area() - 24.0).abs() < 1e-12);

    // Every triangle's vertices must lie on the outer boundary of [0, 2]^3
    // (at least one coordinate x, y, or z is 0.0 or 2.0 on all 3 vertices of the triangle).
    let pts = surface.points();
    for &[i0, i1, i2] in surface.triangles() {
        let p0 = pts[i0 as usize];
        let p1 = pts[i1 as usize];
        let p2 = pts[i2 as usize];
        let on_outer_plane = (0..3).any(|d| {
            (p0[d].abs() < 1e-12 && p1[d].abs() < 1e-12 && p2[d].abs() < 1e-12)
                || ((p0[d] - 2.0).abs() < 1e-12
                    && (p1[d] - 2.0).abs() < 1e-12
                    && (p2[d] - 2.0).abs() < 1e-12)
        });
        assert!(
            on_outer_plane,
            "internal face leaked into hex_linear surface"
        );
    }
}

#[test]
fn mixed_linear_has_no_internal_faces_and_maps_cell_data() {
    let grid = read_vtu_from_path(vtk_dir().join("mixed_linear.vtu")).unwrap();
    let surface = extract_external_surface(&grid).unwrap();

    // 1 hex (6 quads) + 1 wedge (3 quads, 2 tris) + 1 pyramid (1 quad, 4 tris) + 1 tet (4 tris)
    // = 10 quads + 10 tris = 20 cell faces;
    // 4 shared internal faces (2 shared quads = 4 cell quads; 2 shared tris = 4 cell tris) ->
    // 6 external quads + 6 external tris = 12 external faces = 6 * 2 + 6 = 18 triangles.
    assert_eq!(surface.num_faces(), 12);
    assert_eq!(surface.num_triangles(), 18);

    // Verify none of the 4 internal shared faces are present:
    // - hex/wedge shared quad at x = 1, z in [0, 1]: vertices {1, 2, 5, 6}
    // - hex/pyramid shared quad at z = 1, x,y in [0, 1]: vertices {4, 5, 6, 7}
    // - wedge/tet shared triangle at z = 1: vertices {5, 9, 6}
    // - pyramid/tet shared triangle: vertices {5, 6, 10}
    let internal_sets: [&[u32]; 4] = [&[1, 2, 5, 6], &[4, 5, 6, 7], &[5, 6, 9], &[5, 6, 10]];
    for &[a, b, c] in surface.triangles() {
        for internal in internal_sets {
            let all_in_internal =
                internal.contains(&a) && internal.contains(&b) && internal.contains(&c);
            assert!(
                !all_in_internal,
                "triangle [{a}, {b}, {c}] belongs to an internal shared face {internal:?}"
            );
        }
    }

    // Verify point arrays are shared via Arc without copying
    match (
        grid.point_data().get("linear").unwrap().buffer(),
        surface.point_data().get("linear").unwrap().buffer(),
    ) {
        (ArrayBuffer::F64(a), ArrayBuffer::F64(b)) => assert!(Arc::ptr_eq(a, b)),
        _ => panic!("expected F64 buffer"),
    }

    // Verify cell_id is mapped accurately through triangle_source_cells
    let surf_cell_id = surface.cell_data().get("cell_id").unwrap();
    assert_eq!(surf_cell_id.num_tuples(), surface.num_triangles());
    for (tri_idx, &src_cell) in surface.triangle_source_cells().iter().enumerate() {
        assert_eq!(
            surf_cell_id.value_as_f64(tri_idx, 0),
            Some(f64::from(src_cell))
        );
    }
}

#[test]
fn covers_all_cell_families_in_testdata() {
    let dir = vtk_dir();

    // 1. Linear tetrahedra on [0, 1]^3: 12 outer triangles, area = 6.0
    let tet_lin = read_vtu_from_path(dir.join("tet_linear.vtu")).unwrap();
    let surf_tet_lin = extract_external_surface(&tet_lin).unwrap();
    assert_eq!(surf_tet_lin.num_faces(), 12);
    assert_eq!(surf_tet_lin.num_triangles(), 12);
    assert!((surf_tet_lin.total_area() - 6.0).abs() < 1e-12);

    // 2. Quadratic tetrahedra on [0, 1]^3 (corner-node stopgap): 12 outer triangles, area = 6.0
    let tet_quad = read_vtu_from_path(dir.join("tet_quadratic.vtu")).unwrap();
    let surf_tet_quad = extract_external_surface(&tet_quad).unwrap();
    assert_eq!(surf_tet_quad.num_faces(), 12);
    assert_eq!(surf_tet_quad.num_triangles(), 12);
    assert!((surf_tet_quad.total_area() - 6.0).abs() < 1e-12);

    // 3. Third-order Lagrange hexahedra on [0, 2] x [0, 1]^2 (corner-node stopgap):
    // 2 hexes sharing 1 internal face -> 10 external quad faces = 20 triangles, area = 10.0
    let hex_lag = read_vtu_from_path(dir.join("hex_lagrange_p3.vtu")).unwrap();
    let surf_hex_lag = extract_external_surface(&hex_lag).unwrap();
    assert_eq!(surf_hex_lag.num_faces(), 10);
    assert_eq!(surf_hex_lag.num_triangles(), 20);
    assert!((surf_hex_lag.total_area() - 10.0).abs() < 1e-12);

    // 4. Polyhedra: 6-face cube + 7-face beveled prism sharing 1 quad face ->
    // 11 external polygon faces (9 quads + 2 pentagons = 18 + 6 = 24 triangles).
    let poly = read_vtu_from_path(dir.join("polyhedra.vtu")).unwrap();
    let surf_poly = extract_external_surface(&poly).unwrap();
    assert_eq!(surf_poly.num_faces(), 11);
    assert_eq!(surf_poly.num_triangles(), 24);

    // Round-trip surface mesh through UnstructuredGrid and VTU writer
    let surf_grid = surf_poly.to_unstructured_grid().unwrap();
    let vtu_bytes = write_vtu_to_bytes(&surf_grid).unwrap();
    let reloaded = read_vtu(&vtu_bytes).unwrap();
    assert_eq!(surf_grid, reloaded);
}

#[test]
fn empty_grid_produces_empty_surface() {
    let empty =
        UnstructuredGrid::new(Vec::<[f64; 3]>::new(), vec![], vec![], vec![], None).unwrap();
    let surf = extract_external_surface(&empty).unwrap();
    assert_eq!(surf.num_points(), 0);
    assert_eq!(surf.num_faces(), 0);
    assert_eq!(surf.num_triangles(), 0);
    assert_eq!(surf.total_area(), 0.0);
}

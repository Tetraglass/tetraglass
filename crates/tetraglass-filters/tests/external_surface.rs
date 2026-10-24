//! Integration tests for `extract_external_surface` across linear, higher-order, and polyhedral meshes.
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;
use std::sync::Arc;

use tetraglass_core::{
    ArrayBuffer, CellType, ComponentCount, DataArray, SyntheticCellLayout, SyntheticGridSpec,
    UnstructuredGrid,
};
use tetraglass_filters::{extract_external_surface, extract_external_surface_with_subdivisions};
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
    // The 12 external faces have 11 vertices and (by Euler V - E + F = 2) 11 + 12 - 2 = 21
    // original element edges (whereas all 18 triangles have 27 edges including the 6 quad diagonals).
    assert_eq!(surface.num_faces(), 12);
    assert_eq!(surface.num_triangles(), 18);
    assert_eq!(surface.num_wireframe_edges(), 21);

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

    // 2. Quadratic tetrahedra on [0, 1]^3:
    // 12 outer triangle faces, each tessellated into S^2 = 4 sub-triangles at default S = 2
    // (48 triangles total, area = 6.0), and 18 original outer element edges * 2 = 36 wireframe segments.
    let tet_quad = read_vtu_from_path(dir.join("tet_quadratic.vtu")).unwrap();
    let surf_tet_quad = extract_external_surface(&tet_quad).unwrap();
    assert_eq!(surf_tet_quad.num_faces(), 12);
    assert_eq!(surf_tet_quad.num_triangles(), 48);
    assert_eq!(surf_tet_quad.num_wireframe_edges(), 36);
    assert!((surf_tet_quad.total_area() - 6.0).abs() < 1e-12);

    // At S = 1, higher-order faces use only corner nodes (12 triangles, 18 wireframe edges).
    let surf_tet_quad_s1 = extract_external_surface_with_subdivisions(&tet_quad, 1).unwrap();
    assert_eq!(surf_tet_quad_s1.num_faces(), 12);
    assert_eq!(surf_tet_quad_s1.num_triangles(), 12);
    assert_eq!(surf_tet_quad_s1.num_wireframe_edges(), 18);

    // 3. Third-order Lagrange hexahedra on [0, 2] x [0, 1]^2:
    // 2 hexes sharing 1 internal face -> 10 external quad faces.
    // At default S = 2: each quad face is tessellated into 2 * S^2 = 8 sub-triangles (80 triangles),
    // and the 20 original outer element edges are split into 20 * 2 = 40 wireframe segments.
    let hex_lag = read_vtu_from_path(dir.join("hex_lagrange_p3.vtu")).unwrap();
    let surf_hex_lag = extract_external_surface(&hex_lag).unwrap();
    assert_eq!(surf_hex_lag.num_faces(), 10);
    assert_eq!(surf_hex_lag.num_triangles(), 80);
    assert_eq!(surf_hex_lag.num_wireframe_edges(), 40);
    assert!((surf_hex_lag.total_area() - 10.0).abs() < 1e-12);

    // At S = 3 (matching order = 3): 10 * 2 * 9 = 180 triangles, 20 * 3 = 60 wireframe segments,
    // and every face lattice node coincides with an existing VTK p3 node (zero new points allocated).
    let surf_hex_lag_s3 = extract_external_surface_with_subdivisions(&hex_lag, 3).unwrap();
    assert_eq!(surf_hex_lag_s3.num_faces(), 10);
    assert_eq!(surf_hex_lag_s3.num_triangles(), 180);
    assert_eq!(surf_hex_lag_s3.num_wireframe_edges(), 60);
    assert_eq!(surf_hex_lag_s3.num_points(), hex_lag.num_points());

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

#[test]
fn synthetic_tet_and_hex_grids_have_exact_boundary_faces_and_zero_internal_faces() {
    let tet_spec = SyntheticGridSpec::new(10, 8, 6, SyntheticCellLayout::Tetrahedra6);
    let tet_grid = tet_spec.generate().expect("generate 10x8x6 tets");
    let tet_surf = extract_external_surface(&tet_grid).expect("extract tet surface");

    assert_eq!(tet_surf.num_faces(), tet_spec.expected_boundary_faces());
    assert_eq!(tet_surf.num_triangles(), tet_spec.expected_boundary_faces());
    assert!((tet_surf.total_area() - 6.0).abs() < 1e-11);

    let hex_spec = SyntheticGridSpec::new(10, 8, 6, SyntheticCellLayout::Hexahedra);
    let hex_grid = hex_spec.generate().expect("generate 10x8x6 hexes");
    let hex_surf = extract_external_surface(&hex_grid).expect("extract hex surface");

    assert_eq!(hex_surf.num_faces(), hex_spec.expected_boundary_faces());
    assert_eq!(
        hex_surf.num_triangles(),
        hex_spec.expected_boundary_faces() * 2
    );
    assert!((hex_surf.total_area() - 6.0).abs() < 1e-11);
}

#[test]
fn curved_higher_order_surface_converges_and_excludes_internal_wireframe_edges() {
    // Build a 10-node QuadraticTetra whose mid-edge node 4 on edge (0 -> 1) is bowed outward
    // in -y by dy = -0.35, curving the bottom and front triangle faces.
    let mut points = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [0.0, 1.0, 0.0], // 2
        [0.0, 0.0, 1.0], // 3
        [0.5, 0.0, 0.0], // 4: (0,1)
        [0.5, 0.5, 0.0], // 5: (1,2)
        [0.0, 0.5, 0.0], // 6: (2,0)
        [0.0, 0.0, 0.5], // 7: (0,3)
        [0.5, 0.0, 0.5], // 8: (1,3)
        [0.0, 0.5, 0.5], // 9: (2,3)
    ];
    let quad_vals: Vec<f64> = points
        .iter()
        .map(|&[r, s, t]| 1.0 + 2.0 * r * r - 1.5 * r * s + 3.0 * s * t + 0.5 * t * t)
        .collect();
    points[4][1] = -0.35;

    let mut grid = UnstructuredGrid::new(
        points,
        vec![CellType::QuadraticTetra],
        vec![10],
        (0..10).collect(),
        None,
    )
    .unwrap();
    grid.insert_point_data(
        "q",
        DataArray::from_f64(ComponentCount::Scalar, quad_vals).unwrap(),
    )
    .unwrap();

    let s1 = extract_external_surface_with_subdivisions(&grid, 1).unwrap();
    let s2 = extract_external_surface_with_subdivisions(&grid, 2).unwrap();
    let s4 = extract_external_surface_with_subdivisions(&grid, 4).unwrap();

    // 4 logical outer faces, subdivided into 4 * S^2 triangles;
    // 6 original tetrahedron edges, segmented into 6 * S wireframe segments (no interior edges!).
    assert_eq!(s1.num_faces(), 4);
    assert_eq!(s1.num_triangles(), 4);
    assert_eq!(s1.num_wireframe_edges(), 6);

    assert_eq!(s2.num_faces(), 4);
    assert_eq!(s2.num_triangles(), 16);
    assert_eq!(s2.num_wireframe_edges(), 12);

    assert_eq!(s4.num_faces(), 4);
    assert_eq!(s4.num_triangles(), 64);
    assert_eq!(s4.num_wireframe_edges(), 24);

    // Curved surface area strictly increases from S=1 (flat corners) -> S=2 -> S=4 as the
    // parabolic boundary curve is resolved.
    let area1 = s1.total_area();
    let area2 = s2.total_area();
    let area4 = s4.total_area();
    assert!(
        area2 > area1 + 0.15,
        "S=2 surface area ({area2}) must exceed S=1 flat-corner area ({area1})"
    );
    assert!(
        area4 > area2 + 0.01,
        "S=4 surface area ({area4}) must refine S=2 area ({area2})"
    );
}

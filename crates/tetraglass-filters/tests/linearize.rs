//! Integration tests for `linearize` across linear, mixed, polyhedral, and synthetic meshes (`M2.1a`).
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;
use std::sync::Arc;

use tetraglass_core::{ArrayBuffer, SyntheticCellLayout, SyntheticGridSpec, UnstructuredGrid};
use tetraglass_filters::{
    FilterError, LinearizeParams, extract_external_surface, linearize, linearize_default,
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

#[test]
fn tet_linear_preserves_tets_volume_and_arc_buffers() {
    let grid = read_vtu_from_path(vtk_dir().join("tet_linear.vtu")).unwrap();
    let view = linearize_default(&grid).unwrap();

    assert_eq!(view.num_points(), 8);
    assert_eq!(view.num_tets(), 6);
    assert!((view.total_volume() - 1.0).abs() < 1e-12);
    for tet_idx in 0..view.num_tets() {
        assert!((view.tet_signed_volume(tet_idx).unwrap() - 1.0 / 6.0).abs() < 1e-12);
        assert_eq!(view.tet_source_cells()[tet_idx], tet_idx as u32);
    }

    // Points and point_data buffers are shared via Arc with zero copying.
    assert!(Arc::ptr_eq(&grid.points_arc(), &view.points_arc()));
    match (
        grid.point_data().get("linear").unwrap().buffer(),
        view.point_data().get("linear").unwrap().buffer(),
    ) {
        (ArrayBuffer::F64(a), ArrayBuffer::F64(b)) => assert!(Arc::ptr_eq(a, b)),
        _ => panic!("expected F64 buffer"),
    }
}

#[test]
fn hex_linear_preserves_volume_and_has_conformal_internal_faces() {
    let grid = read_vtu_from_path(vtk_dir().join("hex_linear.vtu")).unwrap();
    let view = linearize_default(&grid).unwrap();

    // 8 unit-size hexes on [0, 2]^3 -> 48 tets, total volume = 8.0.
    assert_eq!(view.num_points(), 27);
    assert_eq!(view.num_tets(), 48);
    assert!((view.total_volume() - 8.0).abs() < 1e-12);
    for tet_idx in 0..view.num_tets() {
        assert!(view.tet_signed_volume(tet_idx).unwrap() > 0.0);
    }

    // Because shared quad faces use the minimum-global-vertex diagonal rule,
    // extracting the external surface of the linearized tet mesh must produce
    // the exact 48 boundary triangles (area 24.0) with zero leaked internal triangles.
    let tet_grid = view.to_unstructured_grid().unwrap();
    let tet_surf = extract_external_surface(&tet_grid).unwrap();
    assert_eq!(tet_surf.num_faces(), 48);
    assert_eq!(tet_surf.num_triangles(), 48);
    assert!((tet_surf.total_area() - 24.0).abs() < 1e-12);
}

#[test]
fn mixed_linear_preserves_per_cell_volumes_and_is_watertight_across_cell_types() {
    let grid = read_vtu_from_path(vtk_dir().join("mixed_linear.vtu")).unwrap();
    let orig_surf = extract_external_surface(&grid).unwrap();
    let view = linearize_default(&grid).unwrap();

    // 1 hex (6) + 1 wedge (3) + 1 pyramid (2) + 1 tet (1) = 12 tets.
    assert_eq!(view.num_points(), 11);
    assert_eq!(view.num_tets(), 12);
    assert!((view.total_volume() - 2.0).abs() < 1e-12);

    let mut cell_volumes = [0.0_f64; 4];
    let cell_id_arr = view.cell_data().get("cell_id").unwrap();
    for tet_idx in 0..view.num_tets() {
        let signed_vol = view.tet_signed_volume(tet_idx).unwrap();
        assert!(
            signed_vol > 0.0,
            "tet {tet_idx} has non-positive signed volume {signed_vol}"
        );
        let src = view.tet_source_cells()[tet_idx] as usize;
        cell_volumes[src] += signed_vol;
        assert_eq!(cell_id_arr.value_as_f64(tet_idx, 0), Some(src as f64));
    }

    assert!((cell_volumes[0] - 1.0).abs() < 1e-12); // hex
    assert!((cell_volumes[1] - 0.5).abs() < 1e-12); // wedge
    assert!((cell_volumes[2] - 1.0 / 3.0).abs() < 1e-12); // pyramid
    assert!((cell_volumes[3] - 1.0 / 6.0).abs() < 1e-12); // tet

    // Conformal diagonal check across hex-wedge, hex-pyramid, wedge-tet, and pyramid-tet faces:
    // the external surface of the 12 linearized tets must have 18 outer triangles and the exact
    // same surface area as the original mixed mesh.
    let tet_grid = view.to_unstructured_grid().unwrap();
    let tet_surf = extract_external_surface(&tet_grid).unwrap();
    assert_eq!(tet_surf.num_faces(), 18);
    assert_eq!(tet_surf.num_triangles(), 18);
    assert!((tet_surf.total_area() - orig_surf.total_area()).abs() < 1e-12);
}

#[test]
fn polyhedra_centroid_decomposition_preserves_volume_and_linear_fields_exactly() {
    let grid = read_vtu_from_path(vtk_dir().join("polyhedra.vtu")).unwrap();
    let orig_surf = extract_external_surface(&grid).unwrap();
    let view = linearize_default(&grid).unwrap();

    // 14 original points + 2 polyhedron centroids = 16 points.
    // Cell 0 (cube, 6 quads = 12 tris) + Cell 1 (5 quads + 2 pentagons = 10 + 6 = 16 tris) = 28 tets.
    assert_eq!(view.num_points(), 16);
    assert_eq!(view.num_tets(), 28);
    assert!((view.total_volume() - 1.875).abs() < 1e-12);

    let mut cell_volumes = [0.0_f64; 2];
    for tet_idx in 0..view.num_tets() {
        let signed_vol = view.tet_signed_volume(tet_idx).unwrap();
        assert!(signed_vol > 0.0);
        let src = view.tet_source_cells()[tet_idx] as usize;
        cell_volumes[src] += signed_vol;
    }
    assert!((cell_volumes[0] - 1.0).abs() < 1e-12);
    assert!((cell_volumes[1] - 0.875).abs() < 1e-12);

    // Every point (including the 2 added centroid points) must reproduce linear fields
    // `linear = x + 2y + 3z` and `velocity = (-y, x, 0)` exactly.
    let linear_arr = view.point_data().get("linear").unwrap();
    let vel_arr = view.point_data().get("velocity").unwrap();
    for (pt_idx, &[x, y, z]) in view.points().iter().enumerate() {
        let expected_linear = x + 2.0 * y + 3.0 * z;
        let actual_linear = linear_arr.value_as_f64(pt_idx, 0).unwrap();
        assert!(
            (actual_linear - expected_linear).abs() < 1e-12,
            "point {pt_idx} linear mismatch: expected {expected_linear}, got {actual_linear}"
        );

        let vx = vel_arr.value_as_f64(pt_idx, 0).unwrap();
        let vy = vel_arr.value_as_f64(pt_idx, 1).unwrap();
        let vz = vel_arr.value_as_f64(pt_idx, 2).unwrap();
        assert!((vx - (-y)).abs() < 1e-12);
        assert!((vy - x).abs() < 1e-12);
        assert!(vz.abs() < 1e-12);
    }

    // Shared quad face [4, 5, 6, 7] between the two polyhedra is fan-triangulated from minimum
    // vertex 4 in both cells, so the linearized tet mesh has 24 outer triangles and 0 internal ones.
    let tet_grid = view.to_unstructured_grid().unwrap();
    let tet_surf = extract_external_surface(&tet_grid).unwrap();
    assert_eq!(tet_surf.num_triangles(), 24);
    assert!((tet_surf.total_area() - orig_surf.total_area()).abs() < 1e-12);
}

#[test]
fn synthetic_hex_and_tet_grids_linearize_with_exact_volume_and_watertight_faces() {
    let hex_spec = SyntheticGridSpec::new(10, 8, 6, SyntheticCellLayout::Hexahedra);
    let hex_grid = hex_spec.generate().unwrap();
    let hex_view = linearize_default(&hex_grid).unwrap();

    assert_eq!(hex_view.num_tets(), hex_spec.num_cells() * 6);
    assert!((hex_view.total_volume() - 1.0).abs() < 1e-11);

    let tet_grid = hex_view.to_unstructured_grid().unwrap();
    let tet_surf = extract_external_surface(&tet_grid).unwrap();
    assert_eq!(
        tet_surf.num_triangles(),
        hex_spec.expected_boundary_faces() * 2
    );
    assert!((tet_surf.total_area() - 6.0).abs() < 1e-11);
}

#[test]
fn empty_grid_and_invalid_subdivisions_are_handled() {
    let empty =
        UnstructuredGrid::new(Vec::<[f64; 3]>::new(), vec![], vec![], vec![], None).unwrap();
    let view = linearize_default(&empty).unwrap();
    assert_eq!(view.num_points(), 0);
    assert_eq!(view.num_tets(), 0);
    assert_eq!(view.total_volume(), 0.0);

    assert_eq!(
        linearize(&empty, &LinearizeParams::new(0)),
        Err(FilterError::InvalidSubdivisionLevel { subdivisions: 0 })
    );
}

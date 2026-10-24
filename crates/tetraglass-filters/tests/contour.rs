//! Integration tests for the multi-isovalue `contour` filter (`M2.4`).
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
use tetraglass_filters::{
    ContourComponent, ContourParams, FilterError, LinearizeParams, SliceParams, contour,
    contour_grid, contour_with_chunk_size, linearize_default, slice,
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
fn contour_of_linear_field_matches_equivalent_slice_plane_and_interpolates_all_arrays() {
    let dir = vtk_dir();
    let hex_lin = read_vtu_from_path(dir.join("hex_linear.vtu")).unwrap();
    let view = linearize_default(&hex_lin).unwrap();

    // `linear = x + 2*y + 3*z`. The isosurface `linear == 6.0` is the exact plane
    // `(p - [1, 1, 1]) . [1, 2, 3] == 0`.
    let iso_surf = contour(&view, &ContourParams::new("linear", vec![6.0])).unwrap();
    let plane_surf = slice(
        &view,
        &SliceParams::with_origin([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),
    )
    .unwrap();

    assert_eq!(iso_surf.num_points(), plane_surf.num_points());
    assert_eq!(iso_surf.num_triangles(), plane_surf.num_triangles());
    assert_eq!(iso_surf.num_faces(), plane_surf.num_faces());
    assert!((iso_surf.total_area() - plane_surf.total_area()).abs() < 1e-12);

    // Every vertex on the contour must have interpolated `linear == 6.0` and `velocity == [-y, x, 0]`.
    let lin_arr = iso_surf.point_data().get("linear").unwrap();
    let vel_arr = iso_surf.point_data().get("velocity").unwrap();
    for (idx, &[x, y, z]) in iso_surf.points().iter().enumerate() {
        assert!(((x + 2.0 * y + 3.0 * z) - 6.0).abs() < 1e-11);
        assert!((lin_arr.value_as_f64(idx, 0).unwrap() - 6.0).abs() < 1e-11);
        assert!((vel_arr.value_as_f64(idx, 0).unwrap() - (-y)).abs() < 1e-11);
        assert!((vel_arr.value_as_f64(idx, 1).unwrap() - x).abs() < 1e-11);
        assert!(vel_arr.value_as_f64(idx, 2).unwrap().abs() < 1e-11);
    }
}

#[test]
fn multi_isovalue_contour_combines_disjoint_sheets_and_covers_all_cell_families() {
    let dir = vtk_dir();
    let hex_lin = read_vtu_from_path(dir.join("hex_linear.vtu")).unwrap();
    let view = linearize_default(&hex_lin).unwrap();

    let iso_3 = contour(&view, &ContourParams::new("linear", vec![3.0])).unwrap();
    let iso_6 = contour(&view, &ContourParams::new("linear", vec![6.0])).unwrap();
    let iso_9 = contour(&view, &ContourParams::new("linear", vec![9.0])).unwrap();
    let combined = contour(&view, &ContourParams::new("linear", vec![3.0, 6.0, 9.0])).unwrap();

    assert_eq!(
        combined.num_points(),
        iso_3.num_points() + iso_6.num_points() + iso_9.num_points()
    );
    assert_eq!(
        combined.num_triangles(),
        iso_3.num_triangles() + iso_6.num_triangles() + iso_9.num_triangles()
    );
    assert!(
        (combined.total_area() - (iso_3.total_area() + iso_6.total_area() + iso_9.total_area()))
            .abs()
            < 1e-12
    );

    // Every vertex in the combined mesh must lie on one of the three isovalues {3.0, 6.0, 9.0}.
    let lin_arr = combined.point_data().get("linear").unwrap();
    for idx in 0..combined.num_points() {
        let val = lin_arr.value_as_f64(idx, 0).unwrap();
        let min_err = (val - 3.0)
            .abs()
            .min((val - 6.0).abs())
            .min((val - 9.0).abs());
        assert!(min_err < 1e-11, "vertex {idx} has linear = {val}");
    }

    // Also test vector component (`velocity:1` = `x`) and magnitude (`velocity:magnitude` = sqrt(x^2 + y^2)):
    // On [0, 2]^3, `velocity:1` (`x`) == 1.25 is the plane x = 1.25 with area 2 * 2 = 4.0.
    let vx_iso = contour(
        &view,
        &ContourParams::with_component("velocity", ContourComponent::Component(1), vec![1.25]),
    )
    .unwrap();
    assert!((vx_iso.total_area() - 4.0).abs() < 1e-12);

    let vmag_iso = contour(&view, &ContourParams::magnitude("velocity", vec![1.5])).unwrap();
    assert!(vmag_iso.total_area() > 2.0);

    // Cover all remaining cell families in `testdata/vtk/*.vtu`:
    for (file, isovalues) in [
        ("tet_linear.vtu", vec![2.0, 4.0]),
        ("mixed_linear.vtu", vec![2.0, 4.5]),
        ("tet_quadratic.vtu", vec![2.0, 4.0]),
        ("hex_lagrange_p3.vtu", vec![2.5, 4.5]),
        ("polyhedra.vtu", vec![2.5, 4.5]),
    ] {
        let grid = read_vtu_from_path(dir.join(file)).unwrap();
        let surf = contour_grid(
            &grid,
            &ContourParams::new("linear", isovalues.clone()),
            LinearizeParams::default(),
        )
        .unwrap();
        assert!(
            surf.num_triangles() > 0 && surf.total_area() > 0.0,
            "{file} produced empty contour for {isovalues:?}"
        );
        let cell_id = surf.cell_data().get("cell_id").unwrap();
        assert_eq!(cell_id.num_tuples(), surf.num_triangles());
    }
}

#[test]
fn out_of_range_contour_is_empty_and_invalid_parameters_return_errors() {
    let dir = vtk_dir();
    let tet_lin = read_vtu_from_path(dir.join("tet_linear.vtu")).unwrap();
    let view = linearize_default(&tet_lin).unwrap();

    // `linear` on [0, 1]^3 spans [0, 6]; isovalue 50.0 produces an empty SurfaceMesh.
    let miss = contour(&view, &ContourParams::new("linear", vec![50.0])).unwrap();
    assert_eq!(miss.num_points(), 0);
    assert_eq!(miss.num_triangles(), 0);
    assert_eq!(miss.total_area(), 0.0);

    assert_eq!(
        contour(&view, &ContourParams::new("linear", vec![])),
        Err(FilterError::EmptyContourValues)
    );
    assert!(matches!(
        contour(&view, &ContourParams::new("linear", vec![1.0, f64::NAN])),
        Err(FilterError::InvalidContourValue { .. })
    ));
    assert_eq!(
        contour(&view, &ContourParams::new("nonexistent", vec![1.0])),
        Err(FilterError::MissingPointArray {
            name: "nonexistent".to_string()
        })
    );
    assert_eq!(
        contour(
            &view,
            &ContourParams::with_component("linear", ContourComponent::Component(2), vec![1.0])
        ),
        Err(FilterError::InvalidComponentIndex {
            array: "linear".to_string(),
            component: 2,
            num_components: 1
        })
    );
}

#[test]
fn contour_is_bit_for_bit_deterministic_across_chunk_sizes() {
    let grid = SyntheticGridSpec::new(10, 8, 6, SyntheticCellLayout::Hexahedra)
        .generate()
        .unwrap();
    let view = linearize_default(&grid).unwrap();
    let params = ContourParams::new("linear", vec![1.5, 3.0, 4.5]);

    let single = contour_with_chunk_size(&view, &params, usize::MAX).unwrap();
    let chunk_64 = contour_with_chunk_size(&view, &params, 64).unwrap();
    let chunk_5 = contour_with_chunk_size(&view, &params, 5).unwrap();
    assert!(single.num_triangles() > 100);
    assert_eq!(single, chunk_64);
    assert_eq!(single, chunk_5);
}

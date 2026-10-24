//! Integration tests for the volume `threshold` filter (`M2.6`).
#![cfg(not(target_arch = "wasm32"))]

use std::collections::HashMap;
use std::path::PathBuf;

use tetraglass_core::{
    ComponentCount, DataArray, SyntheticCellLayout, SyntheticGridSpec, TetView, UnstructuredGrid,
};
use tetraglass_filters::{
    FilterError, LinearizeParams, ThresholdAssociation, ThresholdComponent, ThresholdParams,
    extract_tet_view_surface, linearize_default, threshold, threshold_grid,
    threshold_with_chunk_size,
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

fn assert_threshold_invariants(grid: &UnstructuredGrid, kept: &TetView) {
    let pts = kept.points();

    // 1. Every tetrahedron must have strictly positive signed volume.
    for tet_idx in 0..kept.num_tets() {
        let vol = kept.tet_signed_volume(tet_idx).unwrap();
        assert!(
            vol > 0.0,
            "tet {tet_idx} has non-positive signed volume {vol:e}"
        );
    }

    // 2. The boundary surface of a non-empty thresholded volume must be watertight.
    let surf = extract_tet_view_surface(kept).unwrap();
    if kept.num_tets() > 0 {
        assert!(surf.num_triangles() >= 4);
        assert!(surf.num_wireframe_edges() >= 6);
        let mut edge_counts: HashMap<(u32, u32), u32> = HashMap::new();
        for &[i0, i1, i2] in surf.triangles() {
            for (a, b) in [(i0, i1), (i1, i2), (i2, i0)] {
                let key = if a <= b { (a, b) } else { (b, a) };
                *edge_counts.entry(key).or_insert(0) += 1;
            }
        }
        for (edge, count) in edge_counts {
            assert_eq!(
                count % 2,
                0,
                "boundary surface of thresholded volume has open edge {edge:?} (count = {count})"
            );
        }
    }

    // 3. Linear scalar field `linear = x + 2*y + 3*z` and vector field `velocity = [-y, x, 0]`
    // must be preserved at every kept vertex.
    if let Some(linear_arr) = kept.point_data().get("linear") {
        assert_eq!(linear_arr.num_tuples(), kept.num_points());
        for (idx, &[x, y, z]) in pts.iter().enumerate() {
            let expected = x + 2.0 * y + 3.0 * z;
            let actual = linear_arr.value_as_f64(idx, 0).unwrap();
            assert!(
                (actual - expected).abs() < 1e-11,
                "point {idx} `linear` mismatch: {actual} vs {expected}"
            );
        }
    }
    if let Some(vel_arr) = kept.point_data().get("velocity") {
        assert_eq!(vel_arr.num_tuples(), kept.num_points());
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

    // 4. `cell_id` and `tet_source_cells` must map back to the original source cells.
    if let Some(cell_id_arr) = kept.cell_data().get("cell_id") {
        assert_eq!(cell_id_arr.num_tuples(), kept.num_tets());
        for (tet_idx, &src_cell) in kept.tet_source_cells().iter().enumerate() {
            assert!((src_cell as usize) < grid.num_cells());
            assert_eq!(
                cell_id_arr.value_as_f64(tet_idx, 0),
                Some(f64::from(src_cell))
            );
        }
    }
}

#[test]
fn threshold_covers_all_cell_families_point_and_cell_arrays_and_one_sided_bounds() {
    let dir = vtk_dir();

    // 1. Linear tetrahedra (`tet_linear.vtu`, 6 Kuhn tets on [0, 1]^3, each of volume 1/6):
    // Thresholding `cell_id` in [1.0, 3.0] keeps cells 1, 2, 3 -> 3 tets, volume = 3/6 = 0.5.
    let tet_lin = read_vtu_from_path(dir.join("tet_linear.vtu")).unwrap();
    let tet_view = linearize_default(&tet_lin).unwrap();
    let tet_kept = threshold(&tet_view, &ThresholdParams::range("cell_id", 1.0, 3.0)).unwrap();
    assert_eq!(tet_kept.num_tets(), 3);
    assert!((tet_kept.total_volume() - 0.5).abs() < 1e-12);
    assert_eq!(tet_kept.tet_source_cells(), &[1, 2, 3]);
    assert_threshold_invariants(&tet_lin, &tet_kept);

    // 2. Linear hexahedra (`hex_linear.vtu`, 2x2x2 = 8 unit hexes on [0, 2]^3, volume 8.0):
    // - Point scalar `linear = x + 2*y + 3*z`:
    //   - `at_most("linear", 7.0)` (`linear <= 7.0`): keeps hex 0 (`[0, 6]`) and hex 1 (`[1, 7]`) -> 2 whole hexes = 12 tets, volume 2.0.
    //   - `at_least("linear", 5.0)` (`linear >= 5.0`): keeps hex 6 (`[5, 11]`) and hex 7 (`[6, 12]`) -> 2 whole hexes = 12 tets, volume 2.0.
    let hex_lin = read_vtu_from_path(dir.join("hex_linear.vtu")).unwrap();
    let hex_view = linearize_default(&hex_lin).unwrap();
    let hex_low = threshold(&hex_view, &ThresholdParams::at_most("linear", 7.0)).unwrap();
    assert_eq!(hex_low.num_tets(), 12);
    assert!((hex_low.total_volume() - 2.0).abs() < 1e-12);
    assert_threshold_invariants(&hex_lin, &hex_low);

    let hex_high = threshold(&hex_view, &ThresholdParams::at_least("linear", 5.0)).unwrap();
    assert_eq!(hex_high.num_tets(), 12);
    assert!((hex_high.total_volume() - 2.0).abs() < 1e-12);
    assert_threshold_invariants(&hex_lin, &hex_high);

    // Vector component (`velocity:y = x >= 1.0`) keeps the 4 right-half hexes (`x in [1, 2]`, volume 4.0, 24 tets).
    let hex_vel_y = threshold(
        &hex_view,
        &ThresholdParams::at_least("velocity", 1.0)
            .with_component(ThresholdComponent::Component(1)),
    )
    .unwrap();
    assert_eq!(hex_vel_y.num_tets(), 24);
    assert!((hex_vel_y.total_volume() - 4.0).abs() < 1e-12);
    assert_threshold_invariants(&hex_lin, &hex_vel_y);

    // Vector magnitude (`velocity:magnitude = sqrt(x^2 + y^2) >= sqrt(2)`): only hexes with `x in [1, 2]` and `y in [1, 2]`
    // have all 8 corners at distance >= sqrt(2) from the z-axis -> 2 hexes (`z in [0, 1]` and `z in [1, 2]`), volume 2.0, 12 tets.
    let hex_vel_mag = threshold(
        &hex_view,
        &ThresholdParams::at_least("velocity", 2.0_f64.sqrt() - 1e-12)
            .with_component(ThresholdComponent::Magnitude),
    )
    .unwrap();
    assert_eq!(hex_vel_mag.num_tets(), 12);
    assert!((hex_vel_mag.total_volume() - 2.0).abs() < 1e-12);
    assert_threshold_invariants(&hex_lin, &hex_vel_mag);

    // 3. Mixed linear, Quadratic tetrahedra, 3rd-order Lagrange hexahedra, and Polyhedra:
    // Verify cell_id thresholding and point-data `linear` thresholding across every family.
    for filename in [
        "mixed_linear.vtu",
        "tet_quadratic.vtu",
        "hex_lagrange_p3.vtu",
        "polyhedra.vtu",
    ] {
        let grid = read_vtu_from_path(dir.join(filename)).unwrap();
        let view = linearize_default(&grid).unwrap();
        let total_vol = view.total_volume();

        // Keep only source cell 0 via `cell_id in [0.0, 0.0]`.
        let cell0 = threshold(&view, &ThresholdParams::range("cell_id", 0.0, 0.0)).unwrap();
        assert!(cell0.num_tets() > 0);
        assert!(cell0.total_volume() > 0.0 && cell0.total_volume() < total_vol);
        assert!(cell0.tet_source_cells().iter().all(|&c| c == 0));
        assert_threshold_invariants(&grid, &cell0);

        // Keep all cells via `linear >= -100.0`.
        let all_cells = threshold(&view, &ThresholdParams::at_least("linear", -100.0)).unwrap();
        assert_eq!(all_cells.num_tets(), view.num_tets());
        assert!((all_cells.total_volume() - total_vol).abs() < 1e-12);
    }
}

#[test]
fn point_data_wins_by_default_and_explicit_association_selects_point_or_cell_table() {
    let mut grid = SyntheticGridSpec::new(2, 1, 1, SyntheticCellLayout::Hexahedra)
        .generate()
        .unwrap();
    // Insert a same-named array `"shared"` into both point_data (all 10.0) and cell_data ([1.0, 2.0]).
    grid.insert_point_data(
        "shared",
        DataArray::from_f64(ComponentCount::Scalar, vec![10.0; grid.num_points()]).unwrap(),
    )
    .unwrap();
    grid.insert_cell_data(
        "shared",
        DataArray::from_f64(ComponentCount::Scalar, vec![1.0, 2.0]).unwrap(),
    )
    .unwrap();

    let view = linearize_default(&grid).unwrap();

    // Auto association picks point_data (value 10.0), so range [9.0, 11.0] keeps both hexes (12 tets),
    // whereas range [0.5, 1.5] is empty under Auto.
    let auto_kept = threshold(&view, &ThresholdParams::range("shared", 9.0, 11.0)).unwrap();
    assert_eq!(auto_kept.num_tets(), 12);
    let auto_empty = threshold(&view, &ThresholdParams::range("shared", 0.5, 1.5)).unwrap();
    assert_eq!(auto_empty.num_tets(), 0);

    // Explicit `ThresholdAssociation::Cell` picks cell_data ([1.0, 2.0]), keeping only cell 0 (6 tets) for [0.5, 1.5].
    let cell_kept = threshold(
        &view,
        &ThresholdParams::range("shared", 0.5, 1.5).with_association(ThresholdAssociation::Cell),
    )
    .unwrap();
    assert_eq!(cell_kept.num_tets(), 6);
    assert!(cell_kept.tet_source_cells().iter().all(|&c| c == 0));
}

#[test]
fn empty_result_and_invalid_threshold_parameters_return_expected_errors() {
    let dir = vtk_dir();
    let tet_lin = read_vtu_from_path(dir.join("tet_linear.vtu")).unwrap();
    let view = linearize_default(&tet_lin).unwrap();

    // Out-of-range bounds produce an empty TetView with preserved attribute schemas.
    let empty = threshold(&view, &ThresholdParams::at_least("linear", 100.0)).unwrap();
    assert_eq!(empty.num_points(), 0);
    assert_eq!(empty.num_tets(), 0);
    assert_eq!(empty.total_volume(), 0.0);
    assert!(empty.point_data().get("linear").is_some());
    assert!(empty.cell_data().get("cell_id").is_some());

    let grid_out = threshold_grid(
        &tet_lin,
        &ThresholdParams::range("cell_id", 0.0, 1.0),
        LinearizeParams::default(),
    )
    .unwrap();
    assert_eq!(grid_out.num_cells(), 2);

    // Invalid bounds: both None, min > max, or non-finite.
    assert_eq!(
        threshold(&view, &ThresholdParams::new("linear", None, None)),
        Err(FilterError::InvalidThresholdBounds {
            min: None,
            max: None
        })
    );
    assert_eq!(
        threshold(&view, &ThresholdParams::range("linear", 5.0, 2.0)),
        Err(FilterError::InvalidThresholdBounds {
            min: Some(5.0),
            max: Some(2.0)
        })
    );
    assert!(matches!(
        threshold(&view, &ThresholdParams::at_least("linear", f64::NAN)),
        Err(FilterError::InvalidThresholdBounds { .. })
    ));

    // Missing arrays and invalid component index.
    assert_eq!(
        threshold(&view, &ThresholdParams::at_least("nonexistent", 0.0)),
        Err(FilterError::MissingAttributeArray {
            name: "nonexistent".to_string()
        })
    );
    assert_eq!(
        threshold(
            &view,
            &ThresholdParams::at_least("cell_id", 0.0)
                .with_association(ThresholdAssociation::Point)
        ),
        Err(FilterError::MissingPointArray {
            name: "cell_id".to_string()
        })
    );
    assert_eq!(
        threshold(
            &view,
            &ThresholdParams::at_least("linear", 0.0).with_association(ThresholdAssociation::Cell)
        ),
        Err(FilterError::MissingCellArray {
            name: "linear".to_string()
        })
    );
    assert_eq!(
        threshold(
            &view,
            &ThresholdParams::at_least("linear", 0.0)
                .with_component(ThresholdComponent::Component(1))
        ),
        Err(FilterError::InvalidComponentIndex {
            array: "linear".to_string(),
            component: 1,
            num_components: 1
        })
    );
}

#[test]
fn threshold_is_bit_for_bit_deterministic_across_chunk_sizes() {
    let grid = SyntheticGridSpec::new(10, 8, 6, SyntheticCellLayout::Hexahedra)
        .generate()
        .unwrap();
    let view = linearize_default(&grid).unwrap();
    let params = ThresholdParams::range("linear", 1.5, 4.5);

    let single_chunk = threshold_with_chunk_size(&view, &params, usize::MAX).unwrap();
    let chunk_64 = threshold_with_chunk_size(&view, &params, 64).unwrap();
    let chunk_7 = threshold_with_chunk_size(&view, &params, 7).unwrap();
    let chunk_1 = threshold_with_chunk_size(&view, &params, 1).unwrap();

    assert!(single_chunk.num_tets() > 100);
    assert_eq!(single_chunk, chunk_64);
    assert_eq!(single_chunk, chunk_7);
    assert_eq!(single_chunk, chunk_1);
}

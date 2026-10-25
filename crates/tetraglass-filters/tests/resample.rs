//! Integration tests for the `resample` filter and `RegularGrid` (`M4.4a`).
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use tetraglass_core::{BoundingBox, UnstructuredGrid};
use tetraglass_filters::{
    ClipKeep, ClipMode, ClipParams, FilterError, ResampleParams, TetLocator, clip,
    linearize_default, resample, resample_grid, resample_with_chunk_size,
};
use tetraglass_io::{read_msh_from_path, read_vtu_from_path};

fn testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn load_vtu(name: &str) -> UnstructuredGrid {
    read_vtu_from_path(testdata_dir().join("vtk").join(name)).expect("valid VTU test fixture")
}

fn load_msh(name: &str) -> PathBuf {
    testdata_dir().join("gmsh").join(name)
}

#[test]
fn resample_reproduces_analytic_linear_and_vector_fields_on_hex_linear() {
    let grid = load_vtu("hex_linear.vtu");
    let view = linearize_default(&grid).expect("linearize hex_linear");

    let dims = [9, 7, 5];
    let vol = resample(&view, &ResampleParams::new(dims)).expect("resample hex_linear");
    assert_eq!(vol.dimensions(), dims);
    assert_eq!(vol.num_points(), 9 * 7 * 5);
    assert_eq!(vol.num_inside_points(), 9 * 7 * 5);

    let linear_arr = vol.point_data().get("linear").expect("linear array");
    let vel_arr = vol.point_data().get("velocity").expect("velocity array");
    let cell_id_arr = vol.point_data().get("cell_id").expect("cell_id array");

    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let idx = vol.linear_index(ix, iy, iz).unwrap();
                assert!(vol.is_inside(idx));
                let [x, y, z] = vol.point_position(ix, iy, iz).unwrap();
                let expected_linear = x + 2.0 * y + 3.0 * z;
                let got_linear = linear_arr.value_as_f64(idx, 0).unwrap();
                assert!(
                    (got_linear - expected_linear).abs() < 1e-12,
                    "linear mismatch at ({ix},{iy},{iz}): got {got_linear}, expected {expected_linear}"
                );

                let expected_vel = [-y, x, 0.0];
                for (c, &ev) in expected_vel.iter().enumerate() {
                    let gv = vel_arr.value_as_f64(idx, c).unwrap();
                    assert!(
                        (gv - ev).abs() < 1e-12,
                        "velocity[{c}] mismatch at ({ix},{iy},{iz}): got {gv}, expected {ev}"
                    );
                }

                let cid = cell_id_arr.value_as_f64(idx, 0).unwrap();
                assert!((0.0..8.0).contains(&cid));
            }
        }
    }
}

#[test]
fn resample_tracks_inside_mask_on_clipped_and_padded_bounds() {
    let grid = load_vtu("hex_linear.vtu");
    let view = linearize_default(&grid).expect("linearize hex_linear");
    // Clip to x <= 1.0 on [0, 2]^3, but resample over the full [0, 2]^3 bounding box.
    let clipped = clip(
        &view,
        &ClipParams::new([1.0, 0.0, 0.0])
            .origin([1.0, 1.0, 1.0])
            .keep(ClipKeep::Back)
            .mode(ClipMode::Exact),
    )
    .expect("clip hex_linear");

    let full_bounds = BoundingBox {
        min: [0.0, 0.0, 0.0],
        max: [2.0, 2.0, 2.0],
    };
    let dims = [5, 4, 4]; // x = 0.0, 0.5, 1.0, 1.5, 2.0
    let vol = resample(
        &clipped,
        &ResampleParams::new(dims).with_bounds(full_bounds),
    )
    .expect("resample clipped volume over full bounds");

    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let idx = vol.linear_index(ix, iy, iz).unwrap();
                let [x, _, _] = vol.point_position(ix, iy, iz).unwrap();
                if x <= 1.0 + 1e-9 {
                    assert!(vol.is_inside(idx), "expected inside at x = {x}");
                } else {
                    assert!(!vol.is_inside(idx), "expected outside at x = {x}");
                }
            }
        }
    }

    // Only 3 of the 5 X-planes (x = 0.0, 0.5, 1.0) are inside.
    assert_eq!(vol.num_inside_points(), 3 * 4 * 4);

    // ValueRange of `linear` on the resampled clipped volume spans [0.0, 11.0] (since x <= 1, y <= 2, z <= 2 => 1 + 4 + 6 = 11).
    let range = vol
        .point_data()
        .get("linear")
        .unwrap()
        .component_range(0)
        .unwrap();
    assert!((range.min - 0.0).abs() < 1e-12);
    assert!((range.max - 11.0).abs() < 1e-12);
}

#[test]
fn resample_covers_all_vtk_and_gmsh_cell_families() {
    let vtk_files = [
        "tet_linear.vtu",
        "hex_linear.vtu",
        "mixed_linear.vtu",
        "tet_quadratic.vtu",
        "hex_lagrange_p3.vtu",
        "polyhedra.vtu",
    ];

    for file in vtk_files {
        let grid = load_vtu(file);
        let vol = resample_grid(&grid, &ResampleParams::new([6, 6, 6]))
            .unwrap_or_else(|e| panic!("resample_grid failed on {file}: {e}"));
        assert_eq!(vol.dimensions(), [6, 6, 6]);
        assert!(
            vol.num_inside_points() > 0,
            "expected non-zero inside points on {file}"
        );
        assert!(vol.point_data().get("linear").is_some());
    }

    let msh_grid = read_msh_from_path(load_msh("tet2_v41.msh")).expect("load tet2_v41.msh");
    let msh_vol = resample_grid(&msh_grid, &ResampleParams::new([6, 6, 6]))
        .expect("resample_grid on tet2_v41.msh");
    assert!(msh_vol.num_inside_points() > 0);
    assert!(msh_vol.point_data().get("linear").is_some());
}

#[test]
fn resample_is_bit_for_bit_deterministic_across_chunk_sizes() {
    let grid = load_vtu("hex_lagrange_p3.vtu");
    let view = linearize_default(&grid).expect("linearize hex_lagrange_p3");
    let locator = TetLocator::new(&view);
    let params = ResampleParams::new([8, 7, 6]);

    let baseline = resample_with_chunk_size(&view, &locator, &params, 1).expect("chunk_size = 1");
    for chunk_size in [3, 11, 64] {
        let other =
            resample_with_chunk_size(&view, &locator, &params, chunk_size).expect("resample chunk");
        assert_eq!(baseline.inside_mask(), other.inside_mask());
        assert_eq!(baseline.point_data(), other.point_data());
    }
}

#[test]
fn resample_rejects_invalid_dimensions() {
    let grid = load_vtu("tet_linear.vtu");
    let view = linearize_default(&grid).expect("linearize tet_linear");

    for bad_dims in [[0, 4, 4], [1, 4, 4], [4, 1, 4], [4, 4, 1]] {
        let err =
            resample(&view, &ResampleParams::new(bad_dims)).expect_err("dimensions < 2 must fail");
        assert!(matches!(err, FilterError::InvalidResampleDimensions { .. }));
    }
}

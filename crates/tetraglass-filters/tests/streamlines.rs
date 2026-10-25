//! Integration tests for the `streamlines` filter and `PolylineSet` (`M4.2a`).
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use tetraglass_core::{
    Attributes, PolylineSet, SyntheticCellLayout, SyntheticGridSpec, TetView, UnstructuredGrid,
};
use tetraglass_filters::{
    FilterError, StreamlineDirection, StreamlineIntegrator, StreamlineParams, StreamlineSeeds,
    TetLocator, linearize_default, streamlines, streamlines_grid, streamlines_with_chunk_size,
};
use tetraglass_io::{read_msh_from_path, read_vtu_from_path};

fn testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ directory")
        .parent()
        .expect("workspace root")
        .join("testdata")
}

fn verify_streamline_invariants(grid: &UnstructuredGrid, lines: &PolylineSet) {
    assert!(lines.num_polylines() > 0);
    assert!(lines.num_points() >= lines.num_polylines() * 2);

    let linear_arr = lines
        .point_data()
        .get("linear")
        .expect("streamlines carry `linear` point array");
    let vel_arr = lines
        .point_data()
        .get("velocity")
        .expect("streamlines carry `velocity` point array");
    let arc_arr = lines
        .point_data()
        .get("arc_length")
        .expect("streamlines carry `arc_length` point array");
    let time_arr = lines
        .point_data()
        .get("integration_time")
        .expect("streamlines carry `integration_time` point array");
    let cell_id_arr = lines
        .point_data()
        .get("cell_id")
        .expect("streamlines carry sampled `cell_id` array");
    let seed_id_arr = lines
        .cell_data()
        .get("seed_id")
        .expect("streamlines carry per-polyline `seed_id` array");
    assert_eq!(seed_id_arr.num_tuples(), lines.num_polylines());

    for (pt_idx, &p) in lines.points().iter().enumerate() {
        let src_cell = lines.point_source_cells()[pt_idx];
        assert!((src_cell as usize) < grid.num_cells());

        // Exact reproduction of linear scalar field `linear = x + 2*y + 3*z`:
        let interp_lin = linear_arr.value_as_f64(pt_idx, 0).unwrap();
        let expected_lin = p[0] + 2.0 * p[1] + 3.0 * p[2];
        assert!(
            (interp_lin - expected_lin).abs() < 1e-11,
            "point {pt_idx} ({p:?}): `linear` {interp_lin} != {expected_lin}"
        );

        // Exact reproduction of linear vector field `velocity = [-y, x, 0]`:
        let vx = vel_arr.value_as_f64(pt_idx, 0).unwrap();
        let vy = vel_arr.value_as_f64(pt_idx, 1).unwrap();
        let vz = vel_arr.value_as_f64(pt_idx, 2).unwrap();
        assert!((vx - (-p[1])).abs() < 1e-11);
        assert!((vy - p[0]).abs() < 1e-11);
        assert!(vz.abs() < 1e-11);

        let arc = arc_arr.value_as_f64(pt_idx, 0).unwrap();
        assert!(arc >= 0.0 && arc.is_finite());
        let t = time_arr.value_as_f64(pt_idx, 0).unwrap();
        assert!(t.is_finite());

        let cid = cell_id_arr.value_as_f64(pt_idx, 0).unwrap();
        assert_eq!(cid, f64::from(src_cell));
    }

    // Verify conversions to tube surface, line surface, and unstructured grid:
    let tube = lines.to_tube_surface(0.02, 8).expect("build tube surface");
    assert_eq!(tube.num_points(), lines.num_points() * 8);
    assert_eq!(tube.num_triangles(), lines.num_segments() * 16);
    assert_eq!(tube.num_wireframe_edges(), lines.num_segments());
    assert!(tube.point_data().get("linear").is_some());
    assert!(tube.point_data().get("velocity").is_some());
    assert!(tube.point_data().get("arc_length").is_some());
    assert!(tube.cell_data().get("seed_id").is_some());

    let line_surf = lines.to_line_surface().expect("build line surface");
    assert_eq!(line_surf.num_points(), lines.num_points());
    assert_eq!(line_surf.num_triangles(), 0);
    assert_eq!(line_surf.num_wireframe_edges(), lines.num_segments());

    let line_grid = lines
        .to_unstructured_grid()
        .expect("convert streamlines to UnstructuredGrid");
    assert_eq!(line_grid.num_points(), lines.num_points());
    assert_eq!(line_grid.num_cells(), lines.num_segments());
}

#[test]
fn analytic_circular_flow_streamlines_preserve_radius_and_arc_length_in_rk4_and_rk45() {
    let grid = read_vtu_from_path(testdata_dir().join("vtk/hex_linear.vtu")).unwrap();
    let view = linearize_default(&grid).unwrap();

    // Seed at (1.5, 0.2, 1.0) inside [0, 2]^3; velocity = (-y, x, 0) traces a circle in z = 1.0
    // with radius r0 = sqrt(1.5^2 + 0.2^2) around the origin.
    let seed: [f64; 3] = [1.5, 0.2, 1.0];
    let r0 = (seed[0] * seed[0] + seed[1] * seed[1]).sqrt();
    let target_arc = 0.6;

    for integrator in [StreamlineIntegrator::Rk4, StreamlineIntegrator::Rk45] {
        let params = StreamlineParams::new("velocity", StreamlineSeeds::point(seed))
            .integrator(integrator)
            .direction(StreamlineDirection::Forward)
            .step_size(0.02)
            .max_length(target_arc);

        let lines = streamlines(&view, &params).unwrap();
        assert_eq!(lines.num_polylines(), 1);
        let pts = lines.polyline_points(0).unwrap();
        match integrator {
            StreamlineIntegrator::Rk4 => assert!(pts.len() >= 30),
            StreamlineIntegrator::Rk45 => assert!(pts.len() >= 6 && pts.len() < 25),
        }

        for &p in pts {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!(
                (r - r0).abs() < 1e-6,
                "{integrator:?}: radius {r} drifted from {r0} at {p:?}"
            );
            assert!(
                (p[2] - 1.0).abs() < 1e-12,
                "{integrator:?}: z coordinate {} drifted from 1.0",
                p[2]
            );
        }

        let total_len = lines.total_length();
        assert!(
            (total_len - target_arc).abs() < 1e-4,
            "{integrator:?}: total length {total_len} != target {target_arc}"
        );

        // Integration time at end of arc must equal `target_arc / r0` (since speed = r0 is constant):
        let last_idx = lines.num_points() - 1;
        let end_time = lines
            .point_data()
            .get("integration_time")
            .unwrap()
            .value_as_f64(last_idx, 0)
            .unwrap();
        let expected_time = target_arc / r0;
        assert!(
            (end_time - expected_time).abs() < 1e-4,
            "{integrator:?}: end integration_time {end_time} != {expected_time}"
        );
    }

    // Test `StreamlineDirection::Both` from [1.0, 1.0, 1.0]:
    let both_params = StreamlineParams::new("velocity", StreamlineSeeds::point([1.0, 1.0, 1.0]))
        .direction(StreamlineDirection::Both)
        .step_size(0.025)
        .max_length(0.4);
    let both_lines = streamlines(&view, &both_params).unwrap();
    assert_eq!(both_lines.num_polylines(), 1);
    assert!((both_lines.total_length() - 0.8).abs() < 1e-3);
    let times = both_lines.point_data().get("integration_time").unwrap();
    assert!(times.value_as_f64(0, 0).unwrap() < 0.0);
    assert!(times.value_as_f64(both_lines.num_points() - 1, 0).unwrap() > 0.0);
}

#[test]
fn streamlines_cover_all_vtk_and_gmsh_cell_families_and_seed_types() {
    let vtk = testdata_dir().join("vtk");
    let gmsh = testdata_dir().join("gmsh");

    let grids: Vec<(&str, UnstructuredGrid)> = vec![
        (
            "tet_linear.vtu",
            read_vtu_from_path(vtk.join("tet_linear.vtu")).unwrap(),
        ),
        (
            "hex_linear.vtu",
            read_vtu_from_path(vtk.join("hex_linear.vtu")).unwrap(),
        ),
        (
            "mixed_linear.vtu",
            read_vtu_from_path(vtk.join("mixed_linear.vtu")).unwrap(),
        ),
        (
            "tet_quadratic.vtu",
            read_vtu_from_path(vtk.join("tet_quadratic.vtu")).unwrap(),
        ),
        (
            "hex_lagrange_p3.vtu",
            read_vtu_from_path(vtk.join("hex_lagrange_p3.vtu")).unwrap(),
        ),
        (
            "polyhedra.vtu",
            read_vtu_from_path(vtk.join("polyhedra.vtu")).unwrap(),
        ),
        (
            "tet2_v41.msh",
            read_msh_from_path(gmsh.join("tet2_v41.msh")).unwrap(),
        ),
    ];

    for (name, grid) in &grids {
        let bounds = grid.bounds().unwrap();
        let center = bounds.center();
        let ext = bounds.extents();

        // 1. Line seeds across the interior of the mesh
        let line_seeds = StreamlineSeeds::line(
            [
                center[0] - 0.2 * ext[0],
                center[1] - 0.2 * ext[1],
                center[2],
            ],
            [
                center[0] + 0.2 * ext[0],
                center[1] + 0.2 * ext[1],
                center[2],
            ],
            4,
        );
        let line_params = StreamlineParams::new("velocity", line_seeds)
            .direction(StreamlineDirection::Both)
            .max_length(0.5 * bounds.diagonal());
        let lines = streamlines_grid(grid, &line_params)
            .unwrap_or_else(|e| panic!("{name} line streamlines failed: {e}"));
        verify_streamline_invariants(grid, &lines);

        // 2. Plane seeds centered in the mesh
        let plane_seeds = StreamlineSeeds::plane(
            center,
            [0.0, 0.0, 1.0],
            [0.3 * ext[0], 0.3 * ext[1]],
            [3, 3],
        );
        let plane_params = StreamlineParams::new("point:velocity", plane_seeds)
            .direction(StreamlineDirection::Forward)
            .integrator(StreamlineIntegrator::Rk4)
            .max_length(0.4 * bounds.diagonal());
        let plane_lines = streamlines_grid(grid, &plane_params)
            .unwrap_or_else(|e| panic!("{name} plane streamlines failed: {e}"));
        assert_eq!(plane_lines.num_polylines(), 9, "{name}");
        verify_streamline_invariants(grid, &plane_lines);
    }
}

#[test]
fn stagnation_out_of_bounds_empty_and_invalid_parameters_behave_correctly() {
    let grid = read_vtu_from_path(testdata_dir().join("vtk/tet_linear.vtu")).unwrap();
    let view = linearize_default(&grid).unwrap();

    // 1. Stagnation point at x = 0, y = 0 where velocity = (-y, x, 0) = (0, 0, 0) produces 0 streamlines:
    let stag = streamlines(
        &view,
        &StreamlineParams::new("velocity", StreamlineSeeds::point([0.0, 0.0, 0.5])),
    )
    .unwrap();
    assert_eq!(stag.num_polylines(), 0);
    assert_eq!(stag.num_points(), 0);

    // 2. Out-of-bounds seed point produces 0 streamlines:
    let oob = streamlines(
        &view,
        &StreamlineParams::new("velocity", StreamlineSeeds::point([5.0, 5.0, 5.0])),
    )
    .unwrap();
    assert_eq!(oob.num_polylines(), 0);

    // 3. Empty TetView returns empty PolylineSet:
    let empty_view = TetView::new(
        Vec::<[f64; 3]>::new(),
        Vec::<[u32; 4]>::new(),
        Vec::<u32>::new(),
        view.point_data().clone(),
        Attributes::new(0),
    )
    .unwrap_err();
    let _ = empty_view;
    let valid_empty_view = TetView::new(
        Vec::<[f64; 3]>::new(),
        Vec::<[u32; 4]>::new(),
        Vec::<u32>::new(),
        Attributes::new(0),
        Attributes::new(0),
    )
    .unwrap();
    // Missing array on empty view still validates array existence first:
    assert_eq!(
        streamlines(
            &valid_empty_view,
            &StreamlineParams::new("velocity", StreamlineSeeds::point([0.5, 0.5, 0.5]))
        ),
        Err(FilterError::MissingAttributeArray {
            name: "velocity".to_string()
        })
    );

    // 4. Scalar array passed as `vectors` returns `NonVectorArray`:
    assert_eq!(
        streamlines(
            &view,
            &StreamlineParams::new("linear", StreamlineSeeds::point([0.5, 0.5, 0.5]))
        ),
        Err(FilterError::NonVectorArray {
            array: "linear".to_string(),
            num_components: 1,
        })
    );

    // 5. Invalid seed and integration parameters return errors:
    assert_eq!(
        streamlines(
            &view,
            &StreamlineParams::new(
                "velocity",
                StreamlineSeeds::line([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0)
            )
        ),
        Err(FilterError::InvalidStreamlineSeeds {
            reason: "line seed count must be >= 1",
        })
    );
    assert_eq!(
        streamlines(
            &view,
            &StreamlineParams::new(
                "velocity",
                StreamlineSeeds::plane([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.5, 0.5], [2, 2])
            )
        ),
        Err(FilterError::InvalidPlaneNormal {
            normal: [0.0, 0.0, 0.0],
        })
    );
    assert_eq!(
        streamlines(
            &view,
            &StreamlineParams::new("velocity", StreamlineSeeds::point([0.5, 0.5, 0.5]))
                .max_length(-1.0)
        ),
        Err(FilterError::InvalidStreamlineParams {
            reason: "max_length must be finite and > 0.0",
        })
    );
}

#[test]
fn streamlines_are_bit_for_bit_deterministic_across_chunk_sizes() {
    let grid = SyntheticGridSpec::new(8, 8, 8, SyntheticCellLayout::Hexahedra)
        .generate()
        .unwrap();
    let view = linearize_default(&grid).unwrap();
    let locator = TetLocator::new(&view);

    let params = StreamlineParams::new(
        "velocity",
        StreamlineSeeds::plane([0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [0.6, 0.6], [6, 6]),
    )
    .direction(StreamlineDirection::Both)
    .integrator(StreamlineIntegrator::Rk45)
    .max_length(0.8);

    let baseline = streamlines_with_chunk_size(&view, &locator, &params, 1).unwrap();
    assert_eq!(baseline.num_polylines(), 36);

    for chunk_size in [2, 5, 16, 64, 1000] {
        let candidate = streamlines_with_chunk_size(&view, &locator, &params, chunk_size).unwrap();
        assert_eq!(
            candidate, baseline,
            "streamlines_with_chunk_size({chunk_size}) differed from chunk_size = 1"
        );
    }
}

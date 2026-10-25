//! Integration tests for the BVH cell locator (`M4.1`).
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use tetraglass_core::{
    Attributes, SyntheticCellLayout, SyntheticGridSpec, TetView, UnstructuredGrid,
};
use tetraglass_filters::{
    ClipKeep, ClipMode, ClipParams, TetLocation, TetLocator, clip, linearize_default,
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

fn verify_located_point_invariants(
    grid: &UnstructuredGrid,
    view: &TetView,
    loc: TetLocation,
    pt: [f64; 3],
) {
    assert!(
        loc.tet_index < view.num_tets(),
        "tet_index {} out of bounds (num_tets = {})",
        loc.tet_index,
        view.num_tets()
    );
    assert!(
        (loc.source_cell_id as usize) < grid.num_cells(),
        "source_cell_id {} out of bounds (num_cells = {})",
        loc.source_cell_id,
        grid.num_cells()
    );
    assert_eq!(
        loc.source_cell_id,
        view.tet_source_cells()[loc.tet_index],
        "source_cell_id must match view.tet_source_cells()[tet_index]"
    );

    let sum_w: f64 = loc.barycentric.iter().sum();
    assert!(
        (sum_w - 1.0).abs() < 1e-12,
        "barycentric weights {:?} do not sum to 1.0 (sum = {sum_w})",
        loc.barycentric
    );
    assert!(
        loc.min_barycentric() >= -1e-9,
        "min barycentric weight {} < -1e-9 for query {pt:?}",
        loc.min_barycentric()
    );

    let clamped = loc.clamped_barycentric();
    let clamped_sum: f64 = clamped.iter().sum();
    assert!((clamped_sum - 1.0).abs() < 1e-12);
    for cw in clamped {
        assert!((0.0..=1.0).contains(&cw));
    }

    // Reconstructed position must match query point:
    let interp_pos = loc
        .interpolate_position(view)
        .expect("valid tet_index interpolates position");
    for axis in 0..3 {
        assert!(
            (interp_pos[axis] - pt[axis]).abs() < 1e-11,
            "interpolated position {interp_pos:?} differs from query {pt:?} on axis {axis}"
        );
    }

    // Linear scalar field `linear = x + 2*y + 3*z` must be reproduced exactly:
    let linear_arr = view
        .point_data()
        .get("linear")
        .expect("mesh has `linear` point array");
    let interp_linear = loc
        .interpolate_point_scalar(view, linear_arr, 0)
        .expect("interpolate `linear` scalar");
    let expected_linear = pt[0] + 2.0 * pt[1] + 3.0 * pt[2];
    assert!(
        (interp_linear - expected_linear).abs() < 1e-11,
        "interpolated `linear` {interp_linear} != expected {expected_linear} at {pt:?}"
    );

    // Linear vector field `velocity = [-y, x, 0]` must be reproduced exactly:
    let vel_arr = view
        .point_data()
        .get("velocity")
        .expect("mesh has `velocity` point array");
    let interp_vel = loc
        .interpolate_point_vector3(view, vel_arr)
        .expect("interpolate `velocity` vector3");
    let expected_vel = [-pt[1], pt[0], 0.0];
    for c in 0..3 {
        assert!(
            (interp_vel[c] - expected_vel[c]).abs() < 1e-11,
            "interpolated `velocity[{c}]` {} != expected {} at {pt:?}",
            interp_vel[c],
            expected_vel[c]
        );
    }

    let mut tuple_buf = [0.0; 3];
    assert!(loc.interpolate_point_tuple(view, vel_arr, &mut tuple_buf));
    assert_eq!(tuple_buf, interp_vel);

    let interp_mag = loc
        .interpolate_point_magnitude(view, vel_arr)
        .expect("interpolate `velocity` magnitude");
    let expected_mag = (pt[0] * pt[0] + pt[1] * pt[1]).sqrt();
    assert!(
        (interp_mag - expected_mag).abs() < 1e-11,
        "interpolated velocity magnitude {interp_mag} != expected {expected_mag} at {pt:?}"
    );

    // Cell scalar `cell_id` must match `loc.source_cell_id`:
    let cell_id_arr = view
        .cell_data()
        .get("cell_id")
        .expect("mesh has `cell_id` cell array");
    let sampled_cell_id = loc
        .sample_cell_scalar(cell_id_arr, 0)
        .expect("sample `cell_id`");
    assert_eq!(sampled_cell_id, f64::from(loc.source_cell_id));
}

#[test]
fn locator_covers_all_vtk_and_gmsh_cell_families_and_reproduces_linear_fields() {
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
        let view = linearize_default(grid).expect("linearize test mesh");
        let locator = TetLocator::new(&view);
        assert_eq!(locator.num_tets(), view.num_tets(), "{name}");
        assert!(locator.num_nodes() >= 1, "{name}");
        assert_eq!(locator.bounds(), view.bounds(), "{name}");

        // 1. Every mesh vertex in the linearized view must be located on the boundary/interior:
        for (pt_idx, &pt) in view.points().iter().enumerate() {
            let loc = locator.locate(pt).unwrap_or_else(|| {
                panic!("{name}: failed to locate mesh vertex {pt_idx} at {pt:?}")
            });
            verify_located_point_invariants(grid, &view, loc, pt);
        }

        // 2. Every tetrahedron centroid and weighted interior sample point must be located:
        for (tet_idx, &[v0, v1, v2, v3]) in view.tets().iter().enumerate() {
            let p0 = view.points()[v0 as usize];
            let p1 = view.points()[v1 as usize];
            let p2 = view.points()[v2 as usize];
            let p3 = view.points()[v3 as usize];

            for weights in [
                [0.25, 0.25, 0.25, 0.25],
                [0.55, 0.20, 0.15, 0.10],
                [0.5, 0.5, 0.0, 0.0],                   // shared edge midpoint
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], // shared face centroid
            ] {
                let query = [
                    weights[0] * p0[0]
                        + weights[1] * p1[0]
                        + weights[2] * p2[0]
                        + weights[3] * p3[0],
                    weights[0] * p0[1]
                        + weights[1] * p1[1]
                        + weights[2] * p2[1]
                        + weights[3] * p3[1],
                    weights[0] * p0[2]
                        + weights[1] * p1[2]
                        + weights[2] * p2[2]
                        + weights[3] * p3[2],
                ];
                let loc = locator.locate(query).unwrap_or_else(|| {
                    panic!("{name}: failed to locate sample in tet {tet_idx} at {query:?}")
                });
                verify_located_point_invariants(grid, &view, loc, query);

                // Strictly interior points must locate back to `tet_idx` itself:
                if weights[3] > 0.0 {
                    assert_eq!(
                        loc.tet_index, tet_idx,
                        "{name}: strictly interior point in tet {tet_idx} located to tet {}",
                        loc.tet_index
                    );
                }
            }
        }

        // 3. Points outside the bounding box must return None:
        let bounds = locator.bounds().unwrap();
        let outside_points = [
            [bounds.min[0] - 0.01, bounds.center()[1], bounds.center()[2]],
            [bounds.max[0] + 0.01, bounds.center()[1], bounds.center()[2]],
            [bounds.center()[0], bounds.min[1] - 0.01, bounds.center()[2]],
            [bounds.center()[0], bounds.max[1] + 0.01, bounds.center()[2]],
            [bounds.center()[0], bounds.center()[1], bounds.min[2] - 0.01],
            [bounds.center()[0], bounds.center()[1], bounds.max[2] + 0.01],
            [f64::NAN, 0.5, 0.5],
            [f64::INFINITY, 0.5, 0.5],
        ];
        for out_pt in outside_points {
            assert!(
                locator.locate(out_pt).is_none(),
                "{name}: expected None for out-of-bounds point {out_pt:?}"
            );
        }
    }
}

#[test]
fn locator_handles_empty_views_and_clipped_non_convex_domains() {
    // 1. Empty TetView
    let empty_view = TetView::new(
        Vec::<[f64; 3]>::new(),
        Vec::<[u32; 4]>::new(),
        Vec::<u32>::new(),
        Attributes::new(0),
        Attributes::new(0),
    )
    .unwrap();
    let empty_locator = TetLocator::new(&empty_view);
    assert_eq!(empty_locator.num_tets(), 0);
    assert_eq!(empty_locator.num_nodes(), 0);
    assert_eq!(empty_locator.bounds(), None);
    assert_eq!(empty_locator.locate([0.0, 0.0, 0.0]), None);
    assert!(
        empty_locator
            .locate_many(&[[0.0, 0.0, 0.0]])
            .iter()
            .all(Option::is_none)
    );

    // 2. Clipped mesh where points inside the overall AABB lie outside the clipped half-space
    let grid = SyntheticGridSpec::new(6, 6, 6, SyntheticCellLayout::Hexahedra)
        .generate()
        .unwrap();
    let full_view = linearize_default(&grid).unwrap();
    // Clip with oblique plane `x + y + z = 1.5`, keeping `Back` (`x + y + z <= 1.5`).
    // The clipped mesh still has AABB `[0, 1]^3`, while points with `x + y + z > 1.5` (such as
    // `[0.8, 0.8, 0.8]`) are inside the root AABB `[0, 1]^3` yet outside all kept tetrahedra!
    let clipped_view = clip(
        &full_view,
        &ClipParams::new([1.0, 1.0, 1.0])
            .origin([0.5, 0.5, 0.5])
            .keep(ClipKeep::Back)
            .mode(ClipMode::Exact),
    )
    .unwrap();
    let locator = TetLocator::new(&clipped_view);
    assert_eq!(
        locator.bounds().unwrap().max,
        [1.0, 1.0, 1.0],
        "clipped domain still spans [0, 1] along each axis"
    );

    // Point in kept half-space (`0.2 + 0.2 + 0.2 = 0.6 <= 1.5`) is located:
    let kept_hit = locator
        .locate([0.2, 0.2, 0.2])
        .expect("point in kept half-space must be located");
    let pos = kept_hit.interpolate_position(&clipped_view).unwrap();
    assert!((pos[0] - 0.2).abs() < 1e-12);

    // Point on the clip plane (`0.5 + 0.5 + 0.5 = 1.5`) is located:
    assert!(locator.locate([0.5, 0.5, 0.5]).is_some());

    // Point inside AABB `[0, 1]^3` but in the discarded half-space (`0.8 + 0.8 + 0.8 = 2.4 > 1.5`) returns None:
    assert_eq!(locator.locate([0.8, 0.8, 0.8]), None);
}

#[test]
fn locate_with_hint_and_batch_locate_many_are_deterministic() {
    let grid = SyntheticGridSpec::new(10, 9, 8, SyntheticCellLayout::Hexahedra)
        .generate()
        .unwrap();
    let view = linearize_default(&grid).unwrap();
    let locator = TetLocator::new(&view);

    // Generate 500 points along a helical path through `[0, 1]^3` plus some out-of-bounds steps:
    let mut queries = Vec::with_capacity(500);
    for i in 0..500 {
        let t = f64::from(i) / 499.0;
        let angle = t * 6.0 * std::f64::consts::PI;
        let r = 0.45 * (1.0 - 0.3 * t);
        let x = 0.5 + r * angle.cos();
        let y = 0.5 + r * angle.sin();
        let z = -0.05 + 1.10 * t; // starts slightly below 0.0 and ends slightly above 1.0
        queries.push([x, y, z]);
    }

    // 1. Stepping with `locate_with_hint` using previous step's `tet_index` must agree with `locate`:
    let mut hint: Option<usize> = None;
    let mut unhinted_results = Vec::with_capacity(queries.len());
    for &q in &queries {
        let unhinted = locator.locate(q);
        let hinted = locator.locate_with_hint(q, hint);
        let bad_hinted = locator.locate_with_hint(q, Some(999_999));
        assert_eq!(hinted, unhinted, "hint={hint:?} disagreed at query {q:?}");
        assert_eq!(
            bad_hinted, unhinted,
            "out-of-bounds hint disagreed at query {q:?}"
        );
        hint = unhinted.map(|loc| loc.tet_index);
        unhinted_results.push(unhinted);
    }

    // 2. `locate_many_with_chunk_size` must be bit-for-bit identical across chunk sizes and leaf capacities:
    for chunk_size in [1, 7, 32, 128, 10_000] {
        let batch = locator.locate_many_with_chunk_size(&queries, chunk_size);
        assert_eq!(
            batch, unhinted_results,
            "locate_many_with_chunk_size({chunk_size}) disagreed with single-point locate"
        );
    }

    for max_leaf in [1, 2, 4, 16, 64] {
        let alt_locator = TetLocator::with_max_leaf_tets(&view, max_leaf);
        let alt_batch = alt_locator.locate_many(&queries);
        assert_eq!(
            alt_batch, unhinted_results,
            "TetLocator::with_max_leaf_tets({max_leaf}) disagreed with default leaf size"
        );
    }
}

//! Integration tests for the plane `clip` filter in exact and crinkle modes (`M2.5`).
#![cfg(not(target_arch = "wasm32"))]

use std::collections::HashMap;
use std::path::PathBuf;

use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec, TetView, UnstructuredGrid};
use tetraglass_filters::{
    ClipKeep, ClipMode, ClipParams, FilterError, LinearizeParams, clip, clip_grid,
    clip_with_chunk_size, extract_tet_view_surface, linearize_default,
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

fn assert_clip_invariants(
    grid: &UnstructuredGrid,
    clipped: &TetView,
    origin: [f64; 3],
    effective_unit_normal: [f64; 3],
    mode: ClipMode,
) {
    let pts = clipped.points();

    // 1. Every tetrahedron must have strictly positive signed volume.
    for tet_idx in 0..clipped.num_tets() {
        let vol = clipped.tet_signed_volume(tet_idx).unwrap();
        assert!(
            vol > 0.0,
            "tet {tet_idx} has non-positive signed volume {vol:e}"
        );
    }

    // 2. In Exact mode, every vertex must lie in the kept half-space `(p - origin) . n >= -1e-11`.
    if mode == ClipMode::Exact {
        for (idx, &p) in pts.iter().enumerate() {
            let dist = (p[0] - origin[0]) * effective_unit_normal[0]
                + (p[1] - origin[1]) * effective_unit_normal[1]
                + (p[2] - origin[2]) * effective_unit_normal[2];
            assert!(
                dist >= -1e-11,
                "exact-clipped vertex {idx} ({p:?}) is at signed distance {dist:e} (< 0)"
            );
        }
    }

    // 3. The external boundary surface of the clipped volume must be a closed, watertight 2-manifold
    // (every undirected triangle edge on the outer surface is shared by an even number of triangles,
    // specifically 2 for manifold domains).
    let surf = extract_tet_view_surface(clipped).unwrap();
    if clipped.num_tets() > 0 {
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
                "boundary surface of clipped volume has open/non-watertight edge {edge:?} with incident triangle count {count}"
            );
        }
    }

    // 4. Linear scalar field `linear = x + 2*y + 3*z` and vector field `velocity = [-y, x, 0]`
    // must be reproduced accurately at every vertex of the clipped volume.
    if let Some(linear_arr) = clipped.point_data().get("linear") {
        assert_eq!(linear_arr.num_tuples(), clipped.num_points());
        for (idx, &[x, y, z]) in pts.iter().enumerate() {
            let expected = x + 2.0 * y + 3.0 * z;
            let actual = linear_arr.value_as_f64(idx, 0).unwrap();
            assert!(
                (actual - expected).abs() < 1e-11,
                "point {idx} `linear` mismatch: {actual} vs {expected}"
            );
        }
    }
    if let Some(vel_arr) = clipped.point_data().get("velocity") {
        assert_eq!(vel_arr.num_tuples(), clipped.num_points());
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

    // 5. Cell attribute `cell_id` and `tet_source_cells` must map back to the original grid's source cells,
    // both on `clipped` and on `extract_tet_view_surface(clipped)`.
    if let Some(cell_id_arr) = clipped.cell_data().get("cell_id") {
        assert_eq!(cell_id_arr.num_tuples(), clipped.num_tets());
        for (tet_idx, &src_cell) in clipped.tet_source_cells().iter().enumerate() {
            assert!(
                (src_cell as usize) < grid.num_cells(),
                "out-of-range source cell {src_cell}"
            );
            assert_eq!(
                cell_id_arr.value_as_f64(tet_idx, 0),
                Some(f64::from(src_cell))
            );
        }
    }
    if let Some(surf_cell_id) = surf.cell_data().get("cell_id") {
        assert_eq!(surf_cell_id.num_tuples(), surf.num_triangles());
        for (tri_idx, &src_cell) in surf.triangle_source_cells().iter().enumerate() {
            assert!((src_cell as usize) < grid.num_cells());
            assert_eq!(
                surf_cell_id.value_as_f64(tri_idx, 0),
                Some(f64::from(src_cell))
            );
        }
    }
}

#[test]
fn exact_clip_covers_all_cell_families_analytic_volumes_and_partition_of_unity() {
    let dir = vtk_dir();

    // 1. Linear tetrahedra on [0, 1]^3 (volume 1.0):
    // Clip at x = 0.25 -> Front has volume 0.75, Back has volume 0.25.
    let tet_lin = read_vtu_from_path(dir.join("tet_linear.vtu")).unwrap();
    let tet_lin_view = linearize_default(&tet_lin).unwrap();
    let front_025 = clip(
        &tet_lin_view,
        &ClipParams::with_origin([0.25, 0.5, 0.5], [1.0, 0.0, 0.0]),
    )
    .unwrap();
    let back_025 = clip(
        &tet_lin_view,
        &ClipParams::with_origin([0.25, 0.5, 0.5], [1.0, 0.0, 0.0]).keep(ClipKeep::Back),
    )
    .unwrap();
    assert!((front_025.total_volume() - 0.75).abs() < 1e-12);
    assert!((back_025.total_volume() - 0.25).abs() < 1e-12);
    assert_clip_invariants(
        &tet_lin,
        &front_025,
        [0.25, 0.5, 0.5],
        [1.0, 0.0, 0.0],
        ClipMode::Exact,
    );
    assert_clip_invariants(
        &tet_lin,
        &back_025,
        [0.25, 0.5, 0.5],
        [-1.0, 0.0, 0.0],
        ClipMode::Exact,
    );

    // Oblique diagonal plane x + y + z = 1.5 through center [0.5, 0.5, 0.5] bisects [0, 1]^3 into 0.5 and 0.5.
    let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
    let diag_front = clip(&tet_lin_view, &ClipParams::new([1.0, 1.0, 1.0])).unwrap();
    let diag_back = clip(
        &tet_lin_view,
        &ClipParams::new([1.0, 1.0, 1.0]).keep(ClipKeep::Back),
    )
    .unwrap();
    assert!((diag_front.total_volume() - 0.5).abs() < 1e-12);
    assert!((diag_back.total_volume() - 0.5).abs() < 1e-12);
    assert_clip_invariants(
        &tet_lin,
        &diag_front,
        [0.5, 0.5, 0.5],
        [inv_sqrt3, inv_sqrt3, inv_sqrt3],
        ClipMode::Exact,
    );

    // 2. Linear hexahedra on [0, 2]^3 (volume 8.0):
    let hex_lin = read_vtu_from_path(dir.join("hex_linear.vtu")).unwrap();
    let hex_view = linearize_default(&hex_lin).unwrap();
    let hex_front = clip(
        &hex_view,
        &ClipParams::with_origin([0.5, 1.0, 1.0], [1.0, 0.0, 0.0]),
    )
    .unwrap();
    let hex_back = clip(
        &hex_view,
        &ClipParams::with_origin([0.5, 1.0, 1.0], [1.0, 0.0, 0.0]).keep(ClipKeep::Back),
    )
    .unwrap();
    assert!((hex_front.total_volume() - 6.0).abs() < 1e-12);
    assert!((hex_back.total_volume() - 2.0).abs() < 1e-12);
    assert_clip_invariants(
        &hex_lin,
        &hex_front,
        [0.5, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        ClipMode::Exact,
    );

    // 3. Mixed linear cells (Hexahedron + Wedge + Pyramid + Tetra), Quadratic tetrahedra,
    // 3rd-order Lagrange hexahedra, and Polyhedra: verify partition of unity `vol(Front) + vol(Back) == vol(all)`
    // and watertightness across all families.
    for filename in [
        "mixed_linear.vtu",
        "tet_quadratic.vtu",
        "hex_lagrange_p3.vtu",
        "polyhedra.vtu",
    ] {
        let grid = read_vtu_from_path(dir.join(filename)).unwrap();
        let view = linearize_default(&grid).unwrap();
        let total_vol = view.total_volume();

        let n = [0.6, -0.48, 0.64]; // unit normal (0.36 + 0.2304 + 0.4096 = 1.0)
        let center = view.bounds().unwrap().center();
        let front = clip(&view, &ClipParams::with_origin(center, n)).unwrap();
        let back = clip(
            &view,
            &ClipParams::with_origin(center, n).keep(ClipKeep::Back),
        )
        .unwrap();

        assert!(
            front.total_volume() > 0.1 * total_vol,
            "{filename} front volume too small"
        );
        assert!(
            back.total_volume() > 0.1 * total_vol,
            "{filename} back volume too small"
        );
        assert!(
            ((front.total_volume() + back.total_volume()) - total_vol).abs() < 1e-11,
            "{filename}: front ({}) + back ({}) != total ({total_vol})",
            front.total_volume(),
            back.total_volume()
        );
        assert_clip_invariants(&grid, &front, center, n, ClipMode::Exact);
        assert_clip_invariants(&grid, &back, center, [-n[0], -n[1], -n[2]], ClipMode::Exact);
    }
}

#[test]
fn crinkle_mode_keeps_whole_original_cells_and_matches_exact_on_cell_boundaries() {
    let dir = vtk_dir();
    // `hex_linear.vtu` is a 2x2x2 grid of 8 unit hexahedra on [0, 2]^3 (total volume 8.0, 6 tets per hex = 48 tets).
    let hex_lin = read_vtu_from_path(dir.join("hex_linear.vtu")).unwrap();
    let view = linearize_default(&hex_lin).unwrap();

    // 1. Plane at x = 1.5 with normal +X cuts through the 4 right-half hexes (`x in [1, 2]`).
    // - Exact mode cuts those 4 hexes in half -> volume = 4 * 0.5 = 2.0.
    // - Crinkle mode keeps the 4 whole right-half hexes -> volume = 4 * 1.0 = 4.0 (24 tets),
    //   and drops the 4 left-half hexes (`x in [0, 1]`, max x = 1.0 < 1.5).
    let params_crinkle_15 =
        ClipParams::with_origin([1.5, 1.0, 1.0], [1.0, 0.0, 0.0]).mode(ClipMode::Crinkle);
    let crinkle_15 = clip(&view, &params_crinkle_15).unwrap();
    assert_eq!(crinkle_15.num_tets(), 24);
    assert!((crinkle_15.total_volume() - 4.0).abs() < 1e-12);
    assert_clip_invariants(
        &hex_lin,
        &crinkle_15,
        [1.5, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        ClipMode::Crinkle,
    );

    // Verify every kept source cell in crinkle mode has all 6 of its original tetrahedra intact.
    let mut tets_per_src_cell: HashMap<u32, usize> = HashMap::new();
    for &src_cell in crinkle_15.tet_source_cells() {
        *tets_per_src_cell.entry(src_cell).or_insert(0) += 1;
    }
    assert_eq!(tets_per_src_cell.len(), 4);
    for (&src_cell, &count) in &tets_per_src_cell {
        assert_eq!(
            count, 6,
            "source cell {src_cell} did not keep all 6 tetrahedra"
        );
    }

    // 2. Plane at x = 0.5 with normal +X cuts through the 4 left-half hexes (`x in [0, 1]`),
    // which extend up to x = 1.0 > 0.5. Crinkle mode keeps all 8 hexes (volume 8.0, 48 tets).
    let crinkle_05 = clip(
        &view,
        &ClipParams::with_origin([0.5, 1.0, 1.0], [1.0, 0.0, 0.0]).mode(ClipMode::Crinkle),
    )
    .unwrap();
    assert_eq!(crinkle_05.num_tets(), 48);
    assert!((crinkle_05.total_volume() - 8.0).abs() < 1e-12);

    // 3. Plane at x = 1.0 (directly on the shared mid-plane face between left and right hexes):
    // Neither Exact nor Crinkle mode includes the left-half hexes (`x in [0, 1]`) when keeping Front (`x >= 1.0`),
    // because the left-half hexes only touch x = 1.0 on a face and have zero volume in `x > 1.0`.
    let exact_10 = clip(
        &view,
        &ClipParams::with_origin([1.0, 1.0, 1.0], [1.0, 0.0, 0.0]),
    )
    .unwrap();
    let crinkle_10 = clip(
        &view,
        &ClipParams::with_origin([1.0, 1.0, 1.0], [1.0, 0.0, 0.0]).mode(ClipMode::Crinkle),
    )
    .unwrap();
    assert!((exact_10.total_volume() - 4.0).abs() < 1e-12);
    assert!((crinkle_10.total_volume() - 4.0).abs() < 1e-12);
    assert_eq!(crinkle_10.num_tets(), 24);
}

#[test]
fn degenerate_vertex_planes_empty_clips_and_invalid_params_behave_correctly() {
    let dir = vtk_dir();
    let tet_lin = read_vtu_from_path(dir.join("tet_linear.vtu")).unwrap();
    let view = linearize_default(&tet_lin).unwrap();

    // 1. Diagonal plane `x - y = 0` passes directly through 4 vertices of `[0, 1]^3` and shared Kuhn faces.
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let diag_front = clip(
        &view,
        &ClipParams::with_origin([0.5, 0.5, 0.5], [inv_sqrt2, -inv_sqrt2, 0.0]),
    )
    .unwrap();
    let diag_back = clip(
        &view,
        &ClipParams::with_origin([0.5, 0.5, 0.5], [inv_sqrt2, -inv_sqrt2, 0.0])
            .keep(ClipKeep::Back),
    )
    .unwrap();
    assert!((diag_front.total_volume() - 0.5).abs() < 1e-12);
    assert!((diag_back.total_volume() - 0.5).abs() < 1e-12);
    assert_clip_invariants(
        &tet_lin,
        &diag_front,
        [0.5, 0.5, 0.5],
        [inv_sqrt2, -inv_sqrt2, 0.0],
        ClipMode::Exact,
    );

    // 2. Plane at z = 5.0 with normal +Z misses `[0, 1]^3` completely -> empty TetView;
    // with `keep = Back`, keeps the entire `[0, 1]^3` volume (1.0).
    let empty_clip = clip(
        &view,
        &ClipParams::with_origin([0.5, 0.5, 5.0], [0.0, 0.0, 1.0]),
    )
    .unwrap();
    assert_eq!(empty_clip.num_points(), 0);
    assert_eq!(empty_clip.num_tets(), 0);
    assert_eq!(empty_clip.total_volume(), 0.0);
    assert!(empty_clip.point_data().get("linear").is_some());
    assert!(empty_clip.cell_data().get("cell_id").is_some());

    let full_clip = clip_grid(
        &tet_lin,
        &ClipParams::with_origin([0.5, 0.5, 5.0], [0.0, 0.0, 1.0]).keep(ClipKeep::Back),
        LinearizeParams::default(),
    )
    .unwrap();
    assert_eq!(full_clip.num_cells(), view.num_tets());

    // 3. Invalid plane normal / origin errors.
    assert_eq!(
        clip(&view, &ClipParams::new([0.0, 0.0, 0.0])),
        Err(FilterError::InvalidPlaneNormal {
            normal: [0.0, 0.0, 0.0]
        })
    );
    assert!(matches!(
        clip(&view, &ClipParams::new([f64::NAN, 0.0, 1.0])),
        Err(FilterError::InvalidPlaneNormal { .. })
    ));
    assert!(matches!(
        clip(
            &view,
            &ClipParams::with_origin([f64::NEG_INFINITY, 0.0, 0.0], [0.0, 0.0, 1.0])
        ),
        Err(FilterError::InvalidPlaneOrigin { .. })
    ));
}

#[test]
fn clip_is_bit_for_bit_deterministic_across_chunk_sizes_in_both_modes() {
    let grid = SyntheticGridSpec::new(10, 8, 6, SyntheticCellLayout::Hexahedra)
        .generate()
        .unwrap();
    let view = linearize_default(&grid).unwrap();

    for mode in [ClipMode::Exact, ClipMode::Crinkle] {
        let params = ClipParams::with_origin([0.45, 0.52, 0.48], [0.3, -0.5, 0.8]).mode(mode);
        let single_chunk = clip_with_chunk_size(&view, &params, usize::MAX).unwrap();
        let chunk_64 = clip_with_chunk_size(&view, &params, 64).unwrap();
        let chunk_7 = clip_with_chunk_size(&view, &params, 7).unwrap();
        let chunk_1 = clip_with_chunk_size(&view, &params, 1).unwrap();

        assert!(single_chunk.num_tets() > 100);
        assert_eq!(single_chunk, chunk_64);
        assert_eq!(single_chunk, chunk_7);
        assert_eq!(single_chunk, chunk_1);
    }
}

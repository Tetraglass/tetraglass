//! Benchmark harness for the M2 synthetic benchmark mesh (`M1.9`, D-019).
//!
//! Run with:
//! ```sh
//! cargo bench -p tetraglass-filters --bench surface_bench
//! ```

use std::env;
use std::time::Instant;

use tetraglass_core::{
    M2_PERFORMANCE_TARGET, SyntheticCellLayout, SyntheticGridSpec, m2_benchmark_hex_spec,
    m2_benchmark_tet_spec,
};
use tetraglass_filters::{
    ClipKeep, ClipMode, ClipParams, ContourParams, SliceParams, TetLocator, ThresholdParams, clip,
    contour, extract_external_surface, extract_tet_view_surface, linearize_default, slice,
    threshold,
};

fn main() {
    let is_full_bench = env::args().any(|a| a == "--bench");

    if !is_full_bench {
        // Fast smoke mode when invoked by `cargo test --benches`:
        let small = SyntheticGridSpec::new(6, 6, 5, SyntheticCellLayout::Tetrahedra6);
        let grid = small.generate().expect("generate small synthetic mesh");
        let surf = extract_external_surface(&grid).expect("extract external surface");
        assert_eq!(surf.num_faces(), small.expected_boundary_faces());
        let view = linearize_default(&grid).expect("linearize small synthetic mesh");
        assert_eq!(view.num_tets(), small.num_cells());
        let locator = TetLocator::new(&view);
        assert!(locator.locate([0.5, 0.5, 0.5]).is_some());
        let cut = slice(&view, &SliceParams::new([0.0, 0.0, 1.0])).expect("slice small mesh");
        assert!(cut.num_triangles() > 0);
        return;
    }

    println!(
        "=== Tetraglass M2 Performance Target Benchmark (D-019) ===\n\
         Target divisions: {:?} ({} hexes / {} tets, {} points)\n\
         Target viewport frame time: <= {:.1} ms ({}x{}, {}x MSAA)\n\
         Target interactive slice/contour latency: <= {:.1} ms\n\
         Target volume filter / surface extraction latency: <= {:.1} ms\n",
        M2_PERFORMANCE_TARGET.divisions,
        M2_PERFORMANCE_TARGET.num_hex_cells,
        M2_PERFORMANCE_TARGET.num_tet_cells,
        M2_PERFORMANCE_TARGET.num_points,
        M2_PERFORMANCE_TARGET.max_frame_time_ms,
        M2_PERFORMANCE_TARGET.viewport_size[0],
        M2_PERFORMANCE_TARGET.viewport_size[1],
        M2_PERFORMANCE_TARGET.msaa_samples,
        M2_PERFORMANCE_TARGET.max_slice_contour_ms,
        M2_PERFORMANCE_TARGET.max_volume_filter_ms,
    );

    let hex_spec = m2_benchmark_hex_spec();
    let t0 = Instant::now();
    let hex_grid = hex_spec.generate().expect("generate M2 hex benchmark grid");
    let hex_gen_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let hex_surf = extract_external_surface(&hex_grid).expect("extract hex external surface");
    let hex_surf_ms = t1.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(hex_surf.num_faces(), hex_spec.expected_boundary_faces());

    let t1_lin = Instant::now();
    let hex_lin = linearize_default(&hex_grid).expect("linearize 100k hexes");
    let hex_lin_ms = t1_lin.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(hex_lin.num_tets(), hex_spec.num_cells() * 6);

    println!(
        "[Hexahedra 100k] generate: {hex_gen_ms:.2} ms | linearize: {hex_lin_ms:.2} ms ({} tets) | extract_external_surface: {hex_surf_ms:.2} ms ({} boundary faces)",
        hex_lin.num_tets(),
        hex_surf.num_faces()
    );

    let tet_spec = m2_benchmark_tet_spec();
    let t2 = Instant::now();
    let tet_grid = tet_spec.generate().expect("generate M2 tet benchmark grid");
    let tet_gen_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = Instant::now();
    let tet_surf = extract_external_surface(&tet_grid).expect("extract tet external surface");
    let tet_surf_ms = t3.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(tet_surf.num_faces(), tet_spec.expected_boundary_faces());

    let t3_lin = Instant::now();
    let tet_lin = linearize_default(&tet_grid).expect("linearize 600k tets");
    let tet_lin_ms = t3_lin.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(tet_lin.num_tets(), tet_spec.num_cells());

    println!(
        "[Tetrahedra 600k] generate: {tet_gen_ms:.2} ms | linearize: {tet_lin_ms:.2} ms ({} tets) | extract_external_surface: {tet_surf_ms:.2} ms ({} boundary tris)",
        tet_lin.num_tets(),
        tet_surf.num_triangles()
    );

    // Interactive plane slice on 600k tets
    let t_slice = Instant::now();
    let slice_surf =
        slice(&tet_lin, &SliceParams::new([0.6, -0.48, 0.64])).expect("slice 600k tets");
    let slice_ms = t_slice.elapsed().as_secs_f64() * 1000.0;

    // Interactive single-isovalue contour on 600k tets
    let t_contour = Instant::now();
    let contour_surf =
        contour(&tet_lin, &ContourParams::new("radius", vec![0.42])).expect("contour 600k tets");
    let contour_ms = t_contour.elapsed().as_secs_f64() * 1000.0;

    println!(
        "[Interactive <= {:.1} ms] slice: {slice_ms:.2} ms ({} tris) | contour: {contour_ms:.2} ms ({} tris)",
        M2_PERFORMANCE_TARGET.max_slice_contour_ms,
        slice_surf.num_triangles(),
        contour_surf.num_triangles()
    );

    // Exact plane clip + surface extraction on 600k tets (oblique plane cutting tets into prisms)
    let t_clip_exact = Instant::now();
    let clipped_exact = clip(
        &tet_lin,
        &ClipParams::new([0.6, -0.48, 0.64])
            .keep(ClipKeep::Back)
            .mode(ClipMode::Exact),
    )
    .expect("exact clip 600k tets");
    let clip_exact_vol_ms = t_clip_exact.elapsed().as_secs_f64() * 1000.0;
    let t_clip_exact_surf = Instant::now();
    let clipped_exact_surf =
        extract_tet_view_surface(&clipped_exact).expect("extract exact clipped surface");
    let clip_exact_surf_ms = t_clip_exact_surf.elapsed().as_secs_f64() * 1000.0;
    let clip_exact_total_ms = clip_exact_vol_ms + clip_exact_surf_ms;

    // Crinkle plane clip + surface extraction on 600k tets
    let t_clip_crinkle = Instant::now();
    let clipped_crinkle = clip(
        &tet_lin,
        &ClipParams::new([0.6, -0.48, 0.64])
            .keep(ClipKeep::Back)
            .mode(ClipMode::Crinkle),
    )
    .expect("crinkle clip 600k tets");
    let clip_crinkle_vol_ms = t_clip_crinkle.elapsed().as_secs_f64() * 1000.0;
    let t_clip_crinkle_surf = Instant::now();
    let clipped_crinkle_surf =
        extract_tet_view_surface(&clipped_crinkle).expect("extract crinkle clipped surface");
    let clip_crinkle_surf_ms = t_clip_crinkle_surf.elapsed().as_secs_f64() * 1000.0;
    let clip_crinkle_total_ms = clip_crinkle_vol_ms + clip_crinkle_surf_ms;

    // Threshold + surface extraction on 600k tets
    let t_thresh = Instant::now();
    let threshed = threshold(
        &tet_lin,
        &ThresholdParams::new("linear", Some(0.0), Some(3.0)),
    )
    .expect("threshold 600k tets");
    let thresh_vol_ms = t_thresh.elapsed().as_secs_f64() * 1000.0;
    let t_thresh_surf = Instant::now();
    let threshed_surf = extract_tet_view_surface(&threshed).expect("extract thresholded surface");
    let thresh_surf_ms = t_thresh_surf.elapsed().as_secs_f64() * 1000.0;
    let thresh_total_ms = thresh_vol_ms + thresh_surf_ms;

    println!(
        "[Volume <= {:.1} ms]\n  \
         clip (exact):   {clip_exact_vol_ms:.2} ms vol + {clip_exact_surf_ms:.2} ms surf = {clip_exact_total_ms:.2} ms ({} tets -> {} tris)\n  \
         clip (crinkle): {clip_crinkle_vol_ms:.2} ms vol + {clip_crinkle_surf_ms:.2} ms surf = {clip_crinkle_total_ms:.2} ms ({} tets -> {} tris)\n  \
         threshold:      {thresh_vol_ms:.2} ms vol + {thresh_surf_ms:.2} ms surf = {thresh_total_ms:.2} ms ({} tets -> {} tris)",
        M2_PERFORMANCE_TARGET.max_volume_filter_ms,
        clipped_exact.num_tets(),
        clipped_exact_surf.num_triangles(),
        clipped_crinkle.num_tets(),
        clipped_crinkle_surf.num_triangles(),
        threshed.num_tets(),
        threshed_surf.num_triangles(),
    );

    // BVH cell locator build and point location on 600k tets (`M4.1`)
    let t_bvh_build = Instant::now();
    let locator = TetLocator::new(&tet_lin);
    let bvh_build_ms = t_bvh_build.elapsed().as_secs_f64() * 1000.0;

    let mut sample_pts = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        let u = (f64::from(i) + 0.5) / 10_000.0;
        let x = 0.05 + 0.90 * u;
        let y = 0.50 + 0.40 * (u * 20.0).sin();
        let z = 0.50 + 0.40 * (u * 20.0).cos();
        sample_pts.push([x, y, z]);
    }

    let t_bvh_query = Instant::now();
    let batch_hits = locator.locate_many(&sample_pts);
    let bvh_batch_ms = t_bvh_query.elapsed().as_secs_f64() * 1000.0;
    assert!(batch_hits.iter().all(Option::is_some));

    let t_bvh_hint = Instant::now();
    let mut hint = None;
    for &pt in &sample_pts {
        let hit = locator
            .locate_with_hint(pt, hint)
            .expect("locate helical point");
        hint = Some(hit.tet_index);
    }
    let bvh_hint_ms = t_bvh_hint.elapsed().as_secs_f64() * 1000.0;

    println!(
        "[BVH Cell Locator 600k tets] build: {bvh_build_ms:.2} ms ({} nodes) | 10k batch locate: {bvh_batch_ms:.2} ms | 10k hinted walk: {bvh_hint_ms:.2} ms",
        locator.num_nodes(),
    );

    assert!(
        slice_ms <= M2_PERFORMANCE_TARGET.max_slice_contour_ms,
        "slice latency {slice_ms:.2} ms exceeded target {:.1} ms",
        M2_PERFORMANCE_TARGET.max_slice_contour_ms
    );
    assert!(
        contour_ms <= M2_PERFORMANCE_TARGET.max_slice_contour_ms,
        "contour latency {contour_ms:.2} ms exceeded target {:.1} ms",
        M2_PERFORMANCE_TARGET.max_slice_contour_ms
    );
    for (label, ms) in [
        ("hex linearize", hex_lin_ms),
        ("hex extract_external_surface", hex_surf_ms),
        ("tet linearize", tet_lin_ms),
        ("tet extract_external_surface", tet_surf_ms),
        ("clip (exact) + surface", clip_exact_total_ms),
        ("clip (crinkle) + surface", clip_crinkle_total_ms),
        ("threshold + surface", thresh_total_ms),
    ] {
        assert!(
            ms <= M2_PERFORMANCE_TARGET.max_volume_filter_ms,
            "{label} latency {ms:.2} ms exceeded target {:.1} ms",
            M2_PERFORMANCE_TARGET.max_volume_filter_ms
        );
    }
}

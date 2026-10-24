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
use tetraglass_filters::extract_external_surface;

fn main() {
    let is_full_bench = env::args().any(|a| a == "--bench");

    if !is_full_bench {
        // Fast smoke mode when invoked by `cargo test --benches`:
        let small = SyntheticGridSpec::new(6, 6, 5, SyntheticCellLayout::Tetrahedra6);
        let grid = small.generate().expect("generate small synthetic mesh");
        let surf = extract_external_surface(&grid).expect("extract external surface");
        assert_eq!(surf.num_faces(), small.expected_boundary_faces());
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

    println!(
        "[Hexahedra 100k] generate: {hex_gen_ms:.2} ms | extract_external_surface: {hex_surf_ms:.2} ms ({} boundary faces)",
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

    println!(
        "[Tetrahedra 600k] generate: {tet_gen_ms:.2} ms | extract_external_surface: {tet_surf_ms:.2} ms ({} boundary tris)",
        tet_surf.num_triangles()
    );
}

//! GPU viewport rendering benchmark for the M2 performance target (`M1.9`, `M2.9`, `D-019`).
//!
//! Run with:
//! ```sh
//! cargo bench -p tetraglass-render --bench render_bench
//! ```

use std::env;
use std::time::Instant;

use tetraglass_core::{
    M2_PERFORMANCE_TARGET, SyntheticCellLayout, SyntheticGridSpec, m2_benchmark_tet_spec,
};
use tetraglass_filters::{SliceParams, extract_external_surface, linearize_default, slice};
use tetraglass_render::{
    Annotations, Camera, ColorBy, ColorbarAnnotation, ColormapKind, DEFAULT_COLOR_FORMAT,
    HeadlessContext, OffscreenTarget, Renderer, ShaderClipKeep, ShaderClipPlane, SurfaceStyle,
};

fn main() {
    let is_full_bench = env::args().any(|a| a == "--bench");

    let Ok(ctx) = HeadlessContext::new() else {
        println!("Skipping render_bench: no compatible wgpu adapter available.");
        return;
    };

    if !is_full_bench {
        // Fast smoke mode when invoked by `cargo test --benches`:
        let small = SyntheticGridSpec::new(4, 4, 3, SyntheticCellLayout::Tetrahedra6);
        let grid = small.generate().expect("generate small mesh");
        let surf = extract_external_surface(&grid).expect("extract surface");
        let renderer = Renderer::new(ctx.device(), ctx.queue(), DEFAULT_COLOR_FORMAT, 1)
            .expect("create renderer");
        let target = OffscreenTarget::new(ctx.device(), 160, 120, DEFAULT_COLOR_FORMAT, 1)
            .expect("create target");
        let style = SurfaceStyle {
            color_by: ColorBy::point("linear"),
            colormap: ColormapKind::Viridis,
            range: Some([0.0, 6.0]),
            ..SurfaceStyle::default()
        };
        let gpu_surf = renderer
            .upload_surface(ctx.device(), &surf, &style)
            .expect("upload surface");
        let cam = Camera::fit_bounds(surf.bounds().expect("bounds"), 35.0, 25.0);
        let mut enc = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer
            .render(
                &mut enc,
                ctx.queue(),
                &target,
                &cam,
                [0.1, 0.11, 0.13, 1.0],
                &[&gpu_surf],
            )
            .expect("render smoke frame");
        ctx.queue().submit(std::iter::once(enc.finish()));
        let _ = ctx.device().poll(wgpu::PollType::wait_indefinitely());
        return;
    }

    let [width, height] = M2_PERFORMANCE_TARGET.viewport_size;
    let msaa_samples = M2_PERFORMANCE_TARGET.msaa_samples;
    println!(
        "=== Tetraglass M2 GPU Render Benchmark (D-019) ===\n\
         Adapter: {} ({:?})\n\
         Viewport: {}x{}, {}x MSAA\n\
         Target frame time: <= {:.1} ms (60 FPS)\n",
        ctx.adapter_info().name,
        ctx.adapter_info().backend,
        width,
        height,
        msaa_samples,
        M2_PERFORMANCE_TARGET.max_frame_time_ms,
    );

    let tet_spec = m2_benchmark_tet_spec();
    let grid = tet_spec.generate().expect("generate M2 tet benchmark grid");
    let ext_surf = extract_external_surface(&grid).expect("extract tet external surface");
    let tet_view = linearize_default(&grid).expect("linearize 600k tets");
    let slice_surf =
        slice(&tet_view, &SliceParams::new([0.6, -0.48, 0.64])).expect("slice 600k tets");
    let bounds = ext_surf.bounds().expect("non-empty bounds");

    let renderer = Renderer::new(
        ctx.device(),
        ctx.queue(),
        DEFAULT_COLOR_FORMAT,
        msaa_samples,
    )
    .expect("create 4x MSAA renderer");
    let target = OffscreenTarget::new(
        ctx.device(),
        width,
        height,
        DEFAULT_COLOR_FORMAT,
        msaa_samples,
    )
    .expect("create 4x MSAA target");

    let mut style = SurfaceStyle {
        color_by: ColorBy::point("linear"),
        colormap: ColormapKind::Viridis,
        range: Some([0.0, 6.0]),
        use_log_scale: false,
        solid_color_srgb: [0.82, 0.83, 0.86, 1.0],
        draw_surface: true,
        draw_edges: true,
        edge_color_srgb: [0.08, 0.09, 0.11, 1.0],
        clip_plane: None,
        ..SurfaceStyle::default()
    };

    let t_upload = Instant::now();
    let mut gpu_ext = renderer
        .upload_surface_with_center(ctx.device(), &ext_surf, bounds.center(), &style)
        .expect("upload external surface");
    let gpu_slice = renderer
        .upload_surface_with_center(ctx.device(), &slice_surf, bounds.center(), &style)
        .expect("upload slice surface");
    let _ = ctx.device().poll(wgpu::PollType::wait_indefinitely());
    let upload_ms = t_upload.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Uploaded external surface ({} tris) + slice ({} tris) in {upload_ms:.2} ms",
        ext_surf.num_triangles(),
        slice_surf.num_triangles()
    );

    let annotations = Annotations {
        title: Some("M2 600k Tet Benchmark".to_owned()),
        axes: true,
        colorbars: vec![
            ColorbarAnnotation::new(ColormapKind::Viridis, [0.0, 6.0]).with_title("linear"),
        ],
    };

    let mut camera = Camera::fit_bounds(bounds, 35.0, 25.0);
    let background = [0.09, 0.10, 0.12, 1.0];

    // Warm-up frames
    for _ in 0..5 {
        let mut enc = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer
            .render_with_annotations(
                ctx.device(),
                &mut enc,
                ctx.queue(),
                &target,
                &camera,
                background,
                &[&gpu_ext, &gpu_slice],
                &annotations,
            )
            .expect("warmup frame");
        ctx.queue().submit(std::iter::once(enc.finish()));
        let _ = ctx.device().poll(wgpu::PollType::wait_indefinitely());
    }

    let num_frames = 60u32;

    // 1. Camera orbit / pan / zoom with annotations + wireframe edges
    let t_orbit = Instant::now();
    for frame in 0..num_frames {
        camera.orbit(2.5, if frame % 2 == 0 { 0.6 } else { -0.6 });
        camera.pan(0.001, -0.001);
        camera.zoom_by(if frame % 2 == 0 { 0.998 } else { 1.002 });
        let mut enc = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer
            .render_with_annotations(
                ctx.device(),
                &mut enc,
                ctx.queue(),
                &target,
                &camera,
                background,
                &[&gpu_ext, &gpu_slice],
                &annotations,
            )
            .expect("orbit frame");
        ctx.queue().submit(std::iter::once(enc.finish()));
        let _ = ctx.device().poll(wgpu::PollType::wait_indefinitely());
    }
    let orbit_avg_ms = (t_orbit.elapsed().as_secs_f64() * 1000.0) / f64::from(num_frames);

    // 2. Colormap / range uniform updates (`update_surface_style`, no geometry re-upload)
    let colormaps = [
        ColormapKind::Viridis,
        ColormapKind::Cividis,
        ColormapKind::CoolWarm,
    ];
    let t_style = Instant::now();
    for frame in 0..num_frames {
        let cm = colormaps[(frame as usize) % colormaps.len()];
        let max_val = 4.0 + 2.0 * (f64::from(frame) / f64::from(num_frames));
        style.colormap = cm;
        style.range = Some([0.0, max_val]);
        renderer
            .update_surface_style(ctx.queue(), &mut gpu_ext, &ext_surf, &style)
            .expect("update style");
        let mut enc = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer
            .render_with_annotations(
                ctx.device(),
                &mut enc,
                ctx.queue(),
                &target,
                &camera,
                background,
                &[&gpu_ext, &gpu_slice],
                &annotations,
            )
            .expect("style frame");
        ctx.queue().submit(std::iter::once(enc.finish()));
        let _ = ctx.device().poll(wgpu::PollType::wait_indefinitely());
    }
    let style_avg_ms = (t_style.elapsed().as_secs_f64() * 1000.0) / f64::from(num_frames);

    // 3. Interactive shader clip-plane dragging (`set_surface_clip_plane`)
    let t_clip = Instant::now();
    for frame in 0..num_frames {
        let frac = f64::from(frame) / f64::from(num_frames);
        let plane = ShaderClipPlane {
            origin: [0.2 + 0.6 * frac, 0.5, 0.5],
            normal: [0.6, -0.48, 0.64],
            keep: ShaderClipKeep::Back,
        };
        renderer.set_surface_clip_plane(ctx.queue(), &mut gpu_ext, Some(plane));
        let mut enc = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer
            .render_with_annotations(
                ctx.device(),
                &mut enc,
                ctx.queue(),
                &target,
                &camera,
                background,
                &[&gpu_ext, &gpu_slice],
                &annotations,
            )
            .expect("shader clip frame");
        ctx.queue().submit(std::iter::once(enc.finish()));
        let _ = ctx.device().poll(wgpu::PollType::wait_indefinitely());
    }
    renderer.set_surface_clip_plane(ctx.queue(), &mut gpu_ext, None);
    let clip_avg_ms = (t_clip.elapsed().as_secs_f64() * 1000.0) / f64::from(num_frames);

    println!(
        "[Frame time <= {:.1} ms ({} frames synchronous GPU poll)]\n  \
         orbit/pan/zoom + annotations:   {orbit_avg_ms:.2} ms/frame ({:.1} FPS)\n  \
         colormap/range uniform update:  {style_avg_ms:.2} ms/frame ({:.1} FPS)\n  \
         shader clip-plane drag update:  {clip_avg_ms:.2} ms/frame ({:.1} FPS)",
        M2_PERFORMANCE_TARGET.max_frame_time_ms,
        num_frames,
        1000.0 / orbit_avg_ms,
        1000.0 / style_avg_ms,
        1000.0 / clip_avg_ms,
    );

    for (label, ms) in [
        ("orbit/pan/zoom", orbit_avg_ms),
        ("colormap/range update", style_avg_ms),
        ("shader clip-plane drag", clip_avg_ms),
    ] {
        assert!(
            ms <= M2_PERFORMANCE_TARGET.max_frame_time_ms,
            "{label} average frame time {ms:.2} ms exceeded target {:.1} ms",
            M2_PERFORMANCE_TARGET.max_frame_time_ms
        );
    }
}

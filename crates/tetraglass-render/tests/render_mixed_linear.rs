//! Headless rendering integration tests for `tetraglass-render` (M0.7).

use std::path::{Path, PathBuf};

use tetraglass_filters::extract_external_surface;
use tetraglass_io::read_vtu_from_path;
use tetraglass_render::{
    Camera, ColorBy, ColormapKind, DEFAULT_COLOR_FORMAT, HeadlessContext, HeadlessRenderOptions,
    OffscreenTarget, RenderedImage, Renderer, SsaoConfig, SurfaceStyle,
};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn assert_matches_baseline(root: &Path, file_name: &str, image: &RenderedImage) {
    let baseline_path = root.join("tests/baselines").join(file_name);
    assert!(
        baseline_path.exists(),
        "missing baseline image at {}",
        baseline_path.display()
    );
    let baseline = RenderedImage::load_png(&baseline_path).expect("load baseline PNG");
    let cmp = image
        .compare_with_baseline(&baseline)
        .expect("compare with baseline");
    if !cmp.passed {
        let diff_path = root.join("target/render-diffs").join(file_name);
        cmp.diff_image
            .save_png(&diff_path)
            .expect("write diff image");
        panic!(
            "{file_name} differed from baseline in {} / {} pixels ({:.3}%, max channel diff {}), diff written to {}",
            cmp.differing_pixels,
            cmp.total_pixels,
            cmp.diff_fraction * 100.0,
            cmp.max_channel_diff,
            diff_path.display()
        );
    }
}

#[test]
fn render_mixed_linear_colored_by_linear_and_cell_id() {
    let root = workspace_root();
    let vtu_path = root.join("testdata/vtk/mixed_linear.vtu");
    let grid = read_vtu_from_path(&vtu_path).expect("load mixed_linear.vtu");
    let surface = extract_external_surface(&grid).expect("extract external surface");

    let bounds = surface.bounds().expect("non-empty surface bounds");
    let camera = Camera::fit_bounds(bounds, 30.0, 20.0);

    let ctx = HeadlessContext::new().expect("create headless wgpu context");
    let options = HeadlessRenderOptions {
        width: 800,
        height: 600,
        sample_count: 4,
        background_srgb: [1.0, 1.0, 1.0, 1.0],
        style: SurfaceStyle {
            color_by: ColorBy::point("linear"),
            colormap: ColormapKind::Viridis,
            range: None,
            use_log_scale: false,
            solid_color_srgb: [0.8, 0.8, 0.8, 1.0],
            ..SurfaceStyle::default()
        },
        camera: Some(camera.clone()),
        ..HeadlessRenderOptions::default()
    };

    let image = ctx
        .render_surface(&surface, &options)
        .expect("render mixed_linear colored by linear");
    assert_eq!(image.width(), 800);
    assert_eq!(image.height(), 600);

    let out_dir = root.join("target/render-tests");
    let out_path = out_dir.join("mixed_linear_linear.png");
    image
        .save_png(&out_path)
        .expect("save mixed_linear_linear.png");

    // Verify PNG round-trip from disk.
    let reloaded = RenderedImage::load_png(&out_path).expect("reload saved PNG");
    assert_eq!(reloaded, image);

    // Corner pixels must be pure white background.
    for &(x, y) in &[(0, 0), (799, 0), (0, 599), (799, 599)] {
        assert_eq!(image.pixel(x, y), [255, 255, 255, 255]);
    }

    // Analyze foreground pixels to verify geometry coverage and Viridis gradient from left (-X) to right (+X).
    let mut non_white_count = 0usize;
    let mut left_cool_pixels = 0usize;
    let mut right_warm_pixels = 0usize;
    for y in 0..image.height() {
        for x in 0..image.width() {
            let [r, g, b, a] = image.pixel(x, y);
            assert_eq!(a, 255);
            if r < 250 || g < 250 || b < 250 {
                non_white_count += 1;
                if x < 400 && b > r.saturating_add(20) {
                    left_cool_pixels += 1;
                }
                if x >= 400 && g > b.saturating_add(30) && r > b {
                    right_warm_pixels += 1;
                }
            }
        }
    }

    let total_pixels = (image.width() * image.height()) as usize;
    let coverage = (non_white_count as f64) / (total_pixels as f64);
    assert!(
        coverage > 0.08 && coverage < 0.70,
        "unexpected foreground mesh coverage: {coverage:.3}"
    );
    assert!(
        left_cool_pixels > 500,
        "expected Viridis blue/purple pixels on low-scalar left side, got {left_cool_pixels}"
    );
    assert!(
        right_warm_pixels > 500,
        "expected Viridis green/yellow pixels on high-scalar right side, got {right_warm_pixels}"
    );

    // Determinism check: a second render with identical inputs produces identical pixels.
    let image_second = ctx
        .render_surface(&surface, &options)
        .expect("second render");
    let cmp_self = image
        .compare_with_baseline(&image_second)
        .expect("compare two identical renders");
    assert!(cmp_self.passed);
    assert_eq!(cmp_self.differing_pixels, 0);
    assert_eq!(cmp_self.max_channel_diff, 0);

    // Also test flat cell-data rendering (`cell_id`) and style switching on an existing GpuSurface.
    let renderer =
        Renderer::new(ctx.device(), ctx.queue(), DEFAULT_COLOR_FORMAT, 4).expect("create renderer");
    let target =
        OffscreenTarget::new(ctx.device(), 800, 600, DEFAULT_COLOR_FORMAT, 4).expect("target");
    let mut gpu_surface = renderer
        .upload_surface(ctx.device(), &surface, &options.style)
        .expect("upload surface");

    let cell_style = SurfaceStyle {
        color_by: ColorBy::cell("cell_id"),
        colormap: ColormapKind::Cividis,
        range: None,
        use_log_scale: false,
        solid_color_srgb: [0.8, 0.8, 0.8, 1.0],
        ..SurfaceStyle::default()
    };
    renderer
        .update_surface_style(ctx.queue(), &mut gpu_surface, &surface, &cell_style)
        .expect("update style to cell_id");

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    renderer
        .render(
            &mut encoder,
            ctx.queue(),
            &target,
            &camera,
            [1.0, 1.0, 1.0, 1.0],
            &[&gpu_surface],
        )
        .expect("render cell_id");
    ctx.queue().submit(std::iter::once(encoder.finish()));

    let cell_image = ctx.readback_target(&target).expect("readback cell_id");
    let cell_out_path = out_dir.join("mixed_linear_cell_id.png");
    cell_image
        .save_png(&cell_out_path)
        .expect("save mixed_linear_cell_id.png");

    // Point-colored and cell-colored images must differ significantly.
    let cmp_point_vs_cell = image
        .compare_with_baseline(&cell_image)
        .expect("compare point vs cell coloring");
    assert!(
        cmp_point_vs_cell.differing_pixels > 5000,
        "expected cell_id rendering to differ from point linear rendering"
    );

    // Compare both rendered images against the committed baselines in `tests/baselines/`.
    assert_matches_baseline(&root, "mixed_linear_linear.png", &image);
    assert_matches_baseline(&root, "mixed_linear_cell_id.png", &cell_image);
}

#[test]
fn render_validates_errors_and_diff_highlighting() {
    let root = workspace_root();
    let vtu_path = root.join("testdata/vtk/mixed_linear.vtu");
    let grid = read_vtu_from_path(&vtu_path).expect("load mixed_linear.vtu");
    let surface = extract_external_surface(&grid).expect("extract surface");

    let ctx = HeadlessContext::new().expect("headless context");

    // Zero dimensions must return InvalidDimensions error.
    let bad_target = OffscreenTarget::new(ctx.device(), 0, 100, DEFAULT_COLOR_FORMAT, 4);
    assert!(bad_target.is_err());

    // Unsupported sample count (e.g. 2) must return UnsupportedSampleCount error.
    let bad_msaa = OffscreenTarget::new(ctx.device(), 100, 100, DEFAULT_COLOR_FORMAT, 2);
    assert!(bad_msaa.is_err());

    // Non-existent array must return MissingArray error.
    let renderer = Renderer::new(ctx.device(), ctx.queue(), DEFAULT_COLOR_FORMAT, 1)
        .expect("create 1x renderer");
    let bad_style = SurfaceStyle {
        color_by: ColorBy::point("no_such_field"),
        ..SurfaceStyle::default()
    };
    assert!(
        renderer
            .upload_surface(ctx.device(), &surface, &bad_style)
            .is_err()
    );

    // Image comparison detects synthetic pixel perturbation above tolerance and highlights it in magenta.
    let img_a = RenderedImage::new(4, 4, vec![100u8; 4 * 4 * 4]).expect("image a");
    let mut bytes_b = vec![100u8; 4 * 4 * 4];
    bytes_b[0] = 110; // differs by 10 (> 2 tolerance) at pixel (0, 0)
    let img_b = RenderedImage::new(4, 4, bytes_b).expect("image b");
    let cmp = img_a.compare_with_baseline(&img_b).expect("compare");
    assert!(!cmp.passed);
    assert_eq!(cmp.differing_pixels, 1);
    assert_eq!(cmp.max_channel_diff, 10);
    assert_eq!(cmp.diff_image.pixel(0, 0), [255, 0, 255, 255]);
}

#[test]
fn render_surface_with_edges_and_wireframe_draws_original_element_edges() {
    let root = workspace_root();
    let vtu_path = root.join("testdata/vtk/tet_quadratic.vtu");
    let grid = read_vtu_from_path(&vtu_path).expect("load tet_quadratic.vtu");
    let surface = extract_external_surface(&grid).expect("extract curved surface");
    assert_eq!(surface.num_faces(), 12);
    assert_eq!(surface.num_triangles(), 48);
    assert_eq!(surface.num_wireframe_edges(), 36);

    let bounds = surface.bounds().expect("surface bounds");
    let camera = Camera::fit_bounds(bounds, 35.0, 25.0);
    let ctx = HeadlessContext::new().expect("headless context");

    let base_opts = HeadlessRenderOptions {
        width: 400,
        height: 300,
        sample_count: 4,
        background_srgb: [1.0, 1.0, 1.0, 1.0],
        style: SurfaceStyle {
            color_by: ColorBy::point("linear"),
            colormap: ColormapKind::Viridis,
            ..SurfaceStyle::default()
        },
        camera: Some(camera.clone()),
        ..HeadlessRenderOptions::default()
    };
    let img_surface = ctx
        .render_surface(&surface, &base_opts)
        .expect("render surface");

    let mut edges_opts = base_opts.clone();
    edges_opts.style.draw_edges = true;
    edges_opts.style.edge_color_srgb = [0.0, 0.0, 0.0, 1.0];
    let img_with_edges = ctx
        .render_surface(&surface, &edges_opts)
        .expect("render surface_with_edges");

    let mut wire_opts = edges_opts.clone();
    wire_opts.style.draw_surface = false;
    let img_wireframe = ctx
        .render_surface(&surface, &wire_opts)
        .expect("render wireframe");

    let out_dir = root.join("target/render-tests");
    img_with_edges
        .save_png(out_dir.join("tet_quadratic_surface_with_edges.png"))
        .expect("save tet_quadratic_surface_with_edges.png");
    img_wireframe
        .save_png(out_dir.join("tet_quadratic_wireframe.png"))
        .expect("save tet_quadratic_wireframe.png");

    let cmp_edges = img_surface
        .compare(&img_with_edges, 2, 0.0)
        .expect("compare surface vs surface_with_edges");
    assert!(
        cmp_edges.differing_pixels > 200,
        "expected surface_with_edges to draw visible edge pixels, got {}",
        cmp_edges.differing_pixels
    );

    // Wireframe-only image must have dark line pixels on white background, but far fewer
    // non-white pixels than the filled surface render.
    let count_non_white = |img: &RenderedImage| -> usize {
        img.as_rgba8()
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[0] < 250 || px[1] < 250 || px[2] < 250)
            .count()
    };
    let wire_non_white = count_non_white(&img_wireframe);
    let surf_non_white = count_non_white(&img_surface);
    assert!(
        wire_non_white > 200 && wire_non_white * 3 < surf_non_white,
        "unexpected wireframe pixel count {wire_non_white} vs surface {surf_non_white}"
    );
}

#[test]
fn weighted_blended_oit_is_order_independent_and_composites_over_opaque_geometry() {
    use tetraglass_core::{Attributes, SurfaceMesh};
    use tetraglass_filters::{LinearizeParams, SliceParams, linearize, slice};

    let root = workspace_root();
    let hex_path = root.join("testdata/vtk/hex_linear.vtu");
    let grid = read_vtu_from_path(&hex_path).expect("load hex_linear.vtu");
    let shell_surface = extract_external_surface(&grid).expect("extract shell surface");
    let tet_view = linearize(&grid, &LinearizeParams::default()).expect("linearize hex_linear");
    let slice_x = slice(
        &tet_view,
        &SliceParams {
            origin: Some([1.0, 1.0, 1.0]),
            normal: [1.0, 0.0, 0.0],
        },
    )
    .expect("slice x");
    let slice_z = slice(
        &tet_view,
        &SliceParams {
            origin: Some([1.0, 1.0, 1.0]),
            normal: [0.0, 0.0, 1.0],
        },
    )
    .expect("slice z");

    let bounds = shell_surface.bounds().expect("shell bounds");
    let dataset_center = bounds.center();
    let camera = Camera::fit_bounds(bounds, 35.0, 24.0);
    let ctx = HeadlessContext::new().expect("headless context");

    for sample_count in [1u32, 4u32] {
        let renderer = Renderer::new(
            ctx.device(),
            ctx.queue(),
            DEFAULT_COLOR_FORMAT,
            sample_count,
        )
        .expect("create renderer");
        let target =
            OffscreenTarget::new(ctx.device(), 400, 300, DEFAULT_COLOR_FORMAT, sample_count)
                .expect("create target");

        let render_surfaces = |surfaces: &[&tetraglass_render::GpuSurface]| -> RenderedImage {
            let mut encoder = ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            renderer
                .render(
                    &mut encoder,
                    ctx.queue(),
                    &target,
                    &camera,
                    [1.0, 1.0, 1.0, 1.0],
                    surfaces,
                )
                .expect("render surfaces");
            ctx.queue().submit(std::iter::once(encoder.finish()));
            ctx.readback_target(&target).expect("readback")
        };

        // 1. Monotonic opacity response on white background: 0.0 -> 0.35 -> 0.70 -> 1.0.
        let mut mean_luma_by_opacity = Vec::new();
        for &alpha in &[0.0f32, 0.35, 0.70, 1.0] {
            let style = SurfaceStyle {
                color_by: ColorBy::Solid,
                solid_color_srgb: [0.15, 0.30, 0.85, alpha],
                ..SurfaceStyle::default()
            };
            let gpu_surf = renderer
                .upload_surface_with_center(ctx.device(), &shell_surface, dataset_center, &style)
                .expect("upload shell");
            let img = render_surfaces(&[&gpu_surf]);
            if alpha == 0.0 {
                assert!(
                    img.as_rgba8()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|&px| px == [255, 255, 255, 255]),
                    "opacity 0.0 must leave background untouched"
                );
            }
            let sum_r: u64 = img
                .as_rgba8()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|px| u64::from(px[0]) + u64::from(px[1]) + u64::from(px[2]))
                .sum();
            mean_luma_by_opacity.push(sum_r);
        }
        assert!(
            mean_luma_by_opacity[0] > mean_luma_by_opacity[1]
                && mean_luma_by_opacity[1] > mean_luma_by_opacity[2]
                && mean_luma_by_opacity[2] > mean_luma_by_opacity[3],
            "expected monotonic darkening over white background as opacity increases: {mean_luma_by_opacity:?}"
        );

        // 2. Multi-surface order independence: `[slice_x, slice_z]` vs `[slice_z, slice_x]`.
        let gpu_slice_x = renderer
            .upload_surface_with_center(
                ctx.device(),
                &slice_x,
                dataset_center,
                &SurfaceStyle {
                    color_by: ColorBy::Solid,
                    solid_color_srgb: [0.90, 0.20, 0.20, 0.55],
                    ..SurfaceStyle::default()
                },
            )
            .expect("upload slice_x");
        let gpu_slice_z = renderer
            .upload_surface_with_center(
                ctx.device(),
                &slice_z,
                dataset_center,
                &SurfaceStyle {
                    color_by: ColorBy::Solid,
                    solid_color_srgb: [0.15, 0.45, 0.90, 0.55],
                    ..SurfaceStyle::default()
                },
            )
            .expect("upload slice_z");

        let img_xz = render_surfaces(&[&gpu_slice_x, &gpu_slice_z]);
        let img_zx = render_surfaces(&[&gpu_slice_z, &gpu_slice_x]);
        let cmp_order = img_xz
            .compare(&img_zx, 1, 0.0)
            .expect("compare [x, z] vs [z, x]");
        assert_eq!(
            cmp_order.differing_pixels, 0,
            "WBOIT must be independent of transparent surface draw order (sample_count={sample_count}, max_diff={})",
            cmp_order.max_channel_diff
        );

        // 3. Triangle-order independence inside a closed self-overlapping transparent mesh.
        let reversed_triangles: Vec<[u32; 3]> =
            shell_surface.triangles().iter().copied().rev().collect();
        let num_tris = reversed_triangles.len();
        let reversed_shell = SurfaceMesh::new(
            shell_surface.points().to_vec(),
            reversed_triangles,
            vec![0u32; num_tris],
            vec![0u32; num_tris],
            Attributes::new(shell_surface.num_points()),
            Attributes::new(num_tris),
        )
        .expect("build reversed-triangle shell");
        let trans_shell_style = SurfaceStyle {
            color_by: ColorBy::Solid,
            solid_color_srgb: [0.20, 0.65, 0.50, 0.45],
            ..SurfaceStyle::default()
        };
        let gpu_shell_fwd = renderer
            .upload_surface_with_center(
                ctx.device(),
                &shell_surface,
                dataset_center,
                &trans_shell_style,
            )
            .expect("upload forward shell");
        let gpu_shell_rev = renderer
            .upload_surface_with_center(
                ctx.device(),
                &reversed_shell,
                dataset_center,
                &trans_shell_style,
            )
            .expect("upload reversed shell");
        let img_fwd = render_surfaces(&[&gpu_shell_fwd]);
        let img_rev = render_surfaces(&[&gpu_shell_rev]);
        let cmp_tri_order = img_fwd
            .compare(&img_rev, 1, 0.0)
            .expect("compare forward vs reversed triangle order");
        assert_eq!(
            cmp_tri_order.differing_pixels, 0,
            "WBOIT must be independent of internal triangle order (sample_count={sample_count}, max_diff={})",
            cmp_tri_order.max_channel_diff
        );

        // 4. Opaque inner slice inside a semi-transparent outer shell:
        //    `[transparent_shell, opaque_slice]` vs `[opaque_slice, transparent_shell]` must match
        //    and show the inner slice through the outer shell.
        let gpu_opaque_slice = renderer
            .upload_surface_with_center(
                ctx.device(),
                &slice_x,
                dataset_center,
                &SurfaceStyle {
                    color_by: ColorBy::point("linear"),
                    colormap: ColormapKind::Viridis,
                    solid_color_srgb: [1.0, 1.0, 1.0, 1.0],
                    ..SurfaceStyle::default()
                },
            )
            .expect("upload opaque slice");

        let img_shell_then_slice = render_surfaces(&[&gpu_shell_fwd, &gpu_opaque_slice]);
        let img_slice_then_shell = render_surfaces(&[&gpu_opaque_slice, &gpu_shell_fwd]);
        let cmp_mixed = img_shell_then_slice
            .compare(&img_slice_then_shell, 0, 0.0)
            .expect("compare [shell, slice] vs [slice, shell]");
        assert_eq!(
            cmp_mixed.differing_pixels, 0,
            "opaque + transparent draw order must be invariant (sample_count={sample_count})"
        );

        let img_slice_only = render_surfaces(&[&gpu_opaque_slice]);
        let cmp_vs_slice_only = img_shell_then_slice
            .compare(&img_slice_only, 2, 0.0)
            .expect("compare composite vs slice only");
        assert!(
            cmp_vs_slice_only.differing_pixels > 5000,
            "transparent shell should visibly tint and surround the inner slice"
        );
        let cmp_vs_shell_only = img_shell_then_slice
            .compare(&img_fwd, 2, 0.0)
            .expect("compare composite vs shell only");
        assert!(
            cmp_vs_shell_only.differing_pixels > 2000,
            "inner opaque slice should be clearly visible through the transparent shell"
        );

        if sample_count == 4 {
            let out_dir = root.join("target/render-tests");
            img_shell_then_slice
                .save_png(out_dir.join("wboit_shell_and_slice.png"))
                .expect("save wboit_shell_and_slice.png");
        }
    }
}

#[test]
fn volume_ray_marching_modulates_transfer_function_and_terminates_at_opaque_depth() {
    use std::sync::Arc;
    use tetraglass_filters::{
        LinearizeParams, ResampleParams, SliceParams, linearize, resample, slice,
    };

    let root = workspace_root();
    let vtu_path = root.join("testdata/vtk/hex_linear.vtu");
    let mesh = read_vtu_from_path(&vtu_path).expect("read hex_linear.vtu");
    let shell = extract_external_surface(&mesh).expect("extract shell");
    let bounds = shell.bounds().expect("shell bounds");
    let dataset_center = bounds.center();
    let tet_view = linearize(&mesh, &LinearizeParams::default()).expect("linearize hex_linear");
    let grid = Arc::new(
        resample(
            &tet_view,
            &ResampleParams {
                dimensions: [20, 20, 20],
                bounds: None,
            },
        )
        .expect("resample hex_linear"),
    );
    let slice_x = slice(
        &tet_view,
        &SliceParams {
            origin: Some(dataset_center),
            normal: [1.0, 0.0, 0.0],
        },
    )
    .expect("slice x");

    let camera = Camera::fit_bounds(bounds, 35.0, 24.0);
    let ctx = HeadlessContext::new().expect("headless wgpu context");
    let bg = [0.08, 0.09, 0.12, 1.0];

    for sample_count in [1u32, 4u32] {
        let renderer = Renderer::new(
            ctx.device(),
            ctx.queue(),
            DEFAULT_COLOR_FORMAT,
            sample_count,
        )
        .expect("create renderer");
        let target =
            OffscreenTarget::new(ctx.device(), 256, 256, DEFAULT_COLOR_FORMAT, sample_count)
                .expect("create target");

        let render_surfaces = |surfaces: &[&tetraglass_render::GpuSurface]| -> RenderedImage {
            let mut encoder = ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            renderer
                .render(&mut encoder, ctx.queue(), &target, &camera, bg, surfaces)
                .expect("render surfaces");
            ctx.queue().submit(std::iter::once(encoder.finish()));
            ctx.readback_target(&target).expect("readback")
        };

        let img_empty = render_surfaces(&[]);

        // 1. Zero-opacity transfer function produces background-identical image.
        let gpu_vol_zero = renderer
            .upload_surface_with_center(
                ctx.device(),
                &shell,
                dataset_center,
                &SurfaceStyle {
                    color_by: ColorBy::point("linear"),
                    colormap: ColormapKind::Viridis,
                    draw_surface: false,
                    volume_grid: Some(Arc::clone(&grid)),
                    transfer_function: Some(vec![[0.0, 0.0], [1.0, 0.0]]),
                    ..SurfaceStyle::default()
                },
            )
            .expect("upload zero-opacity volume");
        let img_vol_zero = render_surfaces(&[&gpu_vol_zero]);
        let cmp_zero = img_vol_zero
            .compare(&img_empty, 0, 0.0)
            .expect("compare zero-opacity volume with empty");
        assert_eq!(
            cmp_zero.differing_pixels, 0,
            "zero-opacity transfer function must produce background-identical output (sample_count={sample_count})"
        );

        // 2. Low-opacity vs high-opacity transfer function produces visibly denser output.
        let gpu_vol_low = renderer
            .upload_surface_with_center(
                ctx.device(),
                &shell,
                dataset_center,
                &SurfaceStyle {
                    color_by: ColorBy::point("linear"),
                    colormap: ColormapKind::Viridis,
                    draw_surface: false,
                    volume_grid: Some(Arc::clone(&grid)),
                    transfer_function: Some(vec![[0.0, 0.0], [1.0, 0.25]]),
                    ..SurfaceStyle::default()
                },
            )
            .expect("upload low-opacity volume");
        let gpu_vol_high = renderer
            .upload_surface_with_center(
                ctx.device(),
                &shell,
                dataset_center,
                &SurfaceStyle {
                    color_by: ColorBy::point("linear"),
                    colormap: ColormapKind::Viridis,
                    draw_surface: false,
                    volume_grid: Some(Arc::clone(&grid)),
                    transfer_function: Some(vec![[0.0, 0.2], [0.5, 0.6], [1.0, 1.0]]),
                    ..SurfaceStyle::default()
                },
            )
            .expect("upload high-opacity volume");
        let img_vol_low = render_surfaces(&[&gpu_vol_low]);
        let img_vol_high = render_surfaces(&[&gpu_vol_high]);
        let cmp_low_empty = img_vol_low
            .compare(&img_empty, 2, 0.0)
            .expect("compare low-opacity volume with empty");
        assert!(
            cmp_low_empty.differing_pixels > 4000,
            "volume ray marching must visibly render the resampled cube (sample_count={sample_count}, diff={})",
            cmp_low_empty.differing_pixels
        );
        let cmp_low_high = img_vol_low
            .compare(&img_vol_high, 2, 0.0)
            .expect("compare low vs high opacity transfer function");
        assert!(
            cmp_low_high.differing_pixels > 4000,
            "changing transfer function control points must change rendered volume density (sample_count={sample_count})"
        );

        // 3. Depth-buffer termination against an opaque slice inside the volume.
        let gpu_opaque_slice = renderer
            .upload_surface_with_center(
                ctx.device(),
                &slice_x,
                dataset_center,
                &SurfaceStyle {
                    color_by: ColorBy::Solid,
                    solid_color_srgb: [0.95, 0.2, 0.2, 1.0],
                    ..SurfaceStyle::default()
                },
            )
            .expect("upload opaque slice");
        let img_slice_only = render_surfaces(&[&gpu_opaque_slice]);
        let img_vol_and_slice = render_surfaces(&[&gpu_vol_low, &gpu_opaque_slice]);
        let cmp_vs_vol_only = img_vol_and_slice
            .compare(&img_vol_low, 5, 0.0)
            .expect("compare volume+slice vs volume only");
        assert!(
            cmp_vs_vol_only.differing_pixels > 1500,
            "opaque inner slice must remain visible inside the ray-marched volume via depth termination (sample_count={sample_count})"
        );
        let cmp_vs_slice_only = img_vol_and_slice
            .compare(&img_slice_only, 2, 0.0)
            .expect("compare volume+slice vs slice only");
        assert!(
            cmp_vs_slice_only.differing_pixels > 3000,
            "volume in front of and around the opaque slice must visibly composite over it (sample_count={sample_count})"
        );
    }
}

#[test]
fn ssao_darkens_concave_crevices_and_leaves_background_and_disabled_renders_intact() {
    use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
    use tetraglass_filters::{
        ClipMode, ClipParams, LinearizeParams, clip, extract_tet_view_surface, linearize,
    };

    let root = workspace_root();
    let grid = SyntheticGridSpec::new(8, 8, 8, SyntheticCellLayout::Hexahedra)
        .generate()
        .expect("generate 8x8x8 hex grid");
    let tet_view = linearize(&grid, &LinearizeParams::default()).expect("linearize");
    let crinkle = clip(
        &tet_view,
        &ClipParams::with_origin([0.5, 0.5, 0.5], [-1.0, -1.0, -1.0]).mode(ClipMode::Crinkle),
    )
    .expect("crinkle clip");
    let surface = extract_tet_view_surface(&crinkle).expect("extract crinkle surface");

    let bounds = surface.bounds().expect("surface bounds");
    let camera = Camera::fit_bounds(bounds, 35.0, 28.0);
    let ctx = HeadlessContext::new().expect("headless context");

    for sample_count in [1u32, 4u32] {
        let opts = HeadlessRenderOptions {
            width: 480,
            height: 360,
            sample_count,
            background_srgb: [1.0, 1.0, 1.0, 1.0],
            style: SurfaceStyle {
                color_by: ColorBy::point("linear"),
                colormap: ColormapKind::Viridis,
                draw_edges: true,
                edge_color_srgb: [0.12, 0.14, 0.18, 1.0],
                ..SurfaceStyle::default()
            },
            camera: Some(camera.clone()),
            ..HeadlessRenderOptions::default()
        };

        let img_base = ctx.render_surface(&surface, &opts).expect("render base");
        let img_disabled = ctx
            .render_surface_with_ssao(&surface, &opts, SsaoConfig::default())
            .expect("render ssao disabled");

        let cmp_disabled = img_base
            .compare(&img_disabled, 0, 0.0)
            .expect("compare base vs ssao disabled");
        assert_eq!(
            cmp_disabled.differing_pixels, 0,
            "SsaoConfig::default() must produce bit-for-bit identical output (sample_count={sample_count})"
        );

        let ssao_low = SsaoConfig::enabled().with_radius(0.18).with_strength(0.6);
        let ssao_high = SsaoConfig::enabled().with_radius(0.18).with_strength(1.3);

        let img_low = ctx
            .render_surface_with_ssao(&surface, &opts, ssao_low)
            .expect("render ssao low");
        let img_high = ctx
            .render_surface_with_ssao(&surface, &opts, ssao_high)
            .expect("render ssao high");

        if sample_count == 4 {
            let out_dir = root.join("target/render-tests");
            img_disabled
                .save_png(out_dir.join("ssao_disabled.png"))
                .expect("save ssao_disabled.png");
            img_high
                .save_png(out_dir.join("ssao_enabled.png"))
                .expect("save ssao_enabled.png");
        }

        // Background corners must remain pure white.
        for (x, y) in [(4, 4), (475, 4), (4, 355), (475, 355)] {
            assert_eq!(
                img_high.pixel(x, y),
                [255, 255, 255, 255],
                "background pixel ({x}, {y}) must remain untouched by SSAO"
            );
        }

        // Interior staircase crevices must darken measurably compared to disabled SSAO,
        // and higher strength must darken more than lower strength.
        let total_luma = |img: &RenderedImage| -> u64 {
            img.as_rgba8()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|px| u64::from(px[0]) + u64::from(px[1]) + u64::from(px[2]))
                .sum()
        };
        let luma_off = total_luma(&img_disabled);
        let luma_low = total_luma(&img_low);
        let luma_high = total_luma(&img_high);

        assert!(
            luma_low < luma_off,
            "enabling SSAO must reduce total scene luminance in concave staircase crevices (sample_count={sample_count}: luma_low={luma_low} vs luma_off={luma_off})"
        );
        assert!(
            luma_high < luma_low,
            "increasing SSAO strength must monotonically darken occluded crevices (sample_count={sample_count}: luma_high={luma_high} vs luma_low={luma_low})"
        );

        let cmp_on_off = img_disabled
            .compare(&img_high, 2, 0.0)
            .expect("compare disabled vs high ssao");
        assert!(
            cmp_on_off.differing_pixels > 1500,
            "expected > 1500 occluded pixels on crinkle staircase surface, got {} (sample_count={sample_count})",
            cmp_on_off.differing_pixels
        );
    }
}

#[test]
fn compute_path_tracer_accumulates_progressively_and_renders_soft_shadows_and_multi_bounce_gi() {
    use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
    use tetraglass_filters::{
        ClipMode, ClipParams, LinearizeParams, clip, extract_tet_view_surface, linearize,
    };
    use tetraglass_render::{
        Annotations, ColorbarAnnotation, DEFAULT_COLOR_FORMAT, OffscreenTarget, PathTraceConfig,
        PathTracer,
    };

    let root = workspace_root();
    let grid = SyntheticGridSpec::new(8, 8, 8, SyntheticCellLayout::Hexahedra)
        .generate()
        .expect("generate 8x8x8 hex grid");
    let tet_view = linearize(&grid, &LinearizeParams::default()).expect("linearize");
    let crinkle = clip(
        &tet_view,
        &ClipParams::with_origin([0.5, 0.5, 0.5], [-1.0, -1.0, -1.0]).mode(ClipMode::Crinkle),
    )
    .expect("crinkle clip");
    let surface = extract_tet_view_surface(&crinkle).expect("extract crinkle surface");

    let bounds = surface.bounds().expect("surface bounds");
    let camera = Camera::fit_bounds(bounds, 35.0, 28.0);
    let ctx = HeadlessContext::new().expect("headless context");

    let style = SurfaceStyle {
        color_by: ColorBy::point("linear"),
        colormap: ColormapKind::Viridis,
        draw_edges: true,
        edge_color_srgb: [0.12, 0.14, 0.18, 1.0],
        ..SurfaceStyle::default()
    };

    let opts = HeadlessRenderOptions {
        width: 480,
        height: 360,
        sample_count: 4,
        background_srgb: [1.0, 1.0, 1.0, 1.0],
        style: style.clone(),
        camera: Some(camera.clone()),
        annotations: Annotations {
            axes: true,
            colorbars: vec![
                ColorbarAnnotation::new(ColormapKind::Viridis, [0.0, 3.0]).with_title("linear"),
            ],
            ..Annotations::default()
        },
    };

    // 1. Compare 1-bounce direct lighting vs 3-bounce global illumination.
    let img_1bounce = ctx
        .render_surface_path_traced(
            &surface,
            &opts,
            PathTraceConfig::new(16).with_max_bounces(1),
        )
        .expect("render 1-bounce path trace");
    let img_3bounce = ctx
        .render_surface_path_traced(
            &surface,
            &opts,
            PathTraceConfig::new(32).with_max_bounces(3),
        )
        .expect("render 3-bounce path trace");
    let img_1spp = ctx
        .render_surface_path_traced(&surface, &opts, PathTraceConfig::new(1).with_max_bounces(3))
        .expect("render 1-spp path trace");

    let out_dir = root.join("target/render-tests");
    img_1spp
        .save_png(out_dir.join("path_traced_crinkle_1spp.png"))
        .expect("save path_traced_crinkle_1spp.png");
    img_3bounce
        .save_png(out_dir.join("path_traced_crinkle_32spp.png"))
        .expect("save path_traced_crinkle_32spp.png");

    // Background top-left corner must remain pure white.
    assert_eq!(img_3bounce.pixel(4, 4), [255, 255, 255, 255]);

    let total_luma = |img: &RenderedImage| -> u64 {
        img.as_rgba8()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|px| u64::from(px[0]) + u64::from(px[1]) + u64::from(px[2]))
            .sum()
    };
    let luma_1b = total_luma(&img_1bounce);
    let luma_3b = total_luma(&img_3bounce);
    assert!(
        luma_3b > luma_1b,
        "multi-bounce indirect global illumination must brighten shadowed crevices compared to 1-bounce direct lighting (luma_3b={luma_3b} vs luma_1b={luma_1b})"
    );

    // 2. Test progressive accumulation via `PathTracer::step` and automatic reset on camera orbit.
    let mut pt = PathTracer::new(ctx.device(), DEFAULT_COLOR_FORMAT, 1).expect("create PathTracer");
    pt.set_config(
        PathTraceConfig::new(8)
            .with_samples_per_frame(2)
            .with_max_bounces(3),
    );
    let target = OffscreenTarget::new(ctx.device(), 240, 180, DEFAULT_COLOR_FORMAT, 1)
        .expect("create 1x target");
    let style_no_edges = SurfaceStyle {
        draw_edges: false,
        ..style
    };
    let pt_scene = pt
        .upload_surface(ctx.device(), &surface, &style_no_edges)
        .expect("upload path trace scene");

    for expected_samples in [2u32, 4, 6, 8] {
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let accum = pt
            .step(
                ctx.device(),
                &mut encoder,
                ctx.queue(),
                &target,
                &pt_scene,
                &camera,
                [1.0, 1.0, 1.0, 1.0],
            )
            .expect("step path tracer");
        ctx.queue().submit(std::iter::once(encoder.finish()));
        assert_eq!(accum, expected_samples);
        assert_eq!(pt.accumulated_samples(), expected_samples);
    }
    assert!(pt.is_converged());

    // Calling `step` again after convergence keeps `accumulated_samples == 8`.
    {
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let accum = pt
            .step(
                ctx.device(),
                &mut encoder,
                ctx.queue(),
                &target,
                &pt_scene,
                &camera,
                [1.0, 1.0, 1.0, 1.0],
            )
            .expect("step converged");
        ctx.queue().submit(std::iter::once(encoder.finish()));
        assert_eq!(accum, 8);
    }

    // Orbiting the camera must automatically reset accumulation to `samples_per_frame` (`2`).
    let mut moved_camera = camera.clone();
    moved_camera.orbit(15.0, 5.0);
    {
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let accum = pt
            .step(
                ctx.device(),
                &mut encoder,
                ctx.queue(),
                &target,
                &pt_scene,
                &moved_camera,
                [1.0, 1.0, 1.0, 1.0],
            )
            .expect("step after camera orbit");
        ctx.queue().submit(std::iter::once(encoder.finish()));
        assert_eq!(
            accum, 2,
            "camera movement must reset progressive sample accumulation"
        );
        assert!(!pt.is_converged());
    }
}

#[test]
fn points_representation_renders_solid_point_and_cell_colors_scales_with_point_size_and_supports_wboit()
 {
    use tetraglass_render::{DEFAULT_COLOR_FORMAT, OffscreenTarget, Renderer};

    let root = workspace_root();
    let vtu_path = root.join("testdata/vtk/mixed_linear.vtu");
    let grid = read_vtu_from_path(&vtu_path).expect("read mixed_linear.vtu");
    let surface = extract_external_surface(&grid).expect("extract surface");
    let bounds = surface.bounds().expect("surface bounds");
    let camera = Camera::fit_bounds(bounds, 30.0, 20.0);

    let ctx = HeadlessContext::new().expect("headless wgpu context");
    let count_non_white = |img: &RenderedImage| -> usize {
        img.as_rgba8()
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[0] < 250 || px[1] < 250 || px[2] < 250)
            .count()
    };

    // 1. Small vs large `point_size` with point-scalar coloring (`linear`).
    let opts_small = HeadlessRenderOptions {
        width: 400,
        height: 300,
        sample_count: 4,
        style: SurfaceStyle {
            color_by: ColorBy::point("linear"),
            colormap: ColormapKind::Viridis,
            draw_surface: false,
            draw_edges: false,
            draw_points: true,
            point_size: 4.0,
            ..SurfaceStyle::default()
        },
        camera: Some(camera.clone()),
        ..HeadlessRenderOptions::default()
    };
    let opts_large = HeadlessRenderOptions {
        style: SurfaceStyle {
            point_size: 14.0,
            ..opts_small.style.clone()
        },
        ..opts_small.clone()
    };

    let img_small = ctx
        .render_surface(&surface, &opts_small)
        .expect("render small points");
    let img_large = ctx
        .render_surface(&surface, &opts_large)
        .expect("render large points");

    let out_dir = root.join("target/render-tests");
    std::fs::create_dir_all(&out_dir).unwrap();
    img_large
        .save_png(out_dir.join("points_mixed_linear.png"))
        .expect("save points_mixed_linear.png");

    let px_small = count_non_white(&img_small);
    let px_large = count_non_white(&img_large);
    assert!(
        px_small >= 50,
        "expected >= 50 non-white pixels for 4px points on mixed_linear, got {px_small}"
    );
    assert!(
        px_large > px_small * 4,
        "increasing point_size from 4px to 14px must substantially increase covered pixel count (px_large={px_large} vs px_small={px_small})"
    );

    // 2. Cell-scalar (`cell_id`) and solid coloring on `"points"` produce distinct valid images.
    let opts_cell = HeadlessRenderOptions {
        style: SurfaceStyle {
            color_by: ColorBy::cell("cell_id"),
            colormap: ColormapKind::Cividis,
            ..opts_large.style.clone()
        },
        ..opts_large.clone()
    };
    let opts_solid = HeadlessRenderOptions {
        style: SurfaceStyle {
            color_by: ColorBy::Solid,
            solid_color_srgb: [0.85, 0.22, 0.18, 1.0],
            ..opts_large.style.clone()
        },
        ..opts_large.clone()
    };
    let img_cell = ctx
        .render_surface(&surface, &opts_cell)
        .expect("render cell-colored points");
    let img_solid = ctx
        .render_surface(&surface, &opts_solid)
        .expect("render solid-colored points");

    assert!(count_non_white(&img_cell) >= px_large / 2);
    assert!(count_non_white(&img_solid) >= px_large / 2);
    assert!(
        img_large
            .compare(&img_cell, 2, 0.0)
            .expect("compare point vs cell")
            .differing_pixels
            > 200
    );
    assert!(
        img_large
            .compare(&img_solid, 2, 0.0)
            .expect("compare point vs solid")
            .differing_pixels
            > 200
    );

    // 3. `Renderer::update_surface_style` dynamically updates `draw_points`, `point_size`, and WBOIT transparency.
    let renderer =
        Renderer::new(ctx.device(), ctx.queue(), DEFAULT_COLOR_FORMAT, 4).expect("create renderer");
    let mut gpu_surface = renderer
        .upload_surface(ctx.device(), &surface, &opts_small.style)
        .expect("upload points surface");
    assert_eq!(gpu_surface.point_count(), 11);
    assert!(gpu_surface.draw_points());
    assert!((gpu_surface.point_size() - 4.0).abs() < 1e-6);
    assert!(gpu_surface.is_opaque());
    assert!(!gpu_surface.is_transparent());

    let transparent_style = SurfaceStyle {
        color_by: ColorBy::Solid,
        solid_color_srgb: [0.15, 0.45, 0.85, 0.45],
        draw_surface: false,
        draw_edges: false,
        draw_points: true,
        point_size: 16.0,
        ..SurfaceStyle::default()
    };
    renderer
        .update_surface_style(ctx.queue(), &mut gpu_surface, &surface, &transparent_style)
        .expect("update to transparent points style");
    assert!((gpu_surface.point_size() - 16.0).abs() < 1e-6);
    assert!(!gpu_surface.is_opaque());
    assert!(gpu_surface.is_transparent());

    let target = OffscreenTarget::new(ctx.device(), 400, 300, DEFAULT_COLOR_FORMAT, 4)
        .expect("create offscreen target");
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    renderer
        .render(
            &mut encoder,
            ctx.queue(),
            &target,
            &camera,
            [1.0, 1.0, 1.0, 1.0],
            &[&gpu_surface],
        )
        .expect("render WBOIT points");
    ctx.queue().submit(std::iter::once(encoder.finish()));
    let img_wboit = ctx
        .readback_target(&target)
        .expect("read WBOIT points image");
    assert!(
        count_non_white(&img_wboit) > 400,
        "semi-transparent points must render via WBOIT accumulation pass"
    );
}

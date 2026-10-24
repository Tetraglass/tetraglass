//! Headless rendering integration tests for `tetraglass-render` (M0.7).

use std::path::{Path, PathBuf};

use tetraglass_filters::extract_external_surface;
use tetraglass_io::read_vtu_from_path;
use tetraglass_render::{
    Camera, ColorBy, ColormapKind, DEFAULT_COLOR_FORMAT, HeadlessContext, HeadlessRenderOptions,
    OffscreenTarget, RenderedImage, Renderer, SurfaceStyle,
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

//! Headless rendering integration tests for `tetraglass-render` (M0.7).

use std::path::PathBuf;

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
        },
        camera: Some(camera.clone()),
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

    // Optional baseline comparison when `tests/baselines/mixed_linear_linear.png` is present (M0.9).
    let baseline_path = root.join("tests/baselines/mixed_linear_linear.png");
    if baseline_path.exists() {
        let baseline = RenderedImage::load_png(&baseline_path).expect("load baseline PNG");
        let cmp = image
            .compare_with_baseline(&baseline)
            .expect("compare with baseline");
        if !cmp.passed {
            let diff_path = root.join("target/render-diffs/mixed_linear_linear.png");
            cmp.diff_image
                .save_png(&diff_path)
                .expect("write diff image");
            panic!(
                "mixed_linear_linear.png differed from baseline in {} / {} pixels ({:.3}%, max channel diff {}), diff written to {}",
                cmp.differing_pixels,
                cmp.total_pixels,
                cmp.diff_fraction * 100.0,
                cmp.max_channel_diff,
                diff_path.display()
            );
        }
    }

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

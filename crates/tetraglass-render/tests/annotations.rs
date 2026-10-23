//! Integration tests for renderer-drawn annotations: colorbar with tick labels, title text,
//! and 3D orientation axes triad (`M1.5`).

use std::path::PathBuf;

use tetraglass_filters::extract_external_surface;
use tetraglass_io::read_vtu_from_path;
use tetraglass_render::{
    Annotations, Camera, ColorBy, ColorbarAnnotation, ColorbarPlacement, ColormapKind,
    HeadlessContext, HeadlessRenderOptions, SurfaceStyle,
};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn render_annotations_draws_title_colorbar_and_axes_triad() {
    let root = workspace_root();
    let vtu_path = root.join("testdata/vtk/mixed_linear.vtu");
    let grid = read_vtu_from_path(&vtu_path).expect("load mixed_linear.vtu");
    let surface = extract_external_surface(&grid).expect("extract external surface");
    let bounds = surface.bounds().expect("surface bounds");
    let camera = Camera::fit_bounds(bounds, 30.0, 20.0);

    let ctx = HeadlessContext::new().expect("headless context");
    let style = SurfaceStyle {
        color_by: ColorBy::point("linear"),
        colormap: ColormapKind::Viridis,
        range: None,
        use_log_scale: false,
        solid_color_srgb: [0.8, 0.8, 0.8, 1.0],
    };

    let unannotated_opts = HeadlessRenderOptions {
        width: 800,
        height: 600,
        sample_count: 4,
        background_srgb: [1.0, 1.0, 1.0, 1.0],
        style: style.clone(),
        camera: Some(camera.clone()),
        annotations: Annotations::default(),
    };
    let unannotated = ctx
        .render_surface(&surface, &unannotated_opts)
        .expect("render unannotated");

    let annotations = Annotations::default()
        .with_title("Mixed Linear Mesh")
        .with_axes(true)
        .with_colorbar(
            ColorbarAnnotation::new(ColormapKind::Viridis, [0.0, 3.0])
                .with_title("linear")
                .with_position(ColorbarPlacement::Right),
        );

    let annotated_opts = HeadlessRenderOptions {
        annotations,
        ..unannotated_opts
    };
    let annotated = ctx
        .render_surface(&surface, &annotated_opts)
        .expect("render annotated");

    let out_path = root.join("target/render-tests/mixed_linear_annotated.png");
    annotated
        .save_png(&out_path)
        .expect("save mixed_linear_annotated.png");

    // 1. Title region near top-center (`x in 250..550, y in 10..45`) must be white in
    //    `unannotated` and contain dark glyph pixels in `annotated`.
    let mut title_unannotated_nonwhite = 0usize;
    let mut title_annotated_dark = 0usize;
    for y in 10..45 {
        for x in 250..550 {
            let u_px = unannotated.pixel(x, y);
            if u_px[0] < 250 || u_px[1] < 250 || u_px[2] < 250 {
                title_unannotated_nonwhite += 1;
            }
            let a_px = annotated.pixel(x, y);
            if a_px[0] < 120 && a_px[1] < 120 && a_px[2] < 120 {
                title_annotated_dark += 1;
            }
        }
    }
    assert_eq!(
        title_unannotated_nonwhite, 0,
        "expected empty top-center region in unannotated image"
    );
    assert!(
        title_annotated_dark > 120,
        "expected title text pixels in top-center region, got {title_annotated_dark}"
    );

    // 2. Right colorbar region (`x in 680..795, y in 130..480`) must contain Viridis yellow near
    //    the top of the bar, Viridis purple/blue near the bottom of the bar, and dark tick/label pixels.
    let mut cb_top_yellow = 0usize;
    let mut cb_bottom_purple = 0usize;
    let mut cb_dark_labels = 0usize;
    for y in 130..480 {
        for x in 680..795 {
            let [r, g, b, _] = annotated.pixel(x, y);
            if y < 260 && r > 180 && g > 200 && b < 100 {
                cb_top_yellow += 1;
            }
            if y > 340 && b > 80 && r < 100 && g < 100 {
                cb_bottom_purple += 1;
            }
            if r < 90 && g < 90 && b < 90 {
                cb_dark_labels += 1;
            }
        }
    }
    assert!(
        cb_top_yellow > 100,
        "expected Viridis yellow pixels near top of colorbar, got {cb_top_yellow}"
    );
    assert!(
        cb_bottom_purple > 100,
        "expected Viridis purple/blue pixels near bottom of colorbar, got {cb_bottom_purple}"
    );
    assert!(
        cb_dark_labels > 100,
        "expected dark border, tick mark, and label pixels on colorbar, got {cb_dark_labels}"
    );

    // 3. Bottom-left axes triad region (`x in 8..110, y in 490..592`) must contain red (X),
    //    green (Y), and blue (Z) axis pixels.
    let mut triad_red = 0usize;
    let mut triad_green = 0usize;
    let mut triad_blue = 0usize;
    for y in 490..592 {
        for x in 8..110 {
            let [r, g, b, _] = annotated.pixel(x, y);
            if r > 160 && g < 90 && b < 90 {
                triad_red += 1;
            }
            if g > 120 && r < 90 && b < 90 {
                triad_green += 1;
            }
            if b > 160 && r < 90 && g < 130 {
                triad_blue += 1;
            }
        }
    }
    assert!(
        triad_red > 25,
        "expected red +X axis triad pixels in bottom-left corner, got {triad_red}"
    );
    assert!(
        triad_green > 25,
        "expected green +Y axis triad pixels in bottom-left corner, got {triad_green}"
    );
    assert!(
        triad_blue > 25,
        "expected blue +Z axis triad pixels in bottom-left corner, got {triad_blue}"
    );
}

#[test]
fn render_annotations_supports_all_colorbar_positions_and_dark_background() {
    let root = workspace_root();
    let vtu_path = root.join("testdata/vtk/mixed_linear.vtu");
    let grid = read_vtu_from_path(&vtu_path).expect("load mixed_linear.vtu");
    let surface = extract_external_surface(&grid).expect("extract surface");

    let ctx = HeadlessContext::new().expect("headless context");
    for placement in [
        ColorbarPlacement::Left,
        ColorbarPlacement::Bottom,
        ColorbarPlacement::Top,
    ] {
        let annotations = Annotations::default()
            .with_title("Dark Theme Placement Check")
            .with_axes(true)
            .with_colorbar(
                ColorbarAnnotation::new(ColormapKind::CoolWarm, [1.0, 1000.0])
                    .with_title("pressure")
                    .with_position(placement)
                    .with_log_scale(true),
            );
        let opts = HeadlessRenderOptions {
            width: 640,
            height: 480,
            sample_count: 1,
            background_srgb: [0.08, 0.09, 0.12, 1.0],
            style: SurfaceStyle {
                color_by: ColorBy::point("linear"),
                colormap: ColormapKind::CoolWarm,
                ..SurfaceStyle::default()
            },
            camera: None,
            annotations,
        };
        let img = ctx
            .render_surface(&surface, &opts)
            .expect("render dark background placement");
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);
    }
}

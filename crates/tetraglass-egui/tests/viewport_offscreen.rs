//! Integration tests for `tetraglass-egui` (`ViewportState`, `ViewportWidget`, and `available_field_options`).

use std::path::PathBuf;

use egui_wgpu::wgpu;
use tetraglass_egui::{
    ViewportColorBy, ViewportColormap, ViewportScalarComponent, ViewportState,
    ViewportSurfaceStyle, ViewportWidget, available_field_options,
};
use tetraglass_filters::extract_external_surface;
use tetraglass_io::read_vtu_from_path;
use tetraglass_render::{DEFAULT_COLOR_FORMAT, HeadlessContext};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn available_field_options_lists_solid_point_and_cell_arrays() {
    let root = workspace_root();
    let grid =
        read_vtu_from_path(root.join("testdata/vtk/mixed_linear.vtu")).expect("load mixed_linear");
    let surface = extract_external_surface(&grid).expect("extract surface");

    let options = available_field_options(&surface);
    let labels: Vec<&str> = options.iter().map(|o| o.label.as_str()).collect();
    assert!(labels.contains(&"Solid color"));
    assert!(labels.contains(&"Point: linear"));
    assert!(labels.contains(&"Point: radius"));
    assert!(labels.contains(&"Point: velocity (magnitude)"));
    assert!(labels.contains(&"Point: velocity (X)"));
    assert!(labels.contains(&"Point: velocity (Y)"));
    assert!(labels.contains(&"Point: velocity (Z)"));
    assert!(labels.contains(&"Cell: cell_id"));
}

#[test]
fn viewport_widget_resizes_and_runs_prepare_callback_in_egui_wgpu() {
    let root = workspace_root();
    let grid =
        read_vtu_from_path(root.join("testdata/vtk/mixed_linear.vtu")).expect("load mixed_linear");
    let surface = extract_external_surface(&grid).expect("extract surface");

    let headless = HeadlessContext::new().expect("headless wgpu context");
    let mut egui_renderer = egui_wgpu::Renderer::new(
        headless.device(),
        DEFAULT_COLOR_FORMAT,
        egui_wgpu::RendererOptions::default(),
    );

    let mut viewport =
        ViewportState::new(headless.device(), headless.queue(), &mut egui_renderer, 4)
            .expect("create ViewportState");
    assert_eq!(viewport.target_size_px(), (64, 64));

    let style = ViewportSurfaceStyle {
        color_by: ViewportColorBy::point("linear"),
        colormap: ViewportColormap::Viridis,
        range: None,
        use_log_scale: false,
        solid_color_srgb: [0.8, 0.8, 0.8, 1.0],
    };
    viewport
        .set_surface(headless.device(), &surface, style, true)
        .expect("upload surface");
    assert_eq!(viewport.effective_range(), Some([0.0, 7.5]));

    // Run one egui frame with the ViewportWidget at 400x300 logical points (1.0 ppp).
    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(400.0, 300.0),
        )),
        ..Default::default()
    };

    let mut full_output = ctx.run_ui(raw_input, |ui| {
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                let resp = ViewportWidget::new(&mut viewport)
                    .desired_size(egui::Vec2::new(400.0, 300.0))
                    .show(ui, headless.device(), &mut egui_renderer)
                    .expect("show ViewportWidget");
                assert!(resp.resized);
                assert!(!resp.camera_changed);
            });
    });

    assert_eq!(viewport.target_size_px(), (400, 300));

    // Verify `egui_wgpu::Renderer::update_buffers` executes `ViewportPrepareCallback::prepare` cleanly.
    let primitives = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    for (id, deltas) in &full_output.textures_delta.set {
        for delta in deltas {
            egui_renderer.update_texture(headless.device(), headless.queue(), *id, delta);
        }
    }
    for id in &full_output.textures_delta.free {
        egui_renderer.free_texture(id);
    }
    full_output.textures_delta.clear();
    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [400, 300],
        pixels_per_point: full_output.pixels_per_point,
    };
    let mut encoder = headless
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_egui_viewport_encoder"),
        });
    let cmd_bufs = egui_renderer.update_buffers(
        headless.device(),
        headless.queue(),
        &mut encoder,
        &primitives,
        &screen_descriptor,
    );
    headless.queue().submit(
        cmd_bufs
            .into_iter()
            .chain(std::iter::once(encoder.finish())),
    );

    // Switch style to vector magnitude and verify range updates.
    let vec_style = ViewportSurfaceStyle {
        color_by: ViewportColorBy::Point {
            name: "velocity".to_string(),
            component: ViewportScalarComponent::Magnitude,
        },
        colormap: ViewportColormap::Cividis,
        range: None,
        use_log_scale: false,
        solid_color_srgb: [0.8, 0.8, 0.8, 1.0],
    };
    viewport
        .update_style(headless.queue(), &surface, vec_style)
        .expect("update style to velocity magnitude");
    let [lo, hi] = viewport.effective_range().expect("range");
    assert!((lo - 0.0).abs() < 1e-6);
    assert!((hi - 2.0).abs() < 1e-6);

    viewport.destroy(&mut egui_renderer);
}

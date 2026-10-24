//! Integration tests for `tetraglass-egui` (`ViewportState`, `ViewportWidget`, and `available_field_options`).

use std::path::PathBuf;

use egui_wgpu::wgpu;
use tetraglass_egui::{
    ViewportColorBy, ViewportColormap, ViewportScalarComponent, ViewportState,
    ViewportSurfaceStyle, ViewportWidget, available_field_options,
};
use tetraglass_filters::extract_external_surface;
use tetraglass_io::read_vtu_from_path;
use tetraglass_render::{HeadlessContext, OffscreenTarget};

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
        wgpu::TextureFormat::Rgba8Unorm,
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
        ..ViewportSurfaceStyle::default()
    };
    viewport
        .set_surface(headless.device(), &surface, style, true)
        .expect("upload surface");
    assert_eq!(viewport.effective_range(), Some([0.0, 7.5]));

    // Run an egui frame (800x500 logical points at 1.0 ppp) with a left side panel and the 3D ViewportWidget.
    let ctx = egui::Context::default();
    let mut full_output = egui::FullOutput::default();
    for pass in 0..2 {
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(800.0, 500.0),
            )),
            ..Default::default()
        };
        full_output = ctx.run_ui(raw_input, |ui| {
            egui::Panel::left("tetraglass_controls")
                .resizable(false)
                .exact_size(220.0)
                .show(ui, |ui| {
                    ui.heading("Tetraglass Viewport");
                    ui.separator();
                    ui.label("File: mixed_linear.vtu");
                    ui.label(format!(
                        "Volume: {} points, {} cells",
                        grid.num_points(),
                        grid.num_cells()
                    ));
                    ui.label(format!(
                        "Surface: {} faces ({} tris)",
                        surface.num_faces(),
                        surface.num_triangles()
                    ));
                    ui.separator();
                    ui.label("Color by: Point: linear");
                    ui.label("Colormap: Viridis");
                    if let Some([lo, hi]) = viewport.effective_range() {
                        ui.label(format!("Range: [{lo:.4}, {hi:.4}]"));
                    }
                });
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ui, |ui| {
                    let resp = ViewportWidget::new(&mut viewport)
                        .show(ui, headless.device(), &mut egui_renderer)
                        .expect("show ViewportWidget");
                    if pass == 0 {
                        assert!(resp.resized);
                    }
                    assert!(!resp.camera_changed);
                });
        });
        for (id, deltas) in &full_output.textures_delta.set {
            for delta in deltas {
                egui_renderer.update_texture(headless.device(), headless.queue(), *id, delta);
            }
        }
        for id in &full_output.textures_delta.free {
            egui_renderer.free_texture(id);
        }
        full_output.textures_delta.clear();
    }

    let (vp_w, vp_h) = viewport.target_size_px();
    assert!(
        (550..=600).contains(&vp_w),
        "unexpected viewport width {vp_w}"
    );
    assert_eq!(vp_h, 500);

    // Execute `egui_wgpu::Renderer::update_buffers` (runs `ViewportPrepareCallback::prepare`)
    // and `egui_wgpu::Renderer::render` into an offscreen 800x500 target, then read back PNG.
    let primitives = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [800, 500],
        pixels_per_point: full_output.pixels_per_point,
    };
    let ui_target = OffscreenTarget::new(
        headless.device(),
        800,
        500,
        wgpu::TextureFormat::Rgba8Unorm,
        1,
    )
    .expect("create ui target");
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
    {
        let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test_egui_ui_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ui_target.egui_texture_view(),
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.11,
                        g: 0.11,
                        b: 0.12,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        egui_renderer.render(
            &mut rpass.forget_lifetime(),
            &primitives,
            &screen_descriptor,
        );
    }
    headless.queue().submit(
        cmd_bufs
            .into_iter()
            .chain(std::iter::once(encoder.finish())),
    );

    let ui_image = headless
        .readback_target(&ui_target)
        .expect("readback composited egui viewport image");
    let ui_out_path = root.join("target/render-tests/egui_viewport_mixed_linear.png");
    ui_image
        .save_png(&ui_out_path)
        .expect("save egui_viewport_mixed_linear.png");

    // Verify the viewport region (x in 240..790) contains non-white Viridis mesh pixels.
    let mut viewport_mesh_pixels = 0usize;
    for y in 20..480 {
        for x in 240..780 {
            let [r, g, b, _] = ui_image.pixel(x, y);
            if r < 245 || g < 245 || b < 245 {
                viewport_mesh_pixels += 1;
            }
        }
    }
    assert!(
        viewport_mesh_pixels > 5000,
        "expected rendered 3D mesh pixels inside egui viewport region, got {viewport_mesh_pixels}"
    );

    // Probe the center of the viewport via GPU ID buffer picking.
    let pick = viewport
        .pick_at(headless.device(), headless.queue(), vp_w / 2, vp_h / 2)
        .expect("pick_at succeeds")
        .expect("center pixel hits mixed_linear surface");
    assert_eq!(pick.representation_index, 0);
    let [wx, wy, wz] = pick.world_position;
    let expected_linear = wx + 2.0 * wy + 3.0 * wz;
    let actual_linear = pick.active_scalar.expect("active_scalar present");
    assert!(
        (actual_linear - expected_linear).abs() < 1e-4,
        "picked active_scalar {actual_linear} vs expected {expected_linear}"
    );
    assert_eq!(viewport.last_pick(), Some(pick));
    viewport.clear_pick();
    assert!(viewport.last_pick().is_none());

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
        ..ViewportSurfaceStyle::default()
    };
    viewport
        .update_style(headless.queue(), &surface, vec_style)
        .expect("update style to velocity magnitude");
    let [lo, hi] = viewport.effective_range().expect("range");
    assert!((lo - 0.0).abs() < 1e-6);
    assert!((hi - 2.0).abs() < 1e-6);

    viewport.destroy(&mut egui_renderer);
}

//! Integration tests for `tetraglass-viewer` (`M1.8`): scene opening, hot reload on file change,
//! time slider stepping, field and colormap pickers, and offscreen `egui_wgpu` rendering.

use std::fs;
use std::path::PathBuf;

use egui_wgpu::wgpu;
use tetraglass_egui::{ViewportColorBy, ViewportColormap, ViewportScalarComponent, ViewportState};
use tetraglass_render::{HeadlessContext, OffscreenTarget};
use tetraglass_scene::ColormapName;
use tetraglass_viewer::{HotReloadOutcome, ViewerDocument, ViewerSession};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn open_scene_and_raw_mesh_files_and_switch_fields_and_colormaps() {
    let root = workspace_root();
    let vtu_path = root.join("testdata/vtk/mixed_linear.vtu");

    // 1. Open a raw `.vtu` mesh directly (synthesizes a default Scene).
    let mut raw_doc = ViewerDocument::load_path(&vtu_path).expect("load raw mixed_linear.vtu");
    assert_eq!(raw_doc.file_label(), "mixed_linear.vtu");
    assert_eq!(raw_doc.num_time_steps(), 1);
    assert_eq!(raw_doc.evaluator().filter_executions(), 1);
    assert!(raw_doc.colorbar_enabled());
    assert!(raw_doc.axes_enabled());

    let surface = raw_doc.active_surface().expect("active surface");
    assert_eq!(surface.num_faces(), 12);
    assert_eq!(surface.num_triangles(), 18);

    // Verify field options include Solid color, Point arrays, and Cell arrays.
    let labels: Vec<&str> = raw_doc
        .field_options()
        .iter()
        .map(|f| f.label.as_str())
        .collect();
    assert!(labels.contains(&"Solid color"));
    assert!(labels.contains(&"Point: linear"));
    assert!(labels.contains(&"Point: velocity (magnitude)"));
    assert!(labels.contains(&"Point: velocity (Y)"));
    assert!(labels.contains(&"Cell: cell_id"));

    // Switching field picker to "Point: velocity (magnitude)" must NOT re-run the surface filter.
    let vel_mag_idx = raw_doc
        .field_options()
        .iter()
        .position(|f| f.label == "Point: velocity (magnitude)")
        .unwrap();
    assert!(raw_doc.set_field_index(vel_mag_idx).unwrap());
    assert_eq!(raw_doc.evaluator().filter_executions(), 1);
    assert_eq!(
        raw_doc.active_surface_style().color_by,
        ViewportColorBy::Point {
            name: "velocity".to_string(),
            component: ViewportScalarComponent::Magnitude,
        }
    );
    assert_eq!(raw_doc.active_surface_style().range, Some([0.0, 2.0]));

    // Switching colormap to CoolWarm and Cividis must NOT re-run the surface filter.
    assert!(raw_doc.set_colormap(ColormapName::CoolWarm).unwrap());
    assert_eq!(raw_doc.colormap(), ColormapName::CoolWarm);
    assert_eq!(
        raw_doc.active_surface_style().colormap,
        ViewportColormap::CoolWarm
    );
    assert_eq!(raw_doc.evaluator().filter_executions(), 1);

    // Switching to Solid color disables scalar range and colorbar.
    assert!(raw_doc.set_field_index(0).unwrap());
    assert_eq!(
        raw_doc.active_surface_style().color_by,
        ViewportColorBy::Solid
    );
    assert!(raw_doc.annotations().colorbars.is_empty());
    assert_eq!(raw_doc.evaluator().filter_executions(), 1);
}

#[test]
fn time_slider_steps_lazily_through_pvd_series() {
    let root = workspace_root();
    let test_dir = root.join("target/render-tests/viewer-series");
    fs::create_dir_all(&test_dir).unwrap();

    let pvd_abs = root.join("testdata/vtk/series.pvd");
    let scene_path = test_dir.join("series_viewer_scene.json");
    let scene_json = serde_json::json!({
        "version": 1,
        "sources": {
            "series": { "file": pvd_abs.display().to_string() }
        },
        "view": {
            "show": [
                {
                    "input": "series",
                    "color_by": "transient",
                    "colormap": "viridis",
                    "colorbar": true
                }
            ]
        },
        "output": {
            "image": "out.png",
            "time": "first"
        }
    });
    fs::write(
        &scene_path,
        serde_json::to_string_pretty(&scene_json).unwrap(),
    )
    .unwrap();

    let mut doc = ViewerDocument::load_path(&scene_path).expect("load series scene");
    assert_eq!(doc.num_time_steps(), 3);
    assert_eq!(doc.time_steps(), &[0.0, 0.5, 1.0]);
    assert_eq!(doc.current_time_step(), 0);
    assert_eq!(doc.current_time_value(), 0.0);
    assert_eq!(
        doc.evaluator()
            .series_source("series")
            .unwrap()
            .load_executions(),
        1
    );
    assert_eq!(doc.active_surface_style().range, Some([0.0, 6.0]));

    // Advance time slider to step 2 (t = 1.0); step 1 is skipped lazily.
    assert!(doc.set_time_step(2).unwrap());
    assert_eq!(doc.current_time_step(), 2);
    assert_eq!(doc.current_time_value(), 1.0);
    assert_eq!(
        doc.evaluator()
            .series_source("series")
            .unwrap()
            .load_executions(),
        2
    );
    assert_eq!(doc.active_surface_style().range, Some([1.0, 7.0]));

    // Step backward to step 1 (t = 0.5).
    assert!(doc.step_backward(false).unwrap());
    assert_eq!(doc.current_time_step(), 1);
    assert_eq!(doc.current_time_value(), 0.5);
    assert_eq!(
        doc.evaluator()
            .series_source("series")
            .unwrap()
            .load_executions(),
        3
    );
    assert_eq!(doc.active_surface_style().range, Some([0.5, 6.5]));

    // Step forward twice with wrap = true: 1 -> 2 -> 0 (both already cached!).
    assert!(doc.step_forward(true).unwrap());
    assert_eq!(doc.current_time_step(), 2);
    assert!(doc.step_forward(true).unwrap());
    assert_eq!(doc.current_time_step(), 0);
    assert_eq!(
        doc.evaluator()
            .series_source("series")
            .unwrap()
            .load_executions(),
        3
    );
}

#[test]
fn hot_reload_detects_scene_changes_preserves_cache_and_handles_invalid_edits() {
    let root = workspace_root();
    let test_dir = root.join("target/render-tests/viewer-hot-reload");
    fs::create_dir_all(&test_dir).unwrap();

    let pvd_abs = root.join("testdata/vtk/series.pvd");
    let scene_path = test_dir.join("hot_reload_scene.json");

    let initial_json = serde_json::json!({
        "version": 1,
        "sources": {
            "flow": { "file": pvd_abs.display().to_string() }
        },
        "filters": {
            "hull": { "type": "surface", "input": "flow" }
        },
        "view": {
            "show": [
                {
                    "input": "hull",
                    "color_by": "transient",
                    "colormap": "viridis",
                    "colorbar": true
                }
            ],
            "camera": { "fit": "all", "azimuth": 30.0, "elevation": 20.0 }
        },
        "output": {
            "image": "out.png",
            "time": "first"
        }
    });
    fs::write(
        &scene_path,
        serde_json::to_string_pretty(&initial_json).unwrap(),
    )
    .unwrap();

    let mut doc = ViewerDocument::load_path(&scene_path).expect("load initial scene");
    assert_eq!(doc.evaluator().filter_executions(), 1);
    assert_eq!(doc.reload_count(), 0);

    // 1. Polling without file modifications returns `Unchanged`.
    assert_eq!(doc.poll_hot_reload().unwrap(), HotReloadOutcome::Unchanged);

    // 2. Modify colormap and title on disk -> hot reload succeeds with `geometry_changed: false`,
    //    and does NOT re-run the surface filter!
    let mut updated_json = initial_json.clone();
    updated_json["view"]["show"][0]["colormap"] = serde_json::json!("cool_warm");
    updated_json["view"]["title"] = serde_json::json!("Hot Reloaded Title");
    fs::write(
        &scene_path,
        serde_json::to_string_pretty(&updated_json).unwrap(),
    )
    .unwrap();

    let outcome1 = doc.poll_hot_reload().expect("poll after colormap edit");
    assert_eq!(
        outcome1,
        HotReloadOutcome::Reloaded {
            geometry_changed: false,
            camera_changed: false,
        }
    );
    assert_eq!(doc.reload_count(), 1);
    assert_eq!(doc.colormap(), ColormapName::CoolWarm);
    assert_eq!(
        doc.annotations().title.as_deref(),
        Some("Hot Reloaded Title")
    );
    assert_eq!(doc.evaluator().filter_executions(), 1);

    // 3. Modify camera azimuth and output.time ("first" -> "last") on disk -> hot reload reports
    //    `geometry_changed: true, camera_changed: true` and advances to step 2 (t = 1.0).
    updated_json["view"]["camera"]["azimuth"] = serde_json::json!(65.0);
    updated_json["output"]["time"] = serde_json::json!("last");
    fs::write(
        &scene_path,
        serde_json::to_string_pretty(&updated_json).unwrap(),
    )
    .unwrap();

    let outcome2 = doc.poll_hot_reload().expect("poll after time/camera edit");
    assert_eq!(
        outcome2,
        HotReloadOutcome::Reloaded {
            geometry_changed: true,
            camera_changed: true,
        }
    );
    assert_eq!(doc.reload_count(), 2);
    assert_eq!(doc.current_time_step(), 2);
    assert_eq!(doc.current_time_value(), 1.0);
    assert_eq!(doc.camera_from_scene().azimuth_deg, 65.0);

    // 4. Write a scene with a typo in `color_by` ("transent") -> hot reload reports `ReloadFailed`
    //    with a helpful suggestion while preserving the last valid state!
    let mut bad_json = updated_json.clone();
    bad_json["view"]["show"][0]["color_by"] = serde_json::json!("transent");
    fs::write(
        &scene_path,
        serde_json::to_string_pretty(&bad_json).unwrap(),
    )
    .unwrap();

    let outcome3 = doc.poll_hot_reload().expect("poll after invalid edit");
    assert!(matches!(outcome3, HotReloadOutcome::ReloadFailed(_)));
    let err_text = doc.last_reload_error().expect("recorded reload error");
    assert!(
        err_text.contains("did you mean \"transient\"?"),
        "expected suggestion in reload error, got: {err_text}"
    );
    // Previous valid state (step 2, CoolWarm) is still intact.
    assert_eq!(doc.current_time_step(), 2);
    assert_eq!(doc.colormap(), ColormapName::CoolWarm);
    assert_eq!(doc.reload_count(), 2);

    // 5. Fix the file back to valid JSON -> error banner clears and reload_count increments.
    fs::write(
        &scene_path,
        serde_json::to_string_pretty(&updated_json).unwrap(),
    )
    .unwrap();
    let outcome4 = doc.poll_hot_reload().expect("poll after fixing file");
    assert!(matches!(outcome4, HotReloadOutcome::Reloaded { .. }));
    assert!(doc.last_reload_error().is_none());
    assert_eq!(doc.reload_count(), 3);
}

#[test]
fn viewer_session_renders_full_gui_and_viewport_offscreen() {
    let root = workspace_root();
    let test_dir = root.join("target/render-tests");
    fs::create_dir_all(&test_dir).unwrap();

    let pvd_abs = root.join("testdata/vtk/series.pvd");
    let scene_path = test_dir.join("viewer_v0_offscreen_scene.json");
    let scene_json = serde_json::json!({
        "version": 1,
        "sources": {
            "series": { "file": pvd_abs.display().to_string() }
        },
        "view": {
            "show": [
                {
                    "input": "series",
                    "color_by": "transient",
                    "colormap": "viridis",
                    "colorbar": true
                }
            ],
            "camera": { "fit": "all", "azimuth": 30.0, "elevation": 20.0 },
            "axes": true,
            "title": "Transient Cube (t = 1.0)"
        },
        "output": {
            "image": "out.png",
            "time": "last"
        }
    });
    fs::write(
        &scene_path,
        serde_json::to_string_pretty(&scene_json).unwrap(),
    )
    .unwrap();

    let doc = ViewerDocument::load_path(&scene_path).expect("load viewer scene");
    let headless = HeadlessContext::new().expect("headless wgpu context");
    let mut egui_renderer = egui_wgpu::Renderer::new(
        headless.device(),
        wgpu::TextureFormat::Rgba8Unorm,
        egui_wgpu::RendererOptions::default(),
    );
    let viewport = ViewportState::new(headless.device(), headless.queue(), &mut egui_renderer, 4)
        .expect("create ViewportState");
    let mut session =
        ViewerSession::new(doc, viewport, headless.device()).expect("create ViewerSession");

    let ctx = egui::Context::default();
    let mut full_output = egui::FullOutput::default();
    for _ in 0..2 {
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(900.0, 560.0),
            )),
            ..Default::default()
        };
        full_output = ctx.run_ui(raw_input, |ui| {
            session.show_ui(ui, headless.device(), headless.queue(), &mut egui_renderer);
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

    let primitives = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [900, 560],
        pixels_per_point: full_output.pixels_per_point,
    };
    let ui_target = OffscreenTarget::new(
        headless.device(),
        900,
        560,
        wgpu::TextureFormat::Rgba8Unorm,
        1,
    )
    .expect("create ui target");
    let mut encoder = headless
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_viewer_v0_encoder"),
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
            label: Some("test_viewer_v0_pass"),
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
        .expect("readback composited viewer image");
    let out_png = test_dir.join("viewer_v0_series.png");
    ui_image
        .save_png(&out_png)
        .expect("save viewer_v0_series.png");

    // Verify the 3D viewport area (x in 300..880) contains rendered mesh and annotation pixels.
    let mut non_white_pixels = 0usize;
    for y in 20..540 {
        for x in 300..880 {
            let [r, g, b, _] = ui_image.pixel(x, y);
            if r < 245 || g < 245 || b < 245 {
                non_white_pixels += 1;
            }
        }
    }
    assert!(
        non_white_pixels > 10_000,
        "expected > 10,000 non-white mesh & annotation pixels in viewer viewport, got {non_white_pixels}"
    );

    // Probe the center of the viewport via GPU ID buffer picking and verify the Value probe state.
    let (vp_w, vp_h) = session.viewport().target_size_px();
    let pick = session
        .pick_at_pixel(headless.device(), headless.queue(), vp_w / 2, vp_h / 2)
        .expect("pick_at_pixel succeeds")
        .expect("center pixel hits transient cube");
    assert_eq!(pick.representation_index, 0);
    assert_eq!(session.picked_representation_name(), Some("series"));
    let [wx, wy, wz] = pick.world_position;
    let expected_transient = (wx + 2.0 * wy + 3.0 * wz) + 1.0; // step 2 => t = 1.0
    let actual_transient = pick.active_scalar.expect("active_scalar present");
    assert!(
        (actual_transient - expected_transient).abs() < 1e-4,
        "picked transient {actual_transient} vs expected {expected_transient}"
    );

    // Run one more UI frame with the active pick to exercise the Value probe panel and overlay.
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(900.0, 560.0),
        )),
        ..Default::default()
    };
    let mut out = ctx.run_ui(raw_input, |ui| {
        session.show_ui(ui, headless.device(), headless.queue(), &mut egui_renderer);
    });
    out.textures_delta.clear();

    session.clear_pick();
    assert!(session.last_pick().is_none());

    session.destroy(&mut egui_renderer);
}

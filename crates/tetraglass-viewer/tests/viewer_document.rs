//! Integration tests for `tetraglass-viewer` (`M1.8`): scene opening, hot reload on file change,
//! time slider stepping, field and colormap pickers, and offscreen `egui_wgpu` rendering.

use std::fs;
use std::path::PathBuf;

use egui_wgpu::wgpu;
use tetraglass_egui::{ViewportColorBy, ViewportColormap, ViewportScalarComponent, ViewportState};
use tetraglass_render::{HeadlessContext, OffscreenTarget};
use tetraglass_scene::{ClipKeep, ClipMode, ColormapName};
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

#[test]
fn interactive_clip_uses_shader_plane_while_dragging_and_swaps_in_worker_result() {
    let root = workspace_root();
    let hex_path = root.join("testdata/vtk/hex_linear.vtu");

    let doc = ViewerDocument::load_path(&hex_path).expect("load hex_linear.vtu");
    let unclipped_tris = doc
        .active_surface()
        .expect("active surface")
        .num_triangles();

    let headless = HeadlessContext::new().expect("headless wgpu context");
    let mut egui_renderer = egui_wgpu::Renderer::new(
        headless.device(),
        wgpu::TextureFormat::Rgba8Unorm,
        egui_wgpu::RendererOptions::default(),
    );
    let mut viewport =
        ViewportState::new(headless.device(), headless.queue(), &mut egui_renderer, 4)
            .expect("create ViewportState");
    viewport
        .ensure_target_size(headless.device(), &mut egui_renderer, 400, 300)
        .expect("resize viewport");

    let mut session =
        ViewerSession::new(doc, viewport, headless.device()).expect("create ViewerSession");
    let mut cam = session.viewport().camera();
    cam.orbit(35.0, 20.0);
    session.viewport_mut().set_camera(cam);
    assert!(!session.clip_state().enabled);
    assert_eq!(session.viewport().clip_plane(), None);

    // 1. Begin dragging a clip plane at x = 0.5, normal = +X, keeping Back (x <= 0.5).
    session
        .begin_or_update_clip_drag(
            headless.device(),
            headless.queue(),
            [0.5, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            ClipKeep::Back,
        )
        .expect("begin drag at x = 0.5");
    assert!(session.clip_state().enabled);
    assert!(session.clip_state().is_dragging);
    assert!(!session.clip_state().worker_busy);
    assert!(session.viewport().clip_plane().is_some());
    assert!(session.active_clipped_surface().is_none());

    // 2. Continue dragging to x = 1.0 (center of the [0, 2]^3 cube): shader clip plane updates immediately.
    session
        .begin_or_update_clip_drag(
            headless.device(),
            headless.queue(),
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            ClipKeep::Back,
        )
        .expect("update drag to x = 1.0");
    assert_eq!(
        session.viewport().clip_plane().unwrap().origin,
        [1.0, 1.0, 1.0]
    );

    // 3. Release the drag: spawns the background worker thread while keeping the shader clip plane active.
    session.end_clip_drag();
    assert!(!session.clip_state().is_dragging);
    assert!(session.viewport().clip_plane().is_some());

    // 4. Wait for the worker thread to finish and swap in the exact clipped surface.
    let swapped = session
        .wait_for_clip_worker(headless.device(), headless.queue())
        .expect("wait_for_clip_worker");
    assert!(swapped, "expected exact clip worker result to swap in");
    assert!(!session.clip_state().worker_busy);
    // Shader clip plane is cleared once the exact clipped surface is swapped in!
    assert_eq!(session.viewport().clip_plane(), None);

    let exact_surface = session
        .active_clipped_surface()
        .expect("active exact clipped surface");
    assert!(
        exact_surface.num_triangles() > unclipped_tris / 2,
        "exact clipped surface ({}) should include interior cap triangles",
        exact_surface.num_triangles()
    );

    // 5. Picking the center of the viewport now hits the interior cut plane at x = 1.0!
    let pick = session
        .pick_at_pixel(headless.device(), headless.queue(), 200, 150)
        .expect("pick center")
        .expect("center pixel hits clipped cube cap");
    assert!(
        (pick.world_position[0] - 1.0).abs() < 1e-10,
        "expected center pick on interior clip cap x = 1.0, got {:?}",
        pick.world_position
    );

    // 6. Test Crinkle mode via background worker as well.
    session.set_clip_mode(ClipMode::Crinkle);
    session
        .begin_or_update_clip_drag(
            headless.device(),
            headless.queue(),
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            ClipKeep::Back,
        )
        .expect("drag crinkle");
    session.end_clip_drag();
    assert!(
        session
            .wait_for_clip_worker(headless.device(), headless.queue())
            .expect("wait crinkle")
    );
    assert_eq!(session.viewport().clip_plane(), None);

    // 7. Disable interactive clipping: restores the unclipped surface.
    session
        .disable_interactive_clip(headless.device(), headless.queue())
        .expect("disable interactive clip");
    assert!(!session.clip_state().enabled);
    assert!(session.active_clipped_surface().is_none());
    assert_eq!(session.viewport().clip_plane(), None);

    session.destroy(&mut egui_renderer);
}

#[test]
fn webgpu_status_messages_and_wgpu_configuration_enforce_webgpu_only() {
    use tetraglass_viewer::{WebGpuStatus, webgpu_only_wgpu_configuration};

    assert!(WebGpuStatus::Supported.is_supported());
    assert!(WebGpuStatus::Supported.error_message().is_none());

    let insecure = WebGpuStatus::NotSecureContext {
        origin: "http://192.168.1.10:8000".to_string(),
    }
    .error_message()
    .expect("NotSecureContext message");
    assert!(insecure.title.contains("secure context"));
    assert!(insecure.to_plain_text().contains("HTTPS or localhost"));
    assert!(insecure.to_html_fragment().contains("WebGPU"));
    assert!(insecure.to_html_fragment().contains("WebGL2"));

    let missing_gpu = WebGpuStatus::MissingNavigatorGpu {
        user_agent: Some("OldBrowser/1.0".to_string()),
    }
    .error_message()
    .expect("MissingNavigatorGpu message");
    assert!(missing_gpu.summary.contains("navigator.gpu"));
    assert!(missing_gpu.to_plain_text().contains("Chrome"));
    assert!(missing_gpu.to_html_fragment().contains("WebGL2"));

    let no_adapter = WebGpuStatus::NoAdapterAvailable {
        details: Some("adapter <script>blocked</script>".to_string()),
    }
    .error_message()
    .expect("NoAdapterAvailable message");
    assert!(no_adapter.summary.contains("navigator.gpu"));
    // Verify HTML escaping of adapter details.
    let html = no_adapter.to_html_fragment();
    assert!(html.contains("&lt;script&gt;blocked&lt;/script&gt;"));
    assert!(!html.contains("<script>blocked</script>"));

    // Verify D-007: WebGPU-only configuration must request BROWSER_WEBGPU and never GL.
    let cfg = webgpu_only_wgpu_configuration();
    match cfg.wgpu_setup {
        egui_wgpu::WgpuSetup::CreateNew(create_new) => {
            assert_eq!(
                create_new.instance_descriptor.backends,
                wgpu::Backends::BROWSER_WEBGPU
            );
            assert!(
                !create_new
                    .instance_descriptor
                    .backends
                    .contains(wgpu::Backends::GL),
                "WebGL2 backend must never be enabled"
            );
        }
        other => panic!("expected WgpuSetup::CreateNew, got {other:?}"),
    }
}

#[test]
fn url_resolution_and_query_parsing_handle_relative_paths_and_percent_encoding() {
    use tetraglass_viewer::{
        extract_url_from_query_str, parse_query_map, resolve_relative_url, url_display_label,
    };

    assert_eq!(
        resolve_relative_url(
            "https://example.org/tetraglass/gallery/slice.json",
            "../testdata/vtk/mixed_linear.vtu"
        ),
        "https://example.org/tetraglass/testdata/vtk/mixed_linear.vtu"
    );
    assert_eq!(
        resolve_relative_url("https://example.org/data/series.pvd", "series_1.vtu"),
        "https://example.org/data/series_1.vtu"
    );
    assert_eq!(
        resolve_relative_url(
            "https://example.org/a/b/scene.json?token=123",
            "/meshes/tet.msh"
        ),
        "https://example.org/meshes/tet.msh"
    );
    assert_eq!(
        resolve_relative_url("gallery/slice.json", "../testdata/vtk/mixed_linear.vtu"),
        "testdata/vtk/mixed_linear.vtu"
    );
    assert_eq!(
        url_display_label("https://example.org/gallery/slice.json?cache=0#top"),
        "slice.json"
    );

    let map =
        parse_query_map("?scene=https%3A%2F%2Fexample.org%2Fgallery%2Fslice.json&foo=bar+baz");
    assert_eq!(
        map.get("foo").and_then(|v| v.first()).map(String::as_str),
        Some("bar baz")
    );
    assert_eq!(
        extract_url_from_query_str("?scene=https%3A%2F%2Fexample.org%2Fgallery%2Fslice.json"),
        Some("https://example.org/gallery/slice.json".to_string())
    );
    assert_eq!(
        extract_url_from_query_str("?url=testdata/vtk/mixed_linear.vtu"),
        Some("testdata/vtk/mixed_linear.vtu".to_string())
    );
    assert_eq!(
        extract_url_from_query_str("?file=testdata/gmsh/mixed_linear.msh"),
        Some("testdata/gmsh/mixed_linear.msh".to_string())
    );
    assert_eq!(extract_url_from_query_str("?other=123"), None);
}

#[derive(Debug)]
struct MemoryDroppedFile {
    path: PathBuf,
    bytes: Vec<u8>,
}

impl eframe::egui::DroppedFile for MemoryDroppedFile {
    fn path(&self) -> &std::path::Path {
        &self.path
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn bytes(&self) -> Result<Vec<u8>, String> {
        Ok(self.bytes.clone())
    }

    #[cfg(target_arch = "wasm32")]
    fn bytes_async(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>, String>> + '_>> {
        let bytes = self.bytes.clone();
        Box::pin(async move { Ok(bytes) })
    }
}

#[test]
fn viewer_session_supports_drag_and_drop_and_url_loading_for_meshes_series_and_scenes() {
    use std::sync::Arc;
    use tetraglass_viewer::DroppedFilePayload;

    let root = workspace_root();
    let vtu_bytes = fs::read(root.join("testdata/vtk/mixed_linear.vtu")).expect("read vtu");
    let hex_bytes = fs::read(root.join("testdata/vtk/hex_linear.vtu")).expect("read hex_linear");
    let msh_bytes = fs::read(root.join("testdata/gmsh/tet_v22.msh")).expect("read msh");
    let pvd_bytes = fs::read(root.join("testdata/vtk/series.pvd")).expect("read pvd");
    let step0_bytes = fs::read(root.join("testdata/vtk/series_0.vtu")).expect("read step0");
    let step1_bytes = fs::read(root.join("testdata/vtk/series_1.vtu")).expect("read step1");
    let step2_bytes = fs::read(root.join("testdata/vtk/series_2.vtu")).expect("read step2");
    let slice_json_bytes = fs::read(root.join("gallery/slice.json")).expect("read slice.json");

    // Start from the synthetic default demo (used when the web viewer opens with no query URL).
    let demo_doc = ViewerDocument::default_demo().expect("default_demo");
    assert!(demo_doc.active_surface().unwrap().num_triangles() > 0);

    let headless = HeadlessContext::new().expect("headless wgpu context");
    let mut egui_renderer = egui_wgpu::Renderer::new(
        headless.device(),
        wgpu::TextureFormat::Rgba8Unorm,
        egui_wgpu::RendererOptions::default(),
    );
    let mut viewport =
        ViewportState::new(headless.device(), headless.queue(), &mut egui_renderer, 4)
            .expect("create ViewportState");
    viewport
        .ensure_target_size(headless.device(), &mut egui_renderer, 320, 240)
        .expect("resize viewport");

    let mut session =
        ViewerSession::new(demo_doc, viewport, headless.device()).expect("create ViewerSession");

    // 1. Drag-and-drop a raw `.msh` file via `ingest_dropped_files`.
    assert!(
        session
            .ingest_dropped_files(
                [DroppedFilePayload::new(
                    "mixed_linear.msh",
                    msh_bytes.clone()
                )],
                headless.device(),
                headless.queue(),
            )
            .expect("drop .msh")
    );
    assert_eq!(session.document().file_label(), "mixed_linear.msh");
    assert!(session.last_load_error().is_none());

    // 2. Drag-and-drop a `.json` scene together with its companion `.vtu` mesh (even when `.vtu` is listed first).
    assert!(
        session
            .ingest_dropped_files(
                [
                    DroppedFilePayload::new("hex_linear.vtu", hex_bytes.clone()),
                    DroppedFilePayload::new("slice.json", slice_json_bytes.clone()),
                ],
                headless.device(),
                headless.queue(),
            )
            .expect("drop scene + companion mesh")
    );
    assert_eq!(session.document().file_label(), "slice.json");

    // 3. Drag-and-drop a `.pvd` time series together with its step `.vtu` files.
    assert!(
        session
            .ingest_dropped_files(
                [
                    DroppedFilePayload::new("series_0.vtu", step0_bytes.clone()),
                    DroppedFilePayload::new("series_1.vtu", step1_bytes.clone()),
                    DroppedFilePayload::new("series_2.vtu", step2_bytes.clone()),
                    DroppedFilePayload::new("series.pvd", pvd_bytes.clone()),
                ],
                headless.device(),
                headless.queue(),
            )
            .expect("drop pvd + steps")
    );
    assert_eq!(session.document().file_label(), "series.pvd");
    assert_eq!(session.document().num_time_steps(), 3);

    // 4. Simulate dropping a `.vtu` file directly through `egui::RawInput::dropped_files` in `show_ui`.
    let egui_ctx = eframe::egui::Context::default();
    let dropped_handle: eframe::egui::DroppedFileHandle = Arc::new(MemoryDroppedFile {
        path: PathBuf::from("dropped_via_egui.vtu"),
        bytes: vtu_bytes.clone(),
    });
    let raw_input = eframe::egui::RawInput {
        screen_rect: Some(eframe::egui::Rect::from_min_size(
            eframe::egui::Pos2::ZERO,
            eframe::egui::vec2(640.0, 480.0),
        )),
        dropped_files: vec![dropped_handle],
        ..Default::default()
    };
    let mut full_output = egui_ctx.run_ui(raw_input, |ui| {
        session.show_ui(ui, headless.device(), headless.queue(), &mut egui_renderer);
    });
    full_output.textures_delta.clear();
    assert_eq!(session.document().file_label(), "dropped_via_egui.vtu");

    // 5. Load a `.json` scene and a `.pvd` series from simulated remote URLs via `load_from_url_with_fetcher`.
    let fetcher = |requested_url: &str| -> Result<Vec<u8>, String> {
        match requested_url {
            "https://example.org/viz/gallery/slice.json" => Ok(slice_json_bytes.clone()),
            "https://example.org/viz/testdata/vtk/hex_linear.vtu" => Ok(hex_bytes.clone()),
            "https://example.org/viz/testdata/vtk/series.pvd" => Ok(pvd_bytes.clone()),
            "https://example.org/viz/testdata/vtk/series_0.vtu" => Ok(step0_bytes.clone()),
            "https://example.org/viz/testdata/vtk/series_1.vtu" => Ok(step1_bytes.clone()),
            "https://example.org/viz/testdata/vtk/series_2.vtu" => Ok(step2_bytes.clone()),
            other => Err(format!("HTTP 404 for {other}")),
        }
    };

    session
        .load_from_url_with_fetcher(
            "https://example.org/viz/gallery/slice.json",
            fetcher,
            headless.device(),
            headless.queue(),
        )
        .expect("load slice.json from URL");
    assert_eq!(session.document().file_label(), "slice.json");
    assert_eq!(
        session.url_input(),
        "https://example.org/viz/gallery/slice.json"
    );

    session
        .load_from_url_with_fetcher(
            "https://example.org/viz/testdata/vtk/series.pvd",
            fetcher,
            headless.device(),
            headless.queue(),
        )
        .expect("load series.pvd from URL");
    assert_eq!(session.document().file_label(), "series.pvd");
    assert_eq!(session.document().num_time_steps(), 3);

    // 6. Verify that a failed URL load records `last_load_error` while preserving the active document.
    let err = session
        .load_from_url_with_fetcher(
            "https://example.org/viz/missing.vtu",
            fetcher,
            headless.device(),
            headless.queue(),
        )
        .expect_err("404 URL should fail");
    assert!(err.to_string().contains("404"));
    assert!(session.last_load_error().unwrap().contains("404"));
    assert_eq!(session.document().file_label(), "series.pvd");

    session.destroy(&mut egui_renderer);
}

#[test]
fn shareable_link_roundtrips_scene_plus_remote_data_urls_and_view_overrides() {
    use tetraglass_viewer::{ShareableLinkSpec, ShareableViewOverrides};

    let root = workspace_root();
    let hex_bytes = fs::read(root.join("testdata/vtk/hex_linear.vtu")).expect("read hex_linear");
    let slice_json_bytes = fs::read(root.join("gallery/slice.json")).expect("read slice.json");
    let pvd_bytes = fs::read(root.join("testdata/vtk/series.pvd")).expect("read pvd");
    let step0_bytes = fs::read(root.join("testdata/vtk/series_0.vtu")).expect("read step0");
    let step1_bytes = fs::read(root.join("testdata/vtk/series_1.vtu")).expect("read step1");
    let step2_bytes = fs::read(root.join("testdata/vtk/series_2.vtu")).expect("read step2");

    // 1. Build a shareable link pointing to a remote scene URL + a separate CDN mesh URL + camera/style overrides.
    let mut data_urls = std::collections::BTreeMap::new();
    data_urls.insert(
        "mesh".to_string(),
        "https://cdn.example.org/meshes/custom_hex.vtu".to_string(),
    );
    let spec = ShareableLinkSpec {
        primary_url: Some("https://scenes.example.org/gallery/slice.json".to_string()),
        inline_scene: None,
        inline_bundle: None,
        data_urls,
        view_overrides: ShareableViewOverrides {
            azimuth_deg: Some(115.0),
            elevation_deg: Some(28.0),
            roll_deg: Some(5.0),
            zoom: Some(1.4),
            step: None,
            color_by: Some("solid".to_string()),
            colormap: Some(ColormapName::Turbo),
            log_scale: Some(true),
            colorbar: Some(false),
            axes: Some(false),
        },
    };
    let link = spec
        .to_viewer_url("https://tetraglass.example/viewer/")
        .expect("encode viewer URL");
    assert!(link.starts_with("https://tetraglass.example/viewer/?scene="));
    assert!(link.contains("data.mesh=https%3A%2F%2Fcdn.example.org%2Fmeshes%2Fcustom_hex.vtu"));

    // Notice the fetcher ONLY serves the mesh from `cdn.example.org` (not relative to `scenes.example.org`),
    // proving the `data.mesh` query parameter overrode the scene's relative source path.
    let fetcher = |requested_url: &str| -> Result<Vec<u8>, String> {
        match requested_url {
            "https://scenes.example.org/gallery/slice.json" => Ok(slice_json_bytes.clone()),
            "https://cdn.example.org/meshes/custom_hex.vtu" => Ok(hex_bytes.clone()),
            "https://scenes.example.org/data/series.pvd" => Ok(pvd_bytes.clone()),
            "https://scenes.example.org/data/series_0.vtu" => Ok(step0_bytes.clone()),
            "https://scenes.example.org/data/series_1.vtu" => Ok(step1_bytes.clone()),
            "https://scenes.example.org/data/series_2.vtu" => Ok(step2_bytes.clone()),
            other => Err(format!("HTTP 404 for {other}")),
        }
    };

    let doc = ViewerDocument::from_url_with_fetcher(&link, fetcher)
        .expect("load shareable scene + data link");
    assert_eq!(doc.file_label(), "slice.json");
    assert_eq!(
        doc.source_url(),
        Some("https://scenes.example.org/gallery/slice.json")
    );
    let cam = doc.camera_from_scene();
    assert!((cam.azimuth_deg - 115.0).abs() < 1e-6);
    assert!((cam.elevation_deg - 28.0).abs() < 1e-6);
    assert!((cam.roll_deg - 5.0).abs() < 1e-6);
    assert!((cam.zoom - 1.4).abs() < 1e-6);
    assert_eq!(doc.selected_field_index(), 0); // "solid"
    assert_eq!(doc.colormap(), ColormapName::Turbo);
    assert!(doc.use_log_scale());
    assert!(!doc.colorbar_enabled());
    assert!(!doc.axes_enabled());

    // 2. Also verify a shareable link with inline compressed scene JSON (`?scene_b64=...`) + remote data URL.
    let inline_scene_spec = ShareableLinkSpec {
        primary_url: None,
        inline_scene: Some(
            tetraglass_scene::Scene::from_json_bytes(&slice_json_bytes).expect("parse slice.json"),
        ),
        inline_bundle: None,
        data_urls: [(
            "*".to_string(),
            "https://cdn.example.org/meshes/custom_hex.vtu".to_string(),
        )]
        .into_iter()
        .collect(),
        view_overrides: ShareableViewOverrides {
            azimuth_deg: Some(62.5),
            ..Default::default()
        },
    };
    let inline_scene_link = inline_scene_spec
        .to_viewer_url("https://tetraglass.example/viewer/")
        .expect("encode inline scene link");
    assert!(inline_scene_link.contains("scene_b64="));
    assert!(inline_scene_link.contains("data=https%3A%2F%2Fcdn.example.org"));

    let inline_doc = ViewerDocument::from_url_with_fetcher(&inline_scene_link, fetcher)
        .expect("load inline scene_b64 + remote data URL");
    assert!((inline_doc.camera_from_scene().azimuth_deg - 62.5).abs() < 1e-6);
    assert!(inline_doc.active_surface().unwrap().num_triangles() > 0);

    // 3. Verify time-step override (`?scene=...series.pvd&step=2`).
    let series_link = "https://tetraglass.example/viewer/?scene=https%3A%2F%2Fscenes.example.org%2Fdata%2Fseries.pvd&step=2";
    let series_doc =
        ViewerDocument::from_url_with_fetcher(series_link, fetcher).expect("load series link");
    assert_eq!(series_doc.num_time_steps(), 3);
    assert_eq!(series_doc.current_time_step(), 2);
}

#[test]
fn viewer_session_builds_and_opens_self_contained_and_remote_shareable_links() {
    use tetraglass_viewer::DroppedFilePayload;

    let root = workspace_root();
    let hex_bytes = fs::read(root.join("testdata/vtk/hex_linear.vtu")).expect("read hex_linear");
    let slice_json_bytes = fs::read(root.join("gallery/slice.json")).expect("read slice.json");
    let pvd_bytes = fs::read(root.join("testdata/vtk/series.pvd")).expect("read pvd");
    let step0_bytes = fs::read(root.join("testdata/vtk/series_0.vtu")).expect("read step0");
    let step1_bytes = fs::read(root.join("testdata/vtk/series_1.vtu")).expect("read step1");
    let step2_bytes = fs::read(root.join("testdata/vtk/series_2.vtu")).expect("read step2");

    let headless = HeadlessContext::new().expect("headless wgpu context");
    let mut egui_renderer = egui_wgpu::Renderer::new(
        headless.device(),
        wgpu::TextureFormat::Rgba8Unorm,
        egui_wgpu::RendererOptions::default(),
    );
    let viewport1 = ViewportState::new(headless.device(), headless.queue(), &mut egui_renderer, 4)
        .expect("create ViewportState 1");
    let mut session1 = ViewerSession::new(
        ViewerDocument::default_demo().expect("demo"),
        viewport1,
        headless.device(),
    )
    .expect("create session1");

    // 1. Drop `slice.json` + `hex_linear.vtu` into `session1`, customize camera & colormap, and build a shareable link.
    session1
        .ingest_dropped_files(
            [
                DroppedFilePayload::new("slice.json", slice_json_bytes),
                DroppedFilePayload::new("hex_linear.vtu", hex_bytes),
            ],
            headless.device(),
            headless.queue(),
        )
        .expect("ingest dropped scene + mesh");
    let expected_tris = session1
        .document()
        .active_surface()
        .unwrap()
        .num_triangles();
    let mut cam = session1.viewport().camera();
    cam.azimuth_deg = 77.5;
    cam.elevation_deg = 41.25;
    cam.zoom = 1.65;
    session1.viewport_mut().set_camera(cam);
    session1
        .document_mut()
        .set_colormap(ColormapName::CoolWarm)
        .expect("set colormap");
    assert!(session1.document_mut().set_axes_enabled(false));

    let bundle_link = session1
        .build_shareable_link("https://tetraglass.example/viewer/")
        .expect("build self-contained shareable link");
    assert!(bundle_link.starts_with("https://tetraglass.example/viewer/?bundle_b64="));

    // 2. Open `bundle_link` in a brand-new `session2` with a fetcher that rejects all network traffic.
    let viewport2 = ViewportState::new(headless.device(), headless.queue(), &mut egui_renderer, 4)
        .expect("create ViewportState 2");
    let mut session2 = ViewerSession::new(
        ViewerDocument::default_demo().expect("demo 2"),
        viewport2,
        headless.device(),
    )
    .expect("create session2");

    session2
        .load_from_url_with_fetcher(
            &bundle_link,
            |unexpected_url| Err(format!("unexpected network fetch: {unexpected_url}")),
            headless.device(),
            headless.queue(),
        )
        .expect("open self-contained bundle_b64 link offline");

    assert_eq!(session2.document().file_label(), "slice.json");
    assert_eq!(
        session2
            .document()
            .active_surface()
            .unwrap()
            .num_triangles(),
        expected_tris
    );
    assert!((session2.viewport().camera().azimuth_deg - 77.5).abs() < 1e-6);
    assert!((session2.viewport().camera().elevation_deg - 41.25).abs() < 1e-6);
    assert!((session2.viewport().camera().zoom - 1.65).abs() < 1e-6);
    assert_eq!(session2.document().colormap(), ColormapName::CoolWarm);
    assert!(!session2.document().axes_enabled());

    // 3. Load a remote `.pvd` series into `session1`, change step & camera, build a compact URL link, and open it in `session2`.
    let fetcher = |requested_url: &str| -> Result<Vec<u8>, String> {
        match requested_url {
            "https://example.org/viz/testdata/vtk/series.pvd" => Ok(pvd_bytes.clone()),
            "https://example.org/viz/testdata/vtk/series_0.vtu" => Ok(step0_bytes.clone()),
            "https://example.org/viz/testdata/vtk/series_1.vtu" => Ok(step1_bytes.clone()),
            "https://example.org/viz/testdata/vtk/series_2.vtu" => Ok(step2_bytes.clone()),
            other => Err(format!("HTTP 404 for {other}")),
        }
    };
    session1
        .load_from_url_with_fetcher(
            "https://example.org/viz/testdata/vtk/series.pvd",
            fetcher,
            headless.device(),
            headless.queue(),
        )
        .expect("load remote pvd");
    session1
        .document_mut()
        .set_time_step(2)
        .expect("advance to step 2");
    let mut cam2 = session1.viewport().camera();
    cam2.azimuth_deg = 132.0;
    session1.viewport_mut().set_camera(cam2);

    let remote_link = session1
        .build_shareable_link("https://tetraglass.example/viewer/")
        .expect("build remote shareable link");
    assert!(remote_link.starts_with(
        "https://tetraglass.example/viewer/?scene=https%3A%2F%2Fexample.org%2Fviz%2Ftestdata%2Fvtk%2Fseries.pvd"
    ));
    assert!(remote_link.contains("step=2"));
    assert!(remote_link.contains("az=132"));

    session2
        .load_from_url_with_fetcher(&remote_link, fetcher, headless.device(), headless.queue())
        .expect("open remote shareable link in session2");
    assert_eq!(session2.document().file_label(), "series.pvd");
    assert_eq!(session2.document().current_time_step(), 2);
    assert!((session2.viewport().camera().azimuth_deg - 132.0).abs() < 1e-6);

    session1.destroy(&mut egui_renderer);
    session2.destroy(&mut egui_renderer);
}

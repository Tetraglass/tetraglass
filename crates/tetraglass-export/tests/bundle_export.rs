//! Integration and round-trip tests for the Tetraglass bundle format (`M3.5`, `docs/bundle-format.md`).

use std::fs;
use std::path::{Path, PathBuf};

use tetraglass_export::{
    BUNDLE_FORMAT_ID, BUNDLE_FORMAT_VERSION, BundleMeshData, ExportError, TGBIN_HEADER_BYTES,
    TGBIN_MAGIC, decode_tgbin, encode_tgbin, export_evaluated_scene_to_bundle,
    export_scene_to_bundle, normalize_scalar, read_bundle_dir, sample_colormap_srgb,
    write_bundle_dir, write_scene_bundle_output,
};
use tetraglass_io::{
    MeshFormat, detect_format_from_bytes, detect_format_from_extension, read_msh_series,
    read_pvd_series_with_loader, read_vtu_series,
};
use tetraglass_scene::{RepresentationStyle, Scene, SceneEvaluator};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn load_scene_and_evaluator(scene_path: &Path) -> (Scene, SceneEvaluator) {
    let bytes = fs::read(scene_path).expect("read scene json");
    let scene = Scene::from_json_bytes(&bytes).expect("parse scene json");
    let base_dir = scene_path.parent().expect("scene parent dir");
    let mut evaluator = SceneEvaluator::new();

    for (name, src) in &scene.sources {
        let src_path = base_dir.join(&src.file);
        let src_bytes = fs::read(&src_path).expect("read source file");
        let fmt = src
            .format
            .map(MeshFormat::from)
            .or_else(|| {
                src_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .and_then(detect_format_from_extension)
            })
            .or_else(|| detect_format_from_bytes(&src_bytes))
            .expect("detect mesh format");
        let series = match fmt {
            MeshFormat::Vtu => read_vtu_series(src_bytes).expect("read vtu"),
            MeshFormat::Gmsh => read_msh_series(src_bytes).expect("read msh"),
            MeshFormat::Pvd => {
                let pvd_dir = src_path.parent().expect("pvd parent").to_path_buf();
                read_pvd_series_with_loader(&src_bytes, move |rel| Ok(fs::read(pvd_dir.join(rel))?))
                    .expect("read pvd")
            }
            other => panic!("unexpected format {other:?}"),
        };
        evaluator.insert_series_source(name.clone(), series);
    }

    (scene, evaluator)
}

#[test]
fn gallery_scenes_round_trip_through_bundle_writer_and_reader() {
    let root = workspace_root();
    assert!(
        root.join("docs/bundle-format.md").is_file(),
        "docs/bundle-format.md must exist"
    );

    for scene_rel in [
        "gallery/slice.json",
        "gallery/contour.json",
        "gallery/clip.json",
        "gallery/threshold.json",
        "gallery/surface-mixed-linear.json",
        "gallery/gmsh-quadratic-tetra.json",
    ] {
        let scene_path = root.join(scene_rel);
        let (scene, mut evaluator) = load_scene_and_evaluator(&scene_path);
        let evaluated = evaluator.evaluate_scene(&scene).expect("evaluate scene");

        let archive =
            export_evaluated_scene_to_bundle(&scene, &evaluated).expect("export bundle archive");
        assert_eq!(archive.manifest.version, BUNDLE_FORMAT_VERSION);
        assert_eq!(archive.manifest.format, BUNDLE_FORMAT_ID);
        assert_eq!(archive.manifest.units, "meters");
        assert_eq!(archive.manifest.time_steps.len(), 1);
        assert_eq!(
            archive.manifest.objects.len(),
            evaluated.representations.len()
        );

        // Check camera and lighting unit vectors.
        let cam = &archive.manifest.camera;
        for axis in 0..3 {
            assert!(
                (cam.world_position[axis]
                    - (cam.position[axis] + archive.manifest.dataset_center[axis]))
                    .abs()
                    < 1e-12
            );
            assert!(
                (cam.world_focal_point[axis]
                    - (cam.focal_point[axis] + archive.manifest.dataset_center[axis]))
                    .abs()
                    < 1e-12
            );
        }
        let up_len = (cam.up[0] * cam.up[0] + cam.up[1] * cam.up[1] + cam.up[2] * cam.up[2]).sqrt();
        assert!((up_len - 1.0).abs() < 1e-9);
        assert!(cam.clip_end > cam.clip_start && cam.clip_start > 0.0);

        let lights = &archive.manifest.lights;
        for dir in [
            lights.key_direction,
            lights.fill_direction,
            lights.headlight_direction,
        ] {
            let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-9);
        }

        // Write bundle to disk and read it back via `read_bundle_dir`.
        let stem = Path::new(scene_rel).file_stem().unwrap().to_str().unwrap();
        let out_dir = root.join(format!("target/bundle-export-tests/{stem}"));
        let _ = fs::remove_dir_all(&out_dir);
        let summary = write_bundle_dir(&out_dir, &archive).expect("write bundle dir");
        assert!(summary.manifest_file.is_file());
        assert_eq!(summary.mesh_files.len(), archive.mesh_files.len());

        let round_tripped = read_bundle_dir(&out_dir).expect("read bundle dir");
        assert_eq!(round_tripped.manifest.version, archive.manifest.version);
        assert_eq!(
            round_tripped.manifest.objects.len(),
            archive.manifest.objects.len()
        );
        assert_eq!(round_tripped.mesh_files, archive.mesh_files);

        for (obj, rep) in round_tripped
            .manifest
            .objects
            .iter()
            .zip(evaluated.representations.iter())
        {
            if let Some(coloring) = &rep.coloring {
                let cmap = obj.colormap.as_ref().expect("bundle colormap");
                assert_eq!(cmap.name, coloring.colormap);
                assert_eq!(cmap.lut_srgb.len(), 256);
                assert_eq!(
                    cmap.lut_srgb[0],
                    sample_colormap_srgb(coloring.colormap, 0.0)
                );
                assert_eq!(
                    cmap.lut_srgb[255],
                    sample_colormap_srgb(coloring.colormap, 1.0)
                );
            }

            let mesh_ref = &obj.meshes[0];
            let decoded = round_tripped.decode_mesh(mesh_ref).expect("decode mesh");
            assert_eq!(decoded.triangles.len(), rep.surface.num_triangles());
            assert_eq!(decoded.positions.len(), rep.surface.num_triangles() * 3);
            assert_eq!(decoded.normals.len(), decoded.positions.len());
            assert_eq!(decoded.scalars.len(), decoded.positions.len());
            assert_eq!(decoded.raw_scalars.len(), decoded.positions.len());

            if matches!(
                rep.style,
                RepresentationStyle::SurfaceWithEdges | RepresentationStyle::Wireframe
            ) {
                assert_eq!(
                    decoded.edge_positions.len(),
                    rep.surface.wireframe_edges().len()
                );
            } else {
                assert!(decoded.edge_positions.is_empty());
            }

            // Reconstruct first vertex world position (`pos + dataset_center`, `D-014`).
            let tri0 = rep.surface.triangles()[0];
            let expected_p0 = rep.surface.points()[tri0[0] as usize];
            let actual_p0 = [
                f64::from(decoded.positions[0][0]) + round_tripped.manifest.dataset_center[0],
                f64::from(decoded.positions[0][1]) + round_tripped.manifest.dataset_center[1],
                f64::from(decoded.positions[0][2]) + round_tripped.manifest.dataset_center[2],
            ];
            for k in 0..3 {
                assert!(
                    (expected_p0[k] - actual_p0[k]).abs() < 1e-5,
                    "world position mismatch on axis {k} in {scene_rel}"
                );
            }

            if let Some(coloring) = &rep.coloring {
                for (&norm_s, &raw_s) in decoded.scalars.iter().zip(decoded.raw_scalars.iter()) {
                    assert!((0.0..=1.0).contains(&norm_s));
                    let expected_norm = normalize_scalar(
                        f64::from(raw_s),
                        coloring.effective_range,
                        coloring.log_scale,
                        coloring.center,
                    );
                    assert!((norm_s - expected_norm).abs() < 1e-5);
                }
            }
        }
    }
}

#[test]
fn multi_step_series_bundle_exports_all_steps_and_round_trips() {
    let root = workspace_root();
    let (scene, mut evaluator) =
        load_scene_and_evaluator(&root.join("gallery/series-transient-last.json"));

    let out_dir = root.join("target/bundle-export-tests/series_bundle");
    let _ = fs::remove_dir_all(&out_dir);

    let summary =
        write_scene_bundle_output(&out_dir, &scene, &mut evaluator, true).expect("write bundle");
    assert_eq!(summary.num_steps, 3);
    assert_eq!(summary.mesh_files.len(), 3);

    let loaded = read_bundle_dir(&out_dir).expect("read series bundle");
    assert_eq!(loaded.manifest.time_steps.len(), 3);
    assert_eq!(loaded.manifest.objects.len(), 1);
    assert_eq!(loaded.manifest.objects[0].meshes.len(), 3);

    let mut mean_scalars = Vec::new();
    for (idx, mesh_ref) in loaded.manifest.objects[0].meshes.iter().enumerate() {
        assert_eq!(mesh_ref.step, idx);
        let decoded = loaded.decode_mesh(mesh_ref).expect("decode step mesh");
        let sum: f32 = decoded.raw_scalars.iter().sum();
        mean_scalars.push(sum / (decoded.raw_scalars.len() as f32));
    }
    assert!(
        (mean_scalars[0] - mean_scalars[1]).abs() > 1e-3
            && (mean_scalars[1] - mean_scalars[2]).abs() > 1e-3
    );

    // Also verify subset step selection via `export_scene_to_bundle(..., Some(&[0, 2]))`.
    let subset =
        export_scene_to_bundle(&scene, &mut evaluator, Some(&[0, 2])).expect("export subset");
    assert_eq!(subset.manifest.time_steps.len(), 2);
    assert_eq!(subset.manifest.time_steps[0].step, 0);
    assert_eq!(subset.manifest.time_steps[1].step, 2);
}

#[test]
fn tgbin_decoder_rejects_corrupted_headers_and_out_of_bounds_indices() {
    assert!(matches!(
        decode_tgbin(&[0u8; 10]),
        Err(ExportError::InvalidBundle(_))
    ));

    let mut bad_magic = [0u8; TGBIN_HEADER_BYTES];
    bad_magic[0..4].copy_from_slice(b"BAD!");
    assert!(matches!(
        decode_tgbin(&bad_magic),
        Err(ExportError::InvalidBundle(_))
    ));

    let mut bad_len = [0u8; TGBIN_HEADER_BYTES];
    bad_len[0..4].copy_from_slice(&TGBIN_MAGIC.to_le_bytes());
    bad_len[4..8].copy_from_slice(&1u32.to_le_bytes());
    bad_len[8..12].copy_from_slice(&3u32.to_le_bytes()); // V = 3, but no payload bytes follow
    assert!(matches!(
        decode_tgbin(&bad_len),
        Err(ExportError::InvalidBundle(_))
    ));

    let bad_mesh = BundleMeshData {
        positions: vec![[0.0, 0.0, 0.0]; 3],
        triangles: vec![[0, 1, 9]], // index 9 out of bounds for V = 3
        normals: vec![[0.0, 0.0, 1.0]; 3],
        scalars: vec![0.5; 3],
        raw_scalars: vec![1.0; 3],
        edge_positions: Vec::new(),
    };
    assert!(matches!(
        encode_tgbin(&bad_mesh),
        Err(ExportError::InvalidBundle(_))
    ));
}

#[test]
fn blender_scripts_spdx_headers_addon_registration_and_headless_cycles_render() {
    use std::process::Command;

    let root = workspace_root();
    let blender_dir = root.join("blender");

    // 1. Enforce `.agents/rules/blender-gpl.md`: every Python file in `blender/` must start
    //    with `# SPDX-License-Identifier: GPL-3.0-or-later`.
    for rel_py in ["__init__.py", "importer.py", "render.py"] {
        let py_path = blender_dir.join(rel_py);
        let content = fs::read_to_string(&py_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", py_path.display()));
        let first_line = content.lines().next().unwrap_or("");
        assert_eq!(
            first_line,
            "# SPDX-License-Identifier: GPL-3.0-or-later",
            "{} must start with GPL-3.0-or-later SPDX header",
            py_path.display()
        );
    }
    assert!(blender_dir.join("blender_manifest.toml").is_file());

    // 2. Export `gallery/slice.json` and `gallery/series-transient-last.json` to bundle directories.
    let (slice_scene, mut slice_eval) = load_scene_and_evaluator(&root.join("gallery/slice.json"));
    let slice_bundle_dir = root.join("target/bundle-export-tests/blender_slice_bundle");
    let _ = fs::remove_dir_all(&slice_bundle_dir);
    write_scene_bundle_output(&slice_bundle_dir, &slice_scene, &mut slice_eval, false)
        .expect("write slice bundle");

    let (series_scene, mut series_eval) =
        load_scene_and_evaluator(&root.join("gallery/series-transient-last.json"));
    let series_bundle_dir = root.join("target/bundle-export-tests/blender_series_bundle");
    let _ = fs::remove_dir_all(&series_bundle_dir);
    write_scene_bundle_output(&series_bundle_dir, &series_scene, &mut series_eval, true)
        .expect("write series bundle");

    // 3. Locate an installed Blender binary if present (`BLENDER_PATH`, `PATH`, or standard locations).
    let candidate_bins = [
        std::env::var("BLENDER_PATH").ok(),
        Some("blender".to_string()),
        Some("/opt/homebrew/bin/blender".to_string()),
        Some("/Applications/Blender.app/Contents/MacOS/Blender".to_string()),
    ];
    let blender_bin = candidate_bins.into_iter().flatten().find(|bin| {
        Command::new(bin)
            .arg("--version")
            .output()
            .is_ok_and(|out| out.status.success())
    });

    let Some(blender_bin) = blender_bin else {
        // When running on a machine without Blender installed, verify the pure-Python syntax of
        // `blender/*.py` if `python3` is available.
        if let Ok(py_out) = Command::new("python3")
            .arg("-m")
            .arg("py_compile")
            .arg(blender_dir.join("__init__.py"))
            .arg(blender_dir.join("importer.py"))
            .arg(blender_dir.join("render.py"))
            .output()
        {
            assert!(
                py_out.status.success(),
                "python3 py_compile failed: {}",
                String::from_utf8_lossy(&py_out.stderr)
            );
        }
        return;
    };

    // 4. Verify add-on registration and interactive operator invocation inside headless Blender.
    let addon_check_expr = format!(
        "import sys, bpy; sys.path.insert(0, r'{}'); import blender; blender.register(); \
         res = bpy.ops.import_scene.tetraglass_bundle(filepath=r'{}', clear_existing=True); \
         assert 'FINISHED' in res, f'unexpected operator result: {{res}}'; \
         blender.unregister()",
        root.display(),
        slice_bundle_dir.join("manifest.json").display()
    );
    let addon_out = Command::new(&blender_bin)
        .arg("--background")
        .arg("--factory-startup")
        .arg("--python-expr")
        .arg(&addon_check_expr)
        .output()
        .expect("run Blender add-on registration test");
    assert!(
        addon_out.status.success(),
        "Blender add-on registration failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&addon_out.stdout),
        String::from_utf8_lossy(&addon_out.stderr)
    );

    // 5. Run `blender/render.py` on `slice_bundle_dir` with Cycles (`--samples 4 --device cpu`)
    //    and `--save-blend`.
    let out_png = root.join("target/bundle-export-tests/blender_slice_out.png");
    let out_blend = root.join("target/bundle-export-tests/blender_slice_out.blend");
    let _ = fs::remove_file(&out_png);
    let _ = fs::remove_file(&out_blend);

    let render_out = Command::new(&blender_bin)
        .arg("--background")
        .arg("--factory-startup")
        .arg("--python")
        .arg(blender_dir.join("render.py"))
        .arg("--")
        .arg("--bundle")
        .arg(&slice_bundle_dir)
        .arg("--out")
        .arg(&out_png)
        .arg("--samples")
        .arg("4")
        .arg("--device")
        .arg("cpu")
        .arg("--width")
        .arg("240")
        .arg("--height")
        .arg("160")
        .arg("--save-blend")
        .arg(&out_blend)
        .output()
        .expect("run blender/render.py on slice bundle");
    let stdout = String::from_utf8_lossy(&render_out.stdout);
    let stderr = String::from_utf8_lossy(&render_out.stderr);
    assert!(
        render_out.status.success(),
        "blender/render.py failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stdout.contains("PROGRESS stage=import status=starting"));
    assert!(stdout.contains("PROGRESS stage=import status=done objects=3 steps=1"));
    assert!(stdout.contains("PROGRESS stage=cycles device=CPU samples=4 resolution=240x160"));
    assert!(stdout.contains("PROGRESS stage=save_blend"));
    assert!(stdout.contains("PROGRESS stage=render frame=0 index=1 total_frames=1 status=done"));
    assert!(stdout.contains("PROGRESS stage=complete frames=1"));

    assert!(out_blend.is_file(), ".blend file should exist");
    assert!(
        fs::metadata(&out_blend).unwrap().len() > 1024,
        ".blend file should be non-empty"
    );

    assert!(out_png.is_file(), "rendered PNG should exist");
    let png_bytes = fs::read(&out_png).expect("read rendered PNG");
    assert_eq!(&png_bytes[0..8], b"\x89PNG\r\n\x1a\n");
    let width = u32::from_be_bytes(png_bytes[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(png_bytes[20..24].try_into().unwrap());
    assert_eq!((width, height), (240, 160));

    // 6. Run `blender/render.py --all-steps` on the 3-step `series_bundle_dir`.
    let frame_pattern = root.join("target/bundle-export-tests/series_frames/frame_%04d.png");
    let _ = fs::remove_dir_all(frame_pattern.parent().unwrap());
    let series_render_out = Command::new(&blender_bin)
        .arg("--background")
        .arg("--factory-startup")
        .arg("--python")
        .arg(blender_dir.join("render.py"))
        .arg("--")
        .arg("--bundle")
        .arg(&series_bundle_dir)
        .arg("--out")
        .arg(&frame_pattern)
        .arg("--samples")
        .arg("2")
        .arg("--device")
        .arg("cpu")
        .arg("--width")
        .arg("160")
        .arg("--height")
        .arg("120")
        .arg("--all-steps")
        .output()
        .expect("run blender/render.py --all-steps on series bundle");
    assert!(
        series_render_out.status.success(),
        "series blender/render.py failed:\n{}",
        String::from_utf8_lossy(&series_render_out.stderr)
    );
    for step_idx in 0..3 {
        let frame_path = root.join(format!(
            "target/bundle-export-tests/series_frames/frame_{step_idx:04}.png"
        ));
        assert!(
            frame_path.is_file(),
            "missing rendered frame {}",
            frame_path.display()
        );
    }

    // 7. Spawn a persistent Blender worker (`M5.4`), load `slice_bundle_dir`, render once,
    //    push a live camera update (`push_live_scene_view`) without re-importing meshes,
    //    render a second time in the same warm session, and verify the outputs differ.
    let mut worker = tetraglass_export::BlenderWorkerProcess::spawn(
        &blender_bin,
        blender_dir.join("render.py"),
        "127.0.0.1:0",
        std::time::Duration::from_secs(30),
    )
    .expect("spawn persistent Blender worker");
    assert!(!worker.addr().is_empty());
    assert!(!worker.blender_version().is_empty());

    let client = worker.client();
    let ping_resp = client.ping().expect("ping persistent Blender worker");
    assert!(ping_resp.is_ok());
    assert!(ping_resp.loaded_bundle.is_none());

    let load_resp = client
        .load_bundle(&slice_bundle_dir)
        .expect("load slice bundle in worker");
    assert!(load_resp.is_ok());
    assert!(load_resp.loaded_bundle.is_some());

    let worker_png_1 = root.join("target/bundle-export-tests/worker_slice_cam1.png");
    let worker_png_2 = root.join("target/bundle-export-tests/worker_slice_cam2.png");
    let _ = fs::remove_file(&worker_png_1);
    let _ = fs::remove_file(&worker_png_2);

    let mut render_req_1 = tetraglass_export::BlenderWorkerRenderRequest::new(
        worker_png_1.display().to_string(),
        2,
        "cpu",
    );
    render_req_1.width = Some(160);
    render_req_1.height = Some(120);
    let render_resp_1 = client
        .render(&render_req_1)
        .expect("render first view on warm worker");
    assert!(render_resp_1.is_ok());
    assert!(worker_png_1.is_file());

    let mut orbited_scene = slice_scene.clone();
    orbited_scene.view.camera.azimuth += 95.0;
    orbited_scene.view.camera.elevation = -25.0;
    let slice_eval_res = slice_eval
        .evaluate_scene(&slice_scene)
        .expect("evaluate slice scene");
    let bounds = slice_eval_res.camera_bounds.expect("slice camera bounds");
    let cam_resp = client
        .push_live_scene_view(&orbited_scene, bounds, Some(0))
        .expect("push live camera update to warm worker");
    assert!(cam_resp.is_ok());

    let mut render_req_2 = tetraglass_export::BlenderWorkerRenderRequest::new(
        worker_png_2.display().to_string(),
        2,
        "cpu",
    );
    render_req_2.width = Some(160);
    render_req_2.height = Some(120);
    let render_resp_2 = client
        .render(&render_req_2)
        .expect("render second view on warm worker after live camera update");
    assert!(render_resp_2.is_ok());
    assert!(worker_png_2.is_file());

    let bytes_1 = fs::read(&worker_png_1).expect("read worker_png_1");
    let bytes_2 = fs::read(&worker_png_2).expect("read worker_png_2");
    assert_ne!(
        bytes_1, bytes_2,
        "live camera update in persistent Blender worker must change rendered image"
    );

    worker
        .shutdown()
        .expect("shutdown persistent Blender worker");
}

#[test]
fn blender_worker_client_protocol_round_trips_and_propagates_errors() {
    use std::thread;
    use std::time::{Duration, Instant};
    use tetraglass_export::{
        BlenderWorkerClient, BlenderWorkerRenderRequest, BlenderWorkerRequest,
    };

    let root = workspace_root();
    let ipc_dir = root.join("target/bundle-export-tests/mock_worker_ipc");
    let _ = fs::remove_dir_all(&ipc_dir);
    fs::create_dir_all(&ipc_dir).expect("create mock ipc dir");
    let addr = format!("ipc:{}", ipc_dir.display());

    let server_dir = ipc_dir.clone();
    let server_handle = thread::spawn(move || {
        let mut handled = 0;
        let deadline = Instant::now() + Duration::from_secs(10);
        while handled < 4 && Instant::now() < deadline {
            let mut entries: Vec<_> = fs::read_dir(&server_dir)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    name.starts_with("req_") && name.ends_with(".json")
                })
                .collect();
            entries.sort_by_key(|e| e.file_name());
            if entries.is_empty() {
                thread::sleep(Duration::from_millis(5));
                continue;
            }
            for entry in entries {
                let path = entry.path();
                let stem = path.file_stem().unwrap().to_string_lossy();
                let req_id = &stem[4..];
                let line = fs::read_to_string(&path).expect("read request file");
                let _ = fs::remove_file(&path);
                let req: BlenderWorkerRequest =
                    serde_json::from_str(line.trim()).expect("parse request JSON");
                let reply = match req {
                    BlenderWorkerRequest::Ping => serde_json::json!({
                        "status": "ok",
                        "blender_version": "4.2.0-mock",
                        "loaded_bundle": null
                    }),
                    BlenderWorkerRequest::UpdateCamera { camera, step, .. } => serde_json::json!({
                        "status": "ok",
                        "blender_version": "4.2.0-mock",
                        "view": {
                            "position": camera.position,
                            "step": step
                        }
                    }),
                    BlenderWorkerRequest::Render(r) => {
                        if r.out == "fail.png" {
                            serde_json::json!({
                                "status": "error",
                                "error": "simulated render failure"
                            })
                        } else {
                            serde_json::json!({
                                "status": "ok",
                                "files": [r.out],
                                "device": "CPU",
                                "samples": r.samples,
                                "blender_version": "4.2.0-mock"
                            })
                        }
                    }
                    BlenderWorkerRequest::Shutdown => serde_json::json!({
                        "status": "ok",
                        "shutting_down": true
                    }),
                    BlenderWorkerRequest::LoadBundle { bundle } => serde_json::json!({
                        "status": "ok",
                        "loaded_bundle": bundle
                    }),
                };
                let mut out = serde_json::to_string(&reply).unwrap();
                out.push('\n');
                let resp_path = server_dir.join(format!("resp_{req_id}.json"));
                fs::write(&resp_path, out.as_bytes()).unwrap();
                handled += 1;
            }
        }
    });

    let client = BlenderWorkerClient::new(&addr);
    let ping = client.ping().expect("mock ping");
    assert_eq!(ping.blender_version.as_deref(), Some("4.2.0-mock"));

    let ok_render = client
        .render(&BlenderWorkerRenderRequest::new("ok.png", 8, "cpu"))
        .expect("mock render ok");
    assert_eq!(
        ok_render.files.as_deref(),
        Some(&["ok.png".to_string()][..])
    );

    let err_render = client.render(&BlenderWorkerRenderRequest::new("fail.png", 8, "cpu"));
    assert!(err_render.is_err());
    assert!(
        err_render
            .unwrap_err()
            .to_string()
            .contains("simulated render failure")
    );

    let shut = client.shutdown().expect("mock shutdown");
    assert_eq!(shut.shutting_down, Some(true));

    server_handle.join().expect("join mock worker server");
}

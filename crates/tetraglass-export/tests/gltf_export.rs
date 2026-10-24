//! Integration tests for Binary glTF 2.0 (`.glb`) scene and time-series export (`M3.4`).

use std::fs;
use std::path::{Path, PathBuf};

use tetraglass_export::{
    ExportError, GltfSeriesManifest, export_evaluated_scene_to_glb,
    export_time_series_to_glb_frames, linear_to_srgb, normalize_scalar, parse_glb,
    sample_colormap_srgb, write_scene_glb_output,
};
use tetraglass_io::{
    MeshFormat, detect_format_from_bytes, detect_format_from_extension, read_msh_series,
    read_pvd_series_with_loader, read_vtu_series,
};
use tetraglass_scene::{ColormapName, Scene, SceneEvaluator};

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
fn gallery_scenes_export_valid_glb_with_baked_colors_and_normalized_scalar_in_texcoord1() {
    let root = workspace_root();
    for scene_rel in [
        "gallery/slice.json",
        "gallery/contour.json",
        "gallery/clip.json",
        "gallery/threshold.json",
        "gallery/surface-mixed-linear.json",
    ] {
        let scene_path = root.join(scene_rel);
        let (scene, mut evaluator) = load_scene_and_evaluator(&scene_path);
        let evaluated = evaluator.evaluate_scene(&scene).expect("evaluate scene");

        let glb_bytes = export_evaluated_scene_to_glb(&scene, &evaluated).expect("export glb");
        assert!(
            glb_bytes.len().is_multiple_of(4),
            "GLB output for {scene_rel} must be 4-byte aligned"
        );

        let parsed = parse_glb(&glb_bytes).expect("parse exported glb");
        assert_eq!(parsed.root.asset.version, "2.0");
        assert_eq!(parsed.meshes.len(), evaluated.representations.len());

        for (mesh, rep) in parsed.meshes.iter().zip(evaluated.representations.iter()) {
            let expected_verts = rep.surface.num_triangles() * 3;
            assert_eq!(mesh.positions.len(), expected_verts);
            assert_eq!(mesh.normals.len(), expected_verts);
            assert_eq!(mesh.colors_rgba.len(), expected_verts);
            assert_eq!(mesh.texcoord_0.len(), expected_verts);
            assert_eq!(mesh.texcoord_1.len(), expected_verts);
            assert_eq!(mesh.indices.len(), expected_verts);

            // Check that POSITION + node translation reconstructs world-space coordinates (`D-014`).
            let first_tri = rep.surface.triangles()[0];
            let p0_world = rep.surface.points()[first_tri[0] as usize];
            let p0_reconstructed = [
                f64::from(mesh.positions[0][0]) + mesh.translation[0],
                f64::from(mesh.positions[0][1]) + mesh.translation[1],
                f64::from(mesh.positions[0][2]) + mesh.translation[2],
            ];
            for k in 0..3 {
                assert!(
                    (p0_world[k] - p0_reconstructed[k]).abs() < 1e-5,
                    "world coordinate mismatch at axis {k} in {scene_rel}"
                );
            }

            // Check that every normal is unit length and every TEXCOORD_1.x is in [0.0, 1.0].
            for i in 0..expected_verts {
                let n = mesh.normals[i];
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                assert!(
                    (len - 1.0).abs() < 1e-4,
                    "non-unit normal {n:?} in {scene_rel}"
                );

                let norm_scalar = mesh.texcoord_1[i][0];
                let raw_scalar = mesh.texcoord_1[i][1];
                assert!(
                    (0.0..=1.0).contains(&norm_scalar),
                    "normalized scalar {norm_scalar} outside [0, 1] in {scene_rel}"
                );
                assert_eq!(
                    mesh.texcoord_0[i][0], norm_scalar,
                    "TEXCOORD_0.x and TEXCOORD_1.x should carry the same normalized scalar"
                );

                if let Some(coloring) = &rep.coloring {
                    let expected_t = normalize_scalar(
                        f64::from(raw_scalar),
                        coloring.effective_range,
                        coloring.log_scale,
                        coloring.center,
                    );
                    assert!(
                        (norm_scalar - expected_t).abs() < 1e-5,
                        "TEXCOORD_1.x ({norm_scalar}) does not match normalized raw scalar ({expected_t})"
                    );
                    let expected_srgb = sample_colormap_srgb(coloring.colormap, norm_scalar);
                    let actual_srgb = [
                        linear_to_srgb(mesh.colors_rgba[i][0]),
                        linear_to_srgb(mesh.colors_rgba[i][1]),
                        linear_to_srgb(mesh.colors_rgba[i][2]),
                    ];
                    for c in 0..3 {
                        assert!(
                            (expected_srgb[c] - actual_srgb[c]).abs() < 1e-4,
                            "baked vertex color mismatch on channel {c} in {scene_rel}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn transparent_and_cell_colored_representations_set_blend_mode_and_exact_attributes() {
    let root = workspace_root();
    let (mut scene, mut evaluator) =
        load_scene_and_evaluator(&root.join("gallery/surface-mixed-linear.json"));
    scene.view.show[0].color_by = Some("cell:cell_id".to_string());
    scene.view.show[0].colormap = ColormapName::Turbo;
    scene.view.show[0].opacity = 0.45;

    let evaluated = evaluator.evaluate_scene(&scene).expect("evaluate scene");
    let glb_bytes = export_evaluated_scene_to_glb(&scene, &evaluated).expect("export glb");
    let parsed = parse_glb(&glb_bytes).expect("parse glb");

    assert_eq!(parsed.root.materials[0].alpha_mode, "BLEND");
    assert!(parsed.root.materials[0].double_sided);
    let mesh = &parsed.meshes[0];
    let extras = mesh.extras.as_ref().expect("mesh extras");
    assert_eq!(extras.color_by.as_deref(), Some("cell:cell_id"));
    assert_eq!(extras.colormap.as_deref(), Some("turbo"));
    assert!((extras.opacity - 0.45).abs() < 1e-6);

    for rgba in &mesh.colors_rgba {
        assert!((rgba[3] - 0.45).abs() < 1e-5);
    }
    // Cell data must assign identical scalar and color to all 3 vertices of each triangle.
    for tri in 0..evaluated.representations[0].surface.num_triangles() {
        let base = tri * 3;
        assert_eq!(mesh.texcoord_1[base], mesh.texcoord_1[base + 1]);
        assert_eq!(mesh.texcoord_1[base], mesh.texcoord_1[base + 2]);
        assert_eq!(mesh.colors_rgba[base], mesh.colors_rgba[base + 1]);
    }
}

#[test]
fn time_series_exports_one_glb_per_step_and_writes_manifest() {
    let root = workspace_root();
    let (scene, mut evaluator) =
        load_scene_and_evaluator(&root.join("gallery/series-transient-last.json"));

    let series_export =
        export_time_series_to_glb_frames(&scene, &mut evaluator, "time_series", None)
            .expect("export time series frames");
    assert_eq!(series_export.frames.len(), 3);
    assert_eq!(series_export.manifest.steps.len(), 3);
    assert_eq!(series_export.manifest.format, "gltf-series");

    let mut mean_scalar_per_step = Vec::new();
    for (idx, frame) in series_export.frames.iter().enumerate() {
        assert_eq!(frame.step, idx);
        assert_eq!(frame.file_name, format!("time_series_step_{idx:04}.glb"));
        let parsed = parse_glb(&frame.glb_bytes).expect("parse step glb");
        let step_extras = parsed.root.scenes[0].extras.as_ref().expect("scene extras");
        assert_eq!(step_extras.time_step, idx);
        assert!((step_extras.time_value - frame.time).abs() < 1e-9);

        let sum_raw: f32 = parsed.meshes[0].texcoord_1.iter().map(|uv| uv[1]).sum();
        mean_scalar_per_step.push(sum_raw / (parsed.meshes[0].texcoord_1.len() as f32));
    }
    // In `series.pvd`, the field values evolve across steps 0, 1, 2.
    assert!(
        (mean_scalar_per_step[0] - mean_scalar_per_step[1]).abs() > 1e-3
            && (mean_scalar_per_step[1] - mean_scalar_per_step[2]).abs() > 1e-3,
        "expected distinct scalar values across time steps: {mean_scalar_per_step:?}"
    );

    // Verify `write_scene_glb_output` writes all 3 `.glb` files plus `manifest.json` on disk.
    let out_dir = root.join("target/gltf-export-tests/time_series");
    let _ = fs::remove_dir_all(&out_dir);
    let summary =
        write_scene_glb_output(&out_dir, &scene, &mut evaluator, true).expect("write series");
    assert_eq!(summary.num_steps, 3);
    assert_eq!(summary.glb_files.len(), 3);
    for path in &summary.glb_files {
        let bytes = fs::read(path).expect("read written step glb");
        let parsed = parse_glb(&bytes).expect("parse written step glb");
        assert_eq!(parsed.meshes.len(), 1);
    }
    let manifest_path = summary.manifest_file.expect("manifest path");
    let manifest_bytes = fs::read(&manifest_path).expect("read manifest.json");
    let manifest: GltfSeriesManifest =
        serde_json::from_slice(&manifest_bytes).expect("parse manifest.json");
    assert_eq!(manifest, series_export.manifest);
}

#[test]
fn parse_glb_rejects_corrupted_or_truncated_containers() {
    assert!(matches!(
        parse_glb(b"short"),
        Err(ExportError::InvalidGlb(_))
    ));
    let mut bad_magic = [0u8; 24];
    bad_magic[0..4].copy_from_slice(b"NOPE");
    assert!(matches!(
        parse_glb(&bad_magic),
        Err(ExportError::InvalidGlb(_))
    ));
}

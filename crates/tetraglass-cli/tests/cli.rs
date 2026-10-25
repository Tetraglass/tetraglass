//! Integration tests for the `tetraglass` CLI (`M1.4`): `schema`, `inspect`, `validate`, and `render`.

use std::path::PathBuf;
use std::process::Command;

use tetraglass_render::{DEFAULT_CHANNEL_TOLERANCE, DEFAULT_MAX_DIFF_FRACTION, RenderedImage};

fn tetraglass_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_tetraglass"))
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn schema_command_matches_committed_snapshot() {
    let output = tetraglass_bin()
        .arg("schema")
        .output()
        .expect("run `tetraglass schema`");
    assert!(
        output.status.success(),
        "`tetraglass schema` failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).expect("UTF-8 schema output");
    let snapshot_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots/scene-v1.schema.json");
    let expected = std::fs::read_to_string(&snapshot_path).expect("read committed schema snapshot");

    assert_eq!(
        stdout.trim_end(),
        expected.trim_end(),
        "`tetraglass schema` output drifted from {}; update snapshot when changing scene types",
        snapshot_path.display()
    );
}

#[test]
fn inspect_command_outputs_text_and_json() {
    let vtu_path = workspace_root().join("testdata/vtk/mixed_linear.vtu");

    // 1. Human-readable text output
    let text_out = tetraglass_bin()
        .arg("inspect")
        .arg(&vtu_path)
        .output()
        .expect("run `tetraglass inspect`");
    assert!(
        text_out.status.success(),
        "inspect text failed: {}",
        String::from_utf8_lossy(&text_out.stderr)
    );
    let text = String::from_utf8(text_out.stdout).unwrap();
    assert!(text.contains("Points: 11 | Cells: 4"));
    assert!(text.contains("linear (f64, 1 component)"));
    assert!(text.contains("velocity (f64, 3 components)"));
    assert!(text.contains("cell_id"));

    // 2. Structured `--json` output
    let json_out = tetraglass_bin()
        .arg("inspect")
        .arg(&vtu_path)
        .arg("--json")
        .output()
        .expect("run `tetraglass inspect --json`");
    assert!(
        json_out.status.success(),
        "inspect --json failed: {}",
        String::from_utf8_lossy(&json_out.stderr)
    );
    let value: serde_json::Value =
        serde_json::from_slice(&json_out.stdout).expect("parse inspect JSON");
    assert_eq!(value["format"], "vtu");
    assert_eq!(value["num_points"], 11);
    assert_eq!(value["num_cells"], 4);
    assert_eq!(value["cell_types"]["hexahedron"], 1);
    assert_eq!(value["cell_types"]["wedge"], 1);
    assert_eq!(value["cell_types"]["pyramid"], 1);
    assert_eq!(value["cell_types"]["tetra"], 1);
    assert_eq!(value["point_arrays"].as_array().unwrap().len(), 3);
    assert_eq!(value["cell_arrays"].as_array().unwrap().len(), 1);

    // 3. Structured `--json` output on a Gmsh `.msh` file (`M1.6`)
    let msh_path = workspace_root().join("testdata/gmsh/tet2_v41.msh");
    let msh_out = tetraglass_bin()
        .arg("inspect")
        .arg(&msh_path)
        .arg("--json")
        .output()
        .expect("run `tetraglass inspect --json` on .msh");
    assert!(
        msh_out.status.success(),
        "inspect .msh failed: {}",
        String::from_utf8_lossy(&msh_out.stderr)
    );
    let msh_val: serde_json::Value =
        serde_json::from_slice(&msh_out.stdout).expect("parse msh inspect JSON");
    assert_eq!(msh_val["format"], "msh");
    assert_eq!(msh_val["num_points"], 27);
    assert_eq!(msh_val["num_cells"], 6);
    assert_eq!(msh_val["cell_types"]["quadratic_tetra"], 6);
    assert_eq!(msh_val["point_arrays"].as_array().unwrap().len(), 3);
    assert_eq!(msh_val["cell_arrays"].as_array().unwrap().len(), 1);

    // 4. Structured `--json` output on a ParaView `.pvd` time series (`M1.7`)
    let pvd_path = workspace_root().join("testdata/vtk/series.pvd");
    let pvd_out = tetraglass_bin()
        .arg("inspect")
        .arg(&pvd_path)
        .arg("--json")
        .output()
        .expect("run `tetraglass inspect --json` on .pvd");
    assert!(
        pvd_out.status.success(),
        "inspect .pvd failed: {}",
        String::from_utf8_lossy(&pvd_out.stderr)
    );
    let pvd_val: serde_json::Value =
        serde_json::from_slice(&pvd_out.stdout).expect("parse pvd inspect JSON");
    assert_eq!(pvd_val["format"], "pvd");
    assert_eq!(pvd_val["num_points"], 8);
    assert_eq!(pvd_val["num_cells"], 6);
    assert_eq!(pvd_val["time_steps"], serde_json::json!([0.0, 0.5, 1.0]));
}

#[test]
fn validate_command_accepts_valid_scene_and_reports_suggestions_on_invalid_scene() {
    let root = workspace_root();
    let test_dir = root.join("target/render-tests/cli-validate");
    std::fs::create_dir_all(&test_dir).unwrap();

    let vtu_abs = root.join("testdata/vtk/mixed_linear.vtu");
    let valid_scene_path = test_dir.join("valid_scene.json");
    let valid_json = serde_json::json!({
        "version": 1,
        "sources": {
            "mesh": { "file": vtu_abs.display().to_string() }
        },
        "view": {
            "show": [
                { "input": "mesh", "color_by": "linear" }
            ]
        }
    });
    std::fs::write(
        &valid_scene_path,
        serde_json::to_string_pretty(&valid_json).unwrap(),
    )
    .unwrap();

    let ok_out = tetraglass_bin()
        .arg("validate")
        .arg(&valid_scene_path)
        .output()
        .expect("run validate on valid scene");
    assert!(
        ok_out.status.success(),
        "validate failed on valid scene: {}",
        String::from_utf8_lossy(&ok_out.stderr)
    );
    assert!(String::from_utf8_lossy(&ok_out.stdout).contains("is valid"));

    let ok_json_out = tetraglass_bin()
        .arg("validate")
        .arg(&valid_scene_path)
        .arg("--json")
        .output()
        .expect("run validate --json on valid scene");
    assert!(ok_json_out.status.success());
    let ok_val: serde_json::Value = serde_json::from_slice(&ok_json_out.stdout).unwrap();
    assert_eq!(ok_val["valid"], true);

    // Invalid scene with a typo in `color_by` ("linar" instead of "linear")
    let invalid_scene_path = test_dir.join("invalid_scene.json");
    let invalid_json = serde_json::json!({
        "version": 1,
        "sources": {
            "mesh": { "file": vtu_abs.display().to_string() }
        },
        "view": {
            "show": [
                { "input": "mesh", "color_by": "linar" }
            ]
        }
    });
    std::fs::write(
        &invalid_scene_path,
        serde_json::to_string_pretty(&invalid_json).unwrap(),
    )
    .unwrap();

    let bad_out = tetraglass_bin()
        .arg("validate")
        .arg(&invalid_scene_path)
        .output()
        .expect("run validate on invalid scene");
    assert!(!bad_out.status.success());
    let stderr = String::from_utf8_lossy(&bad_out.stderr);
    assert!(
        stderr.contains(
            "view.show[0].color_by: no array \"linar\" on \"mesh\"; did you mean \"linear\"?"
        ),
        "unexpected stderr: {stderr}"
    );

    let bad_json_out = tetraglass_bin()
        .arg("validate")
        .arg(&invalid_scene_path)
        .arg("--json")
        .output()
        .expect("run validate --json on invalid scene");
    assert!(!bad_json_out.status.success());
    let bad_val: serde_json::Value = serde_json::from_slice(&bad_json_out.stdout).unwrap();
    assert_eq!(bad_val["valid"], false);
    assert_eq!(bad_val["issues"][0]["path"], "view.show[0].color_by");
    assert_eq!(
        bad_val["issues"][0]["suggestion"],
        "did you mean \"linear\"?"
    );
}

#[test]
fn render_command_writes_png_and_prints_json_summary() {
    let root = workspace_root();
    let test_dir = root.join("target/render-tests");
    std::fs::create_dir_all(&test_dir).unwrap();

    let vtu_abs = root.join("testdata/vtk/mixed_linear.vtu");
    let scene_path = test_dir.join("cli_mixed_linear_scene.json");
    let png_path = test_dir.join("cli_mixed_linear.png");
    let _ = std::fs::remove_file(&png_path);

    let scene_json = serde_json::json!({
        "version": 1,
        "sources": {
            "mesh": { "file": vtu_abs.display().to_string() }
        },
        "view": {
            "show": [
                {
                    "input": "mesh",
                    "color_by": "linear",
                    "colormap": "viridis"
                }
            ],
            "camera": {
                "fit": "all",
                "azimuth": 30.0,
                "elevation": 20.0
            },
            "axes": false
        },
        "output": {
            "image": "default_out.png",
            "size": [800, 600],
            "msaa": 4
        }
    });
    std::fs::write(
        &scene_path,
        serde_json::to_string_pretty(&scene_json).unwrap(),
    )
    .unwrap();

    let render_out = tetraglass_bin()
        .arg("render")
        .arg(&scene_path)
        .arg("-o")
        .arg(&png_path)
        .output()
        .expect("run `tetraglass render`");
    assert!(
        render_out.status.success(),
        "`tetraglass render` failed: {}",
        String::from_utf8_lossy(&render_out.stderr)
    );

    let summary: serde_json::Value =
        serde_json::from_slice(&render_out.stdout).expect("parse render JSON summary");
    assert_eq!(summary["size"], serde_json::json!([800, 600]));
    assert_eq!(summary["msaa"], 4);
    assert_eq!(summary["engine"], "raster");
    assert_eq!(summary["representations"][0]["input"], "mesh");
    assert_eq!(summary["representations"][0]["color_by"], "linear");
    assert_eq!(summary["representations"][0]["association"], "point");
    assert_eq!(summary["representations"][0]["colormap"], "viridis");
    assert_eq!(summary["representations"][0]["num_faces"], 12);
    assert_eq!(summary["representations"][0]["num_triangles"], 18);
    assert_eq!(summary["camera"]["azimuth"], 30.0);
    assert_eq!(summary["camera"]["elevation"], 20.0);

    let rendered = RenderedImage::load_png(&png_path).expect("load CLI rendered PNG");
    assert_eq!(rendered.width(), 800);
    assert_eq!(rendered.height(), 600);

    // Compare against the M0.9 baseline for `mixed_linear_linear.png` (rendered at the same camera & size).
    let baseline_path = root.join("tests/baselines/mixed_linear_linear.png");
    let baseline = RenderedImage::load_png(&baseline_path).expect("load baseline PNG");
    let cmp = rendered
        .compare(
            &baseline,
            DEFAULT_CHANNEL_TOLERANCE,
            DEFAULT_MAX_DIFF_FRACTION,
        )
        .expect("compare dimensions");
    assert!(
        cmp.passed,
        "CLI render differed from baseline: {} / {} pixels ({:.3}%)",
        cmp.differing_pixels,
        cmp.total_pixels,
        cmp.diff_fraction * 100.0
    );

    // Also render a scene with `colorbar: true`, `axes: true`, and `title` enabled and verify
    // that the CLI draws the annotations into the output PNG.
    let annotated_scene_path = test_dir.join("cli_mixed_linear_annotated_scene.json");
    let annotated_png_path = test_dir.join("cli_mixed_linear_annotated.png");
    let _ = std::fs::remove_file(&annotated_png_path);

    let annotated_scene_json = serde_json::json!({
        "version": 1,
        "sources": {
            "mesh": { "file": vtu_abs.display().to_string() }
        },
        "view": {
            "show": [
                {
                    "input": "mesh",
                    "color_by": "linear",
                    "colormap": "viridis",
                    "colorbar": { "title": "linear", "position": "right" }
                }
            ],
            "camera": {
                "fit": "all",
                "azimuth": 30.0,
                "elevation": 20.0
            },
            "axes": true,
            "title": "Mixed Linear Mesh"
        },
        "output": {
            "image": "annotated_out.png",
            "size": [800, 600],
            "msaa": 4
        }
    });
    std::fs::write(
        &annotated_scene_path,
        serde_json::to_string_pretty(&annotated_scene_json).unwrap(),
    )
    .unwrap();

    let annotated_out = tetraglass_bin()
        .arg("render")
        .arg(&annotated_scene_path)
        .arg("-o")
        .arg(&annotated_png_path)
        .output()
        .expect("run `tetraglass render` with annotations");
    assert!(
        annotated_out.status.success(),
        "`tetraglass render` annotated failed: {}",
        String::from_utf8_lossy(&annotated_out.stderr)
    );

    let annotated_img =
        RenderedImage::load_png(&annotated_png_path).expect("load annotated CLI PNG");
    let cmp_vs_unannotated = annotated_img
        .compare(
            &rendered,
            DEFAULT_CHANNEL_TOLERANCE,
            DEFAULT_MAX_DIFF_FRACTION,
        )
        .expect("compare annotated vs unannotated");
    assert!(
        cmp_vs_unannotated.differing_pixels > 4000,
        "expected annotations (title, colorbar, axes) to add > 4000 foreground pixels, got {}",
        cmp_vs_unannotated.differing_pixels
    );

    // Render a `.pvd` time-series scene at `"time": "last"` (`t = 1.0`) and verify the summary (`M1.7`).
    let pvd_abs = root.join("testdata/vtk/series.pvd");
    let pvd_scene_path = test_dir.join("cli_series_pvd_scene.json");
    let pvd_png_path = test_dir.join("cli_series_pvd_last.png");
    let _ = std::fs::remove_file(&pvd_png_path);

    let pvd_scene_json = serde_json::json!({
        "version": 1,
        "sources": {
            "series": { "file": pvd_abs.display().to_string() }
        },
        "view": {
            "show": [
                {
                    "input": "series",
                    "color_by": "transient",
                    "colormap": "viridis"
                }
            ]
        },
        "output": {
            "image": "series_last.png",
            "size": [400, 300],
            "time": "last"
        }
    });
    std::fs::write(
        &pvd_scene_path,
        serde_json::to_string_pretty(&pvd_scene_json).unwrap(),
    )
    .unwrap();

    let pvd_render_out = tetraglass_bin()
        .arg("render")
        .arg(&pvd_scene_path)
        .arg("-o")
        .arg(&pvd_png_path)
        .output()
        .expect("run `tetraglass render` on pvd series");
    assert!(
        pvd_render_out.status.success(),
        "`tetraglass render` on pvd failed: {}",
        String::from_utf8_lossy(&pvd_render_out.stderr)
    );
    let pvd_summary: serde_json::Value =
        serde_json::from_slice(&pvd_render_out.stdout).expect("parse pvd render summary");
    assert_eq!(pvd_summary["time_step"], 2);
    assert_eq!(pvd_summary["time"], 1.0);
    assert_eq!(
        pvd_summary["representations"][0]["data_range"],
        serde_json::json!([1.0, 7.0])
    );
    assert!(pvd_png_path.exists());
}

#[test]
fn view_command_supports_help_and_validates_scene_before_opening_window() {
    let help_out = tetraglass_bin()
        .arg("view")
        .arg("--help")
        .output()
        .expect("run `tetraglass view --help`");
    assert!(help_out.status.success());
    let stdout = String::from_utf8_lossy(&help_out.stdout);
    assert!(stdout.contains("Usage: tetraglass view <scene.json|mesh>"));

    // Passing a nonexistent scene file fails cleanly before creating a window.
    let missing_out = tetraglass_bin()
        .arg("view")
        .arg("nonexistent_scene.json")
        .output()
        .expect("run `tetraglass view nonexistent_scene.json`");
    assert!(!missing_out.status.success());
    let stderr = String::from_utf8_lossy(&missing_out.stderr);
    assert!(stderr.contains("tetraglass:"));
}

#[test]
fn gallery_scenes_and_author_scene_skill_example_validate_and_match_baselines() {
    let root = workspace_root();
    let gallery_dir = root.join("gallery");
    let out_dir = root.join("target/render-tests");
    let diff_dir = root.join("target/render-diffs");
    std::fs::create_dir_all(&out_dir).unwrap();
    std::fs::create_dir_all(&diff_dir).unwrap();

    let gallery_names = [
        "surface-mixed-linear",
        "surface-cell-id",
        "surface-velocity-magnitude",
        "gmsh-quadratic-tetra",
        "series-transient-last",
        "slice",
        "contour",
        "clip",
        "threshold",
    ];

    let mut failures = Vec::new();
    for name in gallery_names {
        let scene_path = gallery_dir.join(format!("{name}.json"));
        assert!(
            scene_path.is_file(),
            "missing gallery scene at {}",
            scene_path.display()
        );

        let val_out = tetraglass_bin()
            .arg("validate")
            .arg(&scene_path)
            .output()
            .expect("validate gallery scene");
        assert!(
            val_out.status.success(),
            "gallery scene `{name}.json` failed validation: {}",
            String::from_utf8_lossy(&val_out.stderr)
        );

        let rendered_path = out_dir.join(format!("{name}.png"));
        let _ = std::fs::remove_file(&rendered_path);
        let render_out = tetraglass_bin()
            .arg("render")
            .arg(&scene_path)
            .arg("-o")
            .arg(&rendered_path)
            .output()
            .expect("render gallery scene");
        assert!(
            render_out.status.success(),
            "gallery scene `{name}.json` failed to render: {}",
            String::from_utf8_lossy(&render_out.stderr)
        );

        let rendered = RenderedImage::load_png(&rendered_path).expect("load rendered gallery PNG");
        let baseline_path = root.join("tests/baselines").join(format!("{name}.png"));
        assert!(
            baseline_path.is_file(),
            "missing gallery baseline at {}",
            baseline_path.display()
        );
        let baseline = RenderedImage::load_png(&baseline_path).expect("load gallery baseline PNG");
        let cmp = rendered
            .compare_with_baseline(&baseline)
            .expect("compare gallery render with baseline");
        if !cmp.passed {
            let diff_path = diff_dir.join(format!("{name}.png"));
            let _ = cmp.diff_image.save_png(&diff_path);
            failures.push(format!(
                "gallery scene `{name}` differed from baseline in {} / {} pixels ({:.3}%, max channel diff {}), diff written to {}",
                cmp.differing_pixels,
                cmp.total_pixels,
                cmp.diff_fraction * 100.0,
                cmp.max_channel_diff,
                diff_path.display()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "gallery baseline mismatches:\n{}",
        failures.join("\n")
    );

    // Also extract the JSON example from `.agents/skills/author-scene/SKILL.md` and verify it
    // against `tetraglass validate` and `tetraglass render`.
    let skill_path = root.join(".agents/skills/author-scene/SKILL.md");
    let skill_md = std::fs::read_to_string(&skill_path).expect("read author-scene/SKILL.md");
    let json_start = skill_md
        .find("```json\n")
        .expect("find ```json block in author-scene/SKILL.md")
        + "```json\n".len();
    let json_end = skill_md[json_start..]
        .find("\n```")
        .expect("find closing ``` in author-scene/SKILL.md")
        + json_start;
    let skill_scene_json = &skill_md[json_start..json_end];

    // Write the extracted scene JSON at the workspace root (or resolve relative to root).
    let skill_scene_path = root.join("target/author_scene_skill_example_check.json");
    // Since `target/` is 1 directory below `root`, adjust `"testdata/..."` to `"../testdata/..."` or run from root by writing a temp file in `root/target/render-tests` with absolute or root-relative path:
    let parsed_skill: serde_json::Value =
        serde_json::from_str(skill_scene_json).expect("parse author-scene JSON block");
    let rel_file = parsed_skill["sources"]["mesh"]["file"]
        .as_str()
        .expect("sources.mesh.file in author-scene example");
    assert!(
        root.join(rel_file).is_file(),
        "author-scene example references `{rel_file}`, which does not exist under workspace root"
    );

    let mut adjusted_skill = parsed_skill.clone();
    adjusted_skill["sources"]["mesh"]["file"] =
        serde_json::json!(root.join(rel_file).display().to_string());
    std::fs::write(
        &skill_scene_path,
        serde_json::to_string_pretty(&adjusted_skill).unwrap(),
    )
    .unwrap();

    let skill_val = tetraglass_bin()
        .arg("validate")
        .arg(&skill_scene_path)
        .output()
        .expect("validate author-scene example");
    assert!(
        skill_val.status.success(),
        "author-scene SKILL.md example failed validation: {}",
        String::from_utf8_lossy(&skill_val.stderr)
    );

    let skill_png = out_dir.join("author_scene_skill_example.png");
    let skill_render = tetraglass_bin()
        .arg("render")
        .arg(&skill_scene_path)
        .arg("-o")
        .arg(&skill_png)
        .output()
        .expect("render author-scene example");
    let _ = std::fs::remove_file(&skill_scene_path);
    assert!(
        skill_render.status.success(),
        "author-scene SKILL.md example failed to render: {}",
        String::from_utf8_lossy(&skill_render.stderr)
    );
    let skill_img = RenderedImage::load_png(&skill_png).expect("load author-scene rendered PNG");
    assert_eq!(skill_img.width(), 800);
    assert_eq!(skill_img.height(), 600);
}

#[test]
fn mdbook_guide_and_gallery_cover_all_filters_and_chapters_exist() {
    let root = workspace_root();
    let book_dir = root.join("docs/book");
    assert!(
        book_dir.join("book.toml").is_file(),
        "missing docs/book/book.toml"
    );

    let src_dir = book_dir.join("src");
    let summary =
        std::fs::read_to_string(src_dir.join("SUMMARY.md")).expect("read docs/book/src/SUMMARY.md");

    let required_chapters = [
        "introduction.md",
        "getting-started.md",
        "embedding.md",
        "scene-format.md",
        "filters.md",
        "gallery.md",
    ];
    for chapter in required_chapters {
        assert!(
            summary.contains(chapter),
            "SUMMARY.md must link to `{chapter}`"
        );
        let path = src_dir.join(chapter);
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        assert!(
            content.lines().count() >= 25,
            "chapter `{chapter}` is unexpectedly short ({} lines)",
            content.lines().count()
        );
    }

    let gallery_readme =
        std::fs::read_to_string(root.join("gallery/README.md")).expect("read gallery/README.md");
    let gallery_book =
        std::fs::read_to_string(src_dir.join("gallery.md")).expect("read docs/book/src/gallery.md");

    let mut seen_filter_types = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir(root.join("gallery")).expect("read gallery/") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let filename = path.file_name().unwrap().to_str().unwrap();
        assert!(
            gallery_readme.contains(filename),
            "gallery/README.md is missing `{filename}`"
        );
        assert!(
            gallery_book.contains(filename),
            "docs/book/src/gallery.md is missing `{filename}`"
        );
        let raw = std::fs::read_to_string(&path).expect("read gallery json");
        let val: serde_json::Value = serde_json::from_str(&raw).expect("parse gallery json");
        if let Some(filters) = val.get("filters").and_then(|f| f.as_object()) {
            for (_, fdef) in filters {
                if let Some(ty) = fdef.get("type").and_then(|t| t.as_str()) {
                    seen_filter_types.insert(ty.to_owned());
                }
            }
        }
    }

    for expected_filter in ["surface", "slice", "contour", "clip", "threshold"] {
        assert!(
            seen_filter_types.contains(expected_filter),
            "gallery/ scenes do not cover filter type `{expected_filter}`"
        );
    }
}

#[test]
fn export_command_writes_single_glb_and_time_series_manifest() {
    let root = workspace_root();
    let out_dir = root.join("target/cli-export-tests");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();

    // 1. Export single-step scene `gallery/slice.json` to a single `.glb` file.
    let single_glb = out_dir.join("slice.glb");
    let slice_out = tetraglass_bin()
        .arg("export")
        .arg(root.join("gallery/slice.json"))
        .arg("-o")
        .arg(&single_glb)
        .output()
        .expect("run `tetraglass export` on slice.json");
    assert!(
        slice_out.status.success(),
        "`tetraglass export` failed: {}",
        String::from_utf8_lossy(&slice_out.stderr)
    );
    let summary: serde_json::Value =
        serde_json::from_slice(&slice_out.stdout).expect("parse export summary");
    assert_eq!(summary["format"], "gltf");
    assert_eq!(summary["num_steps"], 1);
    assert!(single_glb.is_file());
    let glb_bytes = std::fs::read(&single_glb).expect("read slice.glb");
    assert_eq!(&glb_bytes[0..4], b"glTF");

    // 2. Export multi-step time series `gallery/series-transient-last.json` with `--all-steps`.
    let series_dir = out_dir.join("series_glb");
    let series_out = tetraglass_bin()
        .arg("export")
        .arg(root.join("gallery/series-transient-last.json"))
        .arg("-o")
        .arg(&series_dir)
        .arg("--all-steps")
        .output()
        .expect("run `tetraglass export --all-steps`");
    assert!(
        series_out.status.success(),
        "`tetraglass export --all-steps` failed: {}",
        String::from_utf8_lossy(&series_out.stderr)
    );
    let series_summary: serde_json::Value =
        serde_json::from_slice(&series_out.stdout).expect("parse series export summary");
    assert_eq!(series_summary["num_steps"], 3);
    assert_eq!(series_summary["files"].as_array().unwrap().len(), 3);
    let manifest_path = series_summary["manifest"].as_str().expect("manifest path");
    assert!(std::path::Path::new(manifest_path).is_file());

    // 3. Export `gallery/slice.json` and `gallery/series-transient-last.json` in `--format bundle` mode (`M3.5`).
    let bundle_dir = out_dir.join("slice_bundle");
    let bundle_out = tetraglass_bin()
        .arg("export")
        .arg(root.join("gallery/slice.json"))
        .arg("-o")
        .arg(&bundle_dir)
        .arg("--format")
        .arg("bundle")
        .output()
        .expect("run `tetraglass export --format bundle`");
    assert!(
        bundle_out.status.success(),
        "`tetraglass export --format bundle` failed: {}",
        String::from_utf8_lossy(&bundle_out.stderr)
    );
    let bundle_summary: serde_json::Value =
        serde_json::from_slice(&bundle_out.stdout).expect("parse bundle export summary");
    assert_eq!(bundle_summary["format"], "bundle");
    assert_eq!(bundle_summary["num_steps"], 1);
    assert!(bundle_dir.join("manifest.json").is_file());
    assert_eq!(bundle_summary["files"].as_array().unwrap().len(), 2); // `slice.json` has 2 representations (`hull` + `cut`)

    let series_bundle_dir = out_dir.join("series_bundle");
    let series_bundle_out = tetraglass_bin()
        .arg("export")
        .arg(root.join("gallery/series-transient-last.json"))
        .arg("-o")
        .arg(&series_bundle_dir)
        .arg("--format")
        .arg("bundle")
        .arg("--all-steps")
        .output()
        .expect("run `tetraglass export --format bundle --all-steps`");
    assert!(
        series_bundle_out.status.success(),
        "`tetraglass export --format bundle --all-steps` failed: {}",
        String::from_utf8_lossy(&series_bundle_out.stderr)
    );
    let series_bundle_summary: serde_json::Value =
        serde_json::from_slice(&series_bundle_out.stdout)
            .expect("parse series bundle export summary");
    assert_eq!(series_bundle_summary["format"], "bundle");
    assert_eq!(series_bundle_summary["num_steps"], 3);
    assert!(series_bundle_dir.join("manifest.json").is_file());
    assert_eq!(series_bundle_summary["files"].as_array().unwrap().len(), 3);
}

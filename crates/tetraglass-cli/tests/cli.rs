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

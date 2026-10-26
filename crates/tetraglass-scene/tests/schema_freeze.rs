//! Schema v1 freeze and draft-to-v1 migration integration tests (`M4.6`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::Value;
use tetraglass_io::read_vtu_from_path;
use tetraglass_scene::{
    ColormapName, Filter, FramesConfig, GlyphShapeSpec, RepresentationStyle, SCENE_VERSION, Scene,
    SceneError, SceneEvaluator, StreamlineDirectionSpec, StreamlineIntegratorSpec,
    StreamlineLineSeedSpec, StreamlineSeedsSpec, TimeSelector, migrate_scene_value,
    scene_json_schema_string,
};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn frozen_v1_schema_matches_snapshot_and_declares_all_v1_capabilities() {
    let generated = scene_json_schema_string().expect("generate v1 JSON Schema");
    let snapshot_path =
        workspace_root().join("crates/tetraglass-cli/tests/snapshots/scene-v1.schema.json");
    let committed = std::fs::read_to_string(&snapshot_path).expect("read committed schema");
    assert_eq!(
        generated.trim(),
        committed.trim(),
        "frozen v1 JSON Schema drifted from crates/tetraglass-cli/tests/snapshots/scene-v1.schema.json"
    );

    let schema_val: Value = serde_json::from_str(&generated).expect("parse schema JSON");
    let props = schema_val["properties"]
        .as_object()
        .expect("root properties");
    for required_top in ["version", "sources", "filters", "view", "output"] {
        assert!(
            props.contains_key(required_top),
            "v1 schema missing top-level property `{required_top}`"
        );
    }

    // All 8 v1 filter variants and 7 representation styles must appear in the schema definitions.
    for token in [
        "\"surface\"",
        "\"slice\"",
        "\"contour\"",
        "\"clip\"",
        "\"threshold\"",
        "\"streamlines\"",
        "\"glyphs\"",
        "\"resample\"",
        "\"surface_with_edges\"",
        "\"wireframe\"",
        "\"points\"",
        "\"lines\"",
        "\"tubes\"",
        "\"volume\"",
        "\"transfer_function\"",
        "\"frames\"",
        "\"cycles\"",
    ] {
        assert!(
            generated.contains(token),
            "frozen v1 schema must contain {token}"
        );
    }
}

#[test]
fn all_gallery_scenes_are_canonical_v1_and_roundtrip_losslessly() {
    let gallery_dir = workspace_root().join("gallery");
    let mut count = 0usize;

    for entry in std::fs::read_dir(&gallery_dir).expect("read gallery/") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        count += 1;
        let raw = std::fs::read_to_string(&path).expect("read gallery scene");

        // 1. Canonical v1 scenes must require zero migration modifications.
        let mut val: Value = serde_json::from_str(&raw).expect("parse gallery JSON");
        let report = migrate_scene_value(&mut val).expect("migrate gallery scene");
        assert_eq!(
            report.from_version,
            Some(SCENE_VERSION),
            "gallery scene {} must have version {}",
            path.display(),
            SCENE_VERSION
        );
        assert!(
            !report.was_modified(),
            "gallery scene {} had unexpected draft fields: {:?}",
            path.display(),
            report.changes
        );

        // 2. Strict v1 parse + structural validation + lossless JSON round-trip.
        let scene = Scene::from_json_str(&raw).expect("strict v1 parse");
        assert_eq!(scene.version, SCENE_VERSION);
        scene
            .validate_structure()
            .unwrap_or_else(|e| panic!("gallery scene {} failed validation: {e}", path.display()));

        let serialized = scene.to_json_string_pretty().expect("serialize v1 scene");
        let reparsed = Scene::from_json_str(&serialized).expect("reparse serialized v1 scene");
        assert_eq!(
            reparsed,
            scene,
            "gallery scene {} did not round-trip losslessly",
            path.display()
        );
    }

    assert!(
        count >= 12,
        "expected at least 12 gallery scenes, found {count}"
    );
}

#[test]
fn migrates_m1_through_m4_draft_scenes_to_canonical_v1_and_evaluates() {
    // Draft scene combining pre-freeze M1, M2, and M4 draft conventions:
    // - version 0 (draft)
    // - contour with `array` and single `value`
    // - threshold with `array` and 2-element `range`
    // - streamlines with `vector`
    // - glyphs with `field`
    // - resample written as `"type": "volume"` with `"dims"`
    // - view.show entries using `"dataset"` and `"field"`
    // - output.frames using `"start"` and `"end"`
    let draft_json = r#"{
        "version": 0,
        "sources": {
            "flow": { "file": "testdata/vtk/hex_linear.vtu" }
        },
        "filters": {
            "iso": {
                "type": "contour",
                "input": "flow",
                "array": "linear",
                "value": 3.0
            },
            "band": {
                "type": "threshold",
                "input": "flow",
                "array": "linear",
                "range": [1.0, 5.0]
            },
            "traces": {
                "type": "streamlines",
                "input": "flow",
                "vector": "velocity",
                "seeds": {
                    "line": {
                        "from": [0.2, 0.2, 0.5],
                        "to": [0.2, 0.8, 0.5],
                        "count": 4
                    }
                }
            },
            "arrows": {
                "type": "glyphs",
                "input": "flow",
                "field": "velocity",
                "shape": "arrow",
                "every": 2
            },
            "vol": {
                "type": "volume",
                "input": "flow",
                "dims": [8, 8, 8]
            }
        },
        "view": {
            "show": [
                {
                    "dataset": "iso",
                    "field": "linear",
                    "colormap": "viridis"
                },
                {
                    "dataset": "vol",
                    "style": "volume",
                    "field": "velocity:magnitude",
                    "transfer_function": [[0.0, 0.0], [1.0, 0.8]]
                }
            ]
        },
        "output": {
            "image": "out.png",
            "frames": {
                "start": "first",
                "end": "last",
                "pattern": "frame_%04d.png"
            }
        }
    }"#;

    // Strict v1 parser must reject the unmigrated draft JSON.
    assert!(
        Scene::from_json_str(draft_json).is_err(),
        "strict v1 parser must reject unmigrated draft keys"
    );

    let (scene, report) =
        Scene::from_draft_json_str(draft_json).expect("migrate M1-M4 draft scene to v1");
    assert_eq!(report.from_version, Some(0));
    assert_eq!(report.to_version, SCENE_VERSION);
    assert!(report.was_modified());
    assert_eq!(scene.version, SCENE_VERSION);

    assert_eq!(
        scene.filters["iso"],
        Filter::contour("flow", "linear", vec![3.0])
    );
    assert_eq!(
        scene.filters["band"],
        Filter::threshold("flow", "linear", 1.0, 5.0)
    );
    assert_eq!(
        scene.filters["traces"],
        Filter::Streamlines {
            input: "flow".to_string(),
            vectors: "velocity".to_string(),
            seeds: StreamlineSeedsSpec::Line(StreamlineLineSeedSpec {
                from: [0.2, 0.2, 0.5],
                to: [0.2, 0.8, 0.5],
                count: 4,
            }),
            max_length: None,
            step_size: None,
            direction: StreamlineDirectionSpec::Forward,
            integrator: StreamlineIntegratorSpec::Rk45,
            tube_radius: None,
        }
    );
    assert_eq!(scene.filters["arrows"].input(), "flow");
    if let Filter::Glyphs {
        vectors,
        shape,
        every,
        ..
    } = &scene.filters["arrows"]
    {
        assert_eq!(vectors, "velocity");
        assert_eq!(*shape, GlyphShapeSpec::Arrow);
        assert_eq!(*every, 2);
    } else {
        panic!("expected Filter::Glyphs");
    }
    assert_eq!(scene.filters["vol"], Filter::resample("flow", [8, 8, 8]));

    assert_eq!(scene.view.show[0].input, "iso");
    assert_eq!(scene.view.show[0].color_by.as_deref(), Some("linear"));
    assert_eq!(scene.view.show[0].colormap, ColormapName::Viridis);
    assert_eq!(scene.view.show[1].input, "vol");
    assert_eq!(scene.view.show[1].style, RepresentationStyle::Volume);
    assert_eq!(
        scene.view.show[1].color_by.as_deref(),
        Some("velocity:magnitude")
    );

    let frames = scene
        .output
        .as_ref()
        .and_then(|o| o.frames.as_ref())
        .expect("migrated output.frames");
    assert_eq!(
        *frames,
        FramesConfig::new(TimeSelector::FIRST, TimeSelector::LAST, "frame_%04d.png")
    );

    // Validate and evaluate the migrated scene against `testdata/vtk/hex_linear.vtu`.
    let grid = read_vtu_from_path(workspace_root().join("testdata/vtk/hex_linear.vtu"))
        .expect("load hex_linear.vtu");
    let mut sources = BTreeMap::new();
    sources.insert("flow".to_string(), grid.clone());
    scene
        .validate(&sources)
        .expect("migrated v1 scene must pass semantic validation");

    let mut evaluator = SceneEvaluator::from_sources(sources);
    let evaluated = evaluator
        .evaluate_scene(&scene)
        .expect("migrated v1 scene must evaluate");
    assert_eq!(evaluated.representations.len(), 2);
    assert!(evaluated.representations[1].volume.is_some());
}

#[test]
fn migrates_versionless_minimal_draft_and_rejects_future_versions() {
    let versionless_draft = r#"{
        "sources": {
            "mesh": { "file": "testdata/vtk/mixed_linear.vtu" }
        },
        "view": {
            "show": [
                { "dataset": "mesh", "field": "linear" }
            ]
        }
    }"#;

    let (scene, report) =
        Scene::from_draft_json_str(versionless_draft).expect("migrate versionless draft");
    assert_eq!(report.from_version, None);
    assert_eq!(report.to_version, SCENE_VERSION);
    assert_eq!(scene.version, SCENE_VERSION);
    assert_eq!(scene.view.show[0].input, "mesh");
    assert_eq!(scene.view.show[0].color_by.as_deref(), Some("linear"));

    // Future schema versions must be rejected with a clear validation issue on `version`.
    let future_json = r#"{
        "version": 2,
        "sources": {
            "mesh": { "file": "testdata/vtk/mixed_linear.vtu" }
        },
        "view": {
            "show": [
                { "input": "mesh" }
            ]
        }
    }"#;
    let err = Scene::from_draft_json_str(future_json).expect_err("version 2 must be rejected");
    match err {
        SceneError::Validation(report) => {
            assert_eq!(report.issues()[0].path, "version");
            assert!(
                report.issues()[0]
                    .message
                    .contains("unsupported scene version 2")
            );
        }
        other => panic!("expected SceneError::Validation, got {other:?}"),
    }
}

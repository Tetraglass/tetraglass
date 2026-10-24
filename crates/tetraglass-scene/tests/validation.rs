//! Integration tests for semantic scene validation (`M1.2`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use tetraglass_core::{CellType, ComponentCount, DataArray, UnstructuredGrid};
use tetraglass_io::read_vtu_from_path;
use tetraglass_scene::{
    DatasetMetadata, Scene, ValidationIssue, damerau_levenshtein, validate_scene,
    validate_scene_structure,
};

fn testdata_vtu(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/vtk")
        .join(name)
}

fn flow_grid_with_pressure_and_velocity() -> UnstructuredGrid {
    let mut grid = UnstructuredGrid::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        vec![CellType::Tetra],
        vec![4],
        vec![0, 1, 2, 3],
        None,
    )
    .unwrap();
    grid.insert_point_data(
        "pressure",
        DataArray::from_f64(
            ComponentCount::Scalar,
            vec![101_325.0, 101_400.0, 101_500.0, 101_600.0],
        )
        .unwrap(),
    )
    .unwrap();
    grid.insert_point_data(
        "velocity",
        DataArray::from_f64(
            ComponentCount::Vector3,
            vec![
                1.0, 0.0, 0.0, // p0
                0.0, 2.0, 0.0, // p1
                0.0, 0.0, 3.0, // p2
                1.0, 2.0, 2.0, // p3
            ],
        )
        .unwrap(),
    )
    .unwrap();
    grid.insert_cell_data(
        "cell_id",
        DataArray::from_i32(ComponentCount::Scalar, vec![0]).unwrap(),
    )
    .unwrap();
    grid
}

#[test]
fn valid_scene_passes_with_real_vtu_dataset() {
    let grid = read_vtu_from_path(testdata_vtu("mixed_linear.vtu")).expect("load mixed_linear.vtu");
    let mut sources = BTreeMap::new();
    sources.insert("mesh".to_string(), grid);

    let scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": {
                "mesh": { "file": "testdata/vtk/mixed_linear.vtu" }
            },
            "filters": {
                "hull": { "type": "surface", "input": "mesh" }
            },
            "view": {
                "show": [
                    { "input": "hull", "color_by": "velocity:magnitude", "colorbar": true },
                    { "input": "mesh", "color_by": "cell:cell_id", "style": "surface_with_edges" }
                ],
                "camera": { "fit": "hull", "azimuth": 30, "elevation": 20 }
            },
            "output": { "image": "out.png", "size": [1200, 800] }
        }"#,
    )
    .unwrap();

    assert_eq!(scene.validate(&sources), Ok(()));
}

#[test]
fn reports_exact_json_path_and_edit_distance_suggestion_for_array_typos() {
    let mut sources = BTreeMap::new();
    sources.insert("flow".to_string(), flow_grid_with_pressure_and_velocity());

    let scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": { "flow": { "file": "results.pvd" } },
            "view": {
                "show": [
                    { "input": "flow", "color_by": "presure" }
                ]
            }
        }"#,
    )
    .unwrap();

    let report = validate_scene(&scene, &sources).expect_err("typo must fail validation");
    assert_eq!(
        report.issues(),
        &[ValidationIssue {
            path: "view.show[0].color_by".to_string(),
            message: "no array \"presure\" on \"flow\"".to_string(),
            suggestion: Some("did you mean \"pressure\"?".to_string()),
        }]
    );
    assert_eq!(
        report.to_string(),
        "view.show[0].color_by: no array \"presure\" on \"flow\"; did you mean \"pressure\"?"
    );
}

#[test]
fn detects_wrong_association_prefix_and_invalid_components() {
    let mut sources = BTreeMap::new();
    sources.insert("flow".to_string(), flow_grid_with_pressure_and_velocity());

    let scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": { "flow": { "file": "results.pvd" } },
            "view": {
                "show": [
                    { "input": "flow", "color_by": "point:cell_id" },
                    { "input": "flow", "color_by": "cell:pressure" },
                    { "input": "flow", "color_by": "pressure:x" },
                    { "input": "flow", "color_by": "velocity:mag" },
                    { "input": "flow", "color_by": "velocity:5" }
                ]
            }
        }"#,
    )
    .unwrap();

    let report = scene
        .validate(&sources)
        .expect_err("must report array issues");
    let issues = report.issues();
    assert_eq!(issues.len(), 5);

    assert_eq!(issues[0].path, "view.show[0].color_by");
    assert_eq!(
        issues[0].suggestion.as_deref(),
        Some("did you mean \"cell:cell_id\"?")
    );

    assert_eq!(issues[1].path, "view.show[1].color_by");
    assert_eq!(
        issues[1].suggestion.as_deref(),
        Some("did you mean \"point:pressure\"?")
    );

    assert_eq!(issues[2].path, "view.show[2].color_by");
    assert!(issues[2].message.contains("component \"x\" is not valid"));
    assert!(
        issues[2]
            .suggestion
            .as_deref()
            .unwrap()
            .contains("use \"pressure\" without \":x\"")
    );

    assert_eq!(issues[3].path, "view.show[3].color_by");
    assert_eq!(
        issues[3].suggestion.as_deref(),
        Some("did you mean \"velocity:magnitude\"?")
    );

    assert_eq!(issues[4].path, "view.show[4].color_by");
    assert!(issues[4].message.contains("component \"5\" is not valid"));
}

#[test]
fn detects_unknown_datasets_and_filter_cycles() {
    let scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": {
                "flow": { "file": "results.vtu" }
            },
            "filters": {
                "cut": { "type": "surface", "input": "flow" },
                "iso": { "type": "surface", "input": "flw" },
                "cycle_a": { "type": "surface", "input": "cycle_b" },
                "cycle_b": { "type": "surface", "input": "cycle_a" }
            },
            "view": {
                "show": [
                    { "input": "missing_ds" }
                ]
            }
        }"#,
    )
    .unwrap();

    let report = validate_scene_structure(&scene).expect_err("must report filter/dataset errors");
    let paths: Vec<&str> = report.issues().iter().map(|i| i.path.as_str()).collect();
    assert!(paths.contains(&"filters.iso.input"));
    assert!(paths.contains(&"filters.cycle_a.input"));
    assert!(paths.contains(&"view.show[0].input"));

    let iso_issue = report
        .issues()
        .iter()
        .find(|i| i.path == "filters.iso.input")
        .unwrap();
    assert_eq!(iso_issue.message, "unknown dataset \"flw\"");
    assert!(
        iso_issue
            .suggestion
            .as_deref()
            .unwrap()
            .contains("did you mean \"flow\"?")
    );

    let cycle_issue = report
        .issues()
        .iter()
        .find(|i| i.path == "filters.cycle_a.input")
        .unwrap();
    assert_eq!(
        cycle_issue.message,
        "filter cycle detected: cycle_a -> cycle_b -> cycle_a"
    );
}

#[test]
fn validates_naming_colors_camera_log_scale_and_output() {
    let grid = read_vtu_from_path(testdata_vtu("mixed_linear.vtu")).expect("load mixed_linear.vtu");
    let mut meta_map = BTreeMap::new();
    meta_map.insert("mesh".to_string(), DatasetMetadata::from_grid(&grid));

    let scene = Scene::from_json_str(
        r#"{
            "version": 2,
            "sources": {
                "mesh": { "file": "testdata/vtk/mixed_linear.vtu", "linearize_level": 0 },
                "BadName": { "file": "unknown_ext.xyz" }
            },
            "view": {
                "show": [
                    {
                        "input": "mesh",
                        "color_by": "linear",
                        "log_scale": true,
                        "color": "lightgry",
                        "edge_color": [255.0, 0.0, 0.0],
                        "opacity": 1.5,
                        "point_size": -1.0
                    },
                    {
                        "input": "mesh",
                        "range": [50.0, 10.0]
                    }
                ],
                "camera": {
                    "fit": "msh",
                    "zoom": 0.0,
                    "view_angle": 185.0,
                    "view_up": [0.0, 0.0, 0.0]
                },
                "background": "whte"
            },
            "output": {
                "image": "  ",
                "size": [0, 800],
                "msaa": 8,
                "cycles": {
                    "samples": 0,
                    "device": ""
                }
            }
        }"#,
    )
    .unwrap();

    let report = scene
        .validate_with_metadata(&meta_map)
        .expect_err("must catch multiple semantic errors");
    let by_path: BTreeMap<&str, &ValidationIssue> = report
        .issues()
        .iter()
        .map(|i| (i.path.as_str(), i))
        .collect();

    assert!(by_path.contains_key("version"));
    assert!(by_path.contains_key("sources.mesh.linearize_level"));
    assert_eq!(
        by_path["sources.BadName"].suggestion.as_deref(),
        Some("did you mean \"bad_name\"?")
    );
    assert!(by_path.contains_key("sources.BadName.file"));
    assert_eq!(
        by_path["view.show[0].color"].suggestion.as_deref(),
        Some("did you mean \"lightgray\"?")
    );
    assert!(by_path.contains_key("view.show[0].edge_color"));
    assert!(by_path.contains_key("view.show[0].opacity"));
    assert!(by_path.contains_key("view.show[0].point_size"));
    // `linear` on mixed_linear.vtu has min 0.0, so log_scale with auto range must be rejected!
    assert!(by_path.contains_key("view.show[0].log_scale"));
    assert_eq!(
        by_path["view.show[1].range"].suggestion.as_deref(),
        Some("did you mean [10, 50]?")
    );
    assert!(
        by_path["view.camera.fit"]
            .suggestion
            .as_deref()
            .unwrap()
            .contains("did you mean \"mesh\"?")
    );
    assert!(by_path.contains_key("view.camera.zoom"));
    assert!(by_path.contains_key("view.camera.view_angle"));
    assert_eq!(
        by_path["view.camera.view_up"].message,
        "must not be [0, 0, 0]"
    );
    assert_eq!(
        by_path["view.background"].suggestion.as_deref(),
        Some("did you mean \"white\"?")
    );
    assert!(by_path.contains_key("output.image"));
    assert!(by_path.contains_key("output.size"));
    assert!(by_path.contains_key("output.msaa"));
    assert!(by_path.contains_key("output.cycles.samples"));
    assert!(by_path.contains_key("output.cycles.device"));
}

#[test]
fn damerau_levenshtein_handles_transpositions_and_edits() {
    assert_eq!(damerau_levenshtein("pressure", "pressure"), 0);
    assert_eq!(damerau_levenshtein("presure", "pressure"), 1);
    assert_eq!(damerau_levenshtein("rpessure", "pressure"), 1);
    assert_eq!(damerau_levenshtein("flw", "flow"), 1);
}

#[test]
fn validates_slice_filter_normal_origin_and_volume_input() {
    let scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": {
                "flow": { "file": "results.vtu" }
            },
            "filters": {
                "hull": { "type": "surface", "input": "flow" },
                "cut": { "type": "slice", "input": "flow", "normal": [0.0, 0.0, 0.0] },
                "cut_on_surface": { "type": "slice", "input": "hull", "normal": [0.0, 0.0, 1.0] }
            },
            "view": {
                "show": [
                    { "input": "cut" },
                    { "input": "cut_on_surface" }
                ]
            }
        }"#,
    )
    .unwrap();

    let report = validate_scene_structure(&scene).expect_err("must report slice validation errors");
    let by_path: BTreeMap<&str, &ValidationIssue> = report
        .issues()
        .iter()
        .map(|i| (i.path.as_str(), i))
        .collect();

    assert_eq!(
        by_path["filters.cut.normal"].message,
        "must not be [0, 0, 0]"
    );
    assert!(
        by_path["filters.cut.normal"]
            .to_string()
            .starts_with("filters.cut.normal: must not be [0, 0, 0]")
    );
    assert!(
        by_path["filters.cut_on_surface.input"]
            .message
            .contains("requires a volume dataset input")
    );
    assert_eq!(
        by_path["filters.cut_on_surface.input"]
            .suggestion
            .as_deref(),
        Some("use volume source \"flow\" as input instead")
    );
}

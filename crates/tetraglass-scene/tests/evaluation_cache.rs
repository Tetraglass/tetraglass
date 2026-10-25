//! Integration tests for lazy scene pipeline evaluation and input-hash caching (`M1.3`).

use std::path::PathBuf;
use std::sync::Arc;

use tetraglass_io::{read_pvd_series_from_path, read_vtu_from_path};
use tetraglass_scene::{
    ArrayAssociation, BackgroundSpec, ColorSpec, ColormapName, LightingPreset, RangePreset,
    RangeSpec, ResolvedComponent, Scene, SceneEvaluator, TimeSelector, VectorAxis,
    hash_unstructured_grid,
};

fn testdata_vtu(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/vtk")
        .join(name)
}

#[test]
fn colormap_camera_and_view_changes_never_rerun_filters() {
    let grid = read_vtu_from_path(testdata_vtu("mixed_linear.vtu")).expect("load mixed_linear.vtu");
    let mut evaluator = SceneEvaluator::new();
    evaluator.insert_source("mesh", grid);

    let mut scene = Scene::from_json_str(
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
                    {
                        "input": "hull",
                        "color_by": "linear",
                        "colormap": "viridis",
                        "colorbar": true
                    }
                ],
                "camera": { "fit": "all", "azimuth": 30, "elevation": 20 }
            },
            "output": { "image": "out.png", "size": [1200, 800] }
        }"#,
    )
    .unwrap();

    // First evaluation runs the surface filter once.
    let eval1 = evaluator.evaluate_scene(&scene).expect("initial evaluate");
    assert_eq!(evaluator.filter_executions(), 1);
    assert_eq!(evaluator.cache_hits(), 0);
    assert_eq!(eval1.representations.len(), 1);
    let initial_surface = Arc::clone(&eval1.representations[0].surface);
    assert_eq!(initial_surface.num_faces(), 12);
    assert_eq!(initial_surface.num_triangles(), 18);

    let coloring1 = eval1.representations[0]
        .coloring
        .as_ref()
        .expect("colored by linear");
    assert_eq!(coloring1.association, ArrayAssociation::Point);
    assert_eq!(coloring1.array_name, "linear");
    assert_eq!(coloring1.component, ResolvedComponent::Component(0));
    assert_eq!(coloring1.colormap, ColormapName::Viridis);

    // 1. Change colormap, range, and center -> must NOT re-run the filter.
    scene.view.show[0].colormap = ColormapName::CoolWarm;
    scene.view.show[0].range = RangeSpec::bounds(-5.0, 25.0);
    scene.view.show[0].center = Some(10.0);

    let eval2 = evaluator
        .evaluate_scene(&scene)
        .expect("re-evaluate after colormap/range change");
    assert_eq!(evaluator.filter_executions(), 1);
    assert_eq!(evaluator.cache_hits(), 1);
    assert!(Arc::ptr_eq(
        &initial_surface,
        &eval2.representations[0].surface
    ));
    let coloring2 = eval2.representations[0].coloring.as_ref().unwrap();
    assert_eq!(coloring2.colormap, ColormapName::CoolWarm);
    assert_eq!(coloring2.effective_range, [-5.0, 25.0]);
    assert_eq!(coloring2.center, Some(10.0));

    // 2. Change color_by to a vector component ("velocity:y") and then a cell array ("cell_id") -> must NOT re-run the filter.
    scene.view.show[0].color_by = Some("velocity:y".to_string());
    let eval3 = evaluator
        .evaluate_scene(&scene)
        .expect("re-evaluate after color_by vector component change");
    assert_eq!(evaluator.filter_executions(), 1);
    assert_eq!(evaluator.cache_hits(), 2);
    let coloring3 = eval3.representations[0].coloring.as_ref().unwrap();
    assert_eq!(coloring3.association, ArrayAssociation::Point);
    assert_eq!(coloring3.array_name, "velocity");
    assert_eq!(
        coloring3.component,
        ResolvedComponent::Component(VectorAxis::Y.index())
    );

    scene.view.show[0].color_by = Some("cell_id".to_string());
    let eval4 = evaluator
        .evaluate_scene(&scene)
        .expect("re-evaluate after color_by cell array change");
    assert_eq!(evaluator.filter_executions(), 1);
    assert_eq!(evaluator.cache_hits(), 3);
    let coloring4 = eval4.representations[0].coloring.as_ref().unwrap();
    assert_eq!(coloring4.association, ArrayAssociation::Cell);
    assert_eq!(coloring4.array_name, "cell_id");

    // 3. Change camera angles, zoom, lighting, background, and output -> must NOT re-run the filter.
    scene.view.camera.azimuth = 120.0;
    scene.view.camera.elevation = -15.0;
    scene.view.camera.zoom = 1.75;
    scene.view.lights = LightingPreset::Headlight;
    scene.view.background = BackgroundSpec::Solid(ColorSpec::Named("black".to_string()));
    scene.output.as_mut().unwrap().size = [1920, 1080];

    let eval5 = evaluator
        .evaluate_scene(&scene)
        .expect("re-evaluate after camera/background/output change");
    assert_eq!(evaluator.filter_executions(), 1);
    assert_eq!(evaluator.cache_hits(), 4);
    assert!(Arc::ptr_eq(
        &initial_surface,
        &eval5.representations[0].surface
    ));
}

#[test]
fn lazy_evaluation_skips_unreferenced_filters_and_hidden_representations() {
    let mixed =
        read_vtu_from_path(testdata_vtu("mixed_linear.vtu")).expect("load mixed_linear.vtu");
    let hex = read_vtu_from_path(testdata_vtu("hex_linear.vtu")).expect("load hex_linear.vtu");

    let mut evaluator = SceneEvaluator::new();
    evaluator.insert_source("mixed", mixed);
    evaluator.insert_source("hex", hex);

    let mut scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": {
                "mixed": { "file": "testdata/vtk/mixed_linear.vtu" },
                "hex": { "file": "testdata/vtk/hex_linear.vtu" }
            },
            "filters": {
                "used_surface": { "type": "surface", "input": "mixed" },
                "unused_surface": { "type": "surface", "input": "hex" }
            },
            "view": {
                "show": [
                    { "input": "used_surface", "visible": true },
                    { "input": "unused_surface", "visible": false }
                ]
            }
        }"#,
    )
    .unwrap();

    // Only `used_surface` is visible, so `unused_surface` is never executed.
    let eval1 = evaluator
        .evaluate_scene(&scene)
        .expect("evaluate with hidden second representation");
    assert_eq!(eval1.representations.len(), 1);
    assert_eq!(evaluator.filter_executions(), 1);
    assert_eq!(evaluator.cache_hits(), 0);
    assert_eq!(evaluator.cached_surface_count(), 1);

    // Making the second representation visible evaluates `unused_surface` while hitting the cache for `used_surface`.
    scene.view.show[1].visible = true;
    let eval2 = evaluator
        .evaluate_scene(&scene)
        .expect("evaluate with both representations visible");
    assert_eq!(eval2.representations.len(), 2);
    assert_eq!(evaluator.filter_executions(), 2);
    assert_eq!(evaluator.cache_hits(), 1);
    assert_eq!(evaluator.cached_surface_count(), 2);
}

#[test]
fn deduplicates_explicit_surface_filter_and_implicit_volume_surface() {
    let grid = read_vtu_from_path(testdata_vtu("mixed_linear.vtu")).expect("load mixed_linear.vtu");
    let mut evaluator = SceneEvaluator::new();
    evaluator.insert_source("mesh", grid);

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
                    { "input": "hull", "style": "surface", "color_by": "linear" },
                    { "input": "mesh", "style": "wireframe" }
                ]
            }
        }"#,
    )
    .unwrap();

    let evaluated = evaluator.evaluate_scene(&scene).expect("evaluate scene");
    assert_eq!(evaluated.representations.len(), 2);
    assert_eq!(evaluator.filter_executions(), 1);
    assert_eq!(evaluator.cache_hits(), 1);
    assert!(Arc::ptr_eq(
        &evaluated.representations[0].surface,
        &evaluated.representations[1].surface
    ));
}

#[test]
fn content_hash_is_deterministic_and_invalidates_on_data_change() {
    let mixed_a =
        read_vtu_from_path(testdata_vtu("mixed_linear.vtu")).expect("load mixed_linear.vtu");
    let mixed_b =
        read_vtu_from_path(testdata_vtu("mixed_linear.vtu")).expect("load mixed_linear.vtu again");
    let hex = read_vtu_from_path(testdata_vtu("hex_linear.vtu")).expect("load hex_linear.vtu");

    assert_eq!(
        hash_unstructured_grid(&mixed_a),
        hash_unstructured_grid(&mixed_b)
    );
    assert_ne!(
        hash_unstructured_grid(&mixed_a),
        hash_unstructured_grid(&hex)
    );

    let scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": {
                "mesh": { "file": "testdata/vtk/mixed_linear.vtu" }
            },
            "view": {
                "show": [
                    { "input": "mesh", "color_by": "linear" }
                ]
            }
        }"#,
    )
    .unwrap();

    let mut evaluator = SceneEvaluator::new();
    evaluator.insert_source("mesh", mixed_a);
    evaluator.evaluate_scene(&scene).unwrap();
    assert_eq!(evaluator.filter_executions(), 1);

    // Replacing "mesh" with an identical fresh grid allocation still hits the cache!
    evaluator.insert_source("mesh", mixed_b);
    evaluator.evaluate_scene(&scene).unwrap();
    assert_eq!(evaluator.filter_executions(), 1);
    assert_eq!(evaluator.cache_hits(), 1);

    // Replacing "mesh" with a different grid invalidates the hash and runs extraction once.
    evaluator.insert_source("mesh", hex);
    let eval_hex = evaluator.evaluate_scene(&scene).unwrap();
    assert_eq!(evaluator.filter_executions(), 2);
    assert_eq!(eval_hex.representations[0].surface.num_faces(), 24);
}

#[test]
fn time_series_scene_evaluation_loads_steps_lazily_and_supports_auto_all_steps_range() {
    let pvd_path = testdata_vtu("series.pvd");
    let series = read_pvd_series_from_path(&pvd_path).expect("parse series.pvd");
    assert_eq!(series.load_executions(), 0);

    let mut evaluator = SceneEvaluator::new();
    evaluator.insert_series_source("series", series);

    let mut scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": {
                "series": { "file": "testdata/vtk/series.pvd" }
            },
            "view": {
                "show": [
                    {
                        "input": "series",
                        "color_by": "transient",
                        "range": "auto"
                    }
                ]
            },
            "output": {
                "image": "out.png",
                "time": "first"
            }
        }"#,
    )
    .unwrap();

    // 1. Evaluate at "first" (step 0, t = 0.0). Only step 0 should be loaded.
    let eval_first = evaluator.evaluate_scene(&scene).expect("evaluate at first");
    assert_eq!(eval_first.time_step, 0);
    assert_eq!(eval_first.time_value, 0.0);
    assert_eq!(
        evaluator.series_source("series").unwrap().load_executions(),
        1
    );
    let col_first = eval_first.representations[0].coloring.as_ref().unwrap();
    assert!(
        (col_first.effective_range[0] - 0.0).abs() < 1e-12
            && (col_first.effective_range[1] - 6.0).abs() < 1e-12
    );

    // 2. Evaluate at "last" (step 2, t = 1.0). Step 1 is skipped; only step 2 is loaded.
    scene.output.as_mut().unwrap().time = TimeSelector::LAST;
    let eval_last = evaluator.evaluate_scene(&scene).expect("evaluate at last");
    assert_eq!(eval_last.time_step, 2);
    assert_eq!(eval_last.time_value, 1.0);
    let series_ref = evaluator.series_source("series").unwrap();
    assert_eq!(series_ref.load_executions(), 2);
    assert!(series_ref.is_step_cached(0));
    assert!(!series_ref.is_step_cached(1));
    assert!(series_ref.is_step_cached(2));
    let col_last = eval_last.representations[0].coloring.as_ref().unwrap();
    assert!(
        (col_last.effective_range[0] - 1.0).abs() < 1e-12
            && (col_last.effective_range[1] - 7.0).abs() < 1e-12
    );

    // 3. Evaluate at explicit time value 0.5 (step 1) and then step index { "step": 1 } (cached).
    scene.output.as_mut().unwrap().time = TimeSelector::Value(0.5);
    let eval_mid = evaluator
        .evaluate_scene(&scene)
        .expect("evaluate at t = 0.5");
    assert_eq!(eval_mid.time_step, 1);
    assert_eq!(eval_mid.time_value, 0.5);
    assert_eq!(
        evaluator.series_source("series").unwrap().load_executions(),
        3
    );

    let eval_step1 = evaluator
        .evaluate_scene_at_time(&scene, TimeSelector::step(1))
        .expect("evaluate at step 1");
    assert_eq!(eval_step1.time_step, 1);
    assert_eq!(
        evaluator.series_source("series").unwrap().load_executions(),
        3
    );

    // 4. Evaluate with `range: "auto_all_steps"` at step 0 -> range spans [0.0, 7.0] across steps 0..=2.
    scene.output.as_mut().unwrap().time = TimeSelector::FIRST;
    scene.view.show[0].range = RangeSpec::Preset(RangePreset::AutoAllSteps);
    let eval_all_range = evaluator
        .evaluate_scene(&scene)
        .expect("evaluate with auto_all_steps");
    let col_all = eval_all_range.representations[0].coloring.as_ref().unwrap();
    assert!(
        (col_all.effective_range[0] - 0.0).abs() < 1e-12
            && (col_all.effective_range[1] - 7.0).abs() < 1e-12,
        "expected [0.0, 7.0] across all steps, got {:?}",
        col_all.effective_range
    );
}

#[test]
fn slice_filter_evaluates_on_tet_view_and_caches_results() {
    let hex = read_vtu_from_path(testdata_vtu("hex_linear.vtu")).expect("load hex_linear.vtu");
    let mut evaluator = SceneEvaluator::new();
    evaluator.insert_source("hex", hex);

    let mut scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": {
                "hex": { "file": "testdata/vtk/hex_linear.vtu" }
            },
            "filters": {
                "cut_z": { "type": "slice", "input": "hex", "origin": [1.0, 1.0, 0.75], "normal": [0.0, 0.0, 1.0] },
                "cut_center": { "type": "slice", "input": "hex", "normal": [1.0, 0.0, 0.0] }
            },
            "view": {
                "show": [
                    { "input": "cut_z", "color_by": "linear", "colorbar": true },
                    { "input": "cut_center", "color_by": "velocity:magnitude" }
                ]
            }
        }"#,
    )
    .unwrap();

    let eval1 = evaluator
        .evaluate_scene(&scene)
        .expect("evaluate slice scene");
    assert_eq!(eval1.representations.len(), 2);
    assert_eq!(evaluator.filter_executions(), 2);
    assert_eq!(evaluator.cache_hits(), 0);

    // 2x2x2 hex block on [0, 2]^3 has cross-section area 4.0 on both planes.
    assert!((eval1.representations[0].surface.total_area() - 4.0).abs() < 1e-12);
    assert!((eval1.representations[1].surface.total_area() - 4.0).abs() < 1e-12);

    // At z = 0.75 on [0, 2]^2, `linear = x + 2*y + 3*z` spans [3*0.75, 2 + 4 + 3*0.75] = [2.25, 8.25].
    let col0 = eval1.representations[0].coloring.as_ref().unwrap();
    assert!((col0.effective_range[0] - 2.25).abs() < 1e-12);
    assert!((col0.effective_range[1] - 8.25).abs() < 1e-12);

    // Changing colormap/camera hits the surface cache for both slices without re-running filters.
    scene.view.show[0].colormap = ColormapName::Turbo;
    scene.view.camera.azimuth = 45.0;
    let eval2 = evaluator
        .evaluate_scene(&scene)
        .expect("re-evaluate slice scene");
    assert_eq!(evaluator.filter_executions(), 2);
    assert_eq!(evaluator.cache_hits(), 2);
    assert!(Arc::ptr_eq(
        &eval1.representations[0].surface,
        &eval2.representations[0].surface
    ));
}

#[test]
fn contour_filter_evaluates_multiple_isovalues_and_caches_results() {
    let hex = read_vtu_from_path(testdata_vtu("hex_linear.vtu")).expect("load hex_linear.vtu");
    let mut evaluator = SceneEvaluator::new();
    evaluator.insert_source("hex", hex);

    let mut scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": {
                "hex": { "file": "testdata/vtk/hex_linear.vtu" }
            },
            "filters": {
                "iso": {
                    "type": "contour",
                    "input": "hex",
                    "field": "linear",
                    "values": [3.0, 6.0, 9.0]
                }
            },
            "view": {
                "show": [
                    { "input": "iso", "color_by": "linear", "colorbar": true }
                ]
            }
        }"#,
    )
    .unwrap();

    let eval1 = evaluator
        .evaluate_scene(&scene)
        .expect("evaluate contour scene");
    assert_eq!(eval1.representations.len(), 1);
    assert_eq!(evaluator.filter_executions(), 1);
    assert_eq!(evaluator.cache_hits(), 0);

    let iso_surf = &eval1.representations[0].surface;
    assert!(iso_surf.num_triangles() > 20);
    assert!(iso_surf.total_area() > 6.0);

    // The contoured `linear` field has exact values in {3.0, 6.0, 9.0}, so auto range is [3.0, 9.0].
    let col = eval1.representations[0].coloring.as_ref().unwrap();
    assert!((col.effective_range[0] - 3.0).abs() < 1e-11);
    assert!((col.effective_range[1] - 9.0).abs() < 1e-11);

    // Changing view-only properties hits the surface cache without re-running contour.
    scene.view.show[0].color_by = Some("velocity:magnitude".to_string());
    scene.view.camera.elevation = 35.0;
    let eval2 = evaluator
        .evaluate_scene(&scene)
        .expect("re-evaluate contour scene");
    assert_eq!(evaluator.filter_executions(), 1);
    assert_eq!(evaluator.cache_hits(), 1);
    assert!(Arc::ptr_eq(iso_surf, &eval2.representations[0].surface));
}

#[test]
fn clip_filter_evaluates_exact_and_crinkle_modes_caches_and_chains_into_slice_and_contour() {
    let hex = read_vtu_from_path(testdata_vtu("hex_linear.vtu")).expect("load hex_linear.vtu");
    let mut evaluator = SceneEvaluator::new();
    evaluator.insert_source("hex", hex);

    let mut scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": {
                "hex": { "file": "testdata/vtk/hex_linear.vtu" }
            },
            "filters": {
                "clipped": {
                    "type": "clip",
                    "input": "hex",
                    "origin": [1.0, 1.0, 1.0],
                    "normal": [1.0, 0.0, 0.0],
                    "keep": "front",
                    "mode": "exact"
                },
                "cut_of_clipped": {
                    "type": "slice",
                    "input": "clipped",
                    "origin": [1.5, 1.0, 1.0],
                    "normal": [0.0, 0.0, 1.0]
                }
            },
            "view": {
                "show": [
                    { "input": "clipped", "color_by": "linear", "colorbar": true },
                    { "input": "cut_of_clipped", "color_by": "linear" }
                ]
            }
        }"#,
    )
    .unwrap();

    let eval1 = evaluator
        .evaluate_scene(&scene)
        .expect("evaluate clip + slice scene");
    assert_eq!(eval1.representations.len(), 2);
    // `clipped` executed once (+1) and `cut_of_clipped` executed once (+1) = 2 filter executions;
    // the second reference to `clipped` inside `cut_of_clipped` hit `tet_view_cache` (+1 hit).
    assert_eq!(evaluator.filter_executions(), 2);
    assert_eq!(evaluator.cache_hits(), 1);

    // `hex` on [0, 2]^3 clipped to `x >= 1.0` is the [1, 2] x [0, 2] x [0, 2] block (volume 4.0, surface area 2*(1*2 + 1*2 + 2*2) = 16.0).
    let clipped_surf = &eval1.representations[0].surface;
    assert!((clipped_surf.total_area() - 16.0).abs() < 1e-11);

    // `cut_of_clipped` at z = 1.0 on [1, 2] x [0, 2] has area 1 * 2 = 2.0.
    let slice_surf = &eval1.representations[1].surface;
    assert!((slice_surf.total_area() - 2.0).abs() < 1e-11);

    // Changing view-only properties does not re-execute either filter.
    scene.view.show[0].colormap = ColormapName::Cividis;
    let eval2 = evaluator
        .evaluate_scene(&scene)
        .expect("re-evaluate clip + slice scene");
    assert_eq!(evaluator.filter_executions(), 2);
    assert!(Arc::ptr_eq(clipped_surf, &eval2.representations[0].surface));
}

#[test]
fn threshold_filter_evaluates_point_and_cell_arrays_caches_and_chains_into_slice() {
    let hex = read_vtu_from_path(testdata_vtu("hex_linear.vtu")).expect("load hex_linear.vtu");
    let mut evaluator = SceneEvaluator::new();
    evaluator.insert_source("hex", hex);

    let mut scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": {
                "hex": { "file": "testdata/vtk/hex_linear.vtu" }
            },
            "filters": {
                "lower_layer": {
                    "type": "threshold",
                    "input": "hex",
                    "field": "cell:cell_id",
                    "min": 0.0,
                    "max": 3.0
                },
                "cut_of_layer": {
                    "type": "slice",
                    "input": "lower_layer",
                    "origin": [1.0, 1.0, 0.5],
                    "normal": [0.0, 0.0, 1.0]
                }
            },
            "view": {
                "show": [
                    { "input": "lower_layer", "color_by": "linear", "colorbar": true },
                    { "input": "cut_of_layer", "color_by": "linear" }
                ]
            }
        }"#,
    )
    .unwrap();

    let eval1 = evaluator
        .evaluate_scene(&scene)
        .expect("evaluate threshold + slice scene");
    assert_eq!(eval1.representations.len(), 2);
    assert_eq!(evaluator.filter_executions(), 2);
    assert_eq!(evaluator.cache_hits(), 1);

    // `cell_id` in 0..=3 on hex_linear.vtu is the [0, 2] x [0, 2] x [0, 1] bottom slab (surface area 2*(2*2 + 2*1 + 2*1) = 16.0).
    let layer_surf = &eval1.representations[0].surface;
    assert!((layer_surf.total_area() - 16.0).abs() < 1e-11);

    // `cut_of_layer` at z = 0.5 on [0, 2] x [0, 2] has area 4.0.
    let slice_surf = &eval1.representations[1].surface;
    assert!((slice_surf.total_area() - 4.0).abs() < 1e-11);

    // Changing view-only properties does not re-execute either filter.
    scene.view.show[0].colormap = ColormapName::Turbo;
    let eval2 = evaluator
        .evaluate_scene(&scene)
        .expect("re-evaluate threshold + slice scene");
    assert_eq!(evaluator.filter_executions(), 2);
    assert!(Arc::ptr_eq(layer_surf, &eval2.representations[0].surface));
}

#[test]
fn streamlines_filter_evaluates_tubes_lines_and_wireframe_and_caches_results() {
    let hex = read_vtu_from_path(testdata_vtu("hex_linear.vtu")).expect("load hex_linear.vtu");
    let mut evaluator = SceneEvaluator::new();
    evaluator.insert_source("hex", hex);

    let mut scene = Scene::from_json_str(
        r#"{
            "version": 1,
            "sources": {
                "hex": { "file": "testdata/vtk/hex_linear.vtu" }
            },
            "filters": {
                "traces": {
                    "type": "streamlines",
                    "input": "hex",
                    "vectors": "velocity",
                    "seeds": {
                        "line": {
                            "from": [1.5, 0.2, 1.0],
                            "to": [1.5, 1.2, 1.0],
                            "count": 4
                        }
                    },
                    "max_length": 2.0,
                    "step_size": 0.03,
                    "direction": "both",
                    "integrator": "rk45",
                    "tube_radius": 0.04
                }
            },
            "view": {
                "show": [
                    { "input": "traces", "style": "tubes", "color_by": "arc_length" },
                    { "input": "traces", "style": "lines", "color_by": "velocity:magnitude" },
                    { "input": "traces", "style": "wireframe", "color_by": "seed_id" }
                ]
            }
        }"#,
    )
    .unwrap();

    let eval1 = evaluator
        .evaluate_scene(&scene)
        .expect("evaluate streamlines scene");
    assert_eq!(eval1.representations.len(), 3);
    // `traces` ODE integration executed once (+1), and the 2nd & 3rd representations hit `streamline_cache` (+2 hits).
    assert_eq!(evaluator.filter_executions(), 1);
    assert_eq!(evaluator.cache_hits(), 2);

    let tubes_surf = &eval1.representations[0].surface;
    let lines_surf = &eval1.representations[1].surface;
    let wire_surf = &eval1.representations[2].surface;

    // 12-sided tubes have more triangles than 6-sided lines, and both have positive area, while wireframe lines use wireframe_edges.
    assert!(tubes_surf.num_triangles() > lines_surf.num_triangles());
    assert!(tubes_surf.total_area() > lines_surf.total_area());
    assert!(lines_surf.total_area() > 0.0);
    assert_eq!(wire_surf.num_triangles(), 0);
    assert!(wire_surf.num_wireframe_edges() > 0);

    // Changing colormap/camera hits the streamline and surface caches without re-integrating.
    scene.view.show[0].colormap = ColormapName::Turbo;
    scene.view.camera.azimuth = 50.0;
    let eval2 = evaluator
        .evaluate_scene(&scene)
        .expect("re-evaluate streamlines scene");
    assert_eq!(evaluator.filter_executions(), 1);
    assert_eq!(evaluator.cache_hits(), 5);
    assert!(Arc::ptr_eq(tubes_surf, &eval2.representations[0].surface));
    assert!(Arc::ptr_eq(lines_surf, &eval2.representations[1].surface));
    assert!(Arc::ptr_eq(wire_surf, &eval2.representations[2].surface));
}

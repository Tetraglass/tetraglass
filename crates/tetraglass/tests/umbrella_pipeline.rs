//! End-to-end integration tests for the `tetraglass` umbrella crate (`M6.1`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use tetraglass::prelude::*;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn umbrella_crate_builds_validates_evaluates_exports_and_renders_scene() {
    let vtu_path = workspace_root().join("testdata/vtk/hex_linear.vtu");
    let vtu_bytes = std::fs::read(&vtu_path).expect("read hex_linear.vtu");
    let grid = tetraglass::io::read_vtu(&vtu_bytes).expect("parse hex_linear.vtu");

    // 1. Construct a multi-filter scene using the fluent Rust SceneBuilder API
    let mut builder = Scene::builder();
    let flow = builder.source("flow", Source::file("hex_linear.vtu"));
    let hull = builder.filter("hull", SurfaceFilter::new(&flow));
    let cut = builder.filter(
        "cut",
        SliceFilter::new(&flow, [0.0, 0.0, 1.0]).origin([1.0, 1.0, 1.0]),
    );
    let iso = builder.filter("iso", ContourFilter::new(&flow, "linear", [3.0]));
    let arrows = builder.filter(
        "arrows",
        GlyphsFilter::new(&flow, "velocity")
            .shape(tetraglass::scene::GlyphShapeSpec::Arrow)
            .every(4),
    );

    builder.show(&hull).wireframe().edge_color("dimgray");
    builder
        .show(&cut)
        .surface_with_edges()
        .color_by("velocity")
        .magnitude()
        .colormap(ColormapName::Viridis)
        .colorbar(true);
    builder
        .show(&iso)
        .surface()
        .color_by("linear")
        .colormap(ColormapName::Cividis)
        .opacity(0.6);
    builder
        .show(&arrows)
        .surface()
        .cell_field("cell_id")
        .colormap(ColormapName::CoolWarm);

    builder.camera().fit(&flow).orbit(35.0, 25.0).zoom(1.05);
    builder
        .lights(LightingPreset::Default)
        .background_color("white")
        .axes(true)
        .title("Umbrella Pipeline Test");
    builder.output("umbrella.png").size(400, 300).msaa(4);

    let scene = builder.build_validated().expect("structurally valid scene");

    // 2. Semantic validation against loaded dataset
    let mut sources = BTreeMap::new();
    sources.insert("flow".to_string(), grid.clone());
    scene
        .validate(&sources)
        .expect("scene should validate against hex_linear.vtu");

    // 3. Lazy evaluation and caching via SceneEvaluator
    let mut evaluator = SceneEvaluator::new();
    evaluator.insert_source("flow", grid);
    let evaluated = evaluator.evaluate_scene(&scene).expect("evaluate scene");
    assert_eq!(evaluated.representations.len(), 4);
    assert!(evaluated.representations[1].surface.num_triangles() > 0);
    assert!(evaluated.representations[2].surface.num_triangles() > 0);
    assert!(evaluated.representations[3].surface.num_triangles() > 0);

    let first_execs = evaluator.filter_executions();
    assert!(first_execs > 0);

    // Changing only camera / colormap does not re-execute filters
    let mut modified = scene.clone();
    modified.view.camera.azimuth = 60.0;
    modified.view.show[1].colormap = ColormapName::CoolWarm;
    let _ = evaluator
        .evaluate_scene(&modified)
        .expect("re-evaluate scene");
    assert_eq!(evaluator.filter_executions(), first_execs);
    assert!(evaluator.cache_hits() > 0);

    // 4. Export through re-exported `tetraglass::export`
    let glb = tetraglass::export_evaluated_scene_to_glb(&scene, &evaluated).expect("export glb");
    let parsed_glb = tetraglass::export::parse_glb(&glb).expect("parse exported glb");
    assert!(!parsed_glb.meshes.is_empty());

    let bundle =
        tetraglass::export_evaluated_scene_to_bundle(&scene, &evaluated).expect("export bundle");
    assert_eq!(bundle.manifest.time_steps.len(), 1);
    assert!(!bundle.mesh_files.is_empty());

    // 5. Render headlessly through re-exported `tetraglass::render`
    let ctx = HeadlessContext::new().expect("headless wgpu context");
    let slice_surface = &evaluated.representations[1].surface;
    let style = SurfaceStyle {
        color_by: ColorBy::point("linear"),
        colormap: ColormapKind::Viridis,
        draw_surface: true,
        draw_edges: true,
        ..SurfaceStyle::default()
    };
    let rendered = ctx
        .render_surface(
            slice_surface,
            &tetraglass::render::HeadlessRenderOptions {
                width: 320,
                height: 240,
                sample_count: 4,
                style,
                ..Default::default()
            },
        )
        .expect("headless render");
    assert_eq!(rendered.width(), 320);
    assert_eq!(rendered.height(), 240);
}

#[test]
fn umbrella_crate_reads_partitioned_pvtu_and_extracts_seamless_surface() {
    let pvtu_path = workspace_root().join("testdata/vtk/hex_partitioned.pvtu");
    let grid = tetraglass::io::read_pvtu_from_path(&pvtu_path).expect("read hex_partitioned.pvtu");
    assert_eq!(grid.num_points(), 27);
    assert_eq!(grid.num_cells(), 8);

    // Because shared interface vertices at x = 1.0 are welded and ghost cells are stripped,
    // external surface extraction produces the exact 24 outer quad faces (48 triangles) of [0, 2]^3
    // with zero internal partition faces, and linearized tetrahedral volume is 8.0.
    let surface =
        tetraglass::filters::extract_external_surface(&grid).expect("extract external surface");
    assert_eq!(surface.num_faces(), 24);
    assert_eq!(surface.num_triangles(), 48);

    let tet_view =
        tetraglass::filters::linearize(&grid, &Default::default()).expect("linearize pvtu");
    assert!((tet_view.total_volume() - 8.0).abs() < 1e-12);

    // Chain `calculator` and `warp_by_vector` via `tetraglass::prelude` (`M6.4`)
    let with_q = calculator(
        &tet_view,
        &CalculatorParams::new("dynamic_q", "0.5 * mag(velocity)^2"),
    )
    .expect("compute dynamic_q");
    assert!(with_q.point_data().get("dynamic_q").is_some());

    let warped = warp_by_vector(&with_q, &WarpByVectorParams::with_scale("velocity", 0.1))
        .expect("warp_by_vector");
    assert_eq!(warped.num_points(), 27);
}

//! Integration tests for GPU ID buffer picking and CPU value probing (`M2.7`).

use std::path::PathBuf;

use tetraglass_filters::{SliceParams, extract_external_surface, linearize, slice};
use tetraglass_io::read_vtu_from_path;
use tetraglass_render::{
    Camera, ColorBy, DEFAULT_COLOR_FORMAT, FieldAssociation, HeadlessContext,
    HeadlessRenderOptions, PickTarget, Renderer, SurfaceStyle, probe_surface_triangle,
};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn gpu_pick_returns_miss_on_background_and_exact_analytic_values_on_mesh() {
    let root = workspace_root();
    let grid =
        read_vtu_from_path(root.join("testdata/vtk/mixed_linear.vtu")).expect("load mixed_linear");
    let surface = extract_external_surface(&grid).expect("extract surface");

    let headless = HeadlessContext::new().expect("headless wgpu context");
    let options = HeadlessRenderOptions {
        width: 400,
        height: 300,
        sample_count: 4,
        style: SurfaceStyle {
            color_by: ColorBy::point("linear"),
            ..SurfaceStyle::default()
        },
        ..HeadlessRenderOptions::default()
    };

    // 1. Background corner (5, 5) and out-of-bounds coordinates return `None`.
    assert_eq!(
        headless
            .pick_surface(&surface, &options, 5, 5)
            .expect("pick background"),
        None
    );
    assert_eq!(
        headless
            .pick_surface(&surface, &options, 999, 999)
            .expect("pick out of bounds"),
        None
    );

    // 2. Center pixel (200, 150) hits the mesh and reproduces analytic point & cell fields.
    let hit = headless
        .pick_surface(&surface, &options, 200, 150)
        .expect("pick center")
        .expect("center pixel must hit mixed_linear surface");

    assert_eq!(hit.representation_index, 0);
    assert_eq!(hit.pixel, [200, 150]);
    assert!(hit.triangle_index < surface.num_triangles());
    assert_eq!(
        hit.source_cell_id,
        Some(surface.triangle_source_cells()[hit.triangle_index])
    );
    assert!((hit.point_index as usize) < surface.num_points());
    assert!(hit.depth > 0.0 && hit.depth < 1.0);

    let [w0, w1, w2] = hit.barycentric;
    assert!(w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0);
    assert!(((w0 + w1 + w2) - 1.0).abs() < 1e-12);

    // On `mixed_linear.vtu`, `linear(x, y, z) = x + 2y + 3z` and `velocity(x, y, z) = [y, -x, 0.5z]`.
    let [x, y, z] = hit.world_position;
    let expected_linear = x + 2.0 * y + 3.0 * z;
    let linear_probe = hit
        .point_values
        .iter()
        .find(|pv| pv.name == "linear")
        .expect("linear point value");
    assert_eq!(linear_probe.association, FieldAssociation::Point);
    assert!((linear_probe.values[0] - expected_linear).abs() < 1e-10);
    assert!((hit.active_scalar.expect("active_scalar") - expected_linear).abs() < 1e-10);

    let vel_probe = hit
        .point_values
        .iter()
        .find(|pv| pv.name == "velocity")
        .expect("velocity point value");
    assert_eq!(vel_probe.values.len(), 3);
    assert!((vel_probe.values[0] - (-y)).abs() < 1e-10);
    assert!((vel_probe.values[1] - x).abs() < 1e-10);
    assert!(vel_probe.values[2].abs() < 1e-10);
    let expected_mag = (y * y + x * x).sqrt();
    assert!((vel_probe.magnitude.expect("velocity magnitude") - expected_mag).abs() < 1e-10);

    let cell_id_probe = hit
        .cell_values
        .iter()
        .find(|cv| cv.name == "cell_id")
        .expect("cell_id cell value");
    assert_eq!(cell_id_probe.association, FieldAssociation::Cell);
    assert_eq!(
        cell_id_probe.values[0],
        f64::from(hit.source_cell_id.unwrap())
    );
}

#[test]
fn multi_representation_pick_identifies_representation_and_skips_wireframe() {
    let root = workspace_root();
    let hex = read_vtu_from_path(root.join("testdata/vtk/hex_linear.vtu")).expect("load hex");
    let hull = extract_external_surface(&hex).expect("extract hull");
    let tet_view = linearize(&hex, &Default::default()).expect("linearize hex");
    let cut = slice(&tet_view, &SliceParams::new([0.0, 0.0, 1.0])).expect("slice hex");

    let headless = HeadlessContext::new().expect("headless wgpu context");
    let renderer = Renderer::new(headless.device(), headless.queue(), DEFAULT_COLOR_FORMAT, 4)
        .expect("create renderer");
    let pick_target = PickTarget::new(headless.device(), 400, 300).expect("create pick target");

    let bounds = hull.bounds().expect("hull bounds");
    let center = bounds.center();
    let camera = Camera::fit_bounds(bounds, 35.0, 24.0);

    // Representation 0: wireframe-only outer hull (`draw_surface: false`)
    let wire_style = SurfaceStyle {
        draw_surface: false,
        draw_edges: true,
        ..SurfaceStyle::default()
    };
    // Representation 1: shaded slice plane (`draw_surface: true`)
    let cut_style = SurfaceStyle {
        color_by: ColorBy::point("linear"),
        draw_surface: true,
        draw_edges: true,
        ..SurfaceStyle::default()
    };

    let gpu_wire = renderer
        .upload_surface_with_center(headless.device(), &hull, center, &wire_style)
        .expect("upload wire");
    let gpu_cut = renderer
        .upload_surface_with_center(headless.device(), &cut, center, &cut_style)
        .expect("upload cut");

    let raw = renderer
        .pick_pixel(
            headless.device(),
            headless.queue(),
            &pick_target,
            &camera,
            &[&gpu_wire, &gpu_cut],
            200,
            150,
        )
        .expect("pick_pixel")
        .expect("center pixel must hit slice representation");

    // Representation 0 is wireframe-only (`draw_surface: false`), so the hit must be representation 1 (`gpu_cut`)!
    assert_eq!(raw.representation_index, 1);

    let probed = probe_surface_triangle(
        &cut,
        raw.representation_index,
        [200, 150],
        [400, 300],
        &camera,
        center,
        raw,
        &cut_style.color_by,
    )
    .expect("probe slice triangle");
    assert_eq!(probed.representation_index, 1);
    // The slice plane is z = 1.0, so the probed world z coordinate must be 1.0.
    assert!((probed.world_position[2] - 1.0).abs() < 1e-10);
}

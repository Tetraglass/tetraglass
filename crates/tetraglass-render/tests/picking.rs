//! Integration tests for GPU ID buffer picking and CPU value probing (`M2.7`).

use std::path::PathBuf;

use tetraglass_filters::{SliceParams, extract_external_surface, linearize, slice};
use tetraglass_io::read_vtu_from_path;
use tetraglass_render::{
    Camera, ColorBy, DEFAULT_COLOR_FORMAT, FieldAssociation, HeadlessContext,
    HeadlessRenderOptions, OffscreenTarget, PickTarget, Renderer, ShaderClipKeep, ShaderClipPlane,
    SurfaceStyle, probe_surface_triangle,
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

#[test]
fn shader_clip_plane_clips_color_and_pick_passes_and_restores_when_cleared() {
    let root = workspace_root();
    let hex = read_vtu_from_path(root.join("testdata/vtk/hex_linear.vtu")).expect("load hex");
    let hull = extract_external_surface(&hex).expect("extract hull");

    let headless = HeadlessContext::new().expect("headless wgpu context");
    let renderer = Renderer::new(headless.device(), headless.queue(), DEFAULT_COLOR_FORMAT, 4)
        .expect("create renderer");
    let color_target =
        OffscreenTarget::new(headless.device(), 400, 300, DEFAULT_COLOR_FORMAT, 4).expect("target");
    let pick_target = PickTarget::new(headless.device(), 400, 300).expect("pick target");

    let bounds = hull.bounds().expect("hull bounds");
    let center = bounds.center();
    let camera = Camera::fit_bounds(bounds, 35.0, 24.0);

    let style = SurfaceStyle {
        color_by: ColorBy::point("linear"),
        draw_surface: true,
        draw_edges: true,
        ..SurfaceStyle::default()
    };
    let mut gpu_hull = renderer
        .upload_surface(headless.device(), &hull, &style)
        .expect("upload hull");
    assert_eq!(gpu_hull.clip_plane(), None);

    let count_mesh_pixels = |gpu: &tetraglass_render::GpuSurface| -> usize {
        let mut encoder =
            headless
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("clip_test_enc"),
                });
        renderer
            .render(
                &mut encoder,
                headless.queue(),
                &color_target,
                &camera,
                [1.0, 1.0, 1.0, 1.0],
                &[gpu],
            )
            .expect("render");
        headless.queue().submit(std::iter::once(encoder.finish()));
        let img = headless.readback_target(&color_target).expect("readback");
        let mut count = 0usize;
        for y in 0..300 {
            for x in 0..400 {
                let [r, g, b, _] = img.pixel(x, y);
                if r < 250 || g < 250 || b < 250 {
                    count += 1;
                }
            }
        }
        count
    };

    let full_pixels = count_mesh_pixels(&gpu_hull);
    assert!(full_pixels > 15_000, "unexpected full_pixels={full_pixels}");

    // 1. Set shader clip plane keeping `Back` (`x <= 1.0` on a `[0, 2]^3` cube).
    let back_plane =
        ShaderClipPlane::with_keep([1.0, 1.0, 1.0], [1.0, 0.0, 0.0], ShaderClipKeep::Back);
    renderer.set_surface_clip_plane(headless.queue(), &mut gpu_hull, Some(back_plane));
    assert_eq!(gpu_hull.clip_plane(), Some(back_plane));
    let back_pixels = count_mesh_pixels(&gpu_hull);
    assert!(
        back_pixels > 4_000 && back_pixels < full_pixels - 2_000,
        "expected back-clipped pixel count between 4000 and {}, got {back_pixels}",
        full_pixels - 2_000
    );

    // Every picked pixel under `back_plane` must satisfy `world_position[0] <= 1.0 + 1e-3`.
    for (px, py) in [(140, 150), (200, 150), (260, 150)] {
        if let Some(raw) = renderer
            .pick_pixel(
                headless.device(),
                headless.queue(),
                &pick_target,
                &camera,
                &[&gpu_hull],
                px,
                py,
            )
            .expect("pick")
        {
            let probed = probe_surface_triangle(
                &hull,
                raw.representation_index,
                [px, py],
                [400, 300],
                &camera,
                center,
                raw,
                &style.color_by,
            )
            .expect("probe");
            assert!(
                probed.world_position[0] <= 1.005,
                "expected x <= 1.005 under Back clip plane, got {}",
                probed.world_position[0]
            );
        }
    }

    // 2. Flip to `Front` (`x >= 1.0`): every picked pixel must satisfy `world_position[0] >= 1.0 - 1e-3`.
    let front_plane =
        ShaderClipPlane::with_keep([1.0, 1.0, 1.0], [1.0, 0.0, 0.0], ShaderClipKeep::Front);
    renderer.set_surface_clip_plane(headless.queue(), &mut gpu_hull, Some(front_plane));
    for (px, py) in [(140, 150), (200, 150), (260, 150)] {
        if let Some(raw) = renderer
            .pick_pixel(
                headless.device(),
                headless.queue(),
                &pick_target,
                &camera,
                &[&gpu_hull],
                px,
                py,
            )
            .expect("pick")
        {
            let probed = probe_surface_triangle(
                &hull,
                raw.representation_index,
                [px, py],
                [400, 300],
                &camera,
                center,
                raw,
                &style.color_by,
            )
            .expect("probe");
            assert!(
                probed.world_position[0] >= 0.995,
                "expected x >= 0.995 under Front clip plane, got {}",
                probed.world_position[0]
            );
        }
    }

    // 3. Clear the shader clip plane (`None`): pixel count returns to `full_pixels`.
    renderer.set_surface_clip_plane(headless.queue(), &mut gpu_hull, None);
    assert_eq!(gpu_hull.clip_plane(), None);
    assert_eq!(count_mesh_pixels(&gpu_hull), full_pixels);
}

#[test]
fn picking_points_representation_probes_vertex_and_scalar_values_and_supports_point_only_surfaces()
{
    use tetraglass_filters::{StreamlineParams, StreamlineSeeds, streamlines};

    let root = workspace_root();
    let grid =
        read_vtu_from_path(root.join("testdata/vtk/mixed_linear.vtu")).expect("load mixed_linear");
    let surface = extract_external_surface(&grid).expect("extract surface");
    let bounds = surface.bounds().expect("surface bounds");
    let camera = Camera::fit_bounds(bounds, 30.0, 20.0);

    let headless = HeadlessContext::new().expect("headless wgpu context");
    let style = SurfaceStyle {
        color_by: ColorBy::point("linear"),
        draw_surface: false,
        draw_edges: false,
        draw_points: true,
        point_size: 18.0,
        ..SurfaceStyle::default()
    };
    let options = HeadlessRenderOptions {
        width: 400,
        height: 300,
        sample_count: 4,
        style: style.clone(),
        camera: Some(camera.clone()),
        ..HeadlessRenderOptions::default()
    };

    // Project vertex 0 into screen coordinates and pick at its pixel center.
    let center = bounds.center();
    let vp = camera.view_proj_matrix(400.0 / 300.0);
    let project_point = |p: [f64; 3]| -> [u32; 2] {
        let rx = (p[0] - center[0]) as f32;
        let ry = (p[1] - center[1]) as f32;
        let rz = (p[2] - center[2]) as f32;
        let cx = vp[0][0] * rx + vp[1][0] * ry + vp[2][0] * rz + vp[3][0];
        let cy = vp[0][1] * rx + vp[1][1] * ry + vp[2][1] * rz + vp[3][1];
        let cw = vp[0][3] * rx + vp[1][3] * ry + vp[2][3] * rz + vp[3][3];
        let ndc_x = cx / cw;
        let ndc_y = cy / cw;
        let px = ((ndc_x * 0.5 + 0.5) * 400.0).round().clamp(0.0, 399.0) as u32;
        let py = ((0.5 - ndc_y * 0.5) * 300.0).round().clamp(0.0, 299.0) as u32;
        [px, py]
    };

    let [px, py] = project_point(surface.points()[0]);
    let hit = headless
        .pick_surface(&surface, &options, px, py)
        .expect("pick point impostor")
        .expect("projected vertex pixel must hit a point impostor");
    assert_eq!(hit.representation_index, 0);
    assert!(hit.triangle_index < surface.num_triangles());
    assert!((hit.point_index as usize) < surface.num_points());
    assert!(hit.active_scalar.is_some());

    // Also test picking a 0-triangle streamline station-point surface (`polylines.to_line_surface()`).
    let hex = read_vtu_from_path(root.join("testdata/vtk/hex_linear.vtu")).expect("load hex");
    let tet_view = linearize(&hex, &Default::default()).expect("linearize hex");
    let trace = streamlines(
        &tet_view,
        &StreamlineParams::new("velocity", StreamlineSeeds::Points(vec![[1.2, 0.5, 1.0]])),
    )
    .expect("trace streamline");
    let line_surface = trace.to_line_surface().expect("to_line_surface");
    assert_eq!(line_surface.num_triangles(), 0);
    assert!(line_surface.num_points() > 2);

    let line_bounds = line_surface.bounds().expect("line bounds");
    let line_cam = Camera::fit_bounds(line_bounds, 25.0, 20.0);
    let line_center = line_bounds.center();
    let line_vp = line_cam.view_proj_matrix(400.0 / 300.0);
    let p0 = line_surface.points()[0];
    let rx = (p0[0] - line_center[0]) as f32;
    let ry = (p0[1] - line_center[1]) as f32;
    let rz = (p0[2] - line_center[2]) as f32;
    let cx = line_vp[0][0] * rx + line_vp[1][0] * ry + line_vp[2][0] * rz + line_vp[3][0];
    let cy = line_vp[0][1] * rx + line_vp[1][1] * ry + line_vp[2][1] * rz + line_vp[3][1];
    let cw = line_vp[0][3] * rx + line_vp[1][3] * ry + line_vp[2][3] * rz + line_vp[3][3];
    let lpx = (((cx / cw) * 0.5 + 0.5) * 400.0).round().clamp(0.0, 399.0) as u32;
    let lpy = ((0.5 - (cy / cw) * 0.5) * 300.0).round().clamp(0.0, 299.0) as u32;

    let line_opts = HeadlessRenderOptions {
        width: 400,
        height: 300,
        sample_count: 4,
        style: SurfaceStyle {
            color_by: ColorBy::point("arc_length"),
            draw_surface: false,
            draw_edges: false,
            draw_points: true,
            point_size: 18.0,
            ..SurfaceStyle::default()
        },
        camera: Some(line_cam),
        ..HeadlessRenderOptions::default()
    };
    let line_hit = headless
        .pick_surface(&line_surface, &line_opts, lpx, lpy)
        .expect("pick streamline point")
        .expect("projected streamline station point must hit");
    assert!((line_hit.point_index as usize) < line_surface.num_points());
    assert!(line_hit.active_scalar.is_some());
}

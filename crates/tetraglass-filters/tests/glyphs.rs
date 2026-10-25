//! Integration tests for the `glyphs` filter (`M4.3`).
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

use tetraglass_core::{ArrayBuffer, ComponentCount, DataArray, UnstructuredGrid};
use tetraglass_filters::{
    DEFAULT_GLYPH_LENGTH_FRACTION, FilterError, GlyphAssociation, GlyphParams, GlyphScaleMode,
    GlyphShape, glyph_instances, glyphs, glyphs_grid, glyphs_with_chunk_size, linearize_default,
};
use tetraglass_io::{read_msh_from_path, read_vtu_from_path};

fn testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn load_vtu(name: &str) -> UnstructuredGrid {
    read_vtu_from_path(testdata_dir().join("vtk").join(name)).expect("valid VTU test fixture")
}

fn load_msh(name: &str) -> UnstructuredGrid {
    read_msh_from_path(testdata_dir().join("gmsh").join(name)).expect("valid Gmsh test fixture")
}

fn surface_area(mesh: &tetraglass_core::SurfaceMesh) -> f64 {
    let mut area = 0.0;
    for &[i0, i1, i2] in mesh.triangles() {
        let p0 = mesh.points()[i0 as usize];
        let p1 = mesh.points()[i1 as usize];
        let p2 = mesh.points()[i2 as usize];
        let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cross = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        area += 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    }
    area
}

#[test]
fn glyph_instances_analytic_orientation_and_scaling_on_hex_linear() {
    let grid = load_vtu("hex_linear.vtu");
    let view = linearize_default(&grid).expect("linearize hex_linear");

    let params = GlyphParams::new("velocity");
    let instances = glyph_instances(&view, &params).expect("glyph_instances succeeds");
    // On hex_linear ([0,1]^3, 27 points), 3 points on the z-axis (x=0, y=0) have velocity=[0,0,0]
    // and zero scale in Magnitude mode, so 24 active glyph instances are returned.
    assert_eq!(instances.len(), 24);

    let bbox = view.bounds().expect("non-empty bounds");
    let diag = bbox.diagonal();
    let expected_base = DEFAULT_GLYPH_LENGTH_FRACTION * diag;
    let max_mag = (bbox.max[0].powi(2) + bbox.max[1].powi(2)).sqrt();

    for inst in &instances {
        let [x, y, _z] = inst.position;
        let expected_mag = (x * x + y * y).sqrt();
        assert!(
            (inst.magnitude - expected_mag).abs() < 1e-12,
            "magnitude mismatch at {:?}: got {}, expected {}",
            inst.position,
            inst.magnitude,
            expected_mag
        );
        let expected_dir = [-y / expected_mag, x / expected_mag, 0.0];
        for axis in 0..3 {
            assert!(
                (inst.direction[axis] - expected_dir[axis]).abs() < 1e-12,
                "direction mismatch at {:?}: got {:?}, expected {:?}",
                inst.position,
                inst.direction,
                expected_dir
            );
        }
        let expected_scale = expected_base * (expected_mag / max_mag);
        assert!(
            (inst.scale - expected_scale).abs() < 1e-12,
            "scale mismatch: got {}, expected {}",
            inst.scale,
            expected_scale
        );
    }

    // Fixed scaling mode + scale_factor + every striding
    // In Fixed mode, even zero-magnitude points have positive scale (0.25 * 2.0 = 0.5).
    let fixed_params = GlyphParams::new("velocity")
        .scale_by(GlyphScaleMode::Fixed(0.25))
        .scale_factor(2.0)
        .every(3);
    let fixed_instances = glyph_instances(&view, &fixed_params).expect("fixed glyph_instances");
    assert_eq!(fixed_instances.len(), view.num_points().div_ceil(3));
    for inst in &fixed_instances {
        assert!(
            (inst.scale - 0.5).abs() < 1e-12,
            "fixed scaled length should be 0.25 * 2.0 = 0.5, got {}",
            inst.scale
        );
    }
}

#[test]
fn glyph_shapes_arrow_cone_sphere_produce_valid_surfaces_with_arrays() {
    let grid = load_vtu("hex_linear.vtu");
    let view = linearize_default(&grid).expect("linearize hex_linear");

    let active_instances = 24;

    let cases = [
        (GlyphShape::Arrow, 38, 72),
        (GlyphShape::Cone, 14, 24),
        (GlyphShape::Sphere, 86, 168),
    ];

    for (shape, verts_per_glyph, tris_per_glyph) in cases {
        let params = GlyphParams::new("velocity").shape(shape);
        let mesh = glyphs(&view, &params).expect("glyphs surface succeeds");
        assert_eq!(
            mesh.num_points(),
            active_instances * verts_per_glyph,
            "unexpected point count for {:?}",
            shape
        );
        assert_eq!(
            mesh.num_triangles(),
            active_instances * tris_per_glyph,
            "unexpected triangle count for {:?}",
            shape
        );
        let area = surface_area(&mesh);
        assert!(
            area > 1e-3,
            "surface area should be positive for {:?}, got {}",
            shape,
            area
        );

        // Verify point_data and cell_data replication and glyph_magnitude
        assert!(mesh.point_data().get("glyph_magnitude").is_some());
        assert!(mesh.point_data().get("linear").is_some());
        assert!(mesh.point_data().get("velocity").is_some());
        assert!(mesh.cell_data().get("cell_id").is_some());
        assert_eq!(mesh.triangle_source_cells().len(), mesh.num_triangles());
        for &cid in mesh.triangle_source_cells() {
            assert!(
                (cid as usize) < grid.num_cells(),
                "source cell id {} out of range {}",
                cid,
                grid.num_cells()
            );
        }
    }
}

#[test]
fn glyphs_supports_cell_associated_and_point_associated_vector_arrays() {
    let mut grid = load_vtu("tet_linear.vtu");
    let num_cells = grid.num_cells();
    let mut cell_vecs = Vec::with_capacity(num_cells * 3);
    for c in 0..num_cells {
        let v = (c + 1) as f64;
        cell_vecs.extend_from_slice(&[v, 0.5 * v, 0.25 * v]);
    }
    grid.cell_data_mut()
        .insert(
            "cell_vel",
            DataArray::new(ComponentCount::Vector3, ArrayBuffer::F64(cell_vecs.into()))
                .expect("valid 3-comp array"),
        )
        .expect("insert cell_vel");

    let view = linearize_default(&grid).expect("linearize tet_linear");

    // Auto association finds cell_vel in cell_data and spawns 1 glyph per original source cell
    let auto_instances =
        glyph_instances(&view, &GlyphParams::new("cell_vel")).expect("auto cell_vel instances");
    assert_eq!(auto_instances.len(), num_cells);

    // Explicit Cell association
    let cell_mesh = glyphs(
        &view,
        &GlyphParams::new("cell_vel")
            .with_association(GlyphAssociation::Cell)
            .shape(GlyphShape::Cone),
    )
    .expect("cell glyphs");
    assert_eq!(cell_mesh.num_triangles(), num_cells * 24);

    // Explicit Point association on velocity
    let point_mesh = glyphs(
        &view,
        &GlyphParams::new("velocity")
            .with_association(GlyphAssociation::Point)
            .shape(GlyphShape::Cone),
    )
    .expect("point glyphs");
    assert!(point_mesh.num_triangles() > 0);
}

#[test]
fn glyphs_works_across_all_vtk_and_gmsh_cell_families() {
    let vtk_files = [
        "tet_linear.vtu",
        "hex_linear.vtu",
        "mixed_linear.vtu",
        "tet_quadratic.vtu",
        "hex_lagrange_p3.vtu",
        "polyhedra.vtu",
    ];

    for file in vtk_files {
        let grid = load_vtu(file);
        let mesh = glyphs_grid(
            &grid,
            &GlyphParams::new("velocity")
                .shape(GlyphShape::Arrow)
                .every(2),
        )
        .unwrap_or_else(|e| panic!("glyphs_grid failed on {file}: {e}"));
        assert!(
            mesh.num_triangles() > 0,
            "expected non-empty glyphs on {file}"
        );
        assert!(surface_area(&mesh) > 0.0);
    }

    let msh_grid = load_msh("tet2_v41.msh");
    let msh_mesh = glyphs_grid(
        &msh_grid,
        &GlyphParams::new("velocity")
            .shape(GlyphShape::Sphere)
            .every(2),
    )
    .expect("glyphs_grid on tet2_v41.msh");
    assert!(msh_mesh.num_triangles() > 0);
    assert!(surface_area(&msh_mesh) > 0.0);
}

#[test]
fn glyphs_chunk_parallel_determinism() {
    let grid = load_vtu("hex_lagrange_p3.vtu");
    let view = linearize_default(&grid).expect("linearize hex_lagrange_p3");
    let params = GlyphParams::new("velocity")
        .shape(GlyphShape::Arrow)
        .every(2);

    let baseline = glyphs_with_chunk_size(&view, &params, 1).expect("chunk_size = 1");
    for chunk_size in [7, 32, 256] {
        let other = glyphs_with_chunk_size(&view, &params, chunk_size)
            .expect("glyphs_with_chunk_size succeeds");
        assert_eq!(baseline.points(), other.points());
        assert_eq!(baseline.triangles(), other.triangles());
        assert_eq!(
            baseline.triangle_source_cells(),
            other.triangle_source_cells()
        );
    }
}

#[test]
fn glyphs_rejects_invalid_parameters_and_arrays() {
    let grid = load_vtu("tet_linear.vtu");
    let view = linearize_default(&grid).expect("linearize tet_linear");

    // every == 0
    let err = glyphs(&view, &GlyphParams::new("velocity").every(0)).expect_err("every=0 must fail");
    assert!(matches!(err, FilterError::InvalidGlyphParams { .. }));

    // non-positive Fixed scale
    let err = glyphs(
        &view,
        &GlyphParams::new("velocity").scale_by(GlyphScaleMode::Fixed(0.0)),
    )
    .expect_err("Fixed(0.0) must fail");
    assert!(matches!(err, FilterError::InvalidGlyphParams { .. }));

    // non-positive scale_factor
    let err = glyphs(&view, &GlyphParams::new("velocity").scale_factor(-1.0))
        .expect_err("scale_factor(-1.0) must fail");
    assert!(matches!(err, FilterError::InvalidGlyphParams { .. }));

    // missing array
    let err =
        glyphs(&view, &GlyphParams::new("no_such_vector")).expect_err("missing array must fail");
    assert!(matches!(err, FilterError::MissingAttributeArray { .. }));

    // 1-component scalar array instead of 3-component vector
    let err = glyphs(&view, &GlyphParams::new("linear")).expect_err("scalar array must fail");
    assert!(matches!(
        err,
        FilterError::NonVectorArray {
            num_components: 1,
            ..
        }
    ));
}

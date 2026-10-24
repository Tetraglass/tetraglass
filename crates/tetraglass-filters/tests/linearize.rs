//! Integration tests for `linearize` across linear, mixed, polyhedral, and synthetic meshes (`M2.1a`).
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;
use std::sync::Arc;

use tetraglass_core::{
    ArrayBuffer, CellType, ComponentCount, DataArray, PolyhedronFaces, SyntheticCellLayout,
    SyntheticGridSpec, UnstructuredGrid,
};
use tetraglass_filters::{
    FilterError, LinearizeParams, extract_external_surface, linearize, linearize_default,
};
use tetraglass_io::{read_msh_from_path, read_vtu_from_path};

fn vtk_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ directory")
        .parent()
        .expect("workspace root")
        .join("testdata")
        .join("vtk")
}

fn gmsh_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ directory")
        .parent()
        .expect("workspace root")
        .join("testdata")
        .join("gmsh")
}

#[test]
fn tet_linear_preserves_tets_volume_and_arc_buffers() {
    let grid = read_vtu_from_path(vtk_dir().join("tet_linear.vtu")).unwrap();
    let view = linearize_default(&grid).unwrap();

    assert_eq!(view.num_points(), 8);
    assert_eq!(view.num_tets(), 6);
    assert!((view.total_volume() - 1.0).abs() < 1e-12);
    for tet_idx in 0..view.num_tets() {
        assert!((view.tet_signed_volume(tet_idx).unwrap() - 1.0 / 6.0).abs() < 1e-12);
        assert_eq!(view.tet_source_cells()[tet_idx], tet_idx as u32);
    }

    // Points and point_data buffers are shared via Arc with zero copying.
    assert!(Arc::ptr_eq(&grid.points_arc(), &view.points_arc()));
    match (
        grid.point_data().get("linear").unwrap().buffer(),
        view.point_data().get("linear").unwrap().buffer(),
    ) {
        (ArrayBuffer::F64(a), ArrayBuffer::F64(b)) => assert!(Arc::ptr_eq(a, b)),
        _ => panic!("expected F64 buffer"),
    }
}

#[test]
fn hex_linear_preserves_volume_and_has_conformal_internal_faces() {
    let grid = read_vtu_from_path(vtk_dir().join("hex_linear.vtu")).unwrap();
    let view = linearize_default(&grid).unwrap();

    // 8 unit-size hexes on [0, 2]^3 -> 48 tets, total volume = 8.0.
    assert_eq!(view.num_points(), 27);
    assert_eq!(view.num_tets(), 48);
    assert!((view.total_volume() - 8.0).abs() < 1e-12);
    for tet_idx in 0..view.num_tets() {
        assert!(view.tet_signed_volume(tet_idx).unwrap() > 0.0);
    }

    // Because shared quad faces use the minimum-global-vertex diagonal rule,
    // extracting the external surface of the linearized tet mesh must produce
    // the exact 48 boundary triangles (area 24.0) with zero leaked internal triangles.
    let tet_grid = view.to_unstructured_grid().unwrap();
    let tet_surf = extract_external_surface(&tet_grid).unwrap();
    assert_eq!(tet_surf.num_faces(), 48);
    assert_eq!(tet_surf.num_triangles(), 48);
    assert!((tet_surf.total_area() - 24.0).abs() < 1e-12);
}

#[test]
fn mixed_linear_preserves_per_cell_volumes_and_is_watertight_across_cell_types() {
    let grid = read_vtu_from_path(vtk_dir().join("mixed_linear.vtu")).unwrap();
    let orig_surf = extract_external_surface(&grid).unwrap();
    let view = linearize_default(&grid).unwrap();

    // 1 hex (6) + 1 wedge (3) + 1 pyramid (2) + 1 tet (1) = 12 tets.
    assert_eq!(view.num_points(), 11);
    assert_eq!(view.num_tets(), 12);
    assert!((view.total_volume() - 2.0).abs() < 1e-12);

    let mut cell_volumes = [0.0_f64; 4];
    let cell_id_arr = view.cell_data().get("cell_id").unwrap();
    for tet_idx in 0..view.num_tets() {
        let signed_vol = view.tet_signed_volume(tet_idx).unwrap();
        assert!(
            signed_vol > 0.0,
            "tet {tet_idx} has non-positive signed volume {signed_vol}"
        );
        let src = view.tet_source_cells()[tet_idx] as usize;
        cell_volumes[src] += signed_vol;
        assert_eq!(cell_id_arr.value_as_f64(tet_idx, 0), Some(src as f64));
    }

    assert!((cell_volumes[0] - 1.0).abs() < 1e-12); // hex
    assert!((cell_volumes[1] - 0.5).abs() < 1e-12); // wedge
    assert!((cell_volumes[2] - 1.0 / 3.0).abs() < 1e-12); // pyramid
    assert!((cell_volumes[3] - 1.0 / 6.0).abs() < 1e-12); // tet

    // Conformal diagonal check across hex-wedge, hex-pyramid, wedge-tet, and pyramid-tet faces:
    // the external surface of the 12 linearized tets must have 18 outer triangles and the exact
    // same surface area as the original mixed mesh.
    let tet_grid = view.to_unstructured_grid().unwrap();
    let tet_surf = extract_external_surface(&tet_grid).unwrap();
    assert_eq!(tet_surf.num_faces(), 18);
    assert_eq!(tet_surf.num_triangles(), 18);
    assert!((tet_surf.total_area() - orig_surf.total_area()).abs() < 1e-12);
}

#[test]
fn polyhedra_centroid_decomposition_preserves_volume_and_linear_fields_exactly() {
    let grid = read_vtu_from_path(vtk_dir().join("polyhedra.vtu")).unwrap();
    let orig_surf = extract_external_surface(&grid).unwrap();
    let view = linearize_default(&grid).unwrap();

    // 14 original points + 2 polyhedron centroids = 16 points.
    // Cell 0 (cube, 6 quads = 12 tris) + Cell 1 (5 quads + 2 pentagons = 10 + 6 = 16 tris) = 28 tets.
    assert_eq!(view.num_points(), 16);
    assert_eq!(view.num_tets(), 28);
    assert!((view.total_volume() - 1.875).abs() < 1e-12);

    let mut cell_volumes = [0.0_f64; 2];
    for tet_idx in 0..view.num_tets() {
        let signed_vol = view.tet_signed_volume(tet_idx).unwrap();
        assert!(signed_vol > 0.0);
        let src = view.tet_source_cells()[tet_idx] as usize;
        cell_volumes[src] += signed_vol;
    }
    assert!((cell_volumes[0] - 1.0).abs() < 1e-12);
    assert!((cell_volumes[1] - 0.875).abs() < 1e-12);

    // Every point (including the 2 added centroid points) must reproduce linear fields
    // `linear = x + 2y + 3z` and `velocity = (-y, x, 0)` exactly.
    let linear_arr = view.point_data().get("linear").unwrap();
    let vel_arr = view.point_data().get("velocity").unwrap();
    for (pt_idx, &[x, y, z]) in view.points().iter().enumerate() {
        let expected_linear = x + 2.0 * y + 3.0 * z;
        let actual_linear = linear_arr.value_as_f64(pt_idx, 0).unwrap();
        assert!(
            (actual_linear - expected_linear).abs() < 1e-12,
            "point {pt_idx} linear mismatch: expected {expected_linear}, got {actual_linear}"
        );

        let vx = vel_arr.value_as_f64(pt_idx, 0).unwrap();
        let vy = vel_arr.value_as_f64(pt_idx, 1).unwrap();
        let vz = vel_arr.value_as_f64(pt_idx, 2).unwrap();
        assert!((vx - (-y)).abs() < 1e-12);
        assert!((vy - x).abs() < 1e-12);
        assert!(vz.abs() < 1e-12);
    }

    // Shared quad face [4, 5, 6, 7] between the two polyhedra is fan-triangulated from minimum
    // vertex 4 in both cells, so the linearized tet mesh has 24 outer triangles and 0 internal ones.
    let tet_grid = view.to_unstructured_grid().unwrap();
    let tet_surf = extract_external_surface(&tet_grid).unwrap();
    assert_eq!(tet_surf.num_triangles(), 24);
    assert!((tet_surf.total_area() - orig_surf.total_area()).abs() < 1e-12);
}

#[test]
fn synthetic_hex_and_tet_grids_linearize_with_exact_volume_and_watertight_faces() {
    let hex_spec = SyntheticGridSpec::new(10, 8, 6, SyntheticCellLayout::Hexahedra);
    let hex_grid = hex_spec.generate().unwrap();
    let hex_view = linearize_default(&hex_grid).unwrap();

    assert_eq!(hex_view.num_tets(), hex_spec.num_cells() * 6);
    assert!((hex_view.total_volume() - 1.0).abs() < 1e-11);

    let tet_grid = hex_view.to_unstructured_grid().unwrap();
    let tet_surf = extract_external_surface(&tet_grid).unwrap();
    assert_eq!(
        tet_surf.num_triangles(),
        hex_spec.expected_boundary_faces() * 2
    );
    assert!((tet_surf.total_area() - 6.0).abs() < 1e-11);
}

#[test]
fn empty_grid_and_invalid_subdivisions_are_handled() {
    let empty =
        UnstructuredGrid::new(Vec::<[f64; 3]>::new(), vec![], vec![], vec![], None).unwrap();
    let view = linearize_default(&empty).unwrap();
    assert_eq!(view.num_points(), 0);
    assert_eq!(view.num_tets(), 0);
    assert_eq!(view.total_volume(), 0.0);

    assert_eq!(
        linearize(&empty, &LinearizeParams::new(0)),
        Err(FilterError::InvalidSubdivisionLevel { subdivisions: 0 })
    );
}

#[test]
fn quadratic_tetra_vtu_and_gmsh_subdivide_watertight_and_reproduce_fields() {
    let vtu_grid = read_vtu_from_path(vtk_dir().join("tet_quadratic.vtu")).unwrap();
    let msh_grid = read_msh_from_path(gmsh_dir().join("tet2_v41.msh")).unwrap();

    for grid in [&vtu_grid, &msh_grid] {
        // S = 1: 6 corner tets
        let v1 = linearize(grid, &LinearizeParams::new(1)).unwrap();
        assert_eq!(v1.num_tets(), 6);
        assert!((v1.total_volume() - 1.0).abs() < 1e-12);

        // S = 2 (default): 6 * 2^3 = 48 tets; all 27 quadratic nodes coincide with S=2 lattice
        // points so zero extra points are allocated and point buffers are shared via Arc.
        let v2 = linearize_default(grid).unwrap();
        assert_eq!(v2.num_points(), 27);
        assert_eq!(v2.num_tets(), 48);
        assert!((v2.total_volume() - 1.0).abs() < 1e-12);
        assert!(Arc::ptr_eq(&grid.points_arc(), &v2.points_arc()));

        let surf2 = extract_external_surface(&v2.to_unstructured_grid().unwrap()).unwrap();
        // 12 outer faces * 4 sub-triangles = 48 outer triangles, total area = 6.0
        assert_eq!(surf2.num_triangles(), 48);
        assert!((surf2.total_area() - 6.0).abs() < 1e-12);

        // S = 3: 6 * 3^3 = 162 tets; synthesized points interpolate linear fields exactly
        let v3 = linearize(grid, &LinearizeParams::new(3)).unwrap();
        assert_eq!(v3.num_tets(), 162);
        assert!((v3.total_volume() - 1.0).abs() < 1e-12);
        let lin_arr = v3.point_data().get("linear").unwrap();
        for (pt_idx, &[x, y, z]) in v3.points().iter().enumerate() {
            let expected = x + 2.0 * y + 3.0 * z;
            let actual = lin_arr.value_as_f64(pt_idx, 0).unwrap();
            assert!((actual - expected).abs() < 1e-11);
        }
        let surf3 = extract_external_surface(&v3.to_unstructured_grid().unwrap()).unwrap();
        // 12 outer faces * 9 sub-triangles = 108 outer triangles, total area = 6.0
        assert_eq!(surf3.num_triangles(), 108);
        assert!((surf3.total_area() - 6.0).abs() < 1e-12);
    }
}

#[test]
fn lagrange_hex_p3_subdivides_watertight_and_reuses_nodes_at_order_3() {
    let grid = read_vtu_from_path(vtk_dir().join("hex_lagrange_p3.vtu")).unwrap();

    // S = 2: 2 hexes * 2^3 * 6 = 96 tets on [0, 2] x [0, 1]^2 (volume = 2.0)
    let v2 = linearize(&grid, &LinearizeParams::new(2)).unwrap();
    assert_eq!(v2.num_tets(), 96);
    assert!((v2.total_volume() - 2.0).abs() < 1e-12);

    let lin2 = v2.point_data().get("linear").unwrap();
    let vel2 = v2.point_data().get("velocity").unwrap();
    for (pt_idx, &[x, y, z]) in v2.points().iter().enumerate() {
        assert!((lin2.value_as_f64(pt_idx, 0).unwrap() - (x + 2.0 * y + 3.0 * z)).abs() < 1e-11);
        assert!((vel2.value_as_f64(pt_idx, 0).unwrap() - (-y)).abs() < 1e-11);
        assert!((vel2.value_as_f64(pt_idx, 1).unwrap() - x).abs() < 1e-11);
    }
    // 10 outer quad faces * 2^2 sub-quads * 2 tris = 80 outer triangles, area = 10.0
    let surf2 = extract_external_surface(&v2.to_unstructured_grid().unwrap()).unwrap();
    assert_eq!(surf2.num_triangles(), 80);
    assert!((surf2.total_area() - 10.0).abs() < 1e-11);

    // S = 3: matches polynomial order 3 -> all 112 mesh nodes are reused directly (zero extra points)
    let v3 = linearize(&grid, &LinearizeParams::new(3)).unwrap();
    assert_eq!(v3.num_points(), 112);
    assert_eq!(v3.num_tets(), 2 * 27 * 6);
    assert!((v3.total_volume() - 2.0).abs() < 1e-12);
    assert!(Arc::ptr_eq(&grid.points_arc(), &v3.points_arc()));

    // 10 outer quad faces * 3^2 sub-quads * 2 tris = 180 outer triangles, area = 10.0
    let surf3 = extract_external_surface(&v3.to_unstructured_grid().unwrap()).unwrap();
    assert_eq!(surf3.num_triangles(), 180);
    assert!((surf3.total_area() - 10.0).abs() < 1e-11);
}

#[test]
fn quadratic_tetra_reproduces_quadratic_fields_and_curved_edge_geometry() {
    // Straight-sided 10-node quadratic tetrahedron with a quadratic scalar field
    // q(x, y, z) = 2*x^2 - 1.5*y^2 + 0.75*x*y + 3*z + 1.0.
    // Because the element basis is complete P2, subdividing at S = 3 must reproduce q(x, y, z)
    // to machine precision at all 20 subdivided lattice points!
    let corners = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];
    let mut pts = corners.to_vec();
    for &(a, b) in &edges {
        pts.push([
            0.5 * (corners[a][0] + corners[b][0]),
            0.5 * (corners[a][1] + corners[b][1]),
            0.5 * (corners[a][2] + corners[b][2]),
        ]);
    }
    let quad_fn = |[x, y, z]: [f64; 3]| 2.0 * x * x - 1.5 * y * y + 0.75 * x * y + 3.0 * z + 1.0;
    let quad_vals: Vec<f64> = pts.iter().copied().map(quad_fn).collect();

    let mut grid = UnstructuredGrid::new(
        pts.clone(),
        vec![CellType::QuadraticTetra],
        vec![10],
        (0..10).collect(),
        None,
    )
    .unwrap();
    grid.insert_point_data(
        "quad_field",
        DataArray::from_f64(ComponentCount::Scalar, quad_vals).unwrap(),
    )
    .unwrap();

    let view = linearize(&grid, &LinearizeParams::new(3)).unwrap();
    assert_eq!(view.num_tets(), 27);
    assert!((view.total_volume() - 1.0 / 6.0).abs() < 1e-12);
    let q_arr = view.point_data().get("quad_field").unwrap();
    for (idx, &p) in view.points().iter().enumerate() {
        let expected = quad_fn(p);
        let actual = q_arr.value_as_f64(idx, 0).unwrap();
        assert!(
            (actual - expected).abs() < 1e-12,
            "point {idx} at {p:?}: quadratic field {actual} != {expected}"
        );
    }

    // Now curve mid-edge node 4 on edge (0, 1) outward in -y by dy = -0.3:
    // Along edge (0, 1) (y_linear = 0, z = 0), the quadratic curve is y(x) = -4 * 0.3 * x * (1 - x),
    // and the exact volume of the level-S tetrahedralization is 1/6 + (0.3 / 6) * (1 - 1 / S^2).
    let mut curved_pts = pts;
    curved_pts[4][1] = -0.3;
    let curved_grid = UnstructuredGrid::new(
        curved_pts,
        vec![CellType::QuadraticTetra],
        vec![10],
        (0..10).collect(),
        None,
    )
    .unwrap();
    let curved_view = linearize(&curved_grid, &LinearizeParams::new(3)).unwrap();
    // At x = 1/3, z = 0 on edge (0, 1), y must equal -1.2 * (1/3) * (2/3) = -4 / 15.
    let found_third = curved_view.points().iter().any(|&[x, y, z]| {
        (x - 1.0 / 3.0).abs() < 1e-12 && (y - (-4.0 / 15.0)).abs() < 1e-12 && z.abs() < 1e-12
    });
    assert!(
        found_third,
        "curved edge point at x = 1/3 did not follow quadratic parabola y = -4/15"
    );
    let expected_curved_vol = 1.0 / 6.0 + (0.3 / 6.0) * (1.0 - 1.0 / 9.0);
    assert!(
        (curved_view.total_volume() - expected_curved_vol).abs() < 1e-12,
        "curved volume {} != {expected_curved_vol}",
        curved_view.total_volume()
    );
}

/// Deterministic SplitMix64 PRNG for reproducible property tests without external dependencies.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self, min: f64, max: f64) -> f64 {
        let unit = ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
        min + unit * (max - min)
    }
}

fn reference_cell_template(cell_type: CellType) -> (Vec<[f64; 3]>, Option<PolyhedronFaces>, f64) {
    match cell_type {
        CellType::Tetra => (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            None,
            1.0 / 6.0,
        ),
        CellType::Pyramid => (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            None,
            1.0 / 3.0,
        ),
        CellType::Wedge => (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            None,
            0.5,
        ),
        CellType::Hexahedron => (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            None,
            1.0,
        ),
        CellType::Polyhedron => {
            let pts = vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ];
            let stream = vec![
                6, // 6 quad faces of a cube
                4, 0, 3, 2, 1, // -z
                4, 4, 5, 6, 7, // +z
                4, 0, 1, 5, 4, // -y
                4, 2, 3, 7, 6, // +y
                4, 0, 4, 7, 3, // -x
                4, 1, 2, 6, 5, // +x
            ];
            let pf = PolyhedronFaces::new(vec![stream.len() as u32], stream);
            (pts, Some(pf), 1.0)
        }
        CellType::QuadraticTetra => (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ],
            None,
            1.0 / 6.0,
        ),
        CellType::QuadraticPyramid => (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [0.5, 1.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.25, 0.25, 0.5],
                [0.75, 0.25, 0.5],
                [0.75, 0.75, 0.5],
                [0.25, 0.75, 0.5],
            ],
            None,
            1.0 / 3.0,
        ),
        CellType::QuadraticWedge => (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 1.0],
                [0.5, 0.5, 1.0],
                [0.0, 0.5, 1.0],
                [0.0, 0.0, 0.5],
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
            ],
            None,
            0.5,
        ),
        CellType::QuadraticHexahedron => (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.5, 0.0],
                [0.5, 1.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 1.0],
                [1.0, 0.5, 1.0],
                [0.5, 1.0, 1.0],
                [0.0, 0.5, 1.0],
                [0.0, 0.0, 0.5],
                [1.0, 0.0, 0.5],
                [1.0, 1.0, 0.5],
                [0.0, 1.0, 0.5],
            ],
            None,
            1.0,
        ),
        CellType::LagrangeTetrahedron { order: 2 } => {
            reference_cell_template(CellType::QuadraticTetra)
        }
        CellType::LagrangeHexahedron { order: 2 } => {
            let mut pts = reference_cell_template(CellType::QuadraticHexahedron).0;
            // Face centers (-x, +x, -y, +y, -z, +z) and volume center
            pts.extend_from_slice(&[
                [0.0, 0.5, 0.5],
                [1.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 1.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 1.0],
                [0.5, 0.5, 0.5],
            ]);
            (pts, None, 1.0)
        }
        CellType::LagrangeWedge { order: 2 } => {
            let mut pts = reference_cell_template(CellType::QuadraticWedge).0;
            // 3 quad face centers: (0,1,4,3), (1,2,5,4), (2,0,3,5)
            pts.extend_from_slice(&[[0.5, 0.0, 0.5], [0.5, 0.5, 0.5], [0.0, 0.5, 0.5]]);
            (pts, None, 0.5)
        }
        other => panic!("unsupported template cell type {other:?}"),
    }
}

#[test]
fn property_tests_preserve_volume_linear_fields_and_source_cells_across_all_cell_families() {
    let families = [
        CellType::Tetra,
        CellType::Pyramid,
        CellType::Wedge,
        CellType::Hexahedron,
        CellType::Polyhedron,
        CellType::QuadraticTetra,
        CellType::QuadraticPyramid,
        CellType::QuadraticWedge,
        CellType::QuadraticHexahedron,
        CellType::LagrangeTetrahedron { order: 2 },
        CellType::LagrangeHexahedron { order: 2 },
        CellType::LagrangeWedge { order: 2 },
    ];

    let mut rng = SplitMix64::new(0x5EED_2026_1024_021B);

    for &cell_type in &families {
        let (ref_pts, poly_faces, ref_vol) = reference_cell_template(cell_type);
        let n_nodes = ref_pts.len();

        for trial in 0..20 {
            // Random positive-determinant upper-triangular + shear affine transform A * p + shift
            let d0 = rng.next_f64(0.5, 2.5);
            let d1 = rng.next_f64(0.5, 2.5);
            let d2 = rng.next_f64(0.5, 2.5);
            let s01 = rng.next_f64(-0.6, 0.6);
            let s02 = rng.next_f64(-0.6, 0.6);
            let s12 = rng.next_f64(-0.6, 0.6);
            let shift = [
                rng.next_f64(-10.0, 10.0),
                rng.next_f64(-10.0, 10.0),
                rng.next_f64(-10.0, 10.0),
            ];
            let det_a = d0 * d1 * d2;

            let transformed_pts: Vec<[f64; 3]> = ref_pts
                .iter()
                .map(|&[x, y, z]| {
                    [
                        d0 * x + s01 * y + s02 * z + shift[0],
                        d1 * y + s12 * z + shift[1],
                        d2 * z + shift[2],
                    ]
                })
                .collect();

            // Random linear scalar field f(x, y, z) = c0 + cx*x + cy*y + cz*z
            let c0 = rng.next_f64(-5.0, 5.0);
            let cx = rng.next_f64(-4.0, 4.0);
            let cy = rng.next_f64(-4.0, 4.0);
            let cz = rng.next_f64(-4.0, 4.0);
            let scalar_fn = |[x, y, z]: [f64; 3]| c0 + cx * x + cy * y + cz * z;

            let scalar_vals: Vec<f64> = transformed_pts.iter().copied().map(scalar_fn).collect();
            let vec_vals: Vec<f64> = transformed_pts
                .iter()
                .flat_map(|&[x, y, z]| [x - 2.0 * y, 3.0 * y + z, -x + 0.5 * z])
                .collect();

            let conn: Vec<u32> = (0..n_nodes as u32).collect();
            let mut grid = UnstructuredGrid::new(
                transformed_pts,
                vec![cell_type],
                vec![n_nodes as u32],
                conn,
                poly_faces.clone(),
            )
            .unwrap();
            grid.insert_point_data(
                "linear_scalar",
                DataArray::from_f64(ComponentCount::Scalar, scalar_vals).unwrap(),
            )
            .unwrap();
            grid.insert_point_data(
                "linear_vec",
                DataArray::from_f64(ComponentCount::Vector3, vec_vals).unwrap(),
            )
            .unwrap();
            grid.insert_cell_data(
                "cell_tag",
                DataArray::from_i32(ComponentCount::Scalar, vec![42]).unwrap(),
            )
            .unwrap();

            let subdiv = ((trial % 3) + 1) as u8;
            let view = linearize(&grid, &LinearizeParams::new(subdiv)).unwrap();

            // Property 1: Total volume preserved under affine transformation and subdivision
            let expected_vol = det_a * ref_vol;
            let actual_vol = view.total_volume();
            assert!(
                (actual_vol - expected_vol).abs() < 1e-10 * expected_vol.max(1.0),
                "{cell_type:?} trial {trial} (subdiv={subdiv}): volume {actual_vol} != {expected_vol}"
            );

            // Property 2: Every output tetrahedron has strictly positive signed volume
            // and maps back to valid source cell 0 with cell_tag == 42
            let cell_tag = view.cell_data().get("cell_tag").unwrap();
            for tet_idx in 0..view.num_tets() {
                let sv = view.tet_signed_volume(tet_idx).unwrap();
                assert!(
                    sv > 1e-14,
                    "{cell_type:?} trial {trial} tet {tet_idx} non-positive signed volume {sv}"
                );
                assert_eq!(view.tet_source_cells()[tet_idx], 0);
                assert_eq!(cell_tag.value_as_f64(tet_idx, 0), Some(42.0));
            }

            // Property 3: Linear scalar and vector fields are reproduced exactly at every output point
            let out_scalar = view.point_data().get("linear_scalar").unwrap();
            let out_vec = view.point_data().get("linear_vec").unwrap();
            for (pt_idx, &p) in view.points().iter().enumerate() {
                let expected_s = scalar_fn(p);
                let actual_s = out_scalar.value_as_f64(pt_idx, 0).unwrap();
                assert!(
                    (actual_s - expected_s).abs() < 1e-10,
                    "{cell_type:?} trial {trial} pt {pt_idx}: scalar {actual_s} != {expected_s}"
                );

                let expected_v = [p[0] - 2.0 * p[1], 3.0 * p[1] + p[2], -p[0] + 0.5 * p[2]];
                for (c, &expected_vc) in expected_v.iter().enumerate() {
                    let actual_vc = out_vec.value_as_f64(pt_idx, c).unwrap();
                    assert!(
                        (actual_vc - expected_vc).abs() < 1e-10,
                        "{cell_type:?} trial {trial} pt {pt_idx} comp {c}: {actual_vc} != {expected_vc}",
                    );
                }
            }
        }
    }
}

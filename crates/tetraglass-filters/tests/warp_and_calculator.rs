//! Integration tests for `warp_by_vector` and `calculator` filters (`M6.4`, `/add-filter`).

use std::path::PathBuf;

use tetraglass_core::{Attributes, ComponentCount, DataArray, TetView, UnstructuredGrid};
use tetraglass_filters::{
    CalculatorAssociation, CalculatorParams, ContourParams, DEFAULT_CALCULATOR_CHUNK_SIZE,
    DEFAULT_WARP_CHUNK_SIZE, FilterError, LinearizeParams, SliceParams, WarpByVectorParams,
    calculator, calculator_grid, calculator_unstructured_grid, calculator_with_chunk_size, contour,
    evaluate_calculator, extract_external_surface, extract_tet_view_surface, linearize,
    linearize_default, slice, warp_by_vector, warp_by_vector_grid, warp_by_vector_with_chunk_size,
    warp_unstructured_grid,
};
use tetraglass_io::read_vtu_from_path;

fn testdata_vtk_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .join("testdata/vtk")
}

#[test]
fn warp_by_vector_analytic_dilation_and_shear_on_hex_linear() {
    let grid =
        read_vtu_from_path(testdata_vtk_dir().join("hex_linear.vtu")).expect("read hex_linear.vtu");
    let view = linearize_default(&grid).expect("linearize hex_linear.vtu");
    assert!((view.total_volume() - 8.0).abs() < 1e-12);

    // 1. Use `calculator` to create an analytic dilation vector field `u = [x, y, z]` (`coords`).
    let with_u = calculator(&view, &CalculatorParams::new("u_dilate", "coords"))
        .expect("compute dilation vector field");

    // Displace by scale_factor = 1.0 -> p' = p + p = 2 * p, scaling volume by 2^3 = 8 (8.0 -> 64.0).
    let dilated = warp_by_vector(&with_u, &WarpByVectorParams::with_scale("u_dilate", 1.0))
        .expect("warp by dilation field");
    assert!((dilated.total_volume() - 64.0).abs() < 1e-11);
    let bounds = dilated.bounds().expect("dilated bounds");
    assert_eq!(bounds.min, [0.0, 0.0, 0.0]);
    assert_eq!(bounds.max, [4.0, 4.0, 4.0]);

    // 2. Divergence-free shear `u_shear = [y, 0, 0]` preserves volume = 8.0 for any scale factor.
    let with_shear = calculator(&view, &CalculatorParams::new("u_shear", "[y, 0.0, 0.0]"))
        .expect("compute shear vector field");
    let sheared = warp_by_vector(&with_shear, &WarpByVectorParams::new("u_shear").scale(1.75))
        .expect("warp by shear field");
    assert!((sheared.total_volume() - 8.0).abs() < 1e-11);

    // 3. Scale factor 0.0 leaves coordinates bit-for-bit identical.
    let unwarped = warp_by_vector(&view, &WarpByVectorParams::with_scale("velocity", 0.0))
        .expect("zero scale warp");
    assert_eq!(unwarped.points(), view.points());
}

#[test]
fn calculator_evaluates_scalars_vectors_tensors_and_cell_expressions() {
    let grid =
        read_vtu_from_path(testdata_vtk_dir().join("hex_linear.vtu")).expect("read hex_linear.vtu");
    let view = linearize_default(&grid).expect("linearize hex_linear.vtu");

    // 1. Analytic residual `x + 2*y + 3*z - linear` must be 0 at every vertex.
    let res_arr = evaluate_calculator(
        &view,
        &CalculatorParams::new("residual", "x + 2.0 * y + 3.0 * z - linear"),
    )
    .expect("evaluate residual");
    assert_eq!(res_arr.components(), ComponentCount::Scalar);
    for i in 0..view.num_points() {
        assert!(res_arr.value_as_f64(i, 0).unwrap().abs() < 1e-12);
    }

    // 2. Vector magnitude, dot product, cross product, and basis vectors (`iHat`, `jHat`, `kHat`).
    // On `hex_linear.vtu`, `velocity = [-y, x, 0]`, so `dot(velocity, coords) = -y*x + x*y = 0`.
    let ortho_arr = evaluate_calculator(
        &view,
        &CalculatorParams::new(
            "ortho",
            "dot(velocity, coords) + mag(velocity) - sqrt(x^2 + y^2)",
        ),
    )
    .expect("evaluate ortho");
    for i in 0..view.num_points() {
        assert!(ortho_arr.value_as_f64(i, 0).unwrap().abs() < 1e-12);
    }

    // 3. Construct vector field via `iHat`, `jHat`, `kHat` and `cross` and compare with `velocity`.
    let swirl_arr = evaluate_calculator(
        &view,
        &CalculatorParams::new("swirl", "cross(kHat, [x, y, 0.0])"),
    )
    .expect("evaluate cross(kHat, [x, y, 0]) = [-y, x, 0]");
    assert_eq!(swirl_arr.components(), ComponentCount::Vector3);
    let vel_arr = view.point_data().get("velocity").unwrap();
    for i in 0..view.num_points() {
        for c in 0..3 {
            let a = swirl_arr.value_as_f64(i, c).unwrap();
            let b = vel_arr.value_as_f64(i, c).unwrap();
            assert!((a - b).abs() < 1e-12);
        }
    }

    // 4. Standard math functions (`sin`, `cos`, `log`, `exp`, `sqrt`, `abs`, `pow`, `min`, `max`, `clamp`, `lerp`).
    let math_arr = evaluate_calculator(
        &view,
        &CalculatorParams::new(
            "math_check",
            "sin(pi / 2) + cos(0) + log(e) + sqrt(16) + abs(-3) + pow(2, 3) + min(4, 9) + max(2, 5) + clamp(10, 0, 7) + lerp(10, 20, 0.5)",
        ),
    )
    .expect("evaluate math functions");
    for i in 0..view.num_points() {
        assert!((math_arr.value_as_f64(i, 0).unwrap() - 49.0).abs() < 1e-12);
    }

    // 5. Component access (`.x`, `.y`, `.z`, `:magnitude`, `.0`, `.1`) and symmetric 6-tensor components.
    let mut view_with_tensor = view.clone();
    let sym_values: Vec<f64> = (0..view.num_points())
        .flat_map(|i| {
            let f = i as f64;
            [f, f + 1.0, f + 2.0, f + 3.0, f + 4.0, f + 5.0]
        })
        .collect();
    view_with_tensor
        .insert_point_data(
            "stress",
            DataArray::from_f64(ComponentCount::SymmetricTensor6, sym_values).unwrap(),
        )
        .unwrap();
    let tensor_trace = evaluate_calculator(
        &view_with_tensor,
        &CalculatorParams::new(
            "trace",
            "stress.xx + stress.yy + stress.zz + stress.xy - stress.3",
        ),
    )
    .expect("evaluate tensor components");
    for i in 0..view.num_points() {
        let f = i as f64;
        let expected = f + (f + 1.0) + (f + 2.0);
        assert!((tensor_trace.value_as_f64(i, 0).unwrap() - expected).abs() < 1e-12);
    }
    let tensor_mag = evaluate_calculator(
        &view_with_tensor,
        &CalculatorParams::new("stress_norm", "mag(stress) - stress.magnitude"),
    )
    .expect("evaluate tensor magnitude");
    for i in 0..view.num_points() {
        assert!(tensor_mag.value_as_f64(i, 0).unwrap().abs() < 1e-12);
    }

    // 6. Cell-data association (`CalculatorAssociation::Cell`).
    let cell_calc = calculator(
        &view,
        &CalculatorParams::cell("cell_metric", "cell_id * 10.0 + x + y + z"),
    )
    .expect("evaluate cell calculator");
    let cell_metric = cell_calc
        .cell_data()
        .get("cell_metric")
        .expect("cell_metric present");
    assert_eq!(cell_metric.num_tuples(), view.num_tets());
}

#[test]
fn warp_and_calculator_work_across_all_testdata_cell_families_and_chain_into_slice_and_contour() {
    let vtk_dir = testdata_vtk_dir();
    for filename in [
        "mixed_linear.vtu",
        "tet_quadratic.vtu",
        "hex_lagrange_p3.vtu",
        "polyhedra.vtu",
    ] {
        let grid = read_vtu_from_path(vtk_dir.join(filename)).unwrap_or_else(|e| {
            panic!("failed to read {filename}: {e}");
        });

        // 1. UnstructuredGrid-level calculator + warp_unstructured_grid + surface extraction
        let grid_with_disp = calculator_unstructured_grid(
            &grid,
            &CalculatorParams::new("disp", "[0.1 * x, 0.05 * y, -0.05 * z]"),
        )
        .expect("grid calculator");
        let warped_grid = warp_unstructured_grid(
            &grid_with_disp,
            &WarpByVectorParams::with_scale("disp", 1.0),
        )
        .expect("warp unstructured grid");
        let ext_surface = extract_external_surface(&warped_grid).expect("extract surface");
        assert!(
            ext_surface.num_triangles() > 0,
            "{filename}: expected external surface triangles"
        );

        // 2. Linearized TetView pipeline (`D-005`): calculator -> warp_by_vector -> slice & contour
        let lin_params = LinearizeParams::default();
        let view_with_disp = calculator_grid(
            &grid,
            &CalculatorParams::new("disp", "[0.1 * x, 0.1 * y, 0.1 * z]"),
            &lin_params,
        )
        .expect("calculator_grid");
        let view_with_scalar = calculator(
            &view_with_disp,
            &CalculatorParams::new("warped_scalar", "linear + 0.5 * mag(disp)"),
        )
        .expect("derived scalar");
        let warped_view = warp_by_vector(
            &view_with_scalar,
            &WarpByVectorParams::with_scale("disp", 1.0),
        )
        .expect("warp_by_vector on TetView");

        // Uniform 1.1x scaling in all 3 axes scales volume by 1.1^3 = 1.331
        let orig_vol = view_with_disp.total_volume();
        let new_vol = warped_view.total_volume();
        assert!(
            (new_vol - orig_vol * 1.331).abs() < 1e-9,
            "{filename}: expected volume {} * 1.331, got {new_vol}",
            orig_vol
        );

        let cut = slice(&warped_view, &SliceParams::new([0.0, 0.0, 1.0])).expect("slice warped");
        assert!(
            cut.num_triangles() > 0,
            "{filename}: expected slice triangles"
        );
        assert!(cut.point_data().get("warped_scalar").is_some());

        let range = warped_view
            .point_data()
            .get("warped_scalar")
            .unwrap()
            .component_range(0)
            .unwrap();
        let mid_iso = 0.5 * (range.min + range.max);
        let iso = contour(
            &warped_view,
            &ContourParams::new("warped_scalar", vec![mid_iso]),
        )
        .expect("contour warped");
        assert!(
            iso.num_triangles() > 0,
            "{filename}: expected contour triangles"
        );

        // Also verify `warp_by_vector_grid` helper directly on datasets that have `velocity`
        if grid.point_data().get("velocity").is_some() {
            let wv = warp_by_vector_grid(
                &grid,
                &WarpByVectorParams::with_scale("velocity", 0.1),
                &lin_params,
            )
            .expect("warp_by_vector_grid");
            let surf = extract_tet_view_surface(&wv).expect("extract_tet_view_surface");
            assert!(surf.num_triangles() > 0);
        }
    }
}

#[test]
fn warp_and_calculator_are_deterministic_across_chunk_sizes_and_handle_empty_views() {
    let grid = read_vtu_from_path(testdata_vtk_dir().join("hex_lagrange_p3.vtu"))
        .expect("read hex_lagrange_p3.vtu");
    let view = linearize(&grid, &LinearizeParams::new(2)).expect("linearize");

    let calc_params = CalculatorParams::new(
        " swirl_vec ",
        "[-y + 0.25 * sin(x), x + 0.25 * cos(y), 0.5 * z]",
    );
    let calc_small = calculator_with_chunk_size(&view, &calc_params, 1).expect("chunk 1");
    let calc_med = calculator_with_chunk_size(&view, &calc_params, 7).expect("chunk 7");
    let calc_def = calculator_with_chunk_size(&view, &calc_params, DEFAULT_CALCULATOR_CHUNK_SIZE)
        .expect("default chunk");
    assert_eq!(calc_small, calc_med);
    assert_eq!(calc_med, calc_def);

    let warp_params = WarpByVectorParams::with_scale("swirl_vec", 0.15);
    let warp_small = warp_by_vector_with_chunk_size(&calc_def, &warp_params, 1).expect("chunk 1");
    let warp_med = warp_by_vector_with_chunk_size(&calc_def, &warp_params, 7).expect("chunk 7");
    let warp_def = warp_by_vector_with_chunk_size(&calc_def, &warp_params, DEFAULT_WARP_CHUNK_SIZE)
        .expect("default chunk");
    assert_eq!(warp_small, warp_med);
    assert_eq!(warp_med, warp_def);

    // Empty TetView and UnstructuredGrid degenerate cases
    let mut empty_pt_attrs = Attributes::new(0);
    empty_pt_attrs
        .insert(
            "velocity",
            DataArray::from_f64(ComponentCount::Vector3, Vec::new()).unwrap(),
        )
        .unwrap();
    let empty_view = TetView::new(
        Vec::<[f64; 3]>::new(),
        Vec::<[u32; 4]>::new(),
        Vec::<u32>::new(),
        empty_pt_attrs,
        Attributes::new(0),
    )
    .unwrap();
    let empty_calc = calculator(
        &empty_view,
        &CalculatorParams::new("speed", "mag(velocity)"),
    )
    .expect("empty calculator");
    assert_eq!(empty_calc.num_points(), 0);
    let empty_warped = warp_by_vector(&empty_calc, &WarpByVectorParams::new("velocity"))
        .expect("empty warp_by_vector");
    assert_eq!(empty_warped.num_points(), 0);

    let mut empty_grid = UnstructuredGrid::new(
        Vec::<[f64; 3]>::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        None,
    )
    .unwrap();
    empty_grid
        .insert_point_data(
            "velocity",
            DataArray::from_f64(ComponentCount::Vector3, Vec::new()).unwrap(),
        )
        .unwrap();
    let empty_grid_warped =
        warp_unstructured_grid(&empty_grid, &WarpByVectorParams::new("velocity"))
            .expect("empty grid warp");
    assert_eq!(empty_grid_warped.num_points(), 0);
}

#[test]
fn warp_and_calculator_validate_parameters_and_expressions() {
    let grid =
        read_vtu_from_path(testdata_vtk_dir().join("hex_linear.vtu")).expect("read hex_linear.vtu");
    let view = linearize_default(&grid).expect("linearize");

    // 1. WarpByVector error cases
    assert!(matches!(
        warp_by_vector(&view, &WarpByVectorParams::with_scale("velocity", f64::NAN)),
        Err(FilterError::InvalidWarpScaleFactor { .. })
    ));
    assert!(matches!(
        warp_by_vector(
            &view,
            &WarpByVectorParams::with_scale("velocity", f64::INFINITY)
        ),
        Err(FilterError::InvalidWarpScaleFactor { .. })
    ));
    assert_eq!(
        warp_by_vector(&view, &WarpByVectorParams::new("nonexistent")),
        Err(FilterError::MissingPointArray {
            name: "nonexistent".to_string()
        })
    );
    assert_eq!(
        warp_by_vector(&view, &WarpByVectorParams::new("linear")),
        Err(FilterError::NonVectorArray {
            array: "linear".to_string(),
            num_components: 1
        })
    );

    // 2. Calculator error cases
    assert_eq!(
        calculator(&view, &CalculatorParams::new("   ", "x + y")),
        Err(FilterError::InvalidCalculatorResultName)
    );
    assert_eq!(
        calculator(&view, &CalculatorParams::new("out", "missing_arr + 1")),
        Err(FilterError::MissingPointArray {
            name: "missing_arr".to_string()
        })
    );
    assert_eq!(
        calculator(
            &view,
            &CalculatorParams::cell("out", "missing_cell_arr + 1")
                .with_association(CalculatorAssociation::Cell)
        ),
        Err(FilterError::MissingCellArray {
            name: "missing_cell_arr".to_string()
        })
    );
    assert_eq!(
        calculator(&view, &CalculatorParams::new("out", "velocity.5")),
        Err(FilterError::InvalidComponentIndex {
            array: "velocity".to_string(),
            component: 5,
            num_components: 3
        })
    );

    for bad_expr in [
        "",
        "   ",
        "1 + * 2",
        "(x + y",
        "[x, y]",
        "[x, y, z",
        "unknown_fn(x)",
        "sin(x, y)",
        "pow(x)",
        "dot(x, y)",
        "cross(velocity, x)",
        "normalize(x)",
        "velocity.unknown_comp",
        "coords.w",
        "1e+",
        "x @ y",
    ] {
        assert!(
            matches!(
                calculator(&view, &CalculatorParams::new("out", bad_expr)),
                Err(FilterError::InvalidCalculatorExpression { .. })
            ),
            "expected InvalidCalculatorExpression for `{bad_expr}`"
        );
    }
}

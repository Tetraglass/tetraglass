//! Integration tests for the legacy VTK DataFile (`.vtk`) reader (`M5.1`).
#![cfg(not(target_arch = "wasm32"))]

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use tetraglass_core::{CellType, ComponentCount, ScalarType, UnstructuredGrid};
use tetraglass_io::{
    MeshFormat, detect_format_from_bytes, detect_format_from_extension,
    read_pvd_series_with_loader, read_series_from_path, read_vtk, read_vtk_from_path,
    read_vtk_series, read_vtu, read_vtu_from_path, write_vtu_to_bytes,
};

fn vtk_testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ directory")
        .parent()
        .expect("workspace root")
        .join("testdata")
        .join("vtk")
}

fn assert_grids_equivalent(a: &UnstructuredGrid, b: &UnstructuredGrid) {
    assert_eq!(a.num_points(), b.num_points());
    assert_eq!(a.num_cells(), b.num_cells());
    assert_eq!(a.cell_types(), b.cell_types());
    assert_eq!(a.offsets(), b.offsets());
    assert_eq!(a.connectivity(), b.connectivity());
    assert_eq!(a.polyhedron_faces(), b.polyhedron_faces());

    for (pa, pb) in a.points().iter().zip(b.points().iter()) {
        for k in 0..3 {
            assert!(
                (pa[k] - pb[k]).abs() <= 1e-9,
                "point coordinate mismatch: {pa:?} vs {pb:?}"
            );
        }
    }

    assert_eq!(a.point_data().len(), b.point_data().len());
    for (name, arr_a) in a.point_data().iter() {
        let arr_b = b
            .point_data()
            .get(name)
            .unwrap_or_else(|| panic!("missing point array `{name}`"));
        assert_eq!(arr_a.components(), arr_b.components());
        assert_eq!(arr_a.num_tuples(), arr_b.num_tuples());
        for c in 0..arr_a.components().as_usize() {
            let ra = arr_a.component_range(c).unwrap();
            let rb = arr_b.component_range(c).unwrap();
            assert!((ra.min - rb.min).abs() <= 1e-7);
            assert!((ra.max - rb.max).abs() <= 1e-7);
        }
    }

    assert_eq!(a.cell_data().len(), b.cell_data().len());
    for (name, arr_a) in a.cell_data().iter() {
        let arr_b = b
            .cell_data()
            .get(name)
            .unwrap_or_else(|| panic!("missing cell array `{name}`"));
        assert_eq!(arr_a.components(), arr_b.components());
        assert_eq!(arr_a.num_tuples(), arr_b.num_tuples());
    }
}

/// Converts any loaded [`UnstructuredGrid`] into an ASCII Legacy VTK 4.2 `.vtk` string
/// (using classic `CELLS` and `SCALARS`/`VECTORS`/`FIELD` sections) for round-trip testing.
fn write_grid_to_legacy_vtk42_ascii(grid: &UnstructuredGrid) -> String {
    let mut out = String::new();
    writeln!(out, "# vtk DataFile Version 4.2").unwrap();
    writeln!(out, "Tetraglass test mesh").unwrap();
    writeln!(out, "ASCII").unwrap();
    writeln!(out, "DATASET UNSTRUCTURED_GRID").unwrap();
    writeln!(out, "POINTS {} double", grid.num_points()).unwrap();
    for &[x, y, z] in grid.points() {
        writeln!(out, "{x:.16} {y:.16} {z:.16}").unwrap();
    }

    let mut cell_entries: Vec<Vec<u32>> = Vec::with_capacity(grid.num_cells());
    let mut total_ints = 0usize;
    for cell_idx in 0..grid.num_cells() {
        let ct = grid.cell_types()[cell_idx];
        let entry = if ct == CellType::Polyhedron {
            let pf = grid
                .polyhedron_faces()
                .expect("polyhedron grid has face stream");
            pf.cell_stream(cell_idx)
                .expect("polyhedron cell stream")
                .to_vec()
        } else {
            grid.cell_connectivity(cell_idx)
                .expect("cell connectivity")
                .to_vec()
        };
        total_ints += 1 + entry.len();
        cell_entries.push(entry);
    }

    writeln!(out, "CELLS {} {total_ints}", grid.num_cells()).unwrap();
    for entry in &cell_entries {
        write!(out, "{}", entry.len()).unwrap();
        for &v in entry {
            write!(out, " {v}").unwrap();
        }
        writeln!(out).unwrap();
    }

    writeln!(out, "CELL_TYPES {}", grid.num_cells()).unwrap();
    for &ct in grid.cell_types() {
        writeln!(out, "{}", ct.vtk_type_id()).unwrap();
    }

    if !grid.point_data().is_empty() {
        writeln!(out, "POINT_DATA {}", grid.num_points()).unwrap();
        writeln!(out, "FIELD FieldData {}", grid.point_data().len()).unwrap();
        for (name, arr) in grid.point_data().iter() {
            write_legacy_field_array(&mut out, name, arr);
        }
    }

    if !grid.cell_data().is_empty() {
        writeln!(out, "CELL_DATA {}", grid.num_cells()).unwrap();
        writeln!(out, "FIELD FieldData {}", grid.cell_data().len()).unwrap();
        for (name, arr) in grid.cell_data().iter() {
            write_legacy_field_array(&mut out, name, arr);
        }
    }

    out
}

fn write_legacy_field_array(out: &mut String, name: &str, arr: &tetraglass_core::DataArray) {
    let dtype = match arr.scalar_type() {
        ScalarType::F32 => "float",
        ScalarType::F64 => "double",
        ScalarType::I32 => "int",
        ScalarType::I64 => "vtktypeint64",
        ScalarType::U8 => "unsigned_char",
    };
    writeln!(
        out,
        "{name} {} {} {dtype}",
        arr.components().as_usize(),
        arr.num_tuples()
    )
    .unwrap();
    for tuple_i in 0..arr.num_tuples() {
        for comp_i in 0..arr.components().as_usize() {
            let v = arr.value_as_f64(tuple_i, comp_i).unwrap();
            if comp_i > 0 {
                write!(out, " ").unwrap();
            }
            if matches!(
                arr.scalar_type(),
                ScalarType::I32 | ScalarType::I64 | ScalarType::U8
            ) {
                write!(out, "{}", v as i64).unwrap();
            } else {
                write!(out, "{v:.16}").unwrap();
            }
        }
        writeln!(out).unwrap();
    }
}

#[test]
fn committed_legacy_vtk_42_and_51_ascii_and_binary_match_vtu_and_roundtrip() {
    let dir = vtk_testdata_dir();
    let expected = read_vtu_from_path(dir.join("tet_linear.vtu")).expect("load tet_linear.vtu");

    for file in [
        "tet_linear_legacy.vtk",
        "tet_linear_legacy_binary.vtk",
        "tet_linear_v51.vtk",
        "tet_linear_v51_binary.vtk",
    ] {
        let path = dir.join(file);
        let raw_bytes = fs::read(&path).expect("read .vtk file");
        assert_eq!(
            detect_format_from_extension("vtk"),
            Some(MeshFormat::VtkLegacy)
        );
        assert_eq!(
            detect_format_from_bytes(&raw_bytes),
            Some(MeshFormat::VtkLegacy)
        );

        let grid = read_vtk_from_path(&path)
            .unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
        assert_grids_equivalent(&grid, &expected);

        // Round-trip through the `.vtu` writer.
        let vtu_bytes = write_vtu_to_bytes(&grid).expect("write vtu");
        let roundtripped = read_vtu(&vtu_bytes).expect("read roundtripped vtu");
        assert_grids_equivalent(&roundtripped, &expected);

        // Verify `read_series_from_path` and `read_vtk_series`.
        let (fmt, series) = read_series_from_path(&path, None).expect("read_series_from_path");
        assert_eq!(fmt, MeshFormat::VtkLegacy);
        assert_eq!(series.len(), 1);
        assert_eq!(series.load_executions(), 0, "series must be lazy");
        let step0 = series.step(0).expect("load step 0");
        assert_eq!(series.load_executions(), 1);
        assert_grids_equivalent(&step0, &expected);
    }
}

#[test]
fn legacy_vtk_supports_all_cell_families_higher_order_node_ordering_and_polyhedra() {
    let dir = vtk_testdata_dir();
    for vtu_name in [
        "hex_linear.vtu",
        "mixed_linear.vtu",
        "tet_quadratic.vtu",
        "hex_lagrange_p3.vtu",
        "polyhedra.vtu",
    ] {
        let vtu_grid = read_vtu_from_path(dir.join(vtu_name)).expect("read reference vtu");
        let legacy_ascii = write_grid_to_legacy_vtk42_ascii(&vtu_grid);
        let vtk_grid = read_vtk(legacy_ascii.as_bytes())
            .unwrap_or_else(|e| panic!("failed to parse legacy .vtk for {vtu_name}: {e}"));
        assert_grids_equivalent(&vtk_grid, &vtu_grid);

        // Verify exact linear field `x + 2y + 3z` at every node (including mid-edge/face/interior Lagrange nodes).
        let lin = vtk_grid.point_data().get("linear").expect("linear array");
        for (pt_idx, &[x, y, z]) in vtk_grid.points().iter().enumerate() {
            let actual = lin.value_as_f64(pt_idx, 0).unwrap();
            let expected = x + 2.0 * y + 3.0 * z;
            assert!(
                (actual - expected).abs() < 1e-11,
                "{vtu_name} pt {pt_idx}: {actual} != {expected}"
            );
        }

        // Round-trip through `.vtu` writer.
        let vtu_bytes = write_vtu_to_bytes(&vtk_grid).expect("write vtu");
        let rt = read_vtu(&vtu_bytes).expect("read vtu");
        assert_grids_equivalent(&rt, &vtu_grid);
    }
}

#[test]
fn legacy_vtk_parses_scalars_vectors_normals_tensors_metadata_and_pvd_steps() {
    let vtk_text = "\
# vtk DataFile Version 5.1
Rich attribute and FACES polyhedron test
ASCII
DATASET UNSTRUCTURED_GRID
FIELD FieldData 1
TIME 1 1 double
2.5
POINTS 4 float
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
METADATA
INFORMATION 1
NAME L2_NORM_RANGE LOCATION vtkDataArray
DATA 2 0 1

CELLS 2 4
OFFSETS vtktypeint64
0 4
CONNECTIVITY vtktypeint64
0 1 2 3
FACES 5 12
OFFSETS vtktypeint64
0 3 6 9 12
CONNECTIVITY vtktypeint64
0 2 1 0 1 3 1 2 3 0 3 2
FACE_LOCATIONS 2 4
OFFSETS vtktypeint64
0 4
CONNECTIVITY vtktypeint64
0 1 2 3
CELL_TYPES 1
42
POINT_DATA 4
SCALARS temperature float 1
LOOKUP_TABLE default
10.0 20.0 30.0 40.0
SCALARS flag unsigned_char
1 2 3 4
VECTORS vel double
1.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 3.0
1.0 1.0 1.0
NORMALS norm float
0.0 0.0 1.0
0.0 0.0 1.0
0.0 0.0 1.0
0.0 0.0 1.0
TENSORS6 sym_stress double
1 2 3 4 5 6
1 2 3 4 5 6
1 2 3 4 5 6
1 2 3 4 5 6
TENSORS full_tensor float
1 0 0 0 1 0 0 0 1
1 0 0 0 1 0 0 0 1
1 0 0 0 1 0 0 0 1
1 0 0 0 1 0 0 0 1
CELL_DATA 1
SCALARS region_id int
LOOKUP_TABLE custom_lut
42
LOOKUP_TABLE custom_lut 1
1.0 0.0 0.0 1.0
";

    let series = read_vtk_series(vtk_text.as_bytes().to_vec()).expect("read_vtk_series");
    assert_eq!(series.times(), &[2.5]);
    let grid = series.step(0).expect("load step 0");
    assert_eq!(grid.num_points(), 4);
    assert_eq!(grid.num_cells(), 1);
    assert_eq!(grid.cell_types(), &[CellType::Polyhedron]);
    assert_eq!(grid.polyhedron_cell_faces(0).unwrap().count(), 4);

    assert_eq!(
        grid.point_data().get("temperature").unwrap().components(),
        ComponentCount::Scalar
    );
    assert_eq!(
        grid.point_data().get("flag").unwrap().scalar_type(),
        ScalarType::U8
    );
    assert_eq!(
        grid.point_data().get("vel").unwrap().components(),
        ComponentCount::Vector3
    );
    assert_eq!(
        grid.point_data().get("norm").unwrap().components(),
        ComponentCount::Vector3
    );
    assert_eq!(
        grid.point_data().get("sym_stress").unwrap().components(),
        ComponentCount::SymmetricTensor6
    );
    assert_eq!(
        grid.point_data().get("full_tensor").unwrap().components(),
        ComponentCount::Tensor9
    );
    assert_eq!(
        grid.cell_data()
            .get("region_id")
            .unwrap()
            .value_as_f64(0, 0),
        Some(42.0)
    );

    // Also verify `.pvd` referencing `.vtk` step files.
    let pvd_xml = br#"<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1">
  <Collection>
    <DataSet timestep="0.0" file="step_0.vtk"/>
  </Collection>
</VTKFile>"#;
    let pvd_series = read_pvd_series_with_loader(pvd_xml, move |file| {
        assert_eq!(file, "step_0.vtk");
        Ok(vtk_text.as_bytes().to_vec())
    })
    .expect("read pvd with .vtk step");
    let step_grid = pvd_series.step(0).expect("load .vtk step from .pvd");
    assert_eq!(step_grid.num_points(), 4);
    assert_eq!(step_grid.num_cells(), 1);
}

#[test]
fn truncated_and_corrupted_legacy_vtk_inputs_return_errors_without_panicking() {
    let dir = vtk_testdata_dir();
    for file in [
        "tet_linear_legacy.vtk",
        "tet_linear_legacy_binary.vtk",
        "tet_linear_v51.vtk",
        "tet_linear_v51_binary.vtk",
    ] {
        let bytes = fs::read(dir.join(file)).expect("read .vtk");
        // Truncate at every 17th byte offset and verify no panic occurs.
        for cut in (0..bytes.len()).step_by(17) {
            let _ = read_vtk(&bytes[..cut]);
        }
        // Truncating inside the geometry section must return an error.
        assert!(read_vtk(&bytes[..80]).is_err());
        assert!(read_vtk(&bytes[..180]).is_err());
    }

    // Malformed / unsupported inputs must return clear errors.
    for bad in [
        "",
        "not a vtk file\n",
        "# vtk DataFile Version 4.2\ntitle\nXML\nDATASET UNSTRUCTURED_GRID\n",
        "# vtk DataFile Version 4.2\ntitle\nASCII\nDATASET POLYDATA\n",
        "# vtk DataFile Version 4.2\ntitle\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 2 double\n0 0 0 1 0 0\nCELLS 1 5\n4 0 1 2 99\nCELL_TYPES 1\n10\n",
        "# vtk DataFile Version 4.2\ntitle\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 4 double\n0 0 0 1 0 0 0 1 0 0 0 1\nCELLS 1 5\n4 0 1 2 3\nCELL_TYPES 1\n250\n",
        "# vtk DataFile Version 4.2\ntitle\nASCII\nDATASET UNSTRUCTURED_GRID\nSCALARS orphan float\nLOOKUP_TABLE default\n1.0\n",
    ] {
        assert!(
            read_vtk(bad.as_bytes()).is_err(),
            "expected error for malformed input: {bad:?}"
        );
    }
}

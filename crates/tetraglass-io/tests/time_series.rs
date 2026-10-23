//! Integration, lazy-loading, round-trip, and corruption tests for `.pvd` and multi-step `.msh` time series (`M1.7`).
#![cfg(not(target_arch = "wasm32"))]

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tetraglass_core::ComponentCount;
use tetraglass_io::{
    MeshFormat, detect_format_from_bytes, detect_format_from_extension, parse_pvd_entries,
    read_msh_series, read_pvd_from_path, read_pvd_series_from_path, read_pvd_series_with_loader,
    read_series_from_path, read_vtu, write_vtu_to_bytes,
};

fn vtk_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ directory")
        .parent()
        .expect("workspace root")
        .join("testdata")
        .join("vtk")
}

#[test]
fn pvd_series_parses_lazily_and_loads_only_requested_step() {
    let dir = vtk_dir();
    let pvd_path = dir.join("series.pvd");
    let pvd_bytes = fs::read(&pvd_path).expect("read series.pvd");

    assert_eq!(detect_format_from_bytes(&pvd_bytes), Some(MeshFormat::Pvd));
    assert_eq!(detect_format_from_extension("pvd"), Some(MeshFormat::Pvd));

    let entries = parse_pvd_entries(&pvd_bytes).expect("parse series.pvd entries");
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].timestep, 0.0);
    assert_eq!(entries[0].file, "series_0.vtu");
    assert_eq!(entries[1].timestep, 0.5);
    assert_eq!(entries[1].file, "series_1.vtu");
    assert_eq!(entries[2].timestep, 1.0);
    assert_eq!(entries[2].file, "series_2.vtu");

    // Track which underlying .vtu files are actually read from disk.
    let loaded_files: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let loaded_files_cb = Arc::clone(&loaded_files);
    let base_dir = dir.clone();

    let series = read_pvd_series_with_loader(&pvd_bytes, move |rel_file| {
        loaded_files_cb.lock().unwrap().push(rel_file.to_string());
        Ok(fs::read(base_dir.join(rel_file))?)
    })
    .expect("construct lazy PVD TimeSeries");

    assert_eq!(series.len(), 3);
    assert_eq!(series.times(), &[0.0, 0.5, 1.0]);
    assert_eq!(series.load_executions(), 0);
    assert!(loaded_files.lock().unwrap().is_empty());

    // Request ONLY step 2 (`t = 1.0`): only `series_2.vtu` must be read and parsed.
    let step2 = series.step(2).expect("load step 2");
    assert_eq!(series.load_executions(), 1);
    assert_eq!(&*loaded_files.lock().unwrap(), &["series_2.vtu"]);
    assert!(series.is_step_cached(2));
    assert!(!series.is_step_cached(0));
    assert!(!series.is_step_cached(1));

    // Re-requesting step 2 hits the cache without reading the file again.
    let step2_cached = series.step(2).expect("load step 2 from cache");
    assert!(Arc::ptr_eq(&step2, &step2_cached));
    assert_eq!(series.load_executions(), 1);
    assert_eq!(series.cache_hits(), 1);
    assert_eq!(&*loaded_files.lock().unwrap(), &["series_2.vtu"]);

    // Verify all 3 steps via `read_pvd_series_from_path` and round-trip each step through `write_vtu`.
    let path_series = read_pvd_series_from_path(&pvd_path).expect("read_pvd_series_from_path");
    for (step_idx, &expected_t) in [0.0, 0.5, 1.0].iter().enumerate() {
        let grid = path_series.step(step_idx).expect("load step");
        assert_eq!(grid.num_points(), 8);
        assert_eq!(grid.num_cells(), 6);

        let transient = grid
            .point_data()
            .get("transient")
            .expect("missing 'transient' array");
        assert_eq!(transient.components(), ComponentCount::Scalar);

        for (pt_idx, &[x, y, z]) in grid.points().iter().enumerate() {
            let expected_val = x + 2.0 * y + 3.0 * z + expected_t;
            let actual_val = transient.value_as_f64(pt_idx, 0).unwrap();
            assert!(
                (actual_val - expected_val).abs() < 1e-11,
                "step {step_idx} pt {pt_idx}: {actual_val} != {expected_val}"
            );
        }

        let vtu_bytes = write_vtu_to_bytes(&grid).expect("write_vtu_to_bytes");
        let roundtripped = read_vtu(&vtu_bytes).expect("read_vtu");
        assert_eq!(*grid, roundtripped);
    }

    // Convenience helpers `read_pvd_from_path` and `read_series_from_path`
    let first_grid = read_pvd_from_path(&pvd_path).expect("read_pvd_from_path");
    assert_eq!(first_grid, *path_series.step(0).unwrap());

    let (fmt, unified) = read_series_from_path(&pvd_path, None).expect("read_series_from_path");
    assert_eq!(fmt, MeshFormat::Pvd);
    assert_eq!(unified.times(), &[0.0, 0.5, 1.0]);
}

#[test]
fn multi_step_gmsh_ascii_and_binary_load_lazily_and_match() {
    let times: [f64; 3] = [0.0, 0.5, 1.0];

    // Build a 1-tetrahedron MSH 4.1 mesh with 3 time steps of `$NodeData` ("transient")
    // and 1 static `$ElementData` ("cell_id") in both ASCII and binary formats.
    let mut ascii = String::new();
    ascii.push_str("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n");
    ascii.push_str("$Nodes\n1 4 1 4\n3 1 0 4\n1\n2\n3\n4\n");
    ascii.push_str("0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\n$EndNodes\n");
    ascii.push_str("$Elements\n1 1 1 1\n3 1 4 1\n1 1 2 3 4\n$EndElements\n");

    let pts: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    for (step_idx, &t) in times.iter().enumerate() {
        ascii.push_str(&format!(
            "$NodeData\n1\n\"transient\"\n1\n{t}\n3\n{step_idx}\n1\n4\n"
        ));
        for (k, &[x, y, z]) in pts.iter().enumerate() {
            let val = x + 2.0 * y + 3.0 * z + t;
            ascii.push_str(&format!("{} {val}\n", k + 1));
        }
        ascii.push_str("$EndNodeData\n");
    }
    ascii.push_str("$ElementData\n1\n\"cell_id\"\n1\n0.0\n3\n0\n1\n1\n1 7.0\n$EndElementData\n");

    // Build matching MSH 2.2 binary multi-step buffer
    let mut binary: Vec<u8> = Vec::new();
    binary.extend_from_slice(b"$MeshFormat\n2.2 1 8\n");
    binary.extend_from_slice(&1i32.to_le_bytes());
    binary.extend_from_slice(b"\n$EndMeshFormat\n$Nodes\n4\n");
    for (k, &[x, y, z]) in pts.iter().enumerate() {
        binary.extend_from_slice(&((k as i32) + 1).to_le_bytes());
        binary.extend_from_slice(&x.to_le_bytes());
        binary.extend_from_slice(&y.to_le_bytes());
        binary.extend_from_slice(&z.to_le_bytes());
    }
    binary.extend_from_slice(b"\n$EndNodes\n$Elements\n1\n");
    binary.extend_from_slice(&4i32.to_le_bytes()); // elm_type = 4 (tet)
    binary.extend_from_slice(&1i32.to_le_bytes()); // num_elts = 1
    binary.extend_from_slice(&0i32.to_le_bytes()); // num_tags = 0
    for v in [1i32, 1, 2, 3, 4] {
        binary.extend_from_slice(&v.to_le_bytes());
    }
    binary.extend_from_slice(b"\n$EndElements\n");
    for (step_idx, &t) in times.iter().enumerate() {
        binary.extend_from_slice(
            format!("$NodeData\n1\n\"transient\"\n1\n{t}\n3\n{step_idx}\n1\n4\n").as_bytes(),
        );
        for (k, &[x, y, z]) in pts.iter().enumerate() {
            let val = x + 2.0 * y + 3.0 * z + t;
            binary.extend_from_slice(&((k as i32) + 1).to_le_bytes());
            binary.extend_from_slice(&val.to_le_bytes());
        }
        binary.extend_from_slice(b"\n$EndNodeData\n");
    }
    binary.extend_from_slice(b"$ElementData\n1\n\"cell_id\"\n1\n0.0\n3\n0\n1\n1\n");
    binary.extend_from_slice(&1i32.to_le_bytes());
    binary.extend_from_slice(&7.0f64.to_le_bytes());
    binary.extend_from_slice(b"\n$EndElementData\n");

    let ascii_series = read_msh_series(ascii.into_bytes()).expect("parse multi-step ASCII MSH");
    let binary_series = read_msh_series(binary).expect("parse multi-step binary MSH");

    assert_eq!(ascii_series.len(), 3);
    assert_eq!(ascii_series.times(), &[0.0, 0.5, 1.0]);
    assert_eq!(ascii_series.load_executions(), 0);
    assert_eq!(binary_series.len(), 3);
    assert_eq!(binary_series.times(), &[0.0, 0.5, 1.0]);
    assert_eq!(binary_series.load_executions(), 0);

    // Load step 1 first: only 1 step execution occurs.
    let g1_ascii = ascii_series.step(1).expect("step 1 ascii");
    assert_eq!(g1_ascii.num_points(), 4);
    assert_eq!(ascii_series.load_executions(), 1);
    assert!(ascii_series.is_step_cached(1));
    assert!(!ascii_series.is_step_cached(0));

    for (step_idx, &t) in times.iter().enumerate() {
        let ga = ascii_series.step(step_idx).unwrap();
        let gb = binary_series.step(step_idx).unwrap();
        assert_eq!(*ga, *gb, "step {step_idx} ascii != binary");
        assert_eq!(
            ga.cell_data().get("cell_id").unwrap().value_as_f64(0, 0),
            Some(7.0)
        );
        assert_eq!(
            ga.point_data().get("transient").unwrap().value_as_f64(0, 0),
            Some(t)
        );
    }
}

#[test]
fn truncated_and_corrupted_pvd_inputs_return_errors_without_panicking() {
    let pvd_bytes = fs::read(vtk_dir().join("series.pvd")).unwrap();
    let len = pvd_bytes.len();
    for cut in [0, 1, 10, len / 4, len / 2, (len * 3) / 4, len - 5] {
        assert!(
            parse_pvd_entries(&pvd_bytes[..cut]).is_err(),
            "series.pvd truncated to {cut} bytes unexpectedly succeeded"
        );
    }

    // Wrong VTKFile type
    let wrong_type = br#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid"><Collection><DataSet timestep="0" file="a.vtu"/></Collection></VTKFile>"#;
    assert!(parse_pvd_entries(wrong_type).is_err());

    // Empty Collection
    let empty_col = br#"<?xml version="1.0"?>
<VTKFile type="Collection"><Collection/></VTKFile>"#;
    assert!(parse_pvd_entries(empty_col).is_err());

    // Missing file attribute
    let missing_file = br#"<?xml version="1.0"?>
<VTKFile type="Collection"><Collection><DataSet timestep="0.0"/></Collection></VTKFile>"#;
    assert!(parse_pvd_entries(missing_file).is_err());

    // Multi-part dataset (part > 0) rejected per D-001
    let multi_part = br#"<?xml version="1.0"?>
<VTKFile type="Collection"><Collection><DataSet timestep="0.0" part="1" file="a.vtu"/></Collection></VTKFile>"#;
    assert!(parse_pvd_entries(multi_part).is_err());
}

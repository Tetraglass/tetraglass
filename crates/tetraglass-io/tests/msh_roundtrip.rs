//! Integration, round-trip, node-ordering, and corruption tests for the Gmsh `.msh` reader.
#![cfg(not(target_arch = "wasm32"))]

use std::fs;
use std::path::PathBuf;

use tetraglass_core::{CellType, ComponentCount};
use tetraglass_io::{
    MeshFormat, detect_format_from_bytes, detect_format_from_extension, gmsh_element_spec,
    read_msh, read_msh_from_path, read_vtu, write_vtu_to_bytes,
};

fn gmsh_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ directory")
        .parent()
        .expect("workspace root")
        .join("testdata")
        .join("gmsh")
}

const ALL_MSH_FILES: [(&str, usize, usize, CellType); 4] = [
    ("tet2_v41.msh", 27, 6, CellType::QuadraticTetra),
    ("tet2_v41_binary.msh", 27, 6, CellType::QuadraticTetra),
    ("tet_v22.msh", 8, 6, CellType::Tetra),
    ("tet_v22_binary.msh", 8, 6, CellType::Tetra),
];

#[test]
fn load_and_round_trip_all_msh_test_files() {
    let dir = gmsh_dir();
    for (name, expected_points, expected_cells, expected_cell_type) in ALL_MSH_FILES {
        let path = dir.join(name);
        let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
        assert_eq!(detect_format_from_bytes(&bytes), Some(MeshFormat::Gmsh));
        assert_eq!(detect_format_from_extension("msh"), Some(MeshFormat::Gmsh));

        let grid = read_msh_from_path(&path).unwrap_or_else(|e| panic!("parse {name}: {e}"));
        assert_eq!(grid.num_points(), expected_points, "{name} point count");
        assert_eq!(grid.num_cells(), expected_cells, "{name} cell count");
        for &ct in grid.cell_types() {
            assert_eq!(ct, expected_cell_type, "{name} cell type");
        }

        // Verify standard point arrays: linear, radius, velocity
        let linear = grid
            .point_data()
            .get("linear")
            .unwrap_or_else(|| panic!("{name}: missing 'linear'"));
        assert_eq!(linear.components(), ComponentCount::Scalar);

        let radius = grid
            .point_data()
            .get("radius")
            .unwrap_or_else(|| panic!("{name}: missing 'radius'"));
        assert_eq!(radius.components(), ComponentCount::Scalar);

        let velocity = grid
            .point_data()
            .get("velocity")
            .unwrap_or_else(|| panic!("{name}: missing 'velocity'"));
        assert_eq!(velocity.components(), ComponentCount::Vector3);

        for (pt_idx, &[x, y, z]) in grid.points().iter().enumerate() {
            let expected_lin = x + 2.0 * y + 3.0 * z;
            let actual_lin = linear.value_as_f64(pt_idx, 0).unwrap();
            assert!(
                (actual_lin - expected_lin).abs() < 1e-11,
                "{name} pt {pt_idx}: linear {actual_lin} != {expected_lin}"
            );

            let expected_rad = ((x - 0.5).powi(2) + (y - 0.5).powi(2) + (z - 0.5).powi(2)).sqrt();
            let actual_rad = radius.value_as_f64(pt_idx, 0).unwrap();
            assert!(
                (actual_rad - expected_rad).abs() < 1e-11,
                "{name} pt {pt_idx}: radius {actual_rad} != {expected_rad}"
            );

            assert!(
                (velocity.value_as_f64(pt_idx, 0).unwrap() - (-y)).abs() < 1e-11
                    && (velocity.value_as_f64(pt_idx, 1).unwrap() - x).abs() < 1e-11
                    && velocity.value_as_f64(pt_idx, 2).unwrap().abs() < 1e-11,
                "{name} pt {pt_idx}: velocity mismatch"
            );
        }

        // Verify cell_id array
        let cell_id = grid
            .cell_data()
            .get("cell_id")
            .unwrap_or_else(|| panic!("{name}: missing 'cell_id'"));
        assert_eq!(cell_id.components(), ComponentCount::Scalar);
        for cell_idx in 0..expected_cells {
            assert_eq!(
                cell_id.value_as_f64(cell_idx, 0),
                Some(cell_idx as f64),
                "{name} cell_id[{cell_idx}]"
            );
        }

        // Round-trip through the ASCII .vtu writer
        let written_bytes =
            write_vtu_to_bytes(&grid).unwrap_or_else(|e| panic!("write {name}: {e}"));
        let roundtripped =
            read_vtu(&written_bytes).unwrap_or_else(|e| panic!("re-read {name}: {e}"));
        assert_eq!(
            grid, roundtripped,
            "{name} did not round-trip identically through write_vtu"
        );
    }
}

#[test]
fn msh_ascii_and_binary_encodings_match_exactly() {
    let dir = gmsh_dir();

    let v41_ascii = read_msh_from_path(dir.join("tet2_v41.msh")).unwrap();
    let v41_binary = read_msh_from_path(dir.join("tet2_v41_binary.msh")).unwrap();
    assert_eq!(v41_ascii, v41_binary, "MSH 4.1 binary != ascii");

    let v22_ascii = read_msh_from_path(dir.join("tet_v22.msh")).unwrap();
    let v22_binary = read_msh_from_path(dir.join("tet_v22_binary.msh")).unwrap();
    assert_eq!(v22_ascii, v22_binary, "MSH 2.2 binary != ascii");
}

#[test]
fn quadratic_tetra_nodes_are_permuted_into_vtk_canonical_ordering() {
    let dir = gmsh_dir();
    for name in ["tet2_v41.msh", "tet2_v41_binary.msh"] {
        let grid = read_msh_from_path(dir.join(name)).unwrap();
        let pts = grid.points();
        // VTK_QUADRATIC_TETRA (type 24) edge definitions for mid-edge nodes 4..10:
        // 4: (0,1), 5: (1,2), 6: (2,0), 7: (0,3), 8: (1,3), 9: (2,3).
        // (In Gmsh MSH_TET_10, nodes 8 and 9 are reversed; the reader must permute them.)
        let vtk_edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];
        for cell_idx in 0..grid.num_cells() {
            assert_eq!(grid.cell_types()[cell_idx], CellType::QuadraticTetra);
            let conn = grid.cell_connectivity(cell_idx).unwrap();
            assert_eq!(conn.len(), 10);
            for (edge_idx, &(a, b)) in vtk_edges.iter().enumerate() {
                let pa = pts[conn[a] as usize];
                let pb = pts[conn[b] as usize];
                let pm = pts[conn[4 + edge_idx] as usize];
                for d in 0..3 {
                    let expected = 0.5 * (pa[d] + pb[d]);
                    assert!(
                        (pm[d] - expected).abs() < 1e-12,
                        "{name} cell {cell_idx} edge {edge_idx} coord {d}: {} != {expected}",
                        pm[d]
                    );
                }
            }
        }
    }
}

#[test]
fn all_gmsh_higher_order_permutation_tables_are_bijections_and_remap_correctly() {
    // Verify every supported Gmsh element type's permutation is a valid bijection of 0..N
    // and that reading a single-element MSH 4.1 file places Gmsh node `gmsh_to_vtk[i]` at VTK slot `i`.
    let all_types = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 26,
    ];
    for gmsh_type in all_types {
        let spec = gmsh_element_spec(gmsh_type)
            .unwrap_or_else(|| panic!("missing spec for Gmsh type {gmsh_type}"));
        let n = spec.gmsh_index_for_vtk_slot.len();

        let mut seen = vec![false; n];
        for &src in spec.gmsh_index_for_vtk_slot {
            assert!(src < n, "type {gmsh_type}: index {src} >= {n}");
            assert!(!seen[src], "type {gmsh_type}: duplicate index {src}");
            seen[src] = true;
        }

        // Build a synthetic MSH 4.1 ASCII mesh with `n` nodes where node tag (k+1) has
        // coordinates [k as f64, 0.0, 0.0], and connectivity is [1, 2, ..., n].
        let mut msh = String::new();
        msh.push_str("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n");
        msh.push_str(&format!("$Nodes\n1 {n} 1 {n}\n3 1 0 {n}\n"));
        for k in 0..n {
            msh.push_str(&format!("{}\n", k + 1));
        }
        for k in 0..n {
            msh.push_str(&format!("{k}.0 0.0 0.0\n"));
        }
        msh.push_str("$EndNodes\n");
        msh.push_str(&format!("$Elements\n1 1 1 1\n3 1 {gmsh_type} 1\n1"));
        for k in 0..n {
            msh.push_str(&format!(" {}", k + 1));
        }
        msh.push_str("\n$EndElements\n");

        let grid = read_msh(msh.as_bytes())
            .unwrap_or_else(|e| panic!("failed synthetic MSH for type {gmsh_type}: {e}"));
        assert_eq!(grid.num_cells(), 1);
        assert_eq!(grid.cell_types()[0], spec.cell_type);
        let conn = grid.cell_connectivity(0).unwrap();
        assert_eq!(conn.len(), n);
        for (vtk_slot, &pt_idx) in conn.iter().enumerate() {
            assert_eq!(
                pt_idx as usize, spec.gmsh_index_for_vtk_slot[vtk_slot],
                "Gmsh type {gmsh_type} VTK slot {vtk_slot}"
            );
        }
    }
}

#[test]
fn truncated_and_corrupted_msh_inputs_return_errors_without_panicking() {
    let dir = gmsh_dir();
    for (name, _, _, _) in ALL_MSH_FILES {
        let bytes = fs::read(dir.join(name)).unwrap();
        let len = bytes.len();
        let cut_points = [
            0,
            1,
            15,
            len / 8,
            len / 4,
            len / 2,
            (len * 3) / 4,
            len.saturating_sub(16),
        ];
        for &cut in &cut_points {
            let truncated = &bytes[..cut.min(len)];
            assert!(
                read_msh(truncated).is_err(),
                "{name} truncated to {cut} bytes unexpectedly succeeded"
            );
        }
    }

    // Unsupported version
    let bad_version = b"$MeshFormat\n3.0 0 8\n$EndMeshFormat\n";
    assert!(read_msh(bad_version).is_err());

    // Unsupported element type
    let bad_elem = b"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n\
$Nodes\n1\n1 0.0 0.0 0.0\n$EndNodes\n\
$Elements\n1\n1 999 0 1\n$EndElements\n";
    assert!(read_msh(bad_elem).is_err());

    // Unknown node tag referenced by element
    let bad_node_tag = b"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n\
$Nodes\n1\n1 0.0 0.0 0.0\n$EndNodes\n\
$Elements\n1\n1 15 0 42\n$EndElements\n";
    assert!(read_msh(bad_node_tag).is_err());
}

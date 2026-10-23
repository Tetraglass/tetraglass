//! Integration, round-trip, node-ordering, and corruption tests for the `.vtu` reader and writer.
#![cfg(not(target_arch = "wasm32"))]

use std::fs;
use std::path::PathBuf;

use tetraglass_core::{CellType, ComponentCount};
use tetraglass_io::{
    MeshFormat, detect_format_from_bytes, detect_format_from_extension, read_vtu,
    read_vtu_from_path, write_vtu_to_bytes,
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

const ALL_VTU_FILES: [(&str, usize, usize); 11] = [
    ("tet_linear.vtu", 8, 6),
    ("tet_linear_binary.vtu", 8, 6),
    ("tet_linear_appended.vtu", 8, 6),
    ("hex_linear.vtu", 27, 8),
    ("mixed_linear.vtu", 11, 4),
    ("tet_quadratic.vtu", 27, 6),
    ("hex_lagrange_p3.vtu", 112, 2),
    ("polyhedra.vtu", 14, 2),
    ("series_0.vtu", 8, 6),
    ("series_1.vtu", 8, 6),
    ("series_2.vtu", 8, 6),
];

#[test]
fn load_and_round_trip_all_vtu_test_files() {
    let dir = vtk_dir();
    for (name, expected_points, expected_cells) in ALL_VTU_FILES {
        let path = dir.join(name);
        let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {name}: {e}"));
        assert_eq!(detect_format_from_bytes(&bytes), Some(MeshFormat::Vtu));
        assert_eq!(detect_format_from_extension("vtu"), Some(MeshFormat::Vtu));

        let grid = read_vtu_from_path(&path).unwrap_or_else(|e| panic!("parse {name}: {e}"));
        assert_eq!(grid.num_points(), expected_points, "{name} point count");
        assert_eq!(grid.num_cells(), expected_cells, "{name} cell count");

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
fn ascii_binary_and_appended_zlib_encodings_match_exactly() {
    let dir = vtk_dir();
    let ascii = read_vtu_from_path(dir.join("tet_linear.vtu")).unwrap();
    let binary = read_vtu_from_path(dir.join("tet_linear_binary.vtu")).unwrap();
    let appended = read_vtu_from_path(dir.join("tet_linear_appended.vtu")).unwrap();

    assert_eq!(ascii, binary, "binary != ascii");
    assert_eq!(ascii, appended, "appended != ascii");
}

#[test]
fn higher_order_node_ordering_matches_vtk_reference_elements() {
    let dir = vtk_dir();

    // 1. Quadratic tetrahedra (VTK type 24): mid-edge nodes 4..10 are exact midpoints of edges
    // (0,1), (1,2), (2,0), (0,3), (1,3), (2,3).
    let quad_tet = read_vtu_from_path(dir.join("tet_quadratic.vtu")).unwrap();
    let pts = quad_tet.points();
    let edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];
    for cell_idx in 0..quad_tet.num_cells() {
        assert_eq!(quad_tet.cell_types()[cell_idx], CellType::QuadraticTetra);
        let conn = quad_tet.cell_connectivity(cell_idx).unwrap();
        assert_eq!(conn.len(), 10);
        for (edge_idx, &(a, b)) in edges.iter().enumerate() {
            let pa = pts[conn[a] as usize];
            let pb = pts[conn[b] as usize];
            let pm = pts[conn[4 + edge_idx] as usize];
            for d in 0..3 {
                let expected = 0.5 * (pa[d] + pb[d]);
                assert!(
                    (pm[d] - expected).abs() < 1e-12,
                    "tet_quadratic cell {cell_idx} edge {edge_idx} coord {d}: {} != {expected}",
                    pm[d]
                );
            }
        }
    }

    // 2. Third-order Lagrange hexahedra (VTK type 72, order 3, 64 nodes):
    // First 8 nodes in VTK ordering are the 8 corners of the hexahedron.
    let lag_hex = read_vtu_from_path(dir.join("hex_lagrange_p3.vtu")).unwrap();
    let hpts = lag_hex.points();
    for cell_idx in 0..lag_hex.num_cells() {
        assert_eq!(
            lag_hex.cell_types()[cell_idx],
            CellType::LagrangeHexahedron { order: 3 }
        );
        let conn = lag_hex.cell_connectivity(cell_idx).unwrap();
        assert_eq!(conn.len(), 64);
        let x0 = cell_idx as f64;
        let expected_corners = [
            [x0, 0.0, 0.0],
            [x0 + 1.0, 0.0, 0.0],
            [x0 + 1.0, 1.0, 0.0],
            [x0, 1.0, 0.0],
            [x0, 0.0, 1.0],
            [x0 + 1.0, 0.0, 1.0],
            [x0 + 1.0, 1.0, 1.0],
            [x0, 1.0, 1.0],
        ];
        for (corner_idx, expected) in expected_corners.iter().enumerate() {
            let actual = hpts[conn[corner_idx] as usize];
            for d in 0..3 {
                assert!(
                    (actual[d] - expected[d]).abs() < 1e-12,
                    "hex_lagrange_p3 cell {cell_idx} corner {corner_idx} coord {d}"
                );
            }
        }
    }
}

#[test]
fn polyhedra_loads_from_both_classic_and_vtu23_face_arrays() {
    let dir = vtk_dir();
    let raw_xml = fs::read_to_string(dir.join("polyhedra.vtu")).unwrap();
    let grid_both = read_vtu(raw_xml.as_bytes()).unwrap();
    assert_eq!(
        grid_both.cell_types(),
        &[CellType::Polyhedron, CellType::Polyhedron]
    );
    assert_eq!(grid_both.polyhedron_cell_faces(0).unwrap().count(), 6);
    assert_eq!(grid_both.polyhedron_cell_faces(1).unwrap().count(), 7);

    // Strip the classic `faces` and `faceoffsets` DataArrays so only VTK 9.4+ VTU 2.3
    // (`face_connectivity`, `face_offsets`, `polyhedron_to_faces`, `polyhedron_offsets`) remain.
    let vtu23_only = raw_xml
        .replace("Name=\"faces\"", "Name=\"unused_faces\"")
        .replace("Name=\"faceoffsets\"", "Name=\"unused_faceoffsets\"");
    let grid_vtu23 = read_vtu(vtu23_only.as_bytes()).unwrap();
    assert_eq!(grid_both, grid_vtu23);
}

#[test]
fn truncated_and_corrupted_vtu_inputs_return_errors_without_panicking() {
    let dir = vtk_dir();
    for (name, _, _) in ALL_VTU_FILES {
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
            len.saturating_sub(12),
        ];
        for &cut in &cut_points {
            let truncated = &bytes[..cut.min(len)];
            assert!(
                read_vtu(truncated).is_err(),
                "{name} truncated to {cut} bytes unexpectedly succeeded"
            );
        }
    }

    // Corrupted base64 payload
    let bad_b64 = br#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">
  <UnstructuredGrid>
    <Piece NumberOfPoints="1" NumberOfCells="0">
      <Points>
        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="binary">
          not_valid_base64!!!
        </DataArray>
      </Points>
      <Cells/>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#;
    assert!(read_vtu(bad_b64).is_err());

    // Negative connectivity index
    let neg_idx = br#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1">
  <UnstructuredGrid>
    <Piece NumberOfPoints="4" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">
          0 0 0 1 0 0 0 1 0 0 0 1
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii">0 1 2 -1</DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii">4</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">10</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#;
    assert!(read_vtu(neg_idx).is_err());
}

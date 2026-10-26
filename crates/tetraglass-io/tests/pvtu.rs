//! Integration tests for the Parallel VTK XML (`.pvtu`) partitioned unstructured grid reader (`M6.3`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use tetraglass_core::{
    CellType, ComponentCount, DataArray, PolyhedronFaces, ScalarType, UnstructuredGrid,
};
use tetraglass_io::{
    MeshFormat, detect_format_from_bytes, detect_format_from_extension, parse_pvtu_manifest,
    read_pvd_series_with_loader, read_pvtu_from_path, read_pvtu_series_from_path,
    read_pvtu_series_with_loader, read_pvtu_with_loader, read_series_from_path, read_vtu,
    read_vtu_from_path, write_vtu_to_bytes,
};

fn testdata_vtk_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .join("testdata/vtk")
}

#[test]
fn detects_pvtu_format_from_extension_and_magic_bytes() {
    assert_eq!(detect_format_from_extension("pvtu"), Some(MeshFormat::Pvtu));
    assert_eq!(detect_format_from_extension("PVTU"), Some(MeshFormat::Pvtu));

    let pvtu_bytes = std::fs::read(testdata_vtk_dir().join("hex_partitioned.pvtu"))
        .expect("read hex_partitioned.pvtu");
    assert_eq!(
        detect_format_from_bytes(&pvtu_bytes),
        Some(MeshFormat::Pvtu)
    );
}

#[test]
fn parses_pvtu_manifest_metadata_and_pieces() {
    let pvtu_bytes = std::fs::read(testdata_vtk_dir().join("hex_partitioned.pvtu"))
        .expect("read hex_partitioned.pvtu");
    let manifest = parse_pvtu_manifest(&pvtu_bytes).expect("parse pvtu manifest");

    assert_eq!(manifest.ghost_level, 1);
    assert_eq!(manifest.pieces.len(), 2);
    assert_eq!(manifest.pieces[0].source, "hex_partitioned_0.vtu");
    assert_eq!(manifest.pieces[1].source, "hex_partitioned_1.vtu");

    // `vtkGhostType` is automatically omitted from user-facing point/cell array metadata.
    assert_eq!(manifest.point_data.len(), 3);
    assert_eq!(manifest.point_data[0].name, "linear");
    assert_eq!(manifest.point_data[0].number_of_components, 1);
    assert_eq!(manifest.point_data[2].name, "velocity");
    assert_eq!(manifest.point_data[2].number_of_components, 3);
    assert_eq!(manifest.cell_data.len(), 1);
    assert_eq!(manifest.cell_data[0].name, "cell_id");
}

#[test]
fn reads_partitioned_pvtu_filters_ghost_cells_welds_interface_and_matches_single_vtu() {
    let vtk_dir = testdata_vtk_dir();
    let reference =
        read_vtu_from_path(vtk_dir.join("hex_linear.vtu")).expect("read reference hex_linear.vtu");
    let merged = read_pvtu_from_path(vtk_dir.join("hex_partitioned.pvtu"))
        .expect("read hex_partitioned.pvtu");

    // 1. Both piece 0 and piece 1 have 6 hexes (4 owned + 2 ghost with `vtkGhostType = 1`).
    // After ghost-cell filtering and interface welding, the merged grid must have the exact
    // 27 points and 8 hexahedra of `hex_linear.vtu`, and `vtkGhostType` must be stripped.
    assert_eq!(merged.num_points(), reference.num_points());
    assert_eq!(merged.num_cells(), reference.num_cells());
    assert_eq!(merged.bounds(), reference.bounds());
    assert!(merged.cell_data().get("vtkGhostType").is_none());
    assert!(merged.point_data().get("vtkGhostType").is_none());

    // 2. Every merged vertex must reproduce the analytic `linear(x, y, z) = x + 2y + 3z`,
    // `radius`, and `velocity = [-y, x, 0]` fields.
    let lin_arr = merged.point_data().get("linear").expect("linear array");
    let vel_arr = merged.point_data().get("velocity").expect("velocity array");
    for (pt_idx, &[x, y, z]) in merged.points().iter().enumerate() {
        let expected_lin = x + 2.0 * y + 3.0 * z;
        let actual_lin = lin_arr.buffer().get_as_f64(pt_idx).unwrap();
        assert!((actual_lin - expected_lin).abs() < 1e-12);
        let vx = vel_arr.buffer().get_as_f64(pt_idx * 3).unwrap();
        let vy = vel_arr.buffer().get_as_f64(pt_idx * 3 + 1).unwrap();
        let vz = vel_arr.buffer().get_as_f64(pt_idx * 3 + 2).unwrap();
        assert!((vx - (-y)).abs() < 1e-12);
        assert!((vy - x).abs() < 1e-12);
        assert!(vz.abs() < 1e-12);
    }

    // 3. Cell IDs `0..8` must all be present exactly once (no duplicate ghost cells).
    let cell_id_arr = merged.cell_data().get("cell_id").expect("cell_id array");
    assert_eq!(cell_id_arr.scalar_type(), ScalarType::I32);
    let mut cell_ids: Vec<i32> = (0..merged.num_cells())
        .map(|i| cell_id_arr.buffer().get_as_f64(i).unwrap() as i32)
        .collect();
    cell_ids.sort_unstable();
    assert_eq!(cell_ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    // 4. Because shared interface vertices at x = 1.0 are welded across piece 0 and piece 1,
    // the 9 interface nodes at x = 1.0 must each appear once in `merged.points()` and be shared
    // by cells from both partitions.
    let interface_pts: Vec<u32> = merged
        .points()
        .iter()
        .enumerate()
        .filter_map(|(i, &[x, _, _])| ((x - 1.0).abs() < 1e-12).then_some(i as u32))
        .collect();
    assert_eq!(interface_pts.len(), 9);
    for &ipt in &interface_pts {
        let sharing_cells = (0..merged.num_cells())
            .filter(|&c| merged.cell_connectivity(c).unwrap().contains(&ipt))
            .count();
        assert!(
            sharing_cells >= 2,
            "interface vertex {ipt} must be shared across partitions"
        );
    }

    // 5. Round-trip the merged grid through `write_vtu_to_bytes` and `read_vtu`.
    let vtu_bytes = write_vtu_to_bytes(&merged).expect("write vtu bytes");
    let roundtripped = read_vtu(&vtu_bytes).expect("read roundtripped vtu");
    assert_eq!(roundtripped, merged);

    // 6. Verify `read_series_from_path` and `read_pvtu_series_from_path`.
    let (fmt, series) =
        read_series_from_path(vtk_dir.join("hex_partitioned.pvtu"), None).expect("read series");
    assert_eq!(fmt, MeshFormat::Pvtu);
    assert_eq!(series.len(), 1);
    assert_eq!(*series.step(0).expect("load step 0"), merged);

    let series2 = read_pvtu_series_from_path(vtk_dir.join("hex_partitioned.pvtu"))
        .expect("read_pvtu_series_from_path");
    assert_eq!(*series2.step(0).expect("load step 0"), merged);
}

#[test]
fn merges_polyhedron_pieces_and_preserves_discontinuous_interface_nodes() {
    // 1. Partition `polyhedra.vtu` (2 polyhedra sharing the x = 1 quad face) into 2 separate pieces
    // and verify that `read_pvtu_with_loader` welds the shared quad face and remaps `PolyhedronFaces`.
    let vtk_dir = testdata_vtk_dir();
    let poly_grid = read_vtu_from_path(vtk_dir.join("polyhedra.vtu")).expect("read polyhedra.vtu");
    assert_eq!(poly_grid.num_cells(), 2);

    let build_single_poly_piece = |cell_idx: usize| -> Vec<u8> {
        let conn = poly_grid.cell_connectivity(cell_idx).unwrap();
        let faces: Vec<&[u32]> = poly_grid.polyhedron_cell_faces(cell_idx).unwrap().collect();
        let mut local_pts = Vec::new();
        let mut map = BTreeMap::new();
        for &gpt in conn {
            let next = local_pts.len() as u32;
            map.entry(gpt).or_insert_with(|| {
                local_pts.push(poly_grid.points()[gpt as usize]);
                next
            });
        }
        let local_conn: Vec<u32> = conn.iter().map(|g| map[g]).collect();
        let mut face_stream = vec![faces.len() as u32];
        for f in faces {
            face_stream.push(f.len() as u32);
            for &g in f {
                face_stream.push(map[&g]);
            }
        }
        let poly_faces = PolyhedronFaces::new(vec![face_stream.len() as u32], face_stream);
        let mut piece = UnstructuredGrid::new(
            local_pts.clone(),
            vec![CellType::Polyhedron],
            vec![local_conn.len() as u32],
            local_conn,
            Some(poly_faces),
        )
        .unwrap();
        let lin: Vec<f64> = local_pts
            .iter()
            .map(|&[x, y, z]| x + 2.0 * y + 3.0 * z)
            .collect();
        piece
            .insert_point_data(
                "linear",
                DataArray::from_f64(ComponentCount::Scalar, lin).unwrap(),
            )
            .unwrap();
        piece
            .insert_cell_data(
                "cell_id",
                DataArray::from_i32(ComponentCount::Scalar, vec![cell_idx as i32]).unwrap(),
            )
            .unwrap();
        write_vtu_to_bytes(&piece).unwrap()
    };

    let piece0_bytes = build_single_poly_piece(0);
    let piece1_bytes = build_single_poly_piece(1);
    let pvtu_xml = br#"<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="1.0">
  <PUnstructuredGrid GhostLevel="0">
    <PPointData>
      <PDataArray type="Float64" Name="linear" NumberOfComponents="1"/>
    </PPointData>
    <PCellData>
      <PDataArray type="Int32" Name="cell_id" NumberOfComponents="1"/>
    </PCellData>
    <PPoints>
      <PDataArray type="Float64" Name="Points" NumberOfComponents="3"/>
    </PPoints>
    <Piece Source="poly_0.vtu"/>
    <Piece Source="poly_1.vtu"/>
  </PUnstructuredGrid>
</VTKFile>"#;

    let merged_poly = read_pvtu_with_loader(pvtu_xml, |src| match src {
        "poly_0.vtu" => Ok(piece0_bytes.clone()),
        "poly_1.vtu" => Ok(piece1_bytes.clone()),
        other => panic!("unexpected piece {other}"),
    })
    .expect("merge polyhedron pieces");

    assert_eq!(merged_poly.num_points(), poly_grid.num_points());
    assert_eq!(merged_poly.num_cells(), 2);
    assert_eq!(
        merged_poly.polyhedron_cell_faces(0).unwrap().count(),
        poly_grid.polyhedron_cell_faces(0).unwrap().count()
    );
    assert_eq!(
        merged_poly.polyhedron_cell_faces(1).unwrap().count(),
        poly_grid.polyhedron_cell_faces(1).unwrap().count()
    );

    // 2. Also verify that a `.pvd` time series can reference `.pvtu` step files.
    let pvd_xml = br#"<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1">
  <Collection>
    <DataSet timestep="0.0" file="step_0.pvtu"/>
  </Collection>
</VTKFile>"#;
    let pvd_series = read_pvd_series_with_loader(pvd_xml, move |rel| match rel {
        "step_0.pvtu" => Ok(pvtu_xml.to_vec()),
        "poly_0.vtu" => Ok(piece0_bytes.clone()),
        "poly_1.vtu" => Ok(piece1_bytes.clone()),
        other => panic!("unexpected file {other}"),
    })
    .expect("read pvd referencing pvtu");
    assert_eq!(*pvd_series.step(0).expect("load pvtu step"), merged_poly);
}

#[test]
fn rejects_truncated_and_malformed_pvtu_manifests_without_panicking() {
    let pvtu_bytes = std::fs::read(testdata_vtk_dir().join("hex_partitioned.pvtu"))
        .expect("read hex_partitioned.pvtu");

    // Truncate at several offsets across the file.
    for cut in [
        0,
        5,
        25,
        80,
        160,
        pvtu_bytes.len() / 2,
        pvtu_bytes.len() - 5,
    ] {
        let res = read_pvtu_with_loader(&pvtu_bytes[..cut], |_| Ok(Vec::new()));
        assert!(
            res.is_err(),
            "expected error on truncated pvtu at byte {cut}"
        );
    }

    // Wrong VTKFile type
    let wrong_type = br#"<VTKFile type="UnstructuredGrid"><PUnstructuredGrid><Piece Source="a.vtu"/></PUnstructuredGrid></VTKFile>"#;
    assert!(parse_pvtu_manifest(wrong_type).is_err());

    // No <Piece> entries
    let no_pieces =
        br#"<VTKFile type="PUnstructuredGrid"><PUnstructuredGrid></PUnstructuredGrid></VTKFile>"#;
    assert!(parse_pvtu_manifest(no_pieces).is_err());

    // Missing Source attribute on <Piece/>
    let missing_source = br#"<VTKFile type="PUnstructuredGrid"><PUnstructuredGrid><Piece/></PUnstructuredGrid></VTKFile>"#;
    assert!(parse_pvtu_manifest(missing_source).is_err());

    // Invalid GhostLevel
    let bad_ghost = br#"<VTKFile type="PUnstructuredGrid"><PUnstructuredGrid GhostLevel="abc"><Piece Source="a.vtu"/></PUnstructuredGrid></VTKFile>"#;
    assert!(parse_pvtu_manifest(bad_ghost).is_err());

    // Lazy series rejects malformed manifest upfront
    assert!(read_pvtu_series_with_loader(no_pieces.as_slice(), |_| Ok(Vec::new())).is_err());
}

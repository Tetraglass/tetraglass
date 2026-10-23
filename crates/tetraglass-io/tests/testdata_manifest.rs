//! Integration tests verifying the committed v1 test data manifest.
#![cfg(not(target_arch = "wasm32"))]

use std::fs;
use std::path::PathBuf;

const MAX_TESTDATA_BYTES: u64 = 1_000_000;

fn testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/ directory")
        .parent()
        .expect("workspace root")
        .join("testdata")
}

#[test]
fn required_vtk_files_exist_and_contain_standard_fields() {
    let vtk_dir = testdata_dir().join("vtk");
    let vtu_files = [
        "tet_linear.vtu",
        "tet_linear_binary.vtu",
        "tet_linear_appended.vtu",
        "hex_linear.vtu",
        "mixed_linear.vtu",
        "tet_quadratic.vtu",
        "hex_lagrange_p3.vtu",
        "polyhedra.vtu",
        "series_0.vtu",
        "series_1.vtu",
        "series_2.vtu",
    ];

    for name in vtu_files {
        let path = vtk_dir.join(name);
        let meta =
            fs::metadata(&path).unwrap_or_else(|e| panic!("missing {}: {e}", path.display()));
        assert!(meta.len() > 0, "{} is empty", path.display());
        assert!(
            meta.len() < MAX_TESTDATA_BYTES,
            "{} exceeds 1 MB ({} bytes)",
            path.display(),
            meta.len()
        );

        let contents = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
        assert!(contents.contains("<VTKFile type=\"UnstructuredGrid\""));
        for field in [
            "Name=\"linear\"",
            "Name=\"radius\"",
            "Name=\"velocity\"",
            "Name=\"cell_id\"",
        ] {
            assert!(
                contents.contains(field),
                "{} missing field {}",
                path.display(),
                field
            );
        }
    }

    let pvd_path = vtk_dir.join("series.pvd");
    let pvd = fs::read_to_string(&pvd_path).expect("read series.pvd");
    assert!(pvd.contains("<VTKFile type=\"Collection\""));
    assert!(pvd.contains("file=\"series_0.vtu\""));
    assert!(pvd.contains("file=\"series_1.vtu\""));
    assert!(pvd.contains("file=\"series_2.vtu\""));
}

#[test]
fn required_gmsh_files_exist_and_contain_sections() {
    let gmsh_dir = testdata_dir().join("gmsh");
    let msh_files = [
        ("tet2_v41.msh", b"4.1 0 8".as_slice()),
        ("tet2_v41_binary.msh", b"4.1 1 8".as_slice()),
        ("tet_v22.msh", b"2.2 0 8".as_slice()),
        ("tet_v22_binary.msh", b"2.2 1 8".as_slice()),
    ];

    for (name, expected_format) in msh_files {
        let path = gmsh_dir.join(name);
        let bytes = fs::read(&path).unwrap_or_else(|e| panic!("missing {}: {e}", path.display()));
        assert!(!bytes.is_empty(), "{} is empty", path.display());
        assert!(
            (bytes.len() as u64) < MAX_TESTDATA_BYTES,
            "{} exceeds 1 MB ({} bytes)",
            path.display(),
            bytes.len()
        );
        assert!(
            bytes
                .windows(expected_format.len())
                .any(|w| w == expected_format),
            "{} missing expected MeshFormat header",
            path.display()
        );
        for marker in [
            b"$Nodes".as_slice(),
            b"$Elements".as_slice(),
            b"$NodeData".as_slice(),
            b"$ElementData".as_slice(),
            b"\"linear\"".as_slice(),
            b"\"radius\"".as_slice(),
            b"\"velocity\"".as_slice(),
            b"\"cell_id\"".as_slice(),
        ] {
            assert!(
                bytes.windows(marker.len()).any(|w| w == marker),
                "{} missing marker {:?}",
                path.display(),
                String::from_utf8_lossy(marker)
            );
        }
    }
}

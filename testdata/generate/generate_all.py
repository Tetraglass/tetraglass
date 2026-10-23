#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerates all Tetraglass test meshes and verifies their contents."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import gmsh
import vtk

import generate_gmsh
import generate_vtk

ROOT = Path(__file__).resolve().parent.parent
MAX_FILE_BYTES = 1_000_000


def verify_vtu(path: Path, expected_points: int, expected_cells: int, extra_arrays: tuple[str, ...] = ()) -> None:
    assert path.exists(), f"Missing {path}"
    assert path.stat().st_size < MAX_FILE_BYTES, f"{path} exceeds 1 MB"

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    ugrid = reader.GetOutput()

    assert ugrid.GetNumberOfPoints() == expected_points, (
        f"{path.name}: expected {expected_points} points, got {ugrid.GetNumberOfPoints()}"
    )
    assert ugrid.GetNumberOfCells() == expected_cells, (
        f"{path.name}: expected {expected_cells} cells, got {ugrid.GetNumberOfCells()}"
    )

    pd = ugrid.GetPointData()
    for name, ncomp in (("linear", 1), ("radius", 1), ("velocity", 3)):
        arr = pd.GetArray(name)
        assert arr is not None, f"{path.name}: missing point array '{name}'"
        assert arr.GetNumberOfComponents() == ncomp, (
            f"{path.name}: array '{name}' expected {ncomp} components, got {arr.GetNumberOfComponents()}"
        )
    for name in extra_arrays:
        assert pd.GetArray(name) is not None, f"{path.name}: missing point array '{name}'"

    cd = ugrid.GetCellData()
    cell_id = cd.GetArray("cell_id")
    assert cell_id is not None, f"{path.name}: missing cell array 'cell_id'"
    assert cell_id.GetNumberOfComponents() == 1

    # Verify exact linear field values x + 2y + 3z at every node.
    lin_arr = pd.GetArray("linear")
    for idx in range(ugrid.GetNumberOfPoints()):
        x, y, z = ugrid.GetPoint(idx)
        expected = x + 2.0 * y + 3.0 * z
        actual = lin_arr.GetValue(idx)
        assert abs(actual - expected) < 1e-12, (
            f"{path.name}: point {idx} linear={actual} != {expected}"
        )


def verify_msh(path: Path, expected_nodes: int, expected_cells: int) -> None:
    assert path.exists(), f"Missing {path}"
    assert path.stat().st_size < MAX_FILE_BYTES, f"{path} exceeds 1 MB"

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        _, elem_tags_by_type, _ = gmsh.model.mesh.getElements(3)
        assert len(node_tags) == expected_nodes, (
            f"{path.name}: expected {expected_nodes} nodes, got {len(node_tags)}"
        )
        assert len(elem_tags_by_type) == 1 and len(elem_tags_by_type[0]) == expected_cells, (
            f"{path.name}: expected {expected_cells} 3D elements"
        )
        views = gmsh.view.getTags()
        assert len(views) == 4, f"{path.name}: expected 4 views, got {len(views)}"
        for v in views:
            dtype, tags, _, _, _ = gmsh.view.getModelData(v, 0)
            if dtype == "NodeData":
                assert len(tags) == expected_nodes, (
                    f"{path.name}: NodeData view {v} has {len(tags)} tags != {expected_nodes}"
                )
            elif dtype == "ElementData":
                assert len(tags) == expected_cells, (
                    f"{path.name}: ElementData view {v} has {len(tags)} tags != {expected_cells}"
                )
    finally:
        gmsh.finalize()


def main() -> None:
    generate_vtk.main()
    generate_gmsh.main()

    vtk_dir = ROOT / "vtk"
    gmsh_dir = ROOT / "gmsh"

    verify_vtu(vtk_dir / "tet_linear.vtu", expected_points=8, expected_cells=6)
    verify_vtu(vtk_dir / "tet_linear_binary.vtu", expected_points=8, expected_cells=6)
    verify_vtu(vtk_dir / "tet_linear_appended.vtu", expected_points=8, expected_cells=6)
    verify_vtu(vtk_dir / "hex_linear.vtu", expected_points=27, expected_cells=8)
    verify_vtu(vtk_dir / "mixed_linear.vtu", expected_points=11, expected_cells=4)
    verify_vtu(vtk_dir / "tet_quadratic.vtu", expected_points=27, expected_cells=6)
    verify_vtu(vtk_dir / "hex_lagrange_p3.vtu", expected_points=112, expected_cells=2)
    verify_vtu(vtk_dir / "polyhedra.vtu", expected_points=14, expected_cells=2)

    pvd_path = vtk_dir / "series.pvd"
    assert pvd_path.exists() and pvd_path.stat().st_size < MAX_FILE_BYTES
    pvd_tree = ET.parse(pvd_path)
    datasets = pvd_tree.findall(".//Collection/DataSet")
    assert len(datasets) == 3, f"series.pvd: expected 3 DataSet entries, got {len(datasets)}"
    for step_idx in range(3):
        verify_vtu(
            vtk_dir / f"series_{step_idx}.vtu",
            expected_points=8,
            expected_cells=6,
            extra_arrays=("transient",),
        )

    verify_msh(gmsh_dir / "tet2_v41.msh", expected_nodes=27, expected_cells=6)
    verify_msh(gmsh_dir / "tet2_v41_binary.msh", expected_nodes=27, expected_cells=6)
    verify_msh(gmsh_dir / "tet_v22.msh", expected_nodes=8, expected_cells=6)
    verify_msh(gmsh_dir / "tet_v22_binary.msh", expected_nodes=8, expected_cells=6)

    print("All test meshes generated and verified.")


if __name__ == "__main__":
    main()

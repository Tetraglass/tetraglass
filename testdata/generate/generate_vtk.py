#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Deterministic generator for Tetraglass VTK (.vtu, .pvd) test meshes."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import vtk

ROOT = Path(__file__).resolve().parent.parent
VTK_DIR = ROOT / "vtk"

# Kuhn triangulation of the unit cube [0, 1]^3 into 6 tetrahedra with equal volume 1/6.
CUBE_POINTS: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),  # 0
    (1.0, 0.0, 0.0),  # 1
    (1.0, 1.0, 0.0),  # 2
    (0.0, 1.0, 0.0),  # 3
    (0.0, 0.0, 1.0),  # 4
    (1.0, 0.0, 1.0),  # 5
    (1.0, 1.0, 1.0),  # 6
    (0.0, 1.0, 1.0),  # 7
]

KUHN_TETS: list[tuple[int, int, int, int]] = [
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
]


def domain_center(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return (
        0.5 * (min(xs) + max(xs)),
        0.5 * (min(ys) + max(ys)),
        0.5 * (min(zs) + max(zs)),
    )


def attach_standard_fields(
    ugrid: vtk.vtkUnstructuredGrid,
    points: list[tuple[float, float, float]],
    time_value: float | None = None,
) -> None:
    """Attach the required analytic point and cell arrays to `ugrid`.

    Point arrays:
    - `linear`: x + 2y + 3z (Float64, 1 component)
    - `radius`: Euclidean distance from the bounding-box center (Float64, 1 component)
    - `velocity`: (-y, x, 0) (Float64, 3 components)
    - `transient` (only when `time_value` is given): x + 2y + 3z + t (Float64, 1 component)

    Cell arrays:
    - `cell_id`: 0-based cell index (Int32, 1 component)
    """
    cx, cy, cz = domain_center(points)

    arr_linear = vtk.vtkDoubleArray()
    arr_linear.SetName("linear")
    arr_linear.SetNumberOfComponents(1)
    arr_linear.SetNumberOfTuples(len(points))

    arr_radius = vtk.vtkDoubleArray()
    arr_radius.SetName("radius")
    arr_radius.SetNumberOfComponents(1)
    arr_radius.SetNumberOfTuples(len(points))

    arr_velocity = vtk.vtkDoubleArray()
    arr_velocity.SetName("velocity")
    arr_velocity.SetNumberOfComponents(3)
    arr_velocity.SetNumberOfTuples(len(points))

    arr_transient = None
    if time_value is not None:
        arr_transient = vtk.vtkDoubleArray()
        arr_transient.SetName("transient")
        arr_transient.SetNumberOfComponents(1)
        arr_transient.SetNumberOfTuples(len(points))

    for idx, (x, y, z) in enumerate(points):
        lin = x + 2.0 * y + 3.0 * z
        rad = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        arr_linear.SetValue(idx, lin)
        arr_radius.SetValue(idx, rad)
        arr_velocity.SetTuple3(idx, -y, x, 0.0)
        if arr_transient is not None and time_value is not None:
            arr_transient.SetValue(idx, lin + time_value)

    pd = ugrid.GetPointData()
    pd.AddArray(arr_linear)
    pd.AddArray(arr_radius)
    pd.AddArray(arr_velocity)
    if arr_transient is not None:
        pd.AddArray(arr_transient)

    num_cells = ugrid.GetNumberOfCells()
    arr_cell_id = vtk.vtkIntArray()
    arr_cell_id.SetName("cell_id")
    arr_cell_id.SetNumberOfComponents(1)
    arr_cell_id.SetNumberOfTuples(num_cells)
    for cell_idx in range(num_cells):
        arr_cell_id.SetValue(cell_idx, cell_idx)

    ugrid.GetCellData().AddArray(arr_cell_id)


def make_ugrid(
    points: list[tuple[float, float, float]],
    cells: list[tuple[int, list[int]]],
    time_value: float | None = None,
) -> vtk.vtkUnstructuredGrid:
    ugrid = vtk.vtkUnstructuredGrid()
    vpts = vtk.vtkPoints()
    vpts.SetDataTypeToDouble()
    for x, y, z in points:
        vpts.InsertNextPoint(x, y, z)
    ugrid.SetPoints(vpts)

    for cell_type, conn in cells:
        id_list = vtk.vtkIdList()
        for pt_id in conn:
            id_list.InsertNextId(pt_id)
        ugrid.InsertNextCell(cell_type, id_list)

    attach_standard_fields(ugrid, points, time_value=time_value)
    return ugrid


def write_vtu(
    ugrid: vtk.vtkUnstructuredGrid,
    path: Path,
    mode: str = "ascii",
    compress: bool = True,
) -> None:
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(ugrid)
    writer.SetHeaderTypeToUInt32()
    if mode == "ascii":
        writer.SetDataModeToAscii()
    elif mode == "binary":
        writer.SetDataModeToBinary()
        if not compress:
            writer.SetCompressorTypeToNone()
    elif mode == "appended":
        writer.SetDataModeToAppended()
        if compress:
            writer.SetCompressorTypeToZLib()
        else:
            writer.SetCompressorTypeToNone()
    else:
        raise ValueError(f"Unsupported data mode: {mode}")

    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {path}")


def write_vtk_legacy(
    ugrid: vtk.vtkUnstructuredGrid,
    path: Path,
    mode: str = "ascii",
    file_version: int = 42,
) -> None:
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(ugrid)
    writer.SetFileVersion(file_version)
    if mode == "ascii":
        writer.SetFileTypeToASCII()
    elif mode == "binary":
        writer.SetFileTypeToBinary()
    else:
        raise ValueError(f"Unsupported legacy VTK mode: {mode}")

    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {path}")


def build_tet_linear(time_value: float | None = None) -> vtk.vtkUnstructuredGrid:
    cells = [(vtk.VTK_TETRA, list(tet)) for tet in KUHN_TETS]
    return make_ugrid(CUBE_POINTS, cells, time_value=time_value)


def build_hex_linear() -> vtk.vtkUnstructuredGrid:
    """2x2x2 block of 8 linear hexahedra on [0, 2]^3 (27 points, 8 cells, 24 outer quad faces)."""
    points: list[tuple[float, float, float]] = []
    index_of: dict[tuple[int, int, int], int] = {}
    for k in range(3):
        for j in range(3):
            for i in range(3):
                index_of[(i, j, k)] = len(points)
                points.append((float(i), float(j), float(k)))

    cells: list[tuple[int, list[int]]] = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                conn = [
                    index_of[(i, j, k)],
                    index_of[(i + 1, j, k)],
                    index_of[(i + 1, j + 1, k)],
                    index_of[(i, j + 1, k)],
                    index_of[(i, j, k + 1)],
                    index_of[(i + 1, j, k + 1)],
                    index_of[(i + 1, j + 1, k + 1)],
                    index_of[(i, j + 1, k + 1)],
                ]
                cells.append((vtk.VTK_HEXAHEDRON, conn))

    return make_ugrid(points, cells)


def build_mixed_linear() -> vtk.vtkUnstructuredGrid:
    """1 hex, 1 wedge, 1 pyramid and 1 tet sharing 4 internal faces with no cracks."""
    points: list[tuple[float, float, float]] = [
        (0.0, 0.0, 0.0),  # 0
        (1.0, 0.0, 0.0),  # 1
        (1.0, 1.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
        (0.0, 0.0, 1.0),  # 4
        (1.0, 0.0, 1.0),  # 5
        (1.0, 1.0, 1.0),  # 6
        (0.0, 1.0, 1.0),  # 7
        (2.0, 0.0, 0.0),  # 8
        (2.0, 0.0, 1.0),  # 9
        (0.5, 0.5, 2.0),  # 10
    ]
    cells: list[tuple[int, list[int]]] = [
        (vtk.VTK_HEXAHEDRON, [0, 1, 2, 3, 4, 5, 6, 7]),
        (vtk.VTK_WEDGE, [1, 8, 2, 5, 9, 6]),
        (vtk.VTK_PYRAMID, [4, 5, 6, 7, 10]),
        (vtk.VTK_TETRA, [5, 9, 6, 10]),
    ]
    return make_ugrid(points, cells)


def build_tet_quadratic() -> vtk.vtkUnstructuredGrid:
    """6 quadratic tetrahedra (VTK_QUADRATIC_TETRA = 24, 10 nodes each) on [0, 1]^3."""
    points = list(CUBE_POINTS)
    edge_mid_point: dict[tuple[int, int], int] = {}

    def midpoint_index(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in edge_mid_point:
            pa = points[a]
            pb = points[b]
            edge_mid_point[key] = len(points)
            points.append(
                (
                    0.5 * (pa[0] + pb[0]),
                    0.5 * (pa[1] + pb[1]),
                    0.5 * (pa[2] + pb[2]),
                )
            )
        return edge_mid_point[key]

    # VTK_QUADRATIC_TETRA node ordering:
    # corners: 0, 1, 2, 3
    # mid-edge nodes: (0,1), (1,2), (2,0), (0,3), (1,3), (2,3)
    cells: list[tuple[int, list[int]]] = []
    for n0, n1, n2, n3 in KUHN_TETS:
        conn = [
            n0,
            n1,
            n2,
            n3,
            midpoint_index(n0, n1),
            midpoint_index(n1, n2),
            midpoint_index(n2, n0),
            midpoint_index(n0, n3),
            midpoint_index(n1, n3),
            midpoint_index(n2, n3),
        ]
        cells.append((vtk.VTK_QUADRATIC_TETRA, conn))

    return make_ugrid(points, cells)


def build_hex_lagrange_p3() -> vtk.vtkUnstructuredGrid:
    """Two adjacent third-order Lagrange hexahedra (VTK type 72, 64 nodes each) on [0, 2] x [0, 1]^2."""
    order = (3, 3, 3)
    points: list[tuple[float, float, float]] = []
    grid_index: dict[tuple[int, int, int], int] = {}

    for k in range(4):
        for j in range(4):
            for i in range(7):
                grid_index[(i, j, k)] = len(points)
                points.append((i / 3.0, j / 3.0, k / 3.0))

    cells: list[tuple[int, list[int]]] = []
    for cell_idx in range(2):
        i_offset = 3 * cell_idx
        conn = [0] * 64
        for k in range(4):
            for j in range(4):
                for i in range(4):
                    local_id = vtk.vtkLagrangeHexahedron.PointIndexFromIJK(i, j, k, order)
                    conn[local_id] = grid_index[(i_offset + i, j, k)]
        cells.append((vtk.VTK_LAGRANGE_HEXAHEDRON, conn))

    return make_ugrid(points, cells)


def build_polyhedra_and_write(path: Path) -> None:
    """Two adjacent polyhedra (VTK type 42): a 6-face cube and a 7-face beveled prism with pentagon faces."""
    points: list[tuple[float, float, float]] = [
        # x = 0 plane
        (0.0, 0.0, 0.0),  # 0
        (0.0, 1.0, 0.0),  # 1
        (0.0, 1.0, 1.0),  # 2
        (0.0, 0.0, 1.0),  # 3
        # x = 1 plane (shared quad face)
        (1.0, 0.0, 0.0),  # 4
        (1.0, 1.0, 0.0),  # 5
        (1.0, 1.0, 1.0),  # 6
        (1.0, 0.0, 1.0),  # 7
        # x > 1 beveled prism (y = 0 pentagon)
        (2.0, 0.0, 0.0),  # 8
        (2.0, 0.0, 0.5),  # 9
        (1.5, 0.0, 1.0),  # 10
        # x > 1 beveled prism (y = 1 pentagon)
        (2.0, 1.0, 0.0),  # 11
        (2.0, 1.0, 0.5),  # 12
        (1.5, 1.0, 1.0),  # 13
    ]

    cell_faces: list[list[list[int]]] = [
        # Cell 0: unit cube [0, 1]^3 (6 quad faces, volume = 1.0)
        [
            [0, 1, 2, 3],
            [4, 7, 6, 5],
            [0, 4, 5, 1],
            [3, 2, 6, 7],
            [0, 3, 7, 4],
            [1, 5, 6, 2],
        ],
        # Cell 1: beveled prism on [1, 2] x [0, 1]^2 (5 quad faces + 2 pentagon faces, volume = 0.875)
        [
            [4, 5, 6, 7],
            [4, 8, 11, 5],
            [8, 9, 12, 11],
            [9, 10, 13, 12],
            [7, 6, 13, 10],
            [4, 7, 10, 9, 8],
            [5, 11, 12, 13, 6],
        ],
    ]

    ugrid = vtk.vtkUnstructuredGrid()
    vpts = vtk.vtkPoints()
    vpts.SetDataTypeToDouble()
    for x, y, z in points:
        vpts.InsertNextPoint(x, y, z)
    ugrid.SetPoints(vpts)

    legacy_faces_stream: list[int] = []
    legacy_face_offsets: list[int] = []
    for faces in cell_faces:
        cell_stream = [len(faces)]
        for face in faces:
            cell_stream.append(len(face))
            cell_stream.extend(face)
        id_list = vtk.vtkIdList()
        for val in cell_stream:
            id_list.InsertNextId(val)
        ugrid.InsertNextCell(vtk.VTK_POLYHEDRON, id_list)
        legacy_faces_stream.extend(cell_stream)
        legacy_face_offsets.append(len(legacy_faces_stream))

    attach_standard_fields(ugrid, points)
    write_vtu(ugrid, path, mode="ascii")

    # VTK 9.4+ writes VTU 2.3 polyhedron arrays (`face_connectivity`, `face_offsets`,
    # `polyhedron_to_faces`, `polyhedron_offsets`). Also embed the classic `faces` and
    # `faceoffsets` DataArrays inside <Cells> so both VTU 1.0/2.0 and VTU 2.3 readers
    # can parse the face stream from this file.
    tree = ET.parse(path)
    cells_elem = tree.find(".//Piece/Cells")
    if cells_elem is None:
        raise RuntimeError(f"Missing <Cells> element in {path}")

    faces_da = ET.SubElement(
        cells_elem,
        "DataArray",
        {
            "type": "Int64",
            "Name": "faces",
            "format": "ascii",
            "RangeMin": str(min(legacy_faces_stream)),
            "RangeMax": str(max(legacy_faces_stream)),
        },
    )
    faces_da.text = "\n          " + " ".join(str(v) for v in legacy_faces_stream) + "\n        "

    faceoffsets_da = ET.SubElement(
        cells_elem,
        "DataArray",
        {
            "type": "Int64",
            "Name": "faceoffsets",
            "format": "ascii",
            "RangeMin": str(min(legacy_face_offsets)),
            "RangeMax": str(max(legacy_face_offsets)),
        },
    )
    faceoffsets_da.text = (
        "\n          " + " ".join(str(v) for v in legacy_face_offsets) + "\n        "
    )

    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_series_pvd() -> None:
    timesteps = [(0, 0.0), (1, 0.5), (2, 1.0)]
    for step_idx, t in timesteps:
        ugrid = build_tet_linear(time_value=t)
        write_vtu(ugrid, VTK_DIR / f"series_{step_idx}.vtu", mode="ascii")

    pvd_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian" compressor="vtkZLibDataCompressor">',
        "  <Collection>",
    ]
    for step_idx, t in timesteps:
        pvd_lines.append(
            f'    <DataSet timestep="{t}" group="" part="0" file="series_{step_idx}.vtu"/>'
        )
    pvd_lines.extend(
        [
            "  </Collection>",
            "</VTKFile>",
            "",
        ]
    )
    (VTK_DIR / "series.pvd").write_text("\n".join(pvd_lines), encoding="utf-8")


def main() -> None:
    VTK_DIR.mkdir(parents=True, exist_ok=True)

    tet_linear = build_tet_linear()
    write_vtu(tet_linear, VTK_DIR / "tet_linear.vtu", mode="ascii")
    write_vtu(tet_linear, VTK_DIR / "tet_linear_binary.vtu", mode="binary", compress=False)
    write_vtu(tet_linear, VTK_DIR / "tet_linear_appended.vtu", mode="appended", compress=True)
    write_vtk_legacy(tet_linear, VTK_DIR / "tet_linear_legacy.vtk", mode="ascii", file_version=42)
    write_vtk_legacy(
        tet_linear, VTK_DIR / "tet_linear_legacy_binary.vtk", mode="binary", file_version=42
    )
    write_vtk_legacy(tet_linear, VTK_DIR / "tet_linear_v51.vtk", mode="ascii", file_version=51)
    write_vtk_legacy(
        tet_linear, VTK_DIR / "tet_linear_v51_binary.vtk", mode="binary", file_version=51
    )

    write_vtu(build_hex_linear(), VTK_DIR / "hex_linear.vtu", mode="ascii")
    write_vtu(build_mixed_linear(), VTK_DIR / "mixed_linear.vtu", mode="ascii")
    write_vtu(build_tet_quadratic(), VTK_DIR / "tet_quadratic.vtu", mode="ascii")
    write_vtu(build_hex_lagrange_p3(), VTK_DIR / "hex_lagrange_p3.vtu", mode="ascii")
    build_polyhedra_and_write(VTK_DIR / "polyhedra.vtu")
    write_series_pvd()


if __name__ == "__main__":
    main()

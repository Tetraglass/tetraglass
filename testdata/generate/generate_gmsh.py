#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Deterministic generator for Tetraglass Gmsh (.msh) test meshes."""

from __future__ import annotations

import math
from pathlib import Path

import gmsh

ROOT = Path(__file__).resolve().parent.parent
GMSH_DIR = ROOT / "gmsh"


def generate_gmsh_cube(
    path: Path,
    model_name: str,
    element_order: int,
    msh_version: float,
    binary: bool,
) -> None:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("PostProcessing.SaveMesh", 0)
        gmsh.option.setNumber("PostProcessing.SaveInterpolationMatrices", 0)

        gmsh.model.add(model_name)

        # Deterministic transfinite 2x2x2 unit cube [0, 1]^3 -> 6 tetrahedra.
        p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 1.0)
        p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, 1.0)
        p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, 1.0)
        p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, 1.0)
        p5 = gmsh.model.geo.addPoint(0.0, 0.0, 1.0, 1.0)
        p6 = gmsh.model.geo.addPoint(1.0, 0.0, 1.0, 1.0)
        p7 = gmsh.model.geo.addPoint(1.0, 1.0, 1.0, 1.0)
        p8 = gmsh.model.geo.addPoint(0.0, 1.0, 1.0, 1.0)

        lines = [
            gmsh.model.geo.addLine(p1, p2),
            gmsh.model.geo.addLine(p2, p3),
            gmsh.model.geo.addLine(p3, p4),
            gmsh.model.geo.addLine(p4, p1),
            gmsh.model.geo.addLine(p5, p6),
            gmsh.model.geo.addLine(p6, p7),
            gmsh.model.geo.addLine(p7, p8),
            gmsh.model.geo.addLine(p8, p5),
            gmsh.model.geo.addLine(p1, p5),
            gmsh.model.geo.addLine(p2, p6),
            gmsh.model.geo.addLine(p3, p7),
            gmsh.model.geo.addLine(p4, p8),
        ]
        l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12 = lines

        curve_loops = [
            gmsh.model.geo.addCurveLoop([l1, l2, l3, l4]),
            gmsh.model.geo.addCurveLoop([l5, l6, l7, l8]),
            gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9]),
            gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10]),
            gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11]),
            gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12]),
        ]
        surfaces = [gmsh.model.geo.addPlaneSurface([cl]) for cl in curve_loops]
        surf_loop = gmsh.model.geo.addSurfaceLoop(surfaces)
        vol = gmsh.model.geo.addVolume([surf_loop])

        for line in lines:
            gmsh.model.geo.mesh.setTransfiniteCurve(line, 2)
        for surf in surfaces:
            gmsh.model.geo.mesh.setTransfiniteSurface(surf)
        gmsh.model.geo.mesh.setTransfiniteVolume(vol)

        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(3, [vol], 1, "volume")

        gmsh.option.setNumber("Mesh.ElementOrder", element_order)
        gmsh.model.mesh.generate(3)

        _, elem_tags_by_type, elem_nodes_by_type = gmsh.model.mesh.getElements(3)
        elem_tags = list(elem_tags_by_type[0])
        used_node_set = {int(tag) for tag in elem_nodes_by_type[0]}

        all_node_tags, all_node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags: list[int] = []
        coords: list[tuple[float, float, float]] = []
        for i, raw_tag in enumerate(all_node_tags):
            tag = int(raw_tag)
            x = round(float(all_node_coords[3 * i]) * 2.0) / 2.0
            y = round(float(all_node_coords[3 * i + 1]) * 2.0) / 2.0
            z = round(float(all_node_coords[3 * i + 2]) * 2.0) / 2.0
            gmsh.model.mesh.setNode(tag, [x, y, z], [])
            if tag in used_node_set:
                node_tags.append(tag)
                coords.append((x, y, z))

        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        zs = [p[2] for p in coords]
        cx = 0.5 * (min(xs) + max(xs))
        cy = 0.5 * (min(ys) + max(ys))
        cz = 0.5 * (min(zs) + max(zs))

        linear_vals = [[x + 2.0 * y + 3.0 * z] for x, y, z in coords]
        radius_vals = [
            [math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)]
            for x, y, z in coords
        ]
        velocity_vals = [[-y, x, 0.0] for x, y, z in coords]

        v_linear = gmsh.view.add("linear")
        gmsh.view.addModelData(
            v_linear, 0, model_name, "NodeData", node_tags, linear_vals, 0.0, 1
        )

        v_radius = gmsh.view.add("radius")
        gmsh.view.addModelData(
            v_radius, 0, model_name, "NodeData", node_tags, radius_vals, 0.0, 1
        )

        v_velocity = gmsh.view.add("velocity")
        gmsh.view.addModelData(
            v_velocity, 0, model_name, "NodeData", node_tags, velocity_vals, 0.0, 3
        )

        cell_id_vals = [[float(idx)] for idx in range(len(elem_tags))]
        v_cell_id = gmsh.view.add("cell_id")
        gmsh.view.addModelData(
            v_cell_id, 0, model_name, "ElementData", elem_tags, cell_id_vals, 0.0, 1
        )

        gmsh.option.setNumber("Mesh.MshFileVersion", msh_version)
        gmsh.option.setNumber("Mesh.Binary", 1 if binary else 0)
        gmsh.write(str(path))
        for view_tag in (v_linear, v_radius, v_velocity, v_cell_id):
            gmsh.view.write(view_tag, str(path), True)
    finally:
        gmsh.finalize()


def main() -> None:
    GMSH_DIR.mkdir(parents=True, exist_ok=True)

    generate_gmsh_cube(
        GMSH_DIR / "tet2_v41.msh",
        model_name="tet2_v41",
        element_order=2,
        msh_version=4.1,
        binary=False,
    )
    generate_gmsh_cube(
        GMSH_DIR / "tet2_v41_binary.msh",
        model_name="tet2_v41_binary",
        element_order=2,
        msh_version=4.1,
        binary=True,
    )
    generate_gmsh_cube(
        GMSH_DIR / "tet_v22.msh",
        model_name="tet_v22",
        element_order=1,
        msh_version=2.2,
        binary=False,
    )
    generate_gmsh_cube(
        GMSH_DIR / "tet_v22_binary.msh",
        model_name="tet_v22_binary",
        element_order=1,
        msh_version=2.2,
        binary=True,
    )


if __name__ == "__main__":
    main()

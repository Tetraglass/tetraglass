---
name: make-test-data
description: Generates and maintains the small, reproducible test meshes in testdata/ (VTK .vtu/.pvd and Gmsh .msh, covering linear, quadratic, Lagrange and polyhedral cells with analytic fields) and their generation scripts. Use when tests need a mesh that does not exist yet, when a reader or filter needs coverage for a cell type, or when the user asks for test data.
---

# Make test data

Test meshes are small, analytic and reproducible. Each file is committed together with the script that generates it, so anyone can regenerate it and see exactly what it contains.

## Layout

```
testdata/
  README.md            one line per file: what it contains and which script made it
  generate/            the scripts (Python), with a requirements.txt
  vtk/                 .vtu, .vtk and .pvd files
  gmsh/                .msh files
  reference/           filter outputs from VTK, used for comparison tests (M2)
```

## Required set (version 1, roadmap task M0.2)

| File | Contents |
| --- | --- |
| `vtk/tet_linear.vtu` | A few linear tetrahedra |
| `vtk/hex_linear.vtu` | A small hexahedral block |
| `vtk/mixed_linear.vtu` | Tets, hexes, wedges and pyramids sharing faces |
| `vtk/tet_quadratic.vtu` | Quadratic tetrahedra (VTK type 24) |
| `vtk/hex_lagrange_p3.vtu` | Third-order Lagrange hexahedra (VTK type 72) |
| `vtk/polyhedra.vtu` | Polyhedral cells (VTK type 42) with face streams |
| `vtk/series.pvd` | Three time steps of `tet_linear` with a field that changes over time |
| `gmsh/tet2_v41.msh` | Second-order tets in MSH 4.1 with `$NodeData` and `$ElementData` |
| `gmsh/tet_v22.msh` | Linear tets in MSH 2.2 |

## Fields on every mesh

- `linear`: point data `x + 2y + 3z`. Every interpolation must reproduce it exactly, which makes it the main correctness check.
- `radius`: point data, distance from the domain center; contours are spheres.
- `cell_id`: cell data, the cell index as `i32`.
- `velocity`: point data, 3 components `(-y, x, 0)`, a rotation for streamline tests.

## Tools

- **VTK files:** Python with `vtk` (official VTK wheels) for Lagrange cells and polyhedra; `meshio` is fine for simple linear and quadratic cases.
- **Gmsh files:** the `gmsh` Python package, setting `Mesh.ElementOrder` and `Mesh.MshFileVersion`.
- Pin tool versions in `testdata/generate/requirements.txt`.

## Rules

- Keep each file under about 1 MB. Use a few cells, not thousands; performance tests generate their meshes in code.
- Scripts are deterministic: no random numbers without a fixed seed.
- Generate both ASCII and binary encodings where a reader supports both, for VTK at least for `tet_linear`.
- After changing a script, regenerate everything it produces and rerun `/verify`.
- Record new or changed files in `testdata/README.md` and in `docs/progress.md`.

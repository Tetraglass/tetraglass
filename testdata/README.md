# Test Data

Small, deterministic, analytic test meshes for reader, filter and rendering tests.
Every mesh carries four analytic fields:

- `linear` (point scalar, `f64`): $x + 2y + 3z$
- `radius` (point scalar, `f64`): Euclidean distance from the mesh bounding-box center
- `velocity` (point 3-vector, `f64`): $(-y, x, 0)$
- `cell_id` (cell scalar, `i32` in VTK / `f64` in Gmsh): 0-based cell index

## Files

| File | Contents | Generator |
| --- | --- | --- |
| `vtk/tet_linear.vtu` | 6 linear tetrahedra (VTK type 10, 8 points) partitioning $[0, 1]^3$ in ASCII encoding | `generate/generate_vtk.py` |
| `vtk/tet_linear_binary.vtu` | Same 6 linear tetrahedra as `tet_linear.vtu` in inline base64 binary encoding | `generate/generate_vtk.py` |
| `vtk/tet_linear_appended.vtu` | Same 6 linear tetrahedra as `tet_linear.vtu` in zlib-compressed appended binary encoding | `generate/generate_vtk.py` |
| `vtk/tet_linear_legacy.vtk` | Same 6 linear tetrahedra as `tet_linear.vtu` in legacy VTK 4.2 ASCII (`CELLS 6 30`) encoding | `generate/generate_vtk.py` |
| `vtk/tet_linear_legacy_binary.vtk` | Same 6 linear tetrahedra as `tet_linear.vtu` in legacy VTK 4.2 big-endian `BINARY` encoding | `generate/generate_vtk.py` |
| `vtk/tet_linear_v51.vtk` | Same 6 linear tetrahedra as `tet_linear.vtu` in legacy VTK 5.1 ASCII (`OFFSETS`/`CONNECTIVITY`) encoding | `generate/generate_vtk.py` |
| `vtk/tet_linear_v51_binary.vtk` | Same 6 linear tetrahedra as `tet_linear.vtu` in legacy VTK 5.1 big-endian `BINARY` encoding | `generate/generate_vtk.py` |
| `vtk/hex_linear.vtu` | $2 \times 2 \times 2$ block of 8 linear hexahedra (VTK type 12, 27 points, 24 outer quad faces) on $[0, 2]^3$ | `generate/generate_vtk.py` |
| `vtk/mixed_linear.vtu` | 1 hex (12), 1 wedge (13), 1 pyramid (14) and 1 tet (10) sharing 4 internal faces (11 points, volume $2.0$) | `generate/generate_vtk.py` |
| `vtk/tet_quadratic.vtu` | 6 quadratic 10-node tetrahedra (VTK type 24, 27 points) on $[0, 1]^3$ | `generate/generate_vtk.py` |
| `vtk/hex_lagrange_p3.vtu` | 2 adjacent third-order 64-node Lagrange hexahedra (VTK type 72, 112 points) on $[0, 2] \times [0, 1]^2$ | `generate/generate_vtk.py` |
| `vtk/polyhedra.vtu` | 2 adjacent polyhedral cells (VTK type 42, 14 points): a 6-face cube and a 7-face beveled prism with pentagonal faces | `generate/generate_vtk.py` |
| `vtk/series.pvd` | ParaView collection referencing 3 time steps (`t = 0.0, 0.5, 1.0`) of `tet_linear` | `generate/generate_vtk.py` |
| `vtk/series_0.vtu` | Time step 0 (`t = 0.0`) of `series.pvd` with `transient = x + 2y + 3z + t` | `generate/generate_vtk.py` |
| `vtk/series_1.vtu` | Time step 1 (`t = 0.5`) of `series.pvd` with `transient = x + 2y + 3z + t` | `generate/generate_vtk.py` |
| `vtk/series_2.vtu` | Time step 2 (`t = 1.0`) of `series.pvd` with `transient = x + 2y + 3z + t` | `generate/generate_vtk.py` |
| `vtk/hex_partitioned.pvtu` | Parallel VTK XML UnstructuredGrid (`GhostLevel="1"`) partitioning `hex_linear.vtu` into 2 pieces with `vtkGhostType` ghost cells | `generate/generate_vtk.py` |
| `vtk/hex_partitioned_0.vtu` | Piece 0 (`x in [0, 1]`, 4 owned hexes + 2 ghost hexes with `vtkGhostType = 1`) of `hex_partitioned.pvtu` | `generate/generate_vtk.py` |
| `vtk/hex_partitioned_1.vtu` | Piece 1 (`x in [1, 2]`, 4 owned hexes + 2 ghost hexes with `vtkGhostType = 1`) of `hex_partitioned.pvtu` | `generate/generate_vtk.py` |
| `gmsh/tet2_v41.msh` | 6 second-order 10-node tetrahedra (Gmsh type 11, 27 nodes) on $[0, 1]^3$ in MSH 4.1 ASCII with `$NodeData` and `$ElementData` | `generate/generate_gmsh.py` |
| `gmsh/tet2_v41_binary.msh` | Same 6 second-order tetrahedra as `tet2_v41.msh` in MSH 4.1 binary encoding with `$NodeData` and `$ElementData` | `generate/generate_gmsh.py` |
| `gmsh/tet_v22.msh` | 6 linear 4-node tetrahedra (Gmsh type 4, 8 nodes) on $[0, 1]^3$ in MSH 2.2 ASCII with `$NodeData` and `$ElementData` | `generate/generate_gmsh.py` |
| `gmsh/tet_v22_binary.msh` | Same 6 linear tetrahedra as `tet_v22.msh` in MSH 2.2 binary encoding with `$NodeData` and `$ElementData` | `generate/generate_gmsh.py` |

## Regenerating

Install the pinned generator dependencies from `testdata/generate/requirements.txt` and run:

```sh
python3 testdata/generate/generate_all.py
```

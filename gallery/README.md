# Gallery

Reproducible scene files rendered in CI and compared against `tests/baselines/`. Each scene uses small meshes from `testdata/` with a fixed image size (`[800, 600]`), fixed MSAA (`4`), fixed camera, and fixed time step.

| Scene | What it shows |
| --- | --- |
| `surface-mixed-linear.json` | External surface of `mixed_linear.vtu` (hex, wedge, pyramid, tet) colored by point scalar `linear` with Viridis, right colorbar, and axes triad. |
| `surface-cell-id.json` | Explicit `surface` filter on `mixed_linear.vtu` colored by flat cell scalar `cell:cell_id` with Cividis and right colorbar. |
| `surface-velocity-magnitude.json` | External surface of `hex_linear.vtu` colored by vector magnitude `velocity:magnitude` with diverging Cool-Warm and a bottom horizontal colorbar. |
| `gmsh-quadratic-tetra.json` | Gmsh MSH 4.1 10-node quadratic tetrahedron mesh (`tet2_v41.msh`) colored by point scalar `linear` with Cividis and right colorbar. |
| `series-transient-last.json` | ParaView `.pvd` 3-step transient series (`series.pvd`) evaluated at `"time": "last"` (`t = 1.0`) with `"range": "auto_all_steps"` (`[0.0, 7.0]`). |
| `slice.json` | Diagonal `slice` plane (`normal: [1.0, -0.8, 0.9]`) through `hex_linear.vtu` colored by `linear` with `surface_with_edges` inside a `silver` outer element wireframe. |
| `contour.json` | Multi-isovalue `contour` (`values: [2.5, 5.0, 7.5, 10.0]`) of `linear` on `hex_linear.vtu` colored with Turbo inside a `silver` outer element wireframe. |
| `clip.json` | Exact plane `clip` (`normal: [1.0, -0.8, 0.9]`, `keep: "back"`) of `hex_linear.vtu` colored by `linear` with `surface_with_edges` inside a `silver` outer element wireframe. |
| `threshold.json` | Whole-cell `threshold` (`field: "linear"`, `min: 0.0`, `max: 9.0`) on `hex_linear.vtu` keeping 5 of 8 hexahedra with `surface_with_edges` inside a `silver` outer element wireframe. |

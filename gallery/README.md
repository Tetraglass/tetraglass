# Gallery

Reproducible scene files rendered in CI and compared against `tests/baselines/`. Each scene uses small meshes from `testdata/` with a fixed image size (`[800, 600]`), fixed MSAA (`4`), fixed camera, and fixed time step.

| Scene | What it shows |
| --- | --- |
| `surface-mixed-linear.json` | External surface of `mixed_linear.vtu` (hex, wedge, pyramid, tet) colored by point scalar `linear` with Viridis, right colorbar, and axes triad. |
| `surface-cell-id.json` | Explicit `surface` filter on `mixed_linear.vtu` colored by flat cell scalar `cell:cell_id` with Cividis and right colorbar. |
| `surface-velocity-magnitude.json` | External surface of `hex_linear.vtu` colored by vector magnitude `velocity:magnitude` with diverging Cool-Warm and a bottom horizontal colorbar. |
| `gmsh-quadratic-tetra.json` | Gmsh MSH 4.1 10-node quadratic tetrahedron mesh (`tet2_v41.msh`) colored by point scalar `linear` with Cividis and right colorbar. |
| `series-transient-last.json` | ParaView `.pvd` 3-step transient series (`series.pvd`) evaluated at `"time": "last"` (`t = 1.0`) with `"range": "auto_all_steps"` (`[0.0, 7.0]`). |

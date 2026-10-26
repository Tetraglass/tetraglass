# Roadmap

Milestones run in order. `/next-task` picks the first unticked task of the first unfinished milestone. Each task should fit one or two evening sessions; split bigger ones into lettered sub-tasks (M2.3a, M2.3b) before starting. Dates are estimates at roughly 8 to 10 hours a week.

Status: **M0 complete (2026-10-23); M1 complete (2026-10-24); M2 complete (2026-10-24); M3 complete (2026-10-25); M4 complete (2026-10-26); M5 complete (2026-10-26); M6 in progress.**

---

## M0 Foundations (Nov–Dec 2026 — Completed 2026-10-23)

**Gate:** a .vtu file renders colored by a field, both inside an egui app and to a PNG.

- [x] **M0.1 Bootstrap.** Confirm the `tetraglass-*` crate names are still free, then create the workspace, xtask, CI and license policy (`/bootstrap-workspace`). Reserving the names on crates.io is the developer's step. *Done when:* `cargo xtask verify` passes locally and in CI.
- [x] **M0.2 Test data v1.** Generate the required set in `/make-test-data`. *Done when:* every file listed there exists, with scripts and `testdata/README.md`.
- [x] **M0.3 Reader evaluation.** Try vtkio on every `.vtu` test file and mshio on every `.msh` file. Record what loads, what fails (higher-order, polyhedra, appended or compressed data, `$NodeData`), and decide reuse or own reader. *Done when:* a decision entry exists in `docs/decisions.md` and the open question is closed.
- [x] **M0.4 Core types.** `UnstructuredGrid` (points, cell types, offsets, connectivity, polyhedron face streams), `CellType` including Lagrange order, typed `DataArray` with components and `Arc` storage, point and cell attributes, bounds and cached ranges. *Done when:* unit tests cover construction, validation of inconsistent input, and range computation. Confirms D-013 (u32 indices).
- [x] **M0.5 VTU reader.** Per the M0.3 decision: ASCII, binary (base64) and appended data, zlib compression via a pure-Rust crate; linear cells first, higher-order and polyhedra stored (rendering them comes in M2). Plus a minimal `.vtu` writer for round trips. *Done when:* all `testdata/vtk/*.vtu` files load and round-trip; corruption tests pass.
- [x] **M0.6 External surface.** Face hashing over linear cells, keeping faces with one owner; carry point arrays and map cell arrays through source cell ids. For higher-order cells, use corner nodes as a stopgap until M2. *Done when:* surface of `hex_linear` has the analytic face count; mixed mesh has no internal faces.
- [x] **M0.7 Headless renderer.** `headless` helper that creates its own device; surface pipeline with positions relative to center, normals, scalar attribute and a viridis colormap texture; offscreen color and depth; PNG readback. *Done when:* an integration test renders `mixed_linear.vtu` colored by `linear` to `target/render-tests/` and the image looks right (`/render-check`).
- [x] **M0.8 egui viewport widget.** Renders in egui-wgpu's `prepare` into an offscreen target and shows it as a texture; orbit, pan and zoom camera; resize handling. Example `examples/egui_viewport.rs` loads a .vtu given on the command line and offers a field picker. *Done when:* the example runs on the developer's machine at interactive frame rates for the test meshes.
- [x] **M0.9 CI rendering.** CI runs the render integration test on lavapipe and stores the first baseline image. *Done when:* CI is green with the render test enabled.
- [x] **M0.10 Gate.** `/milestone-gate` (`docs/milestones/m0-gate.md`, completed 2026-10-23).

## M1 Agent loop (Jan–Mar 2027 — Completed 2026-10-24)

**Gate:** a fresh agent session, given only the CLI and `docs/scene-format.md`, goes from a test file to a correct PNG with no compile step.

- [x] **M1.1 Scene types.** Scene structs per `docs/scene-format.md` v1 (sources, view with representations, camera, output; the `surface` filter only), serde and `JsonSchema` derives, explicit defaults.
- [x] **M1.2 Validation.** Semantic validation against loaded data, errors with JSON paths and edit-distance suggestions.
- [x] **M1.3 Evaluation and cache.** Lazy pipeline evaluation cached by input hash; colormap or camera changes never re-run filters (test with a counter).
- [x] **M1.4 CLI.** `inspect` (text and `--json`), `render` (PNG plus JSON summary), `validate`, `schema`. Snapshot test of the schema output.
- [x] **M1.5 Annotations.** Colorbar with tick labels, title text, axes triad, all drawn by the renderer. Pick a pure-Rust font rasterizer with an allowlisted license.
- [x] **M1.6 Gmsh reader.** MSH 4.1 and 2.2, ASCII and binary, `$NodeData` and `$ElementData`, higher-order permutation tables into VTK ordering (`/add-reader`).
- [x] **M1.7 Time series.** `.pvd` collections and multi-step Gmsh files as lazy time series; `"time"` in the scene output.
- [x] **M1.8 Viewer v0.** eframe desktop app: open a scene, hot reload on file change, time slider, field and colormap pickers.
- [x] **M1.9 Performance target.** Define the interactive target for M2 (mesh size, frame time, filter latency) and a synthetic benchmark mesh generated in code; record it as a decision.
- [x] **M1.10 Gallery.** At least five gallery scenes with baselines; `author-scene` skill examples checked against the real CLI.
- [x] **M1.11 Gate.** `/milestone-gate`, including the fresh-agent test (`docs/milestones/m1-gate.md`, completed 2026-10-24).

## M2 Core filters (Apr–Jul 2027 — Completed 2026-10-24)

**Gate:** release 0.1 on crates.io with docs and a gallery.

- [x] **M2.1a Linearize linear and polyhedral cells.** `TetView` in `tetraglass-core`; `linearize` in `tetraglass-filters` with face-consistent diagonal decomposition of linear 3D cells (`Tetra`, `Hexahedron`, `Wedge`, `Pyramid`), centroid decomposition of `Polyhedron` cells, source-cell map, and stopgap corner decomposition for higher-order cells; volume, watertightness, and linear-field tests on `testdata/vtk/*.vtu` and synthetic meshes.
- [x] **M2.1b Linearize higher-order cells and property tests.** Reference-space subdivision at a configurable level (`LinearizeParams`) with Lagrange and serendipity basis evaluation for positions and point fields (`Quadratic*`, `Lagrange*`); property tests verifying volume preservation, exact linear field interpolation, and source-cell mapping across all cell families.
- [x] **M2.2 Curved surfaces.** External surface of higher-order cells with tessellated faces; wireframe from original element edges.
- [x] **M2.3a Marching tetrahedra kernel and slice filter.** Shared marching-tetrahedra isosurface kernel (`D-006`) and plane `slice` filter on `TetView` (`D-005`) in `tetraglass-filters`; analytic area, all-cell-family, degenerate vertex-plane, linear-field interpolation, and chunk-determinism tests (`/add-filter`).
- [x] **M2.3b Slice scene integration and gallery scene.** `Filter::Slice` in `tetraglass-scene`, semantic validation (`origin` default, non-zero `normal`), JSON Schema snapshot, `docs/scene-format.md`, and slice gallery scene with baseline (`/author-scene`, `/render-check`).
- [x] **M2.4 Contour.** Multiple isovalues, all arrays interpolated.
- [x] **M2.5 Clip.** Exact and crinkle modes.
- [x] **M2.6 Threshold.**
- [x] **M2.7 Picking.** GPU ID buffer; value probe in the widget and viewer.
- [x] **M2.8 Interactive clipping.** Shader clip plane while dragging; exact clip on a worker thread, swapped in when ready.
- [x] **M2.9 Performance pass.** Meet the M1.9 target or record what was cut.
- [x] **M2.10 Documentation.** mdBook guide (getting started, embedding, scene format, filters), rustdoc examples, gallery covering every filter.
- [x] **M2.11 Gate and release 0.1.** `/milestone-gate` (`docs/milestones/m2-gate.md`, completed 2026-10-24).

## M3 Web and export (Aug–Oct 2027 — Completed 2026-10-25)

**Gate:** one CFD result shown both in Cycles (via `render --engine cycles`) and in the browser.

- [x] **M3.1 Web viewer.** WASM build on WebGPU, drag-and-drop and URL loading, a clear message without WebGPU.
- [x] **M3.2 Shareable links.** A link to a scene plus data opens in the web viewer.
- [x] **M3.3 Transparency.** Weighted blended order-independent transparency.
- [x] **M3.4 glTF export.** Baked colors, normalized scalar in a second texture coordinate, one .glb per time step.
- [x] **M3.5 Bundle format.** Write `docs/bundle-format.md`, then the export crate's bundle writer (`/blender-bridge`).
- [x] **M3.6 Blender scripts.** GPL importer and headless render driver in `blender/`, installable as an add-on.
- [x] **M3.7 Cycles from the CLI.** `render --engine cycles`, Blender discovery, progress streaming, `--save-blend`.
- [x] **M3.8 Gate.** `/milestone-gate` (`docs/milestones/m3-gate.md`, completed 2026-10-25).

## M4 Flow and volume (Nov 2027–Feb 2028 — Completed 2026-10-26)

**Gate:** release 0.2 with a stable scene schema.

- [x] **M4.1 Cell locator.** BVH over linearized tets for point location.
- [x] **M4.2a Streamline filter, RK4/RK45 integrators, seeds, and tube geometry.** `PolylineSet` and `to_tube_surface` in `tetraglass-core`; `streamlines` filter in `tetraglass-filters` with point, line, and plane seeds, fixed-step RK4 and adaptive Dormand-Prince RK45 integration (`forward`, `backward`, `both`), `TetLocator::locate_with_hint` stepping, point/cell attribute interpolation, and analytic/all-cell-family/determinism tests (`/add-filter`).
- [x] **M4.2b Streamline scene integration, line/tube styles, and gallery scene.** `Filter::Streamlines` and `"lines"` / `"tubes"` representation styles in `tetraglass-scene`, semantic validation, colormapped line/tube rendering in `tetraglass-render`, JSON Schema snapshot, `docs/scene-format.md`, and streamline gallery scene with baseline (`/author-scene`, `/render-check`).
- [x] **M4.3 Glyphs.** Instanced arrows, cones and spheres, oriented and scaled by a vector.
- [x] **M4.4a Regular grid resampling filter and scene integration.** `RegularGrid` in `tetraglass-core`; `resample` filter (`ResampleParams`, `resample`, `resample_with_locator`, `resample_with_chunk_size`, `resample_grid`) in `tetraglass-filters` using `TetLocator` to sample point and cell arrays onto a `[nx, ny, nz]` regular grid with domain occupancy mask; `Filter::Resample` (`"type": "resample"`), `EvaluatedDataset::Volume`, and `transfer_function` opacity control points in `tetraglass-scene` (`/add-filter`).
- [x] **M4.4b GPU volume ray marching, transfer function editor, and gallery scene.** 3D scalar texture upload and front-to-back ray-marching pass with depth-buffer termination and 1D colormap + opacity transfer function in `tetraglass-render`; interactive transfer function curve editor in `tetraglass-viewer`; `gallery/volume.json` and visual regression baseline (`/author-scene`, `/render-check`).
- [x] **M4.5 Animation export.** PNG sequences from the rasterizer and frame ranges through Cycles.
- [x] **M4.6 Schema freeze.** Scene schema v1 final, migration tests from earlier drafts.
- [x] **M4.7 Gate and release 0.2.** `/milestone-gate` (`docs/milestones/m4-gate.md`, completed 2026-10-26).

## M5 Path tracer and beyond (Mar–May 2028 — Completed 2026-10-26)

**Gate:** a scene renders with the built-in compute-shader progressive path tracer, SSAO, and legacy `.vtk` input across the CLI and viewer.

- [x] **M5.1 Legacy VTK (`.vtk`) unstructured grid reader.** Pure-Rust reader (`read_vtk`, `read_vtk_series`, `read_vtk_from_path`) in `tetraglass-io` for legacy VTK DataFile versions 2.0–5.1 (`DATASET UNSTRUCTURED_GRID`, `ASCII` and big-endian `BINARY` encodings, classic 4.x `CELLS` and 5.x `OFFSETS`/`CONNECTIVITY`, `CELL_TYPES`, polyhedra, and `POINT_DATA`/`CELL_DATA` with `SCALARS`, `VECTORS`, `NORMALS`, `TENSORS`, and `FIELD` arrays), wired into `tetraglass-io`, `tetraglass-cli`, and `tetraglass-viewer` (`/add-reader`, `/make-test-data`).
- [x] **M5.2 Screen-space ambient occlusion (SSAO).** Depth/normal reconstruction and hemisphere-kernel SSAO + edge-preserving blur pass in `tetraglass-render`, with interactive SSAO controls in `tetraglass-egui` and `tetraglass-viewer` (`/render-check`).
- [x] **M5.3a GPU triangle BVH and compute-shader progressive path tracer.** Linear BVH over rendered triangles and compute-shader progressive path-tracing pipeline (`path_tracer.wgsl`, `PathTracer`, `PathTraceConfig`) in `tetraglass-render` with multi-bounce shading, soft shadows, colormap evaluation, and progressive sample accumulation across all wgpu backends including WebGPU.
- [x] **M5.3b Progressive path tracer in the viewer and CLI.** Interactive viewport accumulation when stationary in `tetraglass-egui` and `tetraglass-viewer`, plus headless path-traced rendering and tests (`/render-check`).
- [x] **M5.4 Persistent Blender worker and live link.** Socket worker mode in `blender/render.py` and client support in `tetraglass-cli` / `tetraglass-viewer` to keep a Blender session warm across renders and push live camera/scene updates (`/blender-bridge`).
- [x] **M5.5 Gate.** `/milestone-gate` (`docs/milestones/m5-gate.md`, completed 2026-10-26).

## M6 Ergonomics, parallel I/O and array filters (from Jun 2028)

**Gate:** release 0.3 with the `tetraglass` umbrella crate and fluent Rust `SceneBuilder` API, `"points"` representation rendering, `.pvtu` partitioned VTK reader, and `warp_by_vector` / `calculator` filters.

- [x] **M6.1 Umbrella `tetraglass` crate and fluent Rust `SceneBuilder` API.** Create the `tetraglass` umbrella crate (`crates/tetraglass`) re-exporting workspace crates behind feature flags (`io`, `filters`, `scene`, `export`, `render`, `egui`, `viewer`, `full`), and implement the typed fluent Rust builder API (`Scene::builder()`, `SceneBuilder`, `DatasetRef`, filter builders, `RepresentationBuilder`, `CameraBuilder`, `OutputBuilder`) in `tetraglass-scene`.
- [ ] **M6.2 Screen-space `"points"` representation rendering.** Implement screen-space point-cloud rendering (`draw_points`, `point_size`) in `tetraglass-render` supporting solid, point-scalar, and cell-scalar coloring, depth testing, WBOIT transparency, and GPU picking; wire `RepresentationStyle::Points` and `Representation::point_size` in `tetraglass-cli` and `tetraglass-viewer`; add `gallery/points.json` with a visual regression baseline (`/render-check`).
- [ ] **M6.3 Parallel VTK XML (`.pvtu`) partitioned unstructured grid reader.** Pure-Rust `.pvtu` reader (`read_pvtu_with_loader`, `read_pvtu_from_path`) in `tetraglass-io` parsing `<PUnstructuredGrid>` metadata and piece `<Piece Source="..."/>` references, merging partitioned piece `UnstructuredGrid`s with ghost-cell (`vtkGhostType`) filtering, wired into `tetraglass-io`, `tetraglass-cli`, and `tetraglass-viewer` (`/add-reader`, `/make-test-data`).
- [ ] **M6.4 `warp_by_vector` and `calculator` array filters.** Implement `warp_by_vector` (displacing mesh/tet coordinates by `scale_factor * vector_field`) and `calculator` (evaluating mathematical scalar/vector expressions over point and cell arrays into a derived `DataArray`) in `tetraglass-filters` (`/add-filter`).
- [ ] **M6.5 Gate and release 0.3.** `/milestone-gate` (`docs/milestones/m6-gate.md`, `CHANGELOG.md`).

# Progress log

Newest entries at the bottom. `/next-task` appends one entry per finished task using this template:

```
## YYYY-MM-DD: <task id> <title>
- Done:
- Tests:
- Decisions:
- Dependencies added:
- Follow-ups:
```

## 2026-10-22: Planning package imported
- Done: design plan, decisions, roadmap, scene format draft, Antigravity rules and skills added to the repository.
- Tests: none yet.
- Decisions: D-001 to D-017 recorded, including the name Tetraglass.
- Dependencies added: none.
- Follow-ups: run `/bootstrap-workspace` (M0.1).

## 2026-10-23: M0.1 Bootstrap
- Done: confirmed all 10 `tetraglass` / `tetraglass-*` crate names are free on crates.io; created Cargo workspace (`Cargo.toml`, `rust-toolchain.toml`, `.cargo/config.toml`), CI workflow (`.github/workflows/ci.yml`), `xtask`, license policy (`deny.toml`), and initial crate skeletons (`tetraglass-core`, `tetraglass-io`, `tetraglass-filters`, `tetraglass-scene`, `tetraglass-render`, `tetraglass-egui`, `tetraglass-cli`) with repository URL `https://github.com/Tetraglass/tetraglass` and `rust-version = "1.95"`.
- Tests: `cargo xtask verify` (format, clippy, unit/doc tests, `wasm32-unknown-unknown` check on `-core`, `-io`, `-filters`, `-scene`, and `cargo deny check`).
- Decisions: confirmed D-017 crate names remain available; pinned GPU/UI stack to `egui = "0.37.0"`, `egui-wgpu = "0.37.0"`, `wgpu = "30.0.1"`.
- Dependencies added: `egui` (0.37.0), `egui-wgpu` (0.37.0), `wgpu` (30.0.1), `bytemuck` (1, `derive`), `log` (0.4), `thiserror` (2).
- Follow-ups: run `/make-test-data` (M0.2); reserve crate names on crates.io before first public release.

## 2026-10-23: M0.2 Test data v1
- Done: created deterministic test mesh generators (`testdata/generate/generate_vtk.py`, `testdata/generate/generate_gmsh.py`, `testdata/generate/generate_all.py`, `testdata/generate/requirements.txt`), generated the v1 VTK and Gmsh test suite (`vtk/tet_linear.vtu`, `vtk/tet_linear_binary.vtu`, `vtk/tet_linear_appended.vtu`, `vtk/hex_linear.vtu`, `vtk/mixed_linear.vtu`, `vtk/tet_quadratic.vtu`, `vtk/hex_lagrange_p3.vtu`, `vtk/polyhedra.vtu`, `vtk/series.pvd` + `vtk/series_{0,1,2}.vtu`, `gmsh/tet2_v41.msh`, `gmsh/tet2_v41_binary.msh`, `gmsh/tet_v22.msh`, `gmsh/tet_v22_binary.msh`) with analytic fields (`linear`, `radius`, `velocity`, `cell_id`, plus `transient` on the time series), documented every file in `testdata/README.md`, and added `testdata/reference/.gitkeep`.
- Tests: `testdata/generate/generate_all.py` verification suite and `crates/tetraglass-io/tests/testdata_manifest.rs` via `cargo xtask verify`.
- Decisions: included both classic (`faces`/`faceoffsets`) and VTK 9.4+ (`face_connectivity`/`face_offsets`/`polyhedron_to_faces`/`polyhedron_offsets`) face-stream arrays in `vtk/polyhedra.vtu`, and added inline-binary and zlib-appended variants of `tet_linear` plus binary MSH 4.1/2.2 variants for M0.3 reader evaluation.
- Dependencies added: none in Cargo; pinned `vtk==9.7.0`, `gmsh==4.15.2`, `numpy==2.5.3` in `testdata/generate/requirements.txt`.
- Follow-ups: M0.3 Reader evaluation (evaluate `vtkio` and `mshio` against the `testdata/` suite).

## 2026-10-23: M0.3 Reader evaluation
- Done: evaluated `vtkio` 0.7.0-rc2 on all 11 `.vtu` files (plus `series.pvd`) and `mshio` 0.4.2 on all 4 `.msh` files in `testdata/`.
  - `vtkio` 0.7.0-rc2: loaded `tet_linear.vtu` (ASCII), `tet_linear_binary.vtu` (inline base64), `tet_linear_appended.vtu` (zlib appended), `hex_linear.vtu`, `mixed_linear.vtu`, `tet_quadratic.vtu` (type 24), `hex_lagrange_p3.vtu` (type 72), and `series_{0,1,2}.vtu`. Failed on `polyhedra.vtu` (`CellsVisitor` rejects any `DataArray` inside `<Cells>` other than `connectivity`, `offsets`, `types` and stores no face streams), failed on `series.pvd`, and its `compression` feature pulls in `xz2`/`lzma-sys` (a C dependency that breaks `wasm32-unknown-unknown`, violating D-008).
  - `mshio` 0.4.2: parsed `$Nodes` and `$Elements` in `tet2_v41.msh` (ASCII) and `tet2_v41_binary.msh` (binary), but silently discarded all `$NodeData` and `$ElementData` sections (`MshData` has no fields for post-processing views) and rejected `tet_v22.msh` and `tet_v22_binary.msh` (MSH 2.2 unsupported).
- Tests: ran scratch evaluation harness against all 11 `.vtu` and 4 `.msh` files; verified workspace with `cargo xtask verify`.
- Decisions: D-018 — write focused in-house readers in `tetraglass-io` for VTK (`.vtu`, `.pvd`, `.vtk`) and Gmsh (`.msh` 4.1 and 2.2) using pure-Rust building blocks (`quick-xml`, `base64`, `flate2` with `rust_backend`), closing the open question in `docs/decisions.md`.
- Dependencies added: none.
- Follow-ups: M0.4 Core types (`UnstructuredGrid`, `CellType`, `DataArray`, attributes, bounds and cached ranges).

## 2026-10-23: M0.4 Core types
- Done: implemented `UnstructuredGrid`, `BoundingBox`, `CellType` (linear, quadratic, arbitrary-order Lagrange with order inference, and polyhedron), `PolyhedronFaces` / `PolyhedronFaceIter`, typed `DataArray` (`f32`, `f64`, `i32`, `i64`, `u8`) with `ComponentCount` (`1`, `3`, `6`, `9`), `Arc` backing storage, cached per-component and magnitude `ValueRange`s, ordered `Attributes`, `CoreError`, and checked `u32` index conversions (`try_u32_from_u64`, `try_u32_from_i64`, `try_u32_from_usize`).
- Tests: unit and doc tests in `tetraglass-core` covering `u32` index checks, Lagrange order inference, valid linear/higher-order/polyhedral grid construction, inconsistent input rejection, `Arc` sharing, and cached component/magnitude range computation.
- Decisions: confirmed D-013 (`u32` connectivity and offset indices, checked on load/construction).
- Dependencies added: linked existing workspace dependency `thiserror` into `tetraglass-core`.
- Follow-ups: M0.5 VTU reader and writer in `tetraglass-io`.

## 2026-10-23: M0.5 VTU reader
- Done: implemented pure-Rust VTK XML `.vtu` reader (`read_vtu`, `read_vtu_from_path`) supporting `ascii`, `binary` (inline base64), and `appended` (`base64` and `raw`) encodings, uncompressed and `vtkZLibDataCompressor`-compressed payloads, `UInt32`/`UInt64` headers, `LittleEndian`/`BigEndian` byte orders, linear, quadratic, Lagrange, and polyhedral cells (both classic `faces`/`faceoffsets` and VTK 9.4+ `face_connectivity` arrays), plus an ASCII `.vtu` writer (`write_vtu`, `write_vtu_to_bytes`) and format detection helpers (`detect_format_from_bytes`, `detect_format_from_extension`).
- Tests: `crates/tetraglass-io/tests/vtu_roundtrip.rs` covering loading and round-tripping all 11 `testdata/vtk/*.vtu` files, ASCII/binary/appended equivalence, quadratic & Lagrange node-ordering checks, classic & VTU 2.3 polyhedra parsing, and truncation/corruption error handling.
- Decisions: none (implemented per D-018).
- Dependencies added: `quick-xml` (0.41, XML streaming parser), `base64` (0.22, binary/appended base64 decoding), `flate2` (1.1 with `rust_backend`, pure-Rust zlib decompression).
- Follow-ups: M0.6 External surface extraction in `tetraglass-filters`.

## 2026-10-23: M0.6 External surface
- Done: added `SurfaceMesh` in `tetraglass-core` (carrying shared `Arc` points/point attributes, triangulated surface faces, logical face counts, source cell maps, per-triangle mapped cell attributes, `total_area()`, and `to_unstructured_grid()`) and implemented `extract_external_surface` in `tetraglass-filters` using deterministic face hashing over linear cells, higher-order cell corner nodes (stopgap until M2.2), polyhedron face streams, and 2D cells.
- Tests: `crates/tetraglass-filters/tests/external_surface.rs` plus doc tests verifying analytic face counts and surface areas on `hex_linear.vtu` (24 outer quad faces, 48 triangles, area 24.0), zero internal faces on `mixed_linear.vtu` (12 outer faces, 18 triangles), coverage of `tet_linear.vtu`, `tet_quadratic.vtu`, `hex_lagrange_p3.vtu`, `polyhedra.vtu`, empty grids, `Arc` point-array sharing, and source-cell mapping of `cell_id`.
- Decisions: placed `SurfaceMesh` in `tetraglass-core` (and re-exported from `tetraglass-filters`) so `tetraglass-render` can consume `SurfaceMesh` directly without depending on `tetraglass-filters`.
- Dependencies added: linked existing workspace dependency `thiserror` into `tetraglass-filters` (and `tetraglass-io` as a dev-dependency for tests).
- Follow-ups: M0.7 Headless renderer in `tetraglass-render`.

## 2026-10-23: M0.7 Headless renderer
- Done: implemented `Renderer`, `GpuSurface`, `OffscreenTarget` (1x and 4x MSAA with `Depth32Float`), `Camera` (dataset-center-relative orbit/pan/zoom and perspective projection), 256-entry 1D sRGB colormaps (`Viridis`, `Cividis`, `CoolWarm`), WGSL surface pipeline (`common.wgsl` + `surface.wgsl` with vertex scalar attributes, cell scalar storage buffer, and three-point lighting in linear RGB), pure-Rust PNG encoder/decoder (`png.rs`), and `headless` (`HeadlessContext`, `RenderedImage`, `ImageComparison` with `/render-check` diff generation).
- Tests: unit tests in `tetraglass-render` (`camera`, `colormap`, `png`) and integration tests in `crates/tetraglass-render/tests/render_mixed_linear.rs` rendering `testdata/vtk/mixed_linear.vtu` colored by point field `linear` (`target/render-tests/mixed_linear_linear.png`) and cell field `cell_id` (`target/render-tests/mixed_linear_cell_id.png`), plus determinism and validation checks; visually inspected both PNGs per `/render-check`.
- Decisions: none (implemented per D-002, D-014, and `.agents/rules/rendering.md`).
- Dependencies added: linked existing workspace dependencies `flate2`, `log`, and `thiserror` into `tetraglass-render`.
- Follow-ups: M0.8 egui viewport widget in `tetraglass-egui`.

## 2026-10-23: M0.8 egui viewport widget
- Done: implemented `ViewportState`, `ViewportWidget`, `ViewportResponse`, `FieldOption`, and `available_field_options` in `tetraglass-egui` (rendering offscreen during `egui_wgpu::CallbackTrait::prepare`, registering/updating a non-sRGB `Rgba8Unorm` texture view `OffscreenTarget::egui_texture_view` with `egui_wgpu::Renderer`, and handling orbit/pan/zoom/reset camera input and pixel-size resize tracking), plus the interactive desktop example `crates/tetraglass-egui/examples/egui_viewport.rs` with `.vtu` CLI argument loading, field picker, colormap picker, and log-scale toggle.
- Tests: `crates/tetraglass-egui/tests/viewport_offscreen.rs` covering field enumeration (`Solid`, scalar and vector point/cell fields), offscreen target resize, `egui_wgpu::Renderer::update_buffers` execution of the `prepare` callback, and style/range updates.
- Decisions: none (implemented per D-003 and `.agents/rules/rendering.md`).
- Dependencies added: `eframe` (0.37.0 with `wgpu` and `default_fonts`, matching `egui 0.37.0` / `egui-wgpu 0.37.0` / `wgpu 30.0.1`, used as a dev-dependency for `examples/egui_viewport.rs`); linked existing workspace dependencies `tetraglass-core` and `log` into `tetraglass-egui`.
- Follow-ups: M0.9 CI rendering (run render integration tests on Mesa lavapipe in CI and store the first baseline image).

## 2026-10-23: M0.9 CI rendering
- Done: committed initial baseline images (`tests/baselines/mixed_linear_linear.png` and `tests/baselines/mixed_linear_cell_id.png` produced on Mesa `lavapipe` in CI), made baseline comparison mandatory in `crates/tetraglass-render/tests/render_mixed_linear.rs`, added `TETRAGLASS_FORCE_FALLBACK_ADAPTER=1` support in `HeadlessContext::new()`, enabled `x11` on `eframe` so Linux CI builds all targets cleanly, and configured `.github/workflows/ci.yml` to upload `target/render-tests/` and `target/render-diffs/` artifacts on every run.
- Tests: `crates/tetraglass-render/tests/render_mixed_linear.rs` verifying both point-colored (`linear`) and cell-colored (`cell_id`) renders against `tests/baselines/` on CI (`lavapipe`) and locally.
- Decisions: none (implemented per `.agents/rules/testing.md` and `/render-check`).
- Dependencies added: enabled `x11` feature on existing `eframe 0.37.0` dev-dependency for Linux `winit` compilation.
- Follow-ups: M0.10 Milestone gate (`/milestone-gate`).

## 2026-10-23: M0.10 Milestone M0 gate & retrospective
- Done: verified the M0 gate (*"a `.vtu` file renders colored by a field, both inside an egui app and to a PNG"*), extended `crates/tetraglass-egui/tests/viewport_offscreen.rs` to also execute `egui_wgpu::Renderer::render` into an offscreen target and write `target/render-tests/egui_viewport_mixed_linear.png`, and recorded the gate evidence and images in `docs/milestones/m0-gate.md`.
- Retrospective:
  - **Planned vs. actual time:** M0 was estimated for Nov–Dec 2026 and completed ahead of schedule in two days (2026-10-22 to 2026-10-23) because each task was scoped cleanly to a single crate boundary.
  - **What was cut or moved:** Nothing was cut; in-house `.vtu` parsing (D-018) and `SurfaceMesh` placement in `tetraglass-core` simplified downstream integration in `tetraglass-filters`, `tetraglass-render`, and `tetraglass-egui`.
  - **What to carry into M1:** Keep enabling Linux windowing features (`x11` without `pkg-config` build-time requirements) whenever adding desktop binaries (`tetraglass-viewer` in M1.8), and keep uploading CI render artifacts so `lavapipe` baselines stay easy to inspect and update. M1 task breakdown (`M1.1`–`M1.11`) looks well-sized as written.
- Next: M1.1 Scene types (`tetraglass-scene`).

## 2026-10-23: M1.1 Scene types
- Done: implemented the v1 serializable scene model in `tetraglass-scene` (`Scene`, `Source`, `SourceFormat`, internally tagged `Filter::Surface`, `View`, `Representation`, `RepresentationStyle`, `ColorSpec` with CSS color name/hex/RGB resolution, `ColormapName`, `RangeSpec`, `ColorbarSpec`, `CameraConfig`, `ProjectionKind`, `LightingPreset`, `BackgroundSpec`, `Output`, `TimeSelector`, `FramesConfig`, `RenderEngine`, `CyclesConfig`) with `#[serde(deny_unknown_fields, rename_all = "snake_case")]`, explicit default functions, `schemars::JsonSchema` generation (`scene_json_schema`, `scene_json_schema_string`), and array reference parsing/component resolution (`ParsedArrayRef`, `ArrayAssociation`, `ComponentSelector`, `ResolvedComponent`).
- Tests: unit and doc tests in `tetraglass-scene` covering minimal-scene default population, full-scene round-trip serialization, `deny_unknown_fields` rejection across all sections, untagged union parsing, array reference parsing and component resolution across scalar/vector/tensor arrays, and JSON Schema generation with field descriptions.
- Decisions: none (implemented per D-004, `docs/scene-format.md`, and `.agents/rules/scene-schema.md`).
- Dependencies added: `serde` (1.0.229 with `derive`), `serde_json` (1.0.154), and `schemars` (1.2.3) in `[workspace.dependencies]` and `tetraglass-scene`; linked existing workspace dependency `thiserror` into `tetraglass-scene`.
- Follow-ups: M1.2 Validation (semantic validation against loaded data, errors with JSON paths and edit-distance suggestions).

## 2026-10-23: M1.2 Validation
- Done: implemented semantic scene validation in `tetraglass-scene` (`validate_scene`, `validate_scene_structure`, `validate_scene_with_metadata`, `Scene::validate`, `Scene::validate_structure`, `Scene::validate_with_metadata`, `DatasetMetadata`, `ArrayMetadata`, `ValidationIssue`, `ValidationReport`, `closest_match`, `damerau_levenshtein`) checking schema version, `snake_case` naming, source format inference, filter graph existence and acyclicity, representation parameters, CSS/RGB colors, camera framing/angles, `log_scale` positivity, and array/association/component validity against loaded datasets with JSON paths and edit-distance suggestions.
- Tests: `crates/tetraglass-scene/tests/validation.rs` covering valid scenes with `testdata/vtk/mixed_linear.vtu`, array typo suggestions (`"presure"` -> `"pressure"`), association disambiguation (`"point:cell_id"` -> `"cell:cell_id"`), invalid scalar/vector component selectors, filter cycles, `snake_case` name suggestions, color name suggestions, `log_scale` range checks, camera/output bounds, and Damerau-Levenshtein transpositions.
- Decisions: none (implemented per `docs/scene-format.md` and `.agents/rules/scene-schema.md`).
- Dependencies added: none (implemented Damerau-Levenshtein edit distance in-house per `.agents/rules/dependencies.md`).
- Follow-ups: M1.3 Evaluation and cache (lazy pipeline evaluation cached by input hash; colormap or camera changes never re-run filters).

## 2026-10-23: M1.3 Evaluation and cache
- Done: implemented lazy pipeline evaluation and input-hash caching in `tetraglass-scene` (`SceneEvaluator`, `EvaluatedScene`, `EvaluatedRepresentation`, `EvaluatedColoring`, `EvaluatedDataset`, `PipelineHash`, `hash_unstructured_grid`) with deterministic 64-bit FNV-1a content hashing, implicit external surface deduplication for volume sources, camera bounds resolution, and `filter_executions`/`cache_hits` counters.
- Tests: `crates/tetraglass-scene/tests/evaluation_cache.rs` verifying with `filter_executions()` and `cache_hits()` that changing colormap, range, `color_by` array/component, camera, lights, background, or output never re-runs filters, unreferenced filters and hidden representations are skipped lazily, explicit `surface` filters and implicit volume surfaces share the same cache entry, and changing input mesh data invalidates the cache.
- Decisions: none (implemented per D-004 and `docs/design-plan.md`).
- Dependencies added: none.
- Follow-ups: M1.4 CLI (`inspect`, `render`, `validate`, `schema` in `tetraglass-cli`, plus a snapshot test of the schema output).


## 2026-10-23: M1.4 CLI

- Implemented the `tetraglass` command-line binary in `crates/tetraglass-cli/src/main.rs` supporting four commands:
  - `tetraglass schema`: prints the JSON Schema for scene format v1 to stdout.
  - `tetraglass inspect <file> [--json]`: loads a `.vtu` mesh and prints bounds, point/cell counts, cell-type histogram, and point/cell array metadata (component count, data type, value range) in human-readable text or structured JSON (`InspectSummary`).
  - `tetraglass validate <scene.json> [--json]`: parses and semantically validates a scene against its referenced source files relative to the scene directory, reporting JSON paths, messages, and edit-distance suggestions in text or JSON (`ValidationReport`), exiting with code `1` on errors.
  - `tetraglass render <scene.json> [-o <out.png>] [--engine <raster|cycles>]`: validates and evaluates a scene via `SceneEvaluator`, converts the evaluated representation to a `SurfaceMesh`, maps the scene camera (`orbit` / `look_at`, `auto_fit`), background color, MSAA sample count, and output dimensions to `tetraglass-render`'s `HeadlessRenderer`, writes the PNG image, and emits a JSON summary (`RenderSummary`) to stdout.
- Committed the authoritative v1 scene JSON Schema snapshot in `crates/tetraglass-cli/tests/snapshots/scene-v1.schema.json`.
- Tests added: integration test suite in `crates/tetraglass-cli/tests/cli.rs` covering `schema` snapshot comparison, `inspect` (text and `--json`), `validate` (valid and invalid scenes, text and `--json`), and `render` (verifying JSON summary output and comparing the rendered `800x600` PNG against `tests/baselines/mixed_linear_linear.png`).
- Decisions / dependencies: linked workspace dependencies `serde`, `serde_json`, and `thiserror` into `tetraglass-cli`.
- Follow-ups: M1.5 Annotations (colorbar with tick labels, title text, and axes triad drawn by the renderer with a pure-Rust font rasterizer).

## 2026-10-23: M1.5 Annotations
- Done: implemented renderer-drawn 2D/3D viewport annotations in `tetraglass-render` (`Annotations`, `ColorbarAnnotation`, `ColorbarPlacement`, `ColorbarTick`, `compute_colorbar_ticks`, `format_tick_value`, `FontRasterizer`, `EmbeddedFont`, `RasterizedText`, `TextMetrics`, `annotations.wgsl`, `Renderer::render_with_annotations`, and `HeadlessRenderOptions::annotations`) covering colorbars (`Right`, `Left`, `Bottom`, `Top`) with gradient ramps, outlines, tick marks, linear/log tick value labels and titles, top-centered viewport title text, and the bottom-left 3D orientation axes triad (`X`, `Y`, `Z`); wired annotations into `tetraglass-cli` (`tetraglass render`) and `tetraglass-egui` (`ViewportState::set_annotations`).
- Tests: unit tests in `crates/tetraglass-render/src/text.rs` and `crates/tetraglass-render/src/annotations.rs`, integration tests in `crates/tetraglass-render/tests/annotations.rs` and `crates/tetraglass-cli/tests/cli.rs`, and visual inspection of `target/render-tests/mixed_linear_annotated.png` and `target/render-tests/cli_mixed_linear_annotated.png` per `/render-check`.
- Decisions: selected `skrifa` (`0.44.0`, `MIT OR Apache-2.0`), `vello_cpu` (`0.2.0`, `Apache-2.0 OR MIT`), and `epaint_default_fonts` (`0.37.0`, `(MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0`) for pure-Rust font outline parsing and glyph coverage rasterization, reusing the exact font stack already present in `Cargo.lock` via `egui 0.37.0`.
- Dependencies added: promoted existing lockfile dependencies `skrifa` (0.44.0), `vello_cpu` (0.2.0), and `epaint_default_fonts` (0.37.0) to `[workspace.dependencies]` and linked them into `tetraglass-render`.
- Follow-ups: M1.6 Gmsh reader (MSH 4.1 and 2.2, ASCII and binary, `$NodeData` and `$ElementData`, higher-order permutation tables into VTK ordering).

## 2026-10-23: M1.6 Gmsh reader
- Done: implemented pure-Rust Gmsh `.msh` reader in `tetraglass-io` (`read_msh`, `read_msh_from_path`, `GmshElementSpec`, `gmsh_element_spec`, and `IoError::InvalidMshStructure`) supporting MSH `4.1` and `2.2` in both `ascii` (`0`) and `binary` (`1`, little- and big-endian) encodings, `$NodeData` and `$ElementData` sections mapped to `point_data` and `cell_data`, and higher-order node-ordering permutation tables into VTK canonical ordering (`QuadraticEdge`, `QuadraticTriangle`, `QuadraticQuad`, `QuadraticTetra`, `QuadraticHexahedron`, `QuadraticWedge`, `QuadraticPyramid`, `LagrangeQuadrilateral`, `LagrangeHexahedron`, `LagrangeWedge`, `LagrangeCurve`, `LagrangeTriangle`); wired `.msh` loading into `tetraglass-cli` (`inspect`, `validate`, `render`).
- Tests: `crates/tetraglass-io/tests/msh_roundtrip.rs` and `crates/tetraglass-cli/tests/cli.rs` covering loading and `.vtu`-writer round-tripping of all 4 `testdata/gmsh/*.msh` files (`tet2_v41.msh`, `tet2_v41_binary.msh`, `tet_v22.msh`, `tet_v22_binary.msh`), ASCII vs. binary equality, VTK canonical mid-edge node ordering on `tet2_v41.msh`, bijection and synthetic MSH remapping checks across all 20 supported Gmsh element types, CLI `inspect --json` on `.msh`, and truncation/corruption error handling.
- Decisions: none (implemented per D-009 and D-018).
- Dependencies added: none.
- Follow-ups: M1.7 Time series (`.pvd` collections and multi-step Gmsh files as lazy time series; `"time"` in the scene output).

## 2026-10-23: M1.7 Time series
- Done: implemented lazy time-series storage and evaluation across `tetraglass-core`, `tetraglass-io`, `tetraglass-scene`, and `tetraglass-cli`:
  - `tetraglass-core`: added `TimeSeries<E>`, `StepLoader<E>`, and `DEFAULT_STEP_CACHE_CAPACITY` (`4`) with lazy per-step loader closures, bounded LRU step cache eviction, `load_executions()` / `cache_hits()` counters, and time resolution helpers.
  - `tetraglass-io`: added ParaView `.pvd` collection reader (`PvdEntry`, `parse_pvd_entries`, `read_pvd_series_with_loader`, `read_pvd_series_from_path`, `read_pvd_from_path`), multi-step Gmsh `.msh` lazy series reader (`read_msh_series`, `read_msh_series_from_path`) that scans `$NodeData`/`$ElementData` headers and decodes per-step payloads only on demand, and unified series helpers (`read_vtu_series`, `read_series_from_path`).
  - `tetraglass-scene`: added `TimeSelector::resolve_step`, `DatasetMetadata::time_steps` validation for `output.time`, `SceneEvaluator::insert_series_source` / `evaluate_scene_at_time`, `EvaluatedScene::time_step` / `time_value`, and `RangePreset::AutoAllSteps` (`"auto_all_steps"`) scalar range evaluation across all steps.
  - `tetraglass-cli`: wired `.pvd` and multi-step `.msh` lazy series into `inspect` (populating `time_steps`), `validate` (checking `output.time`), and `render` (selecting the step requested by `output.time` and reporting `time_step` and `time` in `RenderSummary`).
- Tests: unit test `time_series_loads_lazily_and_evicts_lru_steps` in `tetraglass-core`; integration tests in `crates/tetraglass-io/tests/time_series.rs` (proving lazy `.vtu` loading for `testdata/vtk/series.pvd`, lazy ASCII/binary step decoding for multi-step `.msh`, and `.pvd` truncation/corruption handling), `crates/tetraglass-scene/tests/evaluation_cache.rs` (testing `"first"`, `"last"`, `0.5`, `{ "step": 1 }`, and `"auto_all_steps"`), and `crates/tetraglass-cli/tests/cli.rs` (`inspect --json` and `render` on `series.pvd`).
- Decisions: none (implemented per D-004, `docs/scene-format.md`, and `.agents/skills/add-reader/SKILL.md`).
- Dependencies added: none.
- Follow-ups: M1.8 Viewer v0 (eframe desktop app: open a scene, hot reload on file change, time slider, field and colormap pickers).

## 2026-10-24: M1.8 Viewer v0
- Done: created the `tetraglass-viewer` crate (`ViewerDocument`, `SceneFileWatcher`, `HotReloadOutcome`, `ViewerSession`, `ViewerApp`, `ViewerOptions`, `run_viewer`) and wired `tetraglass view <scene.json|mesh> [--frames <n>]` into `tetraglass-cli`:
  - Opens `.json` scene files as well as raw `.vtu`, `.pvd`, and `.msh` mesh files (synthesizing a default v1 `Scene`).
  - Hot-reloads when the scene file or its referenced source files change on disk (`SceneFileWatcher` combining filesystem metadata and 64-bit FNV-1a content hashing), reusing the `SceneEvaluator` cache when only view/coloring properties change and keeping the last valid render alive with an error banner when an edit introduces a syntax or validation error.
  - Provides interactive time-series controls (slider, first/prev/play-pause/next/last buttons) backed by lazy `TimeSeries` step loading, field and colormap `ComboBox` pickers, logarithmic scale/colorbar/axes triad toggles, and camera reset.
- Tests: integration test suite in `crates/tetraglass-viewer/tests/viewer_document.rs` covering raw mesh and scene loading, field and colormap picker switching without filter re-execution, lazy time-slider stepping on `testdata/vtk/series.pvd`, hot-reload detection and error recovery, and offscreen `egui_wgpu` rendering (`target/render-tests/viewer_v0_series.png`); plus CLI `view` argument/help tests in `crates/tetraglass-cli/tests/cli.rs`.
- Decisions: none (implemented per D-003, D-004, and `.agents/rules/rendering.md`).
- Dependencies added: none (reused workspace `eframe`, `egui`, `egui-wgpu`, `serde_json`, `thiserror`, `log`, and wrote `SceneFileWatcher` in-house).
- Follow-ups: M1.9 Performance target (define the interactive target for M2: mesh size, frame time, filter latency, and a synthetic benchmark mesh generated in code; record it as a decision).

## 2026-10-24: M1.9 Performance target
- Done: defined the Milestone M2 interactive performance target (`PerformanceTarget`, `M2_PERFORMANCE_TARGET`: `50 × 50 × 40 = 100,000` hexes / `600,000` Kuhn tetrahedra, `106,641` points; viewport frame time `<= 16.6 ms` at `1200 × 800` `4x` MSAA; interactive plane `slice`/single-isovalue `contour` `<= 50 ms`; full-volume `linearize` / `extract_external_surface` / `clip` / `threshold` `<= 150 ms` in `--release`) and implemented the deterministic in-code synthetic benchmark mesh generator in `tetraglass-core` (`SyntheticCellLayout`, `SyntheticGridSpec`, `generate_synthetic_grid`, `m2_benchmark_tet_spec`, `m2_benchmark_hex_spec`) plus the `surface_bench` benchmark target in `tetraglass-filters`.
- Tests: unit test `synthetic_grid_specs_match_m2_performance_target_and_generate_valid_meshes` in `tetraglass-core`; integration test `synthetic_tet_and_hex_grids_have_exact_boundary_faces_and_zero_internal_faces` in `crates/tetraglass-filters/tests/external_surface.rs`; and benchmark harness `crates/tetraglass-filters/benches/surface_bench.rs` (`100k` hexes: `6.9 ms` gen / `32.9 ms` surface; `600k` tets: `7.6 ms` gen / `122.2 ms` surface).
- Decisions: D-019 (Milestone M2 interactive performance target and `50 × 50 × 40` synthetic benchmark mesh).
- Dependencies added: none.
- Follow-ups: M1.10 Gallery (at least five gallery scenes with baselines; `author-scene` skill examples checked against the real CLI).

## 2026-10-24: M1.10 Gallery
- Done: authored five M1 gallery scenes in `gallery/` with `gallery/README.md` and rendered/inspected (`read_file`) their 800×600 4× MSAA baselines in `tests/baselines/`:
  1. `gallery/surface-mixed-linear.json` (`tests/baselines/surface-mixed-linear.png`): `mixed_linear.vtu` external surface colored by point scalar `linear` (`[0.0, 7.5]`, `viridis`, right colorbar, axes triad, title).
  2. `gallery/surface-cell-id.json` (`tests/baselines/surface-cell-id.png`): `mixed_linear.vtu` explicit `surface` filter colored by flat cell scalar `cell:cell_id` (`[0.0, 3.0]`, `cividis`, right colorbar, axes triad, title).
  3. `gallery/surface-velocity-magnitude.json` (`tests/baselines/surface-velocity-magnitude.png`): `hex_linear.vtu` colored by `velocity:magnitude` (`[0.0, 3.0]`, `center: 1.5`, `cool_warm`, bottom horizontal colorbar, axes triad, title).
  4. `gallery/gmsh-quadratic-tetra.json` (`tests/baselines/gmsh-quadratic-tetra.png`): Gmsh 4.1 quadratic tetrahedra (`tet2_v41.msh`) colored by point scalar `linear` (`[0.0, 6.0]`, `cividis`, right colorbar, axes triad, title).
  5. `gallery/series-transient-last.json` (`tests/baselines/series-transient-last.png`): ParaView `.pvd` time series (`series.pvd`) at `"time": "last"` (`t = 1.0`, step `2`) colored by `transient` with `"range": "auto_all_steps"` (`[0.0, 7.0]`, `viridis`, right colorbar, axes triad, title).
  - Updated the minimal scene example in `.agents/skills/author-scene/SKILL.md` so it validates and renders cleanly against `testdata/vtk/mixed_linear.vtu` via the real `tetraglass` CLI.
- Tests: added `gallery_scenes_and_author_scene_skill_example_validate_and_match_baselines` in `crates/tetraglass-cli/tests/cli.rs`, which runs `tetraglass validate` and `tetraglass render` on all five `gallery/*.json` scenes, compares the rendered PNGs against `tests/baselines/*.png` within the `mean_abs_diff <= 0.01` / `pct_over_4 <= 0.5%` visual regression threshold, and extracts, validates, and renders the JSON scene snippet from `.agents/skills/author-scene/SKILL.md`.
- Decisions: none.
- Dependencies added: none.
- Follow-ups: M1.11 Gate (`/milestone-gate`, including the fresh-agent test).

## 2026-10-24: M1.11 Gate (Milestone M1 Complete)
- Done: ran `/milestone-gate` for Milestone M1 (`docs/milestones/m1-gate.md`, `docs/milestones/m1-fresh-agent-render.png`, `docs/milestones/m1-viewer-v0.png`):
  - **Fresh-agent zero-compile test:** launched a fresh subagent session (`f0f0dd8e-2043-4581-bfb0-78c9373b3090`) restricted to only the pre-built `target/debug/tetraglass` CLI and `docs/scene-format.md`. It inspected `testdata/gmsh/tet2_v41.msh` via `tetraglass inspect --json`, authored `target/m1-gate/fresh-agent-scene.json` (`surface` filter, `velocity:magnitude`, `viridis`, right colorbar, axes triad, title, perspective camera), passed `tetraglass validate` on its first attempt, and rendered `docs/milestones/m1-fresh-agent-render.png` (`800 × 600`, `4×` MSAA) with zero compile steps.
  - **Milestone M1 Retrospective:**
    - *Planned vs. actual time:* Planned for Jan–Mar 2027 (~11 evening/weekend sessions); completed in 2 days (2026-10-23 to 2026-10-24) across 11 small, self-contained tasks (`M1.1`–`M1.11`).
    - *What was cut or moved:* Nothing was cut; curved-surface tessellation of higher-order cells remains scheduled for `M2.2` as originally planned (noticed during `M1.10` when `tet2_v41.msh` corner-only external surface had uniform `radius` on `[0, 1]^3` corners).
    - *What to change in M2's plan:* `M2.1 Linearize` covers linear cell tet decomposition with face-consistent diagonals, Lagrange higher-order subdivision with basis evaluation, and polyhedron centroid decomposition; if `M2.1` exceeds ~500 lines, split it into `M2.1a` (linear + polyhedra tet decomposition + source-cell map) and `M2.1b` (Lagrange higher-order subdivision + property tests) at the start of M2.
- Tests: verified all 5 gallery scenes + `author-scene` skill snippet (`cargo test -p tetraglass-cli --test cli`), `tetraglass-viewer` offscreen test (`cargo test -p tetraglass-viewer --test viewer_document`), and full `cargo xtask verify`.
- Decisions: none.
- Dependencies added: none.
- Follow-ups: M2.1 Linearize (first task of Milestone M2 — await user's go-ahead to start M2).

## 2026-10-24: M2.1a Linearize linear and polyhedral cells
- Done: split `M2.1` into `M2.1a` (linear + polyhedral cell decomposition) and `M2.1b` (Lagrange/quadratic higher-order subdivision + property tests) per the M1.11 retrospective; implemented `TetView` and `signed_tet_volume` in `tetraglass-core` and `linearize` / `linearize_default` (`LinearizeParams`, `DEFAULT_SUBDIVISIONS`) in `tetraglass-filters` with Dompierre et al. minimum-global-vertex face-consistent diagonal tetrahedralization of linear 3D cells (`Tetra`, `Hexahedron`, `Wedge`, `Pyramid`), centroid + minimum-vertex face-fan tetrahedralization of `Polyhedron` cells with equal-weight point attribute averaging, `Arc` point/attribute buffer sharing when no new points are added, and per-tet source-cell mapping (`tet_source_cells`); added `linearize` timing to `surface_bench` (`8.1 ms` on `100k` hexes -> `600k` tets).
- Tests: unit test `tet_view_validates_inputs_and_computes_volumes` in `tetraglass-core` and integration tests in `crates/tetraglass-filters/tests/linearize.rs` covering `tet_linear.vtu`, `hex_linear.vtu`, `mixed_linear.vtu`, `polyhedra.vtu`, and synthetic hex grids (verifying exact cell/total volumes, watertight internal shared faces across hex/wedge/pyramid/tet/polyhedron cells, `Arc` buffer sharing, and exact linear scalar/vector field reproduction at polyhedron centroids).
- Decisions: none (implemented per D-005 and D-019).
- Dependencies added: none.
- Follow-ups: M2.1b Linearize higher-order cells and property tests (reference-space subdivision at configurable level with Lagrange and serendipity basis evaluation for positions and point fields, plus property tests).

## 2026-10-24: M2.1b Linearize higher-order cells and property tests
- Done: implemented reference-space subdivision at a configurable level (`LinearizeParams { subdivisions }`) with VTK-ordered Lagrange (`LagrangeTetrahedron`, `LagrangeHexahedron`, `LagrangeWedge`) and quadratic serendipity (`QuadraticTetra`, `QuadraticHexahedron`, `QuadraticWedge`, `QuadraticPyramid`) basis evaluation (`crates/tetraglass-filters/src/basis.rs`) for 3D positions and point attributes in `linearize`, reusing existing cell nodes when sample points coincide with mesh nodes and deduplicating shared boundary sample points across adjacent higher-order cells so subdivided faces remain watertight.
- Tests: unit test `all_higher_order_bases_satisfy_kronecker_delta_partition_of_unity_and_linear_reproduction` in `crates/tetraglass-filters/src/basis.rs`, integration tests in `crates/tetraglass-filters/tests/linearize.rs` on `tet_quadratic.vtu`, `tet2_v41.msh`, `hex_lagrange_p3.vtu`, and curved quadratic tetrahedra, and 240 randomized property-test trials (`property_tests_preserve_volume_linear_fields_and_source_cells_across_all_cell_families`) across all 12 3D cell families verifying volume preservation under affine transforms, positive tet orientation, exact linear scalar/vector field interpolation, and source-cell mapping.
- Decisions: none (implemented per D-005 and D-009).
- Dependencies added: none.
- Follow-ups: M2.2 Curved surfaces (external surface of higher-order cells with tessellated faces; wireframe from original element edges).

## 2026-10-24 — M2.2 Curved surfaces

- Extended `SurfaceMesh` (`crates/tetraglass-core/src/surface.rs`) with `wireframe_edges` (`wireframe_edges(&self) -> &[[u32; 2]]`, `num_wireframe_edges(&self) -> usize`, and `with_wireframe_edges`) storing original element boundary edge segments.
- Extended `crates/tetraglass-filters/src/basis.rs` with 2D higher-order surface cell basis evaluators (`QuadraticTriangle`, `LagrangeTriangle { order }`, `QuadraticQuad`, `LagrangeQuadrilateral { order }`) and `corner_reference_rst`.
- Implemented curved higher-order face tessellation (`extract_external_surface_with_subdivisions`, with `extract_external_surface` defaulting to `DEFAULT_SUBDIVISIONS = 2`) in `crates/tetraglass-filters/src/surface.rs`:
  - Subdivides external higher-order triangle faces into $S^2$ sub-triangles and external higher-order quad faces into $2 S^2$ sub-triangles using the element's Lagrange or serendipity basis functions, reusing original mesh vertex indices whenever a lattice point coincides with an element node and deduplicating shared edge vertices across adjacent faces.
  - Emits only the outer perimeter edges of the original mesh faces into `SurfaceMesh::wireframe_edges` (`S` segments per curved higher-order edge), excluding quad/polygon fan diagonals and internal higher-order face subdivision edges.
- Wired `source.linearize_level` into `SceneEvaluator::get_or_extract_surface` (`crates/tetraglass-scene/src/eval.rs`).
- Added original-element wireframe edge rendering (`edge_pipeline`, `SurfaceStyle::{draw_surface, draw_edges, edge_color_srgb}`) to `crates/tetraglass-render`, and wired `RepresentationStyle::{Surface, SurfaceWithEdges, Wireframe}` and `edge_color` in `tetraglass-cli` and `tetraglass-viewer`.
- Verified with `crates/tetraglass-filters/tests/external_surface.rs`, `crates/tetraglass-render/tests/render_mixed_linear.rs` (`/render-check`), and `cargo xtask verify`.
- Follow-ups: M2.3 Tet kernel and slice (marching tetrahedra kernel, slice filter, scene integration, and gallery scene).

## 2026-10-24 — M2.3a Marching tetrahedra kernel and slice filter

- Split `M2.3` into `M2.3a` (marching-tetrahedra kernel and plane `slice` filter in `tetraglass-filters`) and `M2.3b` (scene schema integration, validation, `docs/scene-format.md`, and gallery scene).
- Added `crates/tetraglass-filters/src/marching_tets.rs` implementing `EdgeKey`, `DEFAULT_TET_CHUNK_SIZE`, `extract_isosurface`, and `extract_isosurfaces_with_chunk_size` on `TetView` (`D-005`, `D-006`) with the 16-case marching-tetrahedra table (`val >= iso` tie-breaking), `iso_tol` snapping and endpoint `EdgeKey(v, v)` deduplication, quad cross-section perimeter wireframe edges (excluding internal quad diagonals), parallel chunk execution via `std::thread::scope` (on non-`wasm32` targets), deterministic chunk-ordered merge, and linear interpolation / mapping of all point and cell `Attributes`.
- Added `crates/tetraglass-filters/src/slice.rs` (`SliceParams`, `slice`, `slice_with_chunk_size`, `slice_grid`) and `FilterError::{InvalidPlaneNormal, InvalidPlaneOrigin}`.
- Added unit and integration tests (`crates/tetraglass-filters/tests/slice.rs`) covering all 14 active marching-tetrahedra cases, analytic cross-section areas across all 6 cell families in `testdata/vtk/*.vtu`, degenerate planes passing directly through mesh vertices and shared faces (`z = 1.0` on `hex_linear.vtu` and `x - y = 0` on `tet_linear.vtu`), out-of-bounds empty cuts, invalid plane parameters, linear scalar/vector field reproduction, cell attribute provenance, and bit-for-bit determinism across chunk sizes and thread counts.
- Follow-ups: M2.3b Slice scene integration and gallery scene (`Filter::Slice` in `tetraglass-scene`, semantic validation, JSON Schema snapshot, `docs/scene-format.md`, and `gallery/slice.json` with baseline).

## 2026-10-24 — M2.3b Slice scene integration and gallery scene

- Done: added `Filter::Slice { input, origin, normal }` (`Filter::slice`, `Filter::slice_at`) to `tetraglass-scene`; added semantic validation for `filters.<name>.normal` (`must not be [0, 0, 0]` and finite coordinates), `filters.<name>.origin` (finite coordinates), and volume dataset input enforcement; cached linearized `TetView`s by `(upstream_hash, subdivisions)` and slice `SurfaceMesh` outputs by `(upstream_hash, origin, normal)` in `SceneEvaluator`; updated `crates/tetraglass-cli/tests/snapshots/scene-v1.schema.json` and `docs/scene-format.md`; authored `gallery/slice.json` and `tests/baselines/slice.png`.
- Tests: unit and integration tests in `crates/tetraglass-scene/src/lib.rs`, `crates/tetraglass-scene/tests/validation.rs`, `crates/tetraglass-scene/tests/evaluation_cache.rs`, and `crates/tetraglass-cli/tests/cli.rs` (`schema_command_matches_committed_snapshot` and `gallery_scenes_and_author_scene_skill_example_validate_and_match_baselines`).
- Decisions: none (implemented per D-004, D-005, and D-006).
- Dependencies added: none.
- Follow-ups: M2.4 Contour (multiple isovalues, all arrays interpolated).

## 2026-10-24 — M2.4 Contour

- Done: implemented multi-isovalue `contour` filter (`ContourComponent`, `ContourParams`, `contour`, `contour_with_chunk_size`, `contour_grid`) in `tetraglass-filters` using the shared marching-tetrahedra kernel (`D-006`) on `TetView` (`D-005`) with all point and cell arrays interpolated; added `Filter::Contour { input, field, values }` (`Filter::contour`) to `tetraglass-scene` with semantic validation (`field` syntax, point-array existence with edit-distance suggestions, component compatibility, non-empty finite `values`, and volume `input`) and cached evaluation in `SceneEvaluator`; updated `crates/tetraglass-cli/tests/snapshots/scene-v1.schema.json` and `docs/scene-format.md`; authored `gallery/contour.json` and `tests/baselines/contour.png`.
- Tests: `crates/tetraglass-filters/tests/contour.rs` (analytic agreement with equivalent `slice` plane, multi-isovalue extraction across all 6 cell families in `testdata/vtk/*.vtu`, vector component and magnitude contouring, linear scalar/vector field reproduction, cell attribute provenance, out-of-range empty contours, error cases, and chunk-size determinism), plus scene validation, evaluation cache, schema snapshot, and gallery baseline tests.
- Decisions: none (implemented per D-004, D-005, and D-006).
- Dependencies added: none.
- Follow-ups: M2.5 Clip (exact and crinkle modes).

## 2026-10-24 — M2.5 Clip

- Done: implemented plane `clip` filter (`ClipKeep`, `ClipMode`, `ClipParams`, `clip`, `clip_with_chunk_size`, `clip_grid`) and `extract_tet_view_surface` in `tetraglass-filters` using the shared marching-tetrahedra kernel (`clip_tets_with_chunk_size` for exact conformal prism decomposition and `clip_crinkle_tets_with_chunk_size` for whole-source-cell crinkle clipping on `TetView`, `D-005`, `D-006`); added `Filter::Clip { input, origin, normal, keep, mode }` (`Filter::clip`, `Filter::clip_at`) and `EvaluatedDataset::TetVolume` to `tetraglass-scene` with semantic validation, cached volume evaluation, and downstream volume-filter chaining (`clip -> slice`, `clip -> contour`, `clip -> clip`); updated `crates/tetraglass-cli/tests/snapshots/scene-v1.schema.json` and `docs/scene-format.md`; authored `gallery/clip.json` and `tests/baselines/clip.png`.
- Tests: `crates/tetraglass-filters/tests/clip.rs` (analytic volumes and `Front` + `Back` partition of unity across all 6 cell families in `testdata/vtk/*.vtu`, closed manifold watertightness on clipped boundary surfaces, exact vs. crinkle mode comparisons, degenerate vertex-passing planes, empty/full clips, linear field reproduction, source-cell provenance, and chunk-size determinism), plus scene validation, evaluation cache, schema snapshot, and gallery baseline tests.
- Decisions: none (implemented per D-004, D-005, and D-006).
- Dependencies added: none.
- Follow-ups: M2.6 Threshold.

## 2026-10-24 — M2.6 Threshold

- Added `crates/tetraglass-filters/src/threshold.rs` implementing `ThresholdAssociation` (`Auto`, `Point`, `Cell`), `ThresholdComponent` (`Component(usize)`, `Magnitude`), `ThresholdParams`, `threshold`, `threshold_with_chunk_size`, and `threshold_grid` on `TetView` (`D-005`), grouping by `tet_source_cells` so whole original cells are kept when their cell scalar or all vertex scalars lie within `[min, max]` (supporting two-sided and one-sided bounds).
- Extended `tetraglass-scene` (`Filter::Threshold` in `types.rs`, semantic validation in `validate.rs`, and cached `EvaluatedDataset::TetVolume` evaluation in `eval.rs`), updated `crates/tetraglass-cli/tests/snapshots/scene-v1.schema.json` and `docs/scene-format.md`.
- Added `gallery/threshold.json` and `tests/baselines/threshold.png`, updated `gallery/README.md`, and registered `"threshold"` in `crates/tetraglass-cli/tests/cli.rs`.
- Follow-ups: M2.7 Picking.

## 2026-10-24 — M2.7 Picking

- Done: implemented GPU ID buffer picking (`PickTarget` with `PICK_FORMAT = Rgba32Uint`, `RawPickHit`, `@fragment fn fs_pick` in `surface.wgsl`, `Renderer::render_pick`, `Renderer::pick_pixel`, and `HeadlessContext::pick_surface`) and CPU value probing (`SurfacePickResult`, `PickProbeValue`, `probe_surface_triangle`, `Camera::pixel_ray_f64`, and `Camera::unproject_pixel_depth_f64`) in `tetraglass-render`; wired synchronous and click-triggered picking plus on-canvas crosshair/badge overlay and multi-surface rendering into `tetraglass-egui` (`ViewportState::pick_at`, `ViewportState::request_pick_at`, `ViewportState::last_pick`, `ViewportState::clear_pick`, `ViewportState::set_shared_surface`, `ViewportState::set_additional_surfaces`, and `ViewportWidget::show_probe_overlay`); added `ViewerSession::pick_at_pixel`, `ViewerSession::last_pick`, `ViewerSession::clear_pick`, `ViewerSession::picked_representation_name`, and the left-panel Value probe section in `tetraglass-viewer`.
- Tests: `crates/tetraglass-render/tests/picking.rs` (single-surface and multi-representation depth-occluded picking, background misses, out-of-bounds pixels, barycentric coordinates, world positions, source cell IDs, and interpolated scalar/vector point and cell values), `crates/tetraglass-render/src/camera.rs` (`pixel_ray_and_unproject_roundtrip_focal_point`), `crates/tetraglass-egui/tests/viewport_offscreen.rs`, and `crates/tetraglass-viewer/tests/viewer_document.rs`.
- Decisions: none (implemented per D-002 and D-003; `PickTarget` is single-sampled `Rgba32Uint` and only rendered when a pick is requested, keeping normal color rendering untouched).
- Dependencies added: none.
- Follow-ups: M2.8 Interactive clipping.

## 2026-10-24 — M2.8 Interactive clipping

- Done: added GPU shader clip-plane support (`ShaderClipKeep`, `ShaderClipPlane`, `SurfaceStyle::clip_plane`, `GpuSurface::clip_plane`, `Renderer::set_surface_clip_plane`, and `clip_plane: vec4<f32>` in `SurfaceUniforms` evaluated per fragment across `fs_main`, `fs_pick`, and `fs_edge`) in `tetraglass-render` so dragging a clip plane updates only a 16-byte GPU uniform (`D-019`); exposed `ViewportState::set_clip_plane`, `ViewportState::set_clip_plane_with_render_state`, `ViewportState::clip_plane`, and preserved `dataset_center` across `fit_camera == false` surface swaps in `tetraglass-egui`; added `SceneEvaluator::evaluate_tet_view` in `tetraglass-scene`; implemented `ViewerDocument::resolve_clip_volume`, `InteractiveClipState`, `ViewerSession::begin_or_update_clip_drag`, `ViewerSession::end_clip_drag`, `ViewerSession::poll_clip_worker`, `ViewerSession::wait_for_clip_worker`, `ViewerSession::disable_interactive_clip`, and the Interactive clip UI controls in `tetraglass-viewer`.
- Tests: `shader_clip_plane_clips_color_and_pick_passes_and_restores_when_cleared` in `crates/tetraglass-render/tests/picking.rs`, shader clip-plane assertions in `crates/tetraglass-egui/tests/viewport_offscreen.rs`, and `interactive_clip_uses_shader_plane_while_dragging_and_swaps_in_worker_result` in `crates/tetraglass-viewer/tests/viewer_document.rs`.
- Decisions: none (implemented per D-002, D-003, D-014, and D-019).
- Dependencies added: `tetraglass-filters` added to `tetraglass-viewer` (`Cargo.toml` workspace dependency).
- Follow-ups: M2.9 Performance pass.

## 2026-10-24 — M2.9 Performance pass

- Done: optimized `extract_external_surface_with_subdivisions` and `extract_tet_view_surface` in `crates/tetraglass-filters/src/surface.rs` by replacing `SipHash` `HashMap<FaceKey, usize>` and heap-bearing `FaceKey::Poly(Vec<u32>)` / `OrientedFace::Poly(Vec<u32>)` variants with `Copy` structs (`FaceRegistry`, `CandidateFace`, `TetCandidateFace`, `TetBoundaryEdgeEntry`), inline 3/4-element sorting networks (`sort3`, `sort4`), and flat vertex-bucketed linked lists (`tri_heads`, `quad_heads`, `edge_heads`) indexed by minimum vertex ID (`0..num_points`); optimized `crates/tetraglass-filters/src/marching_tets.rs` with a deterministic multiplicative `FxHasher` for chunk-local edge maps, an early `num_above == 4` fast path in `process_exact_clip_chunk`, and a flat `vertex_to_vid` lookup table for original mesh vertices (`edge_key.0 == edge_key.1`) in `merge_clip_chunks`; expanded `crates/tetraglass-filters/benches/surface_bench.rs` and added `crates/tetraglass-render/benches/render_bench.rs` to measure and assert every `D-019` (`M2_PERFORMANCE_TARGET`) latency and frame-time budget on the `100,000`-hex / `600,000`-tet synthetic benchmark mesh (`106,641` points).
- Benchmarks (`--release`, Apple M3 Max, `50 × 50 × 40` synthetic mesh = `100,000` hexes / `600,000` tets):
  - `linearize` (`100k` hexes -> `600k` tets): `11.81 ms` (`<= 150.0 ms` target)
  - `extract_external_surface` (`100k` hexes -> `13,000` boundary quads): `16.91 ms` (down from `47.29 ms`, `2.8x` faster; `<= 150.0 ms` target)
  - `linearize` (`600k` tets): `7.70 ms` (`<= 150.0 ms` target)
  - `extract_external_surface` (`600k` tets -> `26,000` boundary tris): `29.02 ms` (down from `120.39 ms`, `4.1x` faster; `<= 150.0 ms` target)
  - `slice` (`600k` tets, oblique plane -> `22,856` tris): `5.32 ms` (`<= 50.0 ms` target)
  - `contour` (`600k` tets, single isovalue -> `42,568` tris): `8.99 ms` (`<= 50.0 ms` target)
  - `clip` (`Exact`, oblique plane) + `extract_tet_view_surface` (`600k` -> `331,578` tets -> `36,228` tris): `8.04 ms` vol + `16.66 ms` surf = `24.71 ms` (`<= 150.0 ms` target)
  - `clip` (`Crinkle`, oblique plane) + `extract_tet_view_surface` (`600k` -> `309,352` tets -> `20,470` tris): `8.72 ms` vol + `12.04 ms` surf = `20.76 ms` (`<= 150.0 ms` target)
  - `threshold` + `extract_tet_view_surface` (`600k` -> `286,818` tets -> `21,548` tris): `7.74 ms` vol + `10.37 ms` surf = `18.11 ms` (`<= 150.0 ms` target)
  - GPU viewport rendering at `1200 × 800` with `4x` MSAA (`48,856` tris + wireframe edges + colorbar + axes triad + title, synchronous GPU poll): `1.53 ms/frame` (`652.8 FPS`) for camera orbit/pan/zoom, `1.22 ms/frame` (`816.6 FPS`) for colormap/range uniform updates, and `0.88 ms/frame` (`1133.8 FPS`) for interactive shader clip-plane dragging (`<= 16.6 ms` / `60 FPS` target).
- Decisions: all M1.9 / D-019 performance targets met with wide headroom; nothing cut.
- Dependencies added: none.
- Follow-ups: M2.10 Documentation.

## 2026-10-24 — M2.10 Documentation

- Done: authored the mdBook user and developer guide in `docs/book/` (`book.toml`, `src/SUMMARY.md`, `src/introduction.md`, `src/getting-started.md`, `src/embedding.md`, `src/scene-format.md`, `src/filters.md`, and `src/gallery.md`); added runnable crate-level rustdoc examples (`//! # Examples`) to `tetraglass-core`, `tetraglass-io`, `tetraglass-filters`, `tetraglass-render`, and `tetraglass-egui`; fixed all intra-doc link warnings so `cargo doc --workspace --no-deps` builds cleanly with zero warnings; updated the root `README.md` with CLI/Rust quickstarts, workspace crate table, and documentation links.
- Tests: crate-level doctests across `tetraglass-core`, `tetraglass-io`, `tetraglass-filters`, `tetraglass-scene`, `tetraglass-render`, and `tetraglass-egui`, plus `mdbook_guide_and_gallery_cover_all_filters_and_chapters_exist` in `crates/tetraglass-cli/tests/cli.rs` verifying all mdBook chapters and gallery coverage across all five filter types (`surface`, `slice`, `contour`, `clip`, `threshold`).
- Decisions: none.
- Dependencies added: none.
- Follow-ups: M2.11 Gate and release 0.1 (`/milestone-gate`).

## 2026-10-24 — M2.11 Gate and release 0.1

- Done: completed the Milestone M2 gate (`docs/milestones/m2-gate.md`, `m2-slice.png`, `m2-contour.png`, `m2-clip.png`, `m2-threshold.png`) and release `0.1.0` checklist: bumped workspace and internal dependency versions to `0.1.0`, set `publish = true`, `readme = "../../README.md"`, and `documentation = "https://docs.rs/<crate>"` across all publishable crates, wrote `CHANGELOG.md` for `0.1.0`, verified `cargo package --no-verify --offline` across all 8 workspace crates (`tetraglass-core` through `tetraglass-cli`), and verified `cargo doc --workspace --no-deps` and `cargo xtask verify`.
- Retrospective (Milestone M2 — Core filters):
  - **Planned vs. actual:** Planned for Apr–Jul 2027 (12 sub-tasks `M2.1a` through `M2.11`); completed on 2026-10-24 in a sequence of focused, self-contained tasks.
  - **What was cut or moved:** Nothing cut. Every planned M2 feature (`TetView` linearization for linear, higher-order Lagrange/serendipity, and polyhedral cells; curved surfaces with original element wireframes; marching-tetrahedra `slice`, `contour`, and `clip` in `exact`/`crinkle` modes; `threshold`; GPU ID-buffer picking and value probe; interactive shader clip-plane dragging with background exact-clip worker swap; `D-019` performance pass; mdBook documentation and 9-scene gallery) shipped and met all `D-019` targets with 5x–15x headroom.
  - **Lessons & adjustments for M3 (Web and export):** Keeping all `wgpu` calls isolated in `tetraglass-render` and keeping `core`, `io`, `filters`, and `scene` `wasm32-unknown-unknown`-clean on every commit paid off—the entire filter and scene pipeline is already browser-ready for `M3.1 Web viewer`.
- Decisions: confirmed and marked `D-015` as `Accepted` with `DEFAULT_MAX_DIFF_FRACTION = 0.0075` (`0.75%`) to accommodate cross-driver 1px `LineList` MSAA rasterization differences between hardware GPUs (Apple Metal) and CI's software Vulkan driver (Mesa `lavapipe`) on wireframe/edge-overlay gallery scenes (`slice`, `contour`, `clip`, `threshold`), and updated `tests/baselines/` with authoritative Mesa `lavapipe` gallery renders.
- Dependencies added: none.
- Follow-ups: Published all 8 workspace crates at `0.1.0` to crates.io and pushed tag `v0.1.0`; next milestone starts with M3.1 Web viewer.

## 2026-10-24 — M3.1 Web viewer

- Done: implemented the `wasm32-unknown-unknown` WebGPU browser viewer in `tetraglass-viewer`, non-blocking GPU pixel picking in `tetraglass-render` (`PendingPixelPick`, `Renderer::begin_pick_pixel`) and `tetraglass-egui`, WebGPU-only `egui_wgpu::WgpuConfiguration` (`webgpu_only_wgpu_configuration` enforcing `Backends::BROWSER_WEBGPU` with no WebGL2 fallback per `D-007`), structured and HTML-escaped diagnostic messages when WebGPU is unavailable (`WebGpuStatus`, `WebGpuErrorMessage`, `render_webgpu_error_banner`), drag-and-drop loading of `.vtu`, `.msh`, `.pvd` (with companion step `.vtu` files), and `.json` scenes (with companion mesh files) via `InMemoryFileStore` and `ViewerSession::ingest_dropped_files`, URL loading via query parameters (`?url=`, `?scene=`, `?file=`), programmatic `WebViewerHandle::load_url`, and an in-panel URL input bar (`FetchedUrlBundle`, `fetch_url_bundle_with`, `fetch_url_bundle_async`, `resolve_relative_url`), plus `crates/tetraglass-viewer/web/index.html` and expanded `cargo xtask wasm-check` / `verify` to cover `tetraglass-render`, `tetraglass-egui`, and `tetraglass-viewer`.
- Tests: `webgpu_status_messages_and_wgpu_configuration_enforce_webgpu_only`, `url_resolution_and_query_parsing_handle_relative_paths_and_percent_encoding`, and `viewer_session_supports_drag_and_drop_and_url_loading_for_meshes_series_and_scenes` in `crates/tetraglass-viewer/tests/viewer_document.rs`, plus doctests on `resolve_relative_url` and `parse_query_map`.
- Decisions: none (enforces `D-007`).
- Dependencies added: promoted existing transitive `Cargo.lock` crates `js-sys = "0.3"`, `wasm-bindgen = "0.2"`, `wasm-bindgen-futures = "0.4"`, and `web-sys = "0.3"` (`MIT OR Apache-2.0`) to `[target.'cfg(target_arch = "wasm32")'.dependencies]` of `tetraglass-viewer`.
- Follow-ups: M3.2 Shareable links.

## 2026-10-24 — M3.2 Shareable links

- Done: implemented shareable web-viewer links (`ShareableLinkSpec`, `ShareableViewOverrides`, `InlineShareableBundle`, `encode_compressed_base64url`, `decode_compressed_base64url`, `percent_encode_query_component`, `fetch_shareable_link_with`) in `tetraglass-viewer` supporting three link modes—(1) remote scene/mesh URL (`?scene=<url>`, `?url=<url>`, `?file=<url>`) with companion data URL overrides (`?data=<url>`, `?data=<source>:<url>`, `?data.<source>=<url>`), (2) inline zlib-compressed + URL-safe base64 scene JSON (`?scene_b64=...` or `?scene_json=...`) paired with remote companion data URLs, and (3) self-contained inline scene + companion mesh data bundles (`?bundle_b64=...`)—along with query/fragment view-state overrides (`az`, `el`, `roll`, `zoom`, `step`, `color_by`, `colormap`, `log`, `colorbar`, `axes`), `ViewerDocument::apply_shareable_overrides`, `ViewerSession::export_scene_with_live_view`, `ViewerSession::build_shareable_link`, `ViewerSession::build_inline_bundle_shareable_link`, `WebViewerHandle::shareable_link`, and a `"Copy shareable link"` button in the viewer control panel.
- Tests: `shareable_link_roundtrips_scene_plus_remote_data_urls_and_view_overrides` and `viewer_session_builds_and_opens_self_contained_and_remote_shareable_links` in `crates/tetraglass-viewer/tests/viewer_document.rs`.
- Decisions: none.
- Dependencies added: none (reused existing workspace dependencies `base64`, `flate2`, and `serde`).
- Follow-ups: M3.3 Transparency.

## 2026-10-25 — M3.3 Transparency (`crates/tetraglass-render`, `crates/tetraglass-viewer`)

- Implemented **Weighted Blended Order-Independent Transparency (WBOIT, McGuire & Bavoil 2013)** in `tetraglass-render`:
  - Added `WBOIT_ACCUM_FORMAT` (`Rgba16Float`) and `WBOIT_REVEAL_FORMAT` (`Rgba8Unorm`) offscreen attachments (both single-sampled and 4x MSAA resolved) and `wboit_bind_group` to `OffscreenTarget`.
  - Added `wboit_weight(ndc_z, alpha)` in `common.wgsl`, `@fragment fn fs_wboit` in `surface.wgsl` (with `f16` quantization for commutative accumulation across draw orders), and `wboit_composite.wgsl` for full-screen triangle compositing (`accum.rgb / max(accum.a, 1e-5)` with coverage `1 - reveal.r`).
  - Extended `GpuSurface` with `opacity()`, `is_opaque()`, and `is_transparent()`, and updated `Renderer::render_internal` to preserve the single-pass fast path when all surfaces are opaque while switching to the 3-pass WBOIT pipeline (Pass 1 opaque depth+color, Pass 2 MRT accumulation+revealage with depth test and no depth write, Pass 3 composite + wireframe edges + annotations) whenever any surface has `0.0 < opacity < 1.0`.
- Exposed interactive opacity controls in `tetraglass-viewer`: `ViewerDocument::opacity`, `ViewerDocument::set_opacity`, `ShareableViewOverrides::opacity` (`?opacity=`), and an `Opacity` slider in `ViewerSession::show_ui`.
- Added headless WBOIT integration tests in `crates/tetraglass-render/tests/render_mixed_linear.rs` (verifying monotonic opacity response, multi-surface draw-order invariance, internal triangle-order invariance, and opaque-inside-transparent compositing at both 1x and 4x MSAA) and opacity round-trip tests in `crates/tetraglass-viewer/tests/viewer_document.rs`.
- Decisions: None.
- Follow-ups: M3.4 glTF export.

## 2026-10-25 — M3.4 glTF export (`crates/tetraglass-export`, `crates/tetraglass-cli`)

- Done: created `tetraglass-export` (`wasm32-unknown-unknown`-compatible, zero GPU or C dependencies) with Binary glTF 2.0 (`.glb`) scene and time-series exporters (`export_evaluated_scene_to_glb`, `export_time_series_to_glb_frames`, `write_evaluated_scene_glb`, `write_scene_glb_output`) and a `.glb` container parser (`parse_glb`). Exported primitives store `POSITION` (`f32` relative to `dataset_center` with node `translation` preserving world coordinates per `D-014`), `NORMAL`, baked linear-light RGBA vertex colors in `COLOR_0` (supporting all 5 colormaps, solid coloring, and `opacity` with `"OPAQUE"` / `"BLEND"` materials), `TEXCOORD_0` (`[normalized_scalar, 0.0]`), and `TEXCOORD_1` (`[normalized_scalar, raw_scalar]` so a Blender user can re-color via a ColorRamp node), plus one `.glb` per time step accompanied by a `GltfSeriesManifest` JSON file. Added `tetraglass export <scene.json> -o <out.glb|dir> [--format <gltf|bundle>] [--all-steps]` to `tetraglass-cli`.
- Tests: unit tests in `crates/tetraglass-export/src/colormap.rs`, crate-level doctest in `crates/tetraglass-export/src/lib.rs`, integration tests in `crates/tetraglass-export/tests/gltf_export.rs` (verifying gallery scenes, transparent and cell-colored representations, multi-step `.pvd` time-series export, and corrupted `.glb` error handling), and `export_command_writes_single_glb_and_time_series_manifest` in `crates/tetraglass-cli/tests/cli.rs`.
- Decisions: none.
- Dependencies added: none.
- Follow-ups: M3.5 Bundle format (`/blender-bridge`).

## 2026-10-25 — M3.5 Bundle format (`docs/bundle-format.md`, `crates/tetraglass-export`, `crates/tetraglass-cli`)

- Done: authored `docs/bundle-format.md` specifying the renderer-neutral v1 bundle directory format (`manifest.json` + `meshes/*.tgbin`) per `D-011` and `.agents/rules/blender-gpl.md`. Implemented the bundle writer and round-trip reader/validator in `crates/tetraglass-export/src/bundle.rs` (`BundleManifest`, `BundleCamera`, `BundleLights`, `BundleBackground`, `BundleTimeStep`, `BundleObject`, `BundleColormap`, `BundleMaterialHints`, `BundleMeshFileRef`, `BundleMeshData`, `BundleArchive`, `encode_tgbin`, `decode_tgbin`, `export_evaluated_scene_to_bundle`, `export_scene_to_bundle`, `write_bundle_dir`, `write_scene_bundle_output`, and `read_bundle_dir`), storing `f32` vertex positions and original element wireframe `edge_positions` relative to `dataset_center` (`D-014`), 256-entry sRGB colormap LUTs, normalized `[0, 1]` and raw scalar attributes, and material/camera/lighting hints. Wired `tetraglass export <scene.json> -o <bundle_dir> --format bundle [--all-steps]` into `crates/tetraglass-cli`.
- Tests: `gallery_scenes_round_trip_through_bundle_writer_and_reader`, `multi_step_series_bundle_exports_all_steps_and_round_trips`, and `tgbin_decoder_rejects_corrupted_headers_and_out_of_bounds_indices` in `crates/tetraglass-export/tests/bundle_export.rs`, plus CLI `--format bundle` and `--all-steps` tests in `crates/tetraglass-cli/tests/cli.rs`.
- Decisions: none (implements `D-011` and `D-014`).
- Dependencies added: none.
- Follow-ups: M3.6 Blender scripts (`/blender-bridge`).

## 2026-10-25 — M3.6 Blender scripts (`blender/`, `crates/tetraglass-export/tests/bundle_export.rs`)

- Done: implemented the GPL-3.0-or-later Blender integration package in `blender/` (`blender/importer.py`, `blender/render.py`, `blender/__init__.py`, `blender/blender_manifest.toml`, `blender/README.md`) using only the Python standard library and `numpy` (plus `bpy`/`mathutils`). `blender/importer.py` decodes `manifest.json` and `.tgbin` mesh buffers (`load_bundle_manifest`, `read_tgbin`) and builds Blender meshes with `tetraglass_scalar`, `tetraglass_raw_scalar`, and named field attributes, a 256x1 linear-light LUT texture + 32-stop ColorRamp shader node tree, non-shadow-casting 3D wireframe edge curves (`_create_blender_edge_object`), perspective/orthographic camera, sun lights, solid/gradient world background (with `Is Camera Ray` separation and `Standard` view transform), and per-step keyframed timeline frames. `blender/render.py` provides the headless Cycles CLI driver (`--bundle`, `--out`, `--samples`, `--device`, `--step`, `--all-steps`, `--width`, `--height`, `--save-blend`) with real-time `PROGRESS` stdout streaming. `blender/__init__.py` and `blender/blender_manifest.toml` package the directory as an installable Blender add-on (`File > Import > Tetraglass Bundle (manifest.json)`).
- Tests: `blender_scripts_spdx_headers_addon_registration_and_headless_cycles_render` in `crates/tetraglass-export/tests/bundle_export.rs` (verifying SPDX GPL headers, add-on registration and operator execution inside headless Blender 5.2 LTS, Cycles rendering of `gallery/slice.json` with `--save-blend` and `PROGRESS` streaming, and multi-step `--all-steps` rendering of `gallery/series-transient-last.json`).
- Decisions: D-020 (supported Blender versions are Blender 4.2 LTS and Blender 5.x LTS; `blender/` stays in this repository under GPL-3.0-or-later).
- Dependencies added: none.
- Follow-ups: M3.7 Cycles from the CLI (`/blender-bridge`).

## 2026-10-25 — M3.7 Cycles from the CLI (`crates/tetraglass-cli`)

- Done: implemented `tetraglass render <scene.json> [-o <out.png>] [--engine cycles] [--samples <n>] [--device <dev>] [--save-blend <path.blend>] [--blender-path <bin>] [--blender-scripts <dir>] [--timeout <secs>]` in `crates/tetraglass-cli/src/main.rs` (`run_cycles_render`, `discover_blender_binary`, `discover_blender_render_script`, `probe_blender_binary`, `extract_progress_field`). Discovers an installed Blender executable (`--blender-path`, `TETRAGLASS_BLENDER_PATH`, `BLENDER_PATH`, `PATH`, and standard installation locations) and locates `blender/render.py` at runtime without embedding GPL scripts (`D-011`, `.agents/rules/blender-gpl.md`). Writes the evaluated scene bundle to a temporary directory, spawns headless Blender with a configurable timeout, streams `PROGRESS` lines to stderr in real time, cleans up the temporary bundle, and outputs the JSON `RenderSummary` enriched with `blender_version`, `cycles_samples`, `cycles_device`, and `save_blend`.
- Tests: `render_cycles_command_discovers_blender_streams_progress_saves_blend_and_prints_json_summary` in `crates/tetraglass-cli/tests/cli.rs`.
- Decisions: none (enforces `D-011` and `D-020`).
- Dependencies added: none.
- Follow-ups: M3.8 Gate (`/milestone-gate`).

## 2026-10-25 — M3.8 Gate & Milestone M3 Retrospective (`docs/milestones/m3-gate.md`)

- Done: verified all M3 tasks (`M3.1`–`M3.7`), demonstrated `gallery/slice.json` rendered both via `tetraglass render --engine cycles` (`docs/milestones/m3-slice-cycles.png`, Blender 5.2.2 LTS) and via `tetraglass render --engine raster` (`docs/milestones/m3-slice-raster.png`) alongside the `wasm32-unknown-unknown` WebGPU browser viewer and shareable link tests, and wrote `docs/milestones/m3-gate.md`.
- Retrospective:
  - **Planned vs. actual:** M3 (WebGPU viewer, shareable links, WBOIT transparency, glTF 2.0 `.glb` export, renderer-neutral `.tgbin` bundle format, GPL Blender add-on/headless Cycles driver, and CLI `--engine cycles`) completed in 8 focused tasks (`M3.1`–`M3.8`) with zero scope cuts.
  - **What worked well:** Keeping `tetraglass-export` 100% `wasm32-unknown-unknown`-compatible and strictly data-only (`D-011`) made both `.glb` and `.tgbin` bundle exports fast and testable without Blender installed, while the pure-Python + `numpy` scripts in `blender/` (`D-020`) work seamlessly with Blender 4.2 LTS and 5.x LTS. Using `Standard` color management and `Is Camera Ray` world background separation in `blender/importer.py` ensured Cycles renders match the rasterizer's calibrated colormaps and white backgrounds out of the box.
  - **Adjustments for M4:** Milestone M4 (`M4.1` Cell locator through `M4.7` Gate and release 0.2) builds flow and volume visualization (`streamlines`, `glyphs`, `volume rendering`, and `animation export`). The multi-step `--all-steps` infrastructure added to `tetraglass-export` and `blender/render.py` in M3 already provides the foundation for `M4.5 Animation export`.
- Follow-ups: M4.1 Cell locator (`BVH over linearized tets for point location`).


## 2026-10-25: M4.1 Cell locator
- Done: implemented a deterministic median-split bounding volume hierarchy (`TetLocator`, `BvhNode`, `TetLocation`, `tet_barycentric_coordinates`, `DEFAULT_LOCATOR_TOLERANCE`, `DEFAULT_MAX_LEAF_TETS`, `DEFAULT_LOCATOR_CHUNK_SIZE`) over `TetView` in `tetraglass-filters` supporting $O(\log N)$ stack-based 3D point location (`locate`), $O(1)$ hint-accelerated stepping (`locate_with_hint`), chunk-parallel batch queries (`locate_many`, `locate_many_with_chunk_size`), source cell provenance (`source_cell_id`), and `f64` barycentric position/point/cell field interpolation; switched `TetView::tets` and `TetView::tet_source_cells` in `tetraglass-core` to `Arc<[[u32; 4]]>` / `Arc<[u32]>` (`tets_arc`, `tet_source_cells_arc`) for zero-copy sharing; added `TetLocator` benchmark to `surface_bench` (`19.35 ms` build on `600k` tets, `1.26 ms` for `10k` batch queries, `0.58 ms` for `10k` hinted steps).
- Tests: unit and doctests in `crates/tetraglass-filters/src/locator.rs` and `crates/tetraglass-filters/src/lib.rs`, plus integration test suite `crates/tetraglass-filters/tests/locator.rs` covering all 6 VTK cell families (`tet_linear`, `hex_linear`, `mixed_linear`, `tet_quadratic`, `hex_lagrange_p3`, `polyhedra`) and Gmsh `tet2_v41.msh`, exact reproduction of linear scalar (`linear`) and vector (`velocity`) fields, `cell_id` provenance, non-convex clipped domains, empty views, hint acceleration, and determinism across chunk sizes and leaf capacities.
- Decisions: none (implemented per D-005, D-008, and D-014).
- Dependencies added: none.
- Follow-ups: M4.2 Streamlines (point, line and plane seeds; RK4, then adaptive RK45; lines and tubes).

## 2026-10-25: M4.2a Streamline filter, RK4/RK45 integrators, seeds, and tube geometry
- Done: split `M4.2` into `M4.2a` (core `PolylineSet` + filter/integrators/seeds/tube geometry) and `M4.2b` (scene integration, line/tube styles, and gallery scene); added `PolylineSet` and `DEFAULT_TUBE_SIDES` (`12`) in `tetraglass-core` with rotation-minimizing parallel-transport tube surface generation (`to_tube_surface`), line surface conversion (`to_line_surface`), and `.vtu`-compatible `CellType::Line` conversion (`to_unstructured_grid`); implemented `streamlines`, `streamlines_with_locator`, `streamlines_with_chunk_size`, and `streamlines_grid` (`StreamlineParams`, `StreamlineSeeds::{Point, Points, Line, Plane}`, `StreamlineDirection::{Forward, Backward, Both}`, `StreamlineIntegrator::{Rk4, Rk45}`) in `tetraglass-filters` with Dormand-Prince 5(4) adaptive step-size control, boundary step-halving, `TetLocator::locate_with_hint` acceleration, and interpolation of all point/cell arrays plus `"arc_length"`, `"integration_time"`, and `"seed_id"` (`4.89 ms` to integrate 64 `Both`-direction `RK45` streamlines and `4.71 ms` to generate `60,672` tube triangles on `600k` tets).
- Tests: unit test `polyline_set_validates_inputs_and_builds_tubes_and_grids` and doctest in `tetraglass-core`, doctest in `crates/tetraglass-filters/src/streamlines.rs`, and integration tests in `crates/tetraglass-filters/tests/streamlines.rs` covering analytic circular flow (`velocity = [-y, x, 0]`) radius/arc-length/time conservation in `Rk4` and `Rk45`, all 6 VTK cell families and Gmsh `tet2_v41.msh`, `Point`/`Line`/`Plane` seeds, stagnation and out-of-bounds seeds, invalid parameter error cases, and chunk-parallel determinism.
- Decisions: none (implemented per D-005, D-008, and D-014).
- Dependencies added: none.
- Follow-ups: M4.2b Streamline scene integration, line/tube styles, and gallery scene.

## 2026-10-25: M4.2b Streamline scene integration, line/tube styles, and gallery scene
- Done: integrated `Filter::Streamlines` (`"type": "streamlines"`) and `"tubes"` / `"lines"` / `"wireframe"` streamline representation styles into `tetraglass-scene`:
  - `types.rs`: added `Filter::Streamlines`, `Filter::streamlines`, `StreamlineSeedsSpec` (`Point`, `Points`, `Line`, `Plane`), `StreamlineLineSeedSpec`, `StreamlinePlaneSeedSpec`, `StreamlineDirectionSpec` (`Forward`, `Backward`, `Both`), and `StreamlineIntegratorSpec` (`Rk45`, `Rk4`) with `#[serde(deny_unknown_fields, rename_all = "snake_case")]` and `schemars::JsonSchema` derives.
  - `validate.rs`: validated volume-input requirement, finite seed coordinates and positive counts/extents, positive `max_length` / `step_size` / `tube_radius`, 3-component vector array existence on `vectors` with edit-distance suggestions, and downstream `color_by` support for streamline-generated `"arc_length"`, `"integration_time"`, and `"seed_id"` arrays alongside interpolated source point and cell arrays.
  - `eval.rs`: added `EvaluatedDataset::Streamlines`, `locator_cache`, `streamline_cache`, `evaluate_surface_for_style`, and style-dependent streamline surface extraction (`12`-sided tubes for `"tubes"`, thin `6`-sided colormapped lines for `"lines"`, and `wireframe_edges` with promoted per-polyline attributes for `"wireframe"`).
  - Updated `crates/tetraglass-cli/tests/snapshots/scene-v1.schema.json`, `docs/scene-format.md`, `docs/book/src/filters.md`, `docs/book/src/gallery.md`, and `gallery/README.md`, and added `gallery/streamlines.json` with its visual regression baseline `tests/baselines/streamlines.png` (inspected per `/render-check`).
- Tests: unit test `parses_streamlines_filter_seed_variants_and_rejects_unknown_fields` in `tetraglass-scene`, `validates_streamlines_filter_vectors_seeds_parameters_and_output_arrays` in `crates/tetraglass-scene/tests/validation.rs`, `streamlines_filter_evaluates_tubes_lines_and_wireframe_and_caches_results` in `crates/tetraglass-scene/tests/evaluation_cache.rs`, and CLI schema snapshot + gallery baseline tests in `crates/tetraglass-cli/tests/cli.rs`.
- Decisions: none (implemented per D-004, D-008, `docs/scene-format.md`, and `.agents/rules/scene-schema.md`).
- Dependencies added: none.
- Follow-ups: M4.3 Glyphs (instanced arrows, cones and spheres, oriented and scaled by a vector).

## 2026-10-25: M4.3 Glyphs
- Done: implemented the `glyphs` filter (`GlyphShape::{Arrow, Cone, Sphere}`, `GlyphScaleMode::{Magnitude, Fixed}`, `GlyphAssociation::{Auto, Point, Cell}`, `GlyphParams`, `GlyphInstance`, `glyph_instances`, `glyphs`, `glyphs_with_chunk_size`, `glyphs_grid`, `DEFAULT_GLYPH_LENGTH_FRACTION`, `DEFAULT_GLYPH_CHUNK_SIZE`) in `tetraglass-filters` over `TetView` (`D-005`) with orthonormal orientation frames, vertex coordinate deduplication across linearized sub-tets, volume-weighted source-cell centroid sampling for cell vector arrays, configurable subsampling stride (`every`), and replication of all point/cell arrays plus `"glyph_magnitude"` and `triangle_source_cells`; integrated `Filter::Glyphs` (`"type": "glyphs"`, `GlyphShapeSpec`, `GlyphScaleBySpec`) into `tetraglass-scene` (`types.rs`, `validate.rs`, `eval.rs`); updated `crates/tetraglass-cli/tests/snapshots/scene-v1.schema.json`, `docs/scene-format.md`, `docs/book/src/filters.md`, `docs/book/src/gallery.md`, and `gallery/README.md`; added `gallery/glyphs.json` and its visual regression baseline `tests/baselines/glyphs.png` (inspected per `/render-check`).
- Tests: doctest in `crates/tetraglass-filters/src/glyphs.rs`, integration tests in `crates/tetraglass-filters/tests/glyphs.rs` (analytic orientation and scaling on `hex_linear.vtu`, `Arrow`/`Cone`/`Sphere` triangle counts and surface areas, point and cell vector arrays, all 6 VTK cell families and Gmsh `tet2_v41.msh`, chunk-parallel determinism, and invalid parameter errors), unit test `parses_glyphs_filter_shapes_and_scale_modes` in `tetraglass-scene`, `validates_glyphs_filter_vectors_parameters_and_augmented_arrays` in `crates/tetraglass-scene/tests/validation.rs`, `glyphs_filter_evaluates_shapes_and_caches_results` in `crates/tetraglass-scene/tests/evaluation_cache.rs`, and CLI schema snapshot + gallery baseline tests in `crates/tetraglass-cli/tests/cli.rs`.
- Decisions: none (implemented per D-004, D-005, D-008, `docs/scene-format.md`, and `.agents/rules/scene-schema.md`).
- Dependencies added: none.
- Follow-ups: M4.4 Volume rendering (resample to a regular grid; GPU ray marching; transfer function editor in the viewer).

## 2026-10-25: M4.4a Regular grid resampling filter and scene integration
- Done: split `M4.4` into `M4.4a` (regular grid resampling filter and scene integration) and `M4.4b` (GPU volume ray marching, transfer function editor, and gallery scene); added `RegularGrid` (`dimensions`, `bounds`, `inside_mask`, `point_data`, `num_points`, `num_inside_points`, `spacing`, `linear_index`, `grid_coords`, `point_position`, `is_inside`, `to_bounding_surface`) in `tetraglass-core`; implemented `resample`, `resample_with_locator`, `resample_with_chunk_size`, and `resample_grid` (`ResampleParams`, `DEFAULT_RESAMPLE_CHUNK_SIZE`) in `tetraglass-filters` using `TetLocator::locate_with_hint` along X scanlines to interpolate all point and cell arrays plus `"source_cell_id"` onto an `[nx, ny, nz]` regular grid while preserving exact interior scalar/magnitude ranges; integrated `Filter::Resample` (`"type": "resample"`), `EvaluatedDataset::Volume`, `EvaluatedRepresentation::volume`, and `Representation::transfer_function` (`Option<Vec<[f64; 2]>>`) into `tetraglass-scene`; updated `crates/tetraglass-cli/tests/snapshots/scene-v1.schema.json`, `docs/scene-format.md`, and `docs/book/src/filters.md`.
- Tests: unit test `regular_grid_validates_inputs_and_computes_positions_and_bounding_surface` in `tetraglass-core`, doctest in `crates/tetraglass-filters/src/resample.rs`, integration tests in `crates/tetraglass-filters/tests/resample.rs` (analytic linear and vector field reproduction, `inside_mask` on clipped and padded domains, all 6 VTK cell families and Gmsh `tet2_v41.msh`, chunk-parallel determinism, and invalid dimension errors), unit test `parses_resample_filter_and_volume_transfer_function` in `tetraglass-scene`, `validates_resample_dimensions_volume_style_and_transfer_function` in `crates/tetraglass-scene/tests/validation.rs`, and `resample_filter_evaluates_volume_and_caches_results` in `crates/tetraglass-scene/tests/evaluation_cache.rs`.
- Decisions: none (implemented per D-004, D-005, D-008, `docs/scene-format.md`, and `.agents/rules/scene-schema.md`).
- Dependencies added: none.
- Follow-ups: M4.4b GPU volume ray marching, transfer function editor, and gallery scene.

## 2026-10-26: M4.4b GPU volume ray marching, transfer function editor, and gallery scene
- Done: implemented 3D scalar texture upload (`VOLUME_TEXTURE_FORMAT = Rgba16Float` with normalized scalar in `R` and occupancy mask in `G` for universal hardware trilinear filtering) and fullscreen-triangle front-to-back GPU volume ray marching (`crates/tetraglass-render/src/shaders/volume.wgsl`, `GpuVolume`, `VolumeUniforms`, `evaluate_transfer_function_lut`, `DEFAULT_TRANSFER_FUNCTION_LUT_SIZE = 256`) in `tetraglass-render` with slab ray-box intersection, continuous partial-step exit integration (`seg_dt = min(dt, rem)`), depth-buffer ray termination (`load_scene_depth` + `inv_view_proj`) against opaque geometry, 1D colormap + piecewise-linear opacity transfer function lookup, gradient-assisted 3-point lighting, and shader clip-plane support; wired `volume_grid` and `transfer_function` into `build_surface_style` in `tetraglass-cli` and `tetraglass-viewer`; added `ViewerDocument::{is_volume_representation, transfer_function, set_transfer_function}` and an interactive 2D transfer function curve editor canvas + presets (`Linear`, `Soft`, `High peak`, `Mid band`) in `ViewerSession::show_ui` (`crates/tetraglass-viewer/src/lib.rs`); authored `gallery/volume.json` and its visual regression baseline `tests/baselines/volume.png` (inspected per `/render-check`), and updated `gallery/README.md` and `docs/book/src/gallery.md`.
- Tests: unit tests `evaluate_transfer_function_lut_interpolates_default_and_custom_points` and `invert_mat4_inverts_camera_view_proj` in `tetraglass-render`, headless integration test `volume_ray_marching_modulates_transfer_function_and_terminates_at_opaque_depth` in `crates/tetraglass-render/tests/render_mixed_linear.rs`, unit test `volume_document_updates_transfer_function_without_recomputing_filter` in `tetraglass-viewer`, and gallery baseline test for `gallery/volume.json` in `crates/tetraglass-cli/tests/cli.rs`.
- Decisions: none (implemented per D-002, D-004, D-008, D-015, and `.agents/rules/gpu-rendering.md`).
- Dependencies added: none.
- Follow-ups: M4.5 Animation export (PNG sequences from the rasterizer and frame ranges through Cycles).

## 2026-10-26: M4.5 Animation export
- Done: added `FramesConfig::{new, all, resolve_step_range, format_filename}`, `format_frame_filename` (`printf`-style `%04d` / `%d`, `{step}` / `{frame}` / `{step:04d}`, and plain filename fallback), semantic validation of `output.frames` (`from`, `to`, `start_step <= end_step`) against loaded time-series metadata, and `SceneEvaluator::{available_time_steps, resolve_frame_steps}` in `tetraglass-scene`; exposed `export_evaluated_steps_to_bundle` in `tetraglass-export`; updated `blender/importer.py` and `blender/render.py` to keyframe arbitrary step subsets (`manifest["time_steps"]`) and support `--from-step <n>` and `--to-step <n>` alongside `--all-steps`; extended `tetraglass render` in `tetraglass-cli` to render PNG sequences from `output.frames` or CLI flags `--all-steps`, `--from-step <n>`, and `--to-step <n>` in both `"raster"` and `"cycles"` modes with `RenderFrameSummary` entries in `RenderSummary::frames`; updated `docs/scene-format.md` and `docs/book/src/scene-format.md`.
- Tests: unit test `frames_config_resolves_step_ranges_and_formats_patterns` in `tetraglass-scene`, validation test `validates_output_frames_against_time_series_metadata` in `crates/tetraglass-scene/tests/validation.rs`, and CLI integration test `render_command_exports_animation_png_sequence_via_raster_and_cycles` (covering rasterizer full-series `output.frames`, rasterizer `--from-step` / `--to-step` subrange, and Cycles `--engine cycles --from-step 1 --to-step 2`) in `crates/tetraglass-cli/tests/cli.rs`.
- Decisions: none (implemented per D-004, D-011, and `docs/scene-format.md`).
- Dependencies added: none.
- Follow-ups: M4.6 Schema freeze (Scene schema v1 final, migration tests from earlier drafts).

## 2026-10-26: M4.6 Schema freeze
- Done: froze version 1 (`SCENE_VERSION = 1`) of the scene schema (`crates/tetraglass-cli/tests/snapshots/scene-v1.schema.json`) and updated `docs/scene-format.md`, `docs/book/src/scene-format.md`, and `AGENTS.md` from draft to final v1 status; added `MigrationReport`, `migrate_scene_value`, `migrate_scene_json_str`, and `Scene::from_draft_json_str` in `crates/tetraglass-scene/src/migrate.rs` to deterministically migrate pre-freeze M1–M4 draft conventions (missing or `0` version, `contour` `array`/`value`, `threshold` `array`/`range`, `streamlines`/`glyphs` `vector`/`field`, `"type": "volume"` / `dims`, `view.show[i]` `dataset`/`field`, and `output.frames` `start`/`end`) into canonical v1 `Scene`s while keeping `Scene::from_json_str` strictly enforcing `version == 1` and `deny_unknown_fields`.
- Tests: integration test suite in `crates/tetraglass-scene/tests/schema_freeze.rs` (`frozen_v1_schema_matches_snapshot_and_declares_all_v1_capabilities`, `all_gallery_scenes_are_canonical_v1_and_roundtrip_losslessly`, `migrates_m1_through_m4_draft_scenes_to_canonical_v1_and_evaluates`, and `migrates_versionless_minimal_draft_and_rejects_future_versions`).
- Decisions: none (implemented per D-004, `docs/scene-format.md`, and `.agents/rules/scene-schema.md`).
- Dependencies added: none.
- Follow-ups: M4.7 Gate and release 0.2 (`/milestone-gate`).

## 2026-10-26: M4.7 Gate, Release 0.2 Prep & Milestone M4 Retrospective (`docs/milestones/m4-gate.md`, `CHANGELOG.md`)
- Done: bumped workspace and internal crate versions to `0.2.0` in `Cargo.toml`; wrote the `[0.2.0] - 2026-10-26` release notes in `CHANGELOG.md` covering M3 and M4; rendered M4 gate demonstration figures (`docs/milestones/m4-streamlines.png`, `docs/milestones/m4-streamlines-cycles.png`, `docs/milestones/m4-glyphs.png`, `docs/milestones/m4-volume.png`); verified `cargo package --no-verify --offline` across all 9 crates at `0.2.0`, `cargo xtask verify`, and `cargo doc --workspace --no-deps`; and recorded the gate evidence in `docs/milestones/m4-gate.md`.
- Retrospective:
  - **Planned vs. actual:** Milestone M4 (`M4.1` Cell locator through `M4.7` Gate and release 0.2) completed in 9 focused sessions (`M4.1`, `M4.2a`, `M4.2b`, `M4.3`, `M4.4a`, `M4.4b`, `M4.5`, `M4.6`, `M4.7`) with zero scope cuts. Splitting `M4.2` (Streamlines) and `M4.4` (Volume rendering) into `a` (core/filter) and `b` (scene/render/viewer/gallery) sub-tasks kept each step under 2 crates and well within a single session.
  - **What worked well:** Building `TetLocator` (`M4.1`) once over `TetView` (`D-005`) powered both `streamlines` (`M4.2`, via `locate_with_hint` along RK4/RK45 trajectories) and `resample` (`M4.4`, via `locate_with_hint` along X scanlines) across all 6 VTK cell families and Gmsh higher-order meshes without special-casing any element type. Using `Rgba16Float` for the 3D volume texture (`M4.4b`) gave universal hardware trilinear filtering across all WebGPU / Metal / Vulkan / DX12 adapters, and scaling the final ray segment (`seg_dt = min(dt, t_exit - t)`) eliminated boundary step-quantization banding.
  - **Next milestone (`M5 Path tracer and beyond`):** M5 is open-ended and driven by user priorities; candidate first tasks in `docs/roadmap.md` are a compute-shader progressive path tracer in the viewer, screen-space ambient occlusion (SSAO), or a persistent Blender worker / live link.
- Follow-ups: Await human approval for `cargo publish` of `0.2.0` and git tagging (`v0.2.0`), or select the first M5 task in `docs/roadmap.md`.


## 2026-10-26: M5.1 Legacy VTK (`.vtk`) unstructured grid reader
- Done: defined the Milestone M5 task breakdown (`M5.1`–`M5.5`) in `docs/roadmap.md` and implemented the pure-Rust legacy VTK DataFile (`.vtk`, `DATASET UNSTRUCTURED_GRID`) reader (`read_vtk`, `read_vtk_series`, `read_vtk_from_path`) in `tetraglass-io` supporting VTK versions 1.0–5.1 in both `ASCII` and big-endian `BINARY` encodings, classic 2.0–4.2 `CELLS <n> <size>` and 5.0–5.1 `OFFSETS`/`CONNECTIVITY` layouts, all linear/quadratic/Lagrange/polyhedron cell types (both inline `CELLS` polyhedron face streams and VTK 5.x `FACES`/`FACE_LOCATIONS`), `POINT_DATA`/`CELL_DATA` (`SCALARS`, `VECTORS`, `NORMALS`, `TENSORS`, `TENSORS6`, `TEXTURE_COORDINATES`, and `FIELD` tables), dataset-level `TIME` metadata, and `METADATA`/`INFORMATION` blocks; generated `testdata/vtk/tet_linear_legacy.vtk`, `tet_linear_legacy_binary.vtk`, `tet_linear_v51.vtk`, and `tet_linear_v51_binary.vtk`; wired `MeshFormat::VtkLegacy` into `read_series_from_path`, `.pvd` step loading, `tetraglass-cli`, and `tetraglass-viewer`.
- Tests: `crates/tetraglass-io/tests/vtk_legacy.rs` (4.2 and 5.1 ASCII/binary equivalence with `.vtu` and round-tripping through `write_vtu_to_bytes`, all cell families including higher-order node ordering and polyhedra, `SCALARS`/`VECTORS`/`NORMALS`/`TENSORS`/`FIELD`/`METADATA`/`.pvd` steps, and truncation/corruption checks), `crates/tetraglass-io/tests/testdata_manifest.rs`, and `inspect_validate_and_render_legacy_vtk_files` in `crates/tetraglass-cli/tests/cli.rs`.
- Decisions: none (implemented per D-008, D-009, and D-018).
- Dependencies added: none.
- Follow-ups: M5.2 Screen-space ambient occlusion (SSAO).

## 2026-10-26: M5.2 Screen-space ambient occlusion (SSAO)
- Done: implemented screen-space ambient occlusion (`crates/tetraglass-render/src/shaders/ssao.wgsl`, `SsaoConfig`, `SSAO_RAW_FORMAT = R8Unorm`, `Renderer::{ssao, set_ssao}`, `HeadlessContext::render_surface_with_ssao`, and `Camera::near_far_f64`) in `tetraglass-render` with view-space position and min-delta depth-derivative normal reconstruction (`dpdx`/`dpdy`), a 16-tap golden-angle hemisphere kernel rotated by a $4 \times 4$ Bayer dither into a 1x `R8Unorm` intermediate attachment (`tetraglass_ssao_raw_pass`), and a $5 \times 5$ depth-aware bilateral blur that multiplicatively modulates opaque geometry (`tetraglass_ssao_apply_pass`) before WBOIT transparency, volume ray marching, wireframe edges, and annotations; exposed `ViewportSsaoConfig` and `ViewportState::{ssao, set_ssao}` in `tetraglass-egui`; added `ViewerDocument` / `ViewerSession` SSAO controls (`ssao`, `set_ssao`, `set_ssao_enabled`, `set_ssao_strength`, `set_ssao_radius`) and interactive `"Ambient occlusion (SSAO)"` checkbox and `"AO strength"` / `"AO radius"` sliders in `tetraglass-viewer`.
- Tests: headless visual regression test `ssao_darkens_concave_crevices_and_leaves_background_and_disabled_renders_intact` (verifying both `sample_count = 1` and `sample_count = 4` MSAA, bit-for-bit identity when `SsaoConfig::default()` is disabled, untouched background corners, and monotonic crevice darkening on a crinkle-clipped staircase mesh; inspected `target/render-tests/ssao_disabled.png` and `ssao_enabled.png` per `/render-check`) in `crates/tetraglass-render/tests/render_mixed_linear.rs`, offscreen viewport test in `crates/tetraglass-egui/tests/viewport_offscreen.rs`, and unit test `viewer_document_ssao_controls_validate_and_update` in `crates/tetraglass-viewer/src/lib.rs`.
- Decisions: none (implemented per D-002, D-003, D-015, and `.agents/rules/gpu-rendering.md`).
- Dependencies added: none.
- Follow-ups: M5.3a GPU triangle BVH and compute-shader progressive path tracer.

## 2026-10-26: M5.3a GPU triangle BVH and compute-shader progressive path tracer
- Done: implemented a 12-bucket Surface Area Heuristic (SAH) pre-order flattened linear triangle BVH (`TriangleBvh`, `BvhNode`, `PathTraceTriangle`, `PathTraceMaterial`, `PathTraceInput`) and compute-shader progressive path tracer (`crates/tetraglass-render/src/shaders/path_tracer.wgsl`, `crates/tetraglass-render/src/path_tracer.rs`, `PathTracer`, `PathTraceConfig`, `GpuPathTraceScene`, and `HeadlessContext::{render_surface_path_traced, render_surfaces_path_traced}`) in `tetraglass-render` with ordered near-first BVH stack traversal, Möller–Trumbore intersection, shader clip-plane culling, stochastic alpha pass-through, linear/log10 colormap evaluation (`Viridis`, `Cividis`, `CoolWarm`), soft area-light shadow rays, multi-bounce cosine-weighted diffuse global illumination, R2 sub-pixel anti-aliasing, progressive per-pixel storage-buffer accumulation with automatic reset on camera/viewport/scene changes, and a depth-writing present pass (`fs_pathtrace_present`) that composites wireframe edges (`draw_edges`) and 2D annotations (`colorbar`, `axes`, `title`) on top across all `wgpu` backends including WebGPU.
- Tests: unit test `triangle_bvh_matches_brute_force_ray_intersections_and_respects_clip_plane` in `crates/tetraglass-render/src/path_tracer.rs` and headless GPU integration test `compute_path_tracer_accumulates_progressively_and_renders_soft_shadows_and_multi_bounce_gi` in `crates/tetraglass-render/tests/render_mixed_linear.rs` (inspected `target/render-tests/path_traced_crinkle_1spp.png` and `path_traced_crinkle_32spp.png` per `/render-check`).
- Decisions: none (implemented per D-002, D-007, D-010, and D-014).
- Dependencies added: none.
- Follow-ups: M5.3b Progressive path tracer in the viewer and CLI.

## 2026-10-26 — M5.3b Progressive path tracer in the viewer and CLI

- Integrated the compute-shader progressive path tracer into `tetraglass-egui`:
  - Extended `ViewportState` (`crates/tetraglass-egui/src/lib.rs`) with `path_tracing_enabled`, `set_path_tracing_enabled`, `path_trace_config`, `set_path_trace_config`, `path_trace_accumulated_samples`, and `is_path_trace_converged`, and re-exported `PathTraceConfig as ViewportPathTraceConfig`.
  - Updated `ViewportPrepareCallback::prepare` to lazily build and upload `GpuPathTraceScene` from the active primary and additional surfaces when `path_tracing_enabled` is true, dispatch `PathTracer::step_with_overlays` (preserving wireframe edges and 2D annotations), and request continuous egui repaints while `!path_tracer.is_converged()`.
- Integrated interactive path-tracing controls into `tetraglass-viewer` (`crates/tetraglass-viewer/src/lib.rs`):
  - Added `ViewerDocument` and `ViewerSession` path-tracing state and validation methods (`set_path_trace_config`, `set_path_trace_samples`, `set_path_trace_bounces`, `set_path_trace_light_softness`).
  - Added a `"Progressive path tracer"` checkbox, sliders for `"Target samples"`, `"Max bounces"`, and `"Light softness"`, and a live convergence status counter (`"Path tracer: {accum} / {target} spp (converging... | converged)"`) in the left control panel.
- Extended `tetraglass render` in `tetraglass-cli` (`crates/tetraglass-cli/src/main.rs`) with `--engine path-tracer` (using `TriangleBvh`, `GpuPathTraceScene`, and `PathTracer::render_converged` with wireframe and 2D annotation overlays) and `--ssao` (enabling `SsaoConfig` in raster mode) without altering the frozen `v1` scene schema.
- Added offscreen progressive accumulation tests in `crates/tetraglass-egui/tests/viewport_offscreen.rs`, unit tests in `crates/tetraglass-viewer/src/lib.rs`, and CLI integration tests in `crates/tetraglass-cli/tests/cli.rs`, and visually verified `threshold_path_traced.png` and `threshold_ssao.png`.
- Verification: `cargo xtask verify` passed (format, clippy `-D warnings`, all workspace unit/integration/visual regression tests, `wasm32-unknown-unknown` build, and `cargo deny check`).
- Follow-ups: M5.4 Persistent Blender worker and live link.

## 2026-10-26: M5.4 Persistent Blender worker and live link
- Done: added `update_camera_and_view` in `blender/importer.py` for in-place camera, headlight, background, resolution, and step updates without re-importing `.tgbin` meshes, and `--serve <host:port|ipc:dir>` persistent worker mode (`run_socket_worker`, `_run_ipc_dir_worker`, `_dispatch_worker_request`) in `blender/render.py` handling `ping`, `load_bundle`, `update_camera`, `render`, and `shutdown`; implemented `BlenderWorkerRequest`, `BlenderWorkerRenderRequest`, `BlenderWorkerResponse`, `BlenderWorkerClient`, and `BlenderWorkerProcess` in `crates/tetraglass-export/src/worker.rs`; added `--worker <addr>` to `tetraglass render --engine cycles` in `tetraglass-cli` and live-link camera syncing (`ViewerSession::{blender_worker_addr, set_blender_worker_addr, blender_live_link_enabled, set_blender_live_link_enabled, build_live_worker_camera_payload, push_live_camera_to_worker}`) in `tetraglass-viewer`.
- Tests: `blender_worker_client_protocol_round_trips_and_propagates_errors` and persistent `BlenderWorkerProcess` live camera update + multi-render test in `crates/tetraglass-export/tests/bundle_export.rs` (visually verified `worker_slice_cam1.png` and `worker_slice_cam2.png`), plus `tetraglass render --engine cycles --worker` integration test in `crates/tetraglass-cli/tests/cli.rs`.
- Decisions: none (implemented per D-002, D-011, and `.agents/rules/blender-gpl.md`).
- Dependencies added: none.
- Follow-ups: M5.5 Gate (`/milestone-gate`).

## 2026-10-26: M5.5 Gate & Milestone M5 Retrospective (`docs/milestones/m5-gate.md`)
- Done: verified the Milestone M5 gate across all 9 crates (`cargo xtask verify` and `cargo doc --workspace --no-deps`); rendered the M5 gate demonstration figures (`docs/milestones/m5-threshold-pathtracer.png`, `docs/milestones/m5-threshold-ssao.png`, and `docs/milestones/m5-streamlines-pathtracer.png`); and recorded the gate evidence in `docs/milestones/m5-gate.md`.
- Retrospective:
  - **Planned vs. actual:** Milestone M5 (`M5.1` Legacy `.vtk` reader, `M5.2` SSAO, `M5.3a`/`M5.3b` GPU triangle BVH and compute-shader progressive path tracer, `M5.4` Persistent Blender worker and live link, and `M5.5` Gate) completed in 6 focused steps with zero scope cuts and zero external C or GPU ray-tracing extension dependencies.
  - **What worked well:** Implementing the progressive path tracer in a pure WGSL compute shader (`path_tracer.wgsl`) over a 12-bucket SAH pre-order flattened linear BVH (`TriangleBvh`) made hardware-agnostic global illumination, soft shadows, and progressive accumulation work identically on Metal, Vulkan, DX12, and WebGPU without requiring vendor-specific RT extensions (`D-010`). Supporting both loopback TCP (`host:port`) and atomic directory IPC (`ipc:<dir>`) in `blender/render.py --serve` and `BlenderWorkerClient` allowed warm multi-render Cycles sessions and live camera updates to run both on standard desktops and inside network-isolated sandboxes.
  - **All roadmap milestones (`M0`–`M5`) complete:** Future enhancements are driven by user feedback and real-world FEM/CFD datasets (e.g., CGNS/EXODUS/VTKHDF readers, Python bindings `D-001` on demand, or multi-volume rendering).
- Follow-ups: All planned roadmap tasks across Milestones `M0`–`M5` are complete.

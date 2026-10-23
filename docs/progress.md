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

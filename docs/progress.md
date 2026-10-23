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

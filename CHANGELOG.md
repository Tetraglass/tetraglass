# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-26

### Added

- **Umbrella `tetraglass` crate & fluent Rust `SceneBuilder` API (`tetraglass`, `tetraglass-scene`):**
  - New `tetraglass` umbrella crate (`crates/tetraglass`) re-exporting `core`, `io`, `filters`, `scene`, `export`, `render`, `egui`, and `viewer` behind optional feature flags (`default = ["io", "filters", "scene", "export", "render", "egui"]`, `viewer`, `full`) along with `tetraglass::prelude` (`D-017`).
  - Typed fluent Rust builder API (`Scene::builder()`, `SceneBuilder`, `DatasetRef`, `SurfaceFilter`, `SliceFilter`, `ContourFilter`, `ClipFilter`, `ThresholdFilter`, `StreamlinesFilter`, `GlyphsFilter`, `ResampleFilter`, `RepresentationBuilder`, `CameraBuilder`, `OutputBuilder`) in `tetraglass-scene`.
- **Legacy VTK (`.vtk`) & Parallel VTK XML (`.pvtu`) readers (`tetraglass-io`, `tetraglass-cli`, `tetraglass-viewer`):**
  - Pure-Rust legacy VTK DataFile reader (`read_vtk`, `read_vtk_series`, `read_vtk_from_path`, `MeshFormat::VtkLegacy`) supporting versions `2.0`–`5.1` in `ASCII` and big-endian `BINARY` encodings, classic 4.x `CELLS` and 5.x `OFFSETS`/`CONNECTIVITY`, `CELL_TYPES`, `FACES`/`FACEOFFSETS` polyhedra, and `POINT_DATA`/`CELL_DATA` (`SCALARS`, `VECTORS`, `NORMALS`, `TENSORS`, `FIELD`).
  - Pure-Rust Parallel VTK XML `.pvtu` reader (`parse_pvtu_manifest`, `read_pvtu_with_loader`, `read_pvtu_series_with_loader`, `read_pvtu_from_path`, `read_pvtu_series_from_path`, `PvtuManifest`, `PvtuPiece`, `MeshFormat::Pvtu`) with `vtkGhostType` ghost-cell and ghost-point filtering, shared partition-interface vertex welding across pieces, polyhedron face-stream remapping, and `.pvd` time-series `.pvtu` step support.
- **Screen-space `"points"` representation, SSAO & compute-shader progressive path tracer (`tetraglass-render`, `tetraglass-egui`, `tetraglass-viewer`, `tetraglass-cli`):**
  - Instanced screen-space circular sphere-impostor point-cloud rendering (`SurfaceStyle::{draw_points, point_size}`, `GpuSurface::{point_count, draw_points, point_size}`) supporting solid, point-scalar, and cell-scalar coloring, view-space sphere-impostor lighting, depth testing, WBOIT transparency, shader clip-plane culling, GPU ID buffer picking, interactive `"Point size"` controls in `tetraglass-viewer`, and `gallery/points.json`.
  - Screen-space ambient occlusion (`ssao.wgsl`, `SsaoConfig`, `Renderer::set_ssao_config`) reconstructing view-space positions and normals from the hardware depth buffer with a 16-sample rotated hemisphere kernel and depth-aware bilateral blur, plus `--ssao` in `tetraglass render` and interactive SSAO controls in `tetraglass-egui` and `tetraglass-viewer`.
  - 12-bucket SAH linear `TriangleBvh` and compute-shader progressive path tracer (`path_tracer.wgsl`, `PathTracer`, `PathTraceConfig`, `GpuPathTraceScene`) with multi-bounce cosine-weighted indirect lighting, soft area-light shadow rays, sky-dome illumination, progressive temporal accumulation when the camera is stationary in `tetraglass-egui` and `tetraglass-viewer`, and `tetraglass render --engine path-tracer [--samples <n>]` (`D-010`).
- **Persistent Blender worker & live camera link (`blender/`, `tetraglass-export`, `tetraglass-cli`, `tetraglass-viewer`):**
  - Persistent worker server mode (`blender/render.py --serve <host:port|ipc:dir>`) and in-place camera/view updater (`update_camera_and_view` in `blender/importer.py`) keeping a warm headless Blender session across renders without re-importing `.tgbin` meshes.
  - `BlenderWorkerClient` and `BlenderWorkerProcess` in `tetraglass-export`, `tetraglass render --engine cycles --worker <addr>` in `tetraglass-cli`, and live-link camera syncing (`ViewerSession::push_live_camera_to_worker`) in `tetraglass-viewer`.
- **`warp_by_vector` & `calculator` array filters (`tetraglass-filters`):**
  - `warp_by_vector` (`WarpByVectorParams`, `warp_by_vector`, `warp_by_vector_with_chunk_size`, `warp_by_vector_grid`, `warp_unstructured_grid`) displacing vertex coordinates in `f64` by `scale_factor * vector_field` while preserving connectivity and attributes zero-copy via `Arc`.
  - `calculator` (`CalculatorAssociation`, `CalculatorParams`, `calculator`, `calculator_with_chunk_size`, `evaluate_calculator`, `calculator_grid`, `calculator_unstructured_grid`) evaluating scalar and 3-vector mathematical expressions over point/cell arrays and spatial coordinates (`x`, `y`, `z`, `coords`) into a derived `f64` `DataArray`.

### Changed

- Updated `tetraglass-scene` streamline surface extraction to emit station points (`polylines.to_line_surface()`) when `RepresentationStyle::Points` is selected.
- Extended `cargo xtask wasm-check` to verify the `tetraglass` umbrella crate on `wasm32-unknown-unknown`.

### Fixed

- Enabled GPU ID buffer picking and CPU vertex probing (`probe_surface_triangle`) on 0-triangle point-cloud and polyline surfaces when `draw_points` is active.

## [0.2.0] - 2026-10-26

### Added

- **Web viewer, shareable links & transparency (`tetraglass-viewer`, `tetraglass-render`):**
  - `wasm32-unknown-unknown` WebGPU browser viewer (`web/index.html`, `ViewerSession`) with drag-and-drop and URL loading (`?scene=`, `?mesh=`, `?data=`) plus a clear diagnostic banner when `navigator.gpu` is unavailable (`D-007`).
  - Self-contained `#scene=<base64url>` and `#scene_z=<deflate-base64url>` shareable URL fragment encoding/decoding (`encode_shareable_scene_fragment`, `parse_location_query`, `resolve_relative_url`, `ViewerSession::build_shareable_link`) with view-state overrides (`?field=`, `?colormap=`, `?step=`, `?az=`, `?el=`, `?zoom=`, `?opacity=`).
  - Weighted Blended Order-Independent Transparency (WBOIT, McGuire & Bavoil 2013) in `tetraglass-render` (`oit_accum.wgsl`, `oit_resolve.wgsl`) with `Rgba16Float` accumulation, `R16Float` revealage, read-only opaque depth testing, and interactive opacity controls in `tetraglass-viewer`.
- **glTF 2.0, renderer-neutral bundle export & Blender Cycles bridge (`tetraglass-export`, `blender/`, `tetraglass-cli`):**
  - New `tetraglass-export` crate (`wasm32-unknown-unknown`-compatible) with Binary glTF 2.0 (`.glb`) exporters (`export_evaluated_scene_to_glb`, `export_time_series_to_glb_frames`, `parse_glb`) storing `f32` positions relative to `dataset_center` (`D-014`), baked linear-light RGBA vertex colors, and `[normalized_scalar, raw_scalar]` in `TEXCOORD_1`.
  - Renderer-neutral v1 bundle exporter and reader (`docs/bundle-format.md`, `BundleArchive`, `encode_tgbin`, `decode_tgbin`, `export_evaluated_scene_to_bundle`, `export_evaluated_steps_to_bundle`, `export_scene_to_bundle`, `write_bundle_dir`, `read_bundle_dir`).
  - GPL-3.0-or-later Blender 4.2 LTS / 5.x LTS add-on and headless Cycles render driver in `blender/` (`importer.py`, `render.py`, `__init__.py`, `blender_manifest.toml`) and CLI `tetraglass export` / `tetraglass render --engine cycles [--samples <n>] [--device <dev>] [--save-blend <path.blend>]` (`D-010`, `D-011`, `D-020`).
- **BVH cell locator, streamlines, glyphs & volume rendering (`tetraglass-core`, `tetraglass-filters`, `tetraglass-render`, `tetraglass-viewer`):**
  - `TetLocator` median-split BVH over `TetView` (`locate`, `locate_with_hint`, `locate_many`) with `f64` barycentric coordinate evaluation and point/cell attribute interpolation (`19.35 ms` build on `600,000` tets).
  - `PolylineSet` (`to_tube_surface`, `to_line_surface`, `to_unstructured_grid`) in `tetraglass-core` and `streamlines` filter (`StreamlineParams`, `StreamlineSeeds::{Point, Points, Line, Plane}`, `StreamlineDirection::{Forward, Backward, Both}`, `StreamlineIntegrator::{Rk4, Rk45}`) in `tetraglass-filters` with `"tubes"`, `"lines"`, and `"wireframe"` styles and `gallery/streamlines.json`.
  - `glyphs` filter (`GlyphShape::{Arrow, Cone, Sphere}`, `GlyphScaleMode::{Magnitude, Fixed}`, `GlyphAssociation`, `GlyphParams`, `glyph_instances`, `glyphs`) in `tetraglass-filters` with orthonormal orientation frames, stride subsampling (`every`), and `gallery/glyphs.json`.
  - `RegularGrid` in `tetraglass-core` and `resample` filter (`ResampleParams`, `resample`, `resample_with_locator`) in `tetraglass-filters` sampling point and cell arrays onto a `[nx, ny, nz]` regular grid with domain occupancy mask (`inside_mask`).
  - Front-to-back GPU volume ray marching (`volume.wgsl`, `GpuVolume`, `evaluate_transfer_function_lut`) in `tetraglass-render` with `Rgba16Float` 3D hardware trilinear filtering, continuous partial-step exit integration, opaque depth-buffer termination, 1D colormap + piecewise-linear `transfer_function` lookup, gradient-assisted 3-point lighting, interactive 2D transfer function curve editor in `tetraglass-viewer`, and `gallery/volume.json`.
- **Animation export & scene schema v1 freeze (`tetraglass-scene`, `tetraglass-cli`):**
  - Multi-frame PNG sequence export in `tetraglass render` from `scene.output.frames` (`FramesConfig`, `format_frame_filename`) or CLI flags `--all-steps`, `--from-step <n>`, and `--to-step <n>` across both `"raster"` and `"cycles"` engines.
  - Frozen scene schema v1 (`SCENE_VERSION = 1`, `crates/tetraglass-cli/tests/snapshots/scene-v1.schema.json`) and deterministic draft-to-v1 migration helpers (`MigrationReport`, `migrate_scene_value`, `migrate_scene_json_str`, `Scene::from_draft_json_str`).

### Changed

- Switched `TetView::tets` and `TetView::tet_source_cells` in `tetraglass-core` to `Arc<[[u32; 4]]>` and `Arc<[u32]>` (`tets_arc`, `tet_source_cells_arc`) for zero-copy sharing with `TetLocator`.
- Marked `docs/scene-format.md` as **final (version 1)** for release `0.2.0`.

### Fixed

- Eliminated boundary step-quantization banding in `volume.wgsl` by scaling the final ray segment (`seg_dt = min(dt, t_exit - t)`) before exiting the volume bounding box.

## [0.1.0] - 2026-10-24

### Added

- **Core data model (`tetraglass-core`):**
  - `UnstructuredGrid` storing `f64` vertex coordinates, `u32` connectivity (`D-013`), and linear (`Vertex`, `Line`, `Triangle`, `Quad`, `Tetra`, `Hexahedron`, `Wedge`, `Pyramid`), quadratic (`Quadratic*`), arbitrary-order Lagrange (`Lagrange*`), and polyhedral (`Polyhedron`) cells.
  - `DataArray` and `Attributes` supporting `f32`, `f64`, `i32`, `i64`, and `u8` storage behind `Arc` with `1`, `3`, `6`, or `9` components and cached per-component and `L2`-magnitude value ranges.
  - Lazy LRU-cached `TimeSeries` (`DEFAULT_STEP_CACHE_CAPACITY = 2`), `SurfaceMesh`, `TetView`, and deterministic synthetic benchmark mesh generators (`SyntheticGridSpec`, `M2_PERFORMANCE_TARGET`).
- **Pure-Rust byte-buffer readers & writers (`tetraglass-io`):**
  - VTK XML `.vtu` reader (`read_vtu`, `read_vtu_from_path`) and writer (`write_vtu`, `write_vtu_to_bytes`) supporting `ascii`, inline `binary` (base64), and `appended` (`base64` and `raw`) encodings, uncompressed and `vtkZLibDataCompressor` payloads, and both classic and VTK 9.4+ polyhedron face streams.
  - ParaView `.pvd` collection reader (`parse_pvd_entries`, `read_pvd_series_with_loader`, `read_pvd_series_from_path`) returning a lazy `TimeSeries`.
  - Gmsh `.msh` reader (`read_msh`, `read_msh_series`, `read_msh_from_path`) supporting MSH `4.1` and `2.2` in ASCII and binary, `$NodeData` and `$ElementData` time steps, and node-ordering permutation tables into canonical VTK ordering (`gmsh_element_spec`).
- **Linearization & visualization filters (`tetraglass-filters`):**
  - `linearize` and `linearize_default` converting any `UnstructuredGrid` into a `TetView` with face-consistent diagonal decomposition for linear 3D cells, reference-space Lagrange/serendipity subdivision for higher-order cells, and volume-weighted centroid decomposition for polyhedra while preserving `tet_source_cells`.
  - `extract_external_surface` and `extract_tet_view_surface` with curved higher-order boundary tessellation and original element perimeter wireframe edges.
  - Shared deterministic chunk-parallel marching-tetrahedra kernel (`marching_tets`) powering:
    - Plane `slice` (`SliceParams`, `slice`, `slice_grid`).
    - Multi-isovalue `contour` (`ContourParams`, `ContourComponent`, `contour`, `contour_grid`) with full attribute interpolation.
    - Half-space `clip` (`ClipParams`, `ClipMode::Exact`, `ClipMode::Crinkle`, `ClipKeep`, `clip`, `clip_grid`) with conformal 1-tet / 3-tet prism decomposition in exact mode and whole-cell retention in crinkle mode.
  - Whole-cell scalar/component/magnitude `threshold` (`ThresholdParams`, `ThresholdAssociation`, `ThresholdComponent`, `threshold`, `threshold_grid`).
- **Declarative scene model (`tetraglass-scene`):**
  - `Scene` (`version: 1`) serde types (`Source`, `Filter`, `Representation`, `View`, `CameraConfig`, `Output`), `schemars` JSON Schema generation (`scene_json_schema`), semantic validation with Damerau-Levenshtein suggestions (`validate_scene`), and input-hash cached pipeline evaluation (`SceneEvaluator`).
- **Offscreen `wgpu` renderer (`tetraglass-render`):**
  - Embeddable `Renderer` and `OffscreenTarget` (`1x` and `4x` MSAA) drawing relative to the dataset center (`D-014`) with headlight + key/fill lighting, 256-entry 1D colormaps (`Viridis`, `Cividis`, `CoolWarm`), flat per-triangle cell scalar storage buffers, and original element wireframes.
  - Renderer-drawn 2D/3D viewport annotations (`Annotations`, `ColorbarAnnotation`, 3D orientation axes triad, and title text rasterized via `skrifa` + `vello_cpu`).
  - Interactive per-fragment `ShaderClipPlane` uniform (`set_surface_clip_plane`).
  - GPU ID buffer picking (`PickTarget`, `Renderer::pick_pixel`) and CPU barycentric value probing (`probe_surface_triangle`, `SurfacePickResult`).
  - `HeadlessContext` and tolerance-based image regression comparator (`RenderedImage::compare_with_baseline`).
- **Embeddable `egui` viewport & desktop viewer (`tetraglass-egui`, `tetraglass-viewer`, `tetraglass-cli`):**
  - `ViewportState` and `ViewportWidget` rendering offscreen in `egui_wgpu::CallbackTrait::prepare` with orbit/pan/zoom camera controls, click-to-probe crosshair/badge overlay, and multi-representation support.
  - `tetraglass-viewer` desktop application (`ViewerApp`, `ViewerSession`) with live scene file hot-reload, time step scrubbing, field/colormap pickers, value probe inspector, and interactive shader clip-plane dragging with background exact-clip worker swap.
  - `tetraglass` CLI (`inspect`, `validate`, `render`, `schema`, `view`).
  - User and developer mdBook guide (`docs/book/`) and 9-scene visual regression gallery (`gallery/`).

### Changed

- **Performance pass (`M2.9`, `D-019`):** Replaced `SipHash` face/edge maps and heap-allocated polygon keys in `tetraglass-filters` with `Copy` structs, inline sorting networks (`sort3`, `sort4`), flat vertex-bucketed linked lists, and `FxHasher`, achieving `5.3 ms` `slice`, `9.0 ms` `contour`, `18–25 ms` volume `clip`/`threshold` + surface extraction on `600,000` tetrahedra and `0.88–1.53 ms/frame` (`650–1130 FPS`) at `1200 × 800` `4x` MSAA.

### Fixed

- Preserved `dataset_center` and interactive camera state across `fit_camera == false` background worker surface swaps in `tetraglass-egui`.

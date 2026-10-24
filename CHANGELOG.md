# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

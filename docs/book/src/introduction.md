# Introduction

**Tetraglass** is a Rust-native scientific visualization toolkit for finite element (FEM) and computational fluid dynamics (CFD) results on unstructured meshes. Built on `wgpu`, it embeds directly into Rust and `egui` applications, renders headlessly to PNG without a window or display server, compiles its data and filter pipeline to `wasm32-unknown-unknown`, and exposes a declarative JSON scene contract and CLI designed for both humans and coding agents.

## Key capabilities (v0.1 / Milestone M2)

- **Native unstructured mesh storage (`tetraglass-core`):** Stores linear (`Vertex`, `Line`, `Triangle`, `Quad`, `Tetra`, `Hexahedron`, `Wedge`, `Pyramid`), quadratic serendipity/Lagrange (`Quadratic*`), arbitrary-order Lagrange (`Lagrange*`), and arbitrary polyhedral (`Polyhedron`) cells natively using `u32` connectivity (`D-013`), `f64` CPU coordinates and attributes (`D-014`), and lazy LRU-cached `TimeSeries`.
- **Pure-Rust byte-buffer readers (`tetraglass-io`):** Reads and writes VTK XML `.vtu` (ASCII, inline base64, raw and base64 appended binary, uncompressed and `vtkZLibDataCompressor`), ParaView `.pvd` transient collections, and Gmsh `.msh` (MSH `4.1` and `2.2`, ASCII and binary, with `$NodeData` and `$ElementData` and automatic permutation into canonical VTK node ordering).
- **Linearized tetrahedral view & filters (`tetraglass-filters`):** Converts any `UnstructuredGrid` into a [`TetView`] with reference-space Lagrange/serendipity subdivision for higher-order cells, centroid decomposition for polyhedra, and a `tet_source_cells` map back to original cells (`D-005`). Provides curved external surface extraction (`extract_external_surface`), plane `slice`, multi-isovalue `contour`, `clip` (`exact` conformal prism decomposition and `crinkle` whole-cell retention), and `threshold` (`D-006`).
- **Declarative scene graph (`tetraglass-scene`):** Plain `serde` structs with a generated JSON Schema (`tetraglass schema`), semantic validation with "did you mean?" typo suggestions, and lazy cached pipeline evaluation (`SceneEvaluator`).
- **Embeddable `wgpu` renderer (`tetraglass-render`):** Takes the host's `wgpu::Device` and `wgpu::Queue` (`D-002`), uploads vertex positions in `f32` relative to the dataset center (`D-014`), draws shaded surfaces, original element wireframes, 256-entry colormaps (`Viridis`, `Cividis`, `CoolWarm`), colorbars, 3D orientation axes triads, titles, interactive shader clip planes, and GPU ID-buffer picking (`PickTarget`).
- **Embeddable `egui` viewport & desktop viewer (`tetraglass-egui`, `tetraglass-viewer`, `tetraglass-cli`):** Drop-in `ViewportWidget` for `egui-wgpu` apps (`D-003`), an `eframe` desktop viewer (`tetraglass view`) with live file watching, interactive shader+worker clipping, and click-to-probe value inspection, plus a scriptable CLI (`inspect`, `validate`, `render`, `schema`, `view`).

## Crate architecture

Dependencies flow strictly one way (`core <- io, filters <- scene <- render <- egui <- viewer / cli`):

| Crate | Purpose | `wasm32` compatible |
| --- | --- | --- |
| `tetraglass-core` | `UnstructuredGrid`, `CellType`, `DataArray`, `TimeSeries`, `SurfaceMesh`, `TetView` | Yes |
| `tetraglass-io` | VTK (`.vtu`, `.pvd`) and Gmsh (`.msh`) byte-buffer readers and `.vtu` writer | Yes |
| `tetraglass-filters` | `linearize`, `extract_external_surface`, `slice`, `contour`, `clip`, `threshold` | Yes |
| `tetraglass-scene` | Declarative `Scene` model, JSON Schema, validation, cached `SceneEvaluator` | Yes |
| `tetraglass-render` | `wgpu` offscreen renderer, colormaps, annotations, GPU ID picking, headless helper | WebGPU / Native |
| `tetraglass-egui` | `ViewportState` and `ViewportWidget` for `egui-wgpu` hosts | WebGPU / Native |
| `tetraglass-viewer` | Desktop `eframe` viewer application library | Native (Web in M3) |
| `tetraglass-cli` | `tetraglass` binary (`inspect`, `validate`, `render`, `schema`, `view`) | Native |

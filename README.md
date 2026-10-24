# Tetraglass

A Rust-native scientific visualization library for FEM/CFD results on unstructured meshes, built on `wgpu`. It embeds in Rust and `egui` apps, renders headlessly to PNG without a window or display server, compiles its data and filter pipeline to `wasm32-unknown-unknown`, and can be driven by coding agents through declarative JSON scene files and a CLI.

## Quickstart

### CLI & Viewer

```sh
# 1. Inspect a VTK (.vtu / .pvd) or Gmsh (.msh) dataset
cargo run -p tetraglass-cli -- inspect testdata/vtk/hex_linear.vtu

# 2. Validate a declarative JSON scene
cargo run -p tetraglass-cli -- validate gallery/slice.json

# 3. Render headlessly to PNG (prints JSON summary to stdout)
cargo run -p tetraglass-cli -- render gallery/slice.json -o slice.png

# 4. Open the interactive desktop viewer (live file reload, click-to-probe, interactive clip)
cargo run -p tetraglass-cli -- view gallery/clip.json

# 5. Print the v1 JSON Schema
cargo run -p tetraglass-cli -- schema
```

## Workspace Crates

| Crate | Description | `wasm32` |
| --- | --- | --- |
| `tetraglass-core` | `UnstructuredGrid`, `CellType` (linear, quadratic, Lagrange, polyhedra), `DataArray`, `TimeSeries`, `SurfaceMesh`, `TetView` | Yes |
| `tetraglass-io` | Pure-Rust byte-buffer readers for `.vtu`, `.pvd`, and `.msh` (MSH 4.1 & 2.2) plus `.vtu` writer | Yes |
| `tetraglass-filters` | `linearize`, `extract_external_surface`, `slice`, `contour`, `clip` (`exact` & `crinkle`), `threshold` | Yes |
| `tetraglass-scene` | Declarative `Scene` model, JSON Schema, semantic validation, and cached `SceneEvaluator` | Yes |
| `tetraglass-render` | Embeddable `wgpu` offscreen renderer, colormaps, annotations, GPU ID picking, headless helper | WebGPU / Native |
| `tetraglass-egui` | Embeddable `ViewportState` and `ViewportWidget` for `egui-wgpu` applications | WebGPU / Native |
| `tetraglass-viewer` | `eframe` desktop viewer application library | Native (Web in M3) |
| `tetraglass-cli` | `tetraglass` command-line binary (`inspect`, `validate`, `render`, `schema`, `view`) | Native |

## Documentation

- **User & Developer Guide (mdBook):** [`docs/book/`](docs/book/src/SUMMARY.md)
  - [Introduction](docs/book/src/introduction.md)
  - [Getting Started](docs/book/src/getting-started.md)
  - [Embedding in Rust & egui](docs/book/src/embedding.md)
  - [Scene Format & Agent Workflow](docs/book/src/scene-format.md)
  - [Linearization & Filters](docs/book/src/filters.md)
  - [Gallery](docs/book/src/gallery.md)
- **Scene Format Specification:** [`docs/scene-format.md`](docs/scene-format.md)
- **Architecture & Design Plan:** [`docs/design-plan.md`](docs/design-plan.md)
- **Binding Decisions:** [`docs/decisions.md`](docs/decisions.md)
- **Roadmap & Progress Log:** [`docs/roadmap.md`](docs/roadmap.md), [`docs/progress.md`](docs/progress.md)
- **API Docs:** run `cargo doc --workspace --no-deps --open`

## Development & Verification

```sh
# Format, clippy (-D warnings), unit/integration/doc tests, wasm32 check, and cargo-deny
cargo xtask verify

# Run CPU filter and GPU rendering benchmarks (D-019 target)
cargo bench -p tetraglass-filters --bench surface_bench
cargo bench -p tetraglass-render --bench render_bench
```

## License

The Rust crates are dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. The contents of `blender/` are licensed separately under GPL-3.0-or-later because they use Blender's Python API; see `blender/README.md`.

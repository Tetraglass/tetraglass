# Getting Started

This chapter walks through inspecting a mesh, validating and rendering a declarative JSON scene from the command line, opening the interactive desktop viewer, and driving the same pipeline from Rust code.

## 1. Command-line quickstart

Build and run the `tetraglass` CLI from the workspace root:

### Inspect a dataset

Use `tetraglass inspect` (with `--json` for machine-readable output) to list cell types, bounding box coordinates, time steps, and point/cell arrays with their cached value ranges:

```sh
cargo run -p tetraglass-cli -- inspect testdata/vtk/hex_linear.vtu
cargo run -p tetraglass-cli -- inspect --json testdata/vtk/series.pvd
```

### Validate and render a scene to PNG

Validate a scene file against both the v1 schema and the referenced dataset's array metadata, then render it headlessly to a PNG image:

```sh
cargo run -p tetraglass-cli -- validate gallery/slice.json
cargo run -p tetraglass-cli -- render gallery/slice.json -o slice.png
```

`tetraglass render` writes `slice.png` and prints a JSON summary to `stdout` reporting the output dimensions, bounding box, camera angles, and every rendered representation's triangle count, field association, colormap, and resolved `[min, max]` range.

### Open the interactive desktop viewer

```sh
cargo run -p tetraglass-cli -- view gallery/clip.json
```

In the viewer:
- **Left-drag** orbits the camera, **Right-drag** (or **Shift + Left-drag**) pans, and **Scroll** zooms.
- Editing and saving `gallery/clip.json` in your editor hot-reloads the viewport automatically while preserving your interactive camera when the JSON camera block is unchanged.
- **Left-clicking** without dragging probes the triangle under the cursor via the GPU ID buffer and displays its 3D world coordinates, source cell ID, triangle ID, and interpolated point/cell field values.
- Enabling **Interactive clip** in the left panel lets you drag a real-time shader clip plane (`<= 1 ms/frame`) while a background worker computes the exact conformal tetrahedral clip when you release the slider.

## 2. Rust library quickstart

Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
tetraglass-core = "0.1"
tetraglass-io = "0.1"
tetraglass-filters = "0.1"
tetraglass-render = "0.1"
```

### Load a mesh, slice it, and render to PNG

```rust,no_run
use tetraglass_filters::{SliceParams, extract_external_surface, linearize_default, slice};
use tetraglass_io::read_vtu_from_path;
use tetraglass_render::{
    Annotations, Camera, ColorBy, ColorbarAnnotation, ColormapKind, DEFAULT_COLOR_FORMAT,
    HeadlessContext, OffscreenTarget, Renderer, SurfaceStyle,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Read an unstructured grid from a VTK .vtu file (or byte slice via `read_vtu(&bytes)`).
    let grid = read_vtu_from_path("testdata/vtk/hex_linear.vtu")?;
    let bounds = grid.bounds().expect("non-empty mesh");

    // 2. Linearize into tetrahedra and extract a diagonal plane slice.
    let tet_view = linearize_default(&grid)?;
    let slice_surface = slice(&tet_view, &SliceParams::new([1.0, -0.8, 0.9]))?;
    let outer_surface = extract_external_surface(&grid)?;

    // 3. Create a headless wgpu context, 4x MSAA renderer, and 800x600 target.
    let ctx = HeadlessContext::new()?;
    let renderer = Renderer::new(ctx.device(), ctx.queue(), DEFAULT_COLOR_FORMAT, 4)?;
    let target = OffscreenTarget::new(ctx.device(), 800, 600, DEFAULT_COLOR_FORMAT, 4)?;

    // 4. Upload both the outer wireframe and the colored slice relative to `bounds.center()`.
    let slice_style = SurfaceStyle {
        color_by: ColorBy::point("linear"),
        colormap: ColormapKind::Viridis,
        range: Some([0.0, 12.0]),
        draw_surface: true,
        draw_edges: true,
        ..SurfaceStyle::default()
    };
    let wire_style = SurfaceStyle {
        draw_surface: false,
        draw_edges: true,
        edge_color_srgb: [0.75, 0.78, 0.82, 1.0],
        ..SurfaceStyle::default()
    };
    let gpu_slice = renderer.upload_surface_with_center(
        ctx.device(),
        &slice_surface,
        bounds.center(),
        &slice_style,
    )?;
    let gpu_wire = renderer.upload_surface_with_center(
        ctx.device(),
        &outer_surface,
        bounds.center(),
        &wire_style,
    )?;

    // 5. Render with a colorbar and orientation axes triad, then read back PNG bytes.
    let camera = Camera::fit_bounds(bounds, 35.0, 25.0);
    let annotations = Annotations {
        axes: true,
        colorbars: vec![
            ColorbarAnnotation::new(ColormapKind::Viridis, [0.0, 12.0]).with_title("linear"),
        ],
        ..Annotations::default()
    };

    let image = ctx.render_surfaces_to_image(
        &renderer,
        &target,
        &camera,
        [0.09, 0.10, 0.12, 1.0],
        &[&gpu_slice, &gpu_wire],
        Some(&annotations),
    )?;
    image.save_png("slice_out.png")?;
    Ok(())
}
```

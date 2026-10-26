# Embedding in Rust & egui

Embedding Tetraglass inside your own Rust engineering or simulation application is the library's top-priority workflow (`D-002`, `D-003`). You can integrate at two levels:

1. **`tetraglass-egui` (`ViewportState` + `ViewportWidget`):** Drop-in 3D viewport widget for `egui` / `eframe` applications using `egui-wgpu`.
2. **`tetraglass-render` (`Renderer` + `OffscreenTarget`):** Direct `wgpu` integration using your application's `wgpu::Device` and `wgpu::Queue`.

## 1. Embedding `ViewportWidget` in an `eframe` / `egui-wgpu` app

`tetraglass-egui` renders into its own offscreen color and depth attachments during `egui_wgpu::CallbackTrait::prepare` and displays the resolved 1x texture inside `egui` as a registered `egui::TextureId`. This keeps depth testing, 4x MSAA, GPU ID picking, and 2D/3D viewport annotations independent of `egui`'s own render pass (`D-003`).

```rust,no_run
use eframe::egui;
use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
use tetraglass_egui::{ViewportState, ViewportWidget, available_field_options};
use tetraglass_filters::extract_external_surface;
use tetraglass_render::{
    Annotations, ColorBy, ColorbarAnnotation, ColormapKind, ShaderClipKeep, ShaderClipPlane,
    SurfaceStyle,
};

struct MySimulationApp {
    viewport: ViewportState,
    style: SurfaceStyle,
    clip_enabled: bool,
    clip_x: f64,
}

impl MySimulationApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must be configured with the wgpu backend");

        let grid = SyntheticGridSpec::new(10, 10, 8, SyntheticCellLayout::Hexahedra)
            .generate()
            .expect("synthetic mesh");
        let surface = extract_external_surface(&grid).expect("external surface");

        let style = SurfaceStyle {
            color_by: ColorBy::point("linear"),
            colormap: ColormapKind::Viridis,
            range: Some([0.0, 6.0]),
            draw_surface: true,
            draw_edges: true,
            ..SurfaceStyle::default()
        };

        let mut viewport = ViewportState::new(&render_state.device, &render_state.queue)
            .expect("create viewport state");
        viewport
            .set_surface(&render_state.device, &surface, &style)
            .expect("upload initial surface");
        viewport.set_annotations(Annotations {
            title: Some("Simulation Viewport".to_owned()),
            axes: true,
            colorbars: vec![
                ColorbarAnnotation::new(ColormapKind::Viridis, [0.0, 6.0]).with_title("linear"),
            ],
        });

        Self {
            viewport,
            style,
            clip_enabled: false,
            clip_x: 0.5,
        }
    }
}

impl eframe::App for MySimulationApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        egui::SidePanel::left("controls").show_inside(ui, |ui| {
            if let Some(rs) = frame.wgpu_render_state() {
                if ui.checkbox(&mut self.clip_enabled, "Shader clip plane").changed()
                    || (self.clip_enabled
                        && ui.add(egui::Slider::new(&mut self.clip_x, 0.0..=1.0)).changed())
                {
                    let plane = self.clip_enabled.then_some(ShaderClipPlane {
                        origin: [self.clip_x, 0.5, 0.5],
                        normal: [1.0, 0.0, 0.0],
                        keep: ShaderClipKeep::Back,
                    });
                    self.viewport.set_clip_plane(&rs.queue, plane);
                }
            }
            if let Some(pick) = self.viewport.last_pick() {
                ui.separator();
                ui.label(format!("Picked cell: {}", pick.source_cell_id));
                ui.label(format!(
                    "World: [{:.3}, {:.3}, {:.3}]",
                    pick.world_position[0], pick.world_position[1], pick.world_position[2]
                ));
            }
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ViewportWidget::new(&mut self.viewport).show(ui);
        });
    }
}
```

### Multi-representation scenes & background worker swaps

- Use `ViewportState::set_shared_surface_preserving_camera(&device, Arc::new(surface), &style, false)` when swapping in a background worker's exact-clipped or re-filtered mesh so the user's interactive orbit/pan/zoom camera and dataset center stay untouched.
- Use `ViewportState::set_additional_surfaces(&device, &[(Arc::new(wireframe_surface), wire_style)])` to overlay additional scene representations (such as an outer element wireframe around a slice or contour).

## 2. Direct `tetraglass-render` integration

If your application manages its own `wgpu::Device`, `wgpu::Queue`, and window/offscreen textures without `egui`, use `tetraglass_render::Renderer` directly:

- `Renderer::new(&device, &queue, format, sample_count)` compiles the surface, wireframe edge, GPU ID pick, and 2D/3D annotation pipelines once.
- `OffscreenTarget::new(&device, width, height, format, sample_count)` allocates the MSAA color, resolve, and `Depth32Float` attachments.
- `renderer.upload_surface(&device, &surface_mesh, &style)` uploads vertex positions in `f32` relative to `surface_mesh.bounds().center()` (`D-014`), per-vertex scalars, and per-triangle cell scalars.
- `renderer.update_surface_style(&queue, &mut gpu_surface, &surface_mesh, &new_style)` updates colormap, range, log scale, or edge visibility via a uniform write with zero geometry re-upload.
- `renderer.set_surface_clip_plane(&queue, &mut gpu_surface, Some(plane))` updates only the 16-byte fragment clip-plane uniform (`<= 1 ms/frame`).
- `renderer.render_with_annotations(&device, &mut encoder, &queue, &target, &camera, bg_srgb, &[&gpu_surface], &annotations)` records the 3D and annotation draw calls into your command encoder.

## 3. Using the `tetraglass` umbrella crate and `SceneBuilder`

Instead of depending on individual sub-crates, Rust applications can depend on the `tetraglass` umbrella crate and build declarative [`Scene`] pipelines in code via `Scene::builder()`:

```rust,no_run
use tetraglass::prelude::*;

let mut builder = Scene::builder();
let flow = builder.source("flow", Source::file("results.pvd"));
let cut = builder.filter("cut", SliceFilter::new(&flow, [0.0, 0.0, 1.0]));
let iso = builder.filter("iso", ContourFilter::new(&flow, "pressure", [101_325.0]));

builder
    .show(&cut)
    .color_by("velocity")
    .magnitude()
    .colormap(ColormapName::Viridis)
    .colorbar(true);
builder.show(&iso).opacity(0.4);
builder.camera().fit("all").orbit(30.0, 20.0);
builder.output("out.png").size(1600, 900).time(TimeSelector::LAST);

let scene = builder.build_validated().expect("valid v1 scene");
```

//! `wgpu` renderer drawing into host-provided offscreen textures, plus a `headless` helper.
//!
//! # Architecture & invariants
//!
//! - [`Renderer`] accepts `&wgpu::Device`, `&wgpu::Queue`, and a target [`wgpu::TextureFormat`]
//!   from the host application and draws into caller-owned [`OffscreenTarget`]s. It never owns a
//!   window, surface, or event loop (`D-002`).
//! - Only [`headless::HeadlessContext`] creates its own `wgpu::Instance`, `Adapter`, `Device`,
//!   and `Queue` (for the CLI and headless tests).
//! - Vertex positions are uploaded in `f32` relative to the dataset center (`D-014`); the
//!   [`Camera`] operates in the same relative space.
//! - Point scalars upload once as an `f32` vertex attribute (`@location(3)`), cell scalars upload
//!   to a storage buffer indexed by triangle `cell_index`, and colormaps are 256-entry 1D sRGB
//!   textures (`Viridis`, `Cividis`, `CoolWarm`). Switching colormap, range, or log scale updates
//!   only a uniform buffer.
//!
//! # Examples
//!
//! Rendering a [`SurfaceMesh`](tetraglass_core::SurfaceMesh) headlessly and probing a pixel:
//!
//! ```no_run
//! use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
//! use tetraglass_filters::extract_external_surface;
//! use tetraglass_render::{
//!     Annotations, Camera, ColorBy, ColorbarAnnotation, ColormapKind, DEFAULT_COLOR_FORMAT,
//!     HeadlessContext, OffscreenTarget, PickTarget, Renderer, SurfaceStyle,
//! };
//!
//! let grid = SyntheticGridSpec::new(4, 4, 4, SyntheticCellLayout::Hexahedra).generate()?;
//! let surface = extract_external_surface(&grid)?;
//! let bounds = surface.bounds().expect("non-empty surface");
//!
//! let ctx = HeadlessContext::new()?;
//! let renderer = Renderer::new(ctx.device(), ctx.queue(), DEFAULT_COLOR_FORMAT, 4)?;
//! let target = OffscreenTarget::new(ctx.device(), 800, 600, DEFAULT_COLOR_FORMAT, 4)?;
//!
//! let style = SurfaceStyle {
//!     color_by: ColorBy::point("linear"),
//!     colormap: ColormapKind::Viridis,
//!     draw_edges: true,
//!     ..SurfaceStyle::default()
//! };
//! let gpu_surface = renderer.upload_surface(ctx.device(), &surface, &style)?;
//! let camera = Camera::fit_bounds(bounds, 35.0, 25.0);
//! let annotations = Annotations {
//!     axes: true,
//!     colorbars: vec![ColorbarAnnotation::new(ColormapKind::Viridis, [0.0, 6.0]).with_title("linear")],
//!     ..Annotations::default()
//! };
//!
//! let mut encoder = ctx.device().create_command_encoder(&Default::default());
//! renderer.render_with_annotations(
//!     ctx.device(),
//!     &mut encoder,
//!     ctx.queue(),
//!     &target,
//!     &camera,
//!     [0.09, 0.10, 0.12, 1.0],
//!     &[&gpu_surface],
//!     &annotations,
//! )?;
//! ctx.queue().submit(std::iter::once(encoder.finish()));
//!
//! // GPU ID buffer picking at the viewport center
//! let pick_target = PickTarget::new(ctx.device(), 800, 600)?;
//! if let Some(hit) = renderer.pick_pixel(
//!     ctx.device(),
//!     ctx.queue(),
//!     &pick_target,
//!     &camera,
//!     &[&gpu_surface],
//!     400,
//!     300,
//! )? {
//!     println!("Hit representation {} triangle {}", hit.representation_index, hit.triangle_index);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod annotations;
pub mod camera;
pub mod colormap;
pub mod error;
pub mod headless;
pub mod pick;
pub mod pipeline;
pub mod png;
pub mod text;

pub use annotations::{
    Annotations, ColorbarAnnotation, ColorbarPlacement, ColorbarTick, DEFAULT_COLORBAR_TICKS,
    compute_colorbar_ticks, format_tick_value,
};
pub use camera::Camera;
pub use colormap::ColormapKind;
pub use error::{FieldAssociation, RenderError};
pub use headless::{
    DEFAULT_CHANNEL_TOLERANCE, DEFAULT_MAX_DIFF_FRACTION, HeadlessContext, HeadlessRenderOptions,
    ImageComparison, RenderedImage,
};
pub use pick::{
    PICK_FORMAT, PickProbeValue, PickTarget, RawPickHit, SurfacePickResult, probe_surface_triangle,
};
pub use pipeline::{
    ColorBy, DEFAULT_COLOR_FORMAT, DEPTH_FORMAT, GpuSurface, OffscreenTarget, Renderer,
    ScalarComponent, ShaderClipKeep, ShaderClipPlane, SurfaceStyle,
};
pub use text::{EmbeddedFont, FontRasterizer, RasterizedText, TextMetrics};

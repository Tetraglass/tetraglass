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
    ScalarComponent, SurfaceStyle,
};
pub use text::{EmbeddedFont, FontRasterizer, RasterizedText, TextMetrics};

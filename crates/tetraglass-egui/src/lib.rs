//! Embeddable `egui` 3D viewport widget built on `tetraglass-render`.
//!
//! # Architecture & invariants
//!
//! - All `wgpu` pipelines, buffers, and shaders live in `tetraglass-render`.
//! - [`ViewportState`] and [`ViewportWidget`] bridge `egui` pointer/scroll input to
//!   [`tetraglass_render::Camera`], manage offscreen target resizing and texture registration
//!   with [`egui_wgpu::Renderer`], and schedule offscreen rendering during
//!   [`egui_wgpu::CallbackTrait::prepare`].
//! - Scene geometry is never drawn inside `egui`'s render pass (`CallbackTrait::paint` is a no-op);
//!   instead, the resolved offscreen target is displayed as a registered `egui::TextureId`.

use std::sync::{Arc, Mutex};

use egui::{
    Color32, CornerRadius, FontId, Pos2, Rect, Response, Sense, Stroke, TextureId, Ui, Vec2, pos2,
};
use egui_wgpu::wgpu;
use tetraglass_core::{BoundingBox, SurfaceMesh};
use tetraglass_render::{
    Annotations, Camera, ColorBy, DEFAULT_COLOR_FORMAT, GpuSurface, OffscreenTarget, PickTarget,
    RenderError, Renderer, ScalarComponent, ShaderClipPlane, SurfacePickResult, SurfaceStyle,
    probe_surface_triangle,
};

/// A selectable coloring option for a [`SurfaceMesh`] in a UI field picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldOption {
    /// Human-readable label (for example `"Point: linear"` or `"Point: velocity (magnitude)"`).
    pub label: String,
    /// Corresponding [`ColorBy`] configuration.
    pub color_by: ColorBy,
}

/// Enumerates all [`FieldOption`]s available on `surface` (`Solid`, point arrays, and cell arrays).
#[must_use]
pub fn available_field_options(surface: &SurfaceMesh) -> Vec<FieldOption> {
    let mut options = vec![FieldOption {
        label: "Solid color".to_string(),
        color_by: ColorBy::Solid,
    }];

    for (name, array) in surface.point_data().iter() {
        append_array_options(
            &mut options,
            "Point",
            name,
            array.components().as_usize(),
            true,
        );
    }
    for (name, array) in surface.cell_data().iter() {
        append_array_options(
            &mut options,
            "Cell",
            name,
            array.components().as_usize(),
            false,
        );
    }

    options
}

fn append_array_options(
    out: &mut Vec<FieldOption>,
    prefix: &str,
    name: &str,
    num_components: usize,
    is_point: bool,
) {
    let make_color_by = |component: ScalarComponent| {
        if is_point {
            ColorBy::Point {
                name: name.to_string(),
                component,
            }
        } else {
            ColorBy::Cell {
                name: name.to_string(),
                component,
            }
        }
    };

    if num_components == 1 {
        out.push(FieldOption {
            label: format!("{prefix}: {name}"),
            color_by: make_color_by(ScalarComponent::First),
        });
    } else {
        out.push(FieldOption {
            label: format!("{prefix}: {name} (magnitude)"),
            color_by: make_color_by(ScalarComponent::Magnitude),
        });
        for c in 0..num_components {
            let suffix = match (num_components, c) {
                (3, 0) => "X".to_string(),
                (3, 1) => "Y".to_string(),
                (3, 2) => "Z".to_string(),
                _ => c.to_string(),
            };
            out.push(FieldOption {
                label: format!("{prefix}: {name} ({suffix})"),
                color_by: make_color_by(ScalarComponent::Index(c)),
            });
        }
    }
}

struct ViewportGpuInner {
    renderer: Renderer,
    target: OffscreenTarget,
    pick_target: PickTarget,
    cpu_surface: Option<Arc<SurfaceMesh>>,
    gpu_surface: Option<GpuSurface>,
    extra_surfaces: Vec<(Arc<SurfaceMesh>, GpuSurface, SurfaceStyle)>,
    camera: Camera,
    default_bounds: Option<BoundingBox>,
    background_srgb: [f32; 4],
    style: SurfaceStyle,
    annotations: Annotations,
    pending_pick_px: Option<[u32; 2]>,
    last_pick: Option<SurfacePickResult>,
}

impl ViewportGpuInner {
    fn execute_pick(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        x_px: u32,
        y_px: u32,
    ) -> Result<Option<SurfacePickResult>, RenderError> {
        let mut gpu_surfaces: Vec<&GpuSurface> = Vec::with_capacity(1 + self.extra_surfaces.len());
        if let Some(primary) = &self.gpu_surface {
            gpu_surfaces.push(primary);
        }
        for (_, extra_gpu, _) in &self.extra_surfaces {
            gpu_surfaces.push(extra_gpu);
        }
        if gpu_surfaces.is_empty() {
            self.last_pick = None;
            return Ok(None);
        }

        let Some(raw) = self.renderer.pick_pixel(
            device,
            queue,
            &self.pick_target,
            &self.camera,
            &gpu_surfaces,
            x_px,
            y_px,
        )?
        else {
            self.last_pick = None;
            return Ok(None);
        };

        let viewport_size = [self.pick_target.width(), self.pick_target.height()];
        let probed = if raw.representation_index == 0 {
            if let (Some(cpu_surf), Some(gpu_surf)) = (&self.cpu_surface, &self.gpu_surface) {
                probe_surface_triangle(
                    cpu_surf,
                    0,
                    [x_px, y_px],
                    viewport_size,
                    &self.camera,
                    gpu_surf.dataset_center(),
                    raw,
                    &self.style.color_by,
                )
            } else {
                None
            }
        } else {
            let extra_idx = raw.representation_index - 1;
            self.extra_surfaces
                .get(extra_idx)
                .and_then(|(cpu_surf, gpu_surf, style)| {
                    probe_surface_triangle(
                        cpu_surf,
                        raw.representation_index,
                        [x_px, y_px],
                        viewport_size,
                        &self.camera,
                        gpu_surf.dataset_center(),
                        raw,
                        &style.color_by,
                    )
                })
        };

        self.last_pick = probed.clone();
        Ok(probed)
    }
}

/// Persistent GPU and camera state for an embeddable 3D [`ViewportWidget`].
pub struct ViewportState {
    inner: Arc<Mutex<ViewportGpuInner>>,
    texture_id: TextureId,
}

impl std::fmt::Debug for ViewportState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViewportState")
            .field("texture_id", &self.texture_id)
            .finish_non_exhaustive()
    }
}

impl ViewportState {
    /// Creates a new [`ViewportState`] using the host's `device`, `queue`, and `egui_renderer`.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        egui_renderer: &mut egui_wgpu::Renderer,
        sample_count: u32,
    ) -> Result<Self, RenderError> {
        let initial_width = 64;
        let initial_height = 64;
        let renderer = Renderer::new(device, queue, DEFAULT_COLOR_FORMAT, sample_count)?;
        let target = OffscreenTarget::new(
            device,
            initial_width,
            initial_height,
            DEFAULT_COLOR_FORMAT,
            sample_count,
        )?;
        let pick_target = PickTarget::new(device, initial_width, initial_height)?;
        let texture_id = egui_renderer.register_native_texture(
            device,
            target.egui_texture_view(),
            wgpu::FilterMode::Linear,
        );

        let inner = ViewportGpuInner {
            renderer,
            target,
            pick_target,
            cpu_surface: None,
            gpu_surface: None,
            extra_surfaces: Vec::new(),
            camera: Camera::default(),
            default_bounds: None,
            background_srgb: [1.0, 1.0, 1.0, 1.0],
            style: SurfaceStyle::default(),
            annotations: Annotations::default(),
            pending_pick_px: None,
            last_pick: None,
        };

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            texture_id,
        })
    }

    /// Creates a new [`ViewportState`] from an [`egui_wgpu::RenderState`] (such as `eframe::CreationContext::wgpu_render_state`).
    pub fn from_render_state(
        render_state: &egui_wgpu::RenderState,
        sample_count: u32,
    ) -> Result<Self, RenderError> {
        let mut egui_renderer = render_state.renderer.write();
        Self::new(
            &render_state.device,
            &render_state.queue,
            &mut egui_renderer,
            sample_count,
        )
    }

    /// Uploads a [`SurfaceMesh`] to the viewport and fits the camera if `fit_camera` is `true`.
    pub fn set_surface(
        &mut self,
        device: &wgpu::Device,
        surface: &SurfaceMesh,
        style: SurfaceStyle,
        fit_camera: bool,
    ) -> Result<(), RenderError> {
        self.set_shared_surface(device, Arc::new(surface.clone()), style, fit_camera)
    }

    /// Uploads a shared [`Arc<SurfaceMesh>`] to the viewport and fits the camera if `fit_camera` is `true`.
    pub fn set_shared_surface(
        &mut self,
        device: &wgpu::Device,
        surface: Arc<SurfaceMesh>,
        style: SurfaceStyle,
        fit_camera: bool,
    ) -> Result<(), RenderError> {
        if let Ok(mut guard) = self.inner.lock() {
            let gpu_surface = if !fit_camera && let Some(existing) = guard.gpu_surface.as_ref() {
                guard.renderer.upload_surface_with_center(
                    device,
                    &surface,
                    existing.dataset_center(),
                    &style,
                )?
            } else {
                guard.renderer.upload_surface(device, &surface, &style)?
            };
            let bounds = surface.bounds();
            if fit_camera {
                guard.default_bounds = bounds;
                if let Some(b) = bounds {
                    guard.camera = Camera::fit_bounds(b, 30.0, 20.0);
                }
            } else if guard.default_bounds.is_none() {
                guard.default_bounds = bounds;
            }
            guard.style = style;
            guard.cpu_surface = Some(surface);
            guard.gpu_surface = Some(gpu_surface);
            guard.extra_surfaces.clear();
            guard.last_pick = None;
            guard.pending_pick_px = None;
        }
        Ok(())
    }

    /// Uploads additional secondary [`SurfaceMesh`] representations (aligned to the primary surface's
    /// dataset center) for multi-representation scenes.
    pub fn set_additional_surfaces(
        &mut self,
        device: &wgpu::Device,
        surfaces: Vec<(Arc<SurfaceMesh>, SurfaceStyle)>,
    ) -> Result<(), RenderError> {
        if let Ok(mut guard) = self.inner.lock() {
            let center = guard
                .gpu_surface
                .as_ref()
                .map_or([0.0, 0.0, 0.0], GpuSurface::dataset_center);
            let mut uploaded = Vec::with_capacity(surfaces.len());
            for (surf, style) in surfaces {
                let gpu_surf = guard
                    .renderer
                    .upload_surface_with_center(device, &surf, center, &style)?;
                uploaded.push((surf, gpu_surf, style));
            }
            guard.extra_surfaces = uploaded;
        }
        Ok(())
    }

    /// Uploads a [`SurfaceMesh`] to the viewport using an [`egui_wgpu::RenderState`].
    pub fn set_surface_with_render_state(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        surface: &SurfaceMesh,
        style: SurfaceStyle,
        fit_camera: bool,
    ) -> Result<(), RenderError> {
        self.set_surface(&render_state.device, surface, style, fit_camera)
    }

    /// Updates the [`SurfaceStyle`] on the active surface using an [`egui_wgpu::RenderState`].
    pub fn update_style_with_render_state(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        surface: &SurfaceMesh,
        style: SurfaceStyle,
    ) -> Result<(), RenderError> {
        self.update_style(&render_state.queue, surface, style)
    }

    /// Updates the [`SurfaceStyle`] (field coloring, colormap, range, or log scale) on the active surface.
    pub fn update_style(
        &mut self,
        queue: &wgpu::Queue,
        surface: &SurfaceMesh,
        style: SurfaceStyle,
    ) -> Result<(), RenderError> {
        if let Ok(mut guard) = self.inner.lock() {
            let ViewportGpuInner {
                renderer,
                gpu_surface,
                style: current_style,
                ..
            } = &mut *guard;
            if let Some(gpu_surf) = gpu_surface.as_mut() {
                renderer.update_surface_style(queue, gpu_surf, surface, &style)?;
            }
            *current_style = style;
        }
        Ok(())
    }

    /// Returns a clone of the current [`SurfaceStyle`].
    #[must_use]
    pub fn style(&self) -> SurfaceStyle {
        self.inner
            .lock()
            .map(|g| g.style.clone())
            .unwrap_or_default()
    }

    /// Returns the active scalar range `[min, max]` mapped across the colormap.
    #[must_use]
    pub fn effective_range(&self) -> Option<[f64; 2]> {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.gpu_surface.as_ref().map(GpuSurface::effective_range))
    }

    /// Returns a clone of the current [`Camera`].
    #[must_use]
    pub fn camera(&self) -> Camera {
        self.inner
            .lock()
            .map(|g| g.camera.clone())
            .unwrap_or_default()
    }

    /// Sets the [`Camera`] directly.
    pub fn set_camera(&mut self, camera: Camera) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.camera = camera;
        }
    }

    /// Resets the camera to frame the active surface's bounding box at `azimuth = 30°, elevation = 20°`.
    pub fn reset_camera(&mut self) {
        if let Ok(mut guard) = self.inner.lock()
            && let Some(bounds) = guard.default_bounds
        {
            guard.camera = Camera::fit_bounds(bounds, 30.0, 20.0);
        }
    }

    /// Sets the viewport background color in sRGB `[r, g, b, a]` (`0.0..=1.0`).
    pub fn set_background_srgb(&mut self, background_srgb: [f32; 4]) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.background_srgb = background_srgb;
        }
    }

    /// Returns a clone of the current viewport [`Annotations`].
    #[must_use]
    pub fn annotations(&self) -> Annotations {
        self.inner
            .lock()
            .map(|g| g.annotations.clone())
            .unwrap_or_default()
    }

    /// Sets the viewport [`Annotations`] (title, axes triad, and colorbars) drawn by the renderer.
    pub fn set_annotations(&mut self, annotations: Annotations) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.annotations = annotations;
        }
    }

    /// Updates only the 16-byte GPU shader clip-plane uniform on the uploaded surface(s) (`M2.8`, `D-019`).
    ///
    /// Pass `Some(plane)` while interactively dragging a clip plane for instant 60 FPS feedback,
    /// and `None` once the exact clipped [`SurfaceMesh`] has been swapped in.
    pub fn set_clip_plane(&mut self, queue: &wgpu::Queue, clip_plane: Option<ShaderClipPlane>) {
        if let Ok(mut guard) = self.inner.lock() {
            let guard = &mut *guard;
            if let Some(ref mut gpu) = guard.gpu_surface {
                guard
                    .renderer
                    .set_surface_clip_plane(queue, gpu, clip_plane);
            }
            for (_, gpu, _) in &mut guard.extra_surfaces {
                guard
                    .renderer
                    .set_surface_clip_plane(queue, gpu, clip_plane);
            }
        }
    }

    /// Updates the GPU shader clip-plane uniform using an [`egui_wgpu::RenderState`].
    pub fn set_clip_plane_with_render_state(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        clip_plane: Option<ShaderClipPlane>,
    ) {
        self.set_clip_plane(&render_state.queue, clip_plane);
    }

    /// Returns the active [`ShaderClipPlane`] on the primary uploaded surface, if any.
    #[must_use]
    pub fn clip_plane(&self) -> Option<ShaderClipPlane> {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.gpu_surface.as_ref().and_then(GpuSurface::clip_plane))
    }

    /// Returns the registered `egui` [`TextureId`] for the resolved offscreen target.
    #[must_use]
    pub const fn texture_id(&self) -> TextureId {
        self.texture_id
    }

    /// Returns the current offscreen target dimensions `(width_px, height_px)` in physical pixels.
    #[must_use]
    pub fn target_size_px(&self) -> (u32, u32) {
        self.inner
            .lock()
            .map(|g| (g.target.width(), g.target.height()))
            .unwrap_or((0, 0))
    }

    /// Ensures the offscreen target matches `(width_px, height_px)` and updates the `egui` texture binding if resized.
    pub fn ensure_target_size(
        &mut self,
        device: &wgpu::Device,
        egui_renderer: &mut egui_wgpu::Renderer,
        width_px: u32,
        height_px: u32,
    ) -> Result<bool, RenderError> {
        let width_px = width_px.max(1);
        let height_px = height_px.max(1);
        if let Ok(mut guard) = self.inner.lock() {
            let resized = guard.target.resize(device, width_px, height_px)?;
            let _ = guard.pick_target.resize(device, width_px, height_px)?;
            if resized {
                egui_renderer.update_egui_texture_from_wgpu_texture(
                    device,
                    guard.target.egui_texture_view(),
                    wgpu::FilterMode::Linear,
                    self.texture_id,
                );
            }
            return Ok(resized);
        }
        Ok(false)
    }

    /// Synchronously renders the GPU ID buffer and probes the surface at physical pixel `(x_px, y_px)`
    /// (`(0, 0)` at top-left of the viewport), updating [`Self::last_pick`].
    pub fn pick_at(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        x_px: u32,
        y_px: u32,
    ) -> Result<Option<SurfacePickResult>, RenderError> {
        if let Ok(mut guard) = self.inner.lock() {
            guard.pending_pick_px = None;
            return guard.execute_pick(device, queue, x_px, y_px);
        }
        Ok(None)
    }

    /// Synchronously probes the surface at physical pixel `(x_px, y_px)` using an [`egui_wgpu::RenderState`].
    pub fn pick_at_with_render_state(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        x_px: u32,
        y_px: u32,
    ) -> Result<Option<SurfacePickResult>, RenderError> {
        self.pick_at(&render_state.device, &render_state.queue, x_px, y_px)
    }

    /// Schedules a pick at physical pixel `(x_px, y_px)` to execute during the next `egui_wgpu` `prepare` step.
    pub fn request_pick_at(&mut self, x_px: u32, y_px: u32) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.pending_pick_px = Some([x_px, y_px]);
        }
    }

    /// Returns the most recent [`SurfacePickResult`] from clicking or calling [`Self::pick_at`].
    #[must_use]
    pub fn last_pick(&self) -> Option<SurfacePickResult> {
        self.inner.lock().ok().and_then(|g| g.last_pick.clone())
    }

    /// Clears the active value-probe result.
    pub fn clear_pick(&mut self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.last_pick = None;
            guard.pending_pick_px = None;
        }
    }

    /// Returns an `egui_wgpu` [`CallbackTrait`](egui_wgpu::CallbackTrait) instance that renders
    /// the current scene into the offscreen target during `prepare`.
    #[must_use]
    pub fn prepare_callback(&self) -> impl egui_wgpu::CallbackTrait + 'static {
        ViewportPrepareCallback {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Frees the registered texture from `egui_renderer`.
    pub fn destroy(&self, egui_renderer: &mut egui_wgpu::Renderer) {
        egui_renderer.free_texture(&self.texture_id);
    }

    /// Frees the registered texture using an [`egui_wgpu::RenderState`].
    pub fn destroy_with_render_state(&self, render_state: &egui_wgpu::RenderState) {
        let mut renderer = render_state.renderer.write();
        self.destroy(&mut renderer);
    }
}

struct ViewportPrepareCallback {
    inner: Arc<Mutex<ViewportGpuInner>>,
}

impl egui_wgpu::CallbackTrait for ViewportPrepareCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some([x_px, y_px]) = guard.pending_pick_px.take()
                && let Err(err) = guard.execute_pick(device, queue, x_px, y_px)
            {
                log::error!("tetraglass-egui pick failed: {err}");
            }
            let surfaces: Vec<&GpuSurface> = guard
                .gpu_surface
                .iter()
                .chain(guard.extra_surfaces.iter().map(|(_, g, _)| g))
                .collect();
            if let Err(err) = guard.renderer.render_with_annotations(
                device,
                egui_encoder,
                queue,
                &guard.target,
                &guard.camera,
                guard.background_srgb,
                &surfaces,
                &guard.annotations,
            ) {
                log::error!("tetraglass-egui offscreen prepare failed: {err}");
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        _render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources,
    ) {
        // Offscreen rendering completed in `prepare`; the resolved texture is drawn by egui as an image quad.
    }
}

/// Output from showing a [`ViewportWidget`] in an `egui::Ui`.
#[derive(Debug)]
pub struct ViewportResponse {
    /// The underlying `egui::Response` for the viewport rectangle.
    pub response: Response,
    /// Whether the camera was modified by orbit, pan, zoom, or reset input this frame.
    pub camera_changed: bool,
    /// Whether the offscreen target was resized this frame.
    pub resized: bool,
    /// The active surface pick result (updated when the user clicks in the viewport or calls [`ViewportState::pick_at`]).
    pub picked: Option<SurfacePickResult>,
}

/// Interactive 3D viewport widget that renders in `egui_wgpu`'s `prepare` step and displays the
/// resolved offscreen target as a texture.
pub struct ViewportWidget<'a> {
    state: &'a mut ViewportState,
    desired_size: Option<Vec2>,
    show_probe_overlay: bool,
}

impl<'a> ViewportWidget<'a> {
    /// Wraps a mutable [`ViewportState`] for display in an `egui::Ui`.
    #[must_use]
    pub fn new(state: &'a mut ViewportState) -> Self {
        Self {
            state,
            desired_size: None,
            show_probe_overlay: true,
        }
    }

    /// Sets an explicit logical point size for the viewport (defaults to `ui.available_size()`).
    #[must_use]
    pub fn desired_size(mut self, size: Vec2) -> Self {
        self.desired_size = Some(size);
        self
    }

    /// Controls whether an on-canvas value-probe overlay badge and crosshair are drawn when a pick is active.
    #[must_use]
    pub fn show_probe_overlay(mut self, show: bool) -> Self {
        self.show_probe_overlay = show;
        self
    }

    /// Shows the viewport widget using an [`egui_wgpu::RenderState`].
    pub fn show_with_render_state(
        self,
        ui: &mut Ui,
        render_state: &egui_wgpu::RenderState,
    ) -> Result<ViewportResponse, RenderError> {
        let mut egui_renderer = render_state.renderer.write();
        self.show(ui, &render_state.device, &mut egui_renderer)
    }

    /// Shows the viewport widget using explicit `&wgpu::Device` and `&mut egui_wgpu::Renderer`.
    pub fn show(
        self,
        ui: &mut Ui,
        device: &wgpu::Device,
        egui_renderer: &mut egui_wgpu::Renderer,
    ) -> Result<ViewportResponse, RenderError> {
        let size = self
            .desired_size
            .unwrap_or_else(|| ui.available_size().max(Vec2::splat(16.0)));
        let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());

        let ppp = ui.ctx().pixels_per_point().max(0.5);
        let width_px = (rect.width() * ppp).round().max(1.0) as u32;
        let height_px = (rect.height() * ppp).round().max(1.0) as u32;

        let resized = self
            .state
            .ensure_target_size(device, egui_renderer, width_px, height_px)?;

        let camera_changed = handle_camera_input(ui, &response, rect, self.state);

        if response.clicked()
            && let Some(pos) = response.interact_pointer_pos()
        {
            let rel_x = ((pos.x - rect.min.x) * ppp)
                .round()
                .clamp(0.0, (width_px.saturating_sub(1)) as f32) as u32;
            let rel_y = ((pos.y - rect.min.y) * ppp)
                .round()
                .clamp(0.0, (height_px.saturating_sub(1)) as f32) as u32;
            self.state.request_pick_at(rel_x, rel_y);
            ui.ctx().request_repaint();
        }

        // 1. Schedule offscreen render pass in `egui_wgpu::CallbackTrait::prepare`.
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            self.state.prepare_callback(),
        ));

        // 2. Draw the resolved offscreen target as a registered egui texture.
        let uv = Rect::from_min_max(pos2(0.0, 0.0), Pos2::new(1.0, 1.0));
        ui.painter()
            .image(self.state.texture_id(), rect, uv, Color32::WHITE);

        let picked = self.state.last_pick();
        if self.show_probe_overlay
            && let Some(ref pick) = picked
        {
            draw_probe_overlay(ui, rect, ppp, pick);
        }

        Ok(ViewportResponse {
            response,
            camera_changed,
            resized,
            picked,
        })
    }
}

fn draw_probe_overlay(ui: &Ui, rect: Rect, ppp: f32, pick: &SurfacePickResult) {
    let painter = ui.painter_at(rect);
    let px_pos = pos2(
        rect.min.x + (pick.pixel[0] as f32 + 0.5) / ppp,
        rect.min.y + (pick.pixel[1] as f32 + 0.5) / ppp,
    );
    if rect.contains(px_pos) {
        let stroke = Stroke::new(1.5, Color32::from_rgb(255, 220, 60));
        painter.circle_stroke(px_pos, 5.0, stroke);
        painter.line_segment(
            [
                pos2(px_pos.x - 8.0, px_pos.y),
                pos2(px_pos.x + 8.0, px_pos.y),
            ],
            stroke,
        );
        painter.line_segment(
            [
                pos2(px_pos.x, px_pos.y - 8.0),
                pos2(px_pos.x, px_pos.y + 8.0),
            ],
            stroke,
        );
    }

    let [wx, wy, wz] = pick.world_position;
    let cell_label = pick
        .source_cell_id
        .map_or_else(|| "n/a".to_string(), |c| format!("#{c}"));
    let mut text = format!(
        "Cell {cell_label} (tri #{}, pt #{})\nx={wx:.4}, y={wy:.4}, z={wz:.4}",
        pick.triangle_index, pick.point_index
    );
    if let Some(val) = pick.active_scalar
        && val.is_finite()
    {
        text.push_str(&format!("\nvalue: {val:.5}"));
    }

    let badge_pos = pos2(rect.min.x + 8.0, rect.min.y + 8.0);
    let galley = painter.layout_no_wrap(text, FontId::monospace(11.0), Color32::from_gray(240));
    let badge_rect = Rect::from_min_size(badge_pos, galley.size() + Vec2::new(12.0, 8.0));
    painter.rect_filled(
        badge_rect,
        CornerRadius::same(4),
        Color32::from_rgba_unmultiplied(20, 24, 30, 210),
    );
    painter.rect_stroke(
        badge_rect,
        CornerRadius::same(4),
        Stroke::new(1.0, Color32::from_gray(90)),
        egui::StrokeKind::Inside,
    );
    painter.galley(
        pos2(badge_pos.x + 6.0, badge_pos.y + 4.0),
        galley,
        Color32::from_gray(240),
    );
}

fn handle_camera_input(
    ui: &Ui,
    response: &Response,
    rect: Rect,
    state: &mut ViewportState,
) -> bool {
    let mut changed = false;

    if response.double_clicked() {
        state.reset_camera();
        return true;
    }

    let drag_delta = response.drag_delta();
    if drag_delta != Vec2::ZERO {
        let shift_held = ui.input(|i| i.modifiers.shift);
        if response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
            || (response.dragged_by(egui::PointerButton::Primary) && shift_held)
        {
            if let Ok(mut guard) = state.inner.lock() {
                let dx = f64::from(drag_delta.x / rect.width().max(1.0));
                let dy = f64::from(drag_delta.y / rect.height().max(1.0));
                guard.camera.pan(dx, dy);
                changed = true;
            }
        } else if response.dragged_by(egui::PointerButton::Primary)
            && let Ok(mut guard) = state.inner.lock()
        {
            let d_az = -f64::from(drag_delta.x) * 0.45;
            let d_el = f64::from(drag_delta.y) * 0.45;
            guard.camera.orbit(d_az, d_el);
            changed = true;
        }
    }

    if response.hovered() {
        let (scroll_y, zoom_delta) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        if scroll_y.abs() > 0.0 || (zoom_delta - 1.0).abs() > 1e-4 {
            let factor = f64::from((scroll_y * 0.0025).exp() * zoom_delta);
            if let Ok(mut guard) = state.inner.lock() {
                guard.camera.zoom_by(factor);
                changed = true;
            }
        }
    }

    changed
}

/// Re-exported colormap, style, clip-plane, and pick types for convenience when configuring [`ViewportState`].
pub use tetraglass_render::{
    Camera as ViewportCamera, ColorBy as ViewportColorBy, ColormapKind as ViewportColormap,
    PickProbeValue as ViewportPickProbeValue, ScalarComponent as ViewportScalarComponent,
    ShaderClipKeep as ViewportClipKeep, ShaderClipPlane as ViewportClipPlane,
    SurfacePickResult as ViewportPickResult, SurfaceStyle as ViewportSurfaceStyle,
};

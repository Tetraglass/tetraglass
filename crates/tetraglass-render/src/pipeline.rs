//! `wgpu` surface rendering pipeline, offscreen targets, and GPU surface buffers.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use tetraglass_core::{BoundingBox, DataArray, RegularGrid, SurfaceMesh, ValueRange};
use wgpu::util::DeviceExt;

use crate::annotations::{
    AnnotationUniforms, AnnotationVertex, Annotations, build_annotation_batch,
};
use crate::camera::Camera;
use crate::colormap::ColormapKind;
use crate::error::{FieldAssociation, RenderError};
use crate::pick::{PICK_FORMAT, PendingPixelPick, PickTarget, RawPickHit};

/// Default offscreen color texture format (sRGB 8-bit RGBA).
pub const DEFAULT_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Depth buffer texture format used by offscreen targets.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Color accumulation texture format for Weighted Blended Order-Independent Transparency (`M3.3`).
pub const WBOIT_ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Transmittance revealage texture format for Weighted Blended Order-Independent Transparency (`M3.3`).
pub const WBOIT_REVEAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Filterable 3D texture format used for GPU volume ray marching (`M4.4b`).
pub const VOLUME_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Intermediate 1x raw ambient-occlusion texture format (`M5.2`).
pub const SSAO_RAW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Number of samples in the pre-evaluated 1D opacity transfer function table (`M4.4b`).
pub const DEFAULT_TRANSFER_FUNCTION_LUT_SIZE: usize = 256;

/// Evaluates a piecewise-linear opacity transfer function `[[t, alpha], ...]` (`t, alpha in 0.0..=1.0`)
/// into a 256-entry `f32` lookup table (`M4.4b`).
///
/// When `control_points` is `None` or has fewer than 2 control points, evaluates the default
/// linear opacity ramp `[[0.0, 0.0], [1.0, 1.0]]`.
#[must_use]
pub fn evaluate_transfer_function_lut(
    control_points: Option<&[[f64; 2]]>,
) -> [f32; DEFAULT_TRANSFER_FUNCTION_LUT_SIZE] {
    let default_pts = [[0.0, 0.0], [1.0, 1.0]];
    let raw_pts = match control_points {
        Some(pts) if pts.len() >= 2 => pts,
        _ => &default_pts,
    };

    let mut sorted: Vec<[f64; 2]> = raw_pts
        .iter()
        .map(|&[t, a]| [t.clamp(0.0, 1.0), a.clamp(0.0, 1.0)])
        .collect();
    sorted.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

    let mut lut = [0.0f32; DEFAULT_TRANSFER_FUNCTION_LUT_SIZE];
    for (i, slot) in lut.iter_mut().enumerate() {
        let t = (i as f64) / ((DEFAULT_TRANSFER_FUNCTION_LUT_SIZE - 1) as f64);
        if t <= sorted[0][0] {
            *slot = sorted[0][1] as f32;
            continue;
        }
        let last = sorted[sorted.len() - 1];
        if t >= last[0] {
            *slot = last[1] as f32;
            continue;
        }
        let mut alpha = last[1];
        for pair in sorted.windows(2) {
            let [t0, a0] = pair[0];
            let [t1, a1] = pair[1];
            if t >= t0 && t <= t1 {
                let span = t1 - t0;
                if span > 1e-12 {
                    let frac = (t - t0) / span;
                    alpha = a0 + frac * (a1 - a0);
                } else {
                    alpha = a1;
                }
                break;
            }
        }
        *slot = alpha.clamp(0.0, 1.0) as f32;
    }
    lut
}

/// Which component of a [`DataArray`] to map through the colormap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ScalarComponent {
    /// Zero-based component index (`0` for scalars or `X` of vectors).
    #[default]
    First,
    /// Explicit zero-based component index (`0..num_components`).
    Index(usize),
    /// Euclidean (`L2`) magnitude across all components of a tuple.
    Magnitude,
}

impl ScalarComponent {
    /// Resolves the component index if not [`ScalarComponent::Magnitude`].
    #[must_use]
    pub const fn index(self) -> Option<usize> {
        match self {
            Self::First => Some(0),
            Self::Index(i) => Some(i),
            Self::Magnitude => None,
        }
    }
}

/// How a [`SurfaceMesh`] is colored.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum ColorBy {
    /// Uniform solid color (`SurfaceStyle::solid_color_srgb`).
    #[default]
    Solid,
    /// Interpolated per-vertex (`point_data`) scalar or vector component.
    Point {
        /// Name of the point attribute array.
        name: String,
        /// Component or magnitude to visualize.
        component: ScalarComponent,
    },
    /// Flat per-triangle (`cell_data`) scalar or vector component read from a GPU storage buffer.
    Cell {
        /// Name of the cell attribute array.
        name: String,
        /// Component or magnitude to visualize.
        component: ScalarComponent,
    },
}

impl ColorBy {
    /// Selects a point scalar array (component 0) by name.
    #[must_use]
    pub fn point(name: impl Into<String>) -> Self {
        Self::Point {
            name: name.into(),
            component: ScalarComponent::First,
        }
    }

    /// Selects a cell scalar array (component 0) by name.
    #[must_use]
    pub fn cell(name: impl Into<String>) -> Self {
        Self::Cell {
            name: name.into(),
            component: ScalarComponent::First,
        }
    }

    /// Resolves `name` against `surface`, preferring `point_data` when present in both.
    pub fn resolve_on_surface(
        surface: &SurfaceMesh,
        name: impl Into<String>,
        component: ScalarComponent,
    ) -> Result<Self, RenderError> {
        let name = name.into();
        if surface.point_data().get(&name).is_some() {
            Ok(Self::Point { name, component })
        } else if surface.cell_data().get(&name).is_some() {
            Ok(Self::Cell { name, component })
        } else {
            Err(RenderError::MissingArray {
                name,
                association: FieldAssociation::Point,
            })
        }
    }
}

/// Which half-space of a [`ShaderClipPlane`] to keep when rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ShaderClipKeep {
    /// Keep fragments in the negative half-space (`dot(p - origin, normal) <= 0`), matching `ClipKeep::Back`.
    #[default]
    Back,
    /// Keep fragments in the positive half-space (`dot(p - origin, normal) >= 0`), matching `ClipKeep::Front`.
    Front,
}

/// World-space half-space clip plane evaluated per fragment in the surface, pick, and wireframe shaders (`M2.8`).
///
/// Used during interactive clip-plane dragging so moving the plane updates only a 16-byte GPU uniform
/// without re-running CPU filters or re-uploading vertex buffers (`D-019`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShaderClipPlane {
    /// Point `[x, y, z]` (`f64`) on the clip plane in world coordinates.
    pub origin: [f64; 3],
    /// Outward plane normal `[nx, ny, nz]` (`f64`).
    pub normal: [f64; 3],
    /// Which side of the plane to keep (`Back` keeps `dot(p - origin, normal) <= 0`).
    pub keep: ShaderClipKeep,
}

impl ShaderClipPlane {
    /// Creates a [`ShaderClipPlane`] keeping [`ShaderClipKeep::Back`] (`dot(p - origin, normal) <= 0`).
    #[must_use]
    pub const fn new(origin: [f64; 3], normal: [f64; 3]) -> Self {
        Self {
            origin,
            normal,
            keep: ShaderClipKeep::Back,
        }
    }

    /// Creates a [`ShaderClipPlane`] with an explicit [`ShaderClipKeep`] side.
    #[must_use]
    pub const fn with_keep(origin: [f64; 3], normal: [f64; 3], keep: ShaderClipKeep) -> Self {
        Self {
            origin,
            normal,
            keep,
        }
    }

    /// Converts this world-space (`f64`) plane into a dataset-center-relative `[nx, ny, nz, d]` (`f32`)
    /// uniform where kept fragments satisfy `dot(v_rel, n_keep) + d >= 0`.
    #[must_use]
    pub fn to_gpu_plane(self, dataset_center: [f64; 3]) -> [f32; 4] {
        let len = (self.normal[0] * self.normal[0]
            + self.normal[1] * self.normal[1]
            + self.normal[2] * self.normal[2])
            .sqrt();
        if !len.is_finite() || len <= 1e-15 {
            return [0.0, 0.0, 0.0, 1.0];
        }
        let sign = match self.keep {
            ShaderClipKeep::Back => -1.0,
            ShaderClipKeep::Front => 1.0,
        };
        let nx = sign * self.normal[0] / len;
        let ny = sign * self.normal[1] / len;
        let nz = sign * self.normal[2] / len;
        let d = (dataset_center[0] - self.origin[0]) * nx
            + (dataset_center[1] - self.origin[1]) * ny
            + (dataset_center[2] - self.origin[2]) * nz;
        [nx as f32, ny as f32, nz as f32, d as f32]
    }
}

/// Display style and colormap parameters for a [`SurfaceMesh`].
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceStyle {
    /// Field coloring mode (solid, point array, or cell array).
    pub color_by: ColorBy,
    /// Colormap lookup table (`Viridis`, `Cividis`, `CoolWarm`).
    pub colormap: ColormapKind,
    /// Explicit scalar range `[min, max]`, or `None` to use the array's cached range.
    pub range: Option<[f64; 2]>,
    /// Whether to apply logarithmic (`log10`) scaling prior to colormap lookup.
    pub use_log_scale: bool,
    /// Fallback / solid color in sRGB `[r, g, b, a]` (`0.0..=1.0`).
    pub solid_color_srgb: [f32; 4],
    /// Whether to draw the shaded triangle faces (`true` for `Surface` and `SurfaceWithEdges`).
    pub draw_surface: bool,
    /// Whether to draw original element wireframe edges (`true` for `SurfaceWithEdges` and `Wireframe`).
    pub draw_edges: bool,
    /// Wireframe edge color in sRGB `[r, g, b, a]` (`0.0..=1.0`).
    pub edge_color_srgb: [f32; 4],
    /// Optional interactive shader clip plane (`None` disables shader clipping).
    pub clip_plane: Option<ShaderClipPlane>,
    /// Optional resampled 3D [`RegularGrid`] for `"volume"` ray-marching representations (`M4.4b`).
    pub volume_grid: Option<Arc<RegularGrid>>,
    /// Optional piecewise-linear opacity transfer function `[[t, alpha], ...]` (`0.0..=1.0`) for `"volume"` representations (`M4.4b`).
    pub transfer_function: Option<Vec<[f64; 2]>>,
}

impl Default for SurfaceStyle {
    fn default() -> Self {
        Self {
            color_by: ColorBy::Solid,
            colormap: ColormapKind::Viridis,
            range: None,
            use_log_scale: false,
            solid_color_srgb: [0.82, 0.83, 0.86, 1.0],
            draw_surface: true,
            draw_edges: false,
            edge_color_srgb: [0.12, 0.14, 0.18, 1.0],
            clip_plane: None,
            volume_grid: None,
            transfer_function: None,
        }
    }
}

/// Screen-space ambient occlusion (SSAO) configuration (`M5.2`).
///
/// Disabled by default (`enabled = false`) so existing gallery scenes and baselines are unchanged
/// unless SSAO is explicitly enabled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsaoConfig {
    /// Whether SSAO post-processing is enabled (`false` by default).
    pub enabled: bool,
    /// Occlusion hemisphere radius as a fraction of the camera's `scene_radius` (`0.15` by default).
    pub radius: f32,
    /// Occlusion darkening strength in `[0.0, 2.0]` (`0.8` by default).
    pub strength: f32,
    /// Depth bias as a fraction of the occlusion radius to prevent self-shadowing (`0.05` by default).
    pub bias: f32,
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            radius: 0.15,
            strength: 0.8,
            bias: 0.05,
        }
    }
}

impl SsaoConfig {
    /// Returns an enabled [`SsaoConfig`] with default `radius = 0.15`, `strength = 0.8`, and `bias = 0.05`.
    #[must_use]
    pub const fn enabled() -> Self {
        Self {
            enabled: true,
            radius: 0.15,
            strength: 0.8,
            bias: 0.05,
        }
    }

    /// Sets the normalized hemisphere radius (`0.005..=2.0`, relative to `scene_radius`).
    #[must_use]
    pub const fn with_radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }

    /// Sets the occlusion darkening strength (`0.0..=2.0`).
    #[must_use]
    pub const fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }

    /// Sets the depth bias fraction (`0.0..=0.5`, relative to the occlusion radius).
    #[must_use]
    pub const fn with_bias(mut self, bias: f32) -> Self {
        self.bias = bias;
        self
    }
}

fn create_ssao_raw_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tetraglass_ssao_raw_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    })
}

fn create_volume_depth_bind_group_layout(
    device: &wgpu::Device,
    sample_count: u32,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tetraglass_volume_depth_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: sample_count > 1,
            },
            count: None,
        }],
    })
}

fn create_wboit_composite_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tetraglass_wboit_composite_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

/// Caller-owned offscreen color and depth render targets.
///
/// Supports `sample_count = 1` (single-sampled) and `sample_count = 4` (4x MSAA resolved into
/// a 1x `color_texture` suitable for both egui texture registration and PNG readback), as well
/// as Weighted Blended Order-Independent Transparency (`M3.3`) accumulation and revealage targets.
#[derive(Debug)]
pub struct OffscreenTarget {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    sample_count: u32,
    color_texture: wgpu::Texture,
    color_view: wgpu::TextureView,
    egui_color_view: wgpu::TextureView,
    msaa_color_view: Option<wgpu::TextureView>,
    depth_view: wgpu::TextureView,
    wboit_accum_view: wgpu::TextureView,
    msaa_wboit_accum_view: Option<wgpu::TextureView>,
    wboit_reveal_view: wgpu::TextureView,
    msaa_wboit_reveal_view: Option<wgpu::TextureView>,
    wboit_bind_group: wgpu::BindGroup,
    volume_depth_bind_group: wgpu::BindGroup,
    ssao_raw_view: wgpu::TextureView,
    ssao_raw_bind_group: wgpu::BindGroup,
}

struct TargetAttachments {
    color_texture: wgpu::Texture,
    color_view: wgpu::TextureView,
    egui_color_view: wgpu::TextureView,
    msaa_color_view: Option<wgpu::TextureView>,
    depth_view: wgpu::TextureView,
    wboit_accum_view: wgpu::TextureView,
    msaa_wboit_accum_view: Option<wgpu::TextureView>,
    wboit_reveal_view: wgpu::TextureView,
    msaa_wboit_reveal_view: Option<wgpu::TextureView>,
    wboit_bind_group: wgpu::BindGroup,
    volume_depth_bind_group: wgpu::BindGroup,
    ssao_raw_view: wgpu::TextureView,
    ssao_raw_bind_group: wgpu::BindGroup,
}

impl OffscreenTarget {
    /// Creates a new offscreen color + depth target of size `width x height`.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Result<Self, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidDimensions { width, height });
        }
        if sample_count != 1 && sample_count != 4 {
            return Err(RenderError::UnsupportedSampleCount { sample_count });
        }

        let att = Self::create_attachments(device, width, height, format, sample_count);

        Ok(Self {
            width,
            height,
            format,
            sample_count,
            color_texture: att.color_texture,
            color_view: att.color_view,
            egui_color_view: att.egui_color_view,
            msaa_color_view: att.msaa_color_view,
            depth_view: att.depth_view,
            wboit_accum_view: att.wboit_accum_view,
            msaa_wboit_accum_view: att.msaa_wboit_accum_view,
            wboit_reveal_view: att.wboit_reveal_view,
            msaa_wboit_reveal_view: att.msaa_wboit_reveal_view,
            wboit_bind_group: att.wboit_bind_group,
            volume_depth_bind_group: att.volume_depth_bind_group,
            ssao_raw_view: att.ssao_raw_view,
            ssao_raw_bind_group: att.ssao_raw_bind_group,
        })
    }

    /// Resizes the offscreen target if `width` or `height` changed.
    ///
    /// Returns `Ok(true)` if the underlying textures were recreated.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Result<bool, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidDimensions { width, height });
        }
        if self.width == width && self.height == height {
            return Ok(false);
        }
        let att = Self::create_attachments(device, width, height, self.format, self.sample_count);
        self.width = width;
        self.height = height;
        self.color_texture = att.color_texture;
        self.color_view = att.color_view;
        self.egui_color_view = att.egui_color_view;
        self.msaa_color_view = att.msaa_color_view;
        self.depth_view = att.depth_view;
        self.wboit_accum_view = att.wboit_accum_view;
        self.msaa_wboit_accum_view = att.msaa_wboit_accum_view;
        self.wboit_reveal_view = att.wboit_reveal_view;
        self.msaa_wboit_reveal_view = att.msaa_wboit_reveal_view;
        self.wboit_bind_group = att.wboit_bind_group;
        self.volume_depth_bind_group = att.volume_depth_bind_group;
        self.ssao_raw_view = att.ssao_raw_view;
        self.ssao_raw_bind_group = att.ssao_raw_bind_group;
        Ok(true)
    }

    /// Returns the width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns the color attachment texture format.
    #[must_use]
    pub const fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Returns the MSAA sample count (`1` or `4`).
    #[must_use]
    pub const fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Returns the 1x resolved color [`wgpu::Texture`] (has `COPY_SRC` and `TEXTURE_BINDING`).
    #[must_use]
    pub const fn color_texture(&self) -> &wgpu::Texture {
        &self.color_texture
    }

    /// Returns the 1x resolved render-attachment [`wgpu::TextureView`].
    #[must_use]
    pub const fn color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }

    /// Returns a non-sRGB (`*Unorm`) [`wgpu::TextureView`] of the resolved color texture
    /// for registration with `egui_wgpu::Renderer::register_native_texture`.
    #[must_use]
    pub const fn egui_texture_view(&self) -> &wgpu::TextureView {
        &self.egui_color_view
    }

    pub(crate) const fn color_attachment_and_resolve(
        &self,
    ) -> (&wgpu::TextureView, Option<&wgpu::TextureView>) {
        match &self.msaa_color_view {
            Some(msaa_view) => (msaa_view, Some(&self.color_view)),
            None => (&self.color_view, None),
        }
    }

    pub(crate) const fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    fn create_attachments(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> TargetAttachments {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let non_srgb_format = format.remove_srgb_suffix();
        let view_formats_storage = [non_srgb_format];
        let view_formats: &[wgpu::TextureFormat] = if non_srgb_format != format {
            &view_formats_storage
        } else {
            &[]
        };

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tetraglass_offscreen_color"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats,
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let egui_color_view = color_texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(non_srgb_format),
            ..Default::default()
        });

        let msaa_color_view = if sample_count > 1 {
            let msaa_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tetraglass_offscreen_msaa_color"),
                size,
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            Some(msaa_tex.create_view(&wgpu::TextureViewDescriptor::default()))
        } else {
            None
        };

        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tetraglass_offscreen_depth"),
            size,
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let volume_depth_bgl = create_volume_depth_bind_group_layout(device, sample_count);
        let volume_depth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tetraglass_volume_depth_bg"),
            layout: &volume_depth_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&depth_view),
            }],
        });

        let wboit_accum_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tetraglass_wboit_accum"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: WBOIT_ACCUM_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let wboit_accum_view = wboit_accum_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let msaa_wboit_accum_view = if sample_count > 1 {
            let msaa_accum = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tetraglass_wboit_msaa_accum"),
                size,
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: WBOIT_ACCUM_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            Some(msaa_accum.create_view(&wgpu::TextureViewDescriptor::default()))
        } else {
            None
        };

        let wboit_reveal_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tetraglass_wboit_reveal"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: WBOIT_REVEAL_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let wboit_reveal_view =
            wboit_reveal_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let msaa_wboit_reveal_view = if sample_count > 1 {
            let msaa_reveal = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tetraglass_wboit_msaa_reveal"),
                size,
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: WBOIT_REVEAL_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            Some(msaa_reveal.create_view(&wgpu::TextureViewDescriptor::default()))
        } else {
            None
        };

        let wboit_bgl = create_wboit_composite_bind_group_layout(device);
        let wboit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tetraglass_wboit_composite_bg"),
            layout: &wboit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&wboit_accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&wboit_reveal_view),
                },
            ],
        });

        let ssao_raw_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tetraglass_ssao_raw"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SSAO_RAW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ssao_raw_view = ssao_raw_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let ssao_raw_bgl = create_ssao_raw_bind_group_layout(device);
        let ssao_raw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tetraglass_ssao_raw_bg"),
            layout: &ssao_raw_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&ssao_raw_view),
            }],
        });

        TargetAttachments {
            color_texture,
            color_view,
            egui_color_view,
            msaa_color_view,
            depth_view,
            wboit_accum_view,
            msaa_wboit_accum_view,
            wboit_reveal_view,
            msaa_wboit_reveal_view,
            wboit_bind_group,
            volume_depth_bind_group,
            ssao_raw_view,
            ssao_raw_bind_group,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SsaoUniforms {
    viewport_size: [f32; 4],
    proj_scale: [f32; 4],
    clip_and_radius: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct VolumeUniforms {
    inv_view_proj: [[f32; 4]; 4],
    box_min: [f32; 4],
    box_max: [f32; 4],
    grid_dims: [f32; 4],
    scalar_params: [f32; 4],
    mode_params: [u32; 4],
    step_params: [f32; 4],
    solid_color: [f32; 4],
    clip_plane: [f32; 4],
    transfer_lut: [[f32; 4]; 64],
}

#[derive(Debug)]
struct GpuVolume {
    grid: Arc<RegularGrid>,
    dimensions: [u32; 3],
    texture: wgpu::Texture,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pending_texels: Mutex<Option<Vec<u8>>>,
    data_min: f32,
    data_scale: f32,
    transfer_function: Option<Vec<[f64; 2]>>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SurfaceGeometryVertex {
    position: [f32; 3],
    normal: [f32; 3],
    cell_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    eye_pos: [f32; 4],
    key_dir: [f32; 4],
    fill_dir: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SurfaceUniforms {
    range_min: f32,
    range_max: f32,
    coloring_mode: u32,
    use_log_scale: u32,
    solid_color: [f32; 4],
    edge_color: [f32; 4],
    pick_info: [u32; 4],
    clip_plane: [f32; 4],
}

/// Uploaded GPU buffers and bind group for a [`SurfaceMesh`].
///
/// Positions are stored in `f32` relative to [`Self::dataset_center`].
/// Point scalars live in a dedicated `f32` vertex buffer (`@location(3)`), and cell scalars
/// live in a storage buffer indexed by triangle `cell_index`. Changing colormap, range,
/// log scale, solid color, or shader clip plane updates only the uniform buffer without
/// re-uploading geometry or scalar buffers.
#[derive(Debug)]
pub struct GpuSurface {
    dataset_center: [f64; 3],
    bounds: Option<BoundingBox>,
    vertex_count: u32,
    edge_vertex_count: u32,
    num_triangles: usize,
    geometry_buffer: wgpu::Buffer,
    edge_vertex_buffer: wgpu::Buffer,
    scalar_buffer: wgpu::Buffer,
    cell_scalar_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    active_color_by: ColorBy,
    active_colormap: ColormapKind,
    active_clip_plane: Option<ShaderClipPlane>,
    opacity: f32,
    draw_surface: bool,
    draw_edges: bool,
    data_range: [f64; 2],
    effective_range: [f64; 2],
    volume: Option<GpuVolume>,
}

impl GpuSurface {
    /// Returns the dataset center in `f64` world coordinates subtracted from GPU vertex positions.
    #[must_use]
    pub const fn dataset_center(&self) -> [f64; 3] {
        self.dataset_center
    }

    /// Returns the original `f64` [`BoundingBox`] of the surface mesh.
    #[must_use]
    pub const fn bounds(&self) -> Option<BoundingBox> {
        self.bounds
    }

    /// Returns the number of vertices (`3 * num_triangles`) in the GPU draw call.
    #[must_use]
    pub const fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Returns the number of wireframe line vertices (`2 * num_wireframe_edges`) in the GPU edge buffer.
    #[must_use]
    pub const fn edge_vertex_count(&self) -> u32 {
        self.edge_vertex_count
    }

    /// Returns the currently active [`ColormapKind`].
    #[must_use]
    pub const fn colormap(&self) -> ColormapKind {
        self.active_colormap
    }

    /// Returns the currently active [`ShaderClipPlane`], if any.
    #[must_use]
    pub const fn clip_plane(&self) -> Option<ShaderClipPlane> {
        self.active_clip_plane
    }

    /// Returns the surface opacity (`solid_color_srgb[3]`, clamped to `0.0..=1.0`) (`M3.3`).
    #[must_use]
    pub const fn opacity(&self) -> f32 {
        self.opacity
    }

    /// Returns `true` if this representation is a 3D volume rendered by GPU ray marching (`M4.4b`).
    #[must_use]
    pub const fn is_volume(&self) -> bool {
        self.volume.is_some()
    }

    /// Returns the piecewise-linear opacity transfer function `[[t, alpha], ...]` if this is a volume representation (`M4.4b`).
    #[must_use]
    pub fn transfer_function(&self) -> Option<&[[f64; 2]]> {
        self.volume
            .as_ref()
            .and_then(|v| v.transfer_function.as_deref())
    }

    /// Returns `true` if this surface has non-empty shaded faces with opaque alpha (`opacity >= 0.9999`) (`M3.3`).
    #[must_use]
    pub fn is_opaque(&self) -> bool {
        !self.is_volume() && self.draw_surface && self.vertex_count > 0 && self.opacity >= 0.9999
    }

    /// Returns `true` if this surface has non-empty shaded faces with semi-transparent alpha (`1e-5 < opacity < 0.9999`) (`M3.3`).
    #[must_use]
    pub fn is_transparent(&self) -> bool {
        !self.is_volume()
            && self.draw_surface
            && self.vertex_count > 0
            && self.opacity > 1e-5
            && self.opacity < 0.9999
    }

    /// Returns the currently active scalar range `[min, max]` mapped across the colormap.
    #[must_use]
    pub const fn effective_range(&self) -> [f64; 2] {
        self.effective_range
    }

    /// Returns the natural data range `[min, max]` of the active scalar array/component.
    #[must_use]
    pub const fn data_range(&self) -> [f64; 2] {
        self.data_range
    }
}

/// Host-embedded `wgpu` surface renderer.
///
/// Never creates a window, surface, or `wgpu::Device`; it takes `&wgpu::Device` and `&wgpu::Queue`
/// from the host application (or [`crate::headless::HeadlessContext`]) and draws into caller-owned
/// [`OffscreenTarget`]s.
#[derive(Debug)]
pub struct Renderer {
    target_format: wgpu::TextureFormat,
    sample_count: u32,
    ssao: SsaoConfig,
    frame_uniform_buffer: wgpu::Buffer,
    frame_bind_groups: [wgpu::BindGroup; 3],
    surface_bind_group_layout: wgpu::BindGroupLayout,
    surface_pipeline: wgpu::RenderPipeline,
    wboit_accum_pipeline: wgpu::RenderPipeline,
    wboit_composite_pipeline: wgpu::RenderPipeline,
    ssao_uniform_buffer: wgpu::Buffer,
    ssao_uniform_bind_group: wgpu::BindGroup,
    ssao_raw_pipeline: wgpu::RenderPipeline,
    ssao_apply_pipeline: wgpu::RenderPipeline,
    volume_sampler: wgpu::Sampler,
    volume_bind_group_layout: wgpu::BindGroupLayout,
    volume_pipeline: wgpu::RenderPipeline,
    pick_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    annotation_uniform_buffer: wgpu::Buffer,
    annotation_sampler: wgpu::Sampler,
    annotation_bind_group_layout: wgpu::BindGroupLayout,
    annotation_pipeline: wgpu::RenderPipeline,
}

impl Renderer {
    /// Creates the surface rendering pipeline and uploads the 256-entry 1D colormap textures.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Result<Self, RenderError> {
        if sample_count != 1 && sample_count != 4 {
            return Err(RenderError::UnsupportedSampleCount { sample_count });
        }

        let shader_source = concat!(
            include_str!("shaders/common.wgsl"),
            "\n",
            include_str!("shaders/surface.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tetraglass_surface_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let frame_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tetraglass_frame_uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let colormap_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tetraglass_colormap_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let frame_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tetraglass_frame_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D1,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let frame_bind_groups = ColormapKind::ALL.map(|kind| {
            let lut = kind.lut_rgba8();
            let tex = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some(kind.as_str()),
                    size: wgpu::Extent3d {
                        width: 256,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D1,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                bytemuck::cast_slice(&lut),
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D1),
                ..Default::default()
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tetraglass_frame_bg"),
                layout: &frame_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame_uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&colormap_sampler),
                    },
                ],
            })
        });

        let surface_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tetraglass_surface_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tetraglass_surface_pipeline_layout"),
            bind_group_layouts: &[
                Some(&frame_bind_group_layout),
                Some(&surface_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let vertex_buffers = [
            // Slot 0: geometry (position, normal, cell_index)
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<SurfaceGeometryVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint32,
                        offset: 24,
                        shader_location: 2,
                    },
                ],
            }),
            // Slot 1: scalar f32 vertex attribute
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<f32>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 0,
                    shader_location: 3,
                }],
            }),
        ];

        let surface_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tetraglass_surface_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let wboit_accum_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tetraglass_wboit_accum_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_wboit"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: WBOIT_ACCUM_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: WBOIT_REVEAL_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::OneMinusSrc,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });

        let wboit_composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tetraglass_wboit_composite_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/wboit_composite.wgsl").into()),
        });
        let wboit_composite_bgl = create_wboit_composite_bind_group_layout(device);
        let wboit_composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tetraglass_wboit_composite_pipeline_layout"),
                bind_group_layouts: &[Some(&wboit_composite_bgl)],
                immediate_size: 0,
            });
        let wboit_composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("tetraglass_wboit_composite_pipeline"),
                layout: Some(&wboit_composite_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &wboit_composite_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &wboit_composite_shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        let pick_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tetraglass_pick_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_pick"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: PICK_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let edge_vertex_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 3]>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        })];

        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tetraglass_edge_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_edge"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &edge_vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_edge"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let annotation_shader_source = concat!(
            include_str!("shaders/common.wgsl"),
            "\n",
            include_str!("shaders/annotations.wgsl")
        );
        let annotation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tetraglass_annotation_shader"),
            source: wgpu::ShaderSource::Wgsl(annotation_shader_source.into()),
        });

        let annotation_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tetraglass_annotation_uniforms"),
            size: std::mem::size_of::<AnnotationUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let annotation_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tetraglass_annotation_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let annotation_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tetraglass_annotation_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let annotation_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tetraglass_annotation_pipeline_layout"),
                bind_group_layouts: &[Some(&annotation_bind_group_layout)],
                immediate_size: 0,
            });

        let annotation_vertex_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<AnnotationVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 8,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 32,
                    shader_location: 3,
                },
            ],
        })];

        let annotation_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tetraglass_annotation_pipeline"),
            layout: Some(&annotation_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &annotation_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &annotation_vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &annotation_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let volume_depth_snippet = if sample_count > 1 {
            "@group(2) @binding(0)\nvar scene_depth_tex: texture_depth_multisampled_2d;\nfn load_scene_depth(px: vec2<i32>) -> f32 { return textureLoad(scene_depth_tex, px, 0); }\n"
        } else {
            "@group(2) @binding(0)\nvar scene_depth_tex: texture_depth_2d;\nfn load_scene_depth(px: vec2<i32>) -> f32 { return textureLoad(scene_depth_tex, px, 0); }\n"
        };
        let volume_shader_source = format!(
            "{}\n{}\n{}",
            include_str!("shaders/common.wgsl"),
            volume_depth_snippet,
            include_str!("shaders/volume.wgsl"),
        );
        let volume_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tetraglass_volume_shader"),
            source: wgpu::ShaderSource::Wgsl(volume_shader_source.into()),
        });

        let volume_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tetraglass_volume_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let volume_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tetraglass_volume_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let volume_depth_bgl = create_volume_depth_bind_group_layout(device, sample_count);
        let volume_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tetraglass_volume_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&frame_bind_group_layout),
                    Some(&volume_bind_group_layout),
                    Some(&volume_depth_bgl),
                ],
                immediate_size: 0,
            });

        let volume_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tetraglass_volume_pipeline"),
            layout: Some(&volume_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &volume_shader,
                entry_point: Some("vs_volume"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &volume_shader,
                entry_point: Some("fs_volume"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let ssao_depth_snippet = if sample_count > 1 {
            "@group(1) @binding(0)\nvar scene_depth_tex: texture_depth_multisampled_2d;\nfn load_scene_depth(px: vec2<i32>) -> f32 { return textureLoad(scene_depth_tex, px, 0); }\n"
        } else {
            "@group(1) @binding(0)\nvar scene_depth_tex: texture_depth_2d;\nfn load_scene_depth(px: vec2<i32>) -> f32 { return textureLoad(scene_depth_tex, px, 0); }\n"
        };
        let ssao_shader_source = format!(
            "{}\n{}",
            ssao_depth_snippet,
            include_str!("shaders/ssao.wgsl"),
        );
        let ssao_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tetraglass_ssao_shader"),
            source: wgpu::ShaderSource::Wgsl(ssao_shader_source.into()),
        });

        let ssao_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tetraglass_ssao_uniforms"),
            size: std::mem::size_of::<SsaoUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ssao_uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tetraglass_ssao_uniform_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let ssao_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tetraglass_ssao_uniform_bg"),
            layout: &ssao_uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ssao_uniform_buffer.as_entire_binding(),
            }],
        });

        let ssao_raw_bgl = create_ssao_raw_bind_group_layout(device);
        let ssao_raw_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tetraglass_ssao_raw_pipeline_layout"),
                bind_group_layouts: &[Some(&ssao_uniform_bgl), Some(&volume_depth_bgl)],
                immediate_size: 0,
            });

        let ssao_raw_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tetraglass_ssao_raw_pipeline"),
            layout: Some(&ssao_raw_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &ssao_shader,
                entry_point: Some("vs_ssao"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_shader,
                entry_point: Some("fs_ssao_raw"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: SSAO_RAW_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let ssao_apply_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tetraglass_ssao_apply_pipeline_layout"),
                bind_group_layouts: &[
                    Some(&ssao_uniform_bgl),
                    Some(&volume_depth_bgl),
                    Some(&ssao_raw_bgl),
                ],
                immediate_size: 0,
            });

        let ssao_apply_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::Src,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let ssao_apply_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tetraglass_ssao_apply_pipeline"),
            layout: Some(&ssao_apply_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &ssao_shader,
                entry_point: Some("vs_ssao"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &ssao_shader,
                entry_point: Some("fs_ssao_apply"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(ssao_apply_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            target_format,
            sample_count,
            ssao: SsaoConfig::default(),
            frame_uniform_buffer,
            frame_bind_groups,
            surface_bind_group_layout,
            surface_pipeline,
            wboit_accum_pipeline,
            wboit_composite_pipeline,
            ssao_uniform_buffer,
            ssao_uniform_bind_group,
            ssao_raw_pipeline,
            ssao_apply_pipeline,
            volume_sampler,
            volume_bind_group_layout,
            volume_pipeline,
            pick_pipeline,
            edge_pipeline,
            annotation_uniform_buffer,
            annotation_sampler,
            annotation_bind_group_layout,
            annotation_pipeline,
        })
    }

    /// Returns the target texture format this renderer was created for.
    #[must_use]
    pub const fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    /// Returns the MSAA sample count this renderer was created for.
    #[must_use]
    pub const fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Returns the active [`SsaoConfig`] (`M5.2`).
    #[must_use]
    pub const fn ssao(&self) -> SsaoConfig {
        self.ssao
    }

    /// Sets the [`SsaoConfig`] used by subsequent [`Self::render`] and [`Self::render_with_annotations`] calls (`M5.2`).
    pub const fn set_ssao(&mut self, ssao: SsaoConfig) {
        self.ssao = ssao;
    }

    /// Uploads a [`SurfaceMesh`] to GPU buffers using `surface.bounds()` center as the dataset center.
    pub fn upload_surface(
        &self,
        device: &wgpu::Device,
        surface: &SurfaceMesh,
        style: &SurfaceStyle,
    ) -> Result<GpuSurface, RenderError> {
        let center = surface.bounds().map_or([0.0, 0.0, 0.0], |b| b.center());
        self.upload_surface_with_center(device, surface, center, style)
    }

    /// Uploads a [`SurfaceMesh`] to GPU buffers with positions relative to `dataset_center`.
    pub fn upload_surface_with_center(
        &self,
        device: &wgpu::Device,
        surface: &SurfaceMesh,
        dataset_center: [f64; 3],
        style: &SurfaceStyle,
    ) -> Result<GpuSurface, RenderError> {
        let points = surface.points();
        let triangles = surface.triangles();
        let num_triangles = triangles.len();
        let vertex_count_usize = num_triangles * 3;
        let vertex_count = u32::try_from(vertex_count_usize).map_err(|_| {
            tetraglass_core::CoreError::IndexOverflow {
                value: vertex_count_usize as u64,
            }
        })?;

        let mut geom_vertices = Vec::with_capacity(vertex_count_usize.max(1));
        for (tri_idx, &[i0, i1, i2]) in triangles.iter().enumerate() {
            let p0 = points[i0 as usize];
            let p1 = points[i1 as usize];
            let p2 = points[i2 as usize];
            let normal = triangle_normal_f32(p0, p1, p2);
            let cell_index = tri_idx as u32;
            for p in [p0, p1, p2] {
                geom_vertices.push(SurfaceGeometryVertex {
                    position: [
                        (p[0] - dataset_center[0]) as f32,
                        (p[1] - dataset_center[1]) as f32,
                        (p[2] - dataset_center[2]) as f32,
                    ],
                    normal,
                    cell_index,
                });
            }
        }
        if geom_vertices.is_empty() {
            geom_vertices.push(SurfaceGeometryVertex::zeroed());
        }

        let wireframe_edges = surface.wireframe_edges();
        let edge_vertex_count_usize = wireframe_edges.len() * 2;
        let edge_vertex_count = u32::try_from(edge_vertex_count_usize).map_err(|_| {
            tetraglass_core::CoreError::IndexOverflow {
                value: edge_vertex_count_usize as u64,
            }
        })?;
        let mut edge_vertices: Vec<[f32; 3]> = Vec::with_capacity(edge_vertex_count_usize.max(1));
        for &[i0, i1] in wireframe_edges {
            for p in [points[i0 as usize], points[i1 as usize]] {
                edge_vertices.push([
                    (p[0] - dataset_center[0]) as f32,
                    (p[1] - dataset_center[1]) as f32,
                    (p[2] - dataset_center[2]) as f32,
                ]);
            }
        }
        if edge_vertices.is_empty() {
            edge_vertices.push([0.0, 0.0, 0.0]);
        }

        let (extracted, gpu_volume) = if let Some(ref grid) = style.volume_grid {
            let (vol_scalars, data_range, coloring_mode) =
                extract_volume_scalars(grid, &style.color_by)?;
            let effective_range = style.range.unwrap_or(data_range);
            let gpu_vol = self.create_gpu_volume(
                device,
                Arc::clone(grid),
                &vol_scalars,
                data_range,
                effective_range,
                coloring_mode,
                dataset_center,
                style,
            );
            let num_triangles = surface.num_triangles();
            let surf_extracted = ExtractedScalars {
                vertex_scalars: vec![0.0f32; (num_triangles * 3).max(1)],
                cell_scalars: vec![0.0f32; num_triangles.max(1)],
                data_range,
                coloring_mode,
            };
            (surf_extracted, Some(gpu_vol))
        } else {
            (extract_scalars(surface, &style.color_by)?, None)
        };

        let effective_range = style.range.unwrap_or(extracted.data_range);
        let gpu_clip_plane = style
            .clip_plane
            .map_or([0.0, 0.0, 0.0, 1.0], |p| p.to_gpu_plane(dataset_center));
        let uniforms = build_surface_uniforms(
            effective_range,
            extracted.coloring_mode,
            style.use_log_scale,
            style.solid_color_srgb,
            style.edge_color_srgb,
            gpu_clip_plane,
        );

        let geometry_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetraglass_surface_geometry"),
            contents: bytemuck::cast_slice(&geom_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let edge_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetraglass_surface_edges"),
            contents: bytemuck::cast_slice(&edge_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let scalar_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetraglass_surface_vertex_scalars"),
            contents: bytemuck::cast_slice(&extracted.vertex_scalars),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let cell_scalar_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetraglass_surface_cell_scalars"),
            contents: bytemuck::cast_slice(&extracted.cell_scalars),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetraglass_surface_uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tetraglass_surface_bg"),
            layout: &self.surface_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_scalar_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(GpuSurface {
            dataset_center,
            bounds: surface.bounds(),
            vertex_count,
            edge_vertex_count,
            num_triangles,
            geometry_buffer,
            edge_vertex_buffer,
            scalar_buffer,
            cell_scalar_buffer,
            uniform_buffer,
            bind_group,
            active_color_by: style.color_by.clone(),
            active_colormap: style.colormap,
            active_clip_plane: style.clip_plane,
            opacity: style.solid_color_srgb[3].clamp(0.0, 1.0),
            draw_surface: style.draw_surface,
            draw_edges: style.draw_edges,
            data_range: extracted.data_range,
            effective_range,
            volume: gpu_volume,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_gpu_volume(
        &self,
        device: &wgpu::Device,
        grid: Arc<RegularGrid>,
        scalars: &[f32],
        data_range: [f64; 2],
        effective_range: [f64; 2],
        coloring_mode: u32,
        dataset_center: [f64; 3],
        style: &SurfaceStyle,
    ) -> GpuVolume {
        let [nx, ny, nz] = grid.dimensions();
        let dims_u32 = [nx as u32, ny as u32, nz as u32];
        let (texels, data_min, data_scale) =
            pack_volume_rgba16f_texels(scalars, grid.inside_mask(), data_range);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tetraglass_volume_texture_3d"),
            size: wgpu::Extent3d {
                width: dims_u32[0],
                height: dims_u32[1],
                depth_or_array_layers: dims_u32[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: VOLUME_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let tex_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        let gpu_clip_plane = style
            .clip_plane
            .map_or([0.0, 0.0, 0.0, 1.0], |p| p.to_gpu_plane(dataset_center));
        let vol_uniforms = build_volume_uniforms(
            &grid,
            dataset_center,
            data_min,
            data_scale,
            effective_range,
            coloring_mode,
            style.use_log_scale,
            style.solid_color_srgb,
            gpu_clip_plane,
            style.transfer_function.as_deref(),
        );
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetraglass_volume_uniforms"),
            contents: bytemuck::bytes_of(&vol_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tetraglass_volume_bg"),
            layout: &self.volume_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.volume_sampler),
                },
            ],
        });

        GpuVolume {
            grid,
            dimensions: dims_u32,
            texture,
            uniform_buffer,
            bind_group,
            pending_texels: Mutex::new(Some(texels)),
            data_min,
            data_scale,
            transfer_function: style.transfer_function.clone(),
        }
    }

    /// Updates a [`GpuSurface`]'s style.
    ///
    /// If `style.color_by` is unchanged, only the uniform buffer is updated (zero vertex or
    /// storage buffer re-upload). If `style.color_by` changed, the scalar buffer is updated too.
    pub fn update_surface_style(
        &self,
        queue: &wgpu::Queue,
        gpu_surface: &mut GpuSurface,
        surface: &SurfaceMesh,
        style: &SurfaceStyle,
    ) -> Result<(), RenderError> {
        if let Some(ref mut gpu_vol) = gpu_surface.volume {
            if gpu_surface.active_color_by != style.color_by {
                let (vol_scalars, data_range, _) =
                    extract_volume_scalars(&gpu_vol.grid, &style.color_by)?;
                let (texels, data_min, data_scale) = pack_volume_rgba16f_texels(
                    &vol_scalars,
                    gpu_vol.grid.inside_mask(),
                    data_range,
                );
                write_volume_texture(queue, &gpu_vol.texture, gpu_vol.dimensions, &texels);
                if let Ok(mut pending) = gpu_vol.pending_texels.lock() {
                    *pending = None;
                }
                gpu_vol.data_min = data_min;
                gpu_vol.data_scale = data_scale;
                gpu_surface.active_color_by = style.color_by.clone();
                gpu_surface.data_range = data_range;
            }
            let coloring_mode = match gpu_surface.active_color_by {
                ColorBy::Solid => 0,
                ColorBy::Point { .. } | ColorBy::Cell { .. } => 1,
            };
            let effective_range = style.range.unwrap_or(gpu_surface.data_range);
            let gpu_clip_plane = style.clip_plane.map_or([0.0, 0.0, 0.0, 1.0], |p| {
                p.to_gpu_plane(gpu_surface.dataset_center)
            });
            let tf_ref = style
                .transfer_function
                .as_deref()
                .or(gpu_vol.transfer_function.as_deref());
            let vol_uniforms = build_volume_uniforms(
                &gpu_vol.grid,
                gpu_surface.dataset_center,
                gpu_vol.data_min,
                gpu_vol.data_scale,
                effective_range,
                coloring_mode,
                style.use_log_scale,
                style.solid_color_srgb,
                gpu_clip_plane,
                tf_ref,
            );
            queue.write_buffer(
                &gpu_vol.uniform_buffer,
                0,
                bytemuck::bytes_of(&vol_uniforms),
            );
            if style.transfer_function.is_some() {
                gpu_vol
                    .transfer_function
                    .clone_from(&style.transfer_function);
            }
            let surf_uniforms = build_surface_uniforms(
                effective_range,
                coloring_mode,
                style.use_log_scale,
                style.solid_color_srgb,
                style.edge_color_srgb,
                gpu_clip_plane,
            );
            queue.write_buffer(
                &gpu_surface.uniform_buffer,
                0,
                bytemuck::bytes_of(&surf_uniforms),
            );
            gpu_surface.active_colormap = style.colormap;
            gpu_surface.active_clip_plane = style.clip_plane;
            gpu_surface.opacity = style.solid_color_srgb[3].clamp(0.0, 1.0);
            gpu_surface.draw_surface = style.draw_surface;
            gpu_surface.draw_edges = style.draw_edges;
            gpu_surface.effective_range = effective_range;
            return Ok(());
        }

        if gpu_surface.active_color_by != style.color_by {
            let extracted = extract_scalars(surface, &style.color_by)?;
            if surface.num_triangles() == gpu_surface.num_triangles {
                queue.write_buffer(
                    &gpu_surface.scalar_buffer,
                    0,
                    bytemuck::cast_slice(&extracted.vertex_scalars),
                );
                queue.write_buffer(
                    &gpu_surface.cell_scalar_buffer,
                    0,
                    bytemuck::cast_slice(&extracted.cell_scalars),
                );
            }
            gpu_surface.active_color_by = style.color_by.clone();
            gpu_surface.data_range = extracted.data_range;
        }

        let coloring_mode = match gpu_surface.active_color_by {
            ColorBy::Solid => 0,
            ColorBy::Point { .. } => 1,
            ColorBy::Cell { .. } => 2,
        };
        let effective_range = style.range.unwrap_or(gpu_surface.data_range);
        let gpu_clip_plane = style.clip_plane.map_or([0.0, 0.0, 0.0, 1.0], |p| {
            p.to_gpu_plane(gpu_surface.dataset_center)
        });
        let uniforms = build_surface_uniforms(
            effective_range,
            coloring_mode,
            style.use_log_scale,
            style.solid_color_srgb,
            style.edge_color_srgb,
            gpu_clip_plane,
        );
        queue.write_buffer(
            &gpu_surface.uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
        gpu_surface.active_colormap = style.colormap;
        gpu_surface.active_clip_plane = style.clip_plane;
        gpu_surface.opacity = style.solid_color_srgb[3].clamp(0.0, 1.0);
        gpu_surface.draw_surface = style.draw_surface;
        gpu_surface.draw_edges = style.draw_edges;
        gpu_surface.effective_range = effective_range;
        Ok(())
    }

    /// Updates only the 16-byte shader clip-plane uniform on `gpu_surface` (`M2.8`, `D-019`).
    pub fn set_surface_clip_plane(
        &self,
        queue: &wgpu::Queue,
        gpu_surface: &mut GpuSurface,
        clip_plane: Option<ShaderClipPlane>,
    ) {
        let gpu_plane = clip_plane.map_or([0.0f32, 0.0, 0.0, 1.0], |p| {
            p.to_gpu_plane(gpu_surface.dataset_center)
        });
        let offset = std::mem::offset_of!(SurfaceUniforms, clip_plane) as u64;
        queue.write_buffer(
            &gpu_surface.uniform_buffer,
            offset,
            bytemuck::bytes_of(&gpu_plane),
        );
        if let Some(ref gpu_vol) = gpu_surface.volume {
            let vol_offset = std::mem::offset_of!(VolumeUniforms, clip_plane) as u64;
            queue.write_buffer(
                &gpu_vol.uniform_buffer,
                vol_offset,
                bytemuck::bytes_of(&gpu_plane),
            );
        }
        gpu_surface.active_clip_plane = clip_plane;
    }

    /// Records a render pass into `encoder`, drawing `surfaces` into `target` with no overlay annotations.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target: &OffscreenTarget,
        camera: &Camera,
        background_srgb: [f32; 4],
        surfaces: &[&GpuSurface],
    ) -> Result<(), RenderError> {
        self.render_internal(
            None,
            encoder,
            queue,
            target,
            (camera, background_srgb),
            surfaces,
        )
    }

    /// Records a render pass into `encoder`, drawing `surfaces` and `annotations` (colorbars,
    /// title text, and 3D orientation axes triad) into `target`.
    // Mirrors `Renderer::render` with the host `device` and `annotations` added so callers do not
    // need to construct an intermediate descriptor struct per frame.
    #[allow(clippy::too_many_arguments)]
    pub fn render_with_annotations(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target: &OffscreenTarget,
        camera: &Camera,
        background_srgb: [f32; 4],
        surfaces: &[&GpuSurface],
        annotations: &Annotations,
    ) -> Result<(), RenderError> {
        self.render_internal(
            Some((device, annotations)),
            encoder,
            queue,
            target,
            (camera, background_srgb),
            surfaces,
        )
    }

    fn render_internal(
        &self,
        annotations_ctx: Option<(&wgpu::Device, &Annotations)>,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target: &OffscreenTarget,
        (camera, background_srgb): (&Camera, [f32; 4]),
        surfaces: &[&GpuSurface],
    ) -> Result<(), RenderError> {
        if target.format() != self.target_format {
            return Err(RenderError::FormatMismatch {
                expected: self.target_format,
                actual: target.format(),
            });
        }
        if target.sample_count() != self.sample_count {
            return Err(RenderError::UnsupportedSampleCount {
                sample_count: target.sample_count(),
            });
        }

        let aspect = (target.width() as f32) / (target.height().max(1) as f32);
        let view_proj = camera.view_proj_matrix(aspect);
        let eye = camera.eye_position();
        let (key_dir, fill_dir) = camera.light_directions();
        let frame_uniforms = FrameUniforms {
            view_proj,
            eye_pos: [eye[0], eye[1], eye[2], 1.0],
            key_dir: [key_dir[0], key_dir[1], key_dir[2], 0.0],
            fill_dir: [fill_dir[0], fill_dir[1], fill_dir[2], 0.0],
        };
        queue.write_buffer(
            &self.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&frame_uniforms),
        );

        let prepared_annotations = if let Some((dev, annotations)) = annotations_ctx
            && !annotations.is_empty()
        {
            let batch = build_annotation_batch(
                annotations,
                camera,
                target.width(),
                target.height(),
                background_srgb,
            );
            if !batch.vertices.is_empty() {
                let overlay_uniforms = AnnotationUniforms {
                    viewport_size: [target.width() as f32, target.height() as f32],
                    _pad: [0.0, 0.0],
                };
                queue.write_buffer(
                    &self.annotation_uniform_buffer,
                    0,
                    bytemuck::bytes_of(&overlay_uniforms),
                );

                let vbuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("tetraglass_annotation_vbuf"),
                    contents: bytemuck::cast_slice(&batch.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                let atlas_tex = dev.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("tetraglass_annotation_atlas"),
                        size: wgpu::Extent3d {
                            width: batch.atlas_width,
                            height: batch.atlas_height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    &batch.atlas_r8,
                );
                let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());
                let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("tetraglass_annotation_bg"),
                    layout: &self.annotation_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.annotation_uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&atlas_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.annotation_sampler),
                        },
                    ],
                });
                Some((vbuf, bg, batch.vertices.len() as u32))
            } else {
                None
            }
        } else {
            None
        };

        let clear_linear = srgb_rgba_to_linear_color(background_srgb);
        let (color_attachment_view, resolve_target) = match &target.msaa_color_view {
            Some(msaa_view) => (msaa_view, Some(&target.color_view)),
            None => (&target.color_view, None),
        };

        let has_opaque = surfaces.iter().any(|s| s.is_opaque());
        let has_transparent = surfaces.iter().any(|s| s.is_transparent());
        let has_volume = surfaces.iter().any(|s| s.is_volume() && s.opacity > 1e-5);
        let use_ssao = self.ssao.enabled && self.ssao.strength > 1e-5 && has_opaque;

        if use_ssao {
            let ssao_uniforms =
                build_ssao_uniforms(camera, target.width(), target.height(), self.ssao);
            queue.write_buffer(
                &self.ssao_uniform_buffer,
                0,
                bytemuck::bytes_of(&ssao_uniforms),
            );
        }

        if has_volume {
            let inv_view_proj = invert_mat4(view_proj);
            for surface in surfaces {
                if let Some(ref gpu_vol) = surface.volume {
                    if let Ok(mut pending) = gpu_vol.pending_texels.lock()
                        && let Some(texels) = pending.take()
                    {
                        write_volume_texture(queue, &gpu_vol.texture, gpu_vol.dimensions, &texels);
                    }
                    queue.write_buffer(
                        &gpu_vol.uniform_buffer,
                        0,
                        bytemuck::bytes_of(&inv_view_proj),
                    );
                }
            }
        }

        if !has_transparent && !has_volume && !use_ssao {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tetraglass_surface_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_attachment_view,
                    depth_slice: None,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_linear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.surface_pipeline);
            for surface in surfaces {
                if !surface.is_opaque() {
                    continue;
                }
                let cm_idx = match surface.active_colormap {
                    ColormapKind::Viridis => 0,
                    ColormapKind::Cividis => 1,
                    ColormapKind::CoolWarm => 2,
                };
                pass.set_bind_group(0, &self.frame_bind_groups[cm_idx], &[]);
                pass.set_bind_group(1, &surface.bind_group, &[]);
                pass.set_vertex_buffer(0, surface.geometry_buffer.slice(..));
                pass.set_vertex_buffer(1, surface.scalar_buffer.slice(..));
                pass.draw(0..surface.vertex_count, 0..1);
            }

            self.draw_edges_and_annotations(&mut pass, surfaces, &prepared_annotations);
        } else {
            // Pass 1: Opaque geometry pass (writes color and depth; defer MSAA resolve to final pass).
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("tetraglass_opaque_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_attachment_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(clear_linear),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &target.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                pass.set_pipeline(&self.surface_pipeline);
                for surface in surfaces {
                    if !surface.is_opaque() {
                        continue;
                    }
                    let cm_idx = match surface.active_colormap {
                        ColormapKind::Viridis => 0,
                        ColormapKind::Cividis => 1,
                        ColormapKind::CoolWarm => 2,
                    };
                    pass.set_bind_group(0, &self.frame_bind_groups[cm_idx], &[]);
                    pass.set_bind_group(1, &surface.bind_group, &[]);
                    pass.set_vertex_buffer(0, surface.geometry_buffer.slice(..));
                    pass.set_vertex_buffer(1, surface.scalar_buffer.slice(..));
                    pass.draw(0..surface.vertex_count, 0..1);
                }
            }

            if use_ssao {
                // Pass 1b: Raw hemisphere SSAO pass into 1x `R8Unorm` `target.ssao_raw_view`.
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("tetraglass_ssao_raw_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &target.ssao_raw_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    pass.set_pipeline(&self.ssao_raw_pipeline);
                    pass.set_bind_group(0, &self.ssao_uniform_bind_group, &[]);
                    pass.set_bind_group(1, &target.volume_depth_bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }

                // Pass 1c: Depth-aware bilateral blur + multiplicative darkening onto opaque color target.
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("tetraglass_ssao_apply_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_attachment_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    pass.set_pipeline(&self.ssao_apply_pipeline);
                    pass.set_bind_group(0, &self.ssao_uniform_bind_group, &[]);
                    pass.set_bind_group(1, &target.volume_depth_bind_group, &[]);
                    pass.set_bind_group(2, &target.ssao_raw_bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }
            }

            if has_transparent {
                // Pass 2: WBOIT accumulation pass (`accum` + `reveal`, depth-tested against opaque pass).
                {
                    let (accum_attachment, accum_resolve) = match &target.msaa_wboit_accum_view {
                        Some(msaa_view) => (msaa_view, Some(&target.wboit_accum_view)),
                        None => (&target.wboit_accum_view, None),
                    };
                    let (reveal_attachment, reveal_resolve) = match &target.msaa_wboit_reveal_view {
                        Some(msaa_view) => (msaa_view, Some(&target.wboit_reveal_view)),
                        None => (&target.wboit_reveal_view, None),
                    };

                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("tetraglass_wboit_accum_pass"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: accum_attachment,
                                depth_slice: None,
                                resolve_target: accum_resolve,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: reveal_attachment,
                                depth_slice: None,
                                resolve_target: reveal_resolve,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &target.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    pass.set_pipeline(&self.wboit_accum_pipeline);
                    for surface in surfaces {
                        if !surface.is_transparent() {
                            continue;
                        }
                        let cm_idx = match surface.active_colormap {
                            ColormapKind::Viridis => 0,
                            ColormapKind::Cividis => 1,
                            ColormapKind::CoolWarm => 2,
                        };
                        pass.set_bind_group(0, &self.frame_bind_groups[cm_idx], &[]);
                        pass.set_bind_group(1, &surface.bind_group, &[]);
                        pass.set_vertex_buffer(0, surface.geometry_buffer.slice(..));
                        pass.set_vertex_buffer(1, surface.scalar_buffer.slice(..));
                        pass.draw(0..surface.vertex_count, 0..1);
                    }
                }

                // Pass 3: WBOIT composite.
                {
                    let composite_resolve = if has_volume { None } else { resolve_target };
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("tetraglass_wboit_composite_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_attachment_view,
                            depth_slice: None,
                            resolve_target: composite_resolve,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &target.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    pass.set_pipeline(&self.wboit_composite_pipeline);
                    pass.set_bind_group(0, &target.wboit_bind_group, &[]);
                    pass.draw(0..3, 0..1);

                    if !has_volume {
                        self.draw_edges_and_annotations(&mut pass, surfaces, &prepared_annotations);
                    }
                }
            }

            if has_volume {
                // Volume ray-marching pass (reads `target.depth_view` via `volume_depth_bind_group`).
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("tetraglass_volume_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_attachment_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    pass.set_pipeline(&self.volume_pipeline);
                    pass.set_bind_group(2, &target.volume_depth_bind_group, &[]);
                    for surface in surfaces {
                        if surface.opacity <= 1e-5 {
                            continue;
                        }
                        let Some(ref gpu_vol) = surface.volume else {
                            continue;
                        };
                        let cm_idx = match surface.active_colormap {
                            ColormapKind::Viridis => 0,
                            ColormapKind::Cividis => 1,
                            ColormapKind::CoolWarm => 2,
                        };
                        pass.set_bind_group(0, &self.frame_bind_groups[cm_idx], &[]);
                        pass.set_bind_group(1, &gpu_vol.bind_group, &[]);
                        pass.draw(0..3, 0..1);
                    }
                }
            }

            if has_volume || (use_ssao && !has_transparent) {
                // Final overlay pass for wireframe edges, annotations, and MSAA resolve.
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("tetraglass_overlay_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_attachment_view,
                            depth_slice: None,
                            resolve_target,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &target.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    self.draw_edges_and_annotations(&mut pass, surfaces, &prepared_annotations);
                }
            }
        }

        Ok(())
    }

    pub(crate) fn prepare_overlays(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &OffscreenTarget,
        camera: &Camera,
        background_srgb: [f32; 4],
        annotations: &Annotations,
    ) -> Option<(wgpu::Buffer, wgpu::BindGroup, u32)> {
        let aspect = (target.width() as f32) / (target.height().max(1) as f32);
        let view_proj = camera.view_proj_matrix(aspect);
        let eye = camera.eye_position();
        let (key_dir, fill_dir) = camera.light_directions();
        let frame_uniforms = FrameUniforms {
            view_proj,
            eye_pos: [eye[0], eye[1], eye[2], 1.0],
            key_dir: [key_dir[0], key_dir[1], key_dir[2], 0.0],
            fill_dir: [fill_dir[0], fill_dir[1], fill_dir[2], 0.0],
        };
        queue.write_buffer(
            &self.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&frame_uniforms),
        );

        if annotations.is_empty() {
            return None;
        }
        let batch = build_annotation_batch(
            annotations,
            camera,
            target.width(),
            target.height(),
            background_srgb,
        );
        if batch.vertices.is_empty() {
            return None;
        }
        let overlay_uniforms = AnnotationUniforms {
            viewport_size: [target.width() as f32, target.height() as f32],
            _pad: [0.0, 0.0],
        };
        queue.write_buffer(
            &self.annotation_uniform_buffer,
            0,
            bytemuck::bytes_of(&overlay_uniforms),
        );

        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetraglass_annotation_vbuf"),
            contents: bytemuck::cast_slice(&batch.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let atlas_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("tetraglass_annotation_atlas"),
                size: wgpu::Extent3d {
                    width: batch.atlas_width,
                    height: batch.atlas_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &batch.atlas_r8,
        );
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tetraglass_annotation_bg"),
            layout: &self.annotation_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.annotation_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.annotation_sampler),
                },
            ],
        });
        Some((vbuf, bg, batch.vertices.len() as u32))
    }

    pub(crate) fn draw_edges_and_annotations<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        surfaces: &[&'a GpuSurface],
        prepared_annotations: &'a Option<(wgpu::Buffer, wgpu::BindGroup, u32)>,
    ) {
        if surfaces
            .iter()
            .any(|s| s.draw_edges && s.edge_vertex_count > 0)
        {
            pass.set_pipeline(&self.edge_pipeline);
            for surface in surfaces {
                if !surface.draw_edges || surface.edge_vertex_count == 0 {
                    continue;
                }
                let cm_idx = match surface.active_colormap {
                    ColormapKind::Viridis => 0,
                    ColormapKind::Cividis => 1,
                    ColormapKind::CoolWarm => 2,
                };
                pass.set_bind_group(0, &self.frame_bind_groups[cm_idx], &[]);
                pass.set_bind_group(1, &surface.bind_group, &[]);
                pass.set_vertex_buffer(0, surface.edge_vertex_buffer.slice(..));
                pass.draw(0..surface.edge_vertex_count, 0..1);
            }
        }

        if let Some((vbuf, bg, vcount)) = prepared_annotations {
            pass.set_pipeline(&self.annotation_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.set_vertex_buffer(0, vbuf.slice(..));
            pass.draw(0..*vcount, 0..1);
        }
    }

    /// Records a GPU ID buffer render pass into `encoder`, drawing `surfaces` into `target`
    /// with 1-based representation IDs (`idx + 1`) and 0-based triangle IDs (`0..num_triangles`).
    pub fn render_pick(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target: &PickTarget,
        camera: &Camera,
        surfaces: &[&GpuSurface],
    ) -> Result<(), RenderError> {
        let aspect = (target.width() as f32) / (target.height().max(1) as f32);
        let view_proj = camera.view_proj_matrix(aspect);
        let eye = camera.eye_position();
        let (key_dir, fill_dir) = camera.light_directions();
        let frame_uniforms = FrameUniforms {
            view_proj,
            eye_pos: [eye[0], eye[1], eye[2], 1.0],
            key_dir: [key_dir[0], key_dir[1], key_dir[2], 0.0],
            fill_dir: [fill_dir[0], fill_dir[1], fill_dir[2], 0.0],
        };
        queue.write_buffer(
            &self.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&frame_uniforms),
        );

        let pick_info_offset = std::mem::offset_of!(SurfaceUniforms, pick_info) as u64;
        for (rep_idx, surface) in surfaces.iter().enumerate() {
            let pick_info: [u32; 4] = [(rep_idx as u32) + 1, 0, 0, 0];
            queue.write_buffer(
                &surface.uniform_buffer,
                pick_info_offset,
                bytemuck::bytes_of(&pick_info),
            );
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tetraglass_pick_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.id_view(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: target.depth_view(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pick_pipeline);
        for surface in surfaces {
            if !surface.draw_surface || surface.vertex_count == 0 || surface.opacity <= 1e-5 {
                continue;
            }
            let cm_idx = match surface.active_colormap {
                ColormapKind::Viridis => 0,
                ColormapKind::Cividis => 1,
                ColormapKind::CoolWarm => 2,
            };
            pass.set_bind_group(0, &self.frame_bind_groups[cm_idx], &[]);
            pass.set_bind_group(1, &surface.bind_group, &[]);
            pass.set_vertex_buffer(0, surface.geometry_buffer.slice(..));
            pass.set_vertex_buffer(1, surface.scalar_buffer.slice(..));
            pass.draw(0..surface.vertex_count, 0..1);
        }

        Ok(())
    }

    /// Submits a GPU ID buffer render pass for `surfaces` into `target` and starts an asynchronous
    /// 1-pixel readback at `(x, y)` without blocking.
    ///
    /// On `wasm32` (WebGPU) this avoids blocking the browser main thread; `on_ready` is invoked
    /// when `map_async` completes so the UI can request a repaint and call [`PendingPixelPick::finish`].
    #[allow(clippy::too_many_arguments)] // Reason: combines device, queue, target, camera, surfaces, pixel coordinates, and completion callback
    pub fn begin_pick_pixel(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &PickTarget,
        camera: &Camera,
        surfaces: &[&GpuSurface],
        x: u32,
        y: u32,
        on_ready: impl FnOnce() + Send + 'static,
    ) -> Result<Option<PendingPixelPick>, RenderError> {
        if x >= target.width() || y >= target.height() {
            return Ok(None);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tetraglass_pick_pixel_staging"),
            size: u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tetraglass_pick_pixel_encoder"),
        });
        self.render_pick(&mut encoder, queue, target, camera, surfaces)?;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target.id_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let map_result: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> =
            Arc::new(Mutex::new(None));
        let map_result_clone = Arc::clone(&map_result);
        staging_buffer
            .slice(..16)
            .map_async(wgpu::MapMode::Read, move |res| {
                if let Ok(mut slot) = map_result_clone.lock() {
                    *slot = Some(res);
                }
                on_ready();
            });

        Ok(Some(PendingPixelPick::new(
            [x, y],
            [target.width(), target.height()],
            staging_buffer,
            map_result,
        )))
    }

    /// Renders `surfaces` into `target` and synchronously reads back the [`RawPickHit`] at pixel `(x, y)`.
    #[allow(clippy::too_many_arguments)] // Reason: combines device, queue, target, camera, surfaces, and pixel coordinates
    pub fn pick_pixel(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &PickTarget,
        camera: &Camera,
        surfaces: &[&GpuSurface],
        x: u32,
        y: u32,
    ) -> Result<Option<RawPickHit>, RenderError> {
        let Some(pending) =
            self.begin_pick_pixel(device, queue, target, camera, surfaces, x, y, || {})?
        else {
            return Ok(None);
        };
        device.poll(wgpu::PollType::wait_indefinitely())?;
        pending.finish()
    }
}

fn build_ssao_uniforms(camera: &Camera, width: u32, height: u32, ssao: SsaoConfig) -> SsaoUniforms {
    let w = (width.max(1)) as f32;
    let h = (height.max(1)) as f32;
    let aspect = (w / h).max(1e-4);
    let tan_half_fov = (0.5 * camera.view_angle_deg.clamp(1.0, 170.0).to_radians()).tan() as f32;
    let scale_x = (aspect * tan_half_fov).max(1e-6);
    let scale_y = tan_half_fov.max(1e-6);
    let (near_f64, far_f64) = camera.near_far_f64();
    let scene_radius = (camera.scene_radius.max(1e-3)) as f32;
    let world_radius = (ssao.radius.clamp(0.005, 2.0) * scene_radius).max(1e-5);
    let world_bias = (ssao.bias.clamp(0.0, 0.5) * world_radius).max(1e-6);
    let strength = ssao.strength.clamp(0.0, 4.0);

    SsaoUniforms {
        viewport_size: [w, h, 1.0 / w, 1.0 / h],
        proj_scale: [scale_x, scale_y, 1.0 / scale_x, 1.0 / scale_y],
        clip_and_radius: [near_f64 as f32, far_f64 as f32, world_radius, world_bias],
        params: [strength, scene_radius, 0.0, 0.0],
    }
}

fn build_surface_uniforms(
    range: [f64; 2],
    coloring_mode: u32,
    use_log_scale: bool,
    solid_color_srgb: [f32; 4],
    edge_color_srgb: [f32; 4],
    clip_plane: [f32; 4],
) -> SurfaceUniforms {
    SurfaceUniforms {
        range_min: range[0] as f32,
        range_max: range[1] as f32,
        coloring_mode,
        use_log_scale: u32::from(use_log_scale),
        solid_color: [
            srgb_channel_to_linear(solid_color_srgb[0]),
            srgb_channel_to_linear(solid_color_srgb[1]),
            srgb_channel_to_linear(solid_color_srgb[2]),
            solid_color_srgb[3],
        ],
        edge_color: [
            srgb_channel_to_linear(edge_color_srgb[0]),
            srgb_channel_to_linear(edge_color_srgb[1]),
            srgb_channel_to_linear(edge_color_srgb[2]),
            edge_color_srgb[3],
        ],
        pick_info: [1, 0, 0, 0],
        clip_plane,
    }
}

pub(crate) struct ExtractedScalars {
    pub(crate) vertex_scalars: Vec<f32>,
    pub(crate) cell_scalars: Vec<f32>,
    pub(crate) data_range: [f64; 2],
    pub(crate) coloring_mode: u32,
}

pub(crate) fn extract_scalars(
    surface: &SurfaceMesh,
    color_by: &ColorBy,
) -> Result<ExtractedScalars, RenderError> {
    let num_triangles = surface.num_triangles();
    let vertex_len = (num_triangles * 3).max(1);
    let cell_len = num_triangles.max(1);

    match color_by {
        ColorBy::Solid => Ok(ExtractedScalars {
            vertex_scalars: vec![0.0f32; vertex_len],
            cell_scalars: vec![0.0f32; cell_len],
            data_range: [0.0, 1.0],
            coloring_mode: 0,
        }),
        ColorBy::Point { name, component } => {
            let array =
                surface
                    .point_data()
                    .get(name)
                    .ok_or_else(|| RenderError::MissingArray {
                        name: name.clone(),
                        association: FieldAssociation::Point,
                    })?;
            let (tuple_scalars, data_range) = evaluate_array_scalars(name, array, *component)?;
            let mut vertex_scalars = Vec::with_capacity(vertex_len);
            for &[i0, i1, i2] in surface.triangles() {
                vertex_scalars.push(tuple_scalars[i0 as usize]);
                vertex_scalars.push(tuple_scalars[i1 as usize]);
                vertex_scalars.push(tuple_scalars[i2 as usize]);
            }
            if vertex_scalars.is_empty() {
                vertex_scalars.push(0.0);
            }
            Ok(ExtractedScalars {
                vertex_scalars,
                cell_scalars: vec![0.0f32; cell_len],
                data_range,
                coloring_mode: 1,
            })
        }
        ColorBy::Cell { name, component } => {
            let array = surface
                .cell_data()
                .get(name)
                .ok_or_else(|| RenderError::MissingArray {
                    name: name.clone(),
                    association: FieldAssociation::Cell,
                })?;
            let (mut cell_scalars, data_range) = evaluate_array_scalars(name, array, *component)?;
            if cell_scalars.is_empty() {
                cell_scalars.push(0.0);
            }
            Ok(ExtractedScalars {
                vertex_scalars: vec![0.0f32; vertex_len],
                cell_scalars,
                data_range,
                coloring_mode: 2,
            })
        }
    }
}

fn evaluate_array_scalars(
    array_name: &str,
    array: &DataArray,
    component: ScalarComponent,
) -> Result<(Vec<f32>, [f64; 2]), RenderError> {
    let num_tuples = array.num_tuples();
    let num_comp = array.components().as_usize();

    if let Some(comp_idx) = component.index() {
        if comp_idx >= num_comp {
            return Err(RenderError::InvalidComponent {
                array: array_name.to_string(),
                component: comp_idx,
                available: num_comp,
            });
        }
        let mut values = Vec::with_capacity(num_tuples);
        for t in 0..num_tuples {
            let v = array.value_as_f64(t, comp_idx).unwrap_or(0.0);
            values.push(v as f32);
        }
        let range = array
            .component_ranges()
            .get(comp_idx)
            .copied()
            .map_or([0.0, 1.0], format_value_range);
        Ok((values, range))
    } else {
        let mut values = Vec::with_capacity(num_tuples);
        for t in 0..num_tuples {
            let mut sum_sq = 0.0f64;
            for c in 0..num_comp {
                let v = array.value_as_f64(t, c).unwrap_or(0.0);
                sum_sq += v * v;
            }
            values.push(sum_sq.sqrt() as f32);
        }
        let range = array
            .magnitude_range()
            .map_or([0.0, 1.0], format_value_range);
        Ok((values, range))
    }
}

fn format_value_range(r: ValueRange) -> [f64; 2] {
    [r.min, r.max]
}

pub(crate) fn triangle_normal_f32(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> [f32; 3] {
    let ux = p1[0] - p0[0];
    let uy = p1[1] - p0[1];
    let uz = p1[2] - p0[2];
    let vx = p2[0] - p0[0];
    let vy = p2[1] - p0[1];
    let vz = p2[2] - p0[2];
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-15 {
        [(nx / len) as f32, (ny / len) as f32, (nz / len) as f32]
    } else {
        [0.0, 0.0, 1.0]
    }
}

pub(crate) fn srgb_channel_to_linear(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub(crate) fn srgb_rgba_to_linear_color(rgba: [f32; 4]) -> wgpu::Color {
    wgpu::Color {
        r: f64::from(srgb_channel_to_linear(rgba[0])),
        g: f64::from(srgb_channel_to_linear(rgba[1])),
        b: f64::from(srgb_channel_to_linear(rgba[2])),
        a: f64::from(rgba[3].clamp(0.0, 1.0)),
    }
}

fn extract_volume_scalars(
    grid: &RegularGrid,
    color_by: &ColorBy,
) -> Result<(Vec<f32>, [f64; 2], u32), RenderError> {
    let num_points = grid.num_points().max(1);
    match color_by {
        ColorBy::Solid => Ok((vec![0.5f32; num_points], [0.0, 1.0], 0)),
        ColorBy::Point { name, component } | ColorBy::Cell { name, component } => {
            let association = match color_by {
                ColorBy::Cell { .. } => FieldAssociation::Cell,
                _ => FieldAssociation::Point,
            };
            let array = grid
                .point_data()
                .get(name)
                .ok_or_else(|| RenderError::MissingArray {
                    name: name.clone(),
                    association,
                })?;
            let (scalars, data_range) = evaluate_array_scalars(name, array, *component)?;
            Ok((scalars, data_range, 1))
        }
    }
}

fn pack_volume_rgba16f_texels(
    scalars: &[f32],
    inside_mask: &[u8],
    data_range: [f64; 2],
) -> (Vec<u8>, f32, f32) {
    let data_min = data_range[0] as f32;
    let span = (data_range[1] - data_range[0]) as f32;
    let data_scale = if span.abs() > 1e-20 { span } else { 1.0 };

    let len = scalars.len().min(inside_mask.len());
    let mut texels_u16: Vec<u16> = Vec::with_capacity(len * 4);
    let zero_h = f32_to_f16_bits(0.0);
    let one_h = f32_to_f16_bits(1.0);
    for i in 0..len {
        let inside = inside_mask[i] != 0;
        if inside {
            let norm = if span.abs() > 1e-20 {
                ((scalars[i] - data_min) / data_scale).clamp(0.0, 1.0)
            } else {
                0.5
            };
            texels_u16.push(f32_to_f16_bits(norm));
            texels_u16.push(one_h);
        } else {
            texels_u16.push(zero_h);
            texels_u16.push(zero_h);
        }
        texels_u16.push(zero_h);
        texels_u16.push(one_h);
    }
    let bytes: Vec<u8> = bytemuck::cast_slice(&texels_u16).to_vec();
    (bytes, data_min, data_scale)
}

fn write_volume_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    dimensions: [u32; 3],
    texels: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(dimensions[0] * 8),
            rows_per_image: Some(dimensions[1]),
        },
        wgpu::Extent3d {
            width: dimensions[0],
            height: dimensions[1],
            depth_or_array_layers: dimensions[2],
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn build_volume_uniforms(
    grid: &RegularGrid,
    dataset_center: [f64; 3],
    data_min: f32,
    data_scale: f32,
    effective_range: [f64; 2],
    coloring_mode: u32,
    use_log_scale: bool,
    solid_color_srgb: [f32; 4],
    clip_plane: [f32; 4],
    transfer_function: Option<&[[f64; 2]]>,
) -> VolumeUniforms {
    let bounds = grid.bounds();
    let box_min = [
        (bounds.min[0] - dataset_center[0]) as f32,
        (bounds.min[1] - dataset_center[1]) as f32,
        (bounds.min[2] - dataset_center[2]) as f32,
        0.0,
    ];
    let box_max = [
        (bounds.max[0] - dataset_center[0]) as f32,
        (bounds.max[1] - dataset_center[1]) as f32,
        (bounds.max[2] - dataset_center[2]) as f32,
        0.0,
    ];
    let [nx, ny, nz] = grid.dimensions();
    let max_dim = nx.max(ny).max(nz).max(2) as f32;
    let max_steps = (max_dim * 2.5).clamp(64.0, 256.0);
    let diag = bounds.diagonal().max(1e-6) as f32;
    let step_size = diag / max_steps;
    let ref_step = diag / 48.0;

    let lut_flat = evaluate_transfer_function_lut(transfer_function);
    let mut transfer_lut = [[0.0f32; 4]; 64];
    for (vec_idx, slot) in transfer_lut.iter_mut().enumerate() {
        let base = vec_idx * 4;
        *slot = [
            lut_flat[base],
            lut_flat[base + 1],
            lut_flat[base + 2],
            lut_flat[base + 3],
        ];
    }

    VolumeUniforms {
        inv_view_proj: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        box_min,
        box_max,
        grid_dims: [nx as f32, ny as f32, nz as f32, max_steps],
        scalar_params: [
            data_min,
            data_scale,
            effective_range[0] as f32,
            effective_range[1] as f32,
        ],
        mode_params: [coloring_mode, u32::from(use_log_scale), 0, 0],
        step_params: [
            solid_color_srgb[3].clamp(0.0, 1.0),
            step_size,
            ref_step,
            0.0,
        ],
        solid_color: [
            srgb_channel_to_linear(solid_color_srgb[0]),
            srgb_channel_to_linear(solid_color_srgb[1]),
            srgb_channel_to_linear(solid_color_srgb[2]),
            solid_color_srgb[3],
        ],
        clip_plane,
        transfer_lut,
    }
}

/// Converts a finite `f32` in `[0.0, 1.0]` (or general `f32`) to IEEE 754 binary16 (`f16`) bits.
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 255 {
        return sign | 0x7c00 | u16::from(mant != 0);
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        return sign | ((m >> shift) as u16);
    }
    let round = (mant + 0x0000_1000) >> 13;
    if round >= 0x0400 {
        let next_exp = (half_exp + 1) as u16;
        if next_exp >= 31 {
            return sign | 0x7c00;
        }
        return sign | (next_exp << 10);
    }
    sign | ((half_exp as u16) << 10) | (round as u16)
}

/// Computes the inverse of a column-major 4x4 matrix in `f64` precision.
fn invert_mat4(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut a = [[0.0f64; 8]; 4];
    for r in 0..4 {
        for c in 0..4 {
            // `m` is column-major `m[col][row]`; invert as standard matrix `M[r][c] = m[c][r]`.
            a[r][c] = f64::from(m[c][r]);
        }
        a[r][4 + r] = 1.0;
    }

    for i in 0..4 {
        let mut pivot_row = i;
        let mut max_abs = a[i][i].abs();
        for (r, row) in a.iter().enumerate().skip(i + 1) {
            if row[i].abs() > max_abs {
                max_abs = row[i].abs();
                pivot_row = r;
            }
        }
        if max_abs <= 1e-18 {
            return [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
        }
        if pivot_row != i {
            a.swap(i, pivot_row);
        }
        let inv_piv = 1.0 / a[i][i];
        for val in &mut a[i] {
            *val *= inv_piv;
        }
        for r in 0..4 {
            if r == i {
                continue;
            }
            let factor = a[r][i];
            if factor.abs() > 0.0 {
                let pivot_slice = a[i];
                for (c, val) in a[r].iter_mut().enumerate() {
                    *val -= factor * pivot_slice[c];
                }
            }
        }
    }

    let mut out = [[0.0f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = a[r][4 + c] as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_transfer_function_lut_interpolates_default_and_custom_points() {
        let default_lut = evaluate_transfer_function_lut(None);
        assert!((default_lut[0] - 0.0).abs() < 1e-6);
        assert!((default_lut[127] - (127.0 / 255.0)).abs() < 1e-4);
        assert!((default_lut[255] - 1.0).abs() < 1e-6);

        let custom = [[0.0, 0.0], [0.5, 0.2], [1.0, 0.8]];
        let lut = evaluate_transfer_function_lut(Some(&custom));
        assert!((lut[0] - 0.0).abs() < 1e-6);
        let mid_idx = 128;
        let t_mid = (mid_idx as f32) / 255.0;
        let expected_mid = 0.2 + ((t_mid - 0.5) / 0.5) * 0.6;
        assert!((lut[mid_idx] - expected_mid).abs() < 1e-4);
        assert!((lut[255] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn invert_mat4_inverts_camera_view_proj() {
        let bounds = BoundingBox {
            min: [0.0, 0.0, 0.0],
            max: [2.0, 2.0, 2.0],
        };
        let cam = Camera::fit_bounds(bounds, 35.0, 25.0);
        let vp = cam.view_proj_matrix(800.0 / 600.0);
        let inv = invert_mat4(vp);

        for (c, inv_col) in inv.iter().enumerate() {
            for r in 0..4usize {
                let sum: f32 = vp
                    .iter()
                    .zip(inv_col.iter())
                    .map(|(vp_col, &inv_val)| vp_col[r] * inv_val)
                    .sum();
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-4,
                    "product at ({r}, {c}) = {sum}, expected {expected}"
                );
            }
        }
    }
}

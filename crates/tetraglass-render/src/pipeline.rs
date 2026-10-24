//! `wgpu` surface rendering pipeline, offscreen targets, and GPU surface buffers.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use tetraglass_core::{BoundingBox, DataArray, SurfaceMesh, ValueRange};
use wgpu::util::DeviceExt;

use crate::annotations::{
    AnnotationUniforms, AnnotationVertex, Annotations, build_annotation_batch,
};
use crate::camera::Camera;
use crate::colormap::ColormapKind;
use crate::error::{FieldAssociation, RenderError};
use crate::pick::{PICK_FORMAT, PickTarget, RawPickHit};

/// Default offscreen color texture format (sRGB 8-bit RGBA).
pub const DEFAULT_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Depth buffer texture format used by offscreen targets.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

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
        }
    }
}

/// Caller-owned offscreen color and depth render targets.
///
/// Supports `sample_count = 1` (single-sampled) and `sample_count = 4` (4x MSAA resolved into
/// a 1x `color_texture` suitable for both egui texture registration and PNG readback).
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
}

struct TargetAttachments {
    color_texture: wgpu::Texture,
    color_view: wgpu::TextureView,
    egui_color_view: wgpu::TextureView,
    msaa_color_view: Option<wgpu::TextureView>,
    depth_view: wgpu::TextureView,
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        TargetAttachments {
            color_texture,
            color_view,
            egui_color_view,
            msaa_color_view,
            depth_view,
        }
    }
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
}

/// Uploaded GPU buffers and bind group for a [`SurfaceMesh`].
///
/// Positions are stored in `f32` relative to [`Self::dataset_center`].
/// Point scalars live in a dedicated `f32` vertex buffer (`@location(3)`), and cell scalars
/// live in a storage buffer indexed by triangle `cell_index`. Changing colormap, range,
/// log scale, or solid color updates only the uniform buffer without re-uploading geometry
/// or scalar buffers.
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
    draw_surface: bool,
    draw_edges: bool,
    data_range: [f64; 2],
    effective_range: [f64; 2],
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
    frame_uniform_buffer: wgpu::Buffer,
    frame_bind_groups: [wgpu::BindGroup; 3],
    surface_bind_group_layout: wgpu::BindGroupLayout,
    surface_pipeline: wgpu::RenderPipeline,
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

        Ok(Self {
            target_format,
            sample_count,
            frame_uniform_buffer,
            frame_bind_groups,
            surface_bind_group_layout,
            surface_pipeline,
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

        let extracted = extract_scalars(surface, &style.color_by)?;

        let effective_range = style.range.unwrap_or(extracted.data_range);
        let uniforms = build_surface_uniforms(
            effective_range,
            extracted.coloring_mode,
            style.use_log_scale,
            style.solid_color_srgb,
            style.edge_color_srgb,
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
            draw_surface: style.draw_surface,
            draw_edges: style.draw_edges,
            data_range: extracted.data_range,
            effective_range,
        })
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
        let uniforms = build_surface_uniforms(
            effective_range,
            coloring_mode,
            style.use_log_scale,
            style.solid_color_srgb,
            style.edge_color_srgb,
        );
        queue.write_buffer(
            &gpu_surface.uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
        gpu_surface.active_colormap = style.colormap;
        gpu_surface.draw_surface = style.draw_surface;
        gpu_surface.draw_edges = style.draw_edges;
        gpu_surface.effective_range = effective_range;
        Ok(())
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
            if !surface.draw_surface || surface.vertex_count == 0 {
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

        if let Some((ref vbuf, ref bg, vcount)) = prepared_annotations {
            pass.set_pipeline(&self.annotation_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.set_vertex_buffer(0, vbuf.slice(..));
            pass.draw(0..vcount, 0..1);
        }

        Ok(())
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
            if !surface.draw_surface || surface.vertex_count == 0 {
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
            });

        device.poll(wgpu::PollType::wait_indefinitely())?;

        let status = map_result
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
            .unwrap_or(Ok(()));
        status?;

        let mapped = staging_buffer.get_mapped_range(..16)?;
        let texel: [u32; 4] = *bytemuck::from_bytes(&mapped[..16]);
        drop(mapped);
        staging_buffer.unmap();

        Ok(RawPickHit::from_rgba_u32(texel))
    }
}

fn build_surface_uniforms(
    range: [f64; 2],
    coloring_mode: u32,
    use_log_scale: bool,
    solid_color_srgb: [f32; 4],
    edge_color_srgb: [f32; 4],
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
    }
}

struct ExtractedScalars {
    vertex_scalars: Vec<f32>,
    cell_scalars: Vec<f32>,
    data_range: [f64; 2],
    coloring_mode: u32,
}

fn extract_scalars(
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

fn triangle_normal_f32(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> [f32; 3] {
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

fn srgb_channel_to_linear(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_rgba_to_linear_color(rgba: [f32; 4]) -> wgpu::Color {
    wgpu::Color {
        r: f64::from(srgb_channel_to_linear(rgba[0])),
        g: f64::from(srgb_channel_to_linear(rgba[1])),
        b: f64::from(srgb_channel_to_linear(rgba[2])),
        a: f64::from(rgba[3].clamp(0.0, 1.0)),
    }
}

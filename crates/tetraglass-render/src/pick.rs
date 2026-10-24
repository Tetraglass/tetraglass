//! GPU ID buffer target (`Rgba32Uint`) and CPU value probing on picked [`SurfaceMesh`] triangles (`M2.7`).
//!
//! During a pick pass, [`crate::Renderer::render_pick`] renders all visible [`crate::GpuSurface`]s
//! into a single-sampled [`PickTarget`] using the `@fragment fn fs_pick` entry point:
//! - `r`: `representation_index + 1` (`0` = background / no hit),
//! - `g`: `triangle_index` (`0..num_triangles`),
//! - `b`: `bitcast<u32>(clip_position.z)` (normalized depth in `0.0..=1.0`),
//! - `a`: `bitcast<u32>(raw_scalar)` (interpolated point scalar or flat cell scalar).
//!
//! Once a pixel (or full ID buffer) is read back as a [`RawPickHit`], [`probe_surface_triangle`]
//! intersects the camera ray through that pixel with the picked triangle in `f64` world space to
//! recover exact barycentric weights, world coordinates, nearest mesh vertex, face and source-cell
//! IDs, and interpolated values across all point and cell [`DataArray`](tetraglass_core::DataArray)s.

use std::sync::{Arc, Mutex};

use tetraglass_core::SurfaceMesh;

use crate::camera::Camera;
use crate::error::{FieldAssociation, RenderError};
use crate::pipeline::{ColorBy, DEPTH_FORMAT, ScalarComponent};

/// Texture format for the GPU ID buffer (`[rep_id_plus_1, triangle_id, depth_bits, scalar_bits]`).
pub const PICK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;

/// Single-sampled offscreen GPU ID buffer (`Rgba32Uint` + `Depth32Float`) for pixel picking.
#[derive(Debug)]
pub struct PickTarget {
    width: u32,
    height: u32,
    id_texture: wgpu::Texture,
    id_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

impl PickTarget {
    /// Creates a new `width x height` GPU ID buffer target.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Result<Self, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidDimensions { width, height });
        }
        let (id_texture, id_view, depth_view) = Self::create_attachments(device, width, height);
        Ok(Self {
            width,
            height,
            id_texture,
            id_view,
            depth_view,
        })
    }

    /// Resizes the pick target if `(width, height)` changed.
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
        let (id_texture, id_view, depth_view) = Self::create_attachments(device, width, height);
        self.width = width;
        self.height = height;
        self.id_texture = id_texture;
        self.id_view = id_view;
        self.depth_view = depth_view;
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

    /// Returns the underlying `Rgba32Uint` [`wgpu::Texture`].
    #[must_use]
    pub const fn id_texture(&self) -> &wgpu::Texture {
        &self.id_texture
    }

    /// Returns the `Rgba32Uint` [`wgpu::TextureView`].
    #[must_use]
    pub const fn id_view(&self) -> &wgpu::TextureView {
        &self.id_view
    }

    /// Returns the `Depth32Float` [`wgpu::TextureView`].
    #[must_use]
    pub(crate) const fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    fn create_attachments(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let id_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tetraglass_pick_id_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PICK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let id_view = id_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tetraglass_pick_depth_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        (id_texture, id_view, depth_view)
    }
}

/// Decoded GPU ID buffer texel (`Rgba32Uint`) for a non-background pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawPickHit {
    /// Zero-based index of the hit surface in the rendered `surfaces` slice.
    pub representation_index: usize,
    /// Zero-based triangle index in the hit [`SurfaceMesh`].
    pub triangle_index: u32,
    /// Normalized device depth (`clip_position.z`) in `0.0..=1.0`.
    pub depth: f32,
    /// Active scalar evaluated by the fragment shader at the pixel.
    pub gpu_scalar: f32,
}

impl RawPickHit {
    /// Decodes a raw `[u32; 4]` texel from the GPU ID buffer (`r == 0` is background / no hit).
    #[must_use]
    pub fn from_rgba_u32(texel: [u32; 4]) -> Option<Self> {
        let rep_plus_one = texel[0];
        if rep_plus_one == 0 {
            return None;
        }
        Some(Self {
            representation_index: (rep_plus_one - 1) as usize,
            triangle_index: texel[1],
            depth: f32::from_bits(texel[2]),
            gpu_scalar: f32::from_bits(texel[3]),
        })
    }
}

/// In-flight single-pixel GPU ID buffer readback started by [`crate::Renderer::begin_pick_pixel`].
///
/// On native targets [`crate::Renderer::pick_pixel`] polls the device synchronously and calls
/// [`Self::finish`] immediately; on `wasm32` (WebGPU) the caller polls [`Self::is_ready`] across
/// animation frames without blocking the browser main thread.
#[derive(Debug)]
pub struct PendingPixelPick {
    pixel: [u32; 2],
    viewport_size: [u32; 2],
    staging_buffer: wgpu::Buffer,
    map_result: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>,
}

impl PendingPixelPick {
    pub(crate) const fn new(
        pixel: [u32; 2],
        viewport_size: [u32; 2],
        staging_buffer: wgpu::Buffer,
        map_result: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>,
    ) -> Self {
        Self {
            pixel,
            viewport_size,
            staging_buffer,
            map_result,
        }
    }

    /// Returns the `[x, y]` physical pixel coordinates being read back.
    #[must_use]
    pub const fn pixel(&self) -> [u32; 2] {
        self.pixel
    }

    /// Returns the `[width, height]` physical viewport dimensions when the pick pass was recorded.
    #[must_use]
    pub const fn viewport_size(&self) -> [u32; 2] {
        self.viewport_size
    }

    /// Returns `true` once the asynchronous `map_async` callback has fired.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.map_result.lock().is_ok_and(|slot| slot.is_some())
    }

    /// Reads the mapped 16-byte `Rgba32Uint` texel, unmaps the staging buffer, and decodes the [`RawPickHit`].
    pub fn finish(self) -> Result<Option<RawPickHit>, RenderError> {
        let status = self
            .map_result
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
            .unwrap_or(Ok(()));
        status?;

        let mapped = self.staging_buffer.get_mapped_range(..16)?;
        let texel: [u32; 4] = *bytemuck::from_bytes(&mapped[..16]);
        drop(mapped);
        self.staging_buffer.unmap();

        Ok(RawPickHit::from_rgba_u32(texel))
    }
}

/// Probed value of a single point or cell attribute array at a picked surface location.
#[derive(Debug, Clone, PartialEq)]
pub struct PickProbeValue {
    /// Name of the attribute array on the [`SurfaceMesh`].
    pub name: String,
    /// Whether this array comes from `point_data` (interpolated) or `cell_data` (per-triangle).
    pub association: FieldAssociation,
    /// Component values at the probed location (`len() == 1` for scalars, `3` for vectors, `6`/`9` for tensors).
    pub values: Vec<f64>,
    /// Euclidean (`L2`) magnitude when `values.len() > 1`, or `None` for scalar arrays.
    pub magnitude: Option<f64>,
}

/// Complete geometry and attribute value-probe result for a picked pixel on a [`SurfaceMesh`].
#[derive(Debug, Clone, PartialEq)]
pub struct SurfacePickResult {
    /// Zero-based index of the hit representation in the rendered surface list.
    pub representation_index: usize,
    /// Pixel coordinates `[x, y]` (`(0, 0)` at top-left) that were probed.
    pub pixel: [u32; 2],
    /// Zero-based triangle index in `surface.triangles()`.
    pub triangle_index: usize,
    /// Source cell index in the originating [`UnstructuredGrid`](tetraglass_core::UnstructuredGrid) (`surface.triangle_source_cells()[triangle_index]`).
    pub source_cell_id: Option<u32>,
    /// Index of the closest mesh vertex (`surface.points()[point_index]`) on the hit triangle.
    pub point_index: u32,
    /// Probed 3D world-space position `[x, y, z]` (`f64`) on the hit triangle.
    pub world_position: [f64; 3],
    /// Clamped barycentric coordinates `[w0, w1, w2]` (`w0 + w1 + w2 == 1.0`) within the hit triangle.
    pub barycentric: [f64; 3],
    /// Normalized device depth in `0.0..=1.0` from the GPU ID buffer.
    pub depth: f32,
    /// Probed scalar value for the surface's active [`ColorBy`] mode, or `None` for [`ColorBy::Solid`].
    pub active_scalar: Option<f64>,
    /// Barycentrically interpolated values for all point attribute arrays on the [`SurfaceMesh`].
    pub point_values: Vec<PickProbeValue>,
    /// Flat per-triangle values for all cell attribute arrays on the [`SurfaceMesh`].
    pub cell_values: Vec<PickProbeValue>,
}

/// Evaluates the exact world position, barycentric weights, cell/vertex IDs, and interpolated
/// point and cell attribute values for a [`RawPickHit`] on `surface`.
#[must_use]
#[allow(clippy::too_many_arguments)] // Reason: combines surface, viewport, camera, and GPU hit context
pub fn probe_surface_triangle(
    surface: &SurfaceMesh,
    representation_index: usize,
    pixel: [u32; 2],
    viewport_size: [u32; 2],
    camera: &Camera,
    dataset_center: [f64; 3],
    raw: RawPickHit,
    color_by: &ColorBy,
) -> Option<SurfacePickResult> {
    let tri_idx = raw.triangle_index as usize;
    let triangles = surface.triangles();
    let &[i0, i1, i2] = triangles.get(tri_idx)?;
    let points = surface.points();
    let p0 = *points.get(i0 as usize)?;
    let p1 = *points.get(i1 as usize)?;
    let p2 = *points.get(i2 as usize)?;

    let (ray_orig_rel, ray_dir) = camera.pixel_ray_f64(
        f64::from(pixel[0]),
        f64::from(pixel[1]),
        f64::from(viewport_size[0].max(1)),
        f64::from(viewport_size[1].max(1)),
    );
    let ray_orig = [
        ray_orig_rel[0] + dataset_center[0],
        ray_orig_rel[1] + dataset_center[1],
        ray_orig_rel[2] + dataset_center[2],
    ];

    let candidate_point = intersect_ray_with_triangle_plane(ray_orig, ray_dir, p0, p1, p2)
        .unwrap_or_else(|| {
            let unproj_rel = camera.unproject_pixel_depth_f64(
                f64::from(pixel[0]),
                f64::from(pixel[1]),
                f64::from(raw.depth),
                f64::from(viewport_size[0].max(1)),
                f64::from(viewport_size[1].max(1)),
            );
            [
                unproj_rel[0] + dataset_center[0],
                unproj_rel[1] + dataset_center[1],
                unproj_rel[2] + dataset_center[2],
            ]
        });

    let barycentric = clamped_barycentric_on_triangle(candidate_point, p0, p1, p2);
    let [w0, w1, w2] = barycentric;
    let world_position = [
        w0 * p0[0] + w1 * p1[0] + w2 * p2[0],
        w0 * p0[1] + w1 * p1[1] + w2 * p2[1],
        w0 * p0[2] + w1 * p1[2] + w2 * p2[2],
    ];

    let point_index = if w0 >= w1 && w0 >= w2 {
        i0
    } else if w1 >= w2 {
        i1
    } else {
        i2
    };

    let source_cell_id = surface.triangle_source_cells().get(tri_idx).copied();

    let mut point_values = Vec::with_capacity(surface.point_data().len());
    for (name, array) in surface.point_data().iter() {
        let num_comp = array.components().as_usize();
        let mut vals = Vec::with_capacity(num_comp);
        let mut sum_sq = 0.0f64;
        for c in 0..num_comp {
            let v0 = array.value_as_f64(i0 as usize, c).unwrap_or(0.0);
            let v1 = array.value_as_f64(i1 as usize, c).unwrap_or(0.0);
            let v2 = array.value_as_f64(i2 as usize, c).unwrap_or(0.0);
            let interp = w0 * v0 + w1 * v1 + w2 * v2;
            sum_sq += interp * interp;
            vals.push(interp);
        }
        let magnitude = (num_comp > 1).then(|| sum_sq.sqrt());
        point_values.push(PickProbeValue {
            name: name.to_string(),
            association: FieldAssociation::Point,
            values: vals,
            magnitude,
        });
    }

    let mut cell_values = Vec::with_capacity(surface.cell_data().len());
    for (name, array) in surface.cell_data().iter() {
        let num_comp = array.components().as_usize();
        let tuple_idx = tri_idx.min(array.num_tuples().saturating_sub(1));
        let mut vals = Vec::with_capacity(num_comp);
        let mut sum_sq = 0.0f64;
        for c in 0..num_comp {
            let v = array.value_as_f64(tuple_idx, c).unwrap_or(0.0);
            sum_sq += v * v;
            vals.push(v);
        }
        let magnitude = (num_comp > 1).then(|| sum_sq.sqrt());
        cell_values.push(PickProbeValue {
            name: name.to_string(),
            association: FieldAssociation::Cell,
            values: vals,
            magnitude,
        });
    }

    let active_scalar = match color_by {
        ColorBy::Solid => None,
        ColorBy::Point { name, component } => point_values
            .iter()
            .find(|pv| pv.name == *name)
            .and_then(|pv| extract_probe_component(pv, *component))
            .or_else(|| Some(f64::from(raw.gpu_scalar))),
        ColorBy::Cell { name, component } => cell_values
            .iter()
            .find(|cv| cv.name == *name)
            .and_then(|cv| extract_probe_component(cv, *component))
            .or_else(|| Some(f64::from(raw.gpu_scalar))),
    };

    Some(SurfacePickResult {
        representation_index,
        pixel,
        triangle_index: tri_idx,
        source_cell_id,
        point_index,
        world_position,
        barycentric,
        depth: raw.depth,
        active_scalar,
        point_values,
        cell_values,
    })
}

fn extract_probe_component(probe: &PickProbeValue, component: ScalarComponent) -> Option<f64> {
    match component {
        ScalarComponent::First => probe.values.first().copied(),
        ScalarComponent::Index(idx) => probe.values.get(idx).copied(),
        ScalarComponent::Magnitude => probe.magnitude.or_else(|| probe.values.first().copied()),
    }
}

fn intersect_ray_with_triangle_plane(
    ray_orig: [f64; 3],
    ray_dir: [f64; 3],
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
) -> Option<[f64; 3]> {
    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let normal = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let denom = normal[0] * ray_dir[0] + normal[1] * ray_dir[1] + normal[2] * ray_dir[2];
    if denom.abs() <= 1e-12 {
        return None;
    }
    let diff = [
        p0[0] - ray_orig[0],
        p0[1] - ray_orig[1],
        p0[2] - ray_orig[2],
    ];
    let t = (normal[0] * diff[0] + normal[1] * diff[1] + normal[2] * diff[2]) / denom;
    if !t.is_finite() || t <= 0.0 {
        return None;
    }
    Some([
        ray_orig[0] + t * ray_dir[0],
        ray_orig[1] + t * ray_dir[1],
        ray_orig[2] + t * ray_dir[2],
    ])
}

/// Projects `p` onto triangle `(p0, p1, p2)` and returns clamped barycentric weights `[w0, w1, w2]`.
fn clamped_barycentric_on_triangle(
    p: [f64; 3],
    p0: [f64; 3],
    p1: [f64; 3],
    p2: [f64; 3],
) -> [f64; 3] {
    let v0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let v2 = [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]];

    let d00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
    let d01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
    let d11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
    let d20 = v2[0] * v0[0] + v2[1] * v0[1] + v2[2] * v0[2];
    let d21 = v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2];

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= 1e-24 {
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    }

    let mut w1 = ((d11 * d20 - d01 * d21) / denom).max(0.0);
    let mut w2 = ((d00 * d21 - d01 * d20) / denom).max(0.0);
    let sum12 = w1 + w2;
    if sum12 > 1.0 {
        w1 /= sum12;
        w2 /= sum12;
    }
    let w0 = (1.0 - w1 - w2).max(0.0);
    [w0, w1, w2]
}

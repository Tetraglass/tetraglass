//! Linear SAH triangle BVH and compute-shader progressive path tracer (`M5.3a`, `D-010`).
//!
//! [`TriangleBvh`] builds a cache-friendly pre-order flattened Surface Area Heuristic (SAH)
//! bounding volume hierarchy over rendered [`SurfaceMesh`] triangles in dataset-center-relative
//! `f32` coordinates (`D-014`). [`PathTracer`] uploads [`GpuPathTraceScene`] buffers and runs a
//! portable WGSL compute shader (`shaders/path_tracer.wgsl`) that evaluates:
//!
//! - multi-bounce cosine-weighted diffuse global illumination and color bleeding,
//! - soft area-light shadow rays toward the camera-anchored key and fill lights,
//! - per-vertex (`point_data`), per-triangle (`cell_data`), and solid colors with linear or
//!   `log10` scaling across all built-in colormaps (`Viridis`, `Cividis`, `CoolWarm`),
//! - interactive [`ShaderClipPlane`](crate::pipeline::ShaderClipPlane) culling and stochastic
//!   semi-transparency, and
//! - progressive Monte Carlo sample accumulation with automatic reset when the camera, viewport
//!   dimensions, or scene changes.
//!
//! In the fullscreen present pass (`fs_pathtrace_present`), accumulated linear RGB radiance and
//! primary-ray WebGPU depth (`ndc_z`) are written into a caller-owned [`OffscreenTarget`] so
//! [`Renderer`] wireframe edges (`draw_edges`) and 2D annotations (`colorbar`, `axes`, `title`)
//! composite cleanly on top of path-traced images.

use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};
use tetraglass_core::{BoundingBox, SurfaceMesh};
use wgpu::util::DeviceExt;

use crate::annotations::Annotations;
use crate::camera::Camera;
use crate::colormap::ColormapKind;
use crate::error::RenderError;
use crate::pipeline::{
    DEPTH_FORMAT, GpuSurface, OffscreenTarget, Renderer, SurfaceStyle, extract_scalars,
    srgb_channel_to_linear, srgb_rgba_to_linear_color, triangle_normal_f32,
};

/// Default total samples per pixel for converged path-traced renders (`32`).
pub const DEFAULT_PATH_TRACE_SAMPLES: u32 = 32;

/// Default samples per pixel traced per interactive frame step (`2`).
pub const DEFAULT_PATH_TRACE_SAMPLES_PER_FRAME: u32 = 2;

/// Default maximum path bounce depth (`3`: primary hit + 2 indirect diffuse bounces).
pub const DEFAULT_PATH_TRACE_MAX_BOUNCES: u32 = 3;

/// Default angular softness (tangent radius) for soft area-light shadows (`0.06`).
pub const DEFAULT_PATH_TRACE_LIGHT_SOFTNESS: f32 = 0.06;

/// Maximum triangles stored in a single BVH leaf node (`4`).
pub const MAX_BVH_LEAF_TRIANGLES: usize = 4;

/// Number of Surface Area Heuristic (SAH) binning buckets evaluated per axis (`12`).
const SAH_BUCKET_COUNT: usize = 12;

static NEXT_SCENE_ID: AtomicU64 = AtomicU64::new(1);

/// Configuration for the compute-shader progressive [`PathTracer`] (`M5.3a`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathTraceConfig {
    /// Target total samples per pixel before progressive accumulation stops (`32` by default).
    pub samples_per_pixel: u32,
    /// Number of samples per pixel traced on each progressive [`PathTracer::step`] call (`2` by default).
    pub samples_per_frame: u32,
    /// Maximum bounce depth (`1..=8`, `3` by default). `1` is direct lighting + soft shadows only;
    /// `>= 2` adds multi-bounce diffuse indirect illumination and color bleeding.
    pub max_bounces: u32,
    /// Angular disk radius (`0.0..=0.5`, `0.06` by default) for soft key/fill light shadows.
    /// `0.0` produces hard pinpoint shadows.
    pub light_softness: f32,
    /// Deterministic random seed offset (`0` by default).
    pub seed_offset: u32,
}

impl Default for PathTraceConfig {
    fn default() -> Self {
        Self {
            samples_per_pixel: DEFAULT_PATH_TRACE_SAMPLES,
            samples_per_frame: DEFAULT_PATH_TRACE_SAMPLES_PER_FRAME,
            max_bounces: DEFAULT_PATH_TRACE_MAX_BOUNCES,
            light_softness: DEFAULT_PATH_TRACE_LIGHT_SOFTNESS,
            seed_offset: 0,
        }
    }
}

impl PathTraceConfig {
    /// Creates a [`PathTraceConfig`] with the given `samples_per_pixel` and default bounce/light settings.
    #[must_use]
    pub const fn new(samples_per_pixel: u32) -> Self {
        Self {
            samples_per_pixel,
            samples_per_frame: DEFAULT_PATH_TRACE_SAMPLES_PER_FRAME,
            max_bounces: DEFAULT_PATH_TRACE_MAX_BOUNCES,
            light_softness: DEFAULT_PATH_TRACE_LIGHT_SOFTNESS,
            seed_offset: 0,
        }
    }

    /// Sets the number of samples per pixel traced per interactive frame step.
    #[must_use]
    pub const fn with_samples_per_frame(mut self, samples_per_frame: u32) -> Self {
        self.samples_per_frame = samples_per_frame;
        self
    }

    /// Sets the maximum bounce depth (`1..=8`).
    #[must_use]
    pub const fn with_max_bounces(mut self, max_bounces: u32) -> Self {
        self.max_bounces = max_bounces;
        self
    }

    /// Sets the angular light softness (`0.0..=0.5`) for soft shadow penumbrae.
    #[must_use]
    pub const fn with_light_softness(mut self, light_softness: f32) -> Self {
        self.light_softness = light_softness;
        self
    }

    /// Sets the deterministic random seed offset.
    #[must_use]
    pub const fn with_seed_offset(mut self, seed_offset: u32) -> Self {
        self.seed_offset = seed_offset;
        self
    }
}

/// Input pairing a [`SurfaceMesh`] with its [`SurfaceStyle`] for BVH construction (`M5.3a`).
#[derive(Debug, Clone, Copy)]
pub struct PathTraceInput<'a> {
    /// Surface mesh geometry and point/cell scalar arrays.
    pub surface: &'a SurfaceMesh,
    /// Display style (coloring mode, colormap, scalar range, opacity, and shader clip plane).
    pub style: &'a SurfaceStyle,
}

impl<'a> PathTraceInput<'a> {
    /// Creates a [`PathTraceInput`] from `surface` and `style`.
    #[must_use]
    pub const fn new(surface: &'a SurfaceMesh, style: &'a SurfaceStyle) -> Self {
        Self { surface, style }
    }
}

/// 32-byte pre-order flattened linear BVH node shared between CPU and WGSL (`M5.3a`).
///
/// - When `count > 0`, the node is a leaf referencing `triangles[right_or_first .. right_or_first + count]`.
/// - When `count == 0` (and the BVH is non-empty), the node is an internal node whose left child is
///   at `node_index + 1` and whose right child is at `right_or_first`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct BvhNode {
    /// Minimum corner `[x, y, z]` of the node's axis-aligned bounding box in dataset-center-relative `f32`.
    pub aabb_min: [f32; 3],
    /// For internal nodes (`count == 0`), index of the right child node; for leaves (`count > 0`),
    /// index of the first triangle in [`TriangleBvh::triangles`].
    pub right_or_first: u32,
    /// Maximum corner `[x, y, z]` of the node's axis-aligned bounding box in dataset-center-relative `f32`.
    pub aabb_max: [f32; 3],
    /// Number of triangles in this leaf node (`> 0`), or `0` for an internal node.
    pub count: u32,
}

impl BvhNode {
    /// Returns `true` if this node is a leaf (`count > 0`).
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.count > 0
    }
}

/// 64-byte GPU triangle primitive for Möller–Trumbore intersection and scalar interpolation (`M5.3a`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct PathTraceTriangle {
    /// Vertex 0 position `[x, y, z]` relative to `dataset_center` and vertex 0 scalar `s0`.
    pub v0_s0: [f32; 4],
    /// Edge vector `v1 - v0` (`[ex, ey, ez]`) and vertex 1 scalar `s1`.
    pub e1_s1: [f32; 4],
    /// Edge vector `v2 - v0` (`[ex, ey, ez]`) and vertex 2 scalar `s2`.
    pub e2_s2: [f32; 4],
    /// Unit geometric normal `[nx, ny, nz]` and `f32::from_bits(material_index)` in `.w`.
    pub normal_and_material: [f32; 4],
}

impl PathTraceTriangle {
    /// Returns the index into [`TriangleBvh::materials`] for this triangle.
    #[must_use]
    pub const fn material_index(&self) -> u32 {
        self.normal_and_material[3].to_bits()
    }
}

/// 64-byte per-surface material and colormap parameters for the compute path tracer (`M5.3a`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct PathTraceMaterial {
    /// Linear RGB solid color `[r, g, b]` and opacity `a` (`0.0..=1.0`).
    pub solid_color: [f32; 4],
    /// Dataset-center-relative clip plane `[nx, ny, nz, d]` where kept points satisfy `dot(p, n) + d >= 0`.
    pub clip_plane: [f32; 4],
    /// Effective scalar range `[range_min, range_max, 0.0, 0.0]`.
    pub scalar_range: [f32; 4],
    /// `[coloring_mode, colormap_index, use_log_scale, has_clip_plane]`.
    pub mode_flags: [u32; 4],
}

/// Pre-order flattened linear SAH triangle BVH built on the CPU for GPU upload (`M5.3a`).
#[derive(Debug, Clone)]
pub struct TriangleBvh {
    dataset_center: [f64; 3],
    bounds: Option<BoundingBox>,
    nodes: Vec<BvhNode>,
    triangles: Vec<PathTraceTriangle>,
    materials: Vec<PathTraceMaterial>,
}

impl TriangleBvh {
    /// Builds a linear SAH [`TriangleBvh`] over the given `inputs` relative to `dataset_center`.
    pub fn build(
        inputs: &[PathTraceInput<'_>],
        dataset_center: [f64; 3],
    ) -> Result<Self, RenderError> {
        let mut raw_triangles: Vec<PathTraceTriangle> = Vec::new();
        let mut tri_aabbs: Vec<[[f32; 3]; 2]> = Vec::new();
        let mut tri_centroids: Vec<[f32; 3]> = Vec::new();
        let mut materials: Vec<PathTraceMaterial> = Vec::with_capacity(inputs.len().max(1));
        let mut combined_bounds: Option<BoundingBox> = None;

        for (surf_idx, input) in inputs.iter().enumerate() {
            let surface = input.surface;
            let style = input.style;
            let mat_idx = surf_idx as u32;

            if let Some(b) = surface.bounds() {
                combined_bounds = Some(match combined_bounds {
                    Some(prev) => BoundingBox {
                        min: [
                            prev.min[0].min(b.min[0]),
                            prev.min[1].min(b.min[1]),
                            prev.min[2].min(b.min[2]),
                        ],
                        max: [
                            prev.max[0].max(b.max[0]),
                            prev.max[1].max(b.max[1]),
                            prev.max[2].max(b.max[2]),
                        ],
                    },
                    None => b,
                });
            }

            let extracted = extract_scalars(surface, &style.color_by)?;
            let effective_range = style.range.unwrap_or(extracted.data_range);
            materials.push(build_material(
                style,
                effective_range,
                extracted.coloring_mode,
                dataset_center,
            ));

            if !style.draw_surface
                || style.volume_grid.is_some()
                || style.solid_color_srgb[3] <= 1e-5
            {
                continue;
            }

            let points = surface.points();
            let triangles = surface.triangles();
            for (tri_idx, &[i0, i1, i2]) in triangles.iter().enumerate() {
                let p0 = points[i0 as usize];
                let p1 = points[i1 as usize];
                let p2 = points[i2 as usize];

                let v0 = [
                    (p0[0] - dataset_center[0]) as f32,
                    (p0[1] - dataset_center[1]) as f32,
                    (p0[2] - dataset_center[2]) as f32,
                ];
                let v1 = [
                    (p1[0] - dataset_center[0]) as f32,
                    (p1[1] - dataset_center[1]) as f32,
                    (p1[2] - dataset_center[2]) as f32,
                ];
                let v2 = [
                    (p2[0] - dataset_center[0]) as f32,
                    (p2[1] - dataset_center[1]) as f32,
                    (p2[2] - dataset_center[2]) as f32,
                ];

                let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                let cx = e1[1] * e2[2] - e1[2] * e2[1];
                let cy = e1[2] * e2[0] - e1[0] * e2[2];
                let cz = e1[0] * e2[1] - e1[1] * e2[0];
                if (cx * cx + cy * cy + cz * cz) <= 1e-24 {
                    continue;
                }

                let normal = triangle_normal_f32(p0, p1, p2);
                let (s0, s1, s2) = match extracted.coloring_mode {
                    1 => {
                        let base = tri_idx * 3;
                        (
                            extracted.vertex_scalars[base],
                            extracted.vertex_scalars[base + 1],
                            extracted.vertex_scalars[base + 2],
                        )
                    }
                    2 => {
                        let cs = extracted.cell_scalars[tri_idx];
                        (cs, cs, cs)
                    }
                    _ => (0.0, 0.0, 0.0),
                };

                let eps = 1e-5_f32;
                let bmin = [
                    v0[0].min(v1[0]).min(v2[0]) - eps,
                    v0[1].min(v1[1]).min(v2[1]) - eps,
                    v0[2].min(v1[2]).min(v2[2]) - eps,
                ];
                let bmax = [
                    v0[0].max(v1[0]).max(v2[0]) + eps,
                    v0[1].max(v1[1]).max(v2[1]) + eps,
                    v0[2].max(v1[2]).max(v2[2]) + eps,
                ];
                let centroid = [
                    (v0[0] + v1[0] + v2[0]) * (1.0 / 3.0),
                    (v0[1] + v1[1] + v2[1]) * (1.0 / 3.0),
                    (v0[2] + v1[2] + v2[2]) * (1.0 / 3.0),
                ];

                raw_triangles.push(PathTraceTriangle {
                    v0_s0: [v0[0], v0[1], v0[2], s0],
                    e1_s1: [e1[0], e1[1], e1[2], s1],
                    e2_s2: [e2[0], e2[1], e2[2], s2],
                    normal_and_material: [normal[0], normal[1], normal[2], f32::from_bits(mat_idx)],
                });
                tri_aabbs.push([bmin, bmax]);
                tri_centroids.push(centroid);
            }
        }

        if materials.is_empty() {
            materials.push(PathTraceMaterial::zeroed());
        }

        let num_tris = raw_triangles.len();
        let _ = u32::try_from(num_tris).map_err(|_| tetraglass_core::CoreError::IndexOverflow {
            value: num_tris as u64,
        })?;

        let mut nodes = Vec::with_capacity((num_tris * 2).max(1));
        let mut ordered_triangles = Vec::with_capacity(num_tris);

        if num_tris == 0 {
            nodes.push(BvhNode {
                aabb_min: [0.0, 0.0, 0.0],
                right_or_first: 0,
                aabb_max: [0.0, 0.0, 0.0],
                count: 0,
            });
        } else {
            let mut indices: Vec<usize> = (0..num_tris).collect();
            build_bvh_recursive(
                &mut indices,
                &raw_triangles,
                &tri_aabbs,
                &tri_centroids,
                &mut nodes,
                &mut ordered_triangles,
            );
        }

        Ok(Self {
            dataset_center,
            bounds: combined_bounds,
            nodes,
            triangles: ordered_triangles,
            materials,
        })
    }

    /// Returns the dataset center in `f64` world coordinates subtracted from GPU positions.
    #[must_use]
    pub const fn dataset_center(&self) -> [f64; 3] {
        self.dataset_center
    }

    /// Returns the combined world-space [`BoundingBox`] of the input surfaces.
    #[must_use]
    pub const fn bounds(&self) -> Option<BoundingBox> {
        self.bounds
    }

    /// Returns the number of triangles stored in the BVH.
    #[must_use]
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Returns the number of [`BvhNode`]s in the flattened hierarchy.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the flattened pre-order [`BvhNode`] slice.
    #[must_use]
    pub fn nodes(&self) -> &[BvhNode] {
        &self.nodes
    }

    /// Returns the reordered [`PathTraceTriangle`] slice referenced by leaf nodes.
    #[must_use]
    pub fn triangles(&self) -> &[PathTraceTriangle] {
        &self.triangles
    }

    /// Returns the per-surface [`PathTraceMaterial`] slice.
    #[must_use]
    pub fn materials(&self) -> &[PathTraceMaterial] {
        &self.materials
    }

    /// Intersects a ray `(origin, dir)` in dataset-center-relative `f32` coordinates with the
    /// BVH on the CPU, returning `Some((t, tri_index, [u, v]))` for the closest hit in
    /// `(t_min, t_max)`.
    #[must_use]
    pub fn intersect_ray(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        t_min: f32,
        t_max: f32,
    ) -> Option<(f32, u32, [f32; 2])> {
        if self.triangles.is_empty() || self.nodes.is_empty() {
            return None;
        }
        let inv_dir = safe_inv_dir_cpu(dir);
        let root = &self.nodes[0];
        if intersect_aabb_cpu(origin, inv_dir, root.aabb_min, root.aabb_max, t_max) >= 1e29 {
            return None;
        }

        let mut closest_t = t_max;
        let mut best_hit: Option<(f32, u32, [f32; 2])> = None;
        let mut stack = [0u32; 64];
        let mut stack_ptr = 0usize;
        let mut node_idx = 0usize;

        loop {
            let node = &self.nodes[node_idx];
            if node.count > 0 {
                let first = node.right_or_first as usize;
                let end = first + (node.count as usize);
                for (ti, tri) in self.triangles[first..end].iter().enumerate() {
                    let global_ti = (first + ti) as u32;
                    if let Some((t, u, v)) =
                        intersect_triangle_cpu(origin, dir, tri, &self.materials, t_min, closest_t)
                    {
                        closest_t = t;
                        best_hit = Some((t, global_ti, [u, v]));
                    }
                }
                if stack_ptr == 0 {
                    break;
                }
                stack_ptr -= 1;
                node_idx = stack[stack_ptr] as usize;
            } else {
                let left_idx = node_idx + 1;
                let right_idx = node.right_or_first as usize;
                let left_node = &self.nodes[left_idx];
                let right_node = &self.nodes[right_idx];

                let t_left = intersect_aabb_cpu(
                    origin,
                    inv_dir,
                    left_node.aabb_min,
                    left_node.aabb_max,
                    closest_t,
                );
                let t_right = intersect_aabb_cpu(
                    origin,
                    inv_dir,
                    right_node.aabb_min,
                    right_node.aabb_max,
                    closest_t,
                );

                let hit_left = t_left < 1e29;
                let hit_right = t_right < 1e29;

                if hit_left && hit_right {
                    let (near_idx, far_idx) = if t_left <= t_right {
                        (left_idx, right_idx)
                    } else {
                        (right_idx, left_idx)
                    };
                    if stack_ptr < stack.len() {
                        stack[stack_ptr] = far_idx as u32;
                        stack_ptr += 1;
                    }
                    node_idx = near_idx;
                } else if hit_left {
                    node_idx = left_idx;
                } else if hit_right {
                    node_idx = right_idx;
                } else {
                    if stack_ptr == 0 {
                        break;
                    }
                    stack_ptr -= 1;
                    node_idx = stack[stack_ptr] as usize;
                }
            }
        }

        best_hit
    }
}

/// Uploaded GPU storage buffers for a [`TriangleBvh`] scene (`M5.3a`).
#[derive(Debug)]
pub struct GpuPathTraceScene {
    scene_id: u64,
    dataset_center: [f64; 3],
    bounds: Option<BoundingBox>,
    num_triangles: u32,
    num_bvh_nodes: u32,
    bvh_buffer: wgpu::Buffer,
    triangle_buffer: wgpu::Buffer,
    material_buffer: wgpu::Buffer,
    materials: Vec<PathTraceMaterial>,
}

impl GpuPathTraceScene {
    /// Uploads a [`TriangleBvh`] into GPU storage buffers.
    #[must_use]
    pub fn from_bvh(device: &wgpu::Device, bvh: &TriangleBvh) -> Self {
        let num_triangles = bvh.triangles.len() as u32;
        let num_bvh_nodes = bvh.nodes.len() as u32;

        let bvh_slice: &[BvhNode] = if bvh.nodes.is_empty() {
            &[BvhNode::zeroed()]
        } else {
            &bvh.nodes
        };
        let tri_slice: &[PathTraceTriangle] = if bvh.triangles.is_empty() {
            &[PathTraceTriangle::zeroed()]
        } else {
            &bvh.triangles
        };
        let mat_slice: &[PathTraceMaterial] = if bvh.materials.is_empty() {
            &[PathTraceMaterial::zeroed()]
        } else {
            &bvh.materials
        };

        let bvh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetraglass_pt_bvh_nodes"),
            contents: bytemuck::cast_slice(bvh_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let triangle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetraglass_pt_triangles"),
            contents: bytemuck::cast_slice(tri_slice),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetraglass_pt_materials"),
            contents: bytemuck::cast_slice(mat_slice),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            scene_id: NEXT_SCENE_ID.fetch_add(1, Ordering::Relaxed),
            dataset_center: bvh.dataset_center,
            bounds: bvh.bounds,
            num_triangles,
            num_bvh_nodes,
            bvh_buffer,
            triangle_buffer,
            material_buffer,
            materials: mat_slice.to_vec(),
        }
    }

    /// Returns the dataset center in `f64` world coordinates subtracted from GPU positions.
    #[must_use]
    pub const fn dataset_center(&self) -> [f64; 3] {
        self.dataset_center
    }

    /// Returns the world-space [`BoundingBox`] of the scene.
    #[must_use]
    pub const fn bounds(&self) -> Option<BoundingBox> {
        self.bounds
    }

    /// Returns the number of uploaded triangles.
    #[must_use]
    pub const fn num_triangles(&self) -> u32 {
        self.num_triangles
    }

    /// Returns the number of uploaded BVH nodes.
    #[must_use]
    pub const fn num_bvh_nodes(&self) -> u32 {
        self.num_bvh_nodes
    }

    /// Updates the material parameters (colormap, scalar range, log scale, solid color, or shader
    /// clip plane) for `surface_index` without rebuilding or re-uploading the BVH.
    pub fn update_material(
        &mut self,
        queue: &wgpu::Queue,
        surface_index: usize,
        style: &SurfaceStyle,
        effective_range: [f64; 2],
    ) {
        let Some(slot) = self.materials.get_mut(surface_index) else {
            return;
        };
        let coloring_mode = slot.mode_flags[0];
        let updated = build_material(style, effective_range, coloring_mode, self.dataset_center);
        if *slot != updated {
            *slot = updated;
            let byte_offset = (surface_index * std::mem::size_of::<PathTraceMaterial>()) as u64;
            queue.write_buffer(
                &self.material_buffer,
                byte_offset,
                bytemuck::bytes_of(&updated),
            );
            self.scene_id = NEXT_SCENE_ID.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PathTraceUniforms {
    eye_and_near: [f32; 4],
    cam_right_and_far: [f32; 4],
    cam_up_and_scale_x: [f32; 4],
    cam_back_and_scale_y: [f32; 4],
    key_dir_and_softness: [f32; 4],
    fill_dir_and_radius: [f32; 4],
    background_linear: [f32; 4],
    image_and_samples: [u32; 4],
    trace_params: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PresentUniforms {
    dims: [u32; 4],
}

#[derive(Debug, Clone, PartialEq)]
struct AccumulationStateKey {
    width: u32,
    height: u32,
    scene_id: u64,
    camera: Camera,
    background_srgb: [f32; 4],
    max_bounces: u32,
    light_softness: f32,
    seed_offset: u32,
}

/// Compute-shader progressive path tracer (`M5.3a`, `D-010`).
///
/// Accumulates Monte Carlo path-traced samples into an internal `wgpu::Buffer` (`vec4<f32>` per
/// pixel) whenever the camera and scene remain stationary, and resolves the running average color
/// and primary-hit depth into a caller-owned [`OffscreenTarget`].
#[derive(Debug)]
pub struct PathTracer {
    target_format: wgpu::TextureFormat,
    sample_count: u32,
    config: PathTraceConfig,
    compute_bgl: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
    present_bgl: wgpu::BindGroupLayout,
    present_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    present_uniform_buffer: wgpu::Buffer,
    colormap_buffer: wgpu::Buffer,
    accum_buffer: Option<(u32, u32, wgpu::Buffer, wgpu::BindGroup)>,
    accumulated_samples: u32,
    last_state: Option<AccumulationStateKey>,
}

impl PathTracer {
    /// Creates a new [`PathTracer`] targeting `target_format` and `sample_count` (`1` or `4`).
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Result<Self, RenderError> {
        if sample_count != 1 && sample_count != 4 {
            return Err(RenderError::UnsupportedSampleCount { sample_count });
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tetraglass_path_tracer_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/path_tracer.wgsl").into()),
        });

        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tetraglass_pt_compute_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tetraglass_pt_compute_layout"),
            bind_group_layouts: &[Some(&compute_bgl)],
            immediate_size: 0,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tetraglass_pt_compute_pipeline"),
            layout: Some(&compute_layout),
            module: &shader,
            entry_point: Some("cs_path_trace"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let present_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tetraglass_pt_present_bgl"),
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

        let present_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tetraglass_pt_present_layout"),
            bind_group_layouts: &[Some(&present_bgl)],
            immediate_size: 0,
        });

        let present_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tetraglass_pt_present_pipeline"),
            layout: Some(&present_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_pathtrace_present"),
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
                depth_write_enabled: Some(true),
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
                module: &shader,
                entry_point: Some("fs_pathtrace_present"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tetraglass_pt_uniforms"),
            size: std::mem::size_of::<PathTraceUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let present_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tetraglass_pt_present_uniforms"),
            size: std::mem::size_of::<PresentUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut colormap_entries = Vec::with_capacity(256 * 3);
        for kind in ColormapKind::ALL {
            let lut = kind.lut_rgba8();
            for &[r, g, b, _] in &lut {
                colormap_entries.push([
                    srgb_channel_to_linear(f32::from(r) / 255.0),
                    srgb_channel_to_linear(f32::from(g) / 255.0),
                    srgb_channel_to_linear(f32::from(b) / 255.0),
                    1.0f32,
                ]);
            }
        }
        let colormap_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tetraglass_pt_colormaps"),
            contents: bytemuck::cast_slice(&colormap_entries),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Ok(Self {
            target_format,
            sample_count,
            config: PathTraceConfig::default(),
            compute_bgl,
            compute_pipeline,
            present_bgl,
            present_pipeline,
            uniform_buffer,
            present_uniform_buffer,
            colormap_buffer,
            accum_buffer: None,
            accumulated_samples: 0,
            last_state: None,
        })
    }

    /// Returns the target texture format this path tracer was created for.
    #[must_use]
    pub const fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    /// Returns the MSAA sample count (`1` or `4`) this path tracer was created for.
    #[must_use]
    pub const fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Returns the active [`PathTraceConfig`].
    #[must_use]
    pub const fn config(&self) -> PathTraceConfig {
        self.config
    }

    /// Sets the active [`PathTraceConfig`]. Resets accumulation if bounce depth, light softness,
    /// or seed offset changed, or if `samples_per_pixel` was lowered below `accumulated_samples`.
    pub fn set_config(&mut self, config: PathTraceConfig) {
        if self.config.max_bounces != config.max_bounces
            || (self.config.light_softness - config.light_softness).abs() > 1e-6
            || self.config.seed_offset != config.seed_offset
            || config.samples_per_pixel.max(1) < self.accumulated_samples
        {
            self.reset_accumulation();
        }
        self.config = config;
    }

    /// Returns the number of samples per pixel accumulated so far for the current stationary view.
    #[must_use]
    pub const fn accumulated_samples(&self) -> u32 {
        self.accumulated_samples
    }

    /// Returns `true` if [`Self::accumulated_samples`] has reached `config.samples_per_pixel`.
    #[must_use]
    pub const fn is_converged(&self) -> bool {
        let target_spp = if self.config.samples_per_pixel > 0 {
            self.config.samples_per_pixel
        } else {
            1
        };
        self.accumulated_samples >= target_spp
    }

    /// Resets the progressive accumulation counter so the next [`Self::step`] starts fresh at 0 samples.
    pub fn reset_accumulation(&mut self) {
        self.accumulated_samples = 0;
        self.last_state = None;
    }

    /// Builds a [`TriangleBvh`] for a single `surface` and uploads it as a [`GpuPathTraceScene`].
    pub fn upload_surface(
        &self,
        device: &wgpu::Device,
        surface: &SurfaceMesh,
        style: &SurfaceStyle,
    ) -> Result<GpuPathTraceScene, RenderError> {
        let center = surface.bounds().map_or([0.0, 0.0, 0.0], |b| b.center());
        self.upload_surfaces(device, &[PathTraceInput::new(surface, style)], center)
    }

    /// Builds a combined [`TriangleBvh`] for `inputs` relative to `dataset_center` and uploads it.
    pub fn upload_surfaces(
        &self,
        device: &wgpu::Device,
        inputs: &[PathTraceInput<'_>],
        dataset_center: [f64; 3],
    ) -> Result<GpuPathTraceScene, RenderError> {
        let bvh = TriangleBvh::build(inputs, dataset_center)?;
        Ok(GpuPathTraceScene::from_bvh(device, &bvh))
    }

    /// Executes one progressive path-tracing step (`config.samples_per_frame` samples per pixel,
    /// up to `config.samples_per_pixel`) and resolves the accumulated image into `target`.
    ///
    /// Automatically resets accumulation to `0` if `camera`, `target` dimensions, `background_srgb`,
    /// or `scene` changed since the previous step. Returns the updated [`Self::accumulated_samples`].
    #[allow(clippy::too_many_arguments)] // Reason: combines device, encoder, queue, target, scene, camera, and background color
    pub fn step(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target: &OffscreenTarget,
        scene: &GpuPathTraceScene,
        camera: &Camera,
        background_srgb: [f32; 4],
    ) -> Result<u32, RenderError> {
        self.step_with_overlays(
            device,
            encoder,
            queue,
            target,
            scene,
            camera,
            background_srgb,
            None,
        )
    }

    /// Executes one progressive path-tracing step and composites optional wireframe edges and
    /// 2D annotations (`colorbar`, `axes`, `title`) from `overlays` on top.
    #[allow(clippy::too_many_arguments)] // Reason: combines device, encoder, queue, target, scene, camera, background, and optional overlays
    pub fn step_with_overlays(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target: &OffscreenTarget,
        scene: &GpuPathTraceScene,
        camera: &Camera,
        background_srgb: [f32; 4],
        overlays: Option<(&Renderer, &[&GpuSurface], &Annotations)>,
    ) -> Result<u32, RenderError> {
        let target_spp = self.config.samples_per_pixel.max(1);
        let step_spp = self.config.samples_per_frame.max(1);
        self.dispatch_and_present(
            device,
            encoder,
            queue,
            target,
            scene,
            camera,
            background_srgb,
            target_spp,
            step_spp,
            false,
            overlays,
        )
    }

    /// Resets accumulation and traces all `config.samples_per_pixel` samples in a single compute
    /// dispatch, then presents into `target` with optional wireframe edges and 2D annotations.
    #[allow(clippy::too_many_arguments)] // Reason: combines device, encoder, queue, target, scene, camera, background, and optional overlays
    pub fn render_converged(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target: &OffscreenTarget,
        scene: &GpuPathTraceScene,
        camera: &Camera,
        background_srgb: [f32; 4],
        overlays: Option<(&Renderer, &[&GpuSurface], &Annotations)>,
    ) -> Result<u32, RenderError> {
        let target_spp = self.config.samples_per_pixel.max(1);
        self.dispatch_and_present(
            device,
            encoder,
            queue,
            target,
            scene,
            camera,
            background_srgb,
            target_spp,
            target_spp,
            true,
            overlays,
        )
    }

    #[allow(clippy::too_many_arguments)] // Reason: internal helper coordinating compute dispatch, accumulation buffer, and present pass
    fn dispatch_and_present(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        target: &OffscreenTarget,
        scene: &GpuPathTraceScene,
        camera: &Camera,
        background_srgb: [f32; 4],
        target_spp: u32,
        max_dispatch_spp: u32,
        force_reset: bool,
        overlays: Option<(&Renderer, &[&GpuSurface], &Annotations)>,
    ) -> Result<u32, RenderError> {
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

        let width = target.width().max(1);
        let height = target.height().max(1);
        let state_key = AccumulationStateKey {
            width,
            height,
            scene_id: scene.scene_id,
            camera: camera.clone(),
            background_srgb,
            max_bounces: self.config.max_bounces.clamp(1, 8),
            light_softness: self.config.light_softness.clamp(0.0, 0.5),
            seed_offset: self.config.seed_offset,
        };

        if force_reset || self.last_state.as_ref() != Some(&state_key) {
            self.accumulated_samples = 0;
            self.last_state = Some(state_key);
        }

        self.ensure_accum_buffer(device, width, height);
        let (_, _, ref accum_buf, ref present_bg) = *self
            .accum_buffer
            .as_ref()
            .expect("accum buffer initialized");

        let remaining = target_spp.saturating_sub(self.accumulated_samples);
        let samples_this_dispatch = remaining.min(max_dispatch_spp);

        if samples_this_dispatch > 0 {
            let uniforms = build_pathtrace_uniforms(
                camera,
                width,
                height,
                background_srgb,
                self.accumulated_samples,
                samples_this_dispatch,
                self.config,
                scene.num_triangles,
                scene.num_bvh_nodes,
            );
            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

            let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tetraglass_pt_compute_bg"),
                layout: &self.compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: scene.bvh_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: scene.triangle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: scene.material_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.colormap_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: accum_buf.as_entire_binding(),
                    },
                ],
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tetraglass_pathtrace_compute_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.compute_pipeline);
                cpass.set_bind_group(0, &compute_bg, &[]);
                cpass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
            }

            self.accumulated_samples += samples_this_dispatch;
        }

        let present_uniforms = PresentUniforms {
            dims: [width, height, 0, 0],
        };
        queue.write_buffer(
            &self.present_uniform_buffer,
            0,
            bytemuck::bytes_of(&present_uniforms),
        );

        let prepared_annotations = if let Some((renderer, _gpu_surfaces, annotations)) = overlays {
            renderer.prepare_overlays(device, queue, target, camera, background_srgb, annotations)
        } else {
            None
        };

        let clear_linear = srgb_rgba_to_linear_color(background_srgb);
        let (color_attachment_view, resolve_target) = target.color_attachment_and_resolve();

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tetraglass_pathtrace_present_pass"),
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

            rpass.set_pipeline(&self.present_pipeline);
            rpass.set_bind_group(0, present_bg, &[]);
            rpass.draw(0..3, 0..1);

            if let Some((renderer, gpu_surfaces, _)) = overlays {
                renderer.draw_edges_and_annotations(
                    &mut rpass,
                    gpu_surfaces,
                    &prepared_annotations,
                );
            }
        }

        Ok(self.accumulated_samples)
    }

    fn ensure_accum_buffer(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if let Some((w, h, _, _)) = &self.accum_buffer
            && *w == width
            && *h == height
        {
            return;
        }
        let pixel_count = u64::from(width.max(1)) * u64::from(height.max(1));
        let byte_size = pixel_count * (std::mem::size_of::<[f32; 4]>() as u64);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tetraglass_pt_accum_buffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let present_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tetraglass_pt_present_bg"),
            layout: &self.present_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.present_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        self.accum_buffer = Some((width, height, buffer, present_bg));
        self.accumulated_samples = 0;
    }
}

#[allow(clippy::too_many_arguments)] // Reason: packs camera basis, light rig, viewport size, sample counters, and scene counts into GPU uniform struct
fn build_pathtrace_uniforms(
    camera: &Camera,
    width: u32,
    height: u32,
    background_srgb: [f32; 4],
    prev_samples: u32,
    samples_this_dispatch: u32,
    config: PathTraceConfig,
    num_triangles: u32,
    num_bvh_nodes: u32,
) -> PathTraceUniforms {
    let aspect = (width.max(1) as f32) / (height.max(1) as f32);
    let tan_half_fov = (0.5 * camera.view_angle_deg.clamp(1.0, 170.0).to_radians()).tan() as f32;
    let scale_x = (aspect * tan_half_fov).max(1e-6);
    let scale_y = tan_half_fov.max(1e-6);
    let eye = camera.eye_position();
    let (cam_right, cam_up, cam_back) = camera.basis();
    let (near_f64, far_f64) = camera.near_far_f64();
    let (key_dir, fill_dir) = camera.light_directions();
    let scene_radius = (camera.scene_radius.max(1e-3)) as f32;

    PathTraceUniforms {
        eye_and_near: [eye[0], eye[1], eye[2], near_f64 as f32],
        cam_right_and_far: [cam_right[0], cam_right[1], cam_right[2], far_f64 as f32],
        cam_up_and_scale_x: [cam_up[0], cam_up[1], cam_up[2], scale_x],
        cam_back_and_scale_y: [cam_back[0], cam_back[1], cam_back[2], scale_y],
        key_dir_and_softness: [
            key_dir[0],
            key_dir[1],
            key_dir[2],
            config.light_softness.clamp(0.0, 0.5),
        ],
        fill_dir_and_radius: [fill_dir[0], fill_dir[1], fill_dir[2], scene_radius],
        background_linear: [
            srgb_channel_to_linear(background_srgb[0]),
            srgb_channel_to_linear(background_srgb[1]),
            srgb_channel_to_linear(background_srgb[2]),
            1.0,
        ],
        image_and_samples: [width, height, prev_samples, samples_this_dispatch.max(1)],
        trace_params: [
            config.max_bounces.clamp(1, 8),
            num_triangles,
            num_bvh_nodes,
            config.seed_offset,
        ],
    }
}

fn build_material(
    style: &SurfaceStyle,
    effective_range: [f64; 2],
    extracted_coloring_mode: u32,
    dataset_center: [f64; 3],
) -> PathTraceMaterial {
    let coloring_mode = u32::from(extracted_coloring_mode != 0);
    let colormap_index = match style.colormap {
        ColormapKind::Viridis => 0,
        ColormapKind::Cividis => 1,
        ColormapKind::CoolWarm => 2,
    };
    let (clip_plane, has_clip) = match style.clip_plane {
        Some(plane) => (plane.to_gpu_plane(dataset_center), 1u32),
        None => ([0.0, 0.0, 0.0, 1.0], 0u32),
    };

    PathTraceMaterial {
        solid_color: [
            srgb_channel_to_linear(style.solid_color_srgb[0]),
            srgb_channel_to_linear(style.solid_color_srgb[1]),
            srgb_channel_to_linear(style.solid_color_srgb[2]),
            style.solid_color_srgb[3].clamp(0.0, 1.0),
        ],
        clip_plane,
        scalar_range: [
            effective_range[0] as f32,
            effective_range[1] as f32,
            0.0,
            0.0,
        ],
        mode_flags: [
            coloring_mode,
            colormap_index,
            u32::from(style.use_log_scale),
            has_clip,
        ],
    }
}

fn build_bvh_recursive(
    indices: &mut [usize],
    raw_tris: &[PathTraceTriangle],
    tri_aabbs: &[[[f32; 3]; 2]],
    tri_centroids: &[[f32; 3]],
    nodes: &mut Vec<BvhNode>,
    ordered_tris: &mut Vec<PathTraceTriangle>,
) {
    let mut node_min = [f32::INFINITY; 3];
    let mut node_max = [f32::NEG_INFINITY; 3];
    let mut c_min = [f32::INFINITY; 3];
    let mut c_max = [f32::NEG_INFINITY; 3];

    for &idx in indices.iter() {
        let [bmin, bmax] = tri_aabbs[idx];
        let c = tri_centroids[idx];
        for axis in 0..3 {
            node_min[axis] = node_min[axis].min(bmin[axis]);
            node_max[axis] = node_max[axis].max(bmax[axis]);
            c_min[axis] = c_min[axis].min(c[axis]);
            c_max[axis] = c_max[axis].max(c[axis]);
        }
    }

    let node_idx = nodes.len();
    nodes.push(BvhNode::zeroed());

    if indices.len() <= MAX_BVH_LEAF_TRIANGLES {
        let first_tri = ordered_tris.len() as u32;
        let count = indices.len() as u32;
        for &idx in indices.iter() {
            ordered_tris.push(raw_tris[idx]);
        }
        nodes[node_idx] = BvhNode {
            aabb_min: node_min,
            right_or_first: first_tri,
            aabb_max: node_max,
            count,
        };
        return;
    }

    // Evaluate 12-bucket Surface Area Heuristic (SAH) across all 3 axes.
    let mut best_cost = f32::INFINITY;
    let mut best_axis = 0usize;
    let mut best_split_bucket = 0usize;

    for axis in 0..3 {
        let extent = c_max[axis] - c_min[axis];
        if extent <= 1e-7 {
            continue;
        }
        let inv_extent = (SAH_BUCKET_COUNT as f32) / extent;
        let mut bucket_counts = [0u32; SAH_BUCKET_COUNT];
        let mut bucket_min = [[f32::INFINITY; 3]; SAH_BUCKET_COUNT];
        let mut bucket_max = [[f32::NEG_INFINITY; 3]; SAH_BUCKET_COUNT];

        for &idx in indices.iter() {
            let b = (((tri_centroids[idx][axis] - c_min[axis]) * inv_extent) as usize)
                .min(SAH_BUCKET_COUNT - 1);
            bucket_counts[b] += 1;
            let [bmin, bmax] = tri_aabbs[idx];
            for k in 0..3 {
                bucket_min[b][k] = bucket_min[b][k].min(bmin[k]);
                bucket_max[b][k] = bucket_max[b][k].max(bmax[k]);
            }
        }

        let mut left_area = [0.0f32; SAH_BUCKET_COUNT - 1];
        let mut left_count = [0u32; SAH_BUCKET_COUNT - 1];
        let mut cur_min = [f32::INFINITY; 3];
        let mut cur_max = [f32::NEG_INFINITY; 3];
        let mut cur_cnt = 0u32;
        for b in 0..(SAH_BUCKET_COUNT - 1) {
            if bucket_counts[b] > 0 {
                cur_cnt += bucket_counts[b];
                for k in 0..3 {
                    cur_min[k] = cur_min[k].min(bucket_min[b][k]);
                    cur_max[k] = cur_max[k].max(bucket_max[b][k]);
                }
            }
            left_count[b] = cur_cnt;
            left_area[b] = if cur_cnt > 0 {
                aabb_half_area(cur_min, cur_max)
            } else {
                0.0
            };
        }

        let mut cur_min = [f32::INFINITY; 3];
        let mut cur_max = [f32::NEG_INFINITY; 3];
        let mut cur_cnt = 0u32;
        for b in (1..SAH_BUCKET_COUNT).rev() {
            if bucket_counts[b] > 0 {
                cur_cnt += bucket_counts[b];
                for k in 0..3 {
                    cur_min[k] = cur_min[k].min(bucket_min[b][k]);
                    cur_max[k] = cur_max[k].max(bucket_max[b][k]);
                }
            }
            let l_cnt = left_count[b - 1];
            let r_cnt = cur_cnt;
            if l_cnt > 0 && r_cnt > 0 {
                let r_area = aabb_half_area(cur_min, cur_max);
                let cost = left_area[b - 1] * (l_cnt as f32) + r_area * (r_cnt as f32);
                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    best_split_bucket = b;
                }
            }
        }
    }

    let mid = if best_cost.is_finite() {
        let extent = c_max[best_axis] - c_min[best_axis];
        let inv_extent = (SAH_BUCKET_COUNT as f32) / extent;
        let mut left = 0usize;
        let mut right = indices.len();
        while left < right {
            let idx = indices[left];
            let b = (((tri_centroids[idx][best_axis] - c_min[best_axis]) * inv_extent) as usize)
                .min(SAH_BUCKET_COUNT - 1);
            if b < best_split_bucket {
                left += 1;
            } else {
                right -= 1;
                indices.swap(left, right);
            }
        }
        if left == 0 || left == indices.len() {
            indices.len() / 2
        } else {
            left
        }
    } else {
        // Fallback: sort along the widest centroid axis and split at the median.
        let mut widest_axis = 0usize;
        for axis in 1..3 {
            if (c_max[axis] - c_min[axis]) > (c_max[widest_axis] - c_min[widest_axis]) {
                widest_axis = axis;
            }
        }
        indices.sort_by(|&a, &b| {
            tri_centroids[a][widest_axis]
                .partial_cmp(&tri_centroids[b][widest_axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.len() / 2
    };

    let (left_slice, right_slice) = indices.split_at_mut(mid);
    // Left child is always built immediately at `node_idx + 1`.
    build_bvh_recursive(
        left_slice,
        raw_tris,
        tri_aabbs,
        tri_centroids,
        nodes,
        ordered_tris,
    );
    let right_child_idx = nodes.len() as u32;
    build_bvh_recursive(
        right_slice,
        raw_tris,
        tri_aabbs,
        tri_centroids,
        nodes,
        ordered_tris,
    );

    nodes[node_idx] = BvhNode {
        aabb_min: node_min,
        right_or_first: right_child_idx,
        aabb_max: node_max,
        count: 0,
    };
}

fn aabb_half_area(bmin: [f32; 3], bmax: [f32; 3]) -> f32 {
    let dx = (bmax[0] - bmin[0]).max(0.0);
    let dy = (bmax[1] - bmin[1]).max(0.0);
    let dz = (bmax[2] - bmin[2]).max(0.0);
    dx * dy + dy * dz + dz * dx
}

fn safe_inv_dir_cpu(d: [f32; 3]) -> [f32; 3] {
    let eps = 1e-8_f32;
    [
        1.0 / if d[0].abs() < eps {
            eps.copysign(d[0])
        } else {
            d[0]
        },
        1.0 / if d[1].abs() < eps {
            eps.copysign(d[1])
        } else {
            d[1]
        },
        1.0 / if d[2].abs() < eps {
            eps.copysign(d[2])
        } else {
            d[2]
        },
    ]
}

fn intersect_aabb_cpu(
    origin: [f32; 3],
    inv_dir: [f32; 3],
    bmin: [f32; 3],
    bmax: [f32; 3],
    t_max: f32,
) -> f32 {
    let tx0 = (bmin[0] - origin[0]) * inv_dir[0];
    let tx1 = (bmax[0] - origin[0]) * inv_dir[0];
    let ty0 = (bmin[1] - origin[1]) * inv_dir[1];
    let ty1 = (bmax[1] - origin[1]) * inv_dir[1];
    let tz0 = (bmin[2] - origin[2]) * inv_dir[2];
    let tz1 = (bmax[2] - origin[2]) * inv_dir[2];

    let t_enter = tx0.min(tx1).max(ty0.min(ty1)).max(tz0.min(tz1)).max(0.0);
    let t_exit = tx0.max(tx1).min(ty0.max(ty1)).min(tz0.max(tz1)).min(t_max);
    if t_enter <= t_exit { t_enter } else { 1e30 }
}

fn intersect_triangle_cpu(
    origin: [f32; 3],
    dir: [f32; 3],
    tri: &PathTraceTriangle,
    materials: &[PathTraceMaterial],
    t_min: f32,
    t_max: f32,
) -> Option<(f32, f32, f32)> {
    let e1 = [tri.e1_s1[0], tri.e1_s1[1], tri.e1_s1[2]];
    let e2 = [tri.e2_s2[0], tri.e2_s2[1], tri.e2_s2[2]];
    let pvec = [
        dir[1] * e2[2] - dir[2] * e2[1],
        dir[2] * e2[0] - dir[0] * e2[2],
        dir[0] * e2[1] - dir[1] * e2[0],
    ];
    let det = e1[0] * pvec[0] + e1[1] * pvec[1] + e1[2] * pvec[2];
    if det.abs() <= 1e-14 {
        return None;
    }
    let inv_det = 1.0 / det;
    let tvec = [
        origin[0] - tri.v0_s0[0],
        origin[1] - tri.v0_s0[1],
        origin[2] - tri.v0_s0[2],
    ];
    let u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let qvec = [
        tvec[1] * e1[2] - tvec[2] * e1[1],
        tvec[2] * e1[0] - tvec[0] * e1[2],
        tvec[0] * e1[1] - tvec[1] * e1[0],
    ];
    let v = (dir[0] * qvec[0] + dir[1] * qvec[1] + dir[2] * qvec[2]) * inv_det;
    if v < 0.0 || (u + v) > 1.0 {
        return None;
    }
    let t = (e2[0] * qvec[0] + e2[1] * qvec[1] + e2[2] * qvec[2]) * inv_det;
    if t <= t_min || t >= t_max {
        return None;
    }
    if let Some(mat) = materials.get(tri.material_index() as usize)
        && mat.mode_flags[3] != 0
    {
        let hit_x = origin[0] + dir[0] * t;
        let hit_y = origin[1] + dir[1] * t;
        let hit_z = origin[2] + dir[2] * t;
        let plane = mat.clip_plane;
        if hit_x * plane[0] + hit_y * plane[1] + hit_z * plane[2] + plane[3] < 0.0 {
            return None;
        }
    }
    Some((t, u, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
    use tetraglass_filters::extract_external_surface;

    #[test]
    fn triangle_bvh_matches_brute_force_ray_intersections_and_respects_clip_plane() {
        let grid = SyntheticGridSpec::new(6, 6, 6, SyntheticCellLayout::Hexahedra)
            .generate()
            .expect("generate 6x6x6 hex grid");
        let surface = extract_external_surface(&grid).expect("extract surface");
        let center = surface.bounds().expect("bounds").center();
        let style = SurfaceStyle {
            color_by: crate::pipeline::ColorBy::point("linear"),
            ..SurfaceStyle::default()
        };

        let bvh = TriangleBvh::build(&[PathTraceInput::new(&surface, &style)], center)
            .expect("build triangle bvh");
        assert_eq!(bvh.num_triangles(), surface.num_triangles());
        assert!(bvh.num_nodes() >= 1);

        // Verify that every ray across a 9x9 grid of directions matches brute-force scan over all triangles.
        for iy in -4..=4 {
            for ix in -4..=4 {
                let origin = [ix as f32 * 0.14, iy as f32 * 0.14, 2.0];
                let dir = [0.0, 0.0, -1.0];
                let bvh_hit = bvh.intersect_ray(origin, dir, 1e-4, 100.0);

                let mut brute_closest = 100.0_f32;
                let mut brute_hit = None;
                for (ti, tri) in bvh.triangles().iter().enumerate() {
                    if let Some((t, u, v)) = intersect_triangle_cpu(
                        origin,
                        dir,
                        tri,
                        bvh.materials(),
                        1e-4,
                        brute_closest,
                    ) {
                        brute_closest = t;
                        brute_hit = Some((t, ti as u32, [u, v]));
                    }
                }

                match (bvh_hit, brute_hit) {
                    (Some((t1, _, _)), Some((t2, _, _))) => {
                        assert!(
                            (t1 - t2).abs() < 1e-5,
                            "ray ({ix}, {iy}) BVH t={t1} != brute-force t={t2}"
                        );
                    }
                    (None, None) => {}
                    other => panic!("mismatch at ({ix}, {iy}): {other:?}"),
                }
            }
        }

        // Now enable a shader clip plane removing the front half (`z > 0.5`) and verify hits start at the back wall (`z = 0.0`, relative `z = -0.5` -> `t = 2.5`).
        let clipped_style = SurfaceStyle {
            clip_plane: Some(crate::pipeline::ShaderClipPlane::new(
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 1.0],
            )),
            ..style
        };
        let clipped_bvh =
            TriangleBvh::build(&[PathTraceInput::new(&surface, &clipped_style)], center)
                .expect("build clipped bvh");
        let (t_clipped, _, _) = clipped_bvh
            .intersect_ray([0.0, 0.0, 2.0], [0.0, 0.0, -1.0], 1e-4, 100.0)
            .expect("hit back wall through clipped front face");
        assert!(
            (t_clipped - 2.5).abs() < 1e-4,
            "expected ray to pass through clipped z=+0.5 face and hit z=-0.5 face at t=2.5, got {t_clipped}"
        );
    }
}

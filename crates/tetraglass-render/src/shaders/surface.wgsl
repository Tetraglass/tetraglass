// Surface pipeline shader (concatenated after common.wgsl).

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

@group(0) @binding(1)
var colormap_tex: texture_1d<f32>;

@group(0) @binding(2)
var colormap_sampler: sampler;

@group(1) @binding(0)
var<uniform> surface: SurfaceUniforms;

@group(1) @binding(1)
var<storage, read> cell_scalars: array<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) cell_index: u32,
    @location(3) scalar: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) scalar: f32,
    @location(3) @interpolate(flat) cell_index: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = frame.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.world_normal = in.normal;
    out.scalar = in.scalar;
    out.cell_index = in.cell_index;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var raw_scalar = in.scalar;
    if (surface.coloring_mode == 2u) {
        raw_scalar = cell_scalars[in.cell_index];
    }

    let u = normalize_scalar_to_lut_u(
        raw_scalar,
        surface.range_min,
        surface.range_max,
        surface.use_log_scale,
    );
    let mapped_color = textureSample(colormap_tex, colormap_sampler, u);

    var base_color = surface.solid_color;
    if (surface.coloring_mode != 0u) {
        base_color = vec4<f32>(mapped_color.rgb, surface.solid_color.a);
    }

    let shaded_rgb = shade_three_point(base_color.rgb, in.world_pos, in.world_normal, frame);
    return vec4<f32>(shaded_rgb, base_color.a);
}

struct EdgeVertexInput {
    @location(0) position: vec3<f32>,
};

struct EdgeVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_edge(in: EdgeVertexInput) -> EdgeVertexOutput {
    var out: EdgeVertexOutput;
    var clip = frame.view_proj * vec4<f32>(in.position, 1.0);
    // Apply a small clip-space depth offset toward the camera so wireframe edges sit cleanly
    // on top of coplanar surface triangles without z-fighting.
    clip.z = max(clip.z - 0.0005 * clip.w, 0.0);
    out.clip_position = clip;
    return out;
}

@fragment
fn fs_edge(_in: EdgeVertexOutput) -> @location(0) vec4<f32> {
    return surface.edge_color;
}

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

fn shade_surface_fragment(in: VertexOutput) -> vec4<f32> {
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
    return vec4<f32>(shaded_rgb, clamp(base_color.a, 0.0, 1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = shade_surface_fragment(in);
    if (is_clipped_by_plane(in.world_pos, surface.clip_plane)) {
        discard;
    }
    return color;
}

struct WboitAccumOutput {
    @location(0) accum: vec4<f32>,
    @location(1) reveal: vec4<f32>,
};

@fragment
fn fs_wboit(in: VertexOutput) -> WboitAccumOutput {
    let color = shade_surface_fragment(in);
    if (is_clipped_by_plane(in.world_pos, surface.clip_plane) || color.a <= 1e-5) {
        discard;
    }
    let w = wboit_weight(in.clip_position.z, color.a);
    let raw_accum = vec4<f32>(color.rgb * w, w);
    var out: WboitAccumOutput;
    out.accum = vec4<f32>(
        unpack2x16float(pack2x16float(raw_accum.xy)),
        unpack2x16float(pack2x16float(raw_accum.zw)),
    );
    out.reveal = vec4<f32>(color.a, color.a, color.a, color.a);
    return out;
}

@fragment
fn fs_pick(in: VertexOutput) -> @location(0) vec4<u32> {
    if (is_clipped_by_plane(in.world_pos, surface.clip_plane)) {
        discard;
    }

    var raw_scalar = in.scalar;
    if (surface.coloring_mode == 2u) {
        raw_scalar = cell_scalars[in.cell_index];
    }
    return vec4<u32>(
        surface.pick_info.x,
        in.cell_index,
        bitcast<u32>(in.clip_position.z),
        bitcast<u32>(raw_scalar),
    );
}

struct EdgeVertexInput {
    @location(0) position: vec3<f32>,
};

struct EdgeVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn vs_edge(in: EdgeVertexInput) -> EdgeVertexOutput {
    var out: EdgeVertexOutput;
    var clip = frame.view_proj * vec4<f32>(in.position, 1.0);
    // Apply a small clip-space depth offset toward the camera so wireframe edges sit cleanly
    // on top of coplanar surface triangles without z-fighting.
    clip.z = max(clip.z - 0.0005 * clip.w, 0.0);
    out.clip_position = clip;
    out.world_pos = in.position;
    return out;
}

@fragment
fn fs_edge(in: EdgeVertexOutput) -> @location(0) vec4<f32> {
    if (is_clipped_by_plane(in.world_pos, surface.clip_plane)) {
        discard;
    }
    return surface.edge_color;
}

struct PointVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) quad_uv: vec2<f32>,
    @location(2) scalar: f32,
    @location(3) @interpolate(flat) cell_index: u32,
};

@vertex
fn vs_point(
    @builtin(vertex_index) vertex_idx: u32,
    in: VertexInput,
) -> PointVertexOutput {
    var quad_uv = vec2<f32>(-1.0, -1.0);
    switch (vertex_idx % 6u) {
        case 0u: { quad_uv = vec2<f32>(-1.0, -1.0); }
        case 1u: { quad_uv = vec2<f32>( 1.0, -1.0); }
        case 2u: { quad_uv = vec2<f32>( 1.0,  1.0); }
        case 3u: { quad_uv = vec2<f32>(-1.0, -1.0); }
        case 4u: { quad_uv = vec2<f32>( 1.0,  1.0); }
        default: { quad_uv = vec2<f32>(-1.0,  1.0); }
    }

    var center_clip = frame.view_proj * vec4<f32>(in.position, 1.0);
    let diameter_px = max(surface.point_params.x, 1.0);
    // Apply a small clip-space depth offset toward the camera proportional to point radius
    // so sphere impostors sit cleanly on top of coplanar surface triangles and wireframe junctions.
    center_clip.z = max(
        center_clip.z - (0.0005 + 0.00035 * diameter_px) * center_clip.w,
        0.0,
    );

    let inv_viewport = frame.viewport_size.zw;
    let ndc_offset = quad_uv * diameter_px * inv_viewport;

    var out: PointVertexOutput;
    out.clip_position = vec4<f32>(
        center_clip.xy + ndc_offset * center_clip.w,
        center_clip.z,
        center_clip.w,
    );
    out.world_pos = in.position;
    out.quad_uv = quad_uv;
    out.scalar = in.scalar;
    out.cell_index = in.cell_index;
    return out;
}

fn shade_point_fragment(in: PointVertexOutput, r2: f32) -> vec4<f32> {
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

    // View-space sphere-impostor lighting across the circular point disc.
    let nz = sqrt(max(1.0 - r2, 0.0));
    let n_view = vec3<f32>(in.quad_uv.x, in.quad_uv.y, nz);
    let key_view = normalize(vec3<f32>(0.45, 0.55, 0.70));
    let fill_view = normalize(vec3<f32>(-0.55, -0.25, 0.80));
    let ambient = 0.22;
    let headlight = 0.35 * nz;
    let key_diff = 0.38 * max(dot(n_view, key_view), 0.0);
    let fill_diff = 0.15 * max(dot(n_view, fill_view), 0.0);
    let half_vec = normalize(vec3<f32>(0.0, 0.0, 1.0) + key_view);
    let spec = 0.08 * pow(max(dot(n_view, half_vec), 0.0), 28.0);
    let shaded_rgb =
        base_color.rgb * (ambient + headlight + key_diff + fill_diff) + vec3<f32>(spec);

    return vec4<f32>(shaded_rgb, clamp(base_color.a, 0.0, 1.0));
}

@fragment
fn fs_point(in: PointVertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(in.quad_uv, in.quad_uv);
    let color = shade_point_fragment(in, r2);
    if (r2 > 1.0 || is_clipped_by_plane(in.world_pos, surface.clip_plane)) {
        discard;
    }
    return color;
}

@fragment
fn fs_point_wboit(in: PointVertexOutput) -> WboitAccumOutput {
    let r2 = dot(in.quad_uv, in.quad_uv);
    let color = shade_point_fragment(in, r2);
    if (r2 > 1.0 || is_clipped_by_plane(in.world_pos, surface.clip_plane) || color.a <= 1e-5) {
        discard;
    }
    let w = wboit_weight(in.clip_position.z, color.a);
    let raw_accum = vec4<f32>(color.rgb * w, w);
    var out: WboitAccumOutput;
    out.accum = vec4<f32>(
        unpack2x16float(pack2x16float(raw_accum.xy)),
        unpack2x16float(pack2x16float(raw_accum.zw)),
    );
    out.reveal = vec4<f32>(color.a, color.a, color.a, color.a);
    return out;
}

@fragment
fn fs_point_pick(in: PointVertexOutput) -> @location(0) vec4<u32> {
    let r2 = dot(in.quad_uv, in.quad_uv);
    if (r2 > 1.0 || is_clipped_by_plane(in.world_pos, surface.clip_plane)) {
        discard;
    }

    var raw_scalar = in.scalar;
    if (surface.coloring_mode == 2u) {
        raw_scalar = cell_scalars[in.cell_index];
    }
    return vec4<u32>(
        surface.pick_info.x,
        in.cell_index,
        bitcast<u32>(in.clip_position.z),
        bitcast<u32>(raw_scalar),
    );
}

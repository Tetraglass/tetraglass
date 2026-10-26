// Shared uniform structs and lighting/colormap helpers for Tetraglass WGSL shaders.

struct FrameUniforms {
    view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    key_dir: vec4<f32>,
    fill_dir: vec4<f32>,
    viewport_size: vec4<f32>,
};

struct SurfaceUniforms {
    range_min: f32,
    range_max: f32,
    coloring_mode: u32,
    use_log_scale: u32,
    solid_color: vec4<f32>,
    edge_color: vec4<f32>,
    pick_info: vec4<u32>,
    clip_plane: vec4<f32>,
    point_params: vec4<f32>,
};

fn is_clipped_by_plane(world_pos: vec3<f32>, clip_plane: vec4<f32>) -> bool {
    let n_sq = dot(clip_plane.xyz, clip_plane.xyz);
    if (n_sq > 1e-12) {
        return (dot(world_pos, clip_plane.xyz) + clip_plane.w) < -1e-6;
    }
    return false;
}

fn normalize_scalar_to_lut_u(
    value: f32,
    range_min: f32,
    range_max: f32,
    use_log_scale: u32,
) -> f32 {
    var v = value;
    var lo = range_min;
    var hi = range_max;
    if (use_log_scale != 0u) {
        let eps = 1e-30;
        v = log(max(v, eps));
        lo = log(max(lo, eps));
        hi = log(max(hi, eps));
    }
    let span = hi - lo;
    var t = 0.5;
    if (abs(span) > 1e-20) {
        t = clamp((v - lo) / span, 0.0, 1.0);
    }
    // Map [0, 1] onto texel centers of a 256-texel 1D LUT texture.
    return (0.5 + t * 255.0) / 256.0;
}

fn shade_three_point(
    albedo: vec3<f32>,
    world_pos: vec3<f32>,
    raw_normal: vec3<f32>,
    frame: FrameUniforms,
) -> vec3<f32> {
    let view_dir = normalize(frame.eye_pos.xyz - world_pos);
    let n_len = length(raw_normal);
    var n = vec3<f32>(0.0, 0.0, 1.0);
    if (n_len > 1e-6) {
        n = raw_normal / n_len;
    }
    if (dot(n, view_dir) < 0.0) {
        n = -n;
    }

    let key_dir = normalize(frame.key_dir.xyz);
    let fill_dir = normalize(frame.fill_dir.xyz);

    let ambient = 0.18;
    let headlight = 0.35 * max(dot(n, view_dir), 0.0);
    let key_diff = 0.42 * max(dot(n, key_dir), 0.0);
    let fill_diff = 0.18 * max(dot(n, fill_dir), 0.0);

    let half_vec = normalize(view_dir + key_dir);
    let spec = 0.06 * pow(max(dot(n, half_vec), 0.0), 28.0);

    return albedo * (ambient + headlight + key_diff + fill_diff) + vec3<f32>(spec);
}

// Weighted Blended Order-Independent Transparency (McGuire & Bavoil 2013) depth-and-alpha weight
// adapted to WebGPU's tight [0, 1] NDC depth range.
// Monotonically decreases with normalized depth `ndc_z` and increases with `alpha`, staying well
// within `Rgba16Float` normal range `[6e-5, 65504]` to prevent overflow or underflow.
fn wboit_weight(ndc_z: f32, alpha: f32) -> f32 {
    let z = clamp(ndc_z, 0.0, 1.0);
    let a = clamp(alpha, 1e-3, 1.0);
    let depth_term = 1.0 - 0.95 * z;
    let d3 = depth_term * depth_term * depth_term;
    return a * clamp(10.0 * d3, 1e-2, 300.0);
}

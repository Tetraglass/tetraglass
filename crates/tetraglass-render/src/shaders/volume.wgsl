// Volume ray-marching shader (concatenated after common.wgsl and scene depth binding).

@group(0) @binding(0)
var<uniform> frame: FrameUniforms;

@group(0) @binding(1)
var colormap_tex: texture_1d<f32>;

@group(0) @binding(2)
var colormap_sampler: sampler;

struct VolumeUniforms {
    inv_view_proj: mat4x4<f32>,
    box_min: vec4<f32>,
    box_max: vec4<f32>,
    grid_dims: vec4<f32>,
    scalar_params: vec4<f32>,
    mode_params: vec4<u32>,
    step_params: vec4<f32>,
    solid_color: vec4<f32>,
    clip_plane: vec4<f32>,
    transfer_lut: array<vec4<f32>, 64>,
};

@group(1) @binding(0)
var<uniform> vol: VolumeUniforms;

@group(1) @binding(1)
var volume_tex: texture_3d<f32>;

@group(1) @binding(2)
var volume_sampler: sampler;

struct VolumeVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc_xy: vec2<f32>,
};

@vertex
fn vs_volume(@builtin(vertex_index) vertex_index: u32) -> VolumeVertexOutput {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let xy = pos[vertex_index];
    var out: VolumeVertexOutput;
    out.clip_position = vec4<f32>(xy, 0.0, 1.0);
    out.ndc_xy = xy;
    return out;
}

fn read_transfer_lut(idx: u32) -> f32 {
    let c = min(idx, 255u);
    let vec_idx = c >> 2u;
    let comp_idx = c & 3u;
    let v = vol.transfer_lut[vec_idx];
    return v[comp_idx];
}

fn sample_transfer_lut(t: f32) -> f32 {
    let scaled = clamp(t, 0.0, 1.0) * 255.0;
    let i0 = u32(floor(scaled));
    let i1 = min(i0 + 1u, 255u);
    let frac = scaled - f32(i0);
    return mix(read_transfer_lut(i0), read_transfer_lut(i1), frac);
}

fn normalize_scalar_01(
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
    if (abs(span) > 1e-20) {
        return clamp((v - lo) / span, 0.0, 1.0);
    }
    return 0.5;
}

fn world_to_tex_uvw(p: vec3<f32>, box_min: vec3<f32>, inv_extent: vec3<f32>, dims: vec3<f32>) -> vec3<f32> {
    let uvw = clamp((p - box_min) * inv_extent, vec3<f32>(0.0), vec3<f32>(1.0));
    return (vec3<f32>(0.5) + uvw * max(dims - vec3<f32>(1.0), vec3<f32>(1.0))) / max(dims, vec3<f32>(1.0));
}

@fragment
fn fs_volume(in: VolumeVertexOutput) -> @location(0) vec4<f32> {
    let near_h = vol.inv_view_proj * vec4<f32>(in.ndc_xy, 0.0, 1.0);
    let far_h = vol.inv_view_proj * vec4<f32>(in.ndc_xy, 1.0, 1.0);
    let ray_orig = near_h.xyz / near_h.w;
    let far_p = far_h.xyz / far_h.w;
    let ray_vec = far_p - ray_orig;
    let ray_len = length(ray_vec);
    if (ray_len <= 1e-8) {
        discard;
    }
    let ray_dir = ray_vec / ray_len;

    let safe_dir = select(
        max(abs(ray_dir), vec3<f32>(1e-8)),
        -max(abs(ray_dir), vec3<f32>(1e-8)),
        ray_dir < vec3<f32>(0.0),
    );
    let inv_dir = 1.0 / safe_dir;
    let t0 = (vol.box_min.xyz - ray_orig) * inv_dir;
    let t1 = (vol.box_max.xyz - ray_orig) * inv_dir;
    let tmin_v = min(t0, t1);
    let tmax_v = max(t0, t1);
    let t_enter = max(max(tmin_v.x, tmin_v.y), max(tmin_v.z, 0.0));
    var t_exit = min(min(tmax_v.x, tmax_v.y), tmax_v.z);
    if (t_exit <= t_enter) {
        discard;
    }

    // Terminate rays against the opaque depth buffer.
    let px = vec2<i32>(in.clip_position.xy);
    let scene_depth = load_scene_depth(px);
    if (scene_depth < 1.0) {
        let depth_h = vol.inv_view_proj * vec4<f32>(in.ndc_xy, scene_depth, 1.0);
        let depth_p = depth_h.xyz / depth_h.w;
        let t_depth = dot(depth_p - ray_orig, ray_dir);
        t_exit = min(t_exit, t_depth);
        if (t_exit <= t_enter) {
            discard;
        }
    }

    let box_min = vol.box_min.xyz;
    let box_extent = max(vol.box_max.xyz - box_min, vec3<f32>(1e-6));
    let inv_extent = 1.0 / box_extent;
    let dims = max(vol.grid_dims.xyz, vec3<f32>(2.0));
    let max_steps = u32(clamp(vol.grid_dims.w, 16.0, 512.0));

    let data_min = vol.scalar_params.x;
    let data_scale = vol.scalar_params.y;
    let range_min = vol.scalar_params.z;
    let range_max = vol.scalar_params.w;
    let coloring_mode = vol.mode_params.x;
    let use_log_scale = vol.mode_params.y;
    let opacity_scale = clamp(vol.step_params.x, 0.0, 1.0);
    let nominal_step = max(vol.step_params.y, 1e-5);
    let ref_step = max(vol.step_params.z, 1e-5);

    let span_t = t_exit - t_enter;
    let dt = max(nominal_step, span_t / f32(max_steps));
    let step_ratio = dt / ref_step;

    let du = vec3<f32>(1.0 / dims.x, 0.0, 0.0);
    let dv = vec3<f32>(0.0, 1.0 / dims.y, 0.0);
    let dw = vec3<f32>(0.0, 0.0, 1.0 / dims.z);

    var accum_rgb = vec3<f32>(0.0);
    var accum_alpha = 0.0;
    var t = t_enter;

    for (var step: u32 = 0u; step <= max_steps; step = step + 1u) {
        let rem = t_exit - t;
        if (rem <= 1e-6 * ref_step || accum_alpha >= 0.995) {
            break;
        }
        let seg_dt = min(dt, rem);
        let seg_ratio = seg_dt / ref_step;
        let p = ray_orig + (t + 0.5 * seg_dt) * ray_dir;
        t += seg_dt;

        if (!is_clipped_by_plane(p, vol.clip_plane)) {
            let tex_uvw = world_to_tex_uvw(p, box_min, inv_extent, dims);
            let sample_val = textureSampleLevel(volume_tex, volume_sampler, tex_uvw, 0.0);
            let inside = sample_val.g;
            if (inside > 0.05) {
                let raw_scalar = data_min + sample_val.r * data_scale;
                let norm_t = normalize_scalar_01(raw_scalar, range_min, range_max, use_log_scale);
                let tf_alpha = sample_transfer_lut(norm_t) * opacity_scale * inside;
                let sample_alpha = clamp(1.0 - pow(max(1.0 - tf_alpha, 0.0), seg_ratio), 0.0, 1.0);

                if (sample_alpha > 1e-4) {
                    var base_rgb = vol.solid_color.rgb;
                    if (coloring_mode != 0u) {
                        let lut_u = (0.5 + norm_t * 255.0) / 256.0;
                        base_rgb = textureSampleLevel(colormap_tex, colormap_sampler, lut_u, 0.0).rgb;
                    }

                    // Estimate scalar gradient for gentle 3D shading cues while keeping flat regions bright.
                    let gx = textureSampleLevel(volume_tex, volume_sampler, clamp(tex_uvw + du, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).r
                           - textureSampleLevel(volume_tex, volume_sampler, clamp(tex_uvw - du, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).r;
                    let gy = textureSampleLevel(volume_tex, volume_sampler, clamp(tex_uvw + dv, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).r
                           - textureSampleLevel(volume_tex, volume_sampler, clamp(tex_uvw - dv, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).r;
                    let gz = textureSampleLevel(volume_tex, volume_sampler, clamp(tex_uvw + dw, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).r
                           - textureSampleLevel(volume_tex, volume_sampler, clamp(tex_uvw - dw, vec3<f32>(0.0), vec3<f32>(1.0)), 0.0).r;
                    let grad = vec3<f32>(gx, gy, gz) * inv_extent;
                    let grad_len = length(grad);
                    var shaded_rgb = base_rgb;
                    if (grad_len > 1e-3) {
                        let lit = shade_three_point(base_rgb, p, -grad, frame);
                        let weight = clamp(grad_len * 2.0, 0.0, 0.35);
                        shaded_rgb = mix(base_rgb, lit, weight);
                    }

                    let weight_alpha = (1.0 - accum_alpha) * sample_alpha;
                    accum_rgb += weight_alpha * shaded_rgb;
                    accum_alpha += weight_alpha;
                }
            }
        }
    }

    if (accum_alpha <= 1e-4) {
        discard;
    }
    return vec4<f32>(accum_rgb, accum_alpha);
}

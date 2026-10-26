// Screen-space ambient occlusion (SSAO) raw hemisphere pass and depth-aware bilateral blur pass (M5.2).
//
// Expects `load_scene_depth(px: vec2<i32>) -> f32` to be prepended at `@group(1) @binding(0)`.

struct SsaoUniforms {
    // `[width, height, 1.0 / width, 1.0 / height]`
    viewport_size: vec4<f32>,
    // `[aspect * tan_half_fov, tan_half_fov, 1.0 / (aspect * tan_half_fov), 1.0 / tan_half_fov]`
    proj_scale: vec4<f32>,
    // `[near, far, world_radius, world_bias]`
    clip_and_radius: vec4<f32>,
    // `[strength, scene_radius, 0.0, 0.0]`
    params: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> ssao: SsaoUniforms;

@group(2) @binding(0)
var ssao_raw_tex: texture_2d<f32>;

const SSAO_SAMPLE_COUNT: u32 = 16u;
const GOLDEN_ANGLE: f32 = 2.399963229728653;
const TWO_PI: f32 = 6.283185307179586;

@vertex
fn vs_ssao(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn linearize_depth(d: f32) -> f32 {
    let near = ssao.clip_and_radius.x;
    let far = ssao.clip_and_radius.y;
    return (near * far) / max(far - d * (far - near), 1e-6);
}

fn reconstruct_view_pos(px: vec2<i32>, d: f32) -> vec3<f32> {
    let uv = (vec2<f32>(px) + vec2<f32>(0.5, 0.5)) * ssao.viewport_size.zw;
    let z_dist = linearize_depth(d);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let xy = ndc * ssao.proj_scale.xy * z_dist;
    return vec3<f32>(xy.x, xy.y, -z_dist);
}

fn bayer4x4_angle(px: vec2<i32>) -> f32 {
    let bx = u32(px.x) & 3u;
    let by = u32(px.y) & 3u;
    let b0 = (bx & 1u) ^ (by & 1u);
    let b1 = by & 1u;
    let b2 = ((bx >> 1u) & 1u) ^ ((by >> 1u) & 1u);
    let b3 = (by >> 1u) & 1u;
    let idx = (b0 << 3u) | (b1 << 2u) | (b2 << 1u) | b3;
    return (f32(idx) + 0.5) * (TWO_PI / 16.0);
}

@fragment
fn fs_ssao_raw(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(i32(ssao.viewport_size.x), i32(ssao.viewport_size.y));
    let px = clamp(vec2<i32>(frag_coord.xy), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    let d_c = load_scene_depth(px);
    if (d_c >= 0.99999) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    let p_c = reconstruct_view_pos(px, d_c);

    let px_l = max(px + vec2<i32>(-1, 0), vec2<i32>(0, 0));
    let px_r = min(px + vec2<i32>(1, 0), dims - vec2<i32>(1, 1));
    let px_u = max(px + vec2<i32>(0, -1), vec2<i32>(0, 0));
    let px_d = min(px + vec2<i32>(0, 1), dims - vec2<i32>(1, 1));

    let d_l = load_scene_depth(px_l);
    let d_r = load_scene_depth(px_r);
    let d_u = load_scene_depth(px_u);
    let d_d = load_scene_depth(px_d);

    let p_l = reconstruct_view_pos(px_l, d_l);
    let p_r = reconstruct_view_pos(px_r, d_r);
    let p_u = reconstruct_view_pos(px_u, d_u);
    let p_d = reconstruct_view_pos(px_d, d_d);

    let valid_l = d_l < 0.99999 && px_l.x < px.x;
    let valid_r = d_r < 0.99999 && px_r.x > px.x;
    let valid_u = d_u < 0.99999 && px_u.y < px.y;
    let valid_d = d_d < 0.99999 && px_d.y > px.y;

    var dpdx = vec3<f32>(1.0, 0.0, 0.0);
    if (valid_l && valid_r) {
        if (abs(p_r.z - p_c.z) < abs(p_c.z - p_l.z)) {
            dpdx = p_r - p_c;
        } else {
            dpdx = p_c - p_l;
        }
    } else if (valid_r) {
        dpdx = p_r - p_c;
    } else if (valid_l) {
        dpdx = p_c - p_l;
    }

    var dpdy = vec3<f32>(0.0, 1.0, 0.0);
    if (valid_u && valid_d) {
        if (abs(p_u.z - p_c.z) < abs(p_c.z - p_d.z)) {
            dpdy = p_u - p_c;
        } else {
            dpdy = p_c - p_d;
        }
    } else if (valid_u) {
        dpdy = p_u - p_c;
    } else if (valid_d) {
        dpdy = p_c - p_d;
    }

    let raw_n = cross(dpdx, dpdy);
    let n_len = length(raw_n);
    var n = select(vec3<f32>(0.0, 0.0, 1.0), raw_n / n_len, n_len > 1e-12);
    if (n.z < 0.0) {
        n = -n;
    }

    let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    let tangent = normalize(cross(up_ref, n));
    let bitangent = cross(n, tangent);

    let radius = ssao.clip_and_radius.z;
    let bias = ssao.clip_and_radius.w;
    let strength = ssao.params.x;
    let phi = bayer4x4_angle(px);

    var occlusion = 0.0;
    var weight_sum = 0.0;

    for (var i: u32 = 0u; i < SSAO_SAMPLE_COUNT; i = i + 1u) {
        let t = (f32(i) + 0.5) / f32(SSAO_SAMPLE_COUNT);
        let radial_scale = mix(0.15, 1.0, t * t);
        let cos_theta = sqrt(1.0 - t * 0.85);
        let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
        let theta = f32(i) * GOLDEN_ANGLE + phi;

        let h_local = vec3<f32>(
            cos(theta) * sin_theta,
            sin(theta) * sin_theta,
            cos_theta,
        ) * radial_scale;

        let offset_view = (tangent * h_local.x + bitangent * h_local.y + n * h_local.z) * radius;
        let sample_pos = p_c + offset_view;

        let sample_dist = max(-sample_pos.z, 1e-6);
        let sample_ndc = (sample_pos.xy / sample_dist) * ssao.proj_scale.zw;
        let sample_uv = vec2<f32>(
            0.5 * (sample_ndc.x + 1.0),
            0.5 * (1.0 - sample_ndc.y),
        );

        weight_sum = weight_sum + cos_theta;

        if (sample_uv.x >= 0.0 && sample_uv.x <= 1.0 && sample_uv.y >= 0.0 && sample_uv.y <= 1.0) {
            let sample_px = clamp(
                vec2<i32>(sample_uv * ssao.viewport_size.xy),
                vec2<i32>(0, 0),
                dims - vec2<i32>(1, 1),
            );
            let d_s = load_scene_depth(sample_px);
            if (d_s < 0.99999) {
                let z_scene = -linearize_depth(d_s);
                let dz = z_scene - sample_pos.z;
                if (dz > bias) {
                    let dist_diff = abs(p_c.z - z_scene);
                    let range_check = smoothstep(0.0, 1.0, radius / max(dist_diff, 1e-5));
                    occlusion = occlusion + range_check * cos_theta;
                }
            }
        }
    }

    let norm_occ = occlusion / max(weight_sum, 1e-5);
    let ao = clamp(1.0 - norm_occ * (1.6 * strength), 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}

@fragment
fn fs_ssao_apply(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(i32(ssao.viewport_size.x), i32(ssao.viewport_size.y));
    let px = clamp(vec2<i32>(frag_coord.xy), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    let d_c = load_scene_depth(px);
    if (d_c >= 0.99999) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }

    let z_c = linearize_depth(d_c);
    let depth_sigma = max(ssao.clip_and_radius.z * 0.35, 1e-4);

    var ao_sum = 0.0;
    var w_sum = 0.0;

    for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
        for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
            let q = clamp(px + vec2<i32>(dx, dy), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
            let d_q = load_scene_depth(q);
            if (d_q < 0.99999) {
                let z_q = linearize_depth(d_q);
                let dz = abs(z_q - z_c) / depth_sigma;
                let w_depth = exp(-dz * dz * 2.0);
                let dist2 = f32(dx * dx + dy * dy);
                let w_spatial = exp(-dist2 * 0.25);
                let w = w_depth * w_spatial;
                let ao_q = textureLoad(ssao_raw_tex, q, 0).r;
                ao_sum = ao_sum + ao_q * w;
                w_sum = w_sum + w;
            }
        }
    }

    let ao = select(1.0, ao_sum / w_sum, w_sum > 1e-5);
    return vec4<f32>(ao, ao, ao, 1.0);
}

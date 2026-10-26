// Compute-shader progressive path tracer and fullscreen present pass (M5.3a).
//
// Operates on a pre-order flattened linear SAH BVH (`bvh_nodes`) and triangle buffer (`triangles`)
// in dataset-center-relative `f32` coordinates (`D-014`). Runs on every `wgpu` backend including
// WebGPU (`wasm32-unknown-unknown`) using storage buffers (`read` and `read_write`) with no
// hardware ray-query extensions required (`D-007`, `D-010`).

struct PathTraceUniforms {
    // Camera eye position in dataset-center-relative coordinates (`xyz`), `w = near`.
    eye_and_near: vec4<f32>,
    // Camera basis right vector (`xyz`), `w = far`.
    cam_right_and_far: vec4<f32>,
    // Camera basis up vector (`xyz`), `w = aspect * tan(fov_y / 2)`.
    cam_up_and_scale_x: vec4<f32>,
    // Camera basis back vector (`xyz`, pointing from focal_point to eye), `w = tan(fov_y / 2)`.
    cam_back_and_scale_y: vec4<f32>,
    // Key directional light unit vector (`xyz`, toward light), `w = light_softness`.
    key_dir_and_softness: vec4<f32>,
    // Fill directional light unit vector (`xyz`, toward light), `w = scene_radius`.
    fill_dir_and_radius: vec4<f32>,
    // Background color in linear RGB (`xyz`), `w = 1.0`.
    background_linear: vec4<f32>,
    // `[width, height, prev_samples, samples_this_dispatch]`
    image_and_samples: vec4<u32>,
    // `[max_bounces, num_triangles, num_bvh_nodes, seed_offset]`
    trace_params: vec4<u32>,
};

struct BvhNode {
    aabb_min: vec3<f32>,
    right_or_first: u32,
    aabb_max: vec3<f32>,
    count: u32,
};

struct PathTraceTriangle {
    v0_s0: vec4<f32>,
    e1_s1: vec4<f32>,
    e2_s2: vec4<f32>,
    normal_and_material: vec4<f32>,
};

struct PathTraceMaterial {
    solid_color: vec4<f32>,
    clip_plane: vec4<f32>,
    scalar_range: vec4<f32>,
    // `[coloring_mode, colormap_index, use_log_scale, has_clip_plane]`
    mode_flags: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> params: PathTraceUniforms;

@group(0) @binding(1)
var<storage, read> bvh_nodes: array<BvhNode>;

@group(0) @binding(2)
var<storage, read> triangles: array<PathTraceTriangle>;

@group(0) @binding(3)
var<storage, read> materials: array<PathTraceMaterial>;

// 3 * 256 = 768 linear RGBA entries (`0..256` = Viridis, `256..512` = Cividis, `512..768` = CoolWarm).
@group(0) @binding(4)
var<storage, read> colormap_luts: array<vec4<f32>>;

// Per-pixel progressive accumulation buffer:
// `xyz` = running-mean linear RGB, `w` = primary-hit WebGPU depth `ndc_z in [0, 1]`.
@group(0) @binding(5)
var<storage, read_write> accum_buffer: array<vec4<f32>>;

const PI: f32 = 3.141592653589793;
const TWO_PI: f32 = 6.283185307179586;
const LOG10_E: f32 = 0.4342944819032518;
const MAX_BVH_STACK: u32 = 64u;

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(rng: ptr<function, u32>) -> f32 {
    *rng = pcg_hash(*rng);
    return f32(*rng) * (1.0 / 4294967296.0);
}

// Low-discrepancy R2 quasi-random 2D sequence in `[0, 1)^2` for sub-pixel anti-aliasing.
fn r2_jitter(sample_index: u32, pixel_hash: u32) -> vec2<f32> {
    let idx = f32(sample_index);
    let u0 = f32(pixel_hash & 0xffffu) * (1.0 / 65536.0);
    let v0 = f32((pixel_hash >> 16u) & 0xffffu) * (1.0 / 65536.0);
    let g1 = 0.7548776662466927;
    let g2 = 0.5698402909980532;
    return fract(vec2<f32>(u0 + idx * g1, v0 + idx * g2));
}

fn safe_inv_dir(d: vec3<f32>) -> vec3<f32> {
    let eps = 1e-8;
    let sx = select(-1.0, 1.0, d.x >= 0.0);
    let sy = select(-1.0, 1.0, d.y >= 0.0);
    let sz = select(-1.0, 1.0, d.z >= 0.0);
    return vec3<f32>(
        1.0 / select(d.x, sx * eps, abs(d.x) < eps),
        1.0 / select(d.y, sy * eps, abs(d.y) < eps),
        1.0 / select(d.z, sz * eps, abs(d.z) < eps),
    );
}

fn intersect_aabb(
    origin: vec3<f32>,
    inv_dir: vec3<f32>,
    bmin: vec3<f32>,
    bmax: vec3<f32>,
    t_max: f32,
) -> f32 {
    let t0 = (bmin - origin) * inv_dir;
    let t1 = (bmax - origin) * inv_dir;
    let tsmaller = min(t0, t1);
    let tbigger = max(t0, t1);
    let t_enter = max(max(tsmaller.x, tsmaller.y), max(tsmaller.z, 0.0));
    let t_exit = min(min(tbigger.x, tbigger.y), min(tbigger.z, t_max));
    return select(1e30, t_enter, t_enter <= t_exit);
}

struct HitInfo {
    hit: bool,
    t: f32,
    u: f32,
    v: f32,
    tri_index: u32,
};

fn intersect_bvh(
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_min: f32,
    t_max: f32,
    rng: ptr<function, u32>,
) -> HitInfo {
    var result = HitInfo(false, t_max, 0.0, 0.0, 0u);
    let num_tris = params.trace_params.y;
    let num_nodes = params.trace_params.z;
    if (num_tris == 0u || num_nodes == 0u) {
        return result;
    }

    let inv_dir = safe_inv_dir(dir);
    let root = bvh_nodes[0u];
    if (intersect_aabb(origin, inv_dir, root.aabb_min, root.aabb_max, result.t) >= 1e29) {
        return result;
    }

    var stack: array<u32, 64>;
    var stack_ptr: u32 = 0u;
    var node_idx: u32 = 0u;

    loop {
        let node = bvh_nodes[node_idx];
        if (node.count > 0u) {
            let first = node.right_or_first;
            let end = first + node.count;
            for (var ti: u32 = first; ti < end; ti = ti + 1u) {
                let tri = triangles[ti];
                let e1 = tri.e1_s1.xyz;
                let e2 = tri.e2_s2.xyz;
                let pvec = cross(dir, e2);
                let det = dot(e1, pvec);
                if (abs(det) > 1e-14) {
                    let inv_det = 1.0 / det;
                    let tvec = origin - tri.v0_s0.xyz;
                    let u = dot(tvec, pvec) * inv_det;
                    if (u >= 0.0 && u <= 1.0) {
                        let qvec = cross(tvec, e1);
                        let v = dot(dir, qvec) * inv_det;
                        if (v >= 0.0 && (u + v) <= 1.0) {
                            let t = dot(e2, qvec) * inv_det;
                            if (t > t_min && t < result.t) {
                                let mat_idx = bitcast<u32>(tri.normal_and_material.w);
                                let mat = materials[mat_idx];
                                var valid_hit = true;
                                if (mat.mode_flags.w != 0u) {
                                    let hit_pos = origin + dir * t;
                                    if (dot(hit_pos, mat.clip_plane.xyz) + mat.clip_plane.w < 0.0) {
                                        valid_hit = false;
                                    }
                                }
                                if (valid_hit && mat.solid_color.a < 0.9999) {
                                    if (rand_f32(rng) > mat.solid_color.a) {
                                        valid_hit = false;
                                    }
                                }
                                if (valid_hit) {
                                    result.hit = true;
                                    result.t = t;
                                    result.u = u;
                                    result.v = v;
                                    result.tri_index = ti;
                                }
                            }
                        }
                    }
                }
            }

            if (stack_ptr == 0u) {
                break;
            }
            stack_ptr = stack_ptr - 1u;
            node_idx = stack[stack_ptr];
        } else {
            let left_idx = node_idx + 1u;
            let right_idx = node.right_or_first;
            let left_node = bvh_nodes[left_idx];
            let right_node = bvh_nodes[right_idx];

            let t_left = intersect_aabb(origin, inv_dir, left_node.aabb_min, left_node.aabb_max, result.t);
            let t_right = intersect_aabb(origin, inv_dir, right_node.aabb_min, right_node.aabb_max, result.t);

            let hit_left = t_left < 1e29;
            let hit_right = t_right < 1e29;

            if (hit_left && hit_right) {
                let near_idx = select(right_idx, left_idx, t_left <= t_right);
                let far_idx = select(left_idx, right_idx, t_left <= t_right);
                if (stack_ptr < MAX_BVH_STACK) {
                    stack[stack_ptr] = far_idx;
                    stack_ptr = stack_ptr + 1u;
                }
                node_idx = near_idx;
            } else if (hit_left) {
                node_idx = left_idx;
            } else if (hit_right) {
                node_idx = right_idx;
            } else {
                if (stack_ptr == 0u) {
                    break;
                }
                stack_ptr = stack_ptr - 1u;
                node_idx = stack[stack_ptr];
            }
        }
    }

    return result;
}

fn any_hit_bvh(
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_min: f32,
    t_max: f32,
    rng: ptr<function, u32>,
) -> bool {
    let num_tris = params.trace_params.y;
    let num_nodes = params.trace_params.z;
    if (num_tris == 0u || num_nodes == 0u) {
        return false;
    }

    let inv_dir = safe_inv_dir(dir);
    let root = bvh_nodes[0u];
    if (intersect_aabb(origin, inv_dir, root.aabb_min, root.aabb_max, t_max) >= 1e29) {
        return false;
    }

    var stack: array<u32, 64>;
    var stack_ptr: u32 = 0u;
    var node_idx: u32 = 0u;

    loop {
        let node = bvh_nodes[node_idx];
        if (node.count > 0u) {
            let first = node.right_or_first;
            let end = first + node.count;
            for (var ti: u32 = first; ti < end; ti = ti + 1u) {
                let tri = triangles[ti];
                let e1 = tri.e1_s1.xyz;
                let e2 = tri.e2_s2.xyz;
                let pvec = cross(dir, e2);
                let det = dot(e1, pvec);
                if (abs(det) > 1e-14) {
                    let inv_det = 1.0 / det;
                    let tvec = origin - tri.v0_s0.xyz;
                    let u = dot(tvec, pvec) * inv_det;
                    if (u >= 0.0 && u <= 1.0) {
                        let qvec = cross(tvec, e1);
                        let v = dot(dir, qvec) * inv_det;
                        if (v >= 0.0 && (u + v) <= 1.0) {
                            let t = dot(e2, qvec) * inv_det;
                            if (t > t_min && t < t_max) {
                                let mat_idx = bitcast<u32>(tri.normal_and_material.w);
                                let mat = materials[mat_idx];
                                var valid_hit = true;
                                if (mat.mode_flags.w != 0u) {
                                    let hit_pos = origin + dir * t;
                                    if (dot(hit_pos, mat.clip_plane.xyz) + mat.clip_plane.w < 0.0) {
                                        valid_hit = false;
                                    }
                                }
                                if (valid_hit && mat.solid_color.a < 0.9999) {
                                    if (rand_f32(rng) > mat.solid_color.a) {
                                        valid_hit = false;
                                    }
                                }
                                if (valid_hit) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }

            if (stack_ptr == 0u) {
                break;
            }
            stack_ptr = stack_ptr - 1u;
            node_idx = stack[stack_ptr];
        } else {
            let left_idx = node_idx + 1u;
            let right_idx = node.right_or_first;
            let left_node = bvh_nodes[left_idx];
            let right_node = bvh_nodes[right_idx];

            let t_left = intersect_aabb(origin, inv_dir, left_node.aabb_min, left_node.aabb_max, t_max);
            let t_right = intersect_aabb(origin, inv_dir, right_node.aabb_min, right_node.aabb_max, t_max);

            let hit_left = t_left < 1e29;
            let hit_right = t_right < 1e29;

            if (hit_left && hit_right) {
                let near_idx = select(right_idx, left_idx, t_left <= t_right);
                let far_idx = select(left_idx, right_idx, t_left <= t_right);
                if (stack_ptr < MAX_BVH_STACK) {
                    stack[stack_ptr] = far_idx;
                    stack_ptr = stack_ptr + 1u;
                }
                node_idx = near_idx;
            } else if (hit_left) {
                node_idx = left_idx;
            } else if (hit_right) {
                node_idx = right_idx;
            } else {
                if (stack_ptr == 0u) {
                    break;
                }
                stack_ptr = stack_ptr - 1u;
                node_idx = stack[stack_ptr];
            }
        }
    }

    return false;
}

fn evaluate_surface_albedo(tri: PathTraceTriangle, u: f32, v: f32) -> vec3<f32> {
    let mat_idx = bitcast<u32>(tri.normal_and_material.w);
    let mat = materials[mat_idx];
    let coloring_mode = mat.mode_flags.x;
    if (coloring_mode == 0u) {
        return mat.solid_color.rgb;
    }

    let w0 = 1.0 - u - v;
    let raw_s = w0 * tri.v0_s0.w + u * tri.e1_s1.w + v * tri.e2_s2.w;
    let r_min = mat.scalar_range.x;
    let r_max = mat.scalar_range.y;
    let use_log = mat.mode_flags.z != 0u;

    var t = 0.5;
    if (use_log) {
        let safe_min = max(r_min, 1e-30);
        let safe_max = max(r_max, safe_min);
        let safe_s = max(raw_s, safe_min);
        let log_min = log(safe_min) * LOG10_E;
        let log_max = log(safe_max) * LOG10_E;
        let log_s = log(safe_s) * LOG10_E;
        let span = log_max - log_min;
        if (abs(span) > 1e-20) {
            t = clamp((log_s - log_min) / span, 0.0, 1.0);
        }
    } else {
        let span = r_max - r_min;
        if (abs(span) > 1e-20) {
            t = clamp((raw_s - r_min) / span, 0.0, 1.0);
        }
    }

    let cm_offset = min(mat.mode_flags.y, 2u) * 256u;
    let scaled = t * 255.0;
    let idx0 = u32(floor(scaled));
    let idx1 = min(idx0 + 1u, 255u);
    let frac = scaled - f32(idx0);
    let c0 = colormap_luts[cm_offset + idx0].rgb;
    let c1 = colormap_luts[cm_offset + idx1].rgb;
    return mix(c0, c1, frac);
}

fn build_orthonormal_basis(n: vec3<f32>, tangent: ptr<function, vec3<f32>>, bitangent: ptr<function, vec3<f32>>) {
    let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.9);
    *tangent = normalize(cross(up_ref, n));
    *bitangent = cross(n, *tangent);
}

fn cosine_sample_hemisphere(n: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let phi = TWO_PI * u2;
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(0.0, 1.0 - u1));
    var t: vec3<f32>;
    var b: vec3<f32>;
    build_orthonormal_basis(n, &t, &b);
    return normalize(t * x + b * y + n * z);
}

fn sample_cone_direction(dir: vec3<f32>, softness: f32, u1: f32, u2: f32) -> vec3<f32> {
    if (softness <= 1e-5) {
        return dir;
    }
    let r = softness * sqrt(u1);
    let phi = TWO_PI * u2;
    var t: vec3<f32>;
    var b: vec3<f32>;
    build_orthonormal_basis(dir, &t, &b);
    return normalize(dir + t * (r * cos(phi)) + b * (r * sin(phi)));
}

fn studio_environment_light(dir: vec3<f32>) -> vec3<f32> {
    let cam_up = params.cam_up_and_scale_x.xyz;
    let key_dir = params.key_dir_and_softness.xyz;
    let fill_dir = params.fill_dir_and_radius.xyz;
    let sky_t = 0.5 * (dot(dir, cam_up) + 1.0);
    let dome = mix(vec3<f32>(0.24, 0.25, 0.28), vec3<f32>(0.78, 0.80, 0.84), sky_t);
    let key_lobe = 0.25 * max(dot(dir, key_dir), 0.0);
    let fill_lobe = 0.12 * max(dot(dir, fill_dir), 0.0);
    return dome + vec3<f32>(key_lobe + fill_lobe);
}

@compute @workgroup_size(8, 8, 1)
fn cs_path_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = params.image_and_samples.x;
    let height = params.image_and_samples.y;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let pixel_idx = gid.y * width + gid.x;
    let prev_samples = params.image_and_samples.z;
    let samples_this_dispatch = max(params.image_and_samples.w, 1u);
    let max_bounces = clamp(params.trace_params.x, 1u, 8u);
    let seed_offset = params.trace_params.w;

    let eye = params.eye_and_near.xyz;
    let near = params.eye_and_near.w;
    let cam_right = params.cam_right_and_far.xyz;
    let far = params.cam_right_and_far.w;
    let cam_up = params.cam_up_and_scale_x.xyz;
    let scale_x = params.cam_up_and_scale_x.w;
    let cam_back = params.cam_back_and_scale_y.xyz;
    let scale_y = params.cam_back_and_scale_y.w;

    let key_dir = params.key_dir_and_softness.xyz;
    let light_softness = params.key_dir_and_softness.w;
    let fill_dir = params.fill_dir_and_radius.xyz;
    let scene_radius = max(params.fill_dir_and_radius.w, 1e-3);
    let ray_eps = max(scene_radius * 2e-4, 1e-5);
    let max_ray_dist = max(far * 4.0, scene_radius * 16.0);

    let inv_w = 1.0 / f32(max(width, 1u));
    let inv_h = 1.0 / f32(max(height, 1u));
    let pixel_hash = pcg_hash(pixel_idx ^ (seed_offset * 1664525u + 1013904223u));

    var batch_rgb = vec3<f32>(0.0, 0.0, 0.0);
    var primary_ndc_z = 1.0;

    for (var s: u32 = 0u; s < samples_this_dispatch; s = s + 1u) {
        let sample_index = prev_samples + s;
        var rng = pcg_hash(pixel_hash ^ (sample_index * 747796405u + 2891336453u));

        // First sample (index 0) uses exact pixel center `(0.5, 0.5)` so 1-spp renders and
        // depth-buffer wireframe overlays align cleanly; subsequent samples use low-discrepancy
        // R2 sub-pixel jitter in `[0, 1)^2`.
        let jitter = select(r2_jitter(sample_index, pixel_hash), vec2<f32>(0.5, 0.5), sample_index == 0u);
        let ndc_x = ((f32(gid.x) + jitter.x) * inv_w) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((f32(gid.y) + jitter.y) * inv_h) * 2.0;
        let cam_x = ndc_x * scale_x;
        let cam_y = ndc_y * scale_y;

        var ray_origin = eye;
        var ray_dir = normalize(cam_x * cam_right + cam_y * cam_up - cam_back);

        var throughput = vec3<f32>(1.0, 1.0, 1.0);
        var sample_radiance = vec3<f32>(0.0, 0.0, 0.0);

        for (var bounce: u32 = 0u; bounce < max_bounces; bounce = bounce + 1u) {
            let hit = intersect_bvh(ray_origin, ray_dir, ray_eps, max_ray_dist, &rng);
            if (!hit.hit) {
                if (bounce == 0u) {
                    sample_radiance = sample_radiance + params.background_linear.rgb;
                } else {
                    sample_radiance = sample_radiance + throughput * studio_environment_light(ray_dir);
                }
                break;
            }

            let hit_pos = ray_origin + ray_dir * hit.t;
            if (bounce == 0u && s == 0u) {
                let z_view = max(dot(hit_pos - eye, -cam_back), near);
                let ndc_z = (far - (near * far) / z_view) / max(far - near, 1e-6);
                primary_ndc_z = clamp(ndc_z, 0.0, 1.0);
            }

            let tri = triangles[hit.tri_index];
            var n = normalize(tri.normal_and_material.xyz);
            if (dot(n, ray_dir) > 0.0) {
                n = -n;
            }

            let albedo = evaluate_surface_albedo(tri, hit.u, hit.v);
            let p_offset = hit_pos + n * ray_eps;

            // Base cavity ambient + subtle headlight on primary hit
            var direct_light = albedo * 0.10;
            if (bounce == 0u) {
                let head_ndl = max(dot(n, -ray_dir), 0.0);
                direct_light = direct_light + albedo * (0.10 * head_ndl);
            }

            // Soft-shadowed key light
            let key_sample_dir = sample_cone_direction(
                key_dir,
                light_softness,
                rand_f32(&rng),
                rand_f32(&rng),
            );
            let key_ndl = max(dot(n, key_sample_dir), 0.0);
            if (key_ndl > 0.0) {
                if (!any_hit_bvh(p_offset, key_sample_dir, ray_eps, max_ray_dist, &rng)) {
                    var spec = 0.0;
                    if (bounce == 0u) {
                        let half_vec = normalize(key_sample_dir - ray_dir);
                        spec = 0.10 * pow(max(dot(n, half_vec), 0.0), 32.0);
                    }
                    direct_light = direct_light + albedo * (0.64 * key_ndl) + vec3<f32>(spec);
                }
            }

            // Soft-shadowed fill light
            let fill_sample_dir = sample_cone_direction(
                fill_dir,
                light_softness * 1.35,
                rand_f32(&rng),
                rand_f32(&rng),
            );
            let fill_ndl = max(dot(n, fill_sample_dir), 0.0);
            if (fill_ndl > 0.0) {
                if (!any_hit_bvh(p_offset, fill_sample_dir, ray_eps, max_ray_dist, &rng)) {
                    direct_light = direct_light + albedo * (0.28 * fill_ndl);
                }
            }

            sample_radiance = sample_radiance + throughput * direct_light;

            if (bounce + 1u >= max_bounces) {
                break;
            }

            // Cosine-weighted diffuse hemisphere bounce
            ray_origin = p_offset;
            ray_dir = cosine_sample_hemisphere(n, rand_f32(&rng), rand_f32(&rng));
            throughput = throughput * albedo * 0.45;
        }

        batch_rgb = batch_rgb + sample_radiance;
    }

    let total_samples = prev_samples + samples_this_dispatch;
    if (prev_samples == 0u) {
        let avg_rgb = batch_rgb / f32(samples_this_dispatch);
        accum_buffer[pixel_idx] = vec4<f32>(avg_rgb, primary_ndc_z);
    } else {
        let prev = accum_buffer[pixel_idx];
        let new_avg = (prev.xyz * f32(prev_samples) + batch_rgb) / f32(total_samples);
        accum_buffer[pixel_idx] = vec4<f32>(new_avg, prev.w);
    }
}

// ============================================================================
// Fullscreen present pass (resolves `accum_buffer` into `OffscreenTarget` color & depth)
// ============================================================================

struct PresentUniforms {
    // `[width, height, 0, 0]`
    dims: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> present_params: PresentUniforms;

@group(0) @binding(1)
var<storage, read> present_accum: array<vec4<f32>>;

struct PresentFragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@vertex
fn vs_pathtrace_present(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_pathtrace_present(@builtin(position) frag_coord: vec4<f32>) -> PresentFragmentOutput {
    let width = max(present_params.dims.x, 1u);
    let height = max(present_params.dims.y, 1u);
    let px = clamp(u32(frag_coord.x), 0u, width - 1u);
    let py = clamp(u32(frag_coord.y), 0u, height - 1u);
    let idx = py * width + px;
    let val = present_accum[idx];
    let rgb = clamp(val.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let depth = clamp(val.w, 0.0, 1.0);
    return PresentFragmentOutput(vec4<f32>(rgb, 1.0), depth);
}

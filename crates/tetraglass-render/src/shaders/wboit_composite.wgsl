// Full-screen composite shader for Weighted Blended Order-Independent Transparency (M3.3).

@group(0) @binding(0)
var wboit_accum_tex: texture_2d<f32>;

@group(0) @binding(1)
var wboit_reveal_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index & 2u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(frag_coord.xy);
    let reveal = clamp(textureLoad(wboit_reveal_tex, coords, 0).r, 0.0, 1.0);
    if (reveal >= 0.99999) {
        discard;
    }
    let accum = textureLoad(wboit_accum_tex, coords, 0);
    let avg_rgb = accum.rgb / max(accum.a, 1e-5);
    let total_alpha = clamp(1.0 - reveal, 0.0, 1.0);
    return vec4<f32>(avg_rgb, total_alpha);
}

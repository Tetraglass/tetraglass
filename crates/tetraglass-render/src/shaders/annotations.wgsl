// 2D screen-space annotation overlay shader (concatenated after common.wgsl).
//
// Draws colorbars, tick marks, borders, 3D orientation axes triad shafts/arrowheads (`mode == 0u`),
// and anti-aliased glyph coverage quads from the text atlas (`mode == 1u`).

struct AnnotationUniforms {
    viewport_size: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> overlay: AnnotationUniforms;

@group(0) @binding(1)
var atlas_tex: texture_2d<f32>;

@group(0) @binding(2)
var atlas_sampler: sampler;

struct AnnotationVertexInput {
    @location(0) position_px: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) mode: u32,
};

struct AnnotationVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) mode: u32,
};

@vertex
fn vs_main(in: AnnotationVertexInput) -> AnnotationVertexOutput {
    var out: AnnotationVertexOutput;
    let width = max(overlay.viewport_size.x, 1.0);
    let height = max(overlay.viewport_size.y, 1.0);
    let x_ndc = (in.position_px.x / width) * 2.0 - 1.0;
    let y_ndc = 1.0 - (in.position_px.y / height) * 2.0;
    out.clip_position = vec4<f32>(x_ndc, y_ndc, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.mode = in.mode;
    return out;
}

@fragment
fn fs_main(in: AnnotationVertexOutput) -> @location(0) vec4<f32> {
    let coverage = textureSample(atlas_tex, atlas_sampler, in.uv).r;
    if (in.mode == 1u) {
        return vec4<f32>(in.color.rgb, in.color.a * coverage);
    }
    return in.color;
}

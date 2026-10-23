//! Viewport annotations drawn directly by the `wgpu` renderer (`M1.5`).
//!
//! Per `.agents/rules/rendering.md`, colorbars with tick marks and labels, title text,
//! and the 3D orientation axes triad are rendered into the offscreen target by
//! [`crate::Renderer`] so both headless PNGs and the `egui` widget include them identically.

use bytemuck::{Pod, Zeroable};
use tetraglass_scene::ColorbarPosition;

use crate::camera::Camera;
use crate::colormap::ColormapKind;
use crate::text::{FontRasterizer, RasterizedText};

/// Default number of tick marks and value labels drawn on a colorbar.
pub const DEFAULT_COLORBAR_TICKS: usize = 5;

/// Placement of a colorbar inside the viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColorbarPlacement {
    /// Vertical colorbar along the right edge of the viewport (default).
    #[default]
    Right,
    /// Vertical colorbar along the left edge of the viewport.
    Left,
    /// Horizontal colorbar along the bottom edge of the viewport.
    Bottom,
    /// Horizontal colorbar along the top edge of the viewport.
    Top,
}

impl From<ColorbarPosition> for ColorbarPlacement {
    fn from(pos: ColorbarPosition) -> Self {
        match pos {
            ColorbarPosition::Right => Self::Right,
            ColorbarPosition::Left => Self::Left,
            ColorbarPosition::Bottom => Self::Bottom,
            ColorbarPosition::Top => Self::Top,
        }
    }
}

/// Configuration for a single colorbar legend drawn by the renderer.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorbarAnnotation {
    /// Colormap lookup table displayed across the bar.
    pub colormap: ColormapKind,
    /// Scalar range `[min, max]` mapped across the colorbar.
    pub range: [f64; 2],
    /// Whether tick values are interpolated logarithmically (`log10`).
    pub use_log_scale: bool,
    /// Optional title text displayed above the colorbar.
    pub title: Option<String>,
    /// Edge placement of the colorbar within the viewport.
    pub position: ColorbarPlacement,
    /// Number of tick marks and value labels (`2..=11`, default [`DEFAULT_COLORBAR_TICKS`]).
    pub tick_count: usize,
}

impl Default for ColorbarAnnotation {
    fn default() -> Self {
        Self {
            colormap: ColormapKind::Viridis,
            range: [0.0, 1.0],
            use_log_scale: false,
            title: None,
            position: ColorbarPlacement::Right,
            tick_count: DEFAULT_COLORBAR_TICKS,
        }
    }
}

impl ColorbarAnnotation {
    /// Creates a new [`ColorbarAnnotation`] for `colormap` and `range` on the right edge.
    #[must_use]
    pub fn new(colormap: ColormapKind, range: [f64; 2]) -> Self {
        Self {
            colormap,
            range,
            ..Self::default()
        }
    }

    /// Sets the optional title displayed above the colorbar.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Sets the viewport placement of the colorbar.
    #[must_use]
    pub const fn with_position(mut self, position: ColorbarPlacement) -> Self {
        self.position = position;
        self
    }

    /// Sets whether ticks are spaced logarithmically.
    #[must_use]
    pub const fn with_log_scale(mut self, use_log_scale: bool) -> Self {
        self.use_log_scale = use_log_scale;
        self
    }
}

/// Complete set of 2D/3D viewport annotations drawn on top of the 3D scene.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Annotations {
    /// Optional title text centered near the top of the viewport.
    pub title: Option<String>,
    /// Whether to draw the 3D orientation axes triad in the bottom-left corner.
    pub axes: bool,
    /// Colorbar legends to draw in the viewport.
    pub colorbars: Vec<ColorbarAnnotation>,
}

impl Annotations {
    /// Returns `true` if no title, axes triad, or colorbars are enabled.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.title.as_ref().is_none_or(|t| t.trim().is_empty())
            && !self.axes
            && self.colorbars.is_empty()
    }

    /// Sets the top-centered viewport title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Enables or disables the bottom-left 3D orientation axes triad.
    #[must_use]
    pub const fn with_axes(mut self, axes: bool) -> Self {
        self.axes = axes;
        self
    }

    /// Appends a [`ColorbarAnnotation`] to the viewport annotations.
    #[must_use]
    pub fn with_colorbar(mut self, colorbar: ColorbarAnnotation) -> Self {
        self.colorbars.push(colorbar);
        self
    }
}

/// A computed tick mark position (`normalized_t in 0.0..=1.0`), scalar value, and formatted label.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorbarTick {
    /// Normalized position along the colorbar (`0.0` at minimum, `1.0` at maximum).
    pub normalized_t: f32,
    /// Scalar value at this tick position.
    pub value: f64,
    /// Formatted string label for display next to the tick mark.
    pub label: String,
}

/// Computes `tick_count` (`2..=11`) evenly spaced linear or logarithmic ticks across `range`.
#[must_use]
pub fn compute_colorbar_ticks(
    range: [f64; 2],
    use_log_scale: bool,
    tick_count: usize,
) -> Vec<ColorbarTick> {
    let count = tick_count.clamp(2, 11);
    let lo = range[0];
    let hi = range[1];
    let can_log = use_log_scale && lo > 0.0 && hi > 0.0;
    let log_lo = if can_log { lo.ln() } else { 0.0 };
    let log_hi = if can_log { hi.ln() } else { 0.0 };

    let mut ticks = Vec::with_capacity(count);
    for i in 0..count {
        let t = (i as f64) / ((count - 1) as f64);
        let value = if can_log {
            (log_lo + t * (log_hi - log_lo)).exp()
        } else {
            lo + t * (hi - lo)
        };
        ticks.push(ColorbarTick {
            normalized_t: t as f32,
            value,
            label: format_tick_value(value),
        });
    }
    ticks
}

/// Formats a floating-point tick value into a compact, deterministic string label.
#[must_use]
pub fn format_tick_value(value: f64) -> String {
    if !value.is_finite() {
        return "0".to_string();
    }
    if value.abs() < 1e-12 {
        return "0".to_string();
    }
    let abs_v = value.abs();
    if (1e-3..1e5).contains(&abs_v) {
        let formatted = if abs_v >= 100.0 {
            format!("{value:.1}")
        } else if abs_v >= 1.0 {
            format!("{value:.2}")
        } else {
            format!("{value:.3}")
        };
        trim_decimal_zeros(&formatted)
    } else {
        format!("{value:.2e}")
    }
}

fn trim_decimal_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed == "-0" || trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct AnnotationVertex {
    pub position_px: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct AnnotationUniforms {
    pub viewport_size: [f32; 2],
    pub _pad: [f32; 2],
}

pub(crate) struct BuiltAnnotationBatch {
    pub vertices: Vec<AnnotationVertex>,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub atlas_r8: Vec<u8>,
}

struct PendingLabel {
    bitmap: RasterizedText,
    x_px: f32,
    y_px: f32,
    color_linear: [f32; 4],
}

pub(crate) fn build_annotation_batch(
    annotations: &Annotations,
    camera: &Camera,
    viewport_width: u32,
    viewport_height: u32,
    background_srgb: [f32; 4],
) -> BuiltAnnotationBatch {
    let mut vertices: Vec<AnnotationVertex> = Vec::new();
    let mut labels: Vec<PendingLabel> = Vec::new();
    let font = FontRasterizer::default();

    let w_f = viewport_width.max(1) as f32;
    let h_f = viewport_height.max(1) as f32;

    // Choose high-contrast foreground color based on background luminance.
    let bg_luma =
        0.2126 * background_srgb[0] + 0.7152 * background_srgb[1] + 0.0722 * background_srgb[2];
    let is_light_bg = bg_luma >= 0.5;
    let fg_srgb = if is_light_bg {
        [0.12, 0.13, 0.16, 1.0]
    } else {
        [0.93, 0.94, 0.97, 1.0]
    };
    let fg_linear = srgb_rgba_to_linear(fg_srgb);

    // 1. Top-centered title text
    if let Some(title) = &annotations.title {
        let trimmed = title.trim();
        if !trimmed.is_empty() {
            let title_size = (w_f * 0.024).clamp(14.0, 22.0);
            let bmp = font.rasterize(trimmed, title_size);
            let x = ((w_f - bmp.width as f32) * 0.5).max(8.0).round();
            let y = 14.0f32;
            labels.push(PendingLabel {
                bitmap: bmp,
                x_px: x,
                y_px: y,
                color_linear: fg_linear,
            });
        }
    }

    // 2. Colorbars with tick marks, tick labels, and optional title
    for cb in &annotations.colorbars {
        append_colorbar(
            &mut vertices,
            &mut labels,
            &font,
            cb,
            [w_f, h_f],
            fg_linear,
            annotations.title.is_some(),
        );
    }

    // 3. Bottom-left 3D orientation axes triad
    if annotations.axes {
        append_axes_triad(&mut vertices, &mut labels, &font, camera, h_f, is_light_bg);
    }

    // Pack all rasterized text bitmaps into a single 2D R8Unorm atlas texture.
    let (atlas_width, atlas_height, atlas_r8) = pack_labels_and_emit_quads(&mut vertices, labels);

    BuiltAnnotationBatch {
        vertices,
        atlas_width,
        atlas_height,
        atlas_r8,
    }
}

fn append_colorbar(
    vertices: &mut Vec<AnnotationVertex>,
    labels: &mut Vec<PendingLabel>,
    font: &FontRasterizer,
    cb: &ColorbarAnnotation,
    [viewport_w, viewport_h]: [f32; 2],
    fg_linear: [f32; 4],
    has_scene_title: bool,
) {
    let ticks = compute_colorbar_ticks(cb.range, cb.use_log_scale, cb.tick_count);
    let tick_font_size = (viewport_h * 0.022).clamp(11.0, 14.0);
    let title_font_size = (tick_font_size + 1.5).clamp(12.0, 16.0);

    let rasterized_ticks: Vec<(f32, RasterizedText)> = ticks
        .iter()
        .map(|t| (t.normalized_t, font.rasterize(&t.label, tick_font_size)))
        .collect();

    let max_label_w = rasterized_ticks
        .iter()
        .map(|(_, b)| b.width as f32)
        .fold(0.0f32, f32::max);
    let max_label_h = rasterized_ticks
        .iter()
        .map(|(_, b)| b.height as f32)
        .fold(12.0f32, f32::max);

    let title_bmp = cb
        .title
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| font.rasterize(s, title_font_size));

    let tick_len = 5.0f32;
    let tick_gap = 4.0f32;
    let border_thickness = 1.5f32;
    let num_ramp_segments = 64u32;

    match cb.position {
        ColorbarPlacement::Right | ColorbarPlacement::Left => {
            let bar_w = 16.0f32;
            let bar_h = (viewport_h * 0.44).clamp(72.0, 280.0);
            let cy = viewport_h * 0.52;
            let bar_y0 = (cy - bar_h * 0.5).round();
            let bar_y1 = (bar_y0 + bar_h).round();

            let bar_x0 = match cb.position {
                ColorbarPlacement::Right => {
                    (viewport_w - 18.0 - max_label_w - tick_gap - tick_len - bar_w)
                        .max(16.0)
                        .round()
                }
                _ => 20.0f32,
            };
            let bar_x1 = bar_x0 + bar_w;

            // Vertical gradient ramp (`t = 0` at bottom `bar_y1`, `t = 1` at top `bar_y0`)
            for seg in 0..num_ramp_segments {
                let t0 = (seg as f32) / (num_ramp_segments as f32);
                let t1 = ((seg + 1) as f32) / (num_ramp_segments as f32);
                let y0 = bar_y1 - t1 * (bar_y1 - bar_y0);
                let y1 = bar_y1 - t0 * (bar_y1 - bar_y0);
                let c_top = srgb_u8_to_linear(cb.colormap.sample_srgb(t1));
                let c_bot = srgb_u8_to_linear(cb.colormap.sample_srgb(t0));
                push_vertical_gradient_quad(vertices, bar_x0, y0, bar_x1, y1, c_top, c_bot);
            }

            // Crisp outline around the bar
            push_rect_outline(
                vertices,
                bar_x0,
                bar_y0,
                bar_x1,
                bar_y1,
                border_thickness,
                fg_linear,
            );

            // Tick marks and tick labels to the right of the bar
            for (t, bmp) in rasterized_ticks {
                let y = (bar_y1 - t * (bar_y1 - bar_y0)).round();
                push_solid_rect(
                    vertices,
                    bar_x1,
                    y - 0.75,
                    bar_x1 + tick_len,
                    y + 0.75,
                    fg_linear,
                );
                let lx = (bar_x1 + tick_len + tick_gap).round();
                let ly = (y - (bmp.height as f32) * 0.5).round();
                labels.push(PendingLabel {
                    bitmap: bmp,
                    x_px: lx,
                    y_px: ly,
                    color_linear: fg_linear,
                });
            }

            // Optional colorbar title above the bar
            if let Some(tb) = title_bmp {
                let total_block_w = bar_w + tick_len + tick_gap + max_label_w;
                let tx = (bar_x0 + (total_block_w - tb.width as f32) * 0.5)
                    .clamp(8.0, (viewport_w - tb.width as f32 - 8.0).max(8.0))
                    .round();
                let ty = (bar_y0 - tb.height as f32 - 8.0).max(6.0).round();
                labels.push(PendingLabel {
                    bitmap: tb,
                    x_px: tx,
                    y_px: ty,
                    color_linear: fg_linear,
                });
            }
        }
        ColorbarPlacement::Bottom | ColorbarPlacement::Top => {
            let bar_h = 14.0f32;
            let bar_w = (viewport_w * 0.42).clamp(96.0, 340.0);
            let cx = viewport_w * 0.5;
            let bar_x0 = (cx - bar_w * 0.5).round();
            let bar_x1 = (bar_x0 + bar_w).round();

            let bar_y0 = match cb.position {
                ColorbarPlacement::Bottom => (viewport_h - 22.0 - max_label_h - tick_len - bar_h)
                    .max(16.0)
                    .round(),
                _ => {
                    if has_scene_title {
                        54.0f32
                    } else {
                        28.0f32
                    }
                }
            };
            let bar_y1 = bar_y0 + bar_h;

            // Horizontal gradient ramp (`t = 0` at left `bar_x0`, `t = 1` at right `bar_x1`)
            for seg in 0..num_ramp_segments {
                let t0 = (seg as f32) / (num_ramp_segments as f32);
                let t1 = ((seg + 1) as f32) / (num_ramp_segments as f32);
                let x0 = bar_x0 + t0 * (bar_x1 - bar_x0);
                let x1 = bar_x0 + t1 * (bar_x1 - bar_x0);
                let c_left = srgb_u8_to_linear(cb.colormap.sample_srgb(t0));
                let c_right = srgb_u8_to_linear(cb.colormap.sample_srgb(t1));
                push_horizontal_gradient_quad(vertices, x0, bar_y0, x1, bar_y1, c_left, c_right);
            }

            push_rect_outline(
                vertices,
                bar_x0,
                bar_y0,
                bar_x1,
                bar_y1,
                border_thickness,
                fg_linear,
            );

            for (t, bmp) in rasterized_ticks {
                let x = (bar_x0 + t * (bar_x1 - bar_x0)).round();
                push_solid_rect(
                    vertices,
                    x - 0.75,
                    bar_y1,
                    x + 0.75,
                    bar_y1 + tick_len,
                    fg_linear,
                );
                let lx = (x - (bmp.width as f32) * 0.5).round();
                let ly = (bar_y1 + tick_len + 3.0).round();
                labels.push(PendingLabel {
                    bitmap: bmp,
                    x_px: lx,
                    y_px: ly,
                    color_linear: fg_linear,
                });
            }

            if let Some(tb) = title_bmp {
                let tx = (cx - (tb.width as f32) * 0.5).max(8.0).round();
                let ty = (bar_y0 - tb.height as f32 - 5.0).max(4.0).round();
                labels.push(PendingLabel {
                    bitmap: tb,
                    x_px: tx,
                    y_px: ty,
                    color_linear: fg_linear,
                });
            }
        }
    }
}

fn append_axes_triad(
    vertices: &mut Vec<AnnotationVertex>,
    labels: &mut Vec<PendingLabel>,
    font: &FontRasterizer,
    camera: &Camera,
    viewport_h: f32,
    is_light_bg: bool,
) {
    let view = camera.view_matrix();
    let cx = 54.0f32;
    let cy = (viewport_h - 54.0).max(48.0);
    let axis_len = (viewport_h * 0.058).clamp(24.0, 38.0);
    let label_dist = axis_len + 11.0;

    let axes_spec: [(&str, [f32; 3], [f32; 4]); 3] = [
        (
            "X",
            [view[0][0], view[0][1], view[0][2]],
            if is_light_bg {
                [0.84, 0.16, 0.18, 1.0]
            } else {
                [0.95, 0.32, 0.34, 1.0]
            },
        ),
        (
            "Y",
            [view[1][0], view[1][1], view[1][2]],
            if is_light_bg {
                [0.14, 0.62, 0.22, 1.0]
            } else {
                [0.30, 0.85, 0.38, 1.0]
            },
        ),
        (
            "Z",
            [view[2][0], view[2][1], view[2][2]],
            if is_light_bg {
                [0.16, 0.38, 0.86, 1.0]
            } else {
                [0.38, 0.62, 0.98, 1.0]
            },
        ),
    ];

    // Sort back-to-front by camera-space Z (`view_z`) so nearer axes draw over farther axes.
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| {
        axes_spec[a].1[2]
            .partial_cmp(&axes_spec[b].1[2])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let label_font_size = 13.0f32;
    for idx in order {
        let (name, dir, color_srgb) = axes_spec[idx];
        let color_linear = srgb_rgba_to_linear(color_srgb);

        let dx = dir[0] * axis_len;
        let dy = -dir[1] * axis_len;
        let len_2d = (dx * dx + dy * dy).sqrt();

        if len_2d > 1.5 {
            let ux = dx / len_2d;
            let uy = dy / len_2d;
            let head_len = 8.0f32.min(len_2d * 0.45);
            let shaft_end_x = cx + dx - ux * (head_len * 0.7);
            let shaft_end_y = cy + dy - uy * (head_len * 0.7);
            push_thick_line(
                vertices,
                cx,
                cy,
                shaft_end_x,
                shaft_end_y,
                2.4,
                color_linear,
            );

            // Arrowhead triangle at `(cx + dx, cy + dy)`
            let tip_x = cx + dx;
            let tip_y = cy + dy;
            let base_x = tip_x - ux * head_len;
            let base_y = tip_y - uy * head_len;
            let px = -uy * 3.6;
            let py = ux * 3.6;
            push_solid_triangle(
                vertices,
                [tip_x, tip_y],
                [base_x + px, base_y + py],
                [base_x - px, base_y - py],
                color_linear,
            );
        }

        let bmp = font.rasterize(name, label_font_size);
        let lx = (cx + dir[0] * label_dist - (bmp.width as f32) * 0.5).round();
        let ly = (cy - dir[1] * label_dist - (bmp.height as f32) * 0.5).round();
        labels.push(PendingLabel {
            bitmap: bmp,
            x_px: lx,
            y_px: ly,
            color_linear,
        });
    }

    // Center hub disc
    let hub_color = if is_light_bg {
        srgb_rgba_to_linear([0.22, 0.24, 0.28, 1.0])
    } else {
        srgb_rgba_to_linear([0.85, 0.87, 0.90, 1.0])
    };
    push_solid_rect(vertices, cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5, hub_color);
}

fn pack_labels_and_emit_quads(
    vertices: &mut Vec<AnnotationVertex>,
    labels: Vec<PendingLabel>,
) -> (u32, u32, Vec<u8>) {
    if labels.is_empty() {
        return (1, 1, vec![0u8]);
    }

    let pad = 2u32;
    let max_row_width = 1024u32;
    let mut atlas_w = 1u32;
    let mut cursor_x = pad;
    let mut cursor_y = pad;
    let mut row_h = 0u32;

    let mut placements: Vec<(u32, u32)> = Vec::with_capacity(labels.len());
    for item in &labels {
        let bw = item.bitmap.width;
        let bh = item.bitmap.height;
        if cursor_x + bw + pad > max_row_width && cursor_x > pad {
            cursor_x = pad;
            cursor_y += row_h + pad;
            row_h = 0;
        }
        placements.push((cursor_x, cursor_y));
        cursor_x += bw + pad;
        row_h = row_h.max(bh);
        atlas_w = atlas_w.max(cursor_x);
    }
    let atlas_h = (cursor_y + row_h + pad).max(1);
    let mut atlas = vec![0u8; (atlas_w * atlas_h) as usize];

    let inv_w = 1.0 / (atlas_w as f32);
    let inv_h = 1.0 / (atlas_h as f32);

    for (item, (ax, ay)) in labels.into_iter().zip(placements) {
        let bw = item.bitmap.width;
        let bh = item.bitmap.height;
        for row in 0..bh {
            let src_start = (row * bw) as usize;
            let src_end = src_start + (bw as usize);
            let dst_start = ((ay + row) * atlas_w + ax) as usize;
            let dst_end = dst_start + (bw as usize);
            atlas[dst_start..dst_end].copy_from_slice(&item.bitmap.coverage[src_start..src_end]);
        }

        let x0 = item.x_px;
        let y0 = item.y_px;
        let x1 = x0 + (bw as f32);
        let y1 = y0 + (bh as f32);

        let u0 = (ax as f32) * inv_w;
        let v0 = (ay as f32) * inv_h;
        let u1 = ((ax + bw) as f32) * inv_w;
        let v1 = ((ay + bh) as f32) * inv_h;

        let c = item.color_linear;
        let v_tl = AnnotationVertex {
            position_px: [x0, y0],
            uv: [u0, v0],
            color: c,
            mode: 1,
        };
        let v_bl = AnnotationVertex {
            position_px: [x0, y1],
            uv: [u0, v1],
            color: c,
            mode: 1,
        };
        let v_br = AnnotationVertex {
            position_px: [x1, y1],
            uv: [u1, v1],
            color: c,
            mode: 1,
        };
        let v_tr = AnnotationVertex {
            position_px: [x1, y0],
            uv: [u1, v0],
            color: c,
            mode: 1,
        };
        vertices.extend_from_slice(&[v_tl, v_bl, v_br, v_tl, v_br, v_tr]);
    }

    (atlas_w, atlas_h, atlas)
}

fn push_solid_triangle(
    vertices: &mut Vec<AnnotationVertex>,
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    color: [f32; 4],
) {
    for p in [p0, p1, p2] {
        vertices.push(AnnotationVertex {
            position_px: p,
            uv: [0.0, 0.0],
            color,
            mode: 0,
        });
    }
}

fn push_solid_rect(
    vertices: &mut Vec<AnnotationVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: [f32; 4],
) {
    push_vertical_gradient_quad(vertices, x0, y0, x1, y1, color, color);
}

fn push_vertical_gradient_quad(
    vertices: &mut Vec<AnnotationVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color_top: [f32; 4],
    color_bottom: [f32; 4],
) {
    let v_tl = AnnotationVertex {
        position_px: [x0, y0],
        uv: [0.0, 0.0],
        color: color_top,
        mode: 0,
    };
    let v_bl = AnnotationVertex {
        position_px: [x0, y1],
        uv: [0.0, 0.0],
        color: color_bottom,
        mode: 0,
    };
    let v_br = AnnotationVertex {
        position_px: [x1, y1],
        uv: [0.0, 0.0],
        color: color_bottom,
        mode: 0,
    };
    let v_tr = AnnotationVertex {
        position_px: [x1, y0],
        uv: [0.0, 0.0],
        color: color_top,
        mode: 0,
    };
    vertices.extend_from_slice(&[v_tl, v_bl, v_br, v_tl, v_br, v_tr]);
}

fn push_horizontal_gradient_quad(
    vertices: &mut Vec<AnnotationVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color_left: [f32; 4],
    color_right: [f32; 4],
) {
    let v_tl = AnnotationVertex {
        position_px: [x0, y0],
        uv: [0.0, 0.0],
        color: color_left,
        mode: 0,
    };
    let v_bl = AnnotationVertex {
        position_px: [x0, y1],
        uv: [0.0, 0.0],
        color: color_left,
        mode: 0,
    };
    let v_br = AnnotationVertex {
        position_px: [x1, y1],
        uv: [0.0, 0.0],
        color: color_right,
        mode: 0,
    };
    let v_tr = AnnotationVertex {
        position_px: [x1, y0],
        uv: [0.0, 0.0],
        color: color_right,
        mode: 0,
    };
    vertices.extend_from_slice(&[v_tl, v_bl, v_br, v_tl, v_br, v_tr]);
}

fn push_rect_outline(
    vertices: &mut Vec<AnnotationVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    thickness: f32,
    color: [f32; 4],
) {
    let t = thickness * 0.5;
    // Top
    push_solid_rect(vertices, x0 - t, y0 - t, x1 + t, y0 + t, color);
    // Bottom
    push_solid_rect(vertices, x0 - t, y1 - t, x1 + t, y1 + t, color);
    // Left
    push_solid_rect(vertices, x0 - t, y0 + t, x0 + t, y1 - t, color);
    // Right
    push_solid_rect(vertices, x1 - t, y0 + t, x1 + t, y1 - t, color);
}

fn push_thick_line(
    vertices: &mut Vec<AnnotationVertex>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    thickness: f32,
    color: [f32; 4],
) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-4 {
        return;
    }
    let nx = -dy / len * (thickness * 0.5);
    let ny = dx / len * (thickness * 0.5);
    let p0 = [x0 + nx, y0 + ny];
    let p1 = [x0 - nx, y0 - ny];
    let p2 = [x1 - nx, y1 - ny];
    let p3 = [x1 + nx, y1 + ny];
    push_solid_triangle(vertices, p0, p1, p2, color);
    push_solid_triangle(vertices, p0, p2, p3, color);
}

fn srgb_channel_to_linear(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_rgba_to_linear(rgba: [f32; 4]) -> [f32; 4] {
    [
        srgb_channel_to_linear(rgba[0]),
        srgb_channel_to_linear(rgba[1]),
        srgb_channel_to_linear(rgba[2]),
        rgba[3].clamp(0.0, 1.0),
    ]
}

fn srgb_u8_to_linear(rgba: [u8; 4]) -> [f32; 4] {
    srgb_rgba_to_linear([
        f32::from(rgba[0]) / 255.0,
        f32::from(rgba[1]) / 255.0,
        f32::from(rgba[2]) / 255.0,
        f32::from(rgba[3]) / 255.0,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_formatting_handles_zero_decimals_and_scientific() {
        assert_eq!(format_tick_value(0.0), "0");
        assert_eq!(format_tick_value(-0.0), "0");
        assert_eq!(format_tick_value(1.0), "1");
        assert_eq!(format_tick_value(1.25), "1.25");
        assert_eq!(format_tick_value(-3.5), "-3.5");
        assert_eq!(format_tick_value(250.0), "250");
        assert_eq!(format_tick_value(1.0e-5), "1.00e-5");
        assert_eq!(format_tick_value(2.5e6), "2.50e6");
    }

    #[test]
    fn linear_and_logarithmic_ticks_are_computed_correctly() {
        let lin = compute_colorbar_ticks([-2.0, 2.0], false, 5);
        assert_eq!(lin.len(), 5);
        assert_eq!(lin[0].label, "-2");
        assert_eq!(lin[1].label, "-1");
        assert_eq!(lin[2].label, "0");
        assert_eq!(lin[3].label, "1");
        assert_eq!(lin[4].label, "2");

        let log_ticks = compute_colorbar_ticks([1.0, 10000.0], true, 5);
        assert_eq!(log_ticks.len(), 5);
        assert_eq!(log_ticks[0].label, "1");
        assert_eq!(log_ticks[1].label, "10");
        assert_eq!(log_ticks[2].label, "100");
        assert_eq!(log_ticks[3].label, "1000");
        assert_eq!(log_ticks[4].label, "10000");
    }

    #[test]
    fn annotation_batch_generates_geometry_and_atlas_for_all_elements() {
        let annotations = Annotations::default()
            .with_title("Mixed Linear Mesh")
            .with_axes(true)
            .with_colorbar(
                ColorbarAnnotation::new(ColormapKind::Viridis, [0.0, 10.0])
                    .with_title("linear")
                    .with_position(ColorbarPlacement::Right),
            );
        let batch = build_annotation_batch(
            &annotations,
            &Camera::default(),
            800,
            600,
            [1.0, 1.0, 1.0, 1.0],
        );
        assert!(!batch.vertices.is_empty());
        assert_eq!(batch.vertices.len() % 3, 0);
        assert!(batch.atlas_width > 20);
        assert!(batch.atlas_height > 10);
        assert!(batch.atlas_r8.iter().any(|&a| a > 128));
    }
}

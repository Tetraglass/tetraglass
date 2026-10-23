//! Pure-Rust font loading and text coverage rasterization (`M1.5`).
//!
//! Uses Google Fonts' [`skrifa`] OpenType outline parser and Linebender's [`vello_cpu`] 2D path
//! rasterizer with the embedded fonts from [`epaint_default_fonts`] (`Hack-Regular.ttf` and
//! `Ubuntu-Light.ttf`). All three crates are pure Rust, `wasm32`-compatible, and carry
//! allowlisted permissive licenses (`MIT OR Apache-2.0`, `OFL-1.1`, `Ubuntu-font-1.0`).

use skrifa::MetadataProvider as _;
use skrifa::instance::{LocationRef, Size};
use skrifa::outline::{DrawSettings, OutlinePen};
use vello_cpu::{color, kurbo};

/// Built-in embedded font faces available to the renderer without filesystem access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EmbeddedFont {
    /// `Hack-Regular.ttf` — regular-weight typeface with crisp strokes and tabular numerals.
    #[default]
    HackRegular,
    /// `Ubuntu-Light.ttf` — light-weight proportional sans-serif typeface.
    UbuntuLight,
}

impl EmbeddedFont {
    /// Returns the raw TrueType font bytes embedded in the binary.
    #[must_use]
    pub const fn bytes(self) -> &'static [u8] {
        match self {
            Self::HackRegular => epaint_default_fonts::HACK_REGULAR,
            Self::UbuntuLight => epaint_default_fonts::UBUNTU_LIGHT,
        }
    }
}

/// Measured pixel dimensions and vertical metrics of a single-line text string.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextMetrics {
    /// Total horizontal advance width in pixels.
    pub width_px: f32,
    /// Total line height (`ascent - descent`) in pixels.
    pub height_px: f32,
    /// Distance from the top of the line box to the baseline in pixels (`> 0`).
    pub ascent_px: f32,
    /// Distance from the baseline to the bottom of the line box in pixels (`<= 0`).
    pub descent_px: f32,
}

/// A single-line string rasterized into an 8-bit alpha coverage mask (`0..=255`).
#[derive(Debug, Clone, PartialEq)]
pub struct RasterizedText {
    /// Bitmap width in pixels (`>= 1`).
    pub width: u32,
    /// Bitmap height in pixels (`>= 1`).
    pub height: u32,
    /// Baseline Y coordinate in pixels relative to the top of the bitmap (`y = 0`).
    pub baseline_y: f32,
    /// Tightly-packed row-major alpha coverage values (`width * height` bytes).
    pub coverage: Vec<u8>,
}

/// Pure-Rust CPU font measurer and glyph outline rasterizer.
#[derive(Debug, Clone, Copy)]
pub struct FontRasterizer {
    font_bytes: &'static [u8],
}

impl Default for FontRasterizer {
    fn default() -> Self {
        Self::new(EmbeddedFont::default())
    }
}

impl FontRasterizer {
    /// Creates a [`FontRasterizer`] for the given [`EmbeddedFont`].
    #[must_use]
    pub const fn new(font: EmbeddedFont) -> Self {
        Self {
            font_bytes: font.bytes(),
        }
    }

    /// Measures the pixel width, height, ascent, and descent of a single-line string `text`
    /// at `font_size_px`.
    #[must_use]
    pub fn measure(&self, text: &str, font_size_px: f32) -> TextMetrics {
        let font_size_px = font_size_px.clamp(4.0, 256.0);
        let Ok(font) = skrifa::FontRef::from_index(self.font_bytes, 0) else {
            return TextMetrics {
                width_px: 0.0,
                height_px: font_size_px,
                ascent_px: font_size_px * 0.8,
                descent_px: -font_size_px * 0.2,
            };
        };

        let size = Size::new(font_size_px);
        let location = LocationRef::default();
        let metrics = font.metrics(size, location);
        let glyph_metrics = font.glyph_metrics(size, location);
        let charmap = font.charmap();

        let fallback_gid = charmap.map('?');
        let mut width_px = 0.0f32;
        for ch in text.chars() {
            if ch == '\n' || ch == '\r' {
                continue;
            }
            if let Some(gid) = charmap.map(ch).or(fallback_gid) {
                width_px += glyph_metrics
                    .advance_width(gid)
                    .unwrap_or(font_size_px * 0.5);
            }
        }

        let ascent_px = metrics.ascent.max(font_size_px * 0.75);
        let descent_px = metrics.descent.min(-font_size_px * 0.2);
        let height_px = (ascent_px - descent_px).max(font_size_px);

        TextMetrics {
            width_px,
            height_px,
            ascent_px,
            descent_px,
        }
    }

    /// Rasterizes a single-line string `text` at `font_size_px` into an 8-bit coverage bitmap.
    #[must_use]
    pub fn rasterize(&self, text: &str, font_size_px: f32) -> RasterizedText {
        let font_size_px = font_size_px.clamp(4.0, 256.0);
        let measured = self.measure(text, font_size_px);

        let pad_x = 2.0f32;
        let pad_y = 2.0f32;
        let width = ((measured.width_px + 2.0 * pad_x).ceil() as u32).clamp(1, 2048);
        let height = ((measured.height_px + 2.0 * pad_y).ceil() as u32).clamp(1, 512);
        let baseline_y = pad_y + measured.ascent_px;

        if text.trim().is_empty() {
            return RasterizedText {
                width,
                height,
                baseline_y,
                coverage: vec![0u8; (width * height) as usize],
            };
        }

        let Ok(font) = skrifa::FontRef::from_index(self.font_bytes, 0) else {
            return RasterizedText {
                width,
                height,
                baseline_y,
                coverage: vec![0u8; (width * height) as usize],
            };
        };

        let size = Size::new(font_size_px);
        let location = LocationRef::default();
        let glyph_metrics = font.glyph_metrics(size, location);
        let outlines = font.outline_glyphs();
        let charmap = font.charmap();
        let fallback_gid = charmap.map('?');

        let mut path = kurbo::BezPath::new();
        let mut cursor_x = f64::from(pad_x);
        let base_y = f64::from(baseline_y);

        for ch in text.chars() {
            if ch == '\n' || ch == '\r' {
                continue;
            }
            let Some(gid) = charmap.map(ch).or(fallback_gid) else {
                continue;
            };
            if let Some(outline) = outlines.get(gid) {
                let mut pen = PathPen {
                    path: &mut path,
                    x_offset: cursor_x,
                    baseline_y: base_y,
                };
                let settings = DrawSettings::unhinted(size, location);
                let _ = outline.draw(settings, &mut pen);
            }
            let adv = glyph_metrics
                .advance_width(gid)
                .unwrap_or(font_size_px * 0.5);
            cursor_x += f64::from(adv);
        }

        let width_u16 = width as u16;
        let height_u16 = height as u16;
        let mut ctx = vello_cpu::RenderContext::new(width_u16, height_u16);
        ctx.set_paint(color::OpaqueColor::<color::Srgb>::WHITE);
        ctx.fill_path(&path);

        let mut pixmap = vello_cpu::Pixmap::new(width_u16, height_u16);
        let mut resources = vello_cpu::Resources::new();
        ctx.render(&mut pixmap, &mut resources);

        let rgba = pixmap.data_as_u8_slice();
        let mut coverage = Vec::with_capacity((width * height) as usize);
        for px in rgba.as_chunks::<4>().0 {
            coverage.push(px[3]);
        }

        RasterizedText {
            width,
            height,
            baseline_y,
            coverage,
        }
    }
}

struct PathPen<'a> {
    path: &'a mut kurbo::BezPath,
    x_offset: f64,
    baseline_y: f64,
}

impl OutlinePen for PathPen<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.path
            .move_to((self.x_offset + f64::from(x), self.baseline_y - f64::from(y)));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.path
            .line_to((self.x_offset + f64::from(x), self.baseline_y - f64::from(y)));
    }

    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.path.quad_to(
            (
                self.x_offset + f64::from(cx0),
                self.baseline_y - f64::from(cy0),
            ),
            (self.x_offset + f64::from(x), self.baseline_y - f64::from(y)),
        );
    }

    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.path.curve_to(
            (
                self.x_offset + f64::from(cx0),
                self.baseline_y - f64::from(cy0),
            ),
            (
                self.x_offset + f64::from(cx1),
                self.baseline_y - f64::from(cy1),
            ),
            (self.x_offset + f64::from(x), self.baseline_y - f64::from(y)),
        );
    }

    fn close(&mut self) {
        self.path.close_path();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_rasterizer_measures_and_rasterizes_non_empty_coverage() {
        for font in [EmbeddedFont::HackRegular, EmbeddedFont::UbuntuLight] {
            let rast = FontRasterizer::new(font);
            let m = rast.measure("Velocity [m/s] 1.25", 16.0);
            assert!(m.width_px > 50.0, "unexpected width: {}", m.width_px);
            assert!(m.height_px >= 16.0, "unexpected height: {}", m.height_px);
            assert!(m.ascent_px > 0.0);
            assert!(m.descent_px < 0.0);

            let bmp = rast.rasterize("Velocity [m/s] 1.25", 16.0);
            assert_eq!(bmp.coverage.len(), (bmp.width * bmp.height) as usize);
            let lit_pixels = bmp.coverage.iter().filter(|&&a| a > 64).count();
            assert!(
                lit_pixels > 80,
                "expected > 80 lit coverage pixels for font {font:?}, got {lit_pixels}"
            );
        }
    }

    #[test]
    fn whitespace_string_rasterizes_to_zero_coverage() {
        let rast = FontRasterizer::default();
        let bmp = rast.rasterize("   ", 14.0);
        assert!(bmp.coverage.iter().all(|&a| a == 0));
    }
}

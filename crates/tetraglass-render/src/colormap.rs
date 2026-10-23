//! Perceptually uniform colormaps (`Viridis`, `Cividis`, `CoolWarm`) as 256-entry sRGB textures.

/// Built-in perceptually uniform colormaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ColormapKind {
    /// Viridis sequential colormap (purple -> blue -> teal -> green -> yellow).
    #[default]
    Viridis,
    /// Cividis sequential colormap (navy -> slate -> olive -> yellow, colorblind-safe).
    Cividis,
    /// Kenneth Moreland's diverging Cool-Warm colormap (blue -> neutral gray -> red).
    CoolWarm,
}

impl ColormapKind {
    /// All built-in colormap variants in canonical order.
    pub const ALL: [Self; 3] = [Self::Viridis, Self::Cividis, Self::CoolWarm];

    /// Returns the canonical lowercase identifier (`"viridis"`, `"cividis"`, `"coolwarm"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Viridis => "viridis",
            Self::Cividis => "cividis",
            Self::CoolWarm => "coolwarm",
        }
    }

    /// Returns the 256-entry sRGB `RGBA8` lookup table for this colormap.
    #[must_use]
    pub fn lut_rgba8(self) -> [[u8; 4]; 256] {
        let stops = match self {
            Self::Viridis => &VIRIDIS_STOPS,
            Self::Cividis => &CIVIDIS_STOPS,
            Self::CoolWarm => &COOLWARM_STOPS,
        };
        let mut lut = [[0u8; 4]; 256];
        for (i, entry) in lut.iter_mut().enumerate() {
            let t = (i as f32) / 255.0;
            *entry = sample_stops(stops, t);
        }
        lut
    }

    /// Samples this colormap in sRGB `RGBA8` at normalized scalar `t` (clamped to `[0.0, 1.0]`).
    #[must_use]
    pub fn sample_srgb(self, t: f32) -> [u8; 4] {
        let stops = match self {
            Self::Viridis => &VIRIDIS_STOPS,
            Self::Cividis => &CIVIDIS_STOPS,
            Self::CoolWarm => &COOLWARM_STOPS,
        };
        sample_stops(stops, t)
    }
}

fn sample_stops(stops: &[[f32; 3]; 17], t: f32) -> [u8; 4] {
    let clamped = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    let scaled = clamped * 16.0;
    let idx = (scaled.floor() as usize).min(15);
    let frac = scaled - (idx as f32);
    let a = stops[idx];
    let b = stops[idx + 1];
    let r = ((a[0] + (b[0] - a[0]) * frac) * 255.0).round() as u8;
    let g = ((a[1] + (b[1] - a[1]) * frac) * 255.0).round() as u8;
    let bl = ((a[2] + (b[2] - a[2]) * frac) * 255.0).round() as u8;
    [r, g, bl, 255]
}

/// 17-point control table for Viridis in normalized sRGB `[0.0, 1.0]`.
const VIRIDIS_STOPS: [[f32; 3]; 17] = [
    [0.267004, 0.004874, 0.329415],
    [0.282623, 0.094955, 0.417331],
    [0.278826, 0.175490, 0.483397],
    [0.258965, 0.251537, 0.524736],
    [0.229739, 0.322361, 0.545706],
    [0.199430, 0.387607, 0.554642],
    [0.172719, 0.448791, 0.557885],
    [0.149039, 0.508051, 0.557250],
    [0.127568, 0.566949, 0.550556],
    [0.120565, 0.625828, 0.533488],
    [0.157851, 0.683765, 0.501686],
    [0.246070, 0.738910, 0.452024],
    [0.369214, 0.788888, 0.382914],
    [0.515992, 0.831158, 0.294279],
    [0.678489, 0.863742, 0.189503],
    [0.845561, 0.887322, 0.099702],
    [0.993248, 0.906157, 0.143936],
];

/// 17-point control table for Cividis in normalized sRGB `[0.0, 1.0]`.
const CIVIDIS_STOPS: [[f32; 3]; 17] = [
    [0.000000, 0.135112, 0.304751],
    [0.000000, 0.186962, 0.438444],
    [0.133399, 0.237705, 0.427811],
    [0.219415, 0.285934, 0.422750],
    [0.288921, 0.334179, 0.425059],
    [0.351688, 0.382942, 0.431654],
    [0.411297, 0.432135, 0.442608],
    [0.470155, 0.482026, 0.459164],
    [0.530075, 0.532825, 0.472874],
    [0.592089, 0.584421, 0.469487],
    [0.656293, 0.637159, 0.457363],
    [0.722456, 0.691188, 0.436625],
    [0.790479, 0.746797, 0.406511],
    [0.860257, 0.804255, 0.365352],
    [0.931608, 0.863996, 0.308857],
    [0.995737, 0.926165, 0.226762],
    [0.995003, 0.910852, 0.276198],
];

/// 17-point control table for Moreland Cool-Warm in normalized sRGB `[0.0, 1.0]`.
const COOLWARM_STOPS: [[f32; 3]; 17] = [
    [0.230, 0.299, 0.754],
    [0.288, 0.384, 0.831],
    [0.353, 0.471, 0.894],
    [0.424, 0.557, 0.945],
    [0.498, 0.639, 0.976],
    [0.576, 0.710, 0.996],
    [0.655, 0.773, 0.992],
    [0.729, 0.824, 0.969],
    [0.865, 0.865, 0.865],
    [0.945, 0.796, 0.733],
    [0.969, 0.718, 0.616],
    [0.965, 0.627, 0.502],
    [0.945, 0.525, 0.396],
    [0.902, 0.416, 0.298],
    [0.839, 0.298, 0.212],
    [0.761, 0.165, 0.149],
    [0.706, 0.016, 0.150],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viridis_lut_has_expected_endpoints() {
        let lut = ColormapKind::Viridis.lut_rgba8();
        assert_eq!(lut[0], [68, 1, 84, 255]);
        assert_eq!(lut[255], [253, 231, 37, 255]);
        // Green channel increases monotonically across Viridis
        for w in lut.windows(2) {
            assert!(w[1][1] >= w[0][1]);
        }
    }

    #[test]
    fn coolwarm_midpoint_is_neutral_gray() {
        let mid = ColormapKind::CoolWarm.sample_srgb(0.5);
        assert_eq!(mid[0], mid[1]);
        assert_eq!(mid[1], mid[2]);
        assert!(mid[0] > 200);
    }
}

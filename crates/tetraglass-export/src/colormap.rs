//! CPU colormap tables and scalar normalization for glTF and bundle exports (`M3.4`, `M3.5`).

use tetraglass_scene::ColormapName;

/// Converts a single sRGB channel in `[0.0, 1.0]` to linear-light intensity in `[0.0, 1.0]`.
#[must_use]
pub fn srgb_to_linear(c: f32) -> f32 {
    let clamped = c.clamp(0.0, 1.0);
    if clamped <= 0.04045 {
        clamped / 12.92
    } else {
        ((clamped + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts a single linear-light channel in `[0.0, 1.0]` to sRGB in `[0.0, 1.0]`.
#[must_use]
pub fn linear_to_srgb(c: f32) -> f32 {
    let clamped = c.clamp(0.0, 1.0);
    if clamped <= 0.003_130_8 {
        clamped * 12.92
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    }
}

/// Normalizes a scalar value `value` into `[0.0, 1.0]` given `effective_range = [lo, hi]`,
/// `log_scale`, and optional diverging `center`.
///
/// Non-finite values (`NaN`, `±inf`) normalize to `0.0`.
#[must_use]
pub fn normalize_scalar(
    value: f64,
    effective_range: [f64; 2],
    log_scale: bool,
    center: Option<f64>,
) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    let [lo, hi] = effective_range;
    if log_scale {
        let safe_min = lo.max(1e-30);
        let safe_max = hi.max(safe_min * (1.0 + 1e-6));
        let safe_val = value.max(safe_min);
        let denom = safe_max.ln() - safe_min.ln();
        if denom.abs() <= 1e-30 {
            0.5
        } else {
            ((safe_val.ln() - safe_min.ln()) / denom).clamp(0.0, 1.0) as f32
        }
    } else if let Some(c) = center
        && c.is_finite()
    {
        let half_span = (c - lo).abs().max((hi - c).abs());
        if half_span <= 1e-30 {
            0.5
        } else {
            (0.5 + 0.5 * (value - c) / half_span).clamp(0.0, 1.0) as f32
        }
    } else {
        let span = hi - lo;
        if span.abs() <= 1e-30 {
            0.5
        } else {
            ((value - lo) / span).clamp(0.0, 1.0) as f32
        }
    }
}

/// Samples a [`ColormapName`] at normalized coordinate `t` in `[0.0, 1.0]`, returning
/// normalized sRGB `[r, g, b]` in `[0.0, 1.0]`.
#[must_use]
pub fn sample_colormap_srgb(colormap: ColormapName, t: f32) -> [f32; 3] {
    let clamped = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    match colormap {
        ColormapName::Viridis => sample_stops_f32(&VIRIDIS_STOPS, clamped),
        ColormapName::Cividis => sample_stops_f32(&CIVIDIS_STOPS, clamped),
        ColormapName::CoolWarm => sample_stops_f32(&COOLWARM_STOPS, clamped),
        ColormapName::Turbo => sample_stops_f32(&TURBO_STOPS, clamped),
        ColormapName::Grayscale => [clamped, clamped, clamped],
    }
}

/// Samples a [`ColormapName`] at normalized coordinate `t` in `[0.0, 1.0]`, returning
/// linear-light RGB `[r, g, b]` in `[0.0, 1.0]` suitable for glTF `COLOR_0`.
#[must_use]
pub fn sample_colormap_linear(colormap: ColormapName, t: f32) -> [f32; 3] {
    let [r, g, b] = sample_colormap_srgb(colormap, t);
    [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)]
}

/// Returns the 256-entry sRGB lookup table (`[[f32; 3]; 256]`) for `colormap`.
#[must_use]
pub fn colormap_lut_srgb_256(colormap: ColormapName) -> [[f32; 3]; 256] {
    let mut lut = [[0.0f32; 3]; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        let t = (i as f32) / 255.0;
        *slot = sample_colormap_srgb(colormap, t);
    }
    lut
}

fn sample_stops_f32(stops: &[[f32; 3]; 17], clamped_t: f32) -> [f32; 3] {
    let scaled = clamped_t * 16.0;
    let idx = (scaled.floor() as usize).min(15);
    let frac = scaled - (idx as f32);
    let a = stops[idx];
    let b = stops[idx + 1];
    [
        (a[0] + (b[0] - a[0]) * frac).clamp(0.0, 1.0),
        (a[1] + (b[1] - a[1]) * frac).clamp(0.0, 1.0),
        (a[2] + (b[2] - a[2]) * frac).clamp(0.0, 1.0),
    ]
}

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

const TURBO_STOPS: [[f32; 3]; 17] = [
    [0.18995, 0.07176, 0.23217],
    [0.25107, 0.25237, 0.63374],
    [0.27628, 0.42118, 0.89123],
    [0.25462, 0.58389, 0.99876],
    [0.18489, 0.73561, 0.92305],
    [0.10342, 0.85806, 0.77246],
    [0.16407, 0.93898, 0.60638],
    [0.32645, 0.98399, 0.43106],
    [0.53006, 0.99724, 0.28352],
    [0.71841, 0.96970, 0.20815],
    [0.86501, 0.89756, 0.19431],
    [0.95889, 0.78505, 0.21024],
    [0.99314, 0.63413, 0.18956],
    [0.97338, 0.46152, 0.12140],
    [0.90167, 0.29409, 0.05634],
    [0.78264, 0.15726, 0.01929],
    [0.47960, 0.01583, 0.01055],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_scalar_handles_linear_log_and_diverging_center() {
        assert!((normalize_scalar(5.0, [0.0, 10.0], false, None) - 0.5).abs() < 1e-6);
        assert!((normalize_scalar(-2.0, [0.0, 10.0], false, None) - 0.0).abs() < 1e-6);
        assert!((normalize_scalar(15.0, [0.0, 10.0], false, None) - 1.0).abs() < 1e-6);
        assert_eq!(normalize_scalar(f64::NAN, [0.0, 10.0], false, None), 0.0);

        // Log10 midpoint: 10 in [1, 100] -> 0.5
        assert!((normalize_scalar(10.0, [1.0, 100.0], true, None) - 0.5).abs() < 1e-6);

        // Diverging center: 2.0 in [0, 10] with center = 2.0 -> 0.5
        assert!((normalize_scalar(2.0, [0.0, 10.0], false, Some(2.0)) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn srgb_and_linear_round_trip() {
        for i in 0..=255 {
            let s = (i as f32) / 255.0;
            let lin = srgb_to_linear(s);
            let back = linear_to_srgb(lin);
            assert!((s - back).abs() < 1e-5);
        }
    }
}

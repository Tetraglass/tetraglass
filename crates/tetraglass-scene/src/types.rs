//! Serializable scene model types for `docs/scene-format.md` (version 1).

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tetraglass_io::MeshFormat;

use crate::SceneError;

/// Current scene format schema version (`1`).
pub const SCENE_VERSION: u32 = 1;

/// Default higher-order subdivision level (`2`) for [`Source::linearize_level`].
pub const DEFAULT_LINEARIZE_LEVEL: u32 = 2;

/// Default point size in pixels (`3.0`) for [`Representation::point_size`].
pub const DEFAULT_POINT_SIZE: f64 = 3.0;

/// Default perspective vertical field-of-view angle in degrees (`30.0`) for [`CameraConfig::view_angle`].
pub const DEFAULT_VIEW_ANGLE_DEGREES: f64 = 30.0;

/// Default headless image dimensions in pixels (`[1200, 800]`) for [`Output::size`].
pub const DEFAULT_OUTPUT_SIZE: [u32; 2] = [1200, 800];

/// Default multisample anti-aliasing sample count (`4`) for [`Output::msaa`].
pub const DEFAULT_MSAA_SAMPLES: u32 = 4;

/// Default Cycles path-tracing sample count (`128`) for [`CyclesConfig::samples`].
pub const DEFAULT_CYCLES_SAMPLES: u32 = 128;

/// Top-level declarative scene description (`docs/scene-format.md`, version 1).
///
/// Every frontend (JSON files, the CLI, the desktop/web viewer, and the Rust builder API)
/// produces and consumes this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct Scene {
    /// Schema version number (must be `1` for v1 scenes).
    pub version: u32,

    /// Named input datasets read from mesh files. Keys must be lowercase `snake_case`
    /// and unique across `sources` and `filters`.
    pub sources: BTreeMap<String, Source>,

    /// Named derived datasets produced by visualization filters. Keys must be lowercase
    /// `snake_case` and unique across `sources` and `filters`.
    #[serde(default = "default_filters")]
    pub filters: BTreeMap<String, Filter>,

    /// Viewport configuration: visible representations, camera, lighting, background and annotations.
    pub view: View,

    /// Headless image output configuration (required by `tetraglass render`, optional for interactive viewing).
    #[serde(default = "default_output")]
    pub output: Option<Output>,
}

impl Scene {
    /// Parses a [`Scene`] from a UTF-8 JSON string.
    pub fn from_json_str(json: &str) -> Result<Self, SceneError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Parses a [`Scene`] from a UTF-8 JSON byte slice.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, SceneError> {
        Ok(serde_json::from_slice(bytes)?)
    }

    /// Serializes this [`Scene`] to a compact JSON string.
    pub fn to_json_string(&self) -> Result<String, SceneError> {
        Ok(serde_json::to_string(self)?)
    }

    /// Serializes this [`Scene`] to a pretty-printed JSON string.
    pub fn to_json_string_pretty(&self) -> Result<String, SceneError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Reads and parses a [`Scene`] from a JSON file path on disk (native targets only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self, SceneError> {
        let bytes = std::fs::read(path)?;
        Self::from_json_bytes(&bytes)
    }

    /// Validates this [`Scene`] structurally (without requiring loaded mesh data).
    pub fn validate_structure(&self) -> Result<(), crate::ValidationReport> {
        crate::validate_scene_structure(self)
    }

    /// Validates this [`Scene`] both structurally and semantically against loaded source [`tetraglass_core::UnstructuredGrid`]s.
    pub fn validate(
        &self,
        sources: &BTreeMap<String, tetraglass_core::UnstructuredGrid>,
    ) -> Result<(), crate::ValidationReport> {
        crate::validate_scene(self, sources)
    }

    /// Validates this [`Scene`] against pre-extracted [`crate::DatasetMetadata`] for each source.
    pub fn validate_with_metadata(
        &self,
        sources: &BTreeMap<String, crate::DatasetMetadata>,
    ) -> Result<(), crate::ValidationReport> {
        crate::validate_scene_with_metadata(self, sources)
    }
}

/// A named input dataset read from a mesh file (`sources.<name>`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct Source {
    /// Path to the mesh file, resolved relative to the scene file (`.vtu`, `.vtk`, `.pvd`, `.msh`).
    pub file: String,

    /// Optional explicit mesh format override. When omitted, the format is inferred from the file extension.
    #[serde(default = "default_source_format")]
    pub format: Option<SourceFormat>,

    /// Reference-space subdivision level for higher-order cells (default: `2`, used from M2 onward).
    #[serde(default = "default_linearize_level")]
    pub linearize_level: u32,
}

impl Source {
    /// Creates a [`Source`] for `file` with default format inference and `linearize_level = 2`.
    #[must_use]
    pub fn new(file: impl Into<String>) -> Self {
        Self {
            file: file.into(),
            format: default_source_format(),
            linearize_level: default_linearize_level(),
        }
    }
}

/// Supported mesh file format identifiers in a scene [`Source`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SourceFormat {
    /// VTK XML UnstructuredGrid (`.vtu`).
    Vtu,
    /// Legacy VTK UnstructuredGrid (`.vtk`).
    Vtk,
    /// ParaView XML time-series collection (`.pvd`).
    Pvd,
    /// Gmsh MSH 2.2 or 4.1 mesh (`.msh`).
    Msh,
}

impl From<SourceFormat> for MeshFormat {
    fn from(format: SourceFormat) -> Self {
        match format {
            SourceFormat::Vtu => Self::Vtu,
            SourceFormat::Vtk => Self::VtkLegacy,
            SourceFormat::Pvd => Self::Pvd,
            SourceFormat::Msh => Self::Gmsh,
        }
    }
}

impl From<MeshFormat> for SourceFormat {
    fn from(format: MeshFormat) -> Self {
        match format {
            MeshFormat::Vtu => Self::Vtu,
            MeshFormat::VtkLegacy => Self::Vtk,
            MeshFormat::Pvd => Self::Pvd,
            MeshFormat::Gmsh => Self::Msh,
        }
    }
}

/// Which half-space to keep relative to a clip plane's outward normal (`filters.<name>.keep`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClipKeep {
    /// Keep the half-space in the direction of the plane normal (`(p - origin) · normal >= 0`).
    #[default]
    Front,
    /// Keep the half-space opposite to the plane normal (`(p - origin) · normal <= 0`).
    Back,
}

impl From<ClipKeep> for tetraglass_filters::ClipKeep {
    fn from(keep: ClipKeep) -> Self {
        match keep {
            ClipKeep::Front => Self::Front,
            ClipKeep::Back => Self::Back,
        }
    }
}

/// Clipping algorithm mode (`filters.<name>.mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClipMode {
    /// Cut tetrahedra along the clip plane via the shared marching-tetrahedra kernel (`D-006`).
    #[default]
    Exact,
    /// Keep whole original source cells that overlap the kept half-space.
    Crinkle,
}

impl From<ClipMode> for tetraglass_filters::ClipMode {
    fn from(mode: ClipMode) -> Self {
        match mode {
            ClipMode::Exact => Self::Exact,
            ClipMode::Crinkle => Self::Crinkle,
        }
    }
}

/// Integration direction for streamlines (`filters.<name>.direction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StreamlineDirectionSpec {
    /// Integrate forward along `+v(x)` (default).
    #[default]
    Forward,
    /// Integrate backward along `-v(x)`.
    Backward,
    /// Integrate both backward and forward from each seed and join into a single polyline.
    Both,
}

impl From<StreamlineDirectionSpec> for tetraglass_filters::StreamlineDirection {
    fn from(direction: StreamlineDirectionSpec) -> Self {
        match direction {
            StreamlineDirectionSpec::Forward => Self::Forward,
            StreamlineDirectionSpec::Backward => Self::Backward,
            StreamlineDirectionSpec::Both => Self::Both,
        }
    }
}

/// Numerical ODE integrator for streamlines (`filters.<name>.integrator`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StreamlineIntegratorSpec {
    /// Adaptive Dormand-Prince RK4(5) integrator (default).
    #[default]
    Rk45,
    /// Fixed-step classical 4th-order Runge-Kutta integrator.
    Rk4,
}

impl From<StreamlineIntegratorSpec> for tetraglass_filters::StreamlineIntegrator {
    fn from(integrator: StreamlineIntegratorSpec) -> Self {
        match integrator {
            StreamlineIntegratorSpec::Rk45 => Self::Rk45,
            StreamlineIntegratorSpec::Rk4 => Self::Rk4,
        }
    }
}

/// Uniform line-segment seed generator (`{ "line": { "from": [...], "to": [...], "count": n } }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct StreamlineLineSeedSpec {
    /// Start point `[x, y, z]` of the seed segment.
    pub from: [f64; 3],
    /// End point `[x, y, z]` of the seed segment.
    pub to: [f64; 3],
    /// Number of seed points (`>= 1`) uniformly spaced along the segment.
    pub count: usize,
}

/// Rectangular plane-grid seed generator (`{ "plane": { "origin": [...], "normal": [...], "extent": [w, h], "count": [nu, nv] } }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct StreamlinePlaneSeedSpec {
    /// Optional center `[x, y, z]` of the seed plane. Defaults to the center of the input dataset bounds when omitted.
    #[serde(default = "default_none_vec3")]
    pub origin: Option<[f64; 3]>,
    /// Plane normal `[nx, ny, nz]` (must be non-zero; normalized automatically).
    pub normal: [f64; 3],
    /// Optional width and height `[w, h]` of the seed grid on the plane. Defaults to `[0.5 * diag, 0.5 * diag]` when omitted.
    #[serde(default = "default_none_vec2")]
    pub extent: Option<[f64; 2]>,
    /// Grid resolution `[nu, nv]` along the two in-plane axes (`nu >= 1`, `nv >= 1`).
    pub count: [usize; 2],
}

/// Seed specification for streamline integration (`filters.<name>.seeds`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum StreamlineSeedsSpec {
    /// A single seed point `[x, y, z]`.
    Point([f64; 3]),
    /// An explicit list of seed points `[[x, y, z], ...]`.
    Points(Vec<[f64; 3]>),
    /// `count` seed points uniformly spaced along the segment from `from` to `to`.
    Line(StreamlineLineSeedSpec),
    /// A `count[0] * count[1]` rectangular grid of seed points on a plane.
    Plane(StreamlinePlaneSeedSpec),
}

impl StreamlineSeedsSpec {
    /// Creates a single-point seed generator (`{ "point": [x, y, z] }`).
    #[must_use]
    pub const fn point(pt: [f64; 3]) -> Self {
        Self::Point(pt)
    }

    /// Creates an explicit point-list seed generator (`{ "points": [[x, y, z], ...] }`).
    #[must_use]
    pub const fn points(pts: Vec<[f64; 3]>) -> Self {
        Self::Points(pts)
    }

    /// Creates a uniform line-segment seed generator (`{ "line": { "from": ..., "to": ..., "count": ... } }`).
    #[must_use]
    pub const fn line(from: [f64; 3], to: [f64; 3], count: usize) -> Self {
        Self::Line(StreamlineLineSeedSpec { from, to, count })
    }

    /// Creates a rectangular plane-grid seed generator (`{ "plane": { ... } }`).
    #[must_use]
    pub const fn plane(
        origin: Option<[f64; 3]>,
        normal: [f64; 3],
        extent: Option<[f64; 2]>,
        count: [usize; 2],
    ) -> Self {
        Self::Plane(StreamlinePlaneSeedSpec {
            origin,
            normal,
            extent,
            count,
        })
    }
}

/// Geometric shape for vector glyphs (`filters.<name>.shape`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GlyphShapeSpec {
    /// Cylindrical shaft with a conical tip oriented along the vector (default).
    #[default]
    Arrow,
    /// Circular cone with base at the sample point and apex oriented along the vector.
    Cone,
    /// UV sphere centered at the sample point.
    Sphere,
}

impl From<GlyphShapeSpec> for tetraglass_filters::GlyphShape {
    fn from(shape: GlyphShapeSpec) -> Self {
        match shape {
            GlyphShapeSpec::Arrow => Self::Arrow,
            GlyphShapeSpec::Cone => Self::Cone,
            GlyphShapeSpec::Sphere => Self::Sphere,
        }
    }
}

/// Named scaling preset for [`GlyphScaleBySpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GlyphScalePreset {
    /// Scale glyph size proportionally to the local vector magnitude `|v|` (`"magnitude"`).
    #[default]
    Magnitude,
}

/// Glyph scaling specification (`"magnitude"` or a fixed positive size in world units).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum GlyphScaleBySpec {
    /// Named scaling preset (`"magnitude"`).
    Preset(GlyphScalePreset),
    /// Uniform world-space base size (`> 0.0`) for all glyphs.
    Factor(f64),
}

impl Default for GlyphScaleBySpec {
    fn default() -> Self {
        Self::Preset(GlyphScalePreset::Magnitude)
    }
}

impl GlyphScaleBySpec {
    /// Scale proportionally to vector magnitude (`"magnitude"`).
    pub const MAGNITUDE: Self = Self::Preset(GlyphScalePreset::Magnitude);

    /// Creates a fixed uniform world-space size (`> 0.0`).
    #[must_use]
    pub const fn fixed(size: f64) -> Self {
        Self::Factor(size)
    }
}

impl From<GlyphScaleBySpec> for tetraglass_filters::GlyphScaleMode {
    fn from(spec: GlyphScaleBySpec) -> Self {
        match spec {
            GlyphScaleBySpec::Preset(GlyphScalePreset::Magnitude) => Self::Magnitude,
            GlyphScaleBySpec::Factor(v) => Self::Fixed(v),
        }
    }
}

/// Visualization filter definition (`filters.<name>`).
///
/// Internally tagged by `"type"`. Milestone M1 implements `"surface"`, M2 adds `"slice"`,
/// `"contour"`, `"clip"`, and `"threshold"`, and M4 adds `"streamlines"`, `"glyphs"`, and `"resample"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Filter {
    /// Extracts the external boundary surface of the input dataset.
    #[serde(rename_all = "snake_case")]
    Surface {
        /// Name of the input source or filter.
        input: String,
    },

    /// Cuts the input volume dataset with a plane (`origin` defaults to the center of input bounds).
    #[serde(rename_all = "snake_case")]
    Slice {
        /// Name of the input source or filter.
        input: String,

        /// Optional point `[x, y, z]` on the slice plane. Defaults to the center of the input dataset bounds when omitted.
        #[serde(default = "default_none_vec3")]
        origin: Option<[f64; 3]>,

        /// Outward plane normal `[nx, ny, nz]` (must be non-zero; normalized automatically).
        normal: [f64; 3],
    },

    /// Extracts one or more isosurfaces from a continuous point attribute array (`"contour"`).
    #[serde(rename_all = "snake_case")]
    Contour {
        /// Name of the input source or filter.
        input: String,

        /// Point array reference (`"<array>"` or `"<array>:<component>"`) to contour.
        field: String,

        /// Non-empty list of finite isovalues to extract.
        values: Vec<f64>,
    },

    /// Clips the input volume dataset with a plane, keeping one side in `"exact"` or `"crinkle"` mode (`"clip"`).
    #[serde(rename_all = "snake_case")]
    Clip {
        /// Name of the input source or filter.
        input: String,

        /// Optional point `[x, y, z]` on the clip plane. Defaults to the center of the input dataset bounds when omitted.
        #[serde(default = "default_none_vec3")]
        origin: Option<[f64; 3]>,

        /// Outward plane normal `[nx, ny, nz]` (must be non-zero; normalized automatically).
        normal: [f64; 3],

        /// Which half-space to keep relative to `normal` (default: `"front"`).
        #[serde(default = "default_clip_keep")]
        keep: ClipKeep,

        /// Clipping mode: `"exact"` (default) or `"crinkle"` (keeps whole original cells).
        #[serde(default = "default_clip_mode")]
        mode: ClipMode,
    },

    /// Keeps cells of the input volume dataset whose attribute value is within `[min, max]` (`"threshold"`).
    #[serde(rename_all = "snake_case")]
    Threshold {
        /// Name of the input source or filter.
        input: String,

        /// Attribute array reference (`"[point:|cell:]<array>[:<component>]"`) to threshold.
        field: String,

        /// Optional inclusive lower bound (omit for a one-sided `<= max` threshold).
        #[serde(default = "default_none_f64")]
        min: Option<f64>,

        /// Optional inclusive upper bound (omit for a one-sided `>= min` threshold).
        #[serde(default = "default_none_f64")]
        max: Option<f64>,
    },

    /// Integrates streamlines through a 3-component vector field on the input volume dataset (`"streamlines"`).
    #[serde(rename_all = "snake_case")]
    Streamlines {
        /// Name of the input source or filter.
        input: String,

        /// Vector attribute array reference (`"[point:|cell:]<array>"`, 3 components) to integrate.
        vectors: String,

        /// Seed point generator (`"point"`, `"points"`, `"line"`, or `"plane"`).
        seeds: StreamlineSeedsSpec,

        /// Optional maximum arc length per integration direction (`> 0.0`). Defaults to `4.0 * bounds.diagonal()`.
        #[serde(default = "default_none_f64")]
        max_length: Option<f64>,

        /// Optional initial / fixed integration step size (`> 0.0`). Defaults to `0.01 * bounds.diagonal()`.
        #[serde(default = "default_none_f64")]
        step_size: Option<f64>,

        /// Integration direction: `"forward"` (default), `"backward"`, or `"both"`.
        #[serde(default = "default_streamline_direction")]
        direction: StreamlineDirectionSpec,

        /// ODE integrator: `"rk45"` (default) or `"rk4"`.
        #[serde(default = "default_streamline_integrator")]
        integrator: StreamlineIntegratorSpec,

        /// Optional tube radius in world units (`> 0.0`) used when displayed with `"tubes"` or `"surface"` style.
        /// Defaults to `0.012 * bounds.diagonal()`.
        #[serde(default = "default_none_f64")]
        tube_radius: Option<f64>,
    },

    /// Instantiates oriented and scaled 3D glyphs (`"arrow"`, `"cone"`, or `"sphere"`) from a 3-component vector field (`"glyphs"`).
    #[serde(rename_all = "snake_case")]
    Glyphs {
        /// Name of the input source or filter.
        input: String,

        /// Vector attribute array reference (`"[point:|cell:]<array>"`, 3 components) to orient and scale glyphs.
        vectors: String,

        /// Glyph geometry shape: `"arrow"` (default), `"cone"`, or `"sphere"`.
        #[serde(default = "default_glyph_shape")]
        shape: GlyphShapeSpec,

        /// Glyph sizing mode: `"magnitude"` (default) or a fixed positive world-space size.
        #[serde(default = "default_glyph_scale_by")]
        scale_by: GlyphScaleBySpec,

        /// Optional positive multiplier applied to the glyph size (`> 0.0`).
        #[serde(default = "default_none_f64")]
        scale_factor: Option<f64>,

        /// Subsampling stride (`>= 1`, default: `1`).
        #[serde(default = "default_glyph_every")]
        every: usize,
    },

    /// Resamples the input volume dataset onto a regular 3D grid `[nx, ny, nz]` for `"volume"` ray-marching (`"resample"`).
    #[serde(rename_all = "snake_case")]
    Resample {
        /// Name of the input source or filter.
        input: String,

        /// Sample grid resolution `[nx, ny, nz]` along the X, Y, and Z axes (each `>= 2`).
        dimensions: [usize; 3],
    },
}

impl Filter {
    /// Creates an external-surface [`Filter::Surface`] reading from `input`.
    #[must_use]
    pub fn surface(input: impl Into<String>) -> Self {
        Self::Surface {
            input: input.into(),
        }
    }

    /// Creates a plane-slice [`Filter::Slice`] reading from `input` through the input bounds center
    /// (`origin: None`) with outward plane `normal`.
    #[must_use]
    pub fn slice(input: impl Into<String>, normal: [f64; 3]) -> Self {
        Self::Slice {
            input: input.into(),
            origin: None,
            normal,
        }
    }

    /// Creates a plane-slice [`Filter::Slice`] reading from `input` at an explicit `origin`
    /// with outward plane `normal`.
    #[must_use]
    pub fn slice_at(input: impl Into<String>, origin: [f64; 3], normal: [f64; 3]) -> Self {
        Self::Slice {
            input: input.into(),
            origin: Some(origin),
            normal,
        }
    }

    /// Creates a multi-isovalue [`Filter::Contour`] reading from `input` on `field` at `values`.
    #[must_use]
    pub fn contour(input: impl Into<String>, field: impl Into<String>, values: Vec<f64>) -> Self {
        Self::Contour {
            input: input.into(),
            field: field.into(),
            values,
        }
    }

    /// Creates a plane-clip [`Filter::Clip`] reading from `input` through the input bounds center
    /// (`origin: None`) with outward plane `normal`, keeping `"front"` in `"exact"` mode.
    #[must_use]
    pub fn clip(input: impl Into<String>, normal: [f64; 3]) -> Self {
        Self::Clip {
            input: input.into(),
            origin: None,
            normal,
            keep: default_clip_keep(),
            mode: default_clip_mode(),
        }
    }

    /// Creates a plane-clip [`Filter::Clip`] reading from `input` at an explicit `origin`
    /// with outward plane `normal`, `keep` side, and `mode`.
    #[must_use]
    pub fn clip_at(
        input: impl Into<String>,
        origin: [f64; 3],
        normal: [f64; 3],
        keep: ClipKeep,
        mode: ClipMode,
    ) -> Self {
        Self::Clip {
            input: input.into(),
            origin: Some(origin),
            normal,
            keep,
            mode,
        }
    }

    /// Creates a two-sided [`Filter::Threshold`] reading from `input` on `field` in `[min, max]`.
    #[must_use]
    pub fn threshold(
        input: impl Into<String>,
        field: impl Into<String>,
        min: f64,
        max: f64,
    ) -> Self {
        Self::Threshold {
            input: input.into(),
            field: field.into(),
            min: Some(min),
            max: Some(max),
        }
    }

    /// Creates a one-sided lower [`Filter::Threshold`] (`value >= min`).
    #[must_use]
    pub fn threshold_min(input: impl Into<String>, field: impl Into<String>, min: f64) -> Self {
        Self::Threshold {
            input: input.into(),
            field: field.into(),
            min: Some(min),
            max: None,
        }
    }

    /// Creates a one-sided upper [`Filter::Threshold`] (`value <= max`).
    #[must_use]
    pub fn threshold_max(input: impl Into<String>, field: impl Into<String>, max: f64) -> Self {
        Self::Threshold {
            input: input.into(),
            field: field.into(),
            min: None,
            max: Some(max),
        }
    }

    /// Creates a [`Filter::Streamlines`] reading from `input` on `vectors` with `seeds` and default integration settings.
    #[must_use]
    pub fn streamlines(
        input: impl Into<String>,
        vectors: impl Into<String>,
        seeds: StreamlineSeedsSpec,
    ) -> Self {
        Self::Streamlines {
            input: input.into(),
            vectors: vectors.into(),
            seeds,
            max_length: None,
            step_size: None,
            direction: default_streamline_direction(),
            integrator: default_streamline_integrator(),
            tube_radius: None,
        }
    }

    /// Creates a [`Filter::Glyphs`] reading from `input` on `vectors` with default arrow shape, magnitude scaling, and stride `1`.
    #[must_use]
    pub fn glyphs(input: impl Into<String>, vectors: impl Into<String>) -> Self {
        Self::Glyphs {
            input: input.into(),
            vectors: vectors.into(),
            shape: default_glyph_shape(),
            scale_by: default_glyph_scale_by(),
            scale_factor: None,
            every: default_glyph_every(),
        }
    }

    /// Creates a [`Filter::Resample`] reading from `input` with grid resolution `dimensions = [nx, ny, nz]`.
    #[must_use]
    pub fn resample(input: impl Into<String>, dimensions: [usize; 3]) -> Self {
        Self::Resample {
            input: input.into(),
            dimensions,
        }
    }

    /// Returns the name of the upstream source or filter that feeds this filter.
    #[must_use]
    pub fn input(&self) -> &str {
        match self {
            Self::Surface { input }
            | Self::Slice { input, .. }
            | Self::Contour { input, .. }
            | Self::Clip { input, .. }
            | Self::Threshold { input, .. }
            | Self::Streamlines { input, .. }
            | Self::Glyphs { input, .. }
            | Self::Resample { input, .. } => input,
        }
    }
}

/// Viewport specification (`view`): representations to draw, camera, lighting, background and annotations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct View {
    /// Ordered list of dataset representations to display in the viewport.
    pub show: Vec<Representation>,

    /// Camera framing, orientation, zoom and projection settings.
    #[serde(default = "default_camera_config")]
    pub camera: CameraConfig,

    /// Lighting rig preset (`"default"` or `"headlight"`).
    #[serde(default = "default_lighting_preset")]
    pub lights: LightingPreset,

    /// Viewport background color or vertical gradient (default: `"white"`).
    #[serde(default = "default_background_spec")]
    pub background: BackgroundSpec,

    /// Whether to draw the orientation axes triad in the corner of the viewport (default: `true`).
    #[serde(default = "default_true")]
    pub axes: bool,

    /// Optional title text drawn at the top of the image.
    #[serde(default = "default_none_string")]
    pub title: Option<String>,
}

/// Visual representation of a source or filter in `view.show`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct Representation {
    /// Name of the source or filter dataset to display.
    pub input: String,

    /// Drawing style (default: `"surface"`).
    #[serde(default = "default_representation_style")]
    pub style: RepresentationStyle,

    /// Optional array reference (`"[point:|cell:]<array>[:<component>]"`) used for scalar coloring.
    /// When omitted, the representation uses the solid [`Self::color`].
    #[serde(default = "default_none_string")]
    pub color_by: Option<String>,

    /// Solid surface color used when [`Self::color_by`] is omitted (default: `"lightgray"`).
    #[serde(default = "default_representation_color")]
    pub color: ColorSpec,

    /// Colormap used when [`Self::color_by`] is set (default: `"viridis"`).
    #[serde(default = "default_colormap_name")]
    pub colormap: ColormapName,

    /// Scalar range mapped across the colormap (default: `"auto"`).
    #[serde(default = "default_range_spec")]
    pub range: RangeSpec,

    /// Whether to map positive scalar values logarithmically (default: `false`).
    #[serde(default = "default_false")]
    pub log_scale: bool,

    /// Optional reference value mapped to the midpoint of a diverging colormap.
    #[serde(default = "default_none_f64")]
    pub center: Option<f64>,

    /// Surface opacity in `0.0..=1.0` (default: `1.0`). Values below `1.0` use order-independent transparency (M3).
    #[serde(default = "default_opacity")]
    pub opacity: f64,

    /// Colorbar legend configuration: `false` (default), `true`, or `{ "title": ..., "position": ... }`.
    #[serde(default = "default_colorbar_spec")]
    pub colorbar: ColorbarSpec,

    /// Wireframe and element-edge color for `"surface_with_edges"` and `"wireframe"` (default: `"black"`).
    #[serde(default = "default_edge_color")]
    pub edge_color: ColorSpec,

    /// Point diameter in pixels for the `"points"` style (default: `3.0`).
    #[serde(default = "default_point_size")]
    pub point_size: f64,

    /// Optional piecewise-linear opacity transfer function `[[t, alpha], ...]` (`t` and `alpha` in `0.0..=1.0`,
    /// sorted by `t`) for `"volume"` representations. When omitted, a linear ramp `[[0.0, 0.0], [1.0, 1.0]]` is used.
    #[serde(default = "default_none_transfer_function")]
    pub transfer_function: Option<Vec<[f64; 2]>>,

    /// Whether this representation is visible (default: `true`).
    #[serde(default = "default_true")]
    pub visible: bool,
}

impl Representation {
    /// Creates a [`Representation`] for `input` with all documented defaults.
    #[must_use]
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            style: default_representation_style(),
            color_by: default_none_string(),
            color: default_representation_color(),
            colormap: default_colormap_name(),
            range: default_range_spec(),
            log_scale: default_false(),
            center: default_none_f64(),
            opacity: default_opacity(),
            colorbar: default_colorbar_spec(),
            edge_color: default_edge_color(),
            point_size: default_point_size(),
            transfer_function: default_none_transfer_function(),
            visible: default_true(),
        }
    }
}

/// Visual drawing style for a [`Representation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RepresentationStyle {
    /// Shaded surface (external faces of volume meshes).
    Surface,
    /// Shaded surface overlaid with element edges.
    SurfaceWithEdges,
    /// Element edges only.
    Wireframe,
    /// Mesh vertices rendered as screen-space points.
    Points,
    /// Polyline rendering for streamlines (M4).
    Lines,
    /// Shaded tube rendering for streamlines (M4).
    Tubes,
    /// Direct volume rendering of a resampled grid (M4).
    Volume,
}

/// A color specification: either a CSS color name (such as `"white"` or `"lightgray"`)
/// or a linear `[r, g, b]` triple with components in `0.0..=1.0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ColorSpec {
    /// Named CSS color (for example `"white"`, `"black"`, `"lightgray"`, `"red"`, `"steelblue"`).
    Named(String),
    /// Explicit `[r, g, b]` color with each channel in `0.0..=1.0`.
    Rgb([f64; 3]),
}

impl ColorSpec {
    /// Resolves this color specification to a `[f32; 3]` RGB triple in `0.0..=1.0`.
    ///
    /// Returns `None` if a named color is unrecognized or if an explicit `[r, g, b]` triple
    /// contains non-finite values or values outside `0.0..=1.0`.
    #[must_use]
    pub fn resolve_rgb(&self) -> Option<[f32; 3]> {
        match self {
            Self::Named(name) => resolve_named_css_color(name),
            Self::Rgb([r, g, b]) => {
                if r.is_finite()
                    && g.is_finite()
                    && b.is_finite()
                    && (0.0..=1.0).contains(r)
                    && (0.0..=1.0).contains(g)
                    && (0.0..=1.0).contains(b)
                {
                    Some([*r as f32, *g as f32, *b as f32])
                } else {
                    None
                }
            }
        }
    }

    /// Returns the list of recognized CSS color names for validation suggestions.
    #[must_use]
    pub const fn known_color_names() -> &'static [&'static str] {
        KNOWN_CSS_COLOR_NAMES
    }
}

const KNOWN_CSS_COLOR_NAMES: &[&str] = &[
    "black",
    "white",
    "lightgray",
    "lightgrey",
    "gray",
    "grey",
    "darkgray",
    "darkgrey",
    "silver",
    "dimgray",
    "dimgrey",
    "slategray",
    "slategrey",
    "red",
    "green",
    "blue",
    "yellow",
    "cyan",
    "magenta",
    "orange",
    "purple",
    "brown",
    "pink",
    "navy",
    "teal",
    "olive",
    "maroon",
    "coral",
    "crimson",
    "gold",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lime",
    "salmon",
    "skyblue",
    "steelblue",
    "tan",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
];

fn resolve_named_css_color(name: &str) -> Option<[f32; 3]> {
    let trimmed = name.trim();
    if let Some(hex) = trimmed.strip_prefix('#') {
        if hex.len() == 6 {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            return Some([
                f32::from(r) / 255.0,
                f32::from(g) / 255.0,
                f32::from(b) / 255.0,
            ]);
        }
        return None;
    }

    let lower = trimmed.to_ascii_lowercase();
    let (r, g, b): (u8, u8, u8) = match lower.as_str() {
        "black" => (0, 0, 0),
        "white" => (255, 255, 255),
        "lightgray" | "lightgrey" => (211, 211, 211),
        "gray" | "grey" => (128, 128, 128),
        "darkgray" | "darkgrey" => (169, 169, 169),
        "silver" => (192, 192, 192),
        "dimgray" | "dimgrey" => (105, 105, 105),
        "slategray" | "slategrey" => (112, 128, 144),
        "red" => (255, 0, 0),
        "green" => (0, 128, 0),
        "blue" => (0, 0, 255),
        "yellow" => (255, 255, 0),
        "cyan" | "aqua" => (0, 255, 255),
        "magenta" | "fuchsia" => (255, 0, 255),
        "orange" => (255, 165, 0),
        "purple" => (128, 0, 128),
        "brown" => (165, 42, 42),
        "pink" => (255, 192, 203),
        "navy" => (0, 0, 128),
        "teal" => (0, 128, 128),
        "olive" => (128, 128, 0),
        "maroon" => (128, 0, 0),
        "coral" => (255, 127, 80),
        "crimson" => (220, 20, 60),
        "gold" => (255, 215, 0),
        "indigo" => (75, 0, 130),
        "ivory" => (255, 255, 240),
        "khaki" => (240, 230, 140),
        "lavender" => (230, 230, 250),
        "lime" => (0, 255, 0),
        "salmon" => (250, 128, 114),
        "skyblue" => (135, 206, 235),
        "steelblue" => (70, 130, 180),
        "tan" => (210, 180, 140),
        "tomato" => (255, 99, 71),
        "turquoise" => (64, 224, 208),
        "violet" => (238, 130, 238),
        "wheat" => (245, 222, 179),
        _ => return None,
    };
    Some([
        f32::from(r) / 255.0,
        f32::from(g) / 255.0,
        f32::from(b) / 255.0,
    ])
}

/// Built-in colormap names (`docs/scene-format.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ColormapName {
    /// Perceptually uniform sequential blue-green-yellow colormap (default).
    Viridis,
    /// Perceptually uniform, color-vision-deficiency-friendly blue-yellow colormap.
    Cividis,
    /// Diverging blue-white-red colormap (`"cool_warm"`).
    CoolWarm,
    /// High-contrast rainbow-like sequential colormap (`"turbo"`).
    Turbo,
    /// Linear black-to-white grayscale ramp (`"grayscale"`).
    Grayscale,
}

/// Automatic range preset for [`RangeSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RangePreset {
    /// Fit to the minimum and maximum of the current time step (`"auto"`).
    Auto,
    /// Fit to the minimum and maximum across all time steps in a series (`"auto_all_steps"`).
    AutoAllSteps,
}

/// Scalar value range specification for colormap mapping (`"auto"`, `"auto_all_steps"`, or `[min, max]`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum RangeSpec {
    /// Automatic range mode (`"auto"` or `"auto_all_steps"`).
    Preset(RangePreset),
    /// Explicit `[min, max]` value bounds.
    Bounds([f64; 2]),
}

impl RangeSpec {
    /// Automatic range from the current time step (`"auto"`).
    pub const AUTO: Self = Self::Preset(RangePreset::Auto);

    /// Automatic range across all time steps (`"auto_all_steps"`).
    pub const AUTO_ALL_STEPS: Self = Self::Preset(RangePreset::AutoAllSteps);

    /// Creates an explicit `[min, max]` range specification.
    #[must_use]
    pub const fn bounds(min: f64, max: f64) -> Self {
        Self::Bounds([min, max])
    }
}

/// Placement of a colorbar legend inside the viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ColorbarPosition {
    /// Vertical colorbar on the right edge of the viewport (default).
    Right,
    /// Vertical colorbar on the left edge of the viewport.
    Left,
    /// Horizontal colorbar along the bottom edge of the viewport.
    Bottom,
    /// Horizontal colorbar along the top edge of the viewport.
    Top,
}

/// Detailed colorbar options (`{ "title": "...", "position": "right" }`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct ColorbarOptions {
    /// Optional title label displayed on the colorbar (defaults to the array name when omitted).
    #[serde(default = "default_none_string")]
    pub title: Option<String>,

    /// Placement of the colorbar in the viewport (default: `"right"`).
    #[serde(default = "default_colorbar_position")]
    pub position: ColorbarPosition,
}

/// Colorbar setting on a [`Representation`]: `false` (default), `true`, or [`ColorbarOptions`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ColorbarSpec {
    /// Simple boolean toggle (`true` or `false`).
    Enabled(bool),
    /// Enabled colorbar with custom title or position.
    Options(ColorbarOptions),
}

impl ColorbarSpec {
    /// Returns `true` if the colorbar should be drawn.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        match self {
            Self::Enabled(enabled) => *enabled,
            Self::Options(_) => true,
        }
    }
}

/// Camera configuration in `view.camera` (`docs/scene-format.md`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct CameraConfig {
    /// Target bounds to frame: `"all"` (default) or the name of a specific source or filter.
    /// Ignored when [`Self::position`] is explicitly provided.
    #[serde(default = "default_camera_fit")]
    pub fit: String,

    /// Horizontal orbit angle in degrees around the focal point, applied after fitting (default: `0.0`).
    #[serde(default = "default_zero_f64")]
    pub azimuth: f64,

    /// Vertical elevation angle in degrees around the focal point, applied after fitting (default: `0.0`).
    #[serde(default = "default_zero_f64")]
    pub elevation: f64,

    /// Roll angle in degrees around the viewing direction, applied after fitting (default: `0.0`).
    #[serde(default = "default_zero_f64")]
    pub roll: f64,

    /// Zoom factor applied after fitting; values greater than `1.0` move closer (default: `1.0`).
    #[serde(default = "default_zoom")]
    pub zoom: f64,

    /// Explicit world-space camera eye position `[x, y, z]`. Overrides `fit`, angles, and `zoom`.
    #[serde(default = "default_none_vec3")]
    pub position: Option<[f64; 3]>,

    /// Explicit world-space focal point `[x, y, z]`. Defaults to the fitted bounds center when omitted.
    #[serde(default = "default_none_vec3")]
    pub focal_point: Option<[f64; 3]>,

    /// Explicit world-space up vector `[x, y, z]`.
    #[serde(default = "default_none_vec3")]
    pub view_up: Option<[f64; 3]>,

    /// Camera projection mode: `"perspective"` (default) or `"orthographic"`.
    #[serde(default = "default_projection_kind")]
    pub projection: ProjectionKind,

    /// Vertical field-of-view angle in degrees for perspective projection (default: `30.0`).
    #[serde(default = "default_view_angle")]
    pub view_angle: f64,
}

impl Default for CameraConfig {
    fn default() -> Self {
        default_camera_config()
    }
}

/// Camera projection mode (`"perspective"` or `"orthographic"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionKind {
    /// Perspective projection with foreshortening (default).
    Perspective,
    /// Parallel orthographic projection.
    Orthographic,
}

/// Lighting rig preset for `view.lights`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LightingPreset {
    /// Headlight plus key and fill lights (default).
    Default,
    /// Single camera-attached headlight.
    Headlight,
}

/// Vertical top-to-bottom background color gradient (`{ "top": ..., "bottom": ... }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct BackgroundGradient {
    /// Color at the top edge of the viewport.
    pub top: ColorSpec,
    /// Color at the bottom edge of the viewport.
    pub bottom: ColorSpec,
}

/// Viewport background specification (`view.background`): a solid [`ColorSpec`] (default: `"white"`)
/// or a vertical [`BackgroundGradient`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum BackgroundSpec {
    /// Solid background color.
    Solid(ColorSpec),
    /// Vertical top-to-bottom color gradient.
    Gradient(BackgroundGradient),
}

/// Headless render output settings (`output` section in `docs/scene-format.md`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct Output {
    /// Output PNG path relative to the scene file (can be overridden by `tetraglass render -o`).
    pub image: String,

    /// Image width and height in pixels `[width, height]` (default: `[1200, 800]`).
    #[serde(default = "default_output_size")]
    pub size: [u32; 2],

    /// Time step selector for transient datasets (default: `"first"`).
    #[serde(default = "default_time_selector")]
    pub time: TimeSelector,

    /// Optional animation frame range settings (M4).
    #[serde(default = "default_none_frames")]
    pub frames: Option<FramesConfig>,

    /// Multisample anti-aliasing sample count (`1` or `4`, default: `4`).
    #[serde(default = "default_msaa")]
    pub msaa: u32,

    /// Rendering engine: `"raster"` (default) or `"cycles"` (M3).
    #[serde(default = "default_render_engine")]
    pub engine: RenderEngine,

    /// Optional Blender Cycles ray-tracing settings (M3).
    #[serde(default = "default_none_cycles")]
    pub cycles: Option<CyclesConfig>,
}

impl Output {
    /// Creates an [`Output`] writing to `image` with all documented defaults.
    #[must_use]
    pub fn new(image: impl Into<String>) -> Self {
        Self {
            image: image.into(),
            size: default_output_size(),
            time: default_time_selector(),
            frames: default_none_frames(),
            msaa: default_msaa(),
            engine: default_render_engine(),
            cycles: default_none_cycles(),
        }
    }
}

/// Named time-step preset (`"first"` or `"last"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TimePreset {
    /// First time step in the time series (default).
    First,
    /// Last time step in the time series.
    Last,
}

/// Explicit zero-based time-step index selector (`{ "step": n }`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct TimeStepIndex {
    /// Zero-based index of the time step in the series.
    pub step: usize,
}

/// Selector for choosing a time step in a transient dataset (`"first"`, `"last"`, `{ "step": n }`, or a time value).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum TimeSelector {
    /// Named time preset (`"first"` or `"last"`).
    Preset(TimePreset),
    /// Explicit zero-based step index (`{ "step": n }`).
    Step(TimeStepIndex),
    /// Continuous simulation time value (for example `0.5`).
    Value(f64),
}

impl TimeSelector {
    /// Selects the first time step (`"first"`).
    pub const FIRST: Self = Self::Preset(TimePreset::First);

    /// Selects the last time step (`"last"`).
    pub const LAST: Self = Self::Preset(TimePreset::Last);

    /// Selects the zero-based step `step` (`{ "step": step }`).
    #[must_use]
    pub const fn step(step: usize) -> Self {
        Self::Step(TimeStepIndex { step })
    }

    /// Resolves this [`TimeSelector`] against a slice of available simulation time steps,
    /// returning `(step_index, time_value)` on success or a descriptive error message.
    pub fn resolve_step(self, times: &[f64]) -> Result<(usize, f64), String> {
        if times.is_empty() {
            return Err("time series contains no time steps".to_string());
        }
        match self {
            Self::Preset(TimePreset::First) => Ok((0, times[0])),
            Self::Preset(TimePreset::Last) => {
                let last = times.len() - 1;
                Ok((last, times[last]))
            }
            Self::Step(TimeStepIndex { step }) => {
                if let Some(&t) = times.get(step) {
                    Ok((step, t))
                } else {
                    Err(format!(
                        "step index {step} is out of bounds (available steps: 0..={})",
                        times.len() - 1
                    ))
                }
            }
            Self::Value(target) => {
                if !target.is_finite() {
                    return Err("output time value must be a finite number".to_string());
                }
                let mut min_t = f64::INFINITY;
                let mut max_t = f64::NEG_INFINITY;
                let mut best_idx = 0;
                let mut best_diff = f64::INFINITY;
                for (idx, &t) in times.iter().enumerate() {
                    min_t = min_t.min(t);
                    max_t = max_t.max(t);
                    let diff = (t - target).abs();
                    if diff < best_diff {
                        best_diff = diff;
                        best_idx = idx;
                    }
                }
                if target < min_t - 1e-9 || target > max_t + 1e-9 {
                    return Err(format!(
                        "time {target} is outside the available time range [{min_t}, {max_t}]"
                    ));
                }
                Ok((best_idx, times[best_idx]))
            }
        }
    }
}

/// Animation frame range configuration (`output.frames`, M4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct FramesConfig {
    /// Starting time step or time value.
    pub from: TimeSelector,
    /// Ending time step or time value.
    pub to: TimeSelector,
    /// Output filename pattern (for example `"frame_%04d.png"`).
    pub pattern: String,
}

/// Rendering engine backend (`output.engine`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RenderEngine {
    /// Built-in wgpu rasterizer (default).
    Raster,
    /// Headless Blender Cycles path tracer (M3).
    Cycles,
}

/// Blender Cycles ray-tracing options (`output.cycles`, M3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct CyclesConfig {
    /// Path-tracing sample count per pixel (default: `128`).
    #[serde(default = "default_cycles_samples")]
    pub samples: u32,

    /// Compute device selection (for example `"auto"`, `"cpu"`, `"gpu"`, default: `"auto"`).
    #[serde(default = "default_cycles_device")]
    pub device: String,

    /// Optional `.blend` output path to save the constructed Blender scene alongside rendering.
    #[serde(default = "default_none_string")]
    pub save_blend: Option<String>,
}

// --- Explicit serde default functions (per `.agents/rules/scene-schema.md`) ---

fn default_filters() -> BTreeMap<String, Filter> {
    BTreeMap::new()
}

const fn default_output() -> Option<Output> {
    None
}

const fn default_source_format() -> Option<SourceFormat> {
    None
}

const fn default_linearize_level() -> u32 {
    DEFAULT_LINEARIZE_LEVEL
}

fn default_camera_config() -> CameraConfig {
    CameraConfig {
        fit: default_camera_fit(),
        azimuth: default_zero_f64(),
        elevation: default_zero_f64(),
        roll: default_zero_f64(),
        zoom: default_zoom(),
        position: default_none_vec3(),
        focal_point: default_none_vec3(),
        view_up: default_none_vec3(),
        projection: default_projection_kind(),
        view_angle: default_view_angle(),
    }
}

const fn default_lighting_preset() -> LightingPreset {
    LightingPreset::Default
}

fn default_background_spec() -> BackgroundSpec {
    BackgroundSpec::Solid(ColorSpec::Named("white".to_string()))
}

const fn default_true() -> bool {
    true
}

const fn default_false() -> bool {
    false
}

const fn default_none_string() -> Option<String> {
    None
}

const fn default_none_f64() -> Option<f64> {
    None
}

const fn default_none_vec3() -> Option<[f64; 3]> {
    None
}

const fn default_none_vec2() -> Option<[f64; 2]> {
    None
}

const fn default_clip_keep() -> ClipKeep {
    ClipKeep::Front
}

const fn default_clip_mode() -> ClipMode {
    ClipMode::Exact
}

const fn default_streamline_direction() -> StreamlineDirectionSpec {
    StreamlineDirectionSpec::Forward
}

const fn default_streamline_integrator() -> StreamlineIntegratorSpec {
    StreamlineIntegratorSpec::Rk45
}

const fn default_glyph_shape() -> GlyphShapeSpec {
    GlyphShapeSpec::Arrow
}

const fn default_glyph_scale_by() -> GlyphScaleBySpec {
    GlyphScaleBySpec::Preset(GlyphScalePreset::Magnitude)
}

const fn default_glyph_every() -> usize {
    1
}

const fn default_representation_style() -> RepresentationStyle {
    RepresentationStyle::Surface
}

fn default_representation_color() -> ColorSpec {
    ColorSpec::Named("lightgray".to_string())
}

const fn default_colormap_name() -> ColormapName {
    ColormapName::Viridis
}

const fn default_range_spec() -> RangeSpec {
    RangeSpec::Preset(RangePreset::Auto)
}

const fn default_opacity() -> f64 {
    1.0
}

const fn default_colorbar_spec() -> ColorbarSpec {
    ColorbarSpec::Enabled(false)
}

const fn default_colorbar_position() -> ColorbarPosition {
    ColorbarPosition::Right
}

fn default_edge_color() -> ColorSpec {
    ColorSpec::Named("black".to_string())
}

const fn default_point_size() -> f64 {
    DEFAULT_POINT_SIZE
}

const fn default_none_transfer_function() -> Option<Vec<[f64; 2]>> {
    None
}

fn default_camera_fit() -> String {
    "all".to_string()
}

const fn default_zero_f64() -> f64 {
    0.0
}

const fn default_zoom() -> f64 {
    1.0
}

const fn default_projection_kind() -> ProjectionKind {
    ProjectionKind::Perspective
}

const fn default_view_angle() -> f64 {
    DEFAULT_VIEW_ANGLE_DEGREES
}

const fn default_output_size() -> [u32; 2] {
    DEFAULT_OUTPUT_SIZE
}

const fn default_time_selector() -> TimeSelector {
    TimeSelector::Preset(TimePreset::First)
}

const fn default_none_frames() -> Option<FramesConfig> {
    None
}

const fn default_msaa() -> u32 {
    DEFAULT_MSAA_SAMPLES
}

const fn default_render_engine() -> RenderEngine {
    RenderEngine::Raster
}

const fn default_none_cycles() -> Option<CyclesConfig> {
    None
}

const fn default_cycles_samples() -> u32 {
    DEFAULT_CYCLES_SAMPLES
}

fn default_cycles_device() -> String {
    "auto".to_string()
}

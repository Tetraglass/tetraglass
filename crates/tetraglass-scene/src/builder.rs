//! Fluent, type-safe Rust builder API for constructing [`Scene`] graphs (`M6.1`, `D-004`).
//!
//! [`SceneBuilder`] produces the exact same [`Scene`] data structure consumed by JSON scene files,
//! the CLI, the desktop/web viewer, and the exporters.
//!
//! # Examples
//!
//! ```
//! use tetraglass_scene::{
//!     ColormapName, ContourFilter, Scene, SliceFilter, Source, TimeSelector,
//! };
//!
//! let mut builder = Scene::builder();
//! let flow = builder.source("flow", Source::file("results.pvd"));
//! let cut = builder.filter("cut", SliceFilter::new(&flow, [0.0, 0.0, 1.0]));
//! let iso = builder.filter("iso", ContourFilter::new(&flow, "pressure", [101325.0]));
//!
//! builder
//!     .show(&cut)
//!     .color_by("velocity")
//!     .magnitude()
//!     .colormap(ColormapName::Viridis)
//!     .colorbar(true);
//! builder.show(&iso).opacity(0.4);
//! builder.camera().fit("all").azimuth(30.0).elevation(20.0);
//! builder.output("out.png").size(1600, 900).time(TimeSelector::LAST);
//!
//! let scene = builder.build_validated().expect("structurally valid v1 scene");
//! assert_eq!(scene.sources.len(), 1);
//! assert_eq!(scene.filters.len(), 2);
//! assert_eq!(scene.view.show.len(), 2);
//! ```

use std::collections::BTreeMap;
use std::fmt;

use crate::{
    BackgroundGradient, BackgroundSpec, CameraConfig, ClipKeep, ClipMode, ColorSpec,
    ColorbarOptions, ColorbarPosition, ColorbarSpec, ColormapName, CyclesConfig,
    DEFAULT_CYCLES_SAMPLES, Filter, FramesConfig, GlyphScaleBySpec, GlyphShapeSpec, LightingPreset,
    Output, ProjectionKind, RangeSpec, RenderEngine, Representation, RepresentationStyle,
    SCENE_VERSION, Scene, Source, SourceFormat, StreamlineDirectionSpec, StreamlineIntegratorSpec,
    StreamlineSeedsSpec, TimeSelector, ValidationReport, View,
};

/// Handle referencing a named source or filter dataset inside a [`SceneBuilder`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DatasetRef(String);

impl DatasetRef {
    /// Creates a [`DatasetRef`] wrapping a dataset identifier.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Returns the dataset name as a string slice.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for DatasetRef {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DatasetRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<DatasetRef> for String {
    fn from(value: DatasetRef) -> Self {
        value.0
    }
}

impl From<&DatasetRef> for String {
    fn from(value: &DatasetRef) -> Self {
        value.0.clone()
    }
}

impl From<&str> for DatasetRef {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for DatasetRef {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// Builder for an external-surface filter (`Filter::Surface`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceFilter {
    input: String,
}

impl SurfaceFilter {
    /// Creates an external-surface filter reading from `input`.
    #[must_use]
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
        }
    }
}

impl From<SurfaceFilter> for Filter {
    fn from(builder: SurfaceFilter) -> Self {
        Self::Surface {
            input: builder.input,
        }
    }
}

/// Builder for a plane-slice filter (`Filter::Slice`).
#[derive(Debug, Clone, PartialEq)]
pub struct SliceFilter {
    input: String,
    origin: Option<[f64; 3]>,
    normal: [f64; 3],
}

impl SliceFilter {
    /// Creates a plane-slice filter reading from `input` with outward plane `normal`.
    #[must_use]
    pub fn new(input: impl Into<String>, normal: [f64; 3]) -> Self {
        Self {
            input: input.into(),
            origin: None,
            normal,
        }
    }

    /// Sets an explicit world-space plane `origin` (defaults to the input dataset center when omitted).
    #[must_use]
    pub const fn origin(mut self, origin: [f64; 3]) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Overrides the outward plane `normal`.
    #[must_use]
    pub const fn normal(mut self, normal: [f64; 3]) -> Self {
        self.normal = normal;
        self
    }
}

impl From<SliceFilter> for Filter {
    fn from(builder: SliceFilter) -> Self {
        Self::Slice {
            input: builder.input,
            origin: builder.origin,
            normal: builder.normal,
        }
    }
}

/// Builder for a multi-isovalue contour filter (`Filter::Contour`).
#[derive(Debug, Clone, PartialEq)]
pub struct ContourFilter {
    input: String,
    field: String,
    values: Vec<f64>,
}

impl ContourFilter {
    /// Creates a contour filter reading from `input` on scalar `field` at `values`.
    #[must_use]
    pub fn new(
        input: impl Into<String>,
        field: impl Into<String>,
        values: impl IntoIterator<Item = f64>,
    ) -> Self {
        Self {
            input: input.into(),
            field: field.into(),
            values: values.into_iter().collect(),
        }
    }

    /// Appends an additional isovalue.
    #[must_use]
    pub fn value(mut self, iso: f64) -> Self {
        self.values.push(iso);
        self
    }
}

impl From<ContourFilter> for Filter {
    fn from(builder: ContourFilter) -> Self {
        Self::Contour {
            input: builder.input,
            field: builder.field,
            values: builder.values,
        }
    }
}

/// Builder for a half-space plane-clip filter (`Filter::Clip`).
#[derive(Debug, Clone, PartialEq)]
pub struct ClipFilter {
    input: String,
    origin: Option<[f64; 3]>,
    normal: [f64; 3],
    keep: ClipKeep,
    mode: ClipMode,
}

impl ClipFilter {
    /// Creates a plane-clip filter reading from `input` with outward plane `normal`,
    /// defaulting to `ClipKeep::Front` and `ClipMode::Exact`.
    #[must_use]
    pub fn new(input: impl Into<String>, normal: [f64; 3]) -> Self {
        Self {
            input: input.into(),
            origin: None,
            normal,
            keep: ClipKeep::Front,
            mode: ClipMode::Exact,
        }
    }

    /// Sets an explicit world-space plane `origin`.
    #[must_use]
    pub const fn origin(mut self, origin: [f64; 3]) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Sets which half-space to keep (`ClipKeep::Front` or `ClipKeep::Back`).
    #[must_use]
    pub const fn keep(mut self, keep: ClipKeep) -> Self {
        self.keep = keep;
        self
    }

    /// Sets the clipping mode (`ClipMode::Exact` or `ClipMode::Crinkle`).
    #[must_use]
    pub const fn mode(mut self, mode: ClipMode) -> Self {
        self.mode = mode;
        self
    }

    /// Configures crinkle clipping (`ClipMode::Crinkle`), keeping whole original source cells.
    #[must_use]
    pub const fn crinkle(mut self) -> Self {
        self.mode = ClipMode::Crinkle;
        self
    }
}

impl From<ClipFilter> for Filter {
    fn from(builder: ClipFilter) -> Self {
        Self::Clip {
            input: builder.input,
            origin: builder.origin,
            normal: builder.normal,
            keep: builder.keep,
            mode: builder.mode,
        }
    }
}

/// Builder for a scalar/component threshold filter (`Filter::Threshold`).
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdFilter {
    input: String,
    field: String,
    min: Option<f64>,
    max: Option<f64>,
}

impl ThresholdFilter {
    /// Creates a threshold filter reading from `input` on `field` with no initial bounds.
    /// Set at least one bound via [`Self::min`], [`Self::max`], or [`Self::between`].
    #[must_use]
    pub fn new(input: impl Into<String>, field: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            field: field.into(),
            min: None,
            max: None,
        }
    }

    /// Sets the inclusive lower bound (`value >= min`).
    #[must_use]
    pub const fn min(mut self, min: f64) -> Self {
        self.min = Some(min);
        self
    }

    /// Sets the inclusive upper bound (`value <= max`).
    #[must_use]
    pub const fn max(mut self, max: f64) -> Self {
        self.max = Some(max);
        self
    }

    /// Sets both inclusive lower and upper bounds (`min <= value <= max`).
    #[must_use]
    pub const fn between(mut self, min: f64, max: f64) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }
}

impl From<ThresholdFilter> for Filter {
    fn from(builder: ThresholdFilter) -> Self {
        Self::Threshold {
            input: builder.input,
            field: builder.field,
            min: builder.min,
            max: builder.max,
        }
    }
}

/// Builder for a streamline integration filter (`Filter::Streamlines`).
#[derive(Debug, Clone, PartialEq)]
pub struct StreamlinesFilter {
    input: String,
    vectors: String,
    seeds: StreamlineSeedsSpec,
    max_length: Option<f64>,
    step_size: Option<f64>,
    direction: StreamlineDirectionSpec,
    integrator: StreamlineIntegratorSpec,
    tube_radius: Option<f64>,
}

impl StreamlinesFilter {
    /// Creates a streamline filter reading from `input` on 3-component vector array `vectors` with `seeds`.
    #[must_use]
    pub fn new(
        input: impl Into<String>,
        vectors: impl Into<String>,
        seeds: StreamlineSeedsSpec,
    ) -> Self {
        Self {
            input: input.into(),
            vectors: vectors.into(),
            seeds,
            max_length: None,
            step_size: None,
            direction: StreamlineDirectionSpec::Forward,
            integrator: StreamlineIntegratorSpec::Rk45,
            tube_radius: None,
        }
    }

    /// Sets the maximum streamline arc length.
    #[must_use]
    pub const fn max_length(mut self, max_length: f64) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Sets the initial or fixed integration step size.
    #[must_use]
    pub const fn step_size(mut self, step_size: f64) -> Self {
        self.step_size = Some(step_size);
        self
    }

    /// Sets the integration direction (`Forward`, `Backward`, or `Both`).
    #[must_use]
    pub const fn direction(mut self, direction: StreamlineDirectionSpec) -> Self {
        self.direction = direction;
        self
    }

    /// Sets the ODE integrator (`Rk45` or `Rk4`).
    #[must_use]
    pub const fn integrator(mut self, integrator: StreamlineIntegratorSpec) -> Self {
        self.integrator = integrator;
        self
    }

    /// Sets the world-space tube radius for `"tubes"` representations.
    #[must_use]
    pub const fn tube_radius(mut self, tube_radius: f64) -> Self {
        self.tube_radius = Some(tube_radius);
        self
    }
}

impl From<StreamlinesFilter> for Filter {
    fn from(builder: StreamlinesFilter) -> Self {
        Self::Streamlines {
            input: builder.input,
            vectors: builder.vectors,
            seeds: builder.seeds,
            max_length: builder.max_length,
            step_size: builder.step_size,
            direction: builder.direction,
            integrator: builder.integrator,
            tube_radius: builder.tube_radius,
        }
    }
}

/// Builder for a vector glyph filter (`Filter::Glyphs`).
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphsFilter {
    input: String,
    vectors: String,
    shape: GlyphShapeSpec,
    scale_by: GlyphScaleBySpec,
    scale_factor: Option<f64>,
    every: usize,
}

impl GlyphsFilter {
    /// Creates a glyph filter reading from `input` on 3-component vector array `vectors`.
    #[must_use]
    pub fn new(input: impl Into<String>, vectors: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            vectors: vectors.into(),
            shape: GlyphShapeSpec::Arrow,
            scale_by: GlyphScaleBySpec::MAGNITUDE,
            scale_factor: None,
            every: 1,
        }
    }

    /// Sets the glyph geometry shape (`Arrow`, `Cone`, or `Sphere`).
    #[must_use]
    pub const fn shape(mut self, shape: GlyphShapeSpec) -> Self {
        self.shape = shape;
        self
    }

    /// Sets the glyph scaling mode (`GlyphScaleBySpec::MAGNITUDE` or `GlyphScaleBySpec::fixed(...)`).
    #[must_use]
    pub const fn scale_by(mut self, scale_by: GlyphScaleBySpec) -> Self {
        self.scale_by = scale_by;
        self
    }

    /// Sets an explicit world-space length multiplier (`scale_factor > 0.0`).
    #[must_use]
    pub const fn scale_factor(mut self, scale_factor: f64) -> Self {
        self.scale_factor = Some(scale_factor);
        self
    }

    /// Sets the subsampling stride (`every >= 1`).
    #[must_use]
    pub const fn every(mut self, every: usize) -> Self {
        self.every = every;
        self
    }
}

impl From<GlyphsFilter> for Filter {
    fn from(builder: GlyphsFilter) -> Self {
        Self::Glyphs {
            input: builder.input,
            vectors: builder.vectors,
            shape: builder.shape,
            scale_by: builder.scale_by,
            scale_factor: builder.scale_factor,
            every: builder.every,
        }
    }
}

/// Builder for a regular-grid volume resampling filter (`Filter::Resample`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResampleFilter {
    input: String,
    dimensions: [usize; 3],
}

impl ResampleFilter {
    /// Creates a regular-grid resampling filter reading from `input` with grid resolution `dimensions = [nx, ny, nz]`.
    #[must_use]
    pub fn new(input: impl Into<String>, dimensions: [usize; 3]) -> Self {
        Self {
            input: input.into(),
            dimensions,
        }
    }
}

impl From<ResampleFilter> for Filter {
    fn from(builder: ResampleFilter) -> Self {
        Self::Resample {
            input: builder.input,
            dimensions: builder.dimensions,
        }
    }
}

/// Fluent builder for constructing a [`Scene`] in Rust (`D-004`).
#[derive(Debug, Clone)]
pub struct SceneBuilder {
    sources: BTreeMap<String, Source>,
    filters: BTreeMap<String, Filter>,
    view: View,
    output: Option<Output>,
}

impl Default for SceneBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneBuilder {
    /// Creates an empty v1 [`SceneBuilder`] with default camera, lighting, white background, and axes enabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: BTreeMap::new(),
            filters: BTreeMap::new(),
            view: View {
                show: Vec::new(),
                camera: CameraConfig::default(),
                lights: LightingPreset::Default,
                background: BackgroundSpec::Solid(ColorSpec::Named("white".to_string())),
                axes: true,
                title: None,
            },
            output: None,
        }
    }

    /// Registers a named [`Source`] dataset (`sources.<name>`) and returns a [`DatasetRef`] handle to it.
    pub fn source(&mut self, name: impl Into<String>, source: Source) -> DatasetRef {
        let name = name.into();
        self.sources.insert(name.clone(), source);
        DatasetRef(name)
    }

    /// Registers a [`Source`] dataset with a unique `snake_case` name derived automatically from its file stem,
    /// returning a [`DatasetRef`] handle to it.
    pub fn add_source(&mut self, source: Source) -> DatasetRef {
        let base = sanitize_snake_case_stem(&source.file, "source");
        let name = self.unique_dataset_name(&base);
        self.sources.insert(name.clone(), source);
        DatasetRef(name)
    }

    /// Registers a named [`Filter`] (`filters.<name>`) and returns a [`DatasetRef`] handle to it.
    pub fn filter(&mut self, name: impl Into<String>, filter: impl Into<Filter>) -> DatasetRef {
        let name = name.into();
        self.filters.insert(name.clone(), filter.into());
        DatasetRef(name)
    }

    /// Registers a [`Filter`] with an automatically generated unique `snake_case` name
    /// (for example `"slice"`, `"slice_2"`, `"contour"`) and returns a [`DatasetRef`] handle to it.
    pub fn add_filter(&mut self, filter: impl Into<Filter>) -> DatasetRef {
        let filter = filter.into();
        let prefix = match &filter {
            Filter::Surface { .. } => "surface",
            Filter::Slice { .. } => "slice",
            Filter::Contour { .. } => "contour",
            Filter::Clip { .. } => "clip",
            Filter::Threshold { .. } => "threshold",
            Filter::Streamlines { .. } => "streamlines",
            Filter::Glyphs { .. } => "glyphs",
            Filter::Resample { .. } => "resample",
        };
        let name = self.unique_dataset_name(prefix);
        self.filters.insert(name.clone(), filter);
        DatasetRef(name)
    }

    /// Appends a visible [`Representation`] for `input` to `view.show` and returns a [`RepresentationBuilder`]
    /// for configuring its style, coloring, colormap, opacity, and colorbar.
    pub fn show(&mut self, input: impl Into<String>) -> RepresentationBuilder<'_> {
        self.view.show.push(Representation::new(input));
        let idx = self.view.show.len() - 1;
        RepresentationBuilder {
            rep: &mut self.view.show[idx],
        }
    }

    /// Returns a [`CameraBuilder`] for configuring `view.camera`.
    pub fn camera(&mut self) -> CameraBuilder<'_> {
        CameraBuilder {
            camera: &mut self.view.camera,
        }
    }

    /// Sets the viewport lighting preset (`LightingPreset::Default` or `LightingPreset::Headlight`).
    pub fn lights(&mut self, lights: LightingPreset) -> &mut Self {
        self.view.lights = lights;
        self
    }

    /// Sets the viewport [`BackgroundSpec`].
    pub fn background(&mut self, background: BackgroundSpec) -> &mut Self {
        self.view.background = background;
        self
    }

    /// Sets a solid viewport background color.
    pub fn background_color(&mut self, color: impl Into<ColorSpec>) -> &mut Self {
        self.view.background = BackgroundSpec::Solid(color.into());
        self
    }

    /// Sets a vertical top-to-bottom viewport background gradient.
    pub fn background_gradient(
        &mut self,
        top: impl Into<ColorSpec>,
        bottom: impl Into<ColorSpec>,
    ) -> &mut Self {
        self.view.background = BackgroundSpec::Gradient(BackgroundGradient {
            top: top.into(),
            bottom: bottom.into(),
        });
        self
    }

    /// Enables or disables the orientation axes triad in the corner of the viewport.
    pub fn axes(&mut self, enabled: bool) -> &mut Self {
        self.view.axes = enabled;
        self
    }

    /// Sets the top-centered viewport title text.
    pub fn title(&mut self, title: impl Into<String>) -> &mut Self {
        self.view.title = Some(title.into());
        self
    }

    /// Configures headless image output writing to `image` and returns an [`OutputBuilder`].
    pub fn output(&mut self, image: impl Into<String>) -> OutputBuilder<'_> {
        let image = image.into();
        let output = self
            .output
            .get_or_insert_with(|| Output::new(image.clone()));
        output.image = image;
        OutputBuilder { output }
    }

    /// Consumes the builder and returns the constructed [`Scene`].
    #[must_use]
    pub fn build(self) -> Scene {
        Scene {
            version: SCENE_VERSION,
            sources: self.sources,
            filters: self.filters,
            view: self.view,
            output: self.output,
        }
    }

    /// Consumes the builder, validates the scene's structural invariants (`Scene::validate_structure`),
    /// and returns the validated [`Scene`].
    pub fn build_validated(self) -> Result<Scene, ValidationReport> {
        let scene = self.build();
        scene.validate_structure()?;
        Ok(scene)
    }

    fn unique_dataset_name(&self, base: &str) -> String {
        if !self.sources.contains_key(base) && !self.filters.contains_key(base) {
            return base.to_string();
        }
        let mut idx = 2usize;
        loop {
            let candidate = format!("{base}_{idx}");
            if !self.sources.contains_key(&candidate) && !self.filters.contains_key(&candidate) {
                return candidate;
            }
            idx += 1;
        }
    }
}

/// Fluent builder for a single [`Representation`] in `view.show`.
#[derive(Debug)]
pub struct RepresentationBuilder<'a> {
    rep: &'a mut Representation,
}

impl<'a> RepresentationBuilder<'a> {
    /// Sets the [`RepresentationStyle`].
    pub fn style(&mut self, style: RepresentationStyle) -> &mut Self {
        self.rep.style = style;
        self
    }

    /// Sets the style to [`RepresentationStyle::Surface`].
    pub fn surface(&mut self) -> &mut Self {
        self.style(RepresentationStyle::Surface)
    }

    /// Sets the style to [`RepresentationStyle::SurfaceWithEdges`].
    pub fn surface_with_edges(&mut self) -> &mut Self {
        self.style(RepresentationStyle::SurfaceWithEdges)
    }

    /// Sets the style to [`RepresentationStyle::Wireframe`].
    pub fn wireframe(&mut self) -> &mut Self {
        self.style(RepresentationStyle::Wireframe)
    }

    /// Sets the style to [`RepresentationStyle::Points`].
    pub fn points(&mut self) -> &mut Self {
        self.style(RepresentationStyle::Points)
    }

    /// Sets the style to [`RepresentationStyle::Lines`] (for streamlines).
    pub fn lines(&mut self) -> &mut Self {
        self.style(RepresentationStyle::Lines)
    }

    /// Sets the style to [`RepresentationStyle::Tubes`] (for streamlines).
    pub fn tubes(&mut self) -> &mut Self {
        self.style(RepresentationStyle::Tubes)
    }

    /// Sets the style to [`RepresentationStyle::Volume`] (for resampled regular grids).
    pub fn volume(&mut self) -> &mut Self {
        self.style(RepresentationStyle::Volume)
    }

    /// Colors the representation by `field` (for example `"pressure"`, `"velocity:magnitude"`, or `"cell:cell_id"`).
    pub fn color_by(&mut self, field: impl Into<String>) -> &mut Self {
        self.rep.color_by = Some(field.into());
        self
    }

    /// Colors the representation by an explicit point-associated array (`"point:<name>"`).
    pub fn point_field(&mut self, name: impl AsRef<str>) -> &mut Self {
        self.rep.color_by = Some(format!("point:{}", name.as_ref()));
        self
    }

    /// Colors the representation by an explicit cell-associated array (`"cell:<name>"`).
    pub fn cell_field(&mut self, name: impl AsRef<str>) -> &mut Self {
        self.rep.color_by = Some(format!("cell:{}", name.as_ref()));
        self
    }

    /// Appends `:magnitude` to the active `color_by` array reference (for example turning `"velocity"` into `"velocity:magnitude"`).
    pub fn magnitude(&mut self) -> &mut Self {
        if let Some(current) = &self.rep.color_by {
            let base = strip_component_suffix(current);
            self.rep.color_by = Some(format!("{base}:magnitude"));
        }
        self
    }

    /// Appends a component selector (such as `"x"`, `"y"`, `"z"`, `"xx"`, or `"0"`) to the active `color_by` array reference.
    pub fn component(&mut self, comp: impl AsRef<str>) -> &mut Self {
        if let Some(current) = &self.rep.color_by {
            let base = strip_component_suffix(current);
            self.rep.color_by = Some(format!("{base}:{}", comp.as_ref()));
        }
        self
    }

    /// Sets the solid fallback color (`view.show[i].color`).
    pub fn color(&mut self, color: impl Into<ColorSpec>) -> &mut Self {
        self.rep.color = color.into();
        self
    }

    /// Sets the colormap (`Viridis`, `Cividis`, `CoolWarm`, `Turbo`, `Grayscale`).
    pub fn colormap(&mut self, colormap: ColormapName) -> &mut Self {
        self.rep.colormap = colormap;
        self
    }

    /// Sets the scalar value range (`RangeSpec::AUTO`, `RangeSpec::AUTO_ALL_STEPS`, or `RangeSpec::bounds(min, max)`).
    pub fn range(&mut self, range: RangeSpec) -> &mut Self {
        self.rep.range = range;
        self
    }

    /// Sets explicit `[min, max]` scalar value bounds for the colormap.
    pub fn range_bounds(&mut self, min: f64, max: f64) -> &mut Self {
        self.rep.range = RangeSpec::bounds(min, max);
        self
    }

    /// Enables or disables logarithmic (`log10`) scalar mapping.
    pub fn log_scale(&mut self, enabled: bool) -> &mut Self {
        self.rep.log_scale = enabled;
        self
    }

    /// Sets the diverging colormap midpoint value.
    pub fn center(&mut self, center: f64) -> &mut Self {
        self.rep.center = Some(center);
        self
    }

    /// Sets the representation opacity (`0.0..=1.0`).
    pub fn opacity(&mut self, opacity: f64) -> &mut Self {
        self.rep.opacity = opacity;
        self
    }

    /// Enables or disables the colorbar legend with default placement (`Right`).
    pub fn colorbar(&mut self, enabled: bool) -> &mut Self {
        self.rep.colorbar = ColorbarSpec::Enabled(enabled);
        self
    }

    /// Enables the colorbar legend with a custom title and [`ColorbarPosition`].
    pub fn colorbar_with(
        &mut self,
        title: Option<impl Into<String>>,
        position: ColorbarPosition,
    ) -> &mut Self {
        self.rep.colorbar = ColorbarSpec::Options(ColorbarOptions {
            title: title.map(Into::into),
            position,
        });
        self
    }

    /// Sets the wireframe / element-edge color for `"surface_with_edges"` and `"wireframe"`.
    pub fn edge_color(&mut self, color: impl Into<ColorSpec>) -> &mut Self {
        self.rep.edge_color = color.into();
        self
    }

    /// Sets the screen-space point diameter in pixels for `"points"`.
    pub fn point_size(&mut self, size: f64) -> &mut Self {
        self.rep.point_size = size;
        self
    }

    /// Sets the piecewise-linear opacity transfer function control points `[[t, alpha], ...]` for `"volume"`.
    pub fn transfer_function(&mut self, points: impl IntoIterator<Item = [f64; 2]>) -> &mut Self {
        self.rep.transfer_function = Some(points.into_iter().collect());
        self
    }

    /// Sets whether this representation is visible.
    pub fn visible(&mut self, visible: bool) -> &mut Self {
        self.rep.visible = visible;
        self
    }
}

/// Fluent builder for `view.camera` ([`CameraConfig`]).
#[derive(Debug)]
pub struct CameraBuilder<'a> {
    camera: &'a mut CameraConfig,
}

impl<'a> CameraBuilder<'a> {
    /// Sets the target bounds to frame (`"all"` or a source/filter name or [`DatasetRef`]).
    pub fn fit(&mut self, target: impl Into<String>) -> &mut Self {
        self.camera.fit = target.into();
        self
    }

    /// Sets the horizontal orbit azimuth angle in degrees.
    pub fn azimuth(&mut self, degrees: f64) -> &mut Self {
        self.camera.azimuth = degrees;
        self
    }

    /// Sets the vertical orbit elevation angle in degrees.
    pub fn elevation(&mut self, degrees: f64) -> &mut Self {
        self.camera.elevation = degrees;
        self
    }

    /// Sets both `azimuth` and `elevation` orbit angles in degrees.
    pub fn orbit(&mut self, azimuth_deg: f64, elevation_deg: f64) -> &mut Self {
        self.camera.azimuth = azimuth_deg;
        self.camera.elevation = elevation_deg;
        self
    }

    /// Sets the camera roll angle in degrees.
    pub fn roll(&mut self, degrees: f64) -> &mut Self {
        self.camera.roll = degrees;
        self
    }

    /// Sets the zoom factor (`> 0.0`; values `> 1.0` move closer).
    pub fn zoom(&mut self, zoom: f64) -> &mut Self {
        self.camera.zoom = zoom;
        self
    }

    /// Sets an explicit world-space camera eye position `[x, y, z]`.
    pub fn position(&mut self, pos: [f64; 3]) -> &mut Self {
        self.camera.position = Some(pos);
        self
    }

    /// Sets an explicit world-space camera focal point `[x, y, z]`.
    pub fn focal_point(&mut self, focal: [f64; 3]) -> &mut Self {
        self.camera.focal_point = Some(focal);
        self
    }

    /// Sets an explicit world-space camera up vector `[x, y, z]`.
    pub fn view_up(&mut self, up: [f64; 3]) -> &mut Self {
        self.camera.view_up = Some(up);
        self
    }

    /// Configures an explicit `position`, `focal_point`, and `view_up` pose.
    pub fn look_at(
        &mut self,
        position: [f64; 3],
        focal_point: [f64; 3],
        view_up: [f64; 3],
    ) -> &mut Self {
        self.camera.position = Some(position);
        self.camera.focal_point = Some(focal_point);
        self.camera.view_up = Some(view_up);
        self
    }

    /// Sets the camera projection mode (`Perspective` or `Orthographic`).
    pub fn projection(&mut self, projection: ProjectionKind) -> &mut Self {
        self.camera.projection = projection;
        self
    }

    /// Selects perspective projection (`ProjectionKind::Perspective`).
    pub fn perspective(&mut self) -> &mut Self {
        self.camera.projection = ProjectionKind::Perspective;
        self
    }

    /// Selects parallel orthographic projection (`ProjectionKind::Orthographic`).
    pub fn orthographic(&mut self) -> &mut Self {
        self.camera.projection = ProjectionKind::Orthographic;
        self
    }

    /// Sets the vertical field-of-view angle in degrees (`1.0..180.0`).
    pub fn view_angle(&mut self, degrees: f64) -> &mut Self {
        self.camera.view_angle = degrees;
        self
    }
}

/// Fluent builder for `output` ([`Output`]).
#[derive(Debug)]
pub struct OutputBuilder<'a> {
    output: &'a mut Output,
}

impl<'a> OutputBuilder<'a> {
    /// Sets the output image dimensions `[width, height]` in pixels.
    pub fn size(&mut self, width: u32, height: u32) -> &mut Self {
        self.output.size = [width, height];
        self
    }

    /// Sets the time-step selector (`TimeSelector::FIRST`, `TimeSelector::LAST`, `TimeSelector::step(n)`, or `TimeSelector::Value(t)`).
    pub fn time(&mut self, time: TimeSelector) -> &mut Self {
        self.output.time = time;
        self
    }

    /// Sets animation frame sequence output (`FramesConfig`).
    pub fn frames(&mut self, frames: FramesConfig) -> &mut Self {
        self.output.frames = Some(frames);
        self
    }

    /// Sets animation frame sequence output spanning all time steps with `pattern` (e.g. `"frame_%04d.png"`).
    pub fn all_frames(&mut self, pattern: impl Into<String>) -> &mut Self {
        self.output.frames = Some(FramesConfig::all(pattern));
        self
    }

    /// Sets the multisample anti-aliasing sample count (`1` or `4`).
    pub fn msaa(&mut self, samples: u32) -> &mut Self {
        self.output.msaa = samples;
        self
    }

    /// Sets the render engine (`RenderEngine::Raster` or `RenderEngine::Cycles`).
    pub fn engine(&mut self, engine: RenderEngine) -> &mut Self {
        self.output.engine = engine;
        self
    }

    /// Selects the headless Blender Cycles engine with `samples` path-tracing samples.
    pub fn cycles(&mut self, samples: u32) -> &mut Self {
        self.output.engine = RenderEngine::Cycles;
        let cfg = self.output.cycles.get_or_insert_with(|| CyclesConfig {
            samples: DEFAULT_CYCLES_SAMPLES,
            device: "auto".to_string(),
            save_blend: None,
        });
        cfg.samples = samples;
        self
    }

    /// Sets full [`CyclesConfig`] options and switches `engine` to [`RenderEngine::Cycles`].
    pub fn cycles_config(&mut self, config: CyclesConfig) -> &mut Self {
        self.output.engine = RenderEngine::Cycles;
        self.output.cycles = Some(config);
        self
    }
}

impl Source {
    /// Creates a [`Source`] for `file` with default format inference and `linearize_level = 2`.
    ///
    /// Alias for [`Source::new`] matching the builder examples in `docs/design-plan.md`.
    #[must_use]
    pub fn file(file: impl Into<String>) -> Self {
        Self::new(file)
    }

    /// Sets an explicit [`SourceFormat`] override.
    #[must_use]
    pub const fn with_format(mut self, format: SourceFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Sets the higher-order reference-space subdivision level (`1..=8`).
    #[must_use]
    pub const fn with_linearize_level(mut self, level: u32) -> Self {
        self.linearize_level = level;
        self
    }
}

impl From<&str> for ColorSpec {
    fn from(name: &str) -> Self {
        Self::Named(name.to_string())
    }
}

impl From<String> for ColorSpec {
    fn from(name: String) -> Self {
        Self::Named(name)
    }
}

impl From<[f64; 3]> for ColorSpec {
    fn from(rgb: [f64; 3]) -> Self {
        Self::Rgb(rgb)
    }
}

impl Scene {
    /// Creates a new [`SceneBuilder`] for constructing a v1 [`Scene`] fluently.
    #[must_use]
    pub fn builder() -> SceneBuilder {
        SceneBuilder::new()
    }
}

fn strip_component_suffix(spec: &str) -> &str {
    let parts: Vec<&str> = spec.split(':').collect();
    match parts.as_slice() {
        [assoc @ ("point" | "cell"), name, _comp] => {
            let prefix_len = assoc.len() + 1 + name.len();
            &spec[..prefix_len]
        }
        [assoc @ ("point" | "cell"), _name] => {
            let _ = assoc;
            spec
        }
        [name, _comp] => &spec[..name.len()],
        _ => spec,
    }
}

fn sanitize_snake_case_stem(path: &str, fallback: &str) -> String {
    let filename = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let stem = filename
        .split('.')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback);
    let mut out = String::with_capacity(stem.len());
    let mut prev_underscore = false;
    for ch in stem.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if !out.is_empty() && !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.is_empty() || out.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("{fallback}_{out}")
            .trim_end_matches('_')
            .to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_builder_constructs_design_plan_scene_and_roundtrips_json() {
        let mut scene = Scene::builder();
        let flow = scene.source(
            "flow",
            Source::file("results.pvd")
                .with_format(SourceFormat::Pvd)
                .with_linearize_level(3),
        );
        let cut = scene.filter(
            "cut",
            SliceFilter::new(&flow, [0.0, 0.0, 1.0]).origin([0.5, 0.5, 0.5]),
        );
        let iso = scene.filter(
            "iso",
            ContourFilter::new(&flow, "pressure", [101_325.0]).value(102_000.0),
        );

        scene
            .show(&cut)
            .surface_with_edges()
            .color_by("velocity")
            .magnitude()
            .colormap(ColormapName::Viridis)
            .range_bounds(0.0, 10.0)
            .colorbar_with(Some("Speed [m/s]"), ColorbarPosition::Right)
            .edge_color("black");
        scene
            .show(&iso)
            .surface()
            .opacity(0.4)
            .color([0.9, 0.2, 0.1]);

        scene
            .camera()
            .fit("all")
            .orbit(30.0, 20.0)
            .zoom(1.1)
            .perspective()
            .view_angle(35.0);
        scene
            .lights(LightingPreset::Default)
            .background_gradient("skyblue", [1.0, 1.0, 1.0])
            .axes(true)
            .title("Flow Slice & Isobar");
        scene
            .output("out.png")
            .size(1600, 900)
            .time(TimeSelector::LAST)
            .msaa(4)
            .cycles(64);

        let built = scene.build_validated().expect("valid scene");
        assert_eq!(built.sources["flow"].file, "results.pvd");
        assert_eq!(built.sources["flow"].format, Some(SourceFormat::Pvd));
        assert_eq!(built.sources["flow"].linearize_level, 3);
        assert_eq!(
            built.filters["cut"],
            Filter::slice_at("flow", [0.5, 0.5, 0.5], [0.0, 0.0, 1.0])
        );
        assert_eq!(
            built.filters["iso"],
            Filter::contour("flow", "pressure", vec![101_325.0, 102_000.0])
        );
        assert_eq!(
            built.view.show[0].color_by.as_deref(),
            Some("velocity:magnitude")
        );
        assert_eq!(built.view.show[1].opacity, 0.4);
        assert_eq!(built.view.camera.azimuth, 30.0);
        assert_eq!(built.view.camera.elevation, 20.0);
        assert_eq!(built.view.title.as_deref(), Some("Flow Slice & Isobar"));

        let out = built.output.as_ref().expect("output present");
        assert_eq!(out.size, [1600, 900]);
        assert_eq!(out.time, TimeSelector::LAST);
        assert_eq!(out.engine, RenderEngine::Cycles);
        assert_eq!(out.cycles.as_ref().unwrap().samples, 64);

        let json = built.to_json_string_pretty().expect("serialize");
        let reparsed = Scene::from_json_str(&json).expect("reparse");
        assert_eq!(reparsed, built);
    }

    #[test]
    fn auto_named_sources_and_all_filter_builders_produce_valid_snake_case_scenes() {
        let mut scene = Scene::builder();
        let src1 = scene.add_source(Source::file("testdata/vtk/mixed-linear.vtu"));
        let src2 = scene.add_source(Source::file("testdata/vtk/mixed-linear.vtu"));
        assert_eq!(src1.name(), "mixed_linear");
        assert_eq!(src2.name(), "mixed_linear_2");

        let hull = scene.add_filter(SurfaceFilter::new(&src1));
        let clipped = scene.add_filter(
            ClipFilter::new(&src1, [1.0, 0.0, 0.0])
                .origin([0.5, 0.5, 0.5])
                .keep(ClipKeep::Back)
                .crinkle(),
        );
        let thresh = scene.add_filter(
            ThresholdFilter::new(&clipped, "cell:cell_id")
                .min(1.0)
                .max(3.0),
        );
        let trace = scene.add_filter(
            StreamlinesFilter::new(
                &thresh,
                "velocity",
                StreamlineSeedsSpec::line([0.1, 0.1, 0.1], [0.9, 0.9, 0.9], 6),
            )
            .max_length(4.0)
            .step_size(0.05)
            .direction(StreamlineDirectionSpec::Both)
            .integrator(StreamlineIntegratorSpec::Rk45)
            .tube_radius(0.02),
        );
        let arrows = scene.add_filter(
            GlyphsFilter::new(&thresh, "velocity")
                .shape(GlyphShapeSpec::Cone)
                .scale_by(GlyphScaleBySpec::fixed(0.15))
                .scale_factor(1.2)
                .every(2),
        );
        let vol = scene.add_filter(ResampleFilter::new(&thresh, [16, 16, 16]));

        scene.show(&hull).wireframe().edge_color("dimgray");
        scene
            .show(&trace)
            .tubes()
            .point_field("velocity")
            .component("z")
            .colormap(ColormapName::CoolWarm)
            .center(0.0);
        scene
            .show(&arrows)
            .surface()
            .cell_field("cell_id")
            .colormap(ColormapName::Cividis);
        scene
            .show(&vol)
            .volume()
            .color_by("linear")
            .transfer_function([[0.0, 0.0], [0.5, 0.25], [1.0, 0.9]]);

        let built = scene
            .build_validated()
            .expect("valid scene with all filters");
        assert_eq!(built.view.show.len(), 4);
        assert_eq!(
            built.view.show[1].color_by.as_deref(),
            Some("point:velocity:z")
        );
        assert_eq!(built.view.show[2].color_by.as_deref(), Some("cell:cell_id"));
    }
}

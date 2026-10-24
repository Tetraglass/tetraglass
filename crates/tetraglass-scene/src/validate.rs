//! Semantic scene validation with JSON paths and edit-distance suggestions (`docs/scene-format.md`).
//!
//! Validation runs after parsing a [`Scene`] and checks both structural invariants
//! (version, lowercase `snake_case` names, filter graph acyclicity, camera/color/output ranges)
//! and data-dependent invariants against loaded datasets (array existence, association disambiguation,
//! component compatibility, and positive ranges for `log_scale`).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tetraglass_core::{
    Attributes, ComponentCount, DataArray, SurfaceMesh, UnstructuredGrid, ValueRange,
};
use tetraglass_io::detect_format_from_extension;

use crate::{
    ArrayAssociation, BackgroundSpec, ColorSpec, Filter, ParsedArrayRef, RangeSpec, Representation,
    ResolvedComponent, SCENE_VERSION, Scene, TimeSelector,
};

/// Metadata for a single attribute array used during semantic validation.
#[derive(Debug, Clone, PartialEq)]
pub struct ArrayMetadata {
    /// Number of components per tuple (`1`, `3`, `6`, or `9`).
    pub components: ComponentCount,
    /// Cached `[min, max]` range for each individual component.
    pub component_ranges: Vec<Option<ValueRange>>,
    /// Cached `[min, max]` range for the Euclidean magnitude.
    pub magnitude_range: Option<ValueRange>,
}

impl ArrayMetadata {
    /// Extracts [`ArrayMetadata`] from a [`DataArray`].
    #[must_use]
    pub fn from_data_array(array: &DataArray) -> Self {
        let count = array.components().as_usize();
        let component_ranges = (0..count).map(|c| array.component_range(c)).collect();
        Self {
            components: array.components(),
            component_ranges,
            magnitude_range: array.magnitude_range(),
        }
    }

    /// Returns the cached [`ValueRange`] for a [`ResolvedComponent`], if non-empty.
    #[must_use]
    pub fn range_for(&self, resolved: ResolvedComponent) -> Option<ValueRange> {
        match resolved {
            ResolvedComponent::Magnitude => self.magnitude_range,
            ResolvedComponent::Component(idx) => self.component_ranges.get(idx).copied().flatten(),
        }
    }
}

/// Summary of point and cell attribute arrays and time steps on a dataset used for scene validation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DatasetMetadata {
    /// Point attribute arrays keyed by name.
    pub point_arrays: BTreeMap<String, ArrayMetadata>,
    /// Cell attribute arrays keyed by name.
    pub cell_arrays: BTreeMap<String, ArrayMetadata>,
    /// Available simulation time steps when the dataset is a time series (empty if unspecified).
    pub time_steps: Vec<f64>,
}

impl DatasetMetadata {
    /// Builds [`DatasetMetadata`] from an [`UnstructuredGrid`].
    #[must_use]
    pub fn from_grid(grid: &UnstructuredGrid) -> Self {
        Self::from_attributes(grid.point_data(), grid.cell_data())
    }

    /// Builds [`DatasetMetadata`] from a [`SurfaceMesh`].
    #[must_use]
    pub fn from_surface(surface: &SurfaceMesh) -> Self {
        Self::from_attributes(surface.point_data(), surface.cell_data())
    }

    /// Attaches the available simulation time steps to this [`DatasetMetadata`].
    #[must_use]
    pub fn with_time_steps(mut self, time_steps: Vec<f64>) -> Self {
        self.time_steps = time_steps;
        self
    }

    fn from_attributes(point_data: &Attributes, cell_data: &Attributes) -> Self {
        let point_arrays = point_data
            .iter()
            .map(|(name, arr)| (name.to_string(), ArrayMetadata::from_data_array(arr)))
            .collect();
        let cell_arrays = cell_data
            .iter()
            .map(|(name, arr)| (name.to_string(), ArrayMetadata::from_data_array(arr)))
            .collect();
        Self {
            point_arrays,
            cell_arrays,
            time_steps: Vec::new(),
        }
    }

    /// Returns all unique array names (point arrays first, then cell arrays) in deterministic order.
    #[must_use]
    pub fn all_array_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.point_arrays.keys().map(String::as_str).collect();
        for k in self.cell_arrays.keys() {
            if !self.point_arrays.contains_key(k) {
                names.push(k.as_str());
            }
        }
        names
    }
}

/// A single semantic validation error anchored at a JSON path (`docs/scene-format.md`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct ValidationIssue {
    /// Dot-and-bracket JSON path to the invalid field (for example `"view.show[0].color_by"`).
    pub path: String,
    /// Explanation of why the value at [`Self::path`] is invalid.
    pub message: String,
    /// Optional fix or edit-distance suggestion (for example `"did you mean \"pressure\"?"`).
    #[serde(default)]
    pub suggestion: Option<String>,
}

impl ValidationIssue {
    /// Creates a new [`ValidationIssue`] with an optional suggestion.
    #[must_use]
    pub fn new(
        path: impl Into<String>,
        message: impl Into<String>,
        suggestion: Option<String>,
    ) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
            suggestion,
        }
    }
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)?;
        if let Some(suggestion) = &self.suggestion {
            write!(f, "; {suggestion}")?;
        }
        Ok(())
    }
}

/// Collection of one or more [`ValidationIssue`]s returned when scene validation fails.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct ValidationReport {
    /// Ordered list of validation issues found in the scene.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Returns `true` if no validation issues were recorded.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.issues.is_empty()
    }

    /// Returns the slice of recorded [`ValidationIssue`]s.
    #[must_use]
    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (idx, issue) in self.issues.iter().enumerate() {
            if idx > 0 {
                writeln!(f)?;
            }
            write!(f, "{issue}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationReport {}

/// Validates a [`Scene`] structurally (without requiring loaded mesh data).
///
/// Checks schema version, `snake_case` dataset naming, source file extensions,
/// filter input existence and cycle freedom, representation parameters, colors,
/// camera parameters, background colors, and output settings.
pub fn validate_scene_structure(scene: &Scene) -> Result<(), ValidationReport> {
    validate_impl(scene, None)
}

/// Validates a [`Scene`] both structurally and semantically against loaded source [`UnstructuredGrid`]s.
///
/// In addition to all checks in [`validate_scene_structure`], verifies that every source in
/// `scene.sources` is present in `sources`, resolves filter outputs back to their source metadata,
/// and validates every `color_by` array reference, component selector, and `log_scale` data range.
pub fn validate_scene(
    scene: &Scene,
    sources: &BTreeMap<String, UnstructuredGrid>,
) -> Result<(), ValidationReport> {
    let metadata: BTreeMap<String, DatasetMetadata> = sources
        .iter()
        .map(|(k, grid)| (k.clone(), DatasetMetadata::from_grid(grid)))
        .collect();
    validate_scene_with_metadata(scene, &metadata)
}

/// Validates a [`Scene`] against pre-extracted [`DatasetMetadata`] for each source.
pub fn validate_scene_with_metadata(
    scene: &Scene,
    sources: &BTreeMap<String, DatasetMetadata>,
) -> Result<(), ValidationReport> {
    validate_impl(scene, Some(sources))
}

fn validate_impl(
    scene: &Scene,
    loaded_sources: Option<&BTreeMap<String, DatasetMetadata>>,
) -> Result<(), ValidationReport> {
    let mut issues = Vec::new();

    // 1. Version
    if scene.version != SCENE_VERSION {
        issues.push(ValidationIssue::new(
            "version",
            format!("unsupported scene version {}", scene.version),
            Some(format!("supported version is {SCENE_VERSION}")),
        ));
    }

    // 2. Sources
    if scene.sources.is_empty() {
        issues.push(ValidationIssue::new(
            "sources",
            "at least one source dataset must be defined",
            None,
        ));
    }

    for (name, source) in &scene.sources {
        let source_path = format!("sources.{name}");
        if !is_valid_snake_case_name(name) {
            let suggested = to_snake_case_suggestion(name);
            let suggestion = if suggested.is_empty() || suggested == *name {
                Some("names must be lowercase snake_case (a-z, 0-9, _)".to_string())
            } else {
                Some(format!("did you mean \"{suggested}\"?"))
            };
            issues.push(ValidationIssue::new(
                &source_path,
                format!("invalid source name \"{name}\"; names must be lowercase snake_case"),
                suggestion,
            ));
        }

        let file_path = format!("sources.{name}.file");
        let trimmed_file = source.file.trim();
        if trimmed_file.is_empty() {
            issues.push(ValidationIssue::new(
                &file_path,
                "source file path must not be empty",
                None,
            ));
        } else if source.format.is_none() {
            let ext = Path::new(trimmed_file)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            if detect_format_from_extension(ext).is_none() {
                let msg = if ext.is_empty() {
                    format!(
                        "cannot infer mesh format from \"{trimmed_file}\" (missing file extension)"
                    )
                } else {
                    format!("unknown mesh file extension \".{ext}\" in \"{trimmed_file}\"")
                };
                issues.push(ValidationIssue::new(
                    &file_path,
                    msg,
                    Some(format!(
                        "use \".vtu\", \".vtk\", \".pvd\", or \".msh\", or set sources.{name}.format"
                    )),
                ));
            }
        }

        if source.linearize_level == 0 {
            issues.push(ValidationIssue::new(
                format!("sources.{name}.linearize_level"),
                "linearize_level must be at least 1",
                Some("use the default 2 or a positive integer".to_string()),
            ));
        }

        if let Some(loaded) = loaded_sources
            && !loaded.contains_key(name)
        {
            issues.push(ValidationIssue::new(
                &file_path,
                format!("source \"{name}\" was not provided in loaded datasets"),
                None,
            ));
        }
    }

    // Collect all defined dataset names in deterministic order (sources first, then filters).
    let mut defined_datasets: Vec<&str> = scene.sources.keys().map(String::as_str).collect();
    for filter_name in scene.filters.keys() {
        if !scene.sources.contains_key(filter_name) {
            defined_datasets.push(filter_name.as_str());
        }
    }

    // 3. Filters: naming, uniqueness, input existence, and cycle detection.
    for (name, filter) in &scene.filters {
        let filter_path = format!("filters.{name}");
        if scene.sources.contains_key(name) {
            issues.push(ValidationIssue::new(
                &filter_path,
                format!("duplicate dataset name \"{name}\" (already defined in sources)"),
                Some("names must be unique across sources and filters".to_string()),
            ));
        } else if !is_valid_snake_case_name(name) {
            let suggested = to_snake_case_suggestion(name);
            let suggestion = if suggested.is_empty() || suggested == *name {
                Some("names must be lowercase snake_case (a-z, 0-9, _)".to_string())
            } else {
                Some(format!("did you mean \"{suggested}\"?"))
            };
            issues.push(ValidationIssue::new(
                &filter_path,
                format!("invalid filter name \"{name}\"; names must be lowercase snake_case"),
                suggestion,
            ));
        }

        let input = filter.input();
        let input_path = format!("filters.{name}.input");
        if input == name {
            issues.push(ValidationIssue::new(
                &input_path,
                format!("filter \"{name}\" cannot use itself as input"),
                suggest_dataset_or_list("", &defined_datasets, Some(name)),
            ));
        } else if !scene.sources.contains_key(input) && !scene.filters.contains_key(input) {
            issues.push(ValidationIssue::new(
                &input_path,
                format!("unknown dataset \"{input}\""),
                suggest_dataset_or_list(input, &defined_datasets, Some(name)),
            ));
        } else if matches!(filter, Filter::Slice { .. } | Filter::Contour { .. })
            && matches!(
                scene.filters.get(input),
                Some(Filter::Surface { .. } | Filter::Slice { .. } | Filter::Contour { .. })
            )
        {
            let kind = match filter {
                Filter::Contour { .. } => "contour",
                _ => "slice",
            };
            let suggestion = resolve_root_source(name, &scene.sources, &scene.filters)
                .map(|src| format!("use volume source \"{src}\" as input instead"));
            issues.push(ValidationIssue::new(
                &input_path,
                format!(
                    "{kind} filter \"{name}\" requires a volume dataset input, but \"{input}\" produces a surface"
                ),
                suggestion,
            ));
        }

        match filter {
            Filter::Surface { .. } => {}
            Filter::Slice { origin, normal, .. } => {
                if let Some(orig) = origin
                    && !orig.iter().all(|v| v.is_finite())
                {
                    issues.push(ValidationIssue::new(
                        format!("filters.{name}.origin"),
                        "slice plane origin coordinates must be finite numbers",
                        None,
                    ));
                }

                if !normal.iter().all(|v| v.is_finite()) {
                    issues.push(ValidationIssue::new(
                        format!("filters.{name}.normal"),
                        "slice plane normal coordinates must be finite numbers",
                        None,
                    ));
                } else if normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
                    <= 1e-24
                {
                    issues.push(ValidationIssue::new(
                        format!("filters.{name}.normal"),
                        "must not be [0, 0, 0]",
                        Some(
                            "use a non-zero plane normal such as [1, 0, 0], [0, 1, 0], or [0, 0, 1]"
                                .to_string(),
                        ),
                    ));
                }
            }
            Filter::Contour { values, .. } => {
                if values.is_empty() {
                    issues.push(ValidationIssue::new(
                        format!("filters.{name}.values"),
                        "contour values list must not be empty",
                        Some("provide at least one finite isovalue such as [1.0]".to_string()),
                    ));
                } else if !values.iter().all(|v| v.is_finite()) {
                    issues.push(ValidationIssue::new(
                        format!("filters.{name}.values"),
                        "contour isovalues must be finite numbers",
                        None,
                    ));
                }
            }
        }
    }

    // Detect cycles in the filter graph.
    detect_filter_cycles(&scene.filters, &mut issues);

    // Resolve dataset metadata for all valid sources and filters when loaded_sources is provided.
    let resolved_datasets: Option<BTreeMap<&str, &DatasetMetadata>> =
        loaded_sources.map(|loaded| {
            let mut map: BTreeMap<&str, &DatasetMetadata> = BTreeMap::new();
            for (src_name, meta) in loaded {
                if scene.sources.contains_key(src_name) {
                    map.insert(src_name.as_str(), meta);
                }
            }
            for filter_name in scene.filters.keys() {
                if let Some(root_source) =
                    resolve_root_source(filter_name, &scene.sources, &scene.filters)
                    && let Some(meta) = loaded.get(root_source)
                {
                    map.insert(filter_name.as_str(), meta);
                }
            }
            map
        });

    // Validate filter field references (such as `contour.field`) against resolved dataset metadata.
    for (name, filter) in &scene.filters {
        if let Filter::Contour { input, field, .. } = filter {
            let dataset_meta = resolved_datasets
                .as_ref()
                .and_then(|map| map.get(input.as_str()).copied());
            validate_contour_field(name, field, input, dataset_meta, &mut issues);
        }
    }

    // 4. View representations
    if scene.view.show.is_empty() {
        issues.push(ValidationIssue::new(
            "view.show",
            "at least one representation must be listed in view.show",
            None,
        ));
    }

    for (idx, rep) in scene.view.show.iter().enumerate() {
        let rep_prefix = format!("view.show[{idx}]");

        let input_exists =
            scene.sources.contains_key(&rep.input) || scene.filters.contains_key(&rep.input);
        if !input_exists {
            issues.push(ValidationIssue::new(
                format!("{rep_prefix}.input"),
                format!("unknown dataset \"{}\"", rep.input),
                suggest_dataset_or_list(&rep.input, &defined_datasets, None),
            ));
        }

        validate_color_spec(&rep.color, &format!("{rep_prefix}.color"), &mut issues);
        validate_color_spec(
            &rep.edge_color,
            &format!("{rep_prefix}.edge_color"),
            &mut issues,
        );

        if !rep.opacity.is_finite() || !(0.0..=1.0).contains(&rep.opacity) {
            issues.push(ValidationIssue::new(
                format!("{rep_prefix}.opacity"),
                format!("opacity must be in 0.0..=1.0, got {}", rep.opacity),
                Some("set opacity between 0.0 (transparent) and 1.0 (opaque)".to_string()),
            ));
        }

        if !rep.point_size.is_finite() || rep.point_size <= 0.0 {
            issues.push(ValidationIssue::new(
                format!("{rep_prefix}.point_size"),
                format!("point_size must be positive, got {}", rep.point_size),
                Some("use a positive pixel size such as 3.0".to_string()),
            ));
        }

        if let Some(center) = rep.center
            && !center.is_finite()
        {
            issues.push(ValidationIssue::new(
                format!("{rep_prefix}.center"),
                "center value must be a finite number",
                None,
            ));
        }

        if let RangeSpec::Bounds([min, max]) = rep.range {
            if !min.is_finite() || !max.is_finite() {
                issues.push(ValidationIssue::new(
                    format!("{rep_prefix}.range"),
                    "range bounds must be finite numbers",
                    None,
                ));
            } else if min > max {
                issues.push(ValidationIssue::new(
                    format!("{rep_prefix}.range"),
                    format!("range minimum ({min}) must not exceed maximum ({max})"),
                    Some(format!("did you mean [{max}, {min}]?")),
                ));
            } else if rep.log_scale && min <= 0.0 {
                issues.push(ValidationIssue::new(
                    format!("{rep_prefix}.log_scale"),
                    format!(
                        "log_scale requires a strictly positive range, but range is [{min}, {max}]"
                    ),
                    Some(
                        "set log_scale to false or choose a positive range [min, max]".to_string(),
                    ),
                ));
            }
        }

        if let Some(raw_color_by) = &rep.color_by {
            validate_color_by(
                raw_color_by,
                rep,
                &rep_prefix,
                resolved_datasets
                    .as_ref()
                    .and_then(|map| map.get(rep.input.as_str()).copied()),
                &mut issues,
            );
        }
    }

    // 5. Camera
    let cam = &scene.view.camera;
    if cam.position.is_none()
        && cam.fit != "all"
        && !scene.sources.contains_key(&cam.fit)
        && !scene.filters.contains_key(&cam.fit)
    {
        let mut fit_candidates = Vec::with_capacity(defined_datasets.len() + 1);
        fit_candidates.push("all");
        fit_candidates.extend(defined_datasets.iter().copied());
        issues.push(ValidationIssue::new(
            "view.camera.fit",
            format!("unknown fit target \"{}\"", cam.fit),
            suggest_dataset_or_list(&cam.fit, &fit_candidates, None),
        ));
    }

    for (field_name, angle) in [
        ("azimuth", cam.azimuth),
        ("elevation", cam.elevation),
        ("roll", cam.roll),
    ] {
        if !angle.is_finite() {
            issues.push(ValidationIssue::new(
                format!("view.camera.{field_name}"),
                format!("{field_name} must be a finite number"),
                None,
            ));
        }
    }

    if !cam.zoom.is_finite() || cam.zoom <= 0.0 {
        issues.push(ValidationIssue::new(
            "view.camera.zoom",
            format!("zoom must be positive, got {}", cam.zoom),
            Some("use 1.0 for default framing or > 1.0 to zoom in".to_string()),
        ));
    }

    if !cam.view_angle.is_finite() || cam.view_angle <= 0.0 || cam.view_angle >= 180.0 {
        issues.push(ValidationIssue::new(
            "view.camera.view_angle",
            format!(
                "view_angle must be between 0 and 180 degrees, got {}",
                cam.view_angle
            ),
            Some("use a perspective field-of-view angle such as 30".to_string()),
        ));
    }

    if let Some(pos) = cam.position
        && !pos.iter().all(|v| v.is_finite())
    {
        issues.push(ValidationIssue::new(
            "view.camera.position",
            "camera position coordinates must be finite numbers",
            None,
        ));
    }

    if let Some(fp) = cam.focal_point
        && !fp.iter().all(|v| v.is_finite())
    {
        issues.push(ValidationIssue::new(
            "view.camera.focal_point",
            "camera focal_point coordinates must be finite numbers",
            None,
        ));
    }

    if let (Some(pos), Some(fp)) = (cam.position, cam.focal_point) {
        let dx = pos[0] - fp[0];
        let dy = pos[1] - fp[1];
        let dz = pos[2] - fp[2];
        if dx * dx + dy * dy + dz * dz <= 1e-24 {
            issues.push(ValidationIssue::new(
                "view.camera.position",
                "camera position and focal_point must not be identical",
                None,
            ));
        }
    }

    if let Some(up) = cam.view_up {
        if !up.iter().all(|v| v.is_finite()) {
            issues.push(ValidationIssue::new(
                "view.camera.view_up",
                "camera view_up coordinates must be finite numbers",
                None,
            ));
        } else if up[0] * up[0] + up[1] * up[1] + up[2] * up[2] <= 1e-24 {
            issues.push(ValidationIssue::new(
                "view.camera.view_up",
                "must not be [0, 0, 0]",
                Some("use a non-zero up vector such as [0, 1, 0] or [0, 0, 1]".to_string()),
            ));
        }
    }

    // 6. Background
    match &scene.view.background {
        BackgroundSpec::Solid(color) => {
            validate_color_spec(color, "view.background", &mut issues);
        }
        BackgroundSpec::Gradient(grad) => {
            validate_color_spec(&grad.top, "view.background.top", &mut issues);
            validate_color_spec(&grad.bottom, "view.background.bottom", &mut issues);
        }
    }

    // 7. Output
    if let Some(output) = &scene.output {
        if output.image.trim().is_empty() {
            issues.push(ValidationIssue::new(
                "output.image",
                "output image path must not be empty",
                None,
            ));
        }

        if output.size[0] == 0 || output.size[1] == 0 {
            issues.push(ValidationIssue::new(
                "output.size",
                format!(
                    "output image dimensions must both be positive, got [{}, {}]",
                    output.size[0], output.size[1]
                ),
                Some("use pixel dimensions such as [1200, 800]".to_string()),
            ));
        }

        if output.msaa != 1 && output.msaa != 4 {
            issues.push(ValidationIssue::new(
                "output.msaa",
                format!("unsupported MSAA sample count {}", output.msaa),
                Some("supported MSAA sample counts are 1 and 4".to_string()),
            ));
        }

        if let TimeSelector::Value(t) = output.time
            && !t.is_finite()
        {
            issues.push(ValidationIssue::new(
                "output.time",
                "output time value must be a finite number",
                None,
            ));
        } else if let Some(loaded) = loaded_sources {
            for (src_name, meta) in loaded {
                if !scene.sources.contains_key(src_name) || meta.time_steps.is_empty() {
                    continue;
                }
                if let Err(err_msg) = output.time.resolve_step(&meta.time_steps) {
                    let path = match output.time {
                        TimeSelector::Step(_) => "output.time.step",
                        _ => "output.time",
                    };
                    let suggestion = Some(format!(
                        "use \"first\", \"last\", a step in 0..={}, or a time in [{}, {}]",
                        meta.time_steps.len().saturating_sub(1),
                        meta.time_steps.first().copied().unwrap_or(0.0),
                        meta.time_steps.last().copied().unwrap_or(0.0)
                    ));
                    issues.push(ValidationIssue::new(
                        path,
                        format!("on source \"{src_name}\": {err_msg}"),
                        suggestion,
                    ));
                    break;
                }
            }
        }

        if let Some(frames) = &output.frames {
            if frames.pattern.trim().is_empty() {
                issues.push(ValidationIssue::new(
                    "output.frames.pattern",
                    "animation filename pattern must not be empty",
                    Some("use a pattern such as \"frame_%04d.png\"".to_string()),
                ));
            }
            if let TimeSelector::Value(t) = frames.from
                && !t.is_finite()
            {
                issues.push(ValidationIssue::new(
                    "output.frames.from",
                    "frame start time must be a finite number",
                    None,
                ));
            }
            if let TimeSelector::Value(t) = frames.to
                && !t.is_finite()
            {
                issues.push(ValidationIssue::new(
                    "output.frames.to",
                    "frame end time must be a finite number",
                    None,
                ));
            }
        }

        if let Some(cycles) = &output.cycles {
            if cycles.samples == 0 {
                issues.push(ValidationIssue::new(
                    "output.cycles.samples",
                    "Cycles sample count must be at least 1",
                    Some("use the default 128 or a positive integer".to_string()),
                ));
            }
            if cycles.device.trim().is_empty() {
                issues.push(ValidationIssue::new(
                    "output.cycles.device",
                    "Cycles device must not be empty",
                    Some("use \"auto\", \"cpu\", or \"gpu\"".to_string()),
                ));
            }
        }
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(ValidationReport { issues })
    }
}

fn validate_contour_field(
    filter_name: &str,
    raw: &str,
    input_name: &str,
    dataset_meta: Option<&DatasetMetadata>,
    issues: &mut Vec<ValidationIssue>,
) {
    let field_path = format!("filters.{filter_name}.field");

    let parsed = match ParsedArrayRef::parse(raw) {
        Ok(parsed) => parsed,
        Err(_) => {
            let suggestion = suggest_component_for_malformed_ref(raw);
            issues.push(ValidationIssue::new(
                &field_path,
                format!("invalid array reference \"{raw}\""),
                suggestion,
            ));
            return;
        }
    };

    if parsed.association == Some(ArrayAssociation::Cell) {
        issues.push(ValidationIssue::new(
            &field_path,
            "contour filter requires a point array, not a cell array",
            Some(format!(
                "use \"point:{}\" or a continuous point array",
                parsed.name
            )),
        ));
        return;
    }

    let Some(meta) = dataset_meta else {
        return;
    };

    let point_candidates: Vec<&str> = meta.point_arrays.keys().map(String::as_str).collect();
    let Some(array_meta) = meta.point_arrays.get(&parsed.name) else {
        if meta.cell_arrays.contains_key(&parsed.name) {
            issues.push(ValidationIssue::new(
                &field_path,
                format!(
                    "no point array \"{}\" on \"{input_name}\" (exists only as a cell array; contour requires a point array)",
                    parsed.name
                ),
                suggest_array_name(&parsed.name, &point_candidates),
            ));
        } else {
            issues.push(ValidationIssue::new(
                &field_path,
                format!("no point array \"{}\" on \"{input_name}\"", parsed.name),
                suggest_array_name(&parsed.name, &point_candidates),
            ));
        }
        return;
    };

    if parsed.resolve_component(array_meta.components).is_none() {
        let comp_str = parsed.component.map_or_else(String::new, |c| c.to_string());
        let suggestion = match array_meta.components {
            ComponentCount::Scalar => Some(format!(
                "\"{}\" is a scalar array (1 component); use \"{}\" without \":{comp_str}\"",
                parsed.name, parsed.name
            )),
            ComponentCount::Vector3 => Some(
                "valid components for a 3-component vector are \"x\", \"y\", \"z\", \"magnitude\", or 0..2"
                    .to_string(),
            ),
            ComponentCount::SymmetricTensor6 => Some(
                "valid components for a 6-component symmetric tensor are \"xx\"..\"zz\", \"magnitude\", or 0..5"
                    .to_string(),
            ),
            ComponentCount::Tensor9 => Some(
                "valid components for a 9-component tensor are \"xx\"..\"zz\", \"magnitude\", or 0..8"
                    .to_string(),
            ),
        };
        issues.push(ValidationIssue::new(
            &field_path,
            format!(
                "component \"{comp_str}\" is not valid for {}-component array \"{}\" on \"{input_name}\"",
                array_meta.components.as_usize(),
                parsed.name
            ),
            suggestion,
        ));
    }
}

fn validate_color_by(
    raw: &str,
    rep: &Representation,
    rep_prefix: &str,
    dataset_meta: Option<&DatasetMetadata>,
    issues: &mut Vec<ValidationIssue>,
) {
    let dataset_name = rep.input.as_str();
    let color_by_path = format!("{rep_prefix}.color_by");
    let log_scale_path = format!("{rep_prefix}.log_scale");

    let parsed = match ParsedArrayRef::parse(raw) {
        Ok(parsed) => parsed,
        Err(_) => {
            // Check if the user wrote a bad component selector so we can suggest a close match.
            let suggestion = suggest_component_for_malformed_ref(raw);
            issues.push(ValidationIssue::new(
                &color_by_path,
                format!("invalid array reference \"{raw}\""),
                suggestion,
            ));
            return;
        }
    };

    let Some(meta) = dataset_meta else {
        return;
    };

    let found_array: Option<(ArrayAssociation, &ArrayMetadata)> = match parsed.association {
        Some(ArrayAssociation::Point) => meta
            .point_arrays
            .get(&parsed.name)
            .map(|a| (ArrayAssociation::Point, a)),
        Some(ArrayAssociation::Cell) => meta
            .cell_arrays
            .get(&parsed.name)
            .map(|a| (ArrayAssociation::Cell, a)),
        None => meta
            .point_arrays
            .get(&parsed.name)
            .map(|a| (ArrayAssociation::Point, a))
            .or_else(|| {
                meta.cell_arrays
                    .get(&parsed.name)
                    .map(|a| (ArrayAssociation::Cell, a))
            }),
    };

    let Some((_assoc, array_meta)) = found_array else {
        // Check if an explicit `point:` or `cell:` qualifier looked in the wrong association table.
        if parsed.association == Some(ArrayAssociation::Point)
            && meta.cell_arrays.contains_key(&parsed.name)
        {
            let fixed = ParsedArrayRef {
                association: Some(ArrayAssociation::Cell),
                name: parsed.name.clone(),
                component: parsed.component,
            };
            issues.push(ValidationIssue::new(
                &color_by_path,
                format!(
                    "no point array \"{}\" on \"{dataset_name}\" (exists as a cell array)",
                    parsed.name
                ),
                Some(format!("did you mean \"{fixed}\"?")),
            ));
            return;
        }
        if parsed.association == Some(ArrayAssociation::Cell)
            && meta.point_arrays.contains_key(&parsed.name)
        {
            let fixed = ParsedArrayRef {
                association: Some(ArrayAssociation::Point),
                name: parsed.name.clone(),
                component: parsed.component,
            };
            issues.push(ValidationIssue::new(
                &color_by_path,
                format!(
                    "no cell array \"{}\" on \"{dataset_name}\" (exists as a point array)",
                    parsed.name
                ),
                Some(format!("did you mean \"{fixed}\"?")),
            ));
            return;
        }

        let candidates = meta.all_array_names();
        let suggestion = suggest_array_name(&parsed.name, &candidates);
        issues.push(ValidationIssue::new(
            &color_by_path,
            format!("no array \"{}\" on \"{dataset_name}\"", parsed.name),
            suggestion,
        ));
        return;
    };

    // Validate component compatibility with the array's ComponentCount.
    let Some(resolved_comp) = parsed.resolve_component(array_meta.components) else {
        let comp_str = parsed.component.map_or_else(String::new, |c| c.to_string());
        let suggestion = match array_meta.components {
            ComponentCount::Scalar => Some(format!(
                "\"{}\" is a scalar array (1 component); use \"{}\" without \":{comp_str}\"",
                parsed.name, parsed.name
            )),
            ComponentCount::Vector3 => Some(
                "valid components for a 3-component vector are \"x\", \"y\", \"z\", \"magnitude\", or 0..2"
                    .to_string(),
            ),
            ComponentCount::SymmetricTensor6 => Some(
                "valid components for a 6-component symmetric tensor are \"xx\"..\"zz\", \"magnitude\", or 0..5"
                    .to_string(),
            ),
            ComponentCount::Tensor9 => Some(
                "valid components for a 9-component tensor are \"xx\"..\"zz\", \"magnitude\", or 0..8"
                    .to_string(),
            ),
        };
        issues.push(ValidationIssue::new(
            &color_by_path,
            format!(
                "component \"{comp_str}\" is not valid for {}-component array \"{}\" on \"{dataset_name}\"",
                array_meta.components.as_usize(),
                parsed.name
            ),
            suggestion,
        ));
        return;
    };

    // If log_scale is enabled with an automatic range, check that the dataset values are positive.
    if rep.log_scale
        && matches!(rep.range, RangeSpec::Preset(_))
        && let Some(val_range) = array_meta.range_for(resolved_comp)
        && val_range.min <= 0.0
    {
        issues.push(ValidationIssue::new(
            &log_scale_path,
            format!(
                "log_scale requires positive values, but \"{raw}\" on \"{dataset_name}\" has range [{}, {}]",
                val_range.min, val_range.max
            ),
            Some("set log_scale to false or specify a positive range [min, max]".to_string()),
        ));
    }
}

fn suggest_component_for_malformed_ref(raw: &str) -> Option<String> {
    let (assoc_prefix, rest) = if let Some(r) = raw.strip_prefix("point:") {
        ("point:", r)
    } else if let Some(r) = raw.strip_prefix("cell:") {
        ("cell:", r)
    } else {
        ("", raw)
    };
    let (arr_name, comp_part) = rest.split_once(':')?;
    if arr_name.is_empty() || comp_part.is_empty() || comp_part.contains(':') {
        return Some(
            "expected \"<array>\", \"<array>:<component>\", \"point:<array>\", or \"cell:<array>\""
                .to_string(),
        );
    }
    const KNOWN_COMPONENTS: &[&str] = &[
        "x",
        "y",
        "z",
        "magnitude",
        "xx",
        "xy",
        "xz",
        "yx",
        "yy",
        "yz",
        "zx",
        "zy",
        "zz",
    ];
    if let Some(best) = closest_match(comp_part, KNOWN_COMPONENTS.iter().copied()) {
        Some(format!("did you mean \"{assoc_prefix}{arr_name}:{best}\"?"))
    } else {
        Some(
            "valid component names are \"x\", \"y\", \"z\", \"magnitude\", \"xx\"..\"zz\", or a zero-based index"
                .to_string(),
        )
    }
}

fn validate_color_spec(color: &ColorSpec, path: &str, issues: &mut Vec<ValidationIssue>) {
    if color.resolve_rgb().is_some() {
        return;
    }
    match color {
        ColorSpec::Named(name) => {
            let suggestion = closest_match(name, ColorSpec::known_color_names().iter().copied())
                .map(|best| format!("did you mean \"{best}\"?"))
                .or_else(|| {
                    Some("use a CSS color name (such as \"white\", \"lightgray\", \"black\"), \"#rrggbb\", or [r, g, b] in 0..1".to_string())
                });
            issues.push(ValidationIssue::new(
                path,
                format!("unknown color \"{name}\""),
                suggestion,
            ));
        }
        ColorSpec::Rgb([r, g, b]) => {
            issues.push(ValidationIssue::new(
                path,
                format!("RGB color [{r}, {g}, {b}] must have finite components in 0.0..=1.0"),
                if *r > 1.0 || *g > 1.0 || *b > 1.0 {
                    Some(format!(
                        "divide 0..255 values by 255: [{:.3}, {:.3}, {:.3}]",
                        (*r / 255.0).clamp(0.0, 1.0),
                        (*g / 255.0).clamp(0.0, 1.0),
                        (*b / 255.0).clamp(0.0, 1.0)
                    ))
                } else {
                    None
                },
            ));
        }
    }
}

fn detect_filter_cycles(filters: &BTreeMap<String, Filter>, issues: &mut Vec<ValidationIssue>) {
    let mut reported_in_cycle: BTreeSet<String> = BTreeSet::new();

    for start_name in filters.keys() {
        if reported_in_cycle.contains(start_name) {
            continue;
        }
        let mut chain: Vec<&str> = vec![start_name.as_str()];
        let mut current = start_name.as_str();

        while let Some(filter) = filters.get(current) {
            let next = filter.input();
            if next == current {
                // Self-cycle is already reported as `cannot use itself as input`.
                break;
            }
            if let Some(pos) = chain.iter().position(|&n| n == next) {
                let cycle_slice = &chain[pos..];
                let mut cycle_str = cycle_slice.join(" -> ");
                cycle_str.push_str(" -> ");
                cycle_str.push_str(next);
                let cycle_start = cycle_slice[0];
                if reported_in_cycle.insert(cycle_start.to_string()) {
                    for &member in cycle_slice {
                        reported_in_cycle.insert(member.to_string());
                    }
                    issues.push(ValidationIssue::new(
                        format!("filters.{cycle_start}.input"),
                        format!("filter cycle detected: {cycle_str}"),
                        Some(
                            "filters must form a directed acyclic graph rooted at sources"
                                .to_string(),
                        ),
                    ));
                }
                break;
            }
            if !filters.contains_key(next) {
                break;
            }
            chain.push(next);
            current = next;
        }
    }
}

fn resolve_root_source<'a>(
    start_filter: &'a str,
    sources: &'a BTreeMap<String, crate::Source>,
    filters: &'a BTreeMap<String, Filter>,
) -> Option<&'a str> {
    let mut visited: BTreeSet<&str> = BTreeSet::new();
    let mut current = start_filter;
    while let Some(filter) = filters.get(current) {
        if !visited.insert(current) {
            return None;
        }
        let input = filter.input();
        if sources.contains_key(input) {
            return Some(input);
        }
        current = input;
    }
    None
}

fn is_valid_snake_case_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    if !bytes[0].is_ascii_lowercase() {
        return false;
    }
    if bytes[bytes.len() - 1] == b'_' {
        return false;
    }
    let mut prev_underscore = false;
    for &b in bytes {
        if b == b'_' {
            if prev_underscore {
                return false;
            }
            prev_underscore = true;
        } else if b.is_ascii_lowercase() || b.is_ascii_digit() {
            prev_underscore = false;
        } else {
            return false;
        }
    }
    true
}

fn to_snake_case_suggestion(raw: &str) -> String {
    let mut out = String::new();
    let mut prev_underscore = false;
    for (i, ch) in raw.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 && !prev_underscore && !out.is_empty() {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if ch.is_ascii_lowercase() || (ch.is_ascii_digit() && !out.is_empty()) {
            out.push(ch);
            prev_underscore = false;
        } else if (ch == '_' || ch == '-' || ch == ' ') && !out.is_empty() && !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

fn suggest_array_name(query: &str, candidates: &[&str]) -> Option<String> {
    if candidates.is_empty() {
        return Some("dataset has no attribute arrays".to_string());
    }
    if let Some(best) = closest_match(query, candidates.iter().copied()) {
        Some(format!("did you mean \"{best}\"?"))
    } else {
        Some(format!("available are {}", format_quoted_list(candidates)))
    }
}

fn suggest_dataset_or_list(
    query: &str,
    candidates: &[&str],
    exclude: Option<&str>,
) -> Option<String> {
    let filtered: Vec<&str> = candidates
        .iter()
        .copied()
        .filter(|c| Some(*c) != exclude)
        .collect();
    if filtered.is_empty() {
        return None;
    }
    if !query.is_empty()
        && let Some(best) = closest_match(query, filtered.iter().copied())
    {
        return Some(format!(
            "did you mean \"{best}\"? (defined are {})",
            format_quoted_list(&filtered)
        ));
    }
    Some(format!("defined are {}", format_quoted_list(&filtered)))
}

fn format_quoted_list(items: &[&str]) -> String {
    items
        .iter()
        .map(|s| format!("\"{s}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Finds the closest candidate to `query` using case-insensitive Damerau-Levenshtein distance,
/// falling back to a unique prefix match when `query` has at least 3 characters.
///
/// Returns `None` if `candidates` is empty or the minimum edit distance exceeds a reasonable
/// typo threshold (`max(2, query.len() / 3)`).
#[must_use]
pub fn closest_match<'a>(
    query: &str,
    candidates: impl IntoIterator<Item = &'a str>,
) -> Option<&'a str> {
    if query.is_empty() {
        return None;
    }
    let query_lower = query.to_ascii_lowercase();
    let max_dist = (query_lower.len() / 3).max(2);

    let mut best: Option<(&'a str, usize)> = None;
    let mut prefix_match: Option<&'a str> = None;
    let mut prefix_count = 0usize;

    for candidate in candidates {
        let cand_lower = candidate.to_ascii_lowercase();
        let dist = damerau_levenshtein(&query_lower, &cand_lower);
        let max_len = query_lower.len().max(cand_lower.len());
        if dist <= max_dist && dist < max_len {
            match best {
                None => best = Some((candidate, dist)),
                Some((_, best_dist)) if dist < best_dist => best = Some((candidate, dist)),
                _ => {}
            }
        }
        if query_lower.len() >= 3 && cand_lower.starts_with(&query_lower) {
            prefix_match = Some(candidate);
            prefix_count += 1;
        }
    }
    if let Some((cand, _)) = best {
        Some(cand)
    } else if prefix_count == 1 {
        prefix_match
    } else {
        None
    }
}

/// Computes the optimal string alignment (restricted Damerau-Levenshtein) edit distance
/// between two strings, counting insertions, deletions, substitutions, and adjacent transpositions.
#[must_use]
pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let width = n + 1;
    let mut dp = vec![0usize; (m + 1) * width];
    for i in 0..=m {
        dp[i * width] = i;
    }
    for (j, cell) in dp.iter_mut().enumerate().take(n + 1) {
        *cell = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
            let deletion = dp[(i - 1) * width + j] + 1;
            let insertion = dp[i * width + (j - 1)] + 1;
            let substitution = dp[(i - 1) * width + (j - 1)] + cost;
            let mut best = deletion.min(insertion).min(substitution);

            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                let transposition = dp[(i - 2) * width + (j - 2)] + 1;
                best = best.min(transposition);
            }
            dp[i * width + j] = best;
        }
    }

    dp[m * width + n]
}

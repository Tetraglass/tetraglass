//! Interactive desktop (`eframe`) viewer for Tetraglass scenes and meshes (`M1.8`).
//!
//! # Architecture & invariants
//!
//! - [`ViewerDocument`] manages the serializable [`Scene`], [`SceneEvaluator`], lazy [`TimeSeries`]
//!   sources, time-step selection, field/colormap pickers, and [`SceneFileWatcher`] hot reload
//!   without depending on `wgpu`.
//! - [`ViewerSession`] and [`ViewerApp`] bridge [`ViewerDocument`] to [`tetraglass_egui::ViewportState`]
//!   and [`tetraglass_egui::ViewportWidget`]. All `wgpu` rendering remains inside `tetraglass-render`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant, SystemTime};

use eframe::egui;
use tetraglass_core::{BoundingBox, SurfaceMesh, TetView, TimeSeries};
use tetraglass_egui::{
    FieldOption, ViewportCamera, ViewportColorBy, ViewportColormap, ViewportScalarComponent,
    ViewportState, ViewportSurfaceStyle, ViewportWidget, available_field_options,
};
use tetraglass_filters::{ClipParams, clip, extract_tet_view_surface};
use tetraglass_io::{
    IoError, MeshFormat, detect_format_from_bytes, detect_format_from_extension, read_msh_series,
    read_pvd_series_with_loader, read_vtu_series,
};
use tetraglass_render::{
    Annotations, ColorbarAnnotation, ColorbarPlacement, DEFAULT_COLORBAR_TICKS, PickProbeValue,
    RenderError, SurfacePickResult,
};
use tetraglass_scene::{
    ArrayAssociation, BackgroundSpec, CameraConfig, ClipKeep, ClipMode, ColorSpec, ColorbarSpec,
    ColormapName, DEFAULT_MSAA_SAMPLES, DatasetMetadata, EvaluatedColoring,
    EvaluatedRepresentation, EvaluatedScene, Filter, LightingPreset, Output, Representation,
    RepresentationStyle, ResolvedComponent, Scene, SceneError, SceneEvaluator, Source,
    TimeSelector, View, validate_scene_with_metadata,
};
use thiserror::Error;

/// Default polling interval for checking scene and source files on disk during hot reload.
pub const DEFAULT_HOT_RELOAD_POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Default playback interval per time step when the time slider's Play toggle is active.
pub const DEFAULT_PLAYBACK_STEP_INTERVAL: Duration = Duration::from_millis(400);

/// All [`ColormapName`] options exposed in the viewer's colormap picker.
pub const VIEWER_COLORMAPS: [ColormapName; 5] = [
    ColormapName::Viridis,
    ColormapName::Cividis,
    ColormapName::CoolWarm,
    ColormapName::Turbo,
    ColormapName::Grayscale,
];

/// Errors returned by [`tetraglass-viewer`](crate) operations.
#[derive(Debug, Error)]
pub enum ViewerError {
    /// Filesystem I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Mesh or time-series reader failure (`tetraglass-io`).
    #[error("mesh I/O error: {0}")]
    MeshIo(#[from] IoError),

    /// Scene parsing, validation, or evaluation failure (`tetraglass-scene`).
    #[error("scene error: {0}")]
    Scene(#[from] SceneError),

    /// GPU renderer or offscreen target failure (`tetraglass-render`).
    #[error("render error: {0}")]
    Render(#[from] RenderError),

    /// Semantic validation failure on a scene file.
    #[error("scene validation failed:\n{0}")]
    Validation(String),

    /// Native window or `eframe` initialization error.
    #[error("viewer window error: {0}")]
    Window(String),
}

/// Outcome of polling or reloading a [`ViewerDocument`] from disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HotReloadOutcome {
    /// Neither the scene file nor any watched source file changed on disk.
    Unchanged,
    /// The scene (and/or its source files) was reloaded and re-evaluated.
    Reloaded {
        /// Whether any source mesh or filter geometry changed (requiring a GPU vertex re-upload).
        geometry_changed: bool,
        /// Whether the scene's `view.camera` specification changed on disk.
        camera_changed: bool,
    },
    /// A file on disk changed, but parsing, loading, or validation failed; the previous valid
    /// state remains active and the error message is recorded on [`ViewerDocument::last_reload_error`].
    ReloadFailed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileFingerprint {
    modified: Option<SystemTime>,
    len: u64,
    content_hash: Option<u64>,
}

impl FileFingerprint {
    fn capture(path: &Path, hash_contents: bool) -> Option<Self> {
        let meta = fs::metadata(path).ok()?;
        let modified = meta.modified().ok();
        let len = meta.len();
        let content_hash = if hash_contents && len <= 1024 * 1024 {
            fs::read(path).ok().map(|bytes| fnv1a_64(&bytes))
        } else {
            None
        };
        Some(Self {
            modified,
            len,
            content_hash,
        })
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// What changed when [`SceneFileWatcher::check_for_changes`] inspected watched files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchedFileChange {
    /// No watched files changed.
    None,
    /// Only the primary scene JSON file changed.
    SceneOnly,
    /// One or more source mesh files (or a raw mesh file opened directly) changed.
    SourcesChanged,
}

/// Polls a scene file and its referenced source files for on-disk modifications.
///
/// Uses both filesystem metadata (`modified()`, byte length) and a fast 64-bit FNV-1a content hash
/// on the scene file so sub-millisecond edits and atomic editor renames are detected reliably.
#[derive(Debug, Clone)]
pub struct SceneFileWatcher {
    primary_path: PathBuf,
    hash_primary: bool,
    primary_state: Option<FileFingerprint>,
    source_states: BTreeMap<PathBuf, Option<FileFingerprint>>,
}

impl SceneFileWatcher {
    /// Creates a new [`SceneFileWatcher`] for `primary_path` and `source_paths`.
    #[must_use]
    pub fn new(
        primary_path: impl Into<PathBuf>,
        hash_primary: bool,
        source_paths: impl IntoIterator<Item = PathBuf>,
    ) -> Self {
        let primary_path = primary_path.into();
        let primary_state = FileFingerprint::capture(&primary_path, hash_primary);
        let mut source_states = BTreeMap::new();
        for src in source_paths {
            let fp = FileFingerprint::capture(&src, false);
            source_states.insert(src, fp);
        }
        Self {
            primary_path,
            hash_primary,
            primary_state,
            source_states,
        }
    }

    /// Returns the watched primary path (scene JSON or mesh file).
    #[must_use]
    pub fn primary_path(&self) -> &Path {
        &self.primary_path
    }

    /// Replaces the set of watched source file paths.
    pub fn set_source_paths(&mut self, source_paths: impl IntoIterator<Item = PathBuf>) {
        self.source_states.clear();
        for src in source_paths {
            let fp = FileFingerprint::capture(&src, false);
            self.source_states.insert(src, fp);
        }
    }

    /// Checks whether the primary file or any watched source file changed since the last check,
    /// updating the stored fingerprints when a change is observed.
    pub fn check_for_changes(&mut self) -> WatchedFileChange {
        let mut sources_changed = false;
        for (path, prev_fp) in &mut self.source_states {
            let current = FileFingerprint::capture(path, false);
            if current != *prev_fp {
                *prev_fp = current;
                sources_changed = true;
            }
        }

        let current_primary = FileFingerprint::capture(&self.primary_path, self.hash_primary);
        let primary_changed = current_primary != self.primary_state;
        if primary_changed {
            self.primary_state = current_primary;
        }

        if sources_changed || (primary_changed && !self.hash_primary) {
            WatchedFileChange::SourcesChanged
        } else if primary_changed {
            WatchedFileChange::SceneOnly
        } else {
            WatchedFileChange::None
        }
    }
}

/// Headless-testable document model for the Tetraglass viewer (`M1.8`).
///
/// Holds the loaded [`Scene`], [`SceneEvaluator`], time-series state, field and colormap picker
/// state, and optional [`SceneFileWatcher`] for hot reload.
#[derive(Debug)]
pub struct ViewerDocument {
    file_label: String,
    scene_path: Option<PathBuf>,
    is_raw_mesh_path: bool,
    base_dir: PathBuf,
    scene: Scene,
    evaluator: SceneEvaluator,
    evaluated: EvaluatedScene,
    time_steps: Vec<f64>,
    current_step: usize,
    field_options: Vec<FieldOption>,
    selected_field_idx: usize,
    colormap: ColormapName,
    use_log_scale: bool,
    show_colorbar: bool,
    show_axes: bool,
    watcher: Option<SceneFileWatcher>,
    last_reload_error: Option<String>,
    reload_count: usize,
}

impl ViewerDocument {
    /// Loads a [`ViewerDocument`] from a `.json` scene file or a mesh file (`.vtu`, `.pvd`, `.msh`).
    pub fn load_path(path: impl AsRef<Path>) -> Result<Self, ViewerError> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        let is_json = is_scene_json_path_or_bytes(path, &bytes);
        let base_dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        let file_label = path.file_name().map_or_else(
            || path.display().to_string(),
            |s| s.to_string_lossy().into_owned(),
        );

        let (scene, mut evaluator, time_steps, source_paths) = if is_json {
            let scene = Scene::from_json_bytes(&bytes)?;
            let (evaluator, time_steps, source_paths) =
                load_and_validate_sources(&scene, &base_dir)?;
            (scene, evaluator, time_steps, source_paths)
        } else {
            let (scene, evaluator, time_steps) = build_default_scene_for_mesh(path, &bytes)?;
            (scene, evaluator, time_steps, vec![path.to_path_buf()])
        };

        let initial_selector = scene.output.as_ref().map_or(TimeSelector::LAST, |o| o.time);
        let evaluated = evaluator.evaluate_scene_at_time(&scene, initial_selector)?;
        let current_step = evaluated.time_step;

        let watcher = Some(SceneFileWatcher::new(path, is_json, source_paths));

        let mut doc = Self {
            file_label,
            scene_path: Some(path.to_path_buf()),
            is_raw_mesh_path: !is_json,
            base_dir,
            scene,
            evaluator,
            evaluated,
            time_steps,
            current_step,
            field_options: Vec::new(),
            selected_field_idx: 0,
            colormap: ColormapName::Viridis,
            use_log_scale: false,
            show_colorbar: false,
            show_axes: true,
            watcher,
            last_reload_error: None,
            reload_count: 0,
        };
        doc.sync_ui_controls_from_scene();
        Ok(doc)
    }

    /// Creates a [`ViewerDocument`] from an in-memory [`Scene`] and `base_dir` for resolving relative source paths.
    pub fn from_scene(scene: Scene, base_dir: impl Into<PathBuf>) -> Result<Self, ViewerError> {
        let base_dir = base_dir.into();
        let (mut evaluator, time_steps, _) = load_and_validate_sources(&scene, &base_dir)?;
        let initial_selector = scene.output.as_ref().map_or(TimeSelector::LAST, |o| o.time);
        let evaluated = evaluator.evaluate_scene_at_time(&scene, initial_selector)?;
        let current_step = evaluated.time_step;

        let mut doc = Self {
            file_label: "in-memory scene".to_string(),
            scene_path: None,
            is_raw_mesh_path: false,
            base_dir,
            scene,
            evaluator,
            evaluated,
            time_steps,
            current_step,
            field_options: Vec::new(),
            selected_field_idx: 0,
            colormap: ColormapName::Viridis,
            use_log_scale: false,
            show_colorbar: false,
            show_axes: true,
            watcher: None,
            last_reload_error: None,
            reload_count: 0,
        };
        doc.sync_ui_controls_from_scene();
        Ok(doc)
    }

    /// Returns the short display label for the opened file.
    #[must_use]
    pub fn file_label(&self) -> &str {
        &self.file_label
    }

    /// Returns the path of the opened scene or mesh file, if backed by a file on disk.
    #[must_use]
    pub fn scene_path(&self) -> Option<&Path> {
        self.scene_path.as_deref()
    }

    /// Returns a reference to the current [`Scene`].
    #[must_use]
    pub const fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Returns a reference to the underlying [`SceneEvaluator`].
    #[must_use]
    pub const fn evaluator(&self) -> &SceneEvaluator {
        &self.evaluator
    }

    /// Returns a reference to the latest [`EvaluatedScene`].
    #[must_use]
    pub const fn evaluated(&self) -> &EvaluatedScene {
        &self.evaluated
    }

    /// Returns the primary [`SurfaceMesh`] from the first visible representation, if any.
    #[must_use]
    pub fn active_surface(&self) -> Option<&SurfaceMesh> {
        self.evaluated
            .representations
            .first()
            .map(|r| r.surface.as_ref())
    }

    /// Returns the [`ViewportSurfaceStyle`] for the first visible representation.
    #[must_use]
    pub fn active_surface_style(&self) -> ViewportSurfaceStyle {
        self.evaluated
            .representations
            .first()
            .map_or_else(ViewportSurfaceStyle::default, build_surface_style)
    }

    /// Computes the initial or updated [`ViewportCamera`] from the scene's `view.camera` and dataset bounds.
    #[must_use]
    pub fn camera_from_scene(&self) -> ViewportCamera {
        let bounds = self.evaluated.camera_bounds.unwrap_or(BoundingBox {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        });
        build_camera_from_scene(&self.scene, bounds)
    }

    /// Builds the renderer-drawn [`Annotations`] (title, axes triad, and colorbars) for the current state.
    #[must_use]
    pub fn annotations(&self) -> Annotations {
        build_annotations_from_scene(&self.scene, &self.evaluated)
    }

    /// Resolves the viewport background color in sRGB `[r, g, b, a]`.
    #[must_use]
    pub fn background_srgb(&self) -> [f32; 4] {
        resolve_background_srgb(&self.scene.view.background)
    }

    /// Returns the MSAA sample count configured on the scene (`4` by default).
    #[must_use]
    pub fn msaa_samples(&self) -> u32 {
        self.scene
            .output
            .as_ref()
            .map_or(DEFAULT_MSAA_SAMPLES, |o| o.msaa)
    }

    /// Returns the available simulation time values across the time series.
    #[must_use]
    pub fn time_steps(&self) -> &[f64] {
        &self.time_steps
    }

    /// Returns the number of time steps in the active series (`>= 1`).
    #[must_use]
    pub fn num_time_steps(&self) -> usize {
        self.time_steps.len().max(1)
    }

    /// Returns the currently selected zero-based time-step index.
    #[must_use]
    pub const fn current_time_step(&self) -> usize {
        self.current_step
    }

    /// Returns the simulation time value at the currently selected step.
    #[must_use]
    pub const fn current_time_value(&self) -> f64 {
        self.evaluated.time_value
    }

    /// Selects a zero-based time-step index, lazily evaluating the pipeline at that step.
    ///
    /// Returns `Ok(true)` if the step changed and the scene was re-evaluated.
    pub fn set_time_step(&mut self, step: usize) -> Result<bool, ViewerError> {
        let max_step = self.num_time_steps().saturating_sub(1);
        let clamped = step.min(max_step);
        if clamped == self.current_step && self.evaluated.time_step == clamped {
            return Ok(false);
        }

        let selector = TimeSelector::step(clamped);
        if let Some(out) = &mut self.scene.output {
            out.time = selector;
        } else {
            let mut out = Output::new("out.png");
            out.time = selector;
            self.scene.output = Some(out);
        }

        self.evaluated = self
            .evaluator
            .evaluate_scene_at_time(&self.scene, selector)?;
        self.current_step = self.evaluated.time_step;
        self.refresh_field_options_preserving_selection();
        Ok(true)
    }

    /// Advances to the next time step (wrapping around to `0` when `wrap` is `true`).
    pub fn step_forward(&mut self, wrap: bool) -> Result<bool, ViewerError> {
        let count = self.num_time_steps();
        if count <= 1 {
            return Ok(false);
        }
        let next = if self.current_step + 1 < count {
            self.current_step + 1
        } else if wrap {
            0
        } else {
            self.current_step
        };
        self.set_time_step(next)
    }

    /// Steps backward to the previous time step (wrapping to the last step when `wrap` is `true`).
    pub fn step_backward(&mut self, wrap: bool) -> Result<bool, ViewerError> {
        let count = self.num_time_steps();
        if count <= 1 {
            return Ok(false);
        }
        let prev = if self.current_step > 0 {
            self.current_step - 1
        } else if wrap {
            count - 1
        } else {
            0
        };
        self.set_time_step(prev)
    }

    /// Returns the selectable [`FieldOption`]s for the active representation's surface.
    #[must_use]
    pub fn field_options(&self) -> &[FieldOption] {
        &self.field_options
    }

    /// Returns the index into [`Self::field_options`] of the currently selected field.
    #[must_use]
    pub const fn selected_field_index(&self) -> usize {
        self.selected_field_idx
    }

    /// Returns the currently selected [`FieldOption`], if any.
    #[must_use]
    pub fn selected_field(&self) -> Option<&FieldOption> {
        self.field_options.get(self.selected_field_idx)
    }

    /// Selects a field by its index in [`Self::field_options`] and re-evaluates coloring without re-running filters.
    pub fn set_field_index(&mut self, index: usize) -> Result<bool, ViewerError> {
        let Some(opt) = self.field_options.get(index) else {
            return Ok(false);
        };
        if index == self.selected_field_idx {
            return Ok(false);
        }
        let color_by_str = color_by_to_scene_string(&opt.color_by);
        self.selected_field_idx = index;
        if let Some(rep) = self.scene.view.show.first_mut() {
            rep.color_by = color_by_str;
        }
        self.evaluated = self
            .evaluator
            .evaluate_scene_at_time(&self.scene, TimeSelector::step(self.current_step))?;
        Ok(true)
    }

    /// Sets the `color_by` array reference string on the primary representation directly.
    pub fn set_color_by(&mut self, color_by: Option<String>) -> Result<bool, ViewerError> {
        if let Some(rep) = self.scene.view.show.first_mut() {
            if rep.color_by == color_by {
                return Ok(false);
            }
            rep.color_by = color_by;
        }
        self.evaluated = self
            .evaluator
            .evaluate_scene_at_time(&self.scene, TimeSelector::step(self.current_step))?;
        self.sync_ui_controls_from_scene();
        Ok(true)
    }

    /// Returns the active [`ColormapName`].
    #[must_use]
    pub const fn colormap(&self) -> ColormapName {
        self.colormap
    }

    /// Updates the active [`ColormapName`] and re-evaluates coloring without re-running filters.
    pub fn set_colormap(&mut self, colormap: ColormapName) -> Result<bool, ViewerError> {
        if self.colormap == colormap {
            return Ok(false);
        }
        self.colormap = colormap;
        if let Some(rep) = self.scene.view.show.first_mut() {
            rep.colormap = colormap;
        }
        self.evaluated = self
            .evaluator
            .evaluate_scene_at_time(&self.scene, TimeSelector::step(self.current_step))?;
        Ok(true)
    }

    /// Returns whether logarithmic scalar mapping is enabled.
    #[must_use]
    pub const fn use_log_scale(&self) -> bool {
        self.use_log_scale
    }

    /// Toggles logarithmic scalar mapping on the primary representation.
    pub fn set_log_scale(&mut self, log_scale: bool) -> Result<bool, ViewerError> {
        if self.use_log_scale == log_scale {
            return Ok(false);
        }
        self.use_log_scale = log_scale;
        if let Some(rep) = self.scene.view.show.first_mut() {
            rep.log_scale = log_scale;
        }
        self.evaluated = self
            .evaluator
            .evaluate_scene_at_time(&self.scene, TimeSelector::step(self.current_step))?;
        Ok(true)
    }

    /// Returns whether the colorbar annotation is enabled on the primary representation.
    #[must_use]
    pub const fn colorbar_enabled(&self) -> bool {
        self.show_colorbar
    }

    /// Enables or disables the colorbar annotation on the primary representation.
    pub fn set_colorbar_enabled(&mut self, enabled: bool) -> Result<bool, ViewerError> {
        if self.show_colorbar == enabled {
            return Ok(false);
        }
        self.show_colorbar = enabled;
        if let Some(rep) = self.scene.view.show.first_mut() {
            rep.colorbar = ColorbarSpec::Enabled(enabled);
        }
        self.evaluated = self
            .evaluator
            .evaluate_scene_at_time(&self.scene, TimeSelector::step(self.current_step))?;
        Ok(true)
    }

    /// Returns whether the 3D orientation axes triad is enabled.
    #[must_use]
    pub const fn axes_enabled(&self) -> bool {
        self.show_axes
    }

    /// Enables or disables the 3D orientation axes triad.
    pub fn set_axes_enabled(&mut self, enabled: bool) -> bool {
        if self.show_axes == enabled {
            return false;
        }
        self.show_axes = enabled;
        self.scene.view.axes = enabled;
        true
    }

    /// Returns the most recent hot-reload error message, if the last reload attempt failed.
    #[must_use]
    pub fn last_reload_error(&self) -> Option<&str> {
        self.last_reload_error.as_deref()
    }

    /// Returns the number of successful hot reloads performed on this document.
    #[must_use]
    pub const fn reload_count(&self) -> usize {
        self.reload_count
    }

    /// Polls the [`SceneFileWatcher`] and reloads the scene (or mesh) if any watched file changed on disk.
    pub fn poll_hot_reload(&mut self) -> Result<HotReloadOutcome, ViewerError> {
        let Some(watcher) = &mut self.watcher else {
            return Ok(HotReloadOutcome::Unchanged);
        };
        let change = watcher.check_for_changes();
        if change == WatchedFileChange::None {
            return Ok(HotReloadOutcome::Unchanged);
        }
        self.reload_with_change(change)
    }

    /// Forces a reload from the watched file path on disk.
    pub fn reload_from_disk(&mut self) -> Result<HotReloadOutcome, ViewerError> {
        self.reload_with_change(WatchedFileChange::SceneOnly)
    }

    fn reload_with_change(
        &mut self,
        change: WatchedFileChange,
    ) -> Result<HotReloadOutcome, ViewerError> {
        let Some(path) = self.scene_path.clone() else {
            return Ok(HotReloadOutcome::Unchanged);
        };

        let result = if self.is_raw_mesh_path {
            self.try_reload_raw_mesh(&path)
        } else {
            self.try_reload_scene_json(&path, change == WatchedFileChange::SourcesChanged)
        };

        match result {
            Ok((geometry_changed, camera_changed)) => {
                self.last_reload_error = None;
                self.reload_count += 1;
                Ok(HotReloadOutcome::Reloaded {
                    geometry_changed,
                    camera_changed,
                })
            }
            Err(err) => {
                let msg = err.to_string();
                self.last_reload_error = Some(msg.clone());
                Ok(HotReloadOutcome::ReloadFailed(msg))
            }
        }
    }

    fn try_reload_raw_mesh(&mut self, path: &Path) -> Result<(bool, bool), ViewerError> {
        let bytes = fs::read(path)?;
        let (new_scene, mut new_evaluator, new_times) = build_default_scene_for_mesh(path, &bytes)?;
        let clamped_step = self.current_step.min(new_times.len().saturating_sub(1));
        let evaluated =
            new_evaluator.evaluate_scene_at_time(&new_scene, TimeSelector::step(clamped_step))?;

        self.scene = new_scene;
        self.evaluator = new_evaluator;
        self.evaluated = evaluated;
        self.time_steps = new_times;
        self.current_step = self.evaluated.time_step;
        self.sync_ui_controls_from_scene();
        Ok((true, false))
    }

    fn try_reload_scene_json(
        &mut self,
        path: &Path,
        force_reload_sources: bool,
    ) -> Result<(bool, bool), ViewerError> {
        let bytes = fs::read(path)?;
        let new_scene = Scene::from_json_bytes(&bytes)?;

        let sources_changed = force_reload_sources || (new_scene.sources != self.scene.sources);
        let filters_changed = new_scene.filters != self.scene.filters;
        let old_rep_inputs: Vec<(&str, _)> = self
            .scene
            .view
            .show
            .iter()
            .filter(|r| r.visible)
            .map(|r| (r.input.as_str(), r.style))
            .collect();
        let new_rep_inputs: Vec<(&str, _)> = new_scene
            .view
            .show
            .iter()
            .filter(|r| r.visible)
            .map(|r| (r.input.as_str(), r.style))
            .collect();
        let camera_changed = new_scene.view.camera != self.scene.view.camera;
        let old_time_spec = self.scene.output.as_ref().map(|o| o.time);
        let new_time_spec = new_scene.output.as_ref().map(|o| o.time);
        let time_spec_changed = new_time_spec != old_time_spec;

        if sources_changed {
            let (mut new_evaluator, new_times, source_paths) =
                load_and_validate_sources(&new_scene, &self.base_dir)?;
            let selector = if time_spec_changed {
                new_time_spec.unwrap_or(TimeSelector::LAST)
            } else {
                TimeSelector::step(self.current_step.min(new_times.len().saturating_sub(1)))
            };
            let evaluated = new_evaluator.evaluate_scene_at_time(&new_scene, selector)?;
            self.scene = new_scene;
            self.evaluator = new_evaluator;
            self.evaluated = evaluated;
            self.time_steps = new_times;
            self.current_step = self.evaluated.time_step;
            if let Some(watcher) = &mut self.watcher {
                watcher.set_source_paths(source_paths);
            }
            self.sync_ui_controls_from_scene();
            return Ok((true, camera_changed));
        }

        // Validate `new_scene` against the already-loaded datasets in `self.evaluator`.
        validate_scene_against_evaluator(&new_scene, &self.evaluator)?;

        let selector = if time_spec_changed {
            new_time_spec.unwrap_or(TimeSelector::LAST)
        } else {
            TimeSelector::step(self.current_step)
        };
        let prev_step = self.current_step;
        let evaluated = self
            .evaluator
            .evaluate_scene_at_time(&new_scene, selector)?;
        let step_changed = evaluated.time_step != prev_step;
        let geometry_changed =
            filters_changed || (old_rep_inputs != new_rep_inputs) || step_changed;

        self.scene = new_scene;
        self.evaluated = evaluated;
        self.current_step = self.evaluated.time_step;
        self.sync_ui_controls_from_scene();
        Ok((geometry_changed, camera_changed))
    }

    fn sync_ui_controls_from_scene(&mut self) {
        if let Some(surface) = self.active_surface() {
            self.field_options = available_field_options(surface);
        } else {
            self.field_options = vec![FieldOption {
                label: "Solid color".to_string(),
                color_by: ViewportColorBy::Solid,
            }];
        }

        let active_style = self.active_surface_style();
        self.selected_field_idx = self
            .field_options
            .iter()
            .position(|opt| opt.color_by == active_style.color_by)
            .unwrap_or(0);

        if let Some(rep) = self.scene.view.show.first() {
            self.colormap = rep.colormap;
            self.use_log_scale = rep.log_scale;
            self.show_colorbar = rep.colorbar.is_enabled();
        }
        self.show_axes = self.scene.view.axes;
    }

    fn refresh_field_options_preserving_selection(&mut self) {
        let prev_color_by = self
            .field_options
            .get(self.selected_field_idx)
            .map(|o| o.color_by.clone());
        if let Some(surface) = self.active_surface() {
            self.field_options = available_field_options(surface);
        }
        if let Some(target) = prev_color_by {
            self.selected_field_idx = self
                .field_options
                .iter()
                .position(|opt| opt.color_by == target)
                .unwrap_or(0);
        }
    }

    /// Resolves the upstream volume dataset for interactive clipping (`M2.8`), returning its
    /// unclipped external [`SurfaceMesh`], linearized [`TetView`], and initial [`InteractiveClipState`].
    pub fn resolve_clip_volume(&mut self) -> Result<Option<ResolvedClipVolume>, ViewerError> {
        let Some(first_show) = self.scene.view.show.first().cloned() else {
            return Ok(None);
        };
        let (vol_name, preset_clip) = resolve_volume_and_clip_spec(&self.scene, &first_show.input);
        let Some(vol_name) = vol_name else {
            return Ok(None);
        };
        let (_, hull_surface) = self.evaluator.evaluate_surface(&self.scene, &vol_name)?;
        let (_, tet_view) = self.evaluator.evaluate_tet_view(&self.scene, &vol_name)?;
        let bounds = hull_surface
            .bounds()
            .or_else(|| tet_view.bounds())
            .unwrap_or(BoundingBox {
                min: [-1.0, -1.0, -1.0],
                max: [1.0, 1.0, 1.0],
            });
        let center = bounds.center();
        let max_offset = (0.55 * bounds.diagonal()).max(1e-3);
        let state = match preset_clip {
            Some((origin, normal, keep, mode)) => {
                let mut st = InteractiveClipState::new(center, max_offset);
                st.enabled = true;
                st.keep = keep;
                st.mode = mode;
                st.set_normal(normal);
                if let Some(orig) = origin {
                    st.set_origin(orig);
                }
                st
            }
            None => InteractiveClipState::new(center, max_offset),
        };
        Ok(Some((hull_surface, tet_view, state)))
    }
}

/// Tuple `(hull_surface, tet_view, initial_clip_state)` returned by [`ViewerDocument::resolve_clip_volume`].
pub type ResolvedClipVolume = (Arc<SurfaceMesh>, Arc<TetView>, InteractiveClipState);

type PresetClipSpec = (Option<[f64; 3]>, [f64; 3], ClipKeep, ClipMode);

fn resolve_volume_and_clip_spec(
    scene: &Scene,
    rep_input: &str,
) -> (Option<String>, Option<PresetClipSpec>) {
    if scene.sources.contains_key(rep_input) {
        return (Some(rep_input.to_string()), None);
    }
    match scene.filters.get(rep_input) {
        Some(Filter::Clip {
            input,
            origin,
            normal,
            keep,
            mode,
        }) => (Some(input.clone()), Some((*origin, *normal, *keep, *mode))),
        Some(Filter::Threshold { .. }) => (Some(rep_input.to_string()), None),
        Some(Filter::Surface { input }) => {
            if let Some(Filter::Clip {
                input: clip_in,
                origin,
                normal,
                keep,
                mode,
            }) = scene.filters.get(input)
            {
                (
                    Some(clip_in.clone()),
                    Some((*origin, *normal, *keep, *mode)),
                )
            } else if scene.sources.contains_key(input)
                || matches!(scene.filters.get(input), Some(Filter::Threshold { .. }))
            {
                (Some(input.clone()), None)
            } else {
                (None, None)
            }
        }
        _ => (None, None),
    }
}

/// Configuration and live status of interactive plane clipping in [`ViewerSession`] (`M2.8`).
#[derive(Debug, Clone, PartialEq)]
pub struct InteractiveClipState {
    /// Whether interactive clipping is currently enabled.
    pub enabled: bool,
    /// Point `[x, y, z]` (`f64`) on the clip plane in world coordinates.
    pub origin: [f64; 3],
    /// Outward unit plane normal `[nx, ny, nz]` (`f64`).
    pub normal: [f64; 3],
    /// Signed offset along `normal` from [`Self::center`].
    pub offset: f64,
    /// Half-diagonal range `[-max_offset, +max_offset]` for the offset slider.
    pub max_offset: f64,
    /// Dataset bounding-box center `[cx, cy, cz]` (`f64`).
    pub center: [f64; 3],
    /// Which half-space of the clip plane to keep (`Back` or `Front`).
    pub keep: ClipKeep,
    /// Whether the background worker computes `Exact` or `Crinkle` clipping on release.
    pub mode: ClipMode,
    /// Whether the clip plane is currently being dragged (using the GPU shader clip plane).
    pub is_dragging: bool,
    /// Whether a background worker thread is currently computing the exact clipped surface.
    pub worker_busy: bool,
}

impl InteractiveClipState {
    /// Creates a default [`InteractiveClipState`] centered at `center` with normal `+X`.
    #[must_use]
    pub const fn new(center: [f64; 3], max_offset: f64) -> Self {
        Self {
            enabled: false,
            origin: center,
            normal: [1.0, 0.0, 0.0],
            offset: 0.0,
            max_offset,
            center,
            keep: ClipKeep::Back,
            mode: ClipMode::Exact,
            is_dragging: false,
            worker_busy: false,
        }
    }

    /// Converts the current plane into a GPU [`ViewportClipPlane`].
    #[must_use]
    pub const fn shader_clip_plane(&self) -> ViewportClipPlane {
        let keep = match self.keep {
            ClipKeep::Back => ViewportClipKeep::Back,
            ClipKeep::Front => ViewportClipKeep::Front,
        };
        ViewportClipPlane::with_keep(self.origin, self.normal, keep)
    }

    /// Sets the plane normal, normalizing it and updating [`Self::origin`] from [`Self::center`] + [`Self::offset`].
    pub fn set_normal(&mut self, normal: [f64; 3]) {
        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len.is_finite() && len > 1e-12 {
            self.normal = [normal[0] / len, normal[1] / len, normal[2] / len];
        }
        self.origin = [
            self.center[0] + self.offset * self.normal[0],
            self.center[1] + self.offset * self.normal[1],
            self.center[2] + self.offset * self.normal[2],
        ];
    }

    /// Sets the signed offset along [`Self::normal`] relative to [`Self::center`] and updates [`Self::origin`].
    pub fn set_offset(&mut self, offset: f64) {
        self.offset = offset.clamp(-self.max_offset, self.max_offset);
        self.origin = [
            self.center[0] + self.offset * self.normal[0],
            self.center[1] + self.offset * self.normal[1],
            self.center[2] + self.offset * self.normal[2],
        ];
    }

    /// Sets [`Self::origin`] directly and updates [`Self::offset`] as the projection along [`Self::normal`].
    pub fn set_origin(&mut self, origin: [f64; 3]) {
        self.origin = origin;
        self.offset = (origin[0] - self.center[0]) * self.normal[0]
            + (origin[1] - self.center[1]) * self.normal[1]
            + (origin[2] - self.center[2]) * self.normal[2];
    }
}

struct CompletedClipJob {
    generation: u64,
    surface: Result<Arc<SurfaceMesh>, String>,
}

fn execute_clip_surface_job(
    tet_view: &TetView,
    origin: [f64; 3],
    normal: [f64; 3],
    keep: ClipKeep,
    mode: ClipMode,
) -> Result<Arc<SurfaceMesh>, String> {
    let params = ClipParams::with_origin(origin, normal)
        .keep(keep.into())
        .mode(mode.into());
    let clipped = clip(tet_view, &params).map_err(|e| e.to_string())?;
    let surface = extract_tet_view_surface(&clipped).map_err(|e| e.to_string())?;
    Ok(Arc::new(surface))
}

/// Interactive session pairing a [`ViewerDocument`] with a GPU-backed [`ViewportState`].
pub struct ViewerSession {
    doc: ViewerDocument,
    viewport: ViewportState,
    is_playing: bool,
    last_reload_poll: Instant,
    last_playback_tick: Instant,
    clip_state: InteractiveClipState,
    clip_base_hull: Option<Arc<SurfaceMesh>>,
    clip_base_tet_view: Option<Arc<TetView>>,
    clip_showing_hull: bool,
    clip_generation: u64,
    clip_tx: mpsc::Sender<CompletedClipJob>,
    clip_rx: mpsc::Receiver<CompletedClipJob>,
    active_exact_clip_surface: Option<Arc<SurfaceMesh>>,
}

impl std::fmt::Debug for ViewerSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViewerSession")
            .field("doc", &self.doc)
            .field("is_playing", &self.is_playing)
            .field("clip_state", &self.clip_state)
            .finish_non_exhaustive()
    }
}

impl ViewerSession {
    /// Creates a new [`ViewerSession`] and uploads the initial surface(s), camera, background, and annotations.
    pub fn new(
        mut doc: ViewerDocument,
        mut viewport: ViewportState,
        device: &egui_wgpu::wgpu::Device,
    ) -> Result<Self, ViewerError> {
        let camera = doc.camera_from_scene();
        let bg = doc.background_srgb();
        let annotations = doc.annotations();
        if let Some(first_rep) = doc.evaluated().representations.first() {
            viewport.set_shared_surface(
                device,
                Arc::clone(&first_rep.surface),
                build_surface_style(first_rep),
                true,
            )?;
            let extra: Vec<(Arc<SurfaceMesh>, ViewportSurfaceStyle)> = doc
                .evaluated()
                .representations
                .iter()
                .skip(1)
                .map(|r| (Arc::clone(&r.surface), build_surface_style(r)))
                .collect();
            viewport.set_additional_surfaces(device, extra)?;
        }
        viewport.set_camera(camera);
        viewport.set_background_srgb(bg);
        viewport.set_annotations(annotations);

        let (clip_base_hull, clip_base_tet_view, clip_state) = match doc.resolve_clip_volume()? {
            Some((hull, tet_view, state)) => (Some(hull), Some(tet_view), state),
            None => (None, None, InteractiveClipState::new([0.0, 0.0, 0.0], 1.0)),
        };
        let clip_showing_hull = !clip_state.enabled;
        let (clip_tx, clip_rx) = mpsc::channel();

        let now = Instant::now();
        Ok(Self {
            doc,
            viewport,
            is_playing: false,
            last_reload_poll: now,
            last_playback_tick: now,
            clip_state,
            clip_base_hull,
            clip_base_tet_view,
            clip_showing_hull,
            clip_generation: 0,
            clip_tx,
            clip_rx,
            active_exact_clip_surface: None,
        })
    }

    /// Creates a new [`ViewerSession`] from an [`egui_wgpu::RenderState`].
    pub fn from_render_state(
        doc: ViewerDocument,
        render_state: &egui_wgpu::RenderState,
    ) -> Result<Self, ViewerError> {
        let msaa = doc.msaa_samples();
        let viewport = ViewportState::from_render_state(render_state, msaa)?;
        Self::new(doc, viewport, &render_state.device)
    }

    /// Returns a shared reference to the [`ViewerDocument`].
    #[must_use]
    pub const fn document(&self) -> &ViewerDocument {
        &self.doc
    }

    /// Returns a mutable reference to the [`ViewerDocument`].
    pub const fn document_mut(&mut self) -> &mut ViewerDocument {
        &mut self.doc
    }

    /// Returns a shared reference to the [`ViewportState`].
    #[must_use]
    pub const fn viewport(&self) -> &ViewportState {
        &self.viewport
    }

    /// Returns a mutable reference to the [`ViewportState`].
    pub const fn viewport_mut(&mut self) -> &mut ViewportState {
        &mut self.viewport
    }

    /// Synchronously probes the surface at physical viewport pixel `(x_px, y_px)` using the GPU ID buffer.
    pub fn pick_at_pixel(
        &mut self,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
        x_px: u32,
        y_px: u32,
    ) -> Result<Option<SurfacePickResult>, ViewerError> {
        Ok(self.viewport.pick_at(device, queue, x_px, y_px)?)
    }

    /// Returns the most recent value-probe result (`Some` after clicking a surface or calling [`Self::pick_at_pixel`]).
    #[must_use]
    pub fn last_pick(&self) -> Option<SurfacePickResult> {
        self.viewport.last_pick()
    }

    /// Clears the active value-probe result.
    pub fn clear_pick(&mut self) {
        self.viewport.clear_pick();
    }

    /// Returns the `input` name of the representation hit by `last_pick()`, if any.
    #[must_use]
    pub fn picked_representation_name(&self) -> Option<&str> {
        let pick = self.viewport.last_pick()?;
        self.doc
            .evaluated()
            .representations
            .get(pick.representation_index)
            .map(|r| r.input.as_str())
    }

    /// Returns the current [`InteractiveClipState`] (`M2.8`).
    #[must_use]
    pub const fn clip_state(&self) -> &InteractiveClipState {
        &self.clip_state
    }

    /// Returns the most recently swapped-in exact clipped [`SurfaceMesh`] from the background worker, if any.
    #[must_use]
    pub const fn active_clipped_surface(&self) -> Option<&Arc<SurfaceMesh>> {
        self.active_exact_clip_surface.as_ref()
    }

    /// Sets the [`ClipMode`] (`Exact` or `Crinkle`) used when the clip drag is released.
    pub const fn set_clip_mode(&mut self, mode: ClipMode) {
        self.clip_state.mode = mode;
    }

    /// Starts or updates an interactive clip-plane drag (`M2.8`, `D-019`).
    ///
    /// Ensures the unclipped external hull is displayed in the viewport and updates only the
    /// 16-byte GPU shader clip-plane uniform (`viewport.set_clip_plane`) for instant 60 FPS feedback.
    pub fn begin_or_update_clip_drag(
        &mut self,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
        origin: [f64; 3],
        normal: [f64; 3],
        keep: ClipKeep,
    ) -> Result<(), ViewerError> {
        if self.clip_base_hull.is_none() || self.clip_base_tet_view.is_none() {
            if let Some((hull, tet_view, init_state)) = self.doc.resolve_clip_volume()? {
                self.clip_base_hull = Some(hull);
                self.clip_base_tet_view = Some(tet_view);
                self.clip_state.center = init_state.center;
                self.clip_state.max_offset = init_state.max_offset;
            } else {
                return Ok(());
            }
        }

        self.clip_generation = self.clip_generation.wrapping_add(1);
        self.clip_state.enabled = true;
        self.clip_state.is_dragging = true;
        self.clip_state.worker_busy = false;
        self.clip_state.keep = keep;
        self.clip_state.set_normal(normal);
        self.clip_state.set_origin(origin);

        let shader_plane = self.clip_state.shader_clip_plane();
        if !self.clip_showing_hull
            && let Some(ref hull) = self.clip_base_hull
        {
            let mut style = self.doc.active_surface_style();
            style.clip_plane = Some(shader_plane);
            self.viewport
                .set_shared_surface(device, Arc::clone(hull), style, false)?;
            self.clip_showing_hull = true;
        } else {
            self.viewport.set_clip_plane(queue, Some(shader_plane));
        }

        Ok(())
    }

    /// Finishes an interactive clip-plane drag and spawns a background worker thread to compute
    /// the exact (or crinkle) clipped [`SurfaceMesh`] (`M2.8`).
    ///
    /// While the worker thread is running, the viewport keeps displaying the shader-clipped preview
    /// until [`Self::poll_clip_worker`] or [`Self::wait_for_clip_worker`] swaps in the result.
    pub fn end_clip_drag(&mut self) {
        if !self.clip_state.enabled {
            return;
        }
        self.clip_state.is_dragging = false;
        self.spawn_clip_worker();
    }

    fn spawn_clip_worker(&mut self) {
        let Some(ref tet_view) = self.clip_base_tet_view else {
            return;
        };
        self.clip_generation = self.clip_generation.wrapping_add(1);
        let generation = self.clip_generation;
        self.clip_state.worker_busy = true;

        let tet_view = Arc::clone(tet_view);
        let origin = self.clip_state.origin;
        let normal = self.clip_state.normal;
        let keep = self.clip_state.keep;
        let mode = self.clip_state.mode;
        let tx = self.clip_tx.clone();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let tv_worker = Arc::clone(&tet_view);
            let spawn_res = std::thread::Builder::new()
                .name("tetraglass-clip-worker".to_string())
                .spawn(move || {
                    let surface = execute_clip_surface_job(&tv_worker, origin, normal, keep, mode);
                    let _ = tx.send(CompletedClipJob {
                        generation,
                        surface,
                    });
                });
            if spawn_res.is_err() {
                let surface = execute_clip_surface_job(&tet_view, origin, normal, keep, mode);
                let _ = self.clip_tx.send(CompletedClipJob {
                    generation,
                    surface,
                });
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            let surface = execute_clip_surface_job(&tet_view, origin, normal, keep, mode);
            let _ = tx.send(CompletedClipJob {
                generation,
                surface,
            });
        }
    }

    /// Polls the background clip worker channel without blocking, swapping in the completed
    /// exact clipped [`SurfaceMesh`] and clearing the GPU shader clip plane when ready (`M2.8`).
    ///
    /// Returns `Ok(true)` if a completed clipped surface was swapped into the viewport.
    pub fn poll_clip_worker(
        &mut self,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
    ) -> Result<bool, ViewerError> {
        let mut latest_match: Option<Result<Arc<SurfaceMesh>, String>> = None;
        while let Ok(job) = self.clip_rx.try_recv() {
            if job.generation == self.clip_generation {
                latest_match = Some(job.surface);
            }
        }
        let Some(res) = latest_match else {
            return Ok(false);
        };
        self.apply_completed_clip_job(device, queue, res)
    }

    /// Blocks until the active background clip worker finishes and swaps the exact clipped
    /// [`SurfaceMesh`] into the viewport (`M2.8`).
    ///
    /// Returns `Ok(true)` if a worker job was swapped in, or `Ok(false)` if no worker was active.
    pub fn wait_for_clip_worker(
        &mut self,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
    ) -> Result<bool, ViewerError> {
        if self.poll_clip_worker(device, queue)? {
            return Ok(true);
        }
        if !self.clip_state.worker_busy {
            return Ok(false);
        }
        while let Ok(job) = self.clip_rx.recv() {
            if job.generation == self.clip_generation {
                return self.apply_completed_clip_job(device, queue, job.surface);
            }
        }
        self.clip_state.worker_busy = false;
        Ok(false)
    }

    fn apply_completed_clip_job(
        &mut self,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
        res: Result<Arc<SurfaceMesh>, String>,
    ) -> Result<bool, ViewerError> {
        self.clip_state.worker_busy = false;
        match res {
            Ok(surface) => {
                let mut style = self.doc.active_surface_style();
                style.clip_plane = None;
                self.viewport
                    .set_shared_surface(device, Arc::clone(&surface), style, false)?;
                self.viewport.set_clip_plane(queue, None);
                self.clip_showing_hull = false;
                self.active_exact_clip_surface = Some(surface);
                Ok(true)
            }
            Err(err) => Err(ViewerError::Validation(format!(
                "interactive clip worker failed: {err}"
            ))),
        }
    }

    /// Disables interactive clipping and restores the unclipped surface in the viewport (`M2.8`).
    pub fn disable_interactive_clip(
        &mut self,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
    ) -> Result<(), ViewerError> {
        self.clip_generation = self.clip_generation.wrapping_add(1);
        self.clip_state.enabled = false;
        self.clip_state.is_dragging = false;
        self.clip_state.worker_busy = false;
        self.active_exact_clip_surface = None;

        let mut style = self.doc.active_surface_style();
        style.clip_plane = None;
        if let Some(ref hull) = self.clip_base_hull {
            self.viewport
                .set_shared_surface(device, Arc::clone(hull), style, false)?;
            self.clip_showing_hull = true;
        } else if let Some(first_rep) = self.doc.evaluated().representations.first() {
            self.viewport.set_shared_surface(
                device,
                Arc::clone(&first_rep.surface),
                style,
                false,
            )?;
            self.clip_showing_hull = true;
        }
        self.viewport.set_clip_plane(queue, None);
        Ok(())
    }

    /// Polls for file changes on disk and synchronizes the GPU [`ViewportState`] if anything changed.
    pub fn poll_hot_reload(
        &mut self,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
    ) -> Result<HotReloadOutcome, ViewerError> {
        let outcome = self.doc.poll_hot_reload()?;
        if let HotReloadOutcome::Reloaded {
            geometry_changed,
            camera_changed,
        } = &outcome
        {
            self.sync_viewport_after_doc_change(device, queue, *geometry_changed, *camera_changed)?;
        }
        Ok(outcome)
    }

    fn sync_viewport_after_doc_change(
        &mut self,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
        geometry_changed: bool,
        camera_changed: bool,
    ) -> Result<(), ViewerError> {
        if let Some(first_rep) = self.doc.evaluated().representations.first() {
            let mut style = build_surface_style(first_rep);
            style.clip_plane = self.viewport.clip_plane();
            if geometry_changed {
                let first_surface = Arc::clone(&first_rep.surface);
                let extra: Vec<(Arc<SurfaceMesh>, ViewportSurfaceStyle)> = self
                    .doc
                    .evaluated()
                    .representations
                    .iter()
                    .skip(1)
                    .map(|r| (Arc::clone(&r.surface), build_surface_style(r)))
                    .collect();
                if let Some((hull, tet_view, init_state)) = self.doc.resolve_clip_volume()? {
                    self.clip_base_hull = Some(hull);
                    self.clip_base_tet_view = Some(tet_view);
                    self.clip_state.center = init_state.center;
                    self.clip_state.max_offset = init_state.max_offset;
                }
                self.active_exact_clip_surface = None;
                style.clip_plane = None;
                self.viewport
                    .set_shared_surface(device, first_surface, style, false)?;
                self.viewport.set_additional_surfaces(device, extra)?;
                self.clip_showing_hull = !self.clip_state.enabled;
                if self.clip_state.enabled {
                    self.spawn_clip_worker();
                }
            } else if let Some(ref exact_surf) = self.active_exact_clip_surface {
                self.viewport.update_style(queue, exact_surf, style)?;
            } else if self.clip_showing_hull
                && let Some(ref hull) = self.clip_base_hull
            {
                self.viewport.update_style(queue, hull, style)?;
            } else {
                self.viewport
                    .update_style(queue, &first_rep.surface, style)?;
            }
        }
        if camera_changed {
            self.viewport.set_camera(self.doc.camera_from_scene());
        }
        self.viewport
            .set_background_srgb(self.doc.background_srgb());
        self.viewport.set_annotations(self.doc.annotations());
        Ok(())
    }

    /// Renders the viewer UI (left control panel, time slider, field/colormap pickers, hot-reload status,
    /// and central 3D [`ViewportWidget`]) using an [`egui_wgpu::RenderState`].
    pub fn show_ui_with_render_state(
        &mut self,
        ui: &mut egui::Ui,
        render_state: &egui_wgpu::RenderState,
    ) {
        let mut egui_renderer = render_state.renderer.write();
        self.show_ui(
            ui,
            &render_state.device,
            &render_state.queue,
            &mut egui_renderer,
        );
    }

    /// Renders the viewer UI using explicit `device`, `queue`, and `egui_renderer` handles.
    pub fn show_ui(
        &mut self,
        ui: &mut egui::Ui,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
        egui_renderer: &mut egui_wgpu::Renderer,
    ) {
        let now = Instant::now();
        if let Err(err) = self.poll_clip_worker(device, queue) {
            log::error!("Interactive clip worker failed: {err}");
        }
        if self.clip_state.worker_busy {
            ui.ctx().request_repaint_after(Duration::from_millis(16));
        }

        if now.duration_since(self.last_reload_poll) >= DEFAULT_HOT_RELOAD_POLL_INTERVAL {
            self.last_reload_poll = now;
            if let Err(err) = self.poll_hot_reload(device, queue) {
                log::error!("Hot reload failed: {err}");
            }
        }
        ui.ctx()
            .request_repaint_after(DEFAULT_HOT_RELOAD_POLL_INTERVAL);

        if self.is_playing && self.doc.num_time_steps() > 1 {
            if now.duration_since(self.last_playback_tick) >= DEFAULT_PLAYBACK_STEP_INTERVAL {
                self.last_playback_tick = now;
                if let Ok(true) = self.doc.step_forward(true)
                    && let Err(err) =
                        self.sync_viewport_after_doc_change(device, queue, true, false)
                {
                    log::error!("Time step advance failed: {err}");
                }
            }
            ui.ctx()
                .request_repaint_after(DEFAULT_PLAYBACK_STEP_INTERVAL);
        }

        let mut step_changed_to: Option<usize> = None;
        let mut field_changed_to: Option<usize> = None;
        let mut colormap_changed_to: Option<ColormapName> = None;
        let mut log_scale_changed_to: Option<bool> = None;
        let mut colorbar_changed_to: Option<bool> = None;
        let mut axes_changed_to: Option<bool> = None;
        let mut trigger_reload = false;
        let mut trigger_reset_camera = false;
        let mut trigger_clear_pick = false;
        let mut clip_drag_update: Option<([f64; 3], [f64; 3], ClipKeep)> = None;
        let mut clip_drag_finished = false;
        let mut clip_disabled = false;

        egui::Panel::left("tetraglass_viewer_panel")
            .resizable(true)
            .default_size(280.0)
            .show(ui, |ui| {
                ui.heading("Tetraglass Viewer");
                ui.separator();

                ui.label(format!("File: {}", self.doc.file_label()));
                if let Some(surface) = self.doc.active_surface() {
                    ui.label(format!(
                        "Surface: {} pts, {} faces ({} tris)",
                        surface.num_points(),
                        surface.num_faces(),
                        surface.num_triangles()
                    ));
                }

                ui.horizontal(|ui| {
                    if ui.button("Reload file").clicked() {
                        trigger_reload = true;
                    }
                    if self.doc.reload_count() > 0 {
                        ui.small(format!("(reloaded {}x)", self.doc.reload_count()));
                    }
                });

                if let Some(err_msg) = self.doc.last_reload_error() {
                    ui.add_space(4.0);
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 50, 47),
                        format!("Hot-reload error: {err_msg}"),
                    );
                }

                // --- Time slider section ---
                let num_steps = self.doc.num_time_steps();
                if num_steps > 1 {
                    ui.separator();
                    ui.label(format!(
                        "Time: t = {:.4} (step {} / {})",
                        self.doc.current_time_value(),
                        self.doc.current_time_step() + 1,
                        num_steps
                    ));
                    let mut step_val = self.doc.current_time_step();
                    if ui
                        .add(
                            egui::Slider::new(&mut step_val, 0..=(num_steps - 1))
                                .text("step")
                                .show_value(true),
                        )
                        .changed()
                    {
                        step_changed_to = Some(step_val);
                    }
                    ui.horizontal(|ui| {
                        if ui.button("|<").on_hover_text("First step").clicked() {
                            step_changed_to = Some(0);
                        }
                        if ui.button("<").on_hover_text("Previous step").clicked()
                            && self.doc.current_time_step() > 0
                        {
                            step_changed_to = Some(self.doc.current_time_step() - 1);
                        }
                        let play_label = if self.is_playing { "Pause" } else { "Play" };
                        if ui.button(play_label).clicked() {
                            self.is_playing = !self.is_playing;
                            self.last_playback_tick = Instant::now();
                        }
                        if ui.button(">").on_hover_text("Next step").clicked()
                            && self.doc.current_time_step() + 1 < num_steps
                        {
                            step_changed_to = Some(self.doc.current_time_step() + 1);
                        }
                        if ui.button(">|").on_hover_text("Last step").clicked() {
                            step_changed_to = Some(num_steps - 1);
                        }
                    });
                }

                // --- Field & Colormap pickers ---
                ui.separator();
                ui.label("Color by:");
                let current_field_label = self
                    .doc
                    .selected_field()
                    .map_or("Solid color", |f| f.label.as_str())
                    .to_string();
                let mut selected_idx = self.doc.selected_field_index();
                egui::ComboBox::from_id_salt("viewer_color_by_picker")
                    .selected_text(current_field_label)
                    .width(230.0)
                    .show_ui(ui, |ui| {
                        for (idx, opt) in self.doc.field_options().iter().enumerate() {
                            if ui
                                .selectable_value(&mut selected_idx, idx, &opt.label)
                                .changed()
                            {
                                field_changed_to = Some(selected_idx);
                            }
                        }
                    });

                ui.add_space(6.0);
                ui.label("Colormap:");
                let mut current_cmap = self.doc.colormap();
                egui::ComboBox::from_id_salt("viewer_colormap_picker")
                    .selected_text(colormap_display_name(current_cmap))
                    .width(230.0)
                    .show_ui(ui, |ui| {
                        for cmap in VIEWER_COLORMAPS {
                            if ui
                                .selectable_value(
                                    &mut current_cmap,
                                    cmap,
                                    colormap_display_name(cmap),
                                )
                                .changed()
                            {
                                colormap_changed_to = Some(current_cmap);
                            }
                        }
                    });

                ui.add_space(6.0);
                let mut log_scale = self.doc.use_log_scale();
                if ui.checkbox(&mut log_scale, "Logarithmic scale").changed() {
                    log_scale_changed_to = Some(log_scale);
                }

                let mut show_colorbar = self.doc.colorbar_enabled();
                if ui.checkbox(&mut show_colorbar, "Show colorbar").changed() {
                    colorbar_changed_to = Some(show_colorbar);
                }

                let mut show_axes = self.doc.axes_enabled();
                if ui.checkbox(&mut show_axes, "Show axes triad").changed() {
                    axes_changed_to = Some(show_axes);
                }

                if let Some([lo, hi]) = self.viewport.effective_range() {
                    ui.label(format!("Range: [{lo:.4}, {hi:.4}]"));
                }

                // --- Interactive clip section (M2.8) ---
                if self.clip_base_tet_view.is_some() {
                    ui.separator();
                    let mut clip_enabled = self.clip_state.enabled;
                    if ui
                        .checkbox(&mut clip_enabled, "Interactive clip plane")
                        .changed()
                    {
                        if clip_enabled {
                            clip_drag_update = Some((
                                self.clip_state.origin,
                                self.clip_state.normal,
                                self.clip_state.keep,
                            ));
                            clip_drag_finished = true;
                        } else {
                            clip_disabled = true;
                        }
                    }

                    if self.clip_state.enabled || clip_enabled {
                        ui.horizontal(|ui| {
                            ui.small("Normal:");
                            for (label, n) in [
                                ("+X", [1.0, 0.0, 0.0]),
                                ("+Y", [0.0, 1.0, 0.0]),
                                ("+Z", [0.0, 0.0, 1.0]),
                                ("-X", [-1.0, 0.0, 0.0]),
                                ("-Y", [0.0, -1.0, 0.0]),
                                ("-Z", [0.0, 0.0, -1.0]),
                            ] {
                                let selected = (self.clip_state.normal[0] - n[0]).abs() < 1e-3
                                    && (self.clip_state.normal[1] - n[1]).abs() < 1e-3
                                    && (self.clip_state.normal[2] - n[2]).abs() < 1e-3;
                                if ui.selectable_label(selected, label).clicked() {
                                    let mut tmp = self.clip_state.clone();
                                    tmp.set_normal(n);
                                    clip_drag_update = Some((tmp.origin, tmp.normal, tmp.keep));
                                    clip_drag_finished = true;
                                }
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.small("Keep:");
                            for (label, k) in [("Back", ClipKeep::Back), ("Front", ClipKeep::Front)]
                            {
                                if ui
                                    .selectable_label(self.clip_state.keep == k, label)
                                    .clicked()
                                {
                                    clip_drag_update =
                                        Some((self.clip_state.origin, self.clip_state.normal, k));
                                    clip_drag_finished = true;
                                }
                            }
                            ui.separator();
                            ui.small("Mode:");
                            for (label, m) in
                                [("Exact", ClipMode::Exact), ("Crinkle", ClipMode::Crinkle)]
                            {
                                if ui
                                    .selectable_label(self.clip_state.mode == m, label)
                                    .clicked()
                                {
                                    self.clip_state.mode = m;
                                    clip_drag_update = Some((
                                        self.clip_state.origin,
                                        self.clip_state.normal,
                                        self.clip_state.keep,
                                    ));
                                    clip_drag_finished = true;
                                }
                            }
                        });

                        let mut offset = self.clip_state.offset;
                        let max_off = self.clip_state.max_offset;
                        let slider_resp = ui.add(
                            egui::Slider::new(&mut offset, -max_off..=max_off)
                                .text("offset")
                                .show_value(true),
                        );
                        if slider_resp.changed() {
                            let mut tmp = self.clip_state.clone();
                            tmp.set_offset(offset);
                            clip_drag_update = Some((tmp.origin, tmp.normal, tmp.keep));
                            if !slider_resp.dragged() {
                                clip_drag_finished = true;
                            }
                        }
                        if slider_resp.drag_stopped() {
                            clip_drag_finished = true;
                        }

                        if self.clip_state.is_dragging {
                            ui.small("Clip status: shader preview (dragging)");
                        } else if self.clip_state.worker_busy {
                            ui.small("Clip status: computing exact clip on worker...");
                        } else if let Some(ref surf) = self.active_exact_clip_surface {
                            ui.small(format!(
                                "Clip status: exact clip ({} tris)",
                                surf.num_triangles()
                            ));
                        }
                    }
                }

                ui.separator();
                let cam = self.viewport.camera();
                ui.label(format!(
                    "Camera: az {:.1}°, el {:.1}°, zoom {:.2}x",
                    cam.azimuth_deg, cam.elevation_deg, cam.zoom
                ));
                if ui.button("Reset camera").clicked() {
                    trigger_reset_camera = true;
                }

                // --- Value probe section ---
                ui.separator();
                ui.label("Value probe:");
                if let Some(pick) = self.viewport.last_pick() {
                    let rep_name = self
                        .doc
                        .evaluated()
                        .representations
                        .get(pick.representation_index)
                        .map_or("surface", |r| r.input.as_str());
                    let cell_label = pick
                        .source_cell_id
                        .map_or_else(|| "n/a".to_string(), |c| format!("#{c}"));
                    ui.monospace(format!(
                        "Input: {rep_name} | Cell {cell_label} (tri #{}, pt #{})",
                        pick.triangle_index, pick.point_index
                    ));
                    let [wx, wy, wz] = pick.world_position;
                    ui.monospace(format!("Pos: [{wx:.4}, {wy:.4}, {wz:.4}]"));
                    if let Some(val) = pick.active_scalar
                        && val.is_finite()
                    {
                        ui.monospace(format!("Active scalar: {val:.5}"));
                    }
                    for pv in &pick.point_values {
                        ui.monospace(format!("pt.{}: {}", pv.name, format_probe_value(pv)));
                    }
                    for cv in &pick.cell_values {
                        ui.monospace(format!("cell.{}: {}", cv.name, format_probe_value(cv)));
                    }
                    if ui.small_button("Clear probe").clicked() {
                        trigger_clear_pick = true;
                    }
                } else {
                    ui.small("Click on mesh in viewport to probe values.");
                }

                ui.separator();
                ui.small("Click: probe value | Left drag: orbit");
                ui.small("Shift/Right drag: pan | Scroll: zoom | Double-click: reset");
            });

        if trigger_reload
            && let Ok(HotReloadOutcome::Reloaded {
                geometry_changed,
                camera_changed,
            }) = self.doc.reload_from_disk()
            && let Err(err) =
                self.sync_viewport_after_doc_change(device, queue, geometry_changed, camera_changed)
        {
            log::error!("Failed to sync viewport after reload: {err}");
        }

        if let Some(new_step) = step_changed_to {
            self.is_playing = false;
            if let Ok(true) = self.doc.set_time_step(new_step)
                && let Err(err) = self.sync_viewport_after_doc_change(device, queue, true, false)
            {
                log::error!("Failed to update viewport time step: {err}");
            }
        }

        let mut style_or_annotations_dirty = false;
        if let Some(new_field_idx) = field_changed_to
            && let Ok(true) = self.doc.set_field_index(new_field_idx)
        {
            style_or_annotations_dirty = true;
        }
        if let Some(new_cmap) = colormap_changed_to
            && let Ok(true) = self.doc.set_colormap(new_cmap)
        {
            style_or_annotations_dirty = true;
        }
        if let Some(new_log) = log_scale_changed_to
            && let Ok(true) = self.doc.set_log_scale(new_log)
        {
            style_or_annotations_dirty = true;
        }
        if let Some(new_cb) = colorbar_changed_to
            && let Ok(true) = self.doc.set_colorbar_enabled(new_cb)
        {
            style_or_annotations_dirty = true;
        }
        if let Some(new_axes) = axes_changed_to
            && self.doc.set_axes_enabled(new_axes)
        {
            style_or_annotations_dirty = true;
        }
        if style_or_annotations_dirty
            && let Err(err) = self.sync_viewport_after_doc_change(device, queue, false, false)
        {
            log::error!("Failed to update viewport style: {err}");
        }

        if trigger_reset_camera {
            self.viewport.set_camera(self.doc.camera_from_scene());
        }
        if trigger_clear_pick {
            self.viewport.clear_pick();
        }
        if clip_disabled {
            if let Err(err) = self.disable_interactive_clip(device, queue) {
                log::error!("Failed to disable interactive clip: {err}");
            }
        } else {
            if let Some((origin, normal, keep)) = clip_drag_update
                && let Err(err) =
                    self.begin_or_update_clip_drag(device, queue, origin, normal, keep)
            {
                log::error!("Failed to update shader clip plane: {err}");
            }
            if clip_drag_finished {
                self.end_clip_drag();
            }
        }

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                if let Err(err) =
                    ViewportWidget::new(&mut self.viewport).show(ui, device, egui_renderer)
                {
                    ui.colored_label(egui::Color32::RED, format!("Viewport error: {err}"));
                }
            });
    }

    /// Frees GPU texture bindings from `egui_renderer`.
    pub fn destroy(&self, egui_renderer: &mut egui_wgpu::Renderer) {
        self.viewport.destroy(egui_renderer);
    }

    /// Frees GPU texture bindings using an [`egui_wgpu::RenderState`].
    pub fn destroy_with_render_state(&self, render_state: &egui_wgpu::RenderState) {
        self.viewport.destroy_with_render_state(render_state);
    }
}

/// Launch options for [`run_viewer`] and [`ViewerApp`].
#[derive(Debug, Clone)]
pub struct ViewerOptions {
    /// Path to the `.json` scene file or mesh file (`.vtu`, `.pvd`, `.msh`) to open.
    pub path: PathBuf,
    /// Optional frame limit for automated smoke tests (`--frames <n>`).
    pub max_frames: Option<u64>,
    /// Initial window size `[width, height]` in logical points.
    pub window_size: [f32; 2],
}

impl ViewerOptions {
    /// Creates [`ViewerOptions`] for `path` with default window dimensions (`[1200.0, 780.0]`).
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            max_frames: None,
            window_size: [1200.0, 780.0],
        }
    }
}

/// Native `eframe` desktop application wrapping a [`ViewerSession`].
pub struct ViewerApp {
    session: ViewerSession,
    render_state: egui_wgpu::RenderState,
    frame_count: u64,
    max_frames: Option<u64>,
}

impl ViewerApp {
    /// Creates a [`ViewerApp`] from an `eframe::CreationContext` and a pre-loaded [`ViewerDocument`].
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        doc: ViewerDocument,
        max_frames: Option<u64>,
    ) -> Result<Self, ViewerError> {
        let render_state = cc.wgpu_render_state.clone().ok_or_else(|| {
            ViewerError::Window("eframe was not initialized with wgpu".to_string())
        })?;
        let session = ViewerSession::from_render_state(doc, &render_state)?;
        Ok(Self {
            session,
            render_state,
            frame_count: 0,
            max_frames,
        })
    }

    /// Returns a reference to the underlying [`ViewerSession`].
    #[must_use]
    pub const fn session(&self) -> &ViewerSession {
        &self.session
    }
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.session
            .show_ui_with_render_state(ui, &self.render_state);

        self.frame_count += 1;
        if let Some(limit) = self.max_frames {
            if self.frame_count >= limit {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            } else {
                ui.ctx().request_repaint();
            }
        }
    }

    fn on_exit(&mut self) {
        self.session.destroy_with_render_state(&self.render_state);
    }
}

/// Validates and loads the scene at `options.path`, then opens the native `eframe` desktop viewer.
pub fn run_viewer(options: ViewerOptions) -> Result<(), ViewerError> {
    // Load and validate the document before opening the window so CLI errors are reported cleanly.
    let doc = ViewerDocument::load_path(&options.path)?;
    let title = format!("Tetraglass Viewer — {}", doc.file_label());
    let max_frames = options.max_frames;

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(options.window_size)
            .with_title(title),
        ..Default::default()
    };

    eframe::run_native(
        "tetraglass_viewer",
        native_options,
        Box::new(move |cc| Ok(Box::new(ViewerApp::new(cc, doc, max_frames)?))),
    )
    .map_err(|e| ViewerError::Window(e.to_string()))
}

// --- Internal helpers ---

fn is_scene_json_path_or_bytes(path: &Path, bytes: &[u8]) -> bool {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    {
        return true;
    }
    let trimmed = bytes.trim_ascii_start();
    trimmed.starts_with(b"{")
}

fn resolve_relative_path(base_dir: &Path, file: &str) -> PathBuf {
    let p = Path::new(file);
    if p.is_absolute() || base_dir.as_os_str().is_empty() {
        p.to_path_buf()
    } else {
        base_dir.join(p)
    }
}

fn load_series_from_path(
    path: &Path,
    explicit_format: Option<MeshFormat>,
) -> Result<(MeshFormat, TimeSeries<IoError>), ViewerError> {
    let bytes = fs::read(path)?;
    load_series_from_bytes(path, bytes, explicit_format)
}

fn load_series_from_bytes(
    path: &Path,
    bytes: Vec<u8>,
    explicit_format: Option<MeshFormat>,
) -> Result<(MeshFormat, TimeSeries<IoError>), ViewerError> {
    let format = explicit_format
        .or_else(|| {
            path.extension()
                .and_then(|e| e.to_str())
                .and_then(detect_format_from_extension)
        })
        .or_else(|| detect_format_from_bytes(&bytes))
        .ok_or_else(|| {
            ViewerError::MeshIo(IoError::UnsupportedFeature {
                offset: 0,
                feature: format!("unknown mesh format for `{}`", path.display()),
            })
        })?;

    match format {
        MeshFormat::Vtu => Ok((format, read_vtu_series(bytes)?)),
        MeshFormat::Gmsh => Ok((format, read_msh_series(bytes)?)),
        MeshFormat::Pvd => {
            let base_dir = path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            let series = read_pvd_series_with_loader(&bytes, move |rel_file| {
                let step_path = resolve_relative_path(&base_dir, rel_file);
                Ok(fs::read(step_path)?)
            })?;
            Ok((format, series))
        }
        other => Err(ViewerError::MeshIo(IoError::UnsupportedFeature {
            offset: 0,
            feature: format!("format `{other:?}` not yet supported"),
        })),
    }
}

fn load_and_validate_sources(
    scene: &Scene,
    base_dir: &Path,
) -> Result<(SceneEvaluator, Vec<f64>, Vec<PathBuf>), ViewerError> {
    let mut evaluator = SceneEvaluator::new();
    let mut datasets = BTreeMap::new();
    let mut primary_times = vec![0.0];
    let mut source_paths = Vec::new();

    let initial_selector = scene.output.as_ref().map_or(TimeSelector::LAST, |o| o.time);

    for (name, source) in &scene.sources {
        let resolved_path = resolve_relative_path(base_dir, &source.file);
        source_paths.push(resolved_path.clone());
        let explicit_format = source.format.map(MeshFormat::from);
        let (_, series) = load_series_from_path(&resolved_path, explicit_format)?;
        let times = series.times().to_vec();
        let probe_step = initial_selector
            .resolve_step(&times)
            .map_or(0, |(idx, _)| idx);
        let probe_grid = series.step(probe_step)?;
        if primary_times.len() <= 1 && !times.is_empty() {
            primary_times.clone_from(&times);
        }
        datasets.insert(
            name.clone(),
            DatasetMetadata::from_grid(&probe_grid).with_time_steps(times),
        );
        evaluator.insert_series_source(name.clone(), series);
    }

    if let Err(report) = validate_scene_with_metadata(scene, &datasets) {
        return Err(ViewerError::Validation(report.to_string()));
    }

    Ok((evaluator, primary_times, source_paths))
}

fn validate_scene_against_evaluator(
    scene: &Scene,
    evaluator: &SceneEvaluator,
) -> Result<(), ViewerError> {
    let initial_selector = scene.output.as_ref().map_or(TimeSelector::LAST, |o| o.time);
    let mut datasets = BTreeMap::new();
    for name in scene.sources.keys() {
        if let Some(series) = evaluator.series_source(name) {
            let times = series.times().to_vec();
            let probe_step = initial_selector
                .resolve_step(&times)
                .map_or(0, |(idx, _)| idx);
            let probe_grid = series.step(probe_step)?;
            datasets.insert(
                name.clone(),
                DatasetMetadata::from_grid(&probe_grid).with_time_steps(times),
            );
        }
    }
    if let Err(report) = validate_scene_with_metadata(scene, &datasets) {
        return Err(ViewerError::Validation(report.to_string()));
    }
    Ok(())
}

fn build_default_scene_for_mesh(
    path: &Path,
    bytes: &[u8],
) -> Result<(Scene, SceneEvaluator, Vec<f64>), ViewerError> {
    let (_, series) = load_series_from_bytes(path, bytes.to_vec(), None)?;
    let first_grid = series.step(0)?;
    let time_steps = series.times().to_vec();

    // Pick the first point array or cell array as default coloring when opening a raw mesh.
    let default_color_by = first_grid
        .point_data()
        .names()
        .next()
        .map(|name| format!("point:{name}"))
        .or_else(|| {
            first_grid
                .cell_data()
                .names()
                .next()
                .map(|name| format!("cell:{name}"))
        });

    let mut rep = Representation::new("mesh");
    rep.colorbar = ColorbarSpec::Enabled(default_color_by.is_some());
    rep.color_by = default_color_by;

    let mut sources = BTreeMap::new();
    sources.insert("mesh".to_string(), Source::new(path.display().to_string()));

    let mut output = Output::new("out.png");
    output.time = TimeSelector::LAST;

    let scene = Scene {
        version: 1,
        sources,
        filters: BTreeMap::new(),
        view: View {
            show: vec![rep],
            camera: CameraConfig::default(),
            lights: LightingPreset::Default,
            background: BackgroundSpec::Solid(ColorSpec::Named("white".to_string())),
            axes: true,
            title: None,
        },
        output: Some(output),
    };

    let mut evaluator = SceneEvaluator::new();
    evaluator.insert_series_source("mesh", series);
    Ok((scene, evaluator, time_steps))
}

fn color_by_to_scene_string(color_by: &ViewportColorBy) -> Option<String> {
    match color_by {
        ViewportColorBy::Solid => None,
        ViewportColorBy::Point { name, component } => {
            Some(format_array_ref("point", name, *component))
        }
        ViewportColorBy::Cell { name, component } => {
            Some(format_array_ref("cell", name, *component))
        }
    }
}

fn format_array_ref(prefix: &str, name: &str, component: ViewportScalarComponent) -> String {
    match component {
        ViewportScalarComponent::First => format!("{prefix}:{name}"),
        ViewportScalarComponent::Magnitude => format!("{prefix}:{name}:magnitude"),
        ViewportScalarComponent::Index(idx) => format!("{prefix}:{name}:{idx}"),
    }
}

const fn colormap_display_name(name: ColormapName) -> &'static str {
    match name {
        ColormapName::Viridis => "viridis",
        ColormapName::Cividis => "cividis",
        ColormapName::CoolWarm => "cool_warm",
        ColormapName::Turbo => "turbo",
        ColormapName::Grayscale => "grayscale",
    }
}

/// Maps a [`ColormapName`] from `tetraglass-scene` to a [`ViewportColormap`] in `tetraglass-render`.
#[must_use]
pub const fn map_colormap_kind(name: ColormapName) -> ViewportColormap {
    match name {
        ColormapName::Viridis | ColormapName::Turbo => ViewportColormap::Viridis,
        ColormapName::Cividis | ColormapName::Grayscale => ViewportColormap::Cividis,
        ColormapName::CoolWarm => ViewportColormap::CoolWarm,
    }
}

/// Builds a [`ViewportSurfaceStyle`] from an [`EvaluatedRepresentation`].
#[must_use]
pub fn build_surface_style(rep: &EvaluatedRepresentation) -> ViewportSurfaceStyle {
    let solid_color_srgb = [rep.color[0], rep.color[1], rep.color[2], rep.opacity as f32];
    let edge_color_srgb = [
        rep.edge_color[0],
        rep.edge_color[1],
        rep.edge_color[2],
        rep.opacity as f32,
    ];
    let (draw_surface, draw_edges) = match rep.style {
        RepresentationStyle::Wireframe => (false, true),
        RepresentationStyle::SurfaceWithEdges => (true, true),
        _ => (true, false),
    };
    match &rep.coloring {
        None => ViewportSurfaceStyle {
            color_by: ViewportColorBy::Solid,
            colormap: ViewportColormap::Viridis,
            range: None,
            use_log_scale: false,
            solid_color_srgb,
            draw_surface,
            draw_edges,
            edge_color_srgb,
            clip_plane: None,
        },
        Some(c) => ViewportSurfaceStyle {
            color_by: build_color_by(c),
            colormap: map_colormap_kind(c.colormap),
            range: Some(c.effective_range),
            use_log_scale: c.log_scale,
            solid_color_srgb,
            draw_surface,
            draw_edges,
            edge_color_srgb,
            clip_plane: None,
        },
    }
}

fn build_color_by(coloring: &EvaluatedColoring) -> ViewportColorBy {
    let component = match coloring.component {
        ResolvedComponent::Magnitude => ViewportScalarComponent::Magnitude,
        ResolvedComponent::Component(0) => ViewportScalarComponent::First,
        ResolvedComponent::Component(idx) => ViewportScalarComponent::Index(idx),
    };
    match coloring.association {
        ArrayAssociation::Point => ViewportColorBy::Point {
            name: coloring.array_name.clone(),
            component,
        },
        ArrayAssociation::Cell => ViewportColorBy::Cell {
            name: coloring.array_name.clone(),
            component,
        },
    }
}

/// Builds renderer-drawn [`Annotations`] from a [`Scene`] and its [`EvaluatedScene`].
#[must_use]
pub fn build_annotations_from_scene(scene: &Scene, evaluated: &EvaluatedScene) -> Annotations {
    let mut colorbars = Vec::new();
    for rep in &evaluated.representations {
        if !rep.colorbar.is_enabled() {
            continue;
        }
        let Some(coloring) = &rep.coloring else {
            continue;
        };
        let (title, position) = match &rep.colorbar {
            ColorbarSpec::Enabled(_) => (Some(coloring.raw_ref.clone()), ColorbarPlacement::Right),
            ColorbarSpec::Options(opts) => (
                opts.title
                    .clone()
                    .or_else(|| Some(coloring.raw_ref.clone())),
                ColorbarPlacement::from(opts.position),
            ),
        };
        colorbars.push(ColorbarAnnotation {
            colormap: map_colormap_kind(coloring.colormap),
            range: coloring.effective_range,
            use_log_scale: coloring.log_scale,
            title,
            position,
            tick_count: DEFAULT_COLORBAR_TICKS,
        });
    }

    Annotations {
        title: scene.view.title.clone(),
        axes: scene.view.axes,
        colorbars,
    }
}

/// Builds a [`ViewportCamera`] from a [`Scene`] and dataset [`BoundingBox`].
#[must_use]
pub fn build_camera_from_scene(scene: &Scene, bounds: BoundingBox) -> ViewportCamera {
    let cam_cfg = &scene.view.camera;
    let mut camera = ViewportCamera::fit_bounds(bounds, cam_cfg.azimuth, cam_cfg.elevation);
    camera.roll_deg = cam_cfg.roll;
    camera.zoom = cam_cfg.zoom;
    camera.view_angle_deg = cam_cfg.view_angle;

    let center = bounds.center();
    if let Some(fp) = cam_cfg.focal_point {
        camera.focal_point = [fp[0] - center[0], fp[1] - center[1], fp[2] - center[2]];
    }
    if let Some(pos) = cam_cfg.position {
        let rel_pos = [pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]];
        let dx = rel_pos[0] - camera.focal_point[0];
        let dy = rel_pos[1] - camera.focal_point[1];
        let dz = rel_pos[2] - camera.focal_point[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-5);
        let el = (dy / dist).clamp(-1.0, 1.0).asin().to_degrees();
        let az = dx.atan2(dz).to_degrees();
        camera.distance = dist;
        camera.zoom = 1.0;
        camera.azimuth_deg = az;
        camera.elevation_deg = el.clamp(-89.0, 89.0);
    }
    camera
}

/// Resolves a [`BackgroundSpec`] to an sRGB `[r, g, b, a]` clear color.
#[must_use]
pub fn resolve_background_srgb(bg: &BackgroundSpec) -> [f32; 4] {
    match bg {
        BackgroundSpec::Solid(color) => {
            let [r, g, b] = color.resolve_rgb().unwrap_or([1.0, 1.0, 1.0]);
            [r, g, b, 1.0]
        }
        BackgroundSpec::Gradient(grad) => {
            let top = grad.top.resolve_rgb().unwrap_or([1.0, 1.0, 1.0]);
            let bottom = grad.bottom.resolve_rgb().unwrap_or([1.0, 1.0, 1.0]);
            [
                0.5 * (top[0] + bottom[0]),
                0.5 * (top[1] + bottom[1]),
                0.5 * (top[2] + bottom[2]),
                1.0,
            ]
        }
    }
}

fn format_probe_value(pv: &PickProbeValue) -> String {
    match pv.values.as_slice() {
        [] => "[]".to_string(),
        [s] => format!("{s:.5}"),
        comps => {
            let joined = comps
                .iter()
                .map(|c| format!("{c:.4}"))
                .collect::<Vec<_>>()
                .join(", ");
            if let Some(mag) = pv.magnitude {
                format!("[{joined}] (|v|={mag:.4})")
            } else {
                format!("[{joined}]")
            }
        }
    }
}

/// Re-exported clip-plane and pick result types (`M2.7`, `M2.8`).
pub use tetraglass_egui::{
    ViewportClipKeep, ViewportClipPlane, ViewportPickProbeValue, ViewportPickResult,
};

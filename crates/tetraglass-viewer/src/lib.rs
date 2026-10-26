//! Interactive desktop and WebGPU (`eframe`) viewer for Tetraglass scenes and meshes (`M1.8`, `M3.1`).
//!
//! # Architecture & invariants
//!
//! - [`ViewerDocument`] manages the serializable [`Scene`], [`SceneEvaluator`], lazy [`TimeSeries`]
//!   sources, time-step selection, field/colormap pickers, and [`SceneFileWatcher`] hot reload
//!   without depending on `wgpu`.
//! - [`ViewerSession`] and [`ViewerApp`] bridge [`ViewerDocument`] to [`tetraglass_egui::ViewportState`]
//!   and [`tetraglass_egui::ViewportWidget`]. All `wgpu` rendering remains inside `tetraglass-render`.
//! - [`web_support`] (and `web` on `wasm32-unknown-unknown`) provides WebGPU-only setup (`D-007`),
//!   clear user-facing diagnostics when WebGPU is unavailable, drag-and-drop byte-buffer ingestion,
//!   and relative-aware URL loading (`M3.1`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::{Duration, SystemTime};

use eframe::egui;
use tetraglass_core::{
    BoundingBox, SurfaceMesh, SyntheticCellLayout, SyntheticGridSpec, TetView, TimeSeries,
    UnstructuredGrid,
};
use tetraglass_egui::{
    FieldOption, ViewportCamera, ViewportColorBy, ViewportColormap, ViewportScalarComponent,
    ViewportState, ViewportSurfaceStyle, ViewportWidget, available_field_options,
};
use tetraglass_filters::{ClipParams, clip, extract_tet_view_surface};
use tetraglass_io::{
    IoError, MeshFormat, detect_format_from_bytes, detect_format_from_extension, read_msh_series,
    read_pvd_series_with_loader, read_vtk_series, read_vtu_series,
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

pub mod web_support;
pub use web_support::{
    DroppedFilePayload, FetchedUrlBundle, InMemoryFileStore, InlineShareableBundle,
    MAX_SHAREABLE_DECOMPRESSED_BYTES, ShareableLinkSpec, ShareableViewOverrides,
    WebGpuErrorMessage, WebGpuStatus, decode_compressed_base64url, encode_compressed_base64url,
    extract_url_from_query_map, extract_url_from_query_str, fetch_shareable_link_with,
    fetch_url_bundle_with, parse_query_map, percent_encode_query_component, resolve_relative_url,
    url_display_label, webgpu_only_wgpu_configuration,
};

/// Browser (`wasm32-unknown-unknown`) entry points, WebGPU capability checks, and `fetch` loader (`M3.1`).
#[cfg(target_arch = "wasm32")]
pub mod web;
#[cfg(target_arch = "wasm32")]
pub use web::{
    WebViewerHandle, check_webgpu_status_sync, detect_webgpu_status, fetch_bytes_from_url,
    fetch_url_bundle_async, render_webgpu_error_banner, start_web_viewer_default,
};

#[derive(Debug, Clone, Copy)]
struct MonotonicInstant {
    #[cfg(not(target_arch = "wasm32"))]
    inner: std::time::Instant,
    #[cfg(target_arch = "wasm32")]
    secs: f64,
}

impl MonotonicInstant {
    fn now() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                inner: std::time::Instant::now(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let secs = web_sys::window()
                .and_then(|w| w.performance())
                .map_or(0.0, |p| p.now() / 1000.0);
            Self { secs }
        }
    }

    fn duration_since(self, earlier: Self) -> Duration {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.duration_since(earlier.inner)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Duration::from_secs_f64((self.secs - earlier.secs).max(0.0))
        }
    }
}

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
    source_url: Option<String>,
    primary_bytes: Option<Vec<u8>>,
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
    opacity: f64,
    transfer_function: Vec<[f64; 2]>,
    ssao: ViewportSsaoConfig,
    path_tracing_enabled: bool,
    path_trace_config: ViewportPathTraceConfig,
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
            source_url: None,
            primary_bytes: Some(bytes),
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
            opacity: 1.0,
            transfer_function: vec![[0.0, 0.0], [1.0, 1.0]],
            ssao: ViewportSsaoConfig::default(),
            path_tracing_enabled: false,
            path_trace_config: ViewportPathTraceConfig::default(),
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
            source_url: None,
            primary_bytes: None,
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
            opacity: 1.0,
            transfer_function: vec![[0.0, 0.0], [1.0, 1.0]],
            ssao: ViewportSsaoConfig::default(),
            path_tracing_enabled: false,
            path_trace_config: ViewportPathTraceConfig::default(),
            watcher: None,
            last_reload_error: None,
            reload_count: 0,
        };
        doc.sync_ui_controls_from_scene();
        Ok(doc)
    }

    /// Creates a synthetic default [`ViewerDocument`] (used when the web viewer opens without a URL parameter).
    pub fn default_demo() -> Result<Self, ViewerError> {
        let grid = SyntheticGridSpec::new(6, 6, 6, SyntheticCellLayout::Hexahedra)
            .generate()
            .map_err(|e| ViewerError::Validation(format!("synthetic demo mesh error: {e}")))?;
        Self::from_grid("synthetic_demo.vtu", grid)
    }

    /// Creates a [`ViewerDocument`] from an in-memory [`UnstructuredGrid`] and display label.
    pub fn from_grid(
        file_label: impl Into<String>,
        grid: UnstructuredGrid,
    ) -> Result<Self, ViewerError> {
        let file_label = file_label.into();
        let default_color_by = grid
            .point_data()
            .names()
            .next()
            .map(|name| format!("point:{name}"))
            .or_else(|| {
                grid.cell_data()
                    .names()
                    .next()
                    .map(|name| format!("cell:{name}"))
            });

        let mut rep = Representation::new("mesh");
        rep.colorbar = ColorbarSpec::Enabled(default_color_by.is_some());
        rep.color_by = default_color_by;

        let mut sources = BTreeMap::new();
        sources.insert("mesh".to_string(), Source::new(file_label.clone()));

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
        evaluator.insert_source("mesh", grid);
        let evaluated = evaluator.evaluate_scene_at_time(&scene, TimeSelector::LAST)?;
        let current_step = evaluated.time_step;

        let mut doc = Self {
            file_label,
            scene_path: None,
            source_url: None,
            primary_bytes: None,
            is_raw_mesh_path: true,
            base_dir: PathBuf::new(),
            scene,
            evaluator,
            evaluated,
            time_steps: vec![0.0],
            current_step,
            field_options: Vec::new(),
            selected_field_idx: 0,
            colormap: ColormapName::Viridis,
            use_log_scale: false,
            show_colorbar: false,
            show_axes: true,
            opacity: 1.0,
            transfer_function: vec![[0.0, 0.0], [1.0, 1.0]],
            ssao: ViewportSsaoConfig::default(),
            path_tracing_enabled: false,
            path_trace_config: ViewportPathTraceConfig::default(),
            watcher: None,
            last_reload_error: None,
            reload_count: 0,
        };
        doc.sync_ui_controls_from_scene();
        Ok(doc)
    }

    /// Creates a [`ViewerDocument`] from in-memory mesh bytes (`.vtu` or `.msh`).
    pub fn from_mesh_bytes(
        file_name: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Result<Self, ViewerError> {
        let store = InMemoryFileStore::new();
        Self::from_file_bytes_with_store(file_name, bytes, &store)
    }

    /// Creates a [`ViewerDocument`] from an in-memory primary file (`file_name`, `bytes`) and an
    /// [`InMemoryFileStore`] supplying any companion files (for `.json` scenes or `.pvd` series).
    pub fn from_file_bytes_with_store(
        file_name: impl Into<String>,
        bytes: Vec<u8>,
        store: &InMemoryFileStore,
    ) -> Result<Self, ViewerError> {
        let file_name = file_name.into();
        let virtual_path = Path::new(&file_name);
        let is_json = is_scene_json_path_or_bytes(virtual_path, &bytes);
        let file_label = virtual_path
            .file_name()
            .map_or_else(|| file_name.clone(), |s| s.to_string_lossy().into_owned());

        let (scene, mut evaluator, time_steps) = if is_json {
            let scene = Scene::from_json_bytes(&bytes)?;
            let (evaluator, time_steps) =
                load_and_validate_sources_from_store(&scene, store, None)?;
            (scene, evaluator, time_steps)
        } else {
            build_default_scene_for_mesh_with_store(virtual_path, &bytes, store)?
        };

        let initial_selector = scene.output.as_ref().map_or(TimeSelector::LAST, |o| o.time);
        let evaluated = evaluator.evaluate_scene_at_time(&scene, initial_selector)?;
        let current_step = evaluated.time_step;

        let mut doc = Self {
            file_label,
            scene_path: None,
            source_url: None,
            primary_bytes: Some(bytes),
            is_raw_mesh_path: !is_json,
            base_dir: PathBuf::new(),
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
            opacity: 1.0,
            transfer_function: vec![[0.0, 0.0], [1.0, 1.0]],
            ssao: ViewportSsaoConfig::default(),
            path_tracing_enabled: false,
            path_trace_config: ViewportPathTraceConfig::default(),
            watcher: None,
            last_reload_error: None,
            reload_count: 0,
        };
        doc.sync_ui_controls_from_scene();
        Ok(doc)
    }

    /// Creates a [`ViewerDocument`] from a pre-fetched [`FetchedUrlBundle`] (`M3.1`, `M3.2`).
    pub fn from_fetched_bundle(bundle: FetchedUrlBundle) -> Result<Self, ViewerError> {
        let mut doc = Self::from_file_bytes_with_store(
            bundle.label.clone(),
            bundle.primary_bytes,
            &bundle.companion_files,
        )?;
        doc.file_label = bundle.label;
        if !bundle.url.starts_with("inline:") {
            doc.source_url = Some(bundle.url);
        }
        doc.apply_shareable_overrides(&bundle.view_overrides)?;
        Ok(doc)
    }

    /// Applies optional [`ShareableViewOverrides`] (camera, time step, field coloring, colormap,
    /// log scale, colorbar, and axes triad) to this document (`M3.2`).
    pub fn apply_shareable_overrides(
        &mut self,
        overrides: &ShareableViewOverrides,
    ) -> Result<(), ViewerError> {
        if overrides.is_empty() {
            return Ok(());
        }

        if let Some(az) = overrides.azimuth_deg {
            self.scene.view.camera.azimuth = az;
            self.scene.view.camera.position = None;
        }
        if let Some(el) = overrides.elevation_deg {
            self.scene.view.camera.elevation = el;
            self.scene.view.camera.position = None;
        }
        if let Some(roll) = overrides.roll_deg {
            self.scene.view.camera.roll = roll;
        }
        if let Some(zoom) = overrides.zoom {
            self.scene.view.camera.zoom = zoom;
            self.scene.view.camera.position = None;
        }

        if let Some(step) = overrides.step {
            self.set_time_step(step)?;
        }
        if let Some(ref color_by) = overrides.color_by {
            if color_by.eq_ignore_ascii_case("solid") || color_by.eq_ignore_ascii_case("none") {
                self.set_color_by(None)?;
            } else {
                self.set_color_by(Some(color_by.clone()))?;
            }
        }
        if let Some(cmap) = overrides.colormap {
            self.set_colormap(cmap)?;
        }
        if let Some(log) = overrides.log_scale {
            self.set_log_scale(log)?;
        }
        if let Some(cb) = overrides.colorbar {
            self.set_colorbar_enabled(cb)?;
        }
        if let Some(axes) = overrides.axes {
            self.set_axes_enabled(axes);
        }
        if let Some(opacity) = overrides.opacity {
            self.set_opacity(opacity)?;
        }
        Ok(())
    }

    /// Fetches `url` (and any referenced scene sources or `.pvd` step files) using `fetcher` and
    /// builds a [`ViewerDocument`] (`M3.1`, `M3.2`).
    pub fn from_url_with_fetcher<F>(url: &str, fetcher: F) -> Result<Self, ViewerError>
    where
        F: FnMut(&str) -> Result<Vec<u8>, String>,
    {
        let bundle = fetch_url_bundle_with(url, fetcher)?;
        Self::from_fetched_bundle(bundle)
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

    /// Returns the remote URL from which this document was loaded, if any (`M3.2`).
    #[must_use]
    pub fn source_url(&self) -> Option<&str> {
        self.source_url.as_deref()
    }

    /// Returns the raw primary file bytes if this document was loaded from a file or URL (`M3.2`).
    #[must_use]
    pub fn primary_bytes(&self) -> Option<&[u8]> {
        self.primary_bytes.as_deref()
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
            if log_scale
                && let tetraglass_scene::RangeSpec::Bounds([lo, hi]) = rep.range
                && lo <= 0.0
            {
                rep.range = if hi > 0.0 {
                    tetraglass_scene::RangeSpec::bounds((hi * 1e-6).min(1e-6), hi)
                } else {
                    tetraglass_scene::RangeSpec::AUTO
                };
            }
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

    /// Returns the surface opacity (`0.0..=1.0`) of the primary representation (`M3.3`).
    #[must_use]
    pub const fn opacity(&self) -> f64 {
        self.opacity
    }

    /// Sets the surface opacity (`0.0..=1.0`) of the primary representation (`M3.3`).
    pub fn set_opacity(&mut self, opacity: f64) -> Result<bool, ViewerError> {
        if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
            return Err(ViewerError::Validation(format!(
                "opacity `{opacity}` must be finite and in [0.0, 1.0]"
            )));
        }
        if (self.opacity - opacity).abs() <= 1e-6 {
            return Ok(false);
        }
        self.opacity = opacity;
        if let Some(rep) = self.scene.view.show.first_mut() {
            rep.opacity = opacity;
        }
        self.evaluated = self
            .evaluator
            .evaluate_scene_at_time(&self.scene, TimeSelector::step(self.current_step))?;
        Ok(true)
    }

    /// Returns whether the primary representation is a volume representation (`M4.4b`).
    #[must_use]
    pub fn is_volume_representation(&self) -> bool {
        self.evaluated
            .representations
            .first()
            .is_some_and(|r| r.volume.is_some())
    }

    /// Returns the active piecewise-linear opacity transfer function control points `[[t, alpha], ...]` (`M4.4b`).
    #[must_use]
    pub fn transfer_function(&self) -> &[[f64; 2]] {
        &self.transfer_function
    }

    /// Updates the piecewise-linear opacity transfer function control points `[[t, alpha], ...]`
    /// on the primary representation without re-running upstream filters (`M4.4b`).
    pub fn set_transfer_function(&mut self, points: Vec<[f64; 2]>) -> Result<bool, ViewerError> {
        if points.len() < 2 {
            return Err(ViewerError::Validation(
                "transfer_function must have at least 2 control points".to_string(),
            ));
        }
        for &[t, alpha] in &points {
            if !t.is_finite()
                || !(0.0..=1.0).contains(&t)
                || !alpha.is_finite()
                || !(0.0..=1.0).contains(&alpha)
            {
                return Err(ViewerError::Validation(format!(
                    "transfer_function point `[{t}, {alpha}]` must have finite coordinates in [0.0, 1.0]"
                )));
            }
        }
        for pair in points.windows(2) {
            if pair[1][0] < pair[0][0] {
                return Err(ViewerError::Validation(
                    "transfer_function control points must have non-decreasing normalized positions"
                        .to_string(),
                ));
            }
        }
        if self.transfer_function == points {
            return Ok(false);
        }
        self.transfer_function.clone_from(&points);
        if let Some(rep) = self.scene.view.show.first_mut() {
            rep.transfer_function = Some(points);
        }
        self.evaluated = self
            .evaluator
            .evaluate_scene_at_time(&self.scene, TimeSelector::step(self.current_step))?;
        Ok(true)
    }

    /// Returns the active [`ViewportSsaoConfig`] (`M5.2`).
    #[must_use]
    pub const fn ssao(&self) -> ViewportSsaoConfig {
        self.ssao
    }

    /// Updates the [`ViewportSsaoConfig`] on this document (`M5.2`).
    pub fn set_ssao(&mut self, ssao: ViewportSsaoConfig) -> bool {
        if self.ssao == ssao {
            return false;
        }
        self.ssao = ssao;
        true
    }

    /// Enables or disables screen-space ambient occlusion (`M5.2`).
    pub fn set_ssao_enabled(&mut self, enabled: bool) -> bool {
        if self.ssao.enabled == enabled {
            return false;
        }
        self.ssao.enabled = enabled;
        true
    }

    /// Sets the SSAO darkening strength in `[0.0, 2.0]` (`M5.2`).
    pub fn set_ssao_strength(&mut self, strength: f32) -> Result<bool, ViewerError> {
        if !strength.is_finite() || !(0.0..=2.0).contains(&strength) {
            return Err(ViewerError::Validation(format!(
                "ssao strength `{strength}` must be finite and in [0.0, 2.0]"
            )));
        }
        if (self.ssao.strength - strength).abs() <= 1e-6 {
            return Ok(false);
        }
        self.ssao.strength = strength;
        Ok(true)
    }

    /// Sets the SSAO normalized hemisphere radius in `[0.005, 1.0]` (`M5.2`).
    pub fn set_ssao_radius(&mut self, radius: f32) -> Result<bool, ViewerError> {
        if !radius.is_finite() || !(0.005..=1.0).contains(&radius) {
            return Err(ViewerError::Validation(format!(
                "ssao radius `{radius}` must be finite and in [0.005, 1.0]"
            )));
        }
        if (self.ssao.radius - radius).abs() <= 1e-6 {
            return Ok(false);
        }
        self.ssao.radius = radius;
        Ok(true)
    }

    /// Returns whether progressive path tracing is enabled on this document (`M5.3b`).
    #[must_use]
    pub const fn path_tracing_enabled(&self) -> bool {
        self.path_tracing_enabled
    }

    /// Enables or disables progressive path tracing (`M5.3b`).
    pub fn set_path_tracing_enabled(&mut self, enabled: bool) -> bool {
        if self.path_tracing_enabled == enabled {
            return false;
        }
        self.path_tracing_enabled = enabled;
        true
    }

    /// Returns the active [`ViewportPathTraceConfig`] (`M5.3b`).
    #[must_use]
    pub const fn path_trace_config(&self) -> ViewportPathTraceConfig {
        self.path_trace_config
    }

    /// Updates the [`ViewportPathTraceConfig`] on this document (`M5.3b`).
    pub fn set_path_trace_config(
        &mut self,
        config: ViewportPathTraceConfig,
    ) -> Result<bool, ViewerError> {
        if !(1..=1024).contains(&config.samples_per_pixel) {
            return Err(ViewerError::Validation(format!(
                "path_trace samples_per_pixel `{}` must be in [1, 1024]",
                config.samples_per_pixel
            )));
        }
        if !(1..=8).contains(&config.max_bounces) {
            return Err(ViewerError::Validation(format!(
                "path_trace max_bounces `{}` must be in [1, 8]",
                config.max_bounces
            )));
        }
        if !config.light_softness.is_finite() || !(0.0..=0.5).contains(&config.light_softness) {
            return Err(ViewerError::Validation(format!(
                "path_trace light_softness `{}` must be finite and in [0.0, 0.5]",
                config.light_softness
            )));
        }
        if self.path_trace_config == config {
            return Ok(false);
        }
        self.path_trace_config = config;
        Ok(true)
    }

    /// Sets the target sample count per pixel in `[1, 1024]` (`M5.3b`).
    pub fn set_path_trace_samples(&mut self, samples_per_pixel: u32) -> Result<bool, ViewerError> {
        let mut cfg = self.path_trace_config;
        cfg.samples_per_pixel = samples_per_pixel;
        self.set_path_trace_config(cfg)
    }

    /// Sets the maximum path bounce depth in `[1, 8]` (`M5.3b`).
    pub fn set_path_trace_bounces(&mut self, max_bounces: u32) -> Result<bool, ViewerError> {
        let mut cfg = self.path_trace_config;
        cfg.max_bounces = max_bounces;
        self.set_path_trace_config(cfg)
    }

    /// Sets the directional light softness radius in `[0.0, 0.5]` (`M5.3b`).
    pub fn set_path_trace_light_softness(&mut self, softness: f32) -> Result<bool, ViewerError> {
        let mut cfg = self.path_trace_config;
        cfg.light_softness = softness;
        self.set_path_trace_config(cfg)
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
            self.opacity = rep.opacity;
            self.transfer_function = rep
                .transfer_function
                .clone()
                .unwrap_or_else(|| vec![[0.0, 0.0], [1.0, 1.0]]);
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
    last_reload_poll: MonotonicInstant,
    last_playback_tick: MonotonicInstant,
    clip_state: InteractiveClipState,
    clip_base_hull: Option<Arc<SurfaceMesh>>,
    clip_base_tet_view: Option<Arc<TetView>>,
    clip_showing_hull: bool,
    clip_generation: u64,
    clip_tx: mpsc::Sender<CompletedClipJob>,
    clip_rx: mpsc::Receiver<CompletedClipJob>,
    active_exact_clip_surface: Option<Arc<SurfaceMesh>>,
    file_store: InMemoryFileStore,
    url_input: String,
    last_load_error: Option<String>,
    last_shared_link: Option<String>,
    url_loading_in_progress: bool,
    async_drop_tx: mpsc::Sender<Result<Vec<DroppedFilePayload>, String>>,
    async_drop_rx: mpsc::Receiver<Result<Vec<DroppedFilePayload>, String>>,
    async_url_tx: mpsc::Sender<Result<FetchedUrlBundle, String>>,
    async_url_rx: mpsc::Receiver<Result<FetchedUrlBundle, String>>,
    active_tf_point: Option<usize>,
    blender_worker_addr: String,
    blender_live_link_enabled: bool,
    last_pushed_camera: Option<ViewportCamera>,
    last_worker_status: Option<String>,
}

impl std::fmt::Debug for ViewerSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViewerSession")
            .field("doc", &self.doc)
            .field("is_playing", &self.is_playing)
            .field("clip_state", &self.clip_state)
            .field("url_input", &self.url_input)
            .field("last_load_error", &self.last_load_error)
            .field("last_shared_link", &self.last_shared_link)
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
        viewport.set_ssao(doc.ssao());
        viewport.set_path_trace_config(doc.path_trace_config());
        viewport.set_path_tracing_enabled(doc.path_tracing_enabled());

        let (clip_base_hull, clip_base_tet_view, clip_state) = match doc.resolve_clip_volume()? {
            Some((hull, tet_view, state)) => (Some(hull), Some(tet_view), state),
            None => (None, None, InteractiveClipState::new([0.0, 0.0, 0.0], 1.0)),
        };
        let clip_showing_hull = !clip_state.enabled;
        let (clip_tx, clip_rx) = mpsc::channel();
        let (async_drop_tx, async_drop_rx) = mpsc::channel();
        let (async_url_tx, async_url_rx) = mpsc::channel();

        let now = MonotonicInstant::now();
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
            file_store: InMemoryFileStore::new(),
            url_input: String::new(),
            last_load_error: None,
            last_shared_link: None,
            url_loading_in_progress: false,
            async_drop_tx,
            async_drop_rx,
            async_url_tx,
            async_url_rx,
            active_tf_point: None,
            blender_worker_addr: String::new(),
            blender_live_link_enabled: false,
            last_pushed_camera: None,
            last_worker_status: None,
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

    /// Returns the most recent drag-and-drop or URL loading error message, if any (`M3.1`).
    #[must_use]
    pub fn last_load_error(&self) -> Option<&str> {
        self.last_load_error.as_deref()
    }

    /// Clears the active drag-and-drop or URL loading error message (`M3.1`).
    pub fn clear_load_error(&mut self) {
        self.last_load_error = None;
    }

    /// Returns the current text in the viewer's URL input box (`M3.1`).
    #[must_use]
    pub fn url_input(&self) -> &str {
        &self.url_input
    }

    /// Sets the text in the viewer's URL input box (`M3.1`).
    pub fn set_url_input(&mut self, url: impl Into<String>) {
        self.url_input = url.into();
    }

    /// Returns a reference to the session's [`InMemoryFileStore`] holding dropped companion files (`M3.1`).
    #[must_use]
    pub const fn file_store(&self) -> &InMemoryFileStore {
        &self.file_store
    }

    /// Returns the most recently generated shareable link copied by the user, if any (`M3.2`).
    #[must_use]
    pub fn last_shared_link(&self) -> Option<&str> {
        self.last_shared_link.as_deref()
    }

    /// Exports a [`Scene`] snapshot reflecting the session's current live camera orientation,
    /// active time step, field selection, colormap, logarithmic scaling, and annotation toggles (`M3.2`).
    #[must_use]
    pub fn export_scene_with_live_view(&self) -> Scene {
        let mut scene = self.doc.scene().clone();
        let cam = self.viewport.camera();
        scene.view.camera.azimuth = cam.azimuth_deg;
        scene.view.camera.elevation = cam.elevation_deg;
        scene.view.camera.roll = cam.roll_deg;
        scene.view.camera.zoom = cam.zoom;
        scene.view.camera.view_angle = cam.view_angle_deg;
        scene.view.camera.position = None;
        if self.doc.num_time_steps() > 1 {
            scene
                .output
                .get_or_insert_with(|| Output::new("out.png"))
                .time = TimeSelector::step(self.doc.current_time_step());
        }
        scene
    }

    /// Builds a shareable web-viewer link (`M3.2`) rooted at `base_viewer_url`.
    ///
    /// - When the active document was loaded from a remote URL (`doc.source_url().is_some()`),
    ///   produces a compact query-string link (`?scene=<url>&az=...&el=...&zoom=...&color_by=...`)
    ///   referencing that URL together with the live camera and styling overrides.
    /// - Otherwise (for example, when files were dropped into the viewer or opened from local bytes),
    ///   delegates to [`Self::build_inline_bundle_shareable_link`] so the resulting link is
    ///   self-contained (`?bundle_b64=<zlib+base64url>`) and carries both the scene and its mesh data.
    pub fn build_shareable_link(&self, base_viewer_url: &str) -> Result<String, ViewerError> {
        if let Some(source_url) = self.doc.source_url() {
            let cam = self.viewport.camera();
            let color_by = self
                .doc
                .scene()
                .view
                .show
                .first()
                .map(|r| r.color_by.clone().unwrap_or_else(|| "solid".to_string()));
            let spec = ShareableLinkSpec {
                primary_url: Some(source_url.to_string()),
                inline_scene: None,
                inline_bundle: None,
                data_urls: std::collections::BTreeMap::new(),
                view_overrides: ShareableViewOverrides {
                    azimuth_deg: Some(cam.azimuth_deg),
                    elevation_deg: Some(cam.elevation_deg),
                    roll_deg: Some(cam.roll_deg),
                    zoom: Some(cam.zoom),
                    step: Some(self.doc.current_time_step()),
                    color_by,
                    colormap: Some(self.doc.colormap()),
                    log_scale: Some(self.doc.use_log_scale()),
                    colorbar: Some(self.doc.colorbar_enabled()),
                    axes: Some(self.doc.axes_enabled()),
                    opacity: Some(self.doc.opacity()),
                },
            };
            spec.to_viewer_url(base_viewer_url)
        } else {
            self.build_inline_bundle_shareable_link(base_viewer_url)
        }
    }

    /// Builds a self-contained shareable web-viewer link (`?bundle_b64=<zlib+base64url>`) that
    /// packages the live [`Scene`] JSON together with all companion mesh files (`M3.2`).
    pub fn build_inline_bundle_shareable_link(
        &self,
        base_viewer_url: &str,
    ) -> Result<String, ViewerError> {
        let scene = self.export_scene_with_live_view();
        let scene_bytes = scene.to_json_string()?.into_bytes();
        let mut companion_files = self.file_store.to_btree_map();
        if self.doc.is_raw_mesh_path
            && let Some(bytes) = self.doc.primary_bytes()
        {
            companion_files.insert(self.doc.file_label().to_string(), bytes.to_vec());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            for source in scene.sources.values() {
                if !companion_files.contains_key(&source.file) {
                    let source_path = self.doc.base_dir.join(&source.file);
                    if let Ok(bytes) = fs::read(&source_path) {
                        companion_files.insert(source.file.clone(), bytes.clone());
                        if source.file.to_ascii_lowercase().ends_with(".pvd")
                            && let Ok(entries) = tetraglass_io::parse_pvd_entries(&bytes)
                        {
                            let pvd_dir =
                                source_path.parent().unwrap_or(self.doc.base_dir.as_path());
                            for entry in entries {
                                if !companion_files.contains_key(&entry.file)
                                    && let Ok(step_bytes) = fs::read(pvd_dir.join(&entry.file))
                                {
                                    companion_files.insert(entry.file, step_bytes);
                                }
                            }
                        }
                    }
                }
            }
        }
        let label = if self
            .doc
            .file_label()
            .to_ascii_lowercase()
            .ends_with(".json")
        {
            self.doc.file_label().to_string()
        } else {
            format!("{}.scene.json", self.doc.file_label())
        };
        let spec = ShareableLinkSpec {
            primary_url: None,
            inline_scene: None,
            inline_bundle: Some(InlineShareableBundle {
                label,
                primary_bytes: scene_bytes,
                companion_files,
            }),
            data_urls: std::collections::BTreeMap::new(),
            view_overrides: ShareableViewOverrides::default(),
        };
        spec.to_viewer_url(base_viewer_url)
    }

    /// Replaces the active [`ViewerDocument`] and uploads its surfaces, camera, background, and
    /// annotations to the GPU [`ViewportState`] (`M3.1`).
    pub fn replace_document(
        &mut self,
        mut doc: ViewerDocument,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
    ) -> Result<(), ViewerError> {
        self.is_playing = false;
        self.last_load_error = None;
        self.last_shared_link = None;
        self.clip_generation = self.clip_generation.wrapping_add(1);
        self.active_exact_clip_surface = None;
        self.viewport.clear_pick();
        self.viewport.set_clip_plane(queue, None);

        let (clip_base_hull, clip_base_tet_view, clip_state) = match doc.resolve_clip_volume()? {
            Some((hull, tet_view, state)) => (Some(hull), Some(tet_view), state),
            None => (None, None, InteractiveClipState::new([0.0, 0.0, 0.0], 1.0)),
        };
        self.clip_base_hull = clip_base_hull;
        self.clip_base_tet_view = clip_base_tet_view;
        self.clip_state = clip_state;
        self.clip_showing_hull = !self.clip_state.enabled;
        self.doc = doc;

        self.sync_viewport_after_doc_change(device, queue, true, true)
    }

    /// Ingests one or more dropped files (`M3.1`), storing all files in the session's
    /// [`InMemoryFileStore`] and loading the primary scene (`.json`) or mesh (`.pvd`, `.vtu`, `.msh`).
    ///
    /// When a user drops a `.json` scene together with its `.vtu`/`.msh` source files (or a `.pvd`
    /// together with its step `.vtu` files), [`InMemoryFileStore::select_primary_dropped_file`]
    /// automatically selects the `.json` or `.pvd` as the entry point while resolving the companion
    /// files from the store.
    pub fn ingest_dropped_files(
        &mut self,
        files: impl IntoIterator<Item = DroppedFilePayload>,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
    ) -> Result<bool, ViewerError> {
        let collected: Vec<DroppedFilePayload> = files.into_iter().collect();
        let Some(primary_idx) = InMemoryFileStore::select_primary_dropped_file(&collected) else {
            return Ok(false);
        };
        let primary = collected[primary_idx].clone();
        self.file_store.insert_all(collected);

        match ViewerDocument::from_file_bytes_with_store(
            primary.name,
            primary.bytes,
            &self.file_store,
        ) {
            Ok(doc) => {
                self.replace_document(doc, device, queue)?;
                Ok(true)
            }
            Err(err) => {
                self.last_load_error = Some(err.to_string());
                Err(err)
            }
        }
    }

    /// Fetches `url` (and any referenced scene sources or `.pvd` step files) using `fetcher` and
    /// replaces the active document (`M3.1`).
    pub fn load_from_url_with_fetcher<F>(
        &mut self,
        url: &str,
        fetcher: F,
        device: &egui_wgpu::wgpu::Device,
        queue: &egui_wgpu::wgpu::Queue,
    ) -> Result<(), ViewerError>
    where
        F: FnMut(&str) -> Result<Vec<u8>, String>,
    {
        let trimmed = url.trim();
        self.url_input = trimmed.to_string();
        match fetch_url_bundle_with(trimmed, fetcher) {
            Ok(bundle) => {
                self.file_store = bundle.companion_files.clone();
                match ViewerDocument::from_fetched_bundle(bundle) {
                    Ok(doc) => {
                        self.replace_document(doc, device, queue)?;
                        Ok(())
                    }
                    Err(err) => {
                        self.last_load_error = Some(err.to_string());
                        Err(err)
                    }
                }
            }
            Err(err) => {
                self.last_load_error = Some(err.to_string());
                Err(err)
            }
        }
    }

    /// Triggers loading a scene or mesh from `url` (`M3.1`).
    ///
    /// - On `wasm32-unknown-unknown`, spawns an asynchronous browser `fetch` task that resolves
    ///   primary and companion URLs and swaps the loaded document in on completion.
    /// - On native targets, resolves local filesystem paths (or `file://` URLs) and queues the
    ///   bundle for the next UI frame.
    pub fn request_load_url(&mut self, url: &str, ctx: Option<egui::Context>) {
        let trimmed = url.trim().to_string();
        if trimmed.is_empty() {
            return;
        }
        self.url_input.clone_from(&trimmed);
        self.last_load_error = None;
        self.url_loading_in_progress = true;

        #[cfg(target_arch = "wasm32")]
        {
            let tx = self.async_url_tx.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let res = crate::web::fetch_url_bundle_async(&trimmed).await;
                let _ = tx.send(res);
                if let Some(c) = ctx {
                    c.request_repaint();
                }
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let res = fetch_url_bundle_with(&trimmed, |target| {
                let path_str = target.strip_prefix("file://").unwrap_or(target);
                fs::read(path_str).map_err(|e| format!("{path_str}: {e}"))
            })
            .map_err(|e| e.to_string());
            let _ = self.async_url_tx.send(res);
            if let Some(c) = ctx {
                c.request_repaint();
            }
        }
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
        self.viewport.set_ssao(self.doc.ssao());
        self.viewport
            .set_path_trace_config(self.doc.path_trace_config());
        self.viewport
            .set_path_tracing_enabled(self.doc.path_tracing_enabled());
        Ok(())
    }

    /// Returns the active [`ViewportSsaoConfig`] on the session (`M5.2`).
    #[must_use]
    pub fn ssao(&self) -> ViewportSsaoConfig {
        self.viewport.ssao()
    }

    /// Updates the [`ViewportSsaoConfig`] on both the document and the GPU viewport (`M5.2`).
    pub fn set_ssao(&mut self, ssao: ViewportSsaoConfig) {
        let _ = self.doc.set_ssao(ssao);
        self.viewport.set_ssao(self.doc.ssao());
    }

    /// Returns whether progressive path tracing is enabled on the session (`M5.3b`).
    #[must_use]
    pub fn path_tracing_enabled(&self) -> bool {
        self.viewport.path_tracing_enabled()
    }

    /// Enables or disables progressive path tracing on both the document and the GPU viewport (`M5.3b`).
    pub fn set_path_tracing_enabled(&mut self, enabled: bool) {
        let _ = self.doc.set_path_tracing_enabled(enabled);
        self.viewport
            .set_path_tracing_enabled(self.doc.path_tracing_enabled());
    }

    /// Returns the active [`ViewportPathTraceConfig`] on the session (`M5.3b`).
    #[must_use]
    pub fn path_trace_config(&self) -> ViewportPathTraceConfig {
        self.viewport.path_trace_config()
    }

    /// Updates the [`ViewportPathTraceConfig`] on both the document and the GPU viewport (`M5.3b`).
    pub fn set_path_trace_config(
        &mut self,
        config: ViewportPathTraceConfig,
    ) -> Result<(), ViewerError> {
        let _ = self.doc.set_path_trace_config(config)?;
        self.viewport
            .set_path_trace_config(self.doc.path_trace_config());
        Ok(())
    }

    /// Returns the number of progressively accumulated samples per pixel (`M5.3b`).
    #[must_use]
    pub fn path_trace_accumulated_samples(&self) -> u32 {
        self.viewport.path_trace_accumulated_samples()
    }

    /// Returns `true` when the progressive path tracer has reached its target sample count (`M5.3b`).
    #[must_use]
    pub fn is_path_trace_converged(&self) -> bool {
        self.viewport.is_path_trace_converged()
    }

    /// Returns the configured persistent Blender worker `host:port` address (`M5.4`).
    #[must_use]
    pub fn blender_worker_addr(&self) -> &str {
        &self.blender_worker_addr
    }

    /// Sets the persistent Blender worker `host:port` address (`M5.4`).
    pub fn set_blender_worker_addr(&mut self, addr: impl Into<String>) {
        self.blender_worker_addr = addr.into();
    }

    /// Returns whether automatic camera live-link syncing to the Blender worker is enabled (`M5.4`).
    #[must_use]
    pub const fn blender_live_link_enabled(&self) -> bool {
        self.blender_live_link_enabled
    }

    /// Enables or disables automatic camera live-link syncing to the Blender worker (`M5.4`).
    pub fn set_blender_live_link_enabled(&mut self, enabled: bool) {
        self.blender_live_link_enabled = enabled;
        if !enabled {
            self.last_pushed_camera = None;
        }
    }

    /// Builds the line-delimited JSON `"update_camera"` payload for the live viewport camera and active step (`M5.4`).
    #[must_use]
    pub fn build_live_worker_camera_payload(&self) -> serde_json::Value {
        let cam = self.viewport.camera();
        let eye = cam.eye_position();
        let (_, up, _) = cam.basis();
        let (clip_start, clip_end) = cam.near_far_f64();
        let half_fov = 0.5 * cam.view_angle_deg.clamp(1.0, 170.0).to_radians();
        let ortho_scale = 2.0 * cam.effective_distance() * half_fov.tan();
        let dataset_center = self
            .doc
            .evaluated()
            .camera_bounds
            .map_or([0.0, 0.0, 0.0], |b| b.center());
        let pos = [f64::from(eye[0]), f64::from(eye[1]), f64::from(eye[2])];
        let focal = cam.focal_point;
        serde_json::json!({
            "command": "update_camera",
            "camera": {
                "position": pos,
                "world_position": [
                    pos[0] + dataset_center[0],
                    pos[1] + dataset_center[1],
                    pos[2] + dataset_center[2]
                ],
                "focal_point": focal,
                "world_focal_point": [
                    focal[0] + dataset_center[0],
                    focal[1] + dataset_center[1],
                    focal[2] + dataset_center[2]
                ],
                "up": [f64::from(up[0]), f64::from(up[1]), f64::from(up[2])],
                "projection": self.doc.scene().view.camera.projection,
                "view_angle_deg": cam.view_angle_deg,
                "ortho_scale": ortho_scale,
                "clip_start": clip_start,
                "clip_end": clip_end
            },
            "step": self.doc.current_time_step()
        })
    }

    /// Pushes the live viewport camera and active step to a persistent Blender worker (`blender/render.py --serve <addr>`) (`M5.4`).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn push_live_camera_to_worker(
        &mut self,
        worker_addr: &str,
    ) -> Result<serde_json::Value, ViewerError> {
        use std::io::{BufRead, BufReader, Write};
        use std::net::{TcpStream, ToSocketAddrs};

        let trimmed = worker_addr.trim();
        if trimmed.is_empty() {
            return Err(ViewerError::Validation(
                "Blender worker address is empty (expected `host:port` or `ipc:<dir>`)".to_string(),
            ));
        }
        let payload = self.build_live_worker_camera_payload();
        let mut line =
            serde_json::to_string(&payload).map_err(|e| ViewerError::Validation(e.to_string()))?;
        line.push('\n');

        let timeout = Duration::from_secs(5);
        let resp_line = if let Some(ipc_path) = trimmed.strip_prefix("ipc:") {
            let ipc_dir = Path::new(ipc_path);
            fs::create_dir_all(ipc_dir)?;
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let req_id = format!("viewer_{}_{nanos}", std::process::id());
            let tmp_req = ipc_dir.join(format!(".req_{req_id}.json.tmp"));
            let final_req = ipc_dir.join(format!("req_{req_id}.json"));
            let final_resp = ipc_dir.join(format!("resp_{req_id}.json"));
            fs::write(&tmp_req, line.as_bytes())?;
            fs::rename(&tmp_req, &final_req)?;
            let start = MonotonicInstant::now();
            loop {
                if final_resp.is_file() {
                    let text = fs::read_to_string(&final_resp)?;
                    let _ = fs::remove_file(&final_resp);
                    break text;
                }
                if MonotonicInstant::now().duration_since(start) >= timeout {
                    let _ = fs::remove_file(&final_req);
                    return Err(ViewerError::Validation(format!(
                        "timed out waiting for Blender worker response in `{}`",
                        ipc_dir.display()
                    )));
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        } else {
            let sock_addr = trimmed
                .to_socket_addrs()
                .map_err(|e| {
                    ViewerError::Validation(format!(
                        "invalid Blender worker address `{trimmed}`: {e}"
                    ))
                })?
                .next()
                .ok_or_else(|| {
                    ViewerError::Validation(format!(
                        "could not resolve Blender worker address `{trimmed}`"
                    ))
                })?;

            let mut stream = TcpStream::connect_timeout(&sock_addr, timeout)?;
            let _ = stream.set_read_timeout(Some(timeout));
            let _ = stream.set_write_timeout(Some(timeout));

            stream.write_all(line.as_bytes())?;
            stream.flush()?;

            let mut reader = BufReader::new(stream);
            let mut buf = String::new();
            let n = reader.read_line(&mut buf)?;
            if n == 0 {
                return Err(ViewerError::Validation(
                    "Blender worker closed connection without replying".to_string(),
                ));
            }
            buf
        };
        let resp: serde_json::Value = serde_json::from_str(resp_line.trim())
            .map_err(|e| ViewerError::Validation(e.to_string()))?;
        if resp.get("status").and_then(|v| v.as_str()) != Some("ok") {
            let err_msg = resp
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown Blender worker error");
            return Err(ViewerError::Validation(err_msg.to_string()));
        }
        self.last_pushed_camera = Some(self.viewport.camera());
        self.last_worker_status = Some(format!("Synced camera to {trimmed}"));
        Ok(resp)
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
        let now = MonotonicInstant::now();
        if let Err(err) = self.poll_clip_worker(device, queue) {
            log::error!("Interactive clip worker failed: {err}");
        }
        if self.clip_state.worker_busy || self.url_loading_in_progress {
            ui.ctx().request_repaint_after(Duration::from_millis(16));
        }

        // Handle newly dropped files from egui's raw input (`M3.1`).
        let dropped_handles = ui.ctx().input(|i| i.raw.dropped_files.clone());
        if !dropped_handles.is_empty() {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let mut payloads = Vec::new();
                let mut read_err = None;
                for handle in dropped_handles {
                    let name = handle.path().to_string_lossy().into_owned();
                    match handle.bytes() {
                        Ok(bytes) => payloads.push(DroppedFilePayload::new(name, bytes)),
                        Err(err) => {
                            read_err = Some(format!("failed to read dropped file `{name}`: {err}"));
                            break;
                        }
                    }
                }
                if let Some(err) = read_err {
                    let _ = self.async_drop_tx.send(Err(err));
                } else {
                    let _ = self.async_drop_tx.send(Ok(payloads));
                }
            }

            #[cfg(target_arch = "wasm32")]
            {
                let tx = self.async_drop_tx.clone();
                let ctx = ui.ctx().clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let mut payloads = Vec::new();
                    for handle in dropped_handles {
                        let name = handle.path().to_string_lossy().into_owned();
                        match handle.bytes_async().await {
                            Ok(bytes) => payloads.push(DroppedFilePayload::new(name, bytes)),
                            Err(err) => {
                                let _ = tx.send(Err(format!("failed to read `{name}`: {err}")));
                                ctx.request_repaint();
                                return;
                            }
                        }
                    }
                    let _ = tx.send(Ok(payloads));
                    ctx.request_repaint();
                });
            }
        }

        // Poll async dropped-file and URL-fetch channels.
        while let Ok(drop_res) = self.async_drop_rx.try_recv() {
            match drop_res {
                Ok(payloads) => {
                    if let Err(err) = self.ingest_dropped_files(payloads, device, queue) {
                        log::error!("Failed to ingest dropped files: {err}");
                    }
                }
                Err(err) => {
                    self.last_load_error = Some(err);
                }
            }
        }
        while let Ok(url_res) = self.async_url_rx.try_recv() {
            self.url_loading_in_progress = false;
            match url_res {
                Ok(bundle) => {
                    self.file_store = bundle.companion_files.clone();
                    match ViewerDocument::from_fetched_bundle(bundle) {
                        Ok(doc) => {
                            if let Err(err) = self.replace_document(doc, device, queue) {
                                self.last_load_error = Some(err.to_string());
                            }
                        }
                        Err(err) => {
                            self.last_load_error = Some(err.to_string());
                        }
                    }
                }
                Err(err) => {
                    self.last_load_error = Some(err);
                }
            }
        }

        if self.doc.scene_path().is_some() {
            if now.duration_since(self.last_reload_poll) >= DEFAULT_HOT_RELOAD_POLL_INTERVAL {
                self.last_reload_poll = now;
                if let Err(err) = self.poll_hot_reload(device, queue) {
                    log::error!("Hot reload failed: {err}");
                }
            }
            ui.ctx()
                .request_repaint_after(DEFAULT_HOT_RELOAD_POLL_INTERVAL);
        }

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
        let mut opacity_changed_to: Option<f64> = None;
        let mut ssao_changed_to: Option<ViewportSsaoConfig> = None;
        let mut path_tracing_enabled_changed_to: Option<bool> = None;
        let mut path_trace_config_changed_to: Option<ViewportPathTraceConfig> = None;
        let mut transfer_function_changed_to: Option<Vec<[f64; 2]>> = None;
        let mut trigger_reload = false;
        let mut trigger_load_url = false;
        let mut trigger_reset_camera = false;
        let mut trigger_copy_shareable_link = false;
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

                if self.doc.scene_path().is_some() {
                    ui.horizontal(|ui| {
                        if ui.button("Reload file").clicked() {
                            trigger_reload = true;
                        }
                        if self.doc.reload_count() > 0 {
                            ui.small(format!("(reloaded {}x)", self.doc.reload_count()));
                        }
                    });
                }

                // --- Drag-and-drop & URL loader section (M3.1) ---
                ui.add_space(4.0);
                ui.label("Open URL (.vtu / .msh / .pvd / .json):");
                ui.horizontal(|ui| {
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.url_input)
                            .desired_width(185.0)
                            .hint_text("https://.../scene.json"),
                    );
                    let enter_pressed =
                        resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    let load_btn =
                        ui.add_enabled(!self.url_loading_in_progress, egui::Button::new("Load"));
                    if load_btn.clicked() || enter_pressed {
                        trigger_load_url = true;
                    }
                });
                if self.url_loading_in_progress {
                    ui.small("Fetching URL...");
                } else {
                    ui.small(
                        "Tip: drag & drop .vtu, .msh, .pvd, or .json files onto the viewport.",
                    );
                }

                if let Some(err_msg) = self.doc.last_reload_error() {
                    ui.add_space(4.0);
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 50, 47),
                        format!("Hot-reload error: {err_msg}"),
                    );
                }

                if let Some(err_msg) = self.last_load_error() {
                    ui.add_space(4.0);
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 50, 47),
                        format!("Load error: {err_msg}"),
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
                            self.last_playback_tick = MonotonicInstant::now();
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

                let mut opacity_val = self.doc.opacity();
                if ui
                    .add(
                        egui::Slider::new(&mut opacity_val, 0.05..=1.0)
                            .text("Opacity")
                            .show_value(true),
                    )
                    .changed()
                {
                    opacity_changed_to = Some(opacity_val);
                }

                let mut ssao_cfg = self.doc.ssao();
                if ui
                    .checkbox(&mut ssao_cfg.enabled, "Ambient occlusion (SSAO)")
                    .changed()
                {
                    ssao_changed_to = Some(ssao_cfg);
                }
                if ssao_cfg.enabled {
                    if ui
                        .add(
                            egui::Slider::new(&mut ssao_cfg.strength, 0.05..=2.0)
                                .text("AO strength")
                                .show_value(true),
                        )
                        .changed()
                    {
                        ssao_changed_to = Some(ssao_cfg);
                    }
                    if ui
                        .add(
                            egui::Slider::new(&mut ssao_cfg.radius, 0.02..=0.5)
                                .text("AO radius")
                                .show_value(true),
                        )
                        .changed()
                    {
                        ssao_changed_to = Some(ssao_cfg);
                    }
                }

                let mut pt_enabled = self.doc.path_tracing_enabled();
                if ui
                    .checkbox(&mut pt_enabled, "Progressive path tracer")
                    .changed()
                {
                    path_tracing_enabled_changed_to = Some(pt_enabled);
                }
                if pt_enabled {
                    let mut pt_cfg = self.doc.path_trace_config();
                    if ui
                        .add(
                            egui::Slider::new(&mut pt_cfg.samples_per_pixel, 1..=256)
                                .text("Target samples")
                                .show_value(true),
                        )
                        .changed()
                    {
                        path_trace_config_changed_to = Some(pt_cfg);
                    }
                    if ui
                        .add(
                            egui::Slider::new(&mut pt_cfg.max_bounces, 1..=8)
                                .text("Max bounces")
                                .show_value(true),
                        )
                        .changed()
                    {
                        path_trace_config_changed_to = Some(pt_cfg);
                    }
                    if ui
                        .add(
                            egui::Slider::new(&mut pt_cfg.light_softness, 0.0..=0.35)
                                .text("Light softness")
                                .show_value(true),
                        )
                        .changed()
                    {
                        path_trace_config_changed_to = Some(pt_cfg);
                    }
                    let accum = self.viewport.path_trace_accumulated_samples();
                    let target = pt_cfg.samples_per_pixel;
                    let status = if self.viewport.is_path_trace_converged() {
                        "converged"
                    } else {
                        "converging..."
                    };
                    ui.small(format!("Path tracer: {accum} / {target} spp ({status})"));
                }

                if let Some([lo, hi]) = self.viewport.effective_range() {
                    ui.label(format!("Range: [{lo:.4}, {hi:.4}]"));
                }

                // --- Transfer function editor (M4.4b) ---
                if self.doc.is_volume_representation() {
                    ui.separator();
                    ui.label("Transfer function (opacity curve):");
                    ui.horizontal_wrapped(|ui| {
                        ui.small("Presets:");
                        if ui.small_button("Linear").clicked() {
                            transfer_function_changed_to = Some(vec![[0.0, 0.0], [1.0, 1.0]]);
                        }
                        if ui.small_button("Soft").clicked() {
                            transfer_function_changed_to =
                                Some(vec![[0.0, 0.0], [0.35, 0.12], [0.7, 0.45], [1.0, 0.9]]);
                        }
                        if ui.small_button("High peak").clicked() {
                            transfer_function_changed_to =
                                Some(vec![[0.0, 0.0], [0.55, 0.05], [0.8, 0.45], [1.0, 1.0]]);
                        }
                        if ui.small_button("Mid band").clicked() {
                            transfer_function_changed_to = Some(vec![
                                [0.0, 0.0],
                                [0.25, 0.05],
                                [0.5, 0.85],
                                [0.75, 0.05],
                                [1.0, 0.0],
                            ]);
                        }
                    });

                    let mut pts = self.doc.transfer_function().to_vec();
                    let (rect, resp) = ui.allocate_exact_size(
                        egui::vec2(230.0, 88.0),
                        egui::Sense::click_and_drag(),
                    );
                    let pad = 6.0_f32;
                    let plot = rect.shrink(pad);
                    let to_screen = |t: f64, a: f64| -> egui::Pos2 {
                        egui::pos2(
                            plot.left() + (t.clamp(0.0, 1.0) as f32) * plot.width(),
                            plot.bottom() - (a.clamp(0.0, 1.0) as f32) * plot.height(),
                        )
                    };
                    let from_screen = |pos: egui::Pos2| -> [f64; 2] {
                        let t = ((pos.x - plot.left()) / plot.width().max(1.0)).clamp(0.0, 1.0);
                        let a = ((plot.bottom() - pos.y) / plot.height().max(1.0)).clamp(0.0, 1.0);
                        [f64::from(t), f64::from(a)]
                    };

                    if resp.drag_started()
                        && let Some(pos) = resp.interact_pointer_pos()
                    {
                        let mut hit = None;
                        let mut best_dist = 10.0_f32;
                        for (idx, &[t, a]) in pts.iter().enumerate() {
                            let d = to_screen(t, a).distance(pos);
                            if d <= best_dist {
                                best_dist = d;
                                hit = Some(idx);
                            }
                        }
                        if let Some(idx) = hit {
                            self.active_tf_point = Some(idx);
                        } else {
                            let [t, a] = from_screen(pos);
                            let insert_idx = pts
                                .iter()
                                .position(|p| p[0] > t)
                                .unwrap_or(pts.len().saturating_sub(1))
                                .clamp(1, pts.len().saturating_sub(1));
                            pts.insert(insert_idx, [t, a]);
                            self.active_tf_point = Some(insert_idx);
                            transfer_function_changed_to = Some(pts.clone());
                        }
                    }

                    if resp.dragged()
                        && let (Some(idx), Some(pos)) =
                            (self.active_tf_point, resp.interact_pointer_pos())
                        && idx < pts.len()
                    {
                        let [raw_t, raw_a] = from_screen(pos);
                        let last = pts.len() - 1;
                        let new_t = if idx == 0 {
                            0.0
                        } else if idx == last {
                            1.0
                        } else {
                            let lo = (pts[idx - 1][0] + 0.01).min(1.0);
                            let hi = (pts[idx + 1][0] - 0.01).max(lo);
                            raw_t.clamp(lo, hi)
                        };
                        pts[idx] = [new_t, raw_a];
                        transfer_function_changed_to = Some(pts.clone());
                    }

                    if resp.drag_stopped() {
                        self.active_tf_point = None;
                    }

                    if resp.secondary_clicked()
                        && pts.len() > 2
                        && let Some(pos) = resp.interact_pointer_pos()
                    {
                        let mut hit = None;
                        let mut best_dist = 10.0_f32;
                        for (idx, &[t, a]) in pts.iter().enumerate() {
                            if idx > 0 && idx + 1 < pts.len() {
                                let d = to_screen(t, a).distance(pos);
                                if d <= best_dist {
                                    best_dist = d;
                                    hit = Some(idx);
                                }
                            }
                        }
                        if let Some(idx) = hit {
                            pts.remove(idx);
                            self.active_tf_point = None;
                            transfer_function_changed_to = Some(pts.clone());
                        }
                    }

                    let painter = ui.painter_at(rect);
                    painter.rect_filled(rect, 3.0, egui::Color32::from_rgb(24, 28, 36));
                    for step in [0.25_f32, 0.5, 0.75] {
                        let y = plot.bottom() - step * plot.height();
                        painter.line_segment(
                            [egui::pos2(plot.left(), y), egui::pos2(plot.right(), y)],
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 52, 66)),
                        );
                    }
                    let screen_pts: Vec<egui::Pos2> =
                        pts.iter().map(|&[t, a]| to_screen(t, a)).collect();
                    for seg in screen_pts.windows(2) {
                        painter.line_segment(
                            [seg[0], seg[1]],
                            egui::Stroke::new(2.0, egui::Color32::from_rgb(90, 200, 250)),
                        );
                    }
                    for (idx, &p) in screen_pts.iter().enumerate() {
                        let fill = if self.active_tf_point == Some(idx) {
                            egui::Color32::from_rgb(255, 210, 90)
                        } else {
                            egui::Color32::WHITE
                        };
                        painter.circle_filled(p, 4.0, fill);
                        painter.circle_stroke(
                            p,
                            4.0,
                            egui::Stroke::new(1.2, egui::Color32::from_rgb(20, 30, 45)),
                        );
                    }
                    ui.small(
                        "Drag points to edit; click to add; right-click interior point to remove.",
                    );
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
                ui.horizontal(|ui| {
                    if ui.button("Reset camera").clicked() {
                        trigger_reset_camera = true;
                    }
                    if ui
                        .button("Copy shareable link")
                        .on_hover_text(
                            "Copy a web-viewer URL containing the scene, data, and live view state",
                        )
                        .clicked()
                    {
                        trigger_copy_shareable_link = true;
                    }
                });
                if self.last_shared_link.is_some() {
                    ui.small("Copied shareable link to clipboard.");
                }

                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.add_space(4.0);
                    ui.label("Blender worker live link (host:port):");
                    let mut trigger_push_worker = false;
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.blender_worker_addr)
                                .desired_width(155.0)
                                .hint_text("127.0.0.1:55123"),
                        );
                        if ui.button("Push camera").clicked() {
                            trigger_push_worker = true;
                        }
                    });
                    ui.checkbox(
                        &mut self.blender_live_link_enabled,
                        "Live sync camera on orbit",
                    );
                    if let Some(status) = &self.last_worker_status {
                        ui.small(status);
                    }
                    let addr = self.blender_worker_addr.clone();
                    let cam_changed = self.last_pushed_camera != Some(self.viewport.camera());
                    let should_push = trigger_push_worker
                        || (self.blender_live_link_enabled
                            && !addr.trim().is_empty()
                            && cam_changed);
                    if should_push && let Err(err) = self.push_live_camera_to_worker(&addr) {
                        self.last_worker_status = Some(format!("Worker error: {err}"));
                    }
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

        if trigger_load_url {
            let url = self.url_input.clone();
            self.request_load_url(&url, Some(ui.ctx().clone()));
        }

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
        let mut volume_field_reupload = false;
        if let Some(new_field_idx) = field_changed_to
            && let Ok(true) = self.doc.set_field_index(new_field_idx)
        {
            if self.doc.is_volume_representation() {
                volume_field_reupload = true;
            } else {
                style_or_annotations_dirty = true;
            }
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
        if let Some(new_opacity) = opacity_changed_to
            && let Ok(true) = self.doc.set_opacity(new_opacity)
        {
            style_or_annotations_dirty = true;
        }
        if let Some(new_ssao) = ssao_changed_to
            && self.doc.set_ssao(new_ssao)
        {
            self.viewport.set_ssao(self.doc.ssao());
        }
        if let Some(new_pt_enabled) = path_tracing_enabled_changed_to
            && self.doc.set_path_tracing_enabled(new_pt_enabled)
        {
            self.viewport
                .set_path_tracing_enabled(self.doc.path_tracing_enabled());
        }
        if let Some(new_pt_cfg) = path_trace_config_changed_to
            && let Ok(true) = self.doc.set_path_trace_config(new_pt_cfg)
        {
            self.viewport
                .set_path_trace_config(self.doc.path_trace_config());
        }
        if let Some(new_tf) = transfer_function_changed_to
            && let Ok(true) = self.doc.set_transfer_function(new_tf)
        {
            style_or_annotations_dirty = true;
        }
        if volume_field_reupload {
            if let Err(err) = self.sync_viewport_after_doc_change(device, queue, true, false) {
                log::error!("Failed to re-upload volume field: {err}");
            }
        } else if style_or_annotations_dirty
            && let Err(err) = self.sync_viewport_after_doc_change(device, queue, false, false)
        {
            log::error!("Failed to update viewport style: {err}");
        }

        if trigger_reset_camera {
            self.viewport.set_camera(self.doc.camera_from_scene());
        }
        if trigger_copy_shareable_link {
            #[cfg(target_arch = "wasm32")]
            let base_url = web_sys::window()
                .and_then(|w| w.location().href().ok())
                .unwrap_or_else(|| "http://localhost:8080/".to_string());
            #[cfg(not(target_arch = "wasm32"))]
            let base_url = "http://localhost:8080/".to_string();

            match self.build_shareable_link(&base_url) {
                Ok(link) => {
                    ui.ctx().copy_text(link.clone());
                    self.last_shared_link = Some(link);
                }
                Err(err) => {
                    self.last_load_error = Some(err.to_string());
                }
            }
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

                // Draw a drop overlay when the user hovers files over the window (`M3.1`).
                let hovered = ui.ctx().input(|i| i.raw.hovered_files.clone());
                if !hovered.is_empty() {
                    let rect = ui.max_rect();
                    let painter = ui.ctx().layer_painter(egui::LayerId::new(
                        egui::Order::Foreground,
                        ui.scope_id().with("tetraglass_drop_overlay"),
                    ));
                    painter.rect_filled(
                        rect,
                        0.0,
                        egui::Color32::from_rgba_unmultiplied(15, 25, 45, 190),
                    );
                    let mut label =
                        "Drop .vtu, .msh, .pvd, or .json scene files to load\n".to_string();
                    for file in &hovered {
                        if let Some(p) = &file.path {
                            label.push_str(&format!("\n• {}", p.display()));
                        } else if !file.mime.is_empty() {
                            label.push_str(&format!("\n• ({})", file.mime));
                        }
                    }
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        label,
                        egui::FontId::proportional(18.0),
                        egui::Color32::WHITE,
                    );
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

/// Launch options for `run_viewer` and [`ViewerApp`].
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

/// Native and WebGPU `eframe` application wrapping a [`ViewerSession`].
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

    /// Returns a mutable reference to the underlying [`ViewerSession`].
    pub const fn session_mut(&mut self) -> &mut ViewerSession {
        &mut self.session
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

    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

/// Validates and loads the scene at `options.path`, then opens the native `eframe` desktop viewer.
#[cfg(not(target_arch = "wasm32"))]
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
        MeshFormat::VtkLegacy => Ok((format, read_vtk_series(bytes)?)),
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
    build_default_scene_from_series(path, series)
}

fn build_default_scene_for_mesh_with_store(
    path: &Path,
    bytes: &[u8],
    store: &InMemoryFileStore,
) -> Result<(Scene, SceneEvaluator, Vec<f64>), ViewerError> {
    let (_, series) = load_series_from_store(path, bytes.to_vec(), None, store, None)?;
    build_default_scene_from_series(path, series)
}

fn build_default_scene_from_series(
    path: &Path,
    series: TimeSeries<IoError>,
) -> Result<(Scene, SceneEvaluator, Vec<f64>), ViewerError> {
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

fn load_series_from_store(
    path: &Path,
    bytes: Vec<u8>,
    explicit_format: Option<MeshFormat>,
    store: &InMemoryFileStore,
    fallback_dir: Option<&Path>,
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
        MeshFormat::VtkLegacy => Ok((format, read_vtk_series(bytes)?)),
        MeshFormat::Gmsh => Ok((format, read_msh_series(bytes)?)),
        MeshFormat::Pvd => {
            let store_clone = store.clone();
            let pvd_dir = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .map(Path::to_path_buf)
                .or_else(|| fallback_dir.map(Path::to_path_buf));
            let series = read_pvd_series_with_loader(&bytes, move |rel_file| {
                if let Some(step_bytes) = store_clone.get(rel_file) {
                    return Ok(step_bytes.to_vec());
                }
                if let Some(ref base) = pvd_dir {
                    let candidate = resolve_relative_path(base, rel_file);
                    let candidate_str = candidate.to_string_lossy();
                    if let Some(step_bytes) = store_clone.get(&candidate_str) {
                        return Ok(step_bytes.to_vec());
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if candidate.exists() {
                        return Ok(fs::read(candidate)?);
                    }
                }
                Err(IoError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "missing PVD step file `{rel_file}` (drop the step .vtu files alongside the .pvd file)"
                    ),
                )))
            })?;
            Ok((format, series))
        }
    }
}

fn load_and_validate_sources_from_store(
    scene: &Scene,
    store: &InMemoryFileStore,
    fallback_dir: Option<&Path>,
) -> Result<(SceneEvaluator, Vec<f64>), ViewerError> {
    let mut evaluator = SceneEvaluator::new();
    let mut datasets = BTreeMap::new();
    let mut primary_times = vec![0.0];

    let initial_selector = scene.output.as_ref().map_or(TimeSelector::LAST, |o| o.time);

    for (name, source) in &scene.sources {
        let src_bytes = if let Some(b) = store.get(&source.file) {
            b.to_vec()
        } else if let Some(base) = fallback_dir {
            let resolved = resolve_relative_path(base, &source.file);
            let resolved_str = resolved.to_string_lossy();
            if let Some(b) = store.get(&resolved_str) {
                b.to_vec()
            } else {
                fs::read(&resolved)?
            }
        } else {
            return Err(ViewerError::Validation(format!(
                "source `{name}` references `{}` which is not loaded in memory; drop `{}` together with the scene JSON file",
                source.file, source.file
            )));
        };

        let explicit_format = source.format.map(MeshFormat::from);
        let (_, series) = load_series_from_store(
            Path::new(&source.file),
            src_bytes,
            explicit_format,
            store,
            fallback_dir,
        )?;
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

    Ok((evaluator, primary_times))
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
    let is_volume = rep.volume.is_some();
    let (draw_surface, draw_edges) = match rep.style {
        RepresentationStyle::Wireframe => (false, true),
        RepresentationStyle::SurfaceWithEdges => (!is_volume, true),
        RepresentationStyle::Volume => (false, false),
        _ => (!is_volume, false),
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
            volume_grid: rep.volume.clone(),
            transfer_function: rep.transfer_function.clone(),
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
            volume_grid: rep.volume.clone(),
            transfer_function: rep.transfer_function.clone(),
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

/// Re-exported clip-plane, SSAO, path-tracing, and pick result types (`M2.7`, `M2.8`, `M5.2`, `M5.3b`).
pub use tetraglass_egui::{
    ViewportClipKeep, ViewportClipPlane, ViewportPathTraceConfig, ViewportPickProbeValue,
    ViewportPickResult, ViewportSsaoConfig,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewer_document_path_trace_controls_validate_and_update() {
        let mut doc = ViewerDocument::default_demo().expect("default demo document");
        assert!(!doc.path_tracing_enabled());
        assert_eq!(doc.path_trace_config().samples_per_pixel, 32);
        assert_eq!(doc.path_trace_config().max_bounces, 3);

        assert!(doc.set_path_tracing_enabled(true));
        assert!(doc.path_tracing_enabled());
        assert!(!doc.set_path_tracing_enabled(true));

        assert!(doc.set_path_trace_samples(128).expect("set valid samples"));
        assert_eq!(doc.path_trace_config().samples_per_pixel, 128);
        assert!(doc.set_path_trace_samples(0).is_err());
        assert!(doc.set_path_trace_samples(2048).is_err());

        assert!(doc.set_path_trace_bounces(4).expect("set valid bounces"));
        assert_eq!(doc.path_trace_config().max_bounces, 4);
        assert!(doc.set_path_trace_bounces(0).is_err());
        assert!(doc.set_path_trace_bounces(16).is_err());

        assert!(
            doc.set_path_trace_light_softness(0.12)
                .expect("set valid softness")
        );
        assert!((doc.path_trace_config().light_softness - 0.12).abs() < 1e-6);
        assert!(doc.set_path_trace_light_softness(-0.05).is_err());
        assert!(doc.set_path_trace_light_softness(0.9).is_err());
    }

    #[test]
    fn viewer_document_ssao_controls_validate_and_update() {
        let mut doc = ViewerDocument::default_demo().expect("default demo document");
        assert!(!doc.ssao().enabled);
        assert!((doc.ssao().strength - 0.8).abs() < 1e-6);
        assert!((doc.ssao().radius - 0.15).abs() < 1e-6);

        assert!(doc.set_ssao_enabled(true));
        assert!(doc.ssao().enabled);
        assert!(!doc.set_ssao_enabled(true));

        assert!(doc.set_ssao_strength(1.25).expect("set valid strength"));
        assert!((doc.ssao().strength - 1.25).abs() < 1e-6);
        assert!(doc.set_ssao_strength(-0.1).is_err());
        assert!(doc.set_ssao_strength(2.5).is_err());

        assert!(doc.set_ssao_radius(0.25).expect("set valid radius"));
        assert!((doc.ssao().radius - 0.25).abs() < 1e-6);
        assert!(doc.set_ssao_radius(0.0).is_err());
        assert!(doc.set_ssao_radius(1.5).is_err());
    }

    #[test]
    fn volume_document_updates_transfer_function_without_recomputing_filter() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root")
            .to_path_buf();
        let json = r#"{
            "version": 1,
            "sources": {
                "mesh": { "file": "testdata/vtk/hex_linear.vtu" }
            },
            "filters": {
                "vol": {
                    "type": "resample",
                    "input": "mesh",
                    "dimensions": [12, 12, 12]
                }
            },
            "view": {
                "show": [
                    {
                        "input": "vol",
                        "style": "volume",
                        "color_by": "linear",
                        "transfer_function": [[0.0, 0.0], [0.5, 0.25], [1.0, 0.9]]
                    }
                ]
            }
        }"#;
        let scene = Scene::from_json_str(json).expect("parse volume scene");
        let mut doc = ViewerDocument::from_scene(scene, &root).expect("load ViewerDocument");
        assert!(doc.is_volume_representation());
        assert_eq!(
            doc.transfer_function(),
            &[[0.0, 0.0], [0.5, 0.25], [1.0, 0.9]]
        );

        let initial_vol = doc
            .evaluated()
            .representations
            .first()
            .and_then(|r| r.volume.clone())
            .expect("initial volume grid");

        let new_pts = vec![[0.0, 0.0], [0.3, 0.15], [0.7, 0.6], [1.0, 1.0]];
        assert!(
            doc.set_transfer_function(new_pts.clone())
                .expect("update transfer function")
        );
        assert_eq!(doc.transfer_function(), new_pts.as_slice());
        assert_eq!(
            doc.active_surface_style().transfer_function.as_deref(),
            Some(new_pts.as_slice())
        );
        let updated_vol = doc
            .evaluated()
            .representations
            .first()
            .and_then(|r| r.volume.clone())
            .expect("updated volume grid");
        assert!(
            Arc::ptr_eq(&initial_vol, &updated_vol),
            "updating transfer function must reuse cached RegularGrid without re-running resample"
        );
    }
}

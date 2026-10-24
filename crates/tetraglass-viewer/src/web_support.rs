//! Platform-independent support types for the WebGPU browser viewer (`M3.1`).
//!
//! All types and helpers in this module compile on both native and `wasm32-unknown-unknown`
//! targets so that WebGPU status formatting, WebGPU-only `wgpu` configuration (`D-007`),
//! drag-and-drop byte stores, URL resolution, and multi-file scene/series bundle fetching
//! can be unit- and integration-tested on native CI as well as used in the browser.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use tetraglass_io::{
    IoError, MeshFormat, detect_format_from_bytes, detect_format_from_extension, parse_pvd_entries,
};
use tetraglass_scene::Scene;

use crate::ViewerError;

/// Browsers with default WebGPU support cited in diagnostic messages (`docs/design-plan.md`).
pub const SUPPORTED_WEBGPU_BROWSERS: [&str; 4] = [
    "Chrome or Microsoft Edge 113+ (Windows, macOS, ChromeOS)",
    "Safari 26+ (macOS, iOS / iPadOS)",
    "Firefox 141+ on Windows",
    "Firefox 145+ on Apple Silicon macOS",
];

/// Status of WebGPU support in the current browser environment (`M3.1`, `D-007`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebGpuStatus {
    /// WebGPU (`navigator.gpu`) is present in a secure context and a GPU adapter is available.
    Supported,
    /// The page is not running in a secure context (`https://` or `http://localhost`), so the
    /// browser hides `navigator.gpu`.
    NotSecureContext {
        /// Current page origin or URL.
        origin: String,
    },
    /// `navigator.gpu` is undefined or null in this browser.
    MissingNavigatorGpu {
        /// Optional browser `navigator.userAgent` string.
        user_agent: Option<String>,
    },
    /// `navigator.gpu` exists, but requesting a WebGPU adapter or device failed.
    NoAdapterAvailable {
        /// Optional adapter or device initialization error details.
        details: Option<String>,
    },
}

impl WebGpuStatus {
    /// Returns `true` if WebGPU is supported and ready to use.
    #[must_use]
    pub const fn is_supported(&self) -> bool {
        matches!(self, Self::Supported)
    }

    /// Builds a structured, user-facing [`WebGpuErrorMessage`] when WebGPU is unavailable,
    /// or `None` when [`Self::Supported`].
    #[must_use]
    pub fn error_message(&self) -> Option<WebGpuErrorMessage> {
        match self {
            Self::Supported => None,
            Self::NotSecureContext { origin } => Some(WebGpuErrorMessage {
                title: "WebGPU requires a secure context (HTTPS or localhost)".to_string(),
                summary: format!(
                    "Tetraglass requires WebGPU for 3D mesh rendering and does not fall back to WebGL2. \
                     The current page ({origin}) is not loaded over HTTPS or localhost, so the browser \
                     disables navigator.gpu."
                ),
                details: None,
                supported_browsers: &SUPPORTED_WEBGPU_BROWSERS,
                troubleshooting_steps: vec![
                    "Serve the viewer over http://localhost, http://127.0.0.1, or https://."
                        .to_string(),
                    "Avoid opening index.html directly via file:// or plain HTTP on a LAN IP."
                        .to_string(),
                ],
            }),
            Self::MissingNavigatorGpu { user_agent } => Some(WebGpuErrorMessage {
                title: "WebGPU is not available in this browser".to_string(),
                summary: "Tetraglass is built on WebGPU only (no WebGL2 fallback) because its \
                          unstructured-mesh shaders and picking pipeline require storage buffers \
                          and modern GPU features. Your browser does not expose navigator.gpu."
                    .to_string(),
                details: user_agent
                    .as_ref()
                    .filter(|ua| !ua.trim().is_empty())
                    .map(|ua| format!("Detected user-agent: {ua}")),
                supported_browsers: &SUPPORTED_WEBGPU_BROWSERS,
                troubleshooting_steps: vec![
                    "Open this page in a browser with WebGPU enabled (Chrome/Edge 113+, Safari 26+, or Firefox 141+/145+)."
                        .to_string(),
                    "Ensure hardware acceleration is enabled in your browser settings.".to_string(),
                ],
            }),
            Self::NoAdapterAvailable { details } => Some(WebGpuErrorMessage {
                title: "No compatible WebGPU graphics adapter found".to_string(),
                summary: "Your browser exposes the WebGPU API (navigator.gpu), but could not \
                          initialize a compatible WebGPU adapter or device. Tetraglass requires \
                          WebGPU and does not fall back to WebGL2."
                    .to_string(),
                details: details.clone().filter(|d| !d.trim().is_empty()),
                supported_browsers: &SUPPORTED_WEBGPU_BROWSERS,
                troubleshooting_steps: vec![
                    "Verify that hardware acceleration is enabled in your browser settings."
                        .to_string(),
                    "Update your graphics drivers or try Chrome/Edge 113+ or Safari 26+."
                        .to_string(),
                ],
            }),
        }
    }
}

/// User-facing diagnostic message displayed when WebGPU is unavailable (`M3.1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebGpuErrorMessage {
    /// Short heading summarizing why the viewer cannot start.
    pub title: String,
    /// Clear explanation of the requirement (including why WebGL2 is not used).
    pub summary: String,
    /// Optional technical detail (user-agent or adapter error).
    pub details: Option<String>,
    /// Browsers that support WebGPU out of the box.
    pub supported_browsers: &'static [&'static str],
    /// Actionable steps the user can take to resolve the issue.
    pub troubleshooting_steps: Vec<String>,
}

impl WebGpuErrorMessage {
    /// Formats the diagnostic message as readable plain text (for logs, errors, and CLI/tests).
    #[must_use]
    pub fn to_plain_text(&self) -> String {
        let mut out = format!("{}\n\n{}\n", self.title, self.summary);
        if let Some(ref details) = self.details {
            out.push_str(&format!("\nDetails: {details}\n"));
        }
        out.push_str("\nSupported browsers:\n");
        for browser in self.supported_browsers {
            out.push_str(&format!("  - {browser}\n"));
        }
        if !self.troubleshooting_steps.is_empty() {
            out.push_str("\nHow to fix:\n");
            for step in &self.troubleshooting_steps {
                out.push_str(&format!("  - {step}\n"));
            }
        }
        out
    }

    /// Formats the diagnostic message as a self-contained HTML card for injection into the page DOM.
    #[must_use]
    pub fn to_html_fragment(&self) -> String {
        let title = escape_html(&self.title);
        let summary = escape_html(&self.summary);
        let details_html = self.details.as_ref().map_or_else(String::new, |d| {
            format!(
                "<p style=\"margin:12px 0;padding:8px 12px;background:#1e222a;border-radius:6px;\
                 font-family:monospace;font-size:13px;color:#cbd5e1;word-break:break-word;\">{}</p>",
                escape_html(d)
            )
        });
        let mut browsers_html = String::new();
        for b in self.supported_browsers {
            browsers_html.push_str(&format!("<li>{}</li>", escape_html(b)));
        }
        let mut steps_html = String::new();
        for s in &self.troubleshooting_steps {
            steps_html.push_str(&format!("<li>{}</li>", escape_html(s)));
        }

        format!(
            "<div role=\"alert\" data-tetraglass-webgpu-error=\"true\" \
             style=\"max-width:640px;margin:48px auto;padding:24px 28px;background:#161920;\
             color:#f1f5f9;border:1px solid #ef4444;border-radius:10px;\
             font-family:system-ui,-apple-system,sans-serif;line-height:1.5;\">\
             <h2 style=\"margin:0 0 12px 0;color:#f87171;font-size:20px;\">{title}</h2>\
             <p style=\"margin:0 0 12px 0;color:#e2e8f0;\">{summary}</p>\
             {details_html}\
             <h3 style=\"margin:16px 0 6px 0;font-size:15px;color:#94a3b8;\">Supported browsers</h3>\
             <ul style=\"margin:0 0 12px 20px;padding:0;color:#cbd5e1;\">{browsers_html}</ul>\
             <h3 style=\"margin:16px 0 6px 0;font-size:15px;color:#94a3b8;\">How to fix</h3>\
             <ul style=\"margin:0 0 0 20px;padding:0;color:#cbd5e1;\">{steps_html}</ul>\
             </div>"
        )
    }
}

fn escape_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

/// Returns an [`egui_wgpu::WgpuConfiguration`] restricted strictly to `Backends::BROWSER_WEBGPU`
/// (`D-007`: WebGPU only on the web, with no WebGL2 fallback).
#[must_use]
pub fn webgpu_only_wgpu_configuration() -> egui_wgpu::WgpuConfiguration {
    let mut create_new = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    create_new.instance_descriptor.backends = egui_wgpu::wgpu::Backends::BROWSER_WEBGPU;
    egui_wgpu::WgpuConfiguration {
        wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(create_new),
        ..egui_wgpu::WgpuConfiguration::default()
    }
}

/// A file payload delivered via drag-and-drop or in-memory loading (`M3.1`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedFilePayload {
    /// File name or relative path (for example `"tet_linear.vtu"` or `"slice.json"`).
    pub name: String,
    /// Optional filesystem path (populated on desktop drag-and-drop when available).
    pub path: Option<PathBuf>,
    /// Raw file bytes.
    pub bytes: Vec<u8>,
}

impl DroppedFilePayload {
    /// Creates a new [`DroppedFilePayload`] from a file name and byte buffer.
    #[must_use]
    pub fn new(name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            path: None,
            bytes,
        }
    }

    /// Attaches an optional filesystem path to this payload.
    #[must_use]
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }
}

/// In-memory lookup table of companion files (for `.json` scenes and `.pvd` time series)
/// loaded via drag-and-drop or URL fetching (`M3.1`).
///
/// Entries are indexed both by their normalized relative path and by their basename so that
/// scenes referencing `"../testdata/vtk/mixed_linear.vtu"` or `"mixed_linear.vtu"` match a
/// dropped `"mixed_linear.vtu"` file.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InMemoryFileStore {
    files_by_key: BTreeMap<String, Vec<u8>>,
    basenames: BTreeMap<String, Vec<u8>>,
}

impl InMemoryFileStore {
    /// Creates an empty [`InMemoryFileStore`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts `bytes` under `name_or_path` (indexing both normalized path and file basename).
    pub fn insert(&mut self, name_or_path: impl AsRef<str>, bytes: Vec<u8>) {
        let raw = name_or_path.as_ref();
        let norm = normalize_lookup_key(raw);
        let base = basename_of_key(&norm).to_string();
        self.basenames.insert(base, bytes.clone());
        self.files_by_key.insert(norm, bytes);
    }

    /// Inserts all [`DroppedFilePayload`] entries into the store.
    pub fn insert_all(&mut self, files: impl IntoIterator<Item = DroppedFilePayload>) {
        for f in files {
            if let Some(p) = &f.path {
                self.insert(p.to_string_lossy(), f.bytes.clone());
            }
            self.insert(f.name, f.bytes);
        }
    }

    /// Selects the index of the primary entry point when multiple files are dropped simultaneously:
    /// prefers `.json` scene files first, then `.pvd` time-series descriptors, then the first file.
    #[must_use]
    pub fn select_primary_dropped_file(files: &[DroppedFilePayload]) -> Option<usize> {
        if files.is_empty() {
            return None;
        }
        if let Some(idx) = files
            .iter()
            .position(|f| is_json_label_or_bytes(Path::new(&f.name), &f.bytes))
        {
            return Some(idx);
        }
        if let Some(idx) = files.iter().position(|f| {
            detect_mesh_format(Path::new(&f.name), &f.bytes, None) == Some(MeshFormat::Pvd)
        }) {
            return Some(idx);
        }
        Some(0)
    }

    /// Looks up file bytes by exact normalized path first, falling back to basename match.
    #[must_use]
    pub fn get(&self, name_or_path: &str) -> Option<&[u8]> {
        let norm = normalize_lookup_key(name_or_path);
        if let Some(bytes) = self.files_by_key.get(&norm) {
            return Some(bytes.as_slice());
        }
        let base = basename_of_key(&norm);
        self.basenames.get(base).map(Vec::as_slice)
    }

    /// Returns the number of stored files.
    #[must_use]
    pub fn len(&self) -> usize {
        self.files_by_key.len()
    }

    /// Returns `true` if no files are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files_by_key.is_empty()
    }

    /// Clears all stored files.
    pub fn clear(&mut self) {
        self.files_by_key.clear();
        self.basenames.clear();
    }
}

fn normalize_lookup_key(raw: &str) -> String {
    let without_query = raw.split(['?', '#']).next().unwrap_or(raw).trim();
    let replaced = without_query.replace('\\', "/");
    let mut parts: Vec<&str> = Vec::new();
    for seg in replaced.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|last| *last != "..") {
                    parts.pop();
                } else {
                    parts.push("..");
                }
            }
            other => parts.push(other),
        }
    }
    if parts.is_empty() {
        replaced
    } else {
        parts.join("/")
    }
}

fn basename_of_key(norm: &str) -> &str {
    norm.rsplit('/').next().unwrap_or(norm)
}

/// Resolves a relative file or URL reference (`relative_ref`) against `base_url` (`M3.1`).
///
/// # Examples
/// ```
/// use tetraglass_viewer::resolve_relative_url;
///
/// assert_eq!(
///     resolve_relative_url(
///         "https://example.com/tetraglass/gallery/slice.json",
///         "../testdata/vtk/mixed_linear.vtu"
///     ),
///     "https://example.com/tetraglass/testdata/vtk/mixed_linear.vtu"
/// );
/// assert_eq!(
///     resolve_relative_url("vtk/series.pvd", "series_0.vtu"),
///     "vtk/series_0.vtu"
/// );
/// ```
#[must_use]
pub fn resolve_relative_url(base_url: &str, relative_ref: &str) -> String {
    let rel = relative_ref.trim();
    if rel.is_empty() {
        return base_url.to_string();
    }
    if rel.starts_with("http://")
        || rel.starts_with("https://")
        || rel.starts_with("data:")
        || rel.starts_with("blob:")
    {
        return rel.to_string();
    }

    let base_no_query = base_url.split(['?', '#']).next().unwrap_or(base_url).trim();
    let (scheme_origin, base_path) = split_url_origin_and_path(base_no_query);

    if let Some(stripped_root) = rel.strip_prefix('/') {
        let normalized = normalize_path_segments(stripped_root, true);
        return match scheme_origin {
            Some(origin) => format!("{origin}/{normalized}"),
            None => format!("/{normalized}"),
        };
    }

    let base_dir = match base_path.rfind('/') {
        Some(idx) => &base_path[..idx],
        None => "",
    };
    let has_leading_slash = base_path.starts_with('/');
    let combined = if base_dir.is_empty() {
        rel.to_string()
    } else {
        format!("{base_dir}/{rel}")
    };
    let normalized =
        normalize_path_segments(&combined, scheme_origin.is_some() || has_leading_slash);

    if let Some(origin) = scheme_origin {
        if normalized.is_empty() {
            origin.to_string()
        } else {
            format!("{origin}/{normalized}")
        }
    } else if has_leading_slash {
        format!("/{normalized}")
    } else {
        normalized
    }
}

fn split_url_origin_and_path(url: &str) -> (Option<&str>, &str) {
    if let Some(scheme_end) = url.find("://") {
        let after_scheme = &url[scheme_end + 3..];
        if let Some(slash_offset) = after_scheme.find('/') {
            let origin_end = scheme_end + 3 + slash_offset;
            (Some(&url[..origin_end]), &url[origin_end..])
        } else {
            (Some(url), "/")
        }
    } else {
        (None, url)
    }
}

fn normalize_path_segments(path: &str, clamp_at_root: bool) -> String {
    let mut stack: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                if stack.last().is_some_and(|last| *last != "..") {
                    stack.pop();
                } else if !clamp_at_root {
                    stack.push("..");
                }
            }
            other => stack.push(other),
        }
    }
    stack.join("/")
}

/// Extracts a short display label (typically the file name) from a URL or path string.
#[must_use]
pub fn url_display_label(url: &str) -> String {
    let clean = url.split(['?', '#']).next().unwrap_or(url).trim();
    let trimmed = clean.trim_end_matches('/');
    let name = trimmed.rsplit('/').next().unwrap_or(trimmed);
    if name.is_empty() {
        url.to_string()
    } else {
        name.to_string()
    }
}

/// Parses a URL query string (with or without a leading `'?'`) into a key-to-values map,
/// decoding `%XX` hex escapes and `'+'` spaces.
///
/// # Examples
/// ```
/// use tetraglass_viewer::parse_query_map;
///
/// let map = parse_query_map("?url=gallery%2Fslice.json&theme=dark");
/// assert_eq!(
///     map.get("url").and_then(|v| v.first()).map(String::as_str),
///     Some("gallery/slice.json")
/// );
/// ```
#[must_use]
pub fn parse_query_map(query: &str) -> BTreeMap<String, Vec<String>> {
    let raw = query.trim().strip_prefix('?').unwrap_or(query.trim());
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for pair in raw.split('&') {
        if pair.is_empty() {
            continue;
        }
        if let Some((k, v)) = pair.split_once('=') {
            map.entry(percent_decode_query(k))
                .or_default()
                .push(percent_decode_query(v));
        } else {
            map.entry(percent_decode_query(pair))
                .or_default()
                .push(String::new());
        }
    }
    map
}

/// Extracts the initial scene or mesh URL from a parsed query parameter map, checking
/// `"url"`, `"scene"`, and `"file"` in order (`M3.1`).
#[must_use]
pub fn extract_url_from_query_map(query_map: &BTreeMap<String, Vec<String>>) -> Option<String> {
    for key in ["url", "scene", "file"] {
        if let Some(values) = query_map.get(key) {
            for val in values {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.to_string());
                }
            }
        }
    }
    None
}

/// Extracts the initial scene or mesh URL directly from a raw query string (`?url=...`, `?scene=...`, or `?file=...`).
#[must_use]
pub fn extract_url_from_query_str(query: &str) -> Option<String> {
    let map = parse_query_map(query);
    extract_url_from_query_map(&map)
}

fn percent_decode_query(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                if let (Some(hi), Some(lo)) = (hex_nibble(bytes[i + 1]), hex_nibble(bytes[i + 2])) {
                    out.push((hi << 4) | lo);
                    i += 3;
                } else {
                    out.push(bytes[i]);
                    i += 1;
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// A primary file and any companion source/step files fetched from a URL (`M3.1`).
#[derive(Debug, Clone)]
pub struct FetchedUrlBundle {
    /// The requested URL.
    pub url: String,
    /// Short display label derived from `url`.
    pub label: String,
    /// Raw bytes of the primary file at `url` (a `.json` scene, `.pvd`, `.vtu`, or `.msh`).
    pub primary_bytes: Vec<u8>,
    /// Companion source meshes and `.pvd` step files resolved relative to `url`.
    pub companion_files: InMemoryFileStore,
}

/// Fetches a scene or mesh at `url` (plus any referenced source meshes and `.pvd` step files)
/// using a caller-supplied synchronous `fetcher` closure.
///
/// This powers URL loading in tests and native environments; the `wasm32` browser loader
/// mirrors the exact same traversal asynchronously over `fetch()`.
pub fn fetch_url_bundle_with<F>(url: &str, mut fetcher: F) -> Result<FetchedUrlBundle, ViewerError>
where
    F: FnMut(&str) -> Result<Vec<u8>, String>,
{
    let trimmed_url = url.trim();
    if trimmed_url.is_empty() {
        return Err(ViewerError::Validation("URL must not be empty".to_string()));
    }

    let primary_bytes = fetcher(trimmed_url).map_err(|msg| {
        ViewerError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("failed to load `{trimmed_url}`: {msg}"),
        ))
    })?;

    let label = url_display_label(trimmed_url);
    let label_path = Path::new(&label);
    let mut companion_files = InMemoryFileStore::new();

    if is_json_label_or_bytes(label_path, &primary_bytes) {
        let scene = Scene::from_json_bytes(&primary_bytes)?;
        for source in scene.sources.values() {
            let source_url = resolve_relative_url(trimmed_url, &source.file);
            let source_bytes = fetcher(&source_url).map_err(|msg| {
                ViewerError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "failed to load scene source `{}` from `{source_url}`: {msg}",
                        source.file
                    ),
                ))
            })?;
            let explicit_format = source.format.map(MeshFormat::from);
            let is_pvd =
                detect_mesh_format(Path::new(&source.file), &source_bytes, explicit_format)
                    == Some(MeshFormat::Pvd);
            if is_pvd {
                let entries = parse_pvd_entries(&source_bytes)?;
                for entry in entries {
                    let step_url = resolve_relative_url(&source_url, &entry.file);
                    let step_bytes = fetcher(&step_url).map_err(|msg| {
                        ViewerError::Io(std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            format!(
                                "failed to load PVD step `{}` from `{step_url}`: {msg}",
                                entry.file
                            ),
                        ))
                    })?;
                    companion_files.insert(&entry.file, step_bytes.clone());
                    companion_files.insert(&step_url, step_bytes);
                }
            }
            companion_files.insert(&source.file, source_bytes.clone());
            companion_files.insert(&source_url, source_bytes);
        }
    } else if detect_mesh_format(label_path, &primary_bytes, None) == Some(MeshFormat::Pvd) {
        let entries = parse_pvd_entries(&primary_bytes)?;
        for entry in entries {
            let step_url = resolve_relative_url(trimmed_url, &entry.file);
            let step_bytes = fetcher(&step_url).map_err(|msg| {
                ViewerError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "failed to load PVD step `{}` from `{step_url}`: {msg}",
                        entry.file
                    ),
                ))
            })?;
            companion_files.insert(&entry.file, step_bytes.clone());
            companion_files.insert(&step_url, step_bytes);
        }
    } else if detect_mesh_format(label_path, &primary_bytes, None).is_none() {
        return Err(ViewerError::MeshIo(IoError::UnsupportedFeature {
            offset: 0,
            feature: format!("unknown mesh or scene format for `{trimmed_url}`"),
        }));
    }

    Ok(FetchedUrlBundle {
        url: trimmed_url.to_string(),
        label,
        primary_bytes,
        companion_files,
    })
}

pub(crate) fn is_json_label_or_bytes(path: &Path, bytes: &[u8]) -> bool {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    {
        return true;
    }
    bytes.trim_ascii_start().starts_with(b"{")
}

pub(crate) fn detect_mesh_format(
    path: &Path,
    bytes: &[u8],
    explicit: Option<MeshFormat>,
) -> Option<MeshFormat> {
    explicit
        .or_else(|| {
            path.extension()
                .and_then(|e| e.to_str())
                .and_then(detect_format_from_extension)
        })
        .or_else(|| detect_format_from_bytes(bytes))
}

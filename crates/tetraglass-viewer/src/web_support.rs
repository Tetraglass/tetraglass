//! Platform-independent support types for the WebGPU browser viewer (`M3.1`, `M3.2`).
//!
//! All types and helpers in this module compile on both native and `wasm32-unknown-unknown`
//! targets so that WebGPU status formatting, WebGPU-only `wgpu` configuration (`D-007`),
//! drag-and-drop byte stores, URL resolution, multi-file scene/series bundle fetching,
//! and shareable links (`M3.2`) can be unit- and integration-tested on native CI as well
//! as used in the browser.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use base64::Engine as _;
use base64::engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD};
use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use serde::{Deserialize, Serialize};
use tetraglass_io::{
    IoError, MeshFormat, detect_format_from_bytes, detect_format_from_extension, parse_pvd_entries,
    parse_pvtu_manifest,
};
use tetraglass_scene::{ColormapName, Scene};

use crate::ViewerError;

/// Maximum decompressed payload size (`64 MiB`) allowed when decoding `?scene_b64=` or `?bundle_b64=`.
pub const MAX_SHAREABLE_DECOMPRESSED_BYTES: usize = 64 * 1024 * 1024;

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
        if let Some(idx) = files.iter().position(|f| {
            detect_mesh_format(Path::new(&f.name), &f.bytes, None) == Some(MeshFormat::Pvtu)
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

    /// Returns an iterator over all stored `(normalized_key, bytes)` entries.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &[u8])> {
        self.files_by_key
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
    }

    /// Exports all stored files as a [`BTreeMap`] of normalized keys to byte vectors.
    #[must_use]
    pub fn to_btree_map(&self) -> BTreeMap<String, Vec<u8>> {
        self.files_by_key.clone()
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

/// Percent-encodes a string for safe inclusion as a URL query parameter value (`M3.2`).
#[must_use]
pub fn percent_encode_query_component(input: &str) -> String {
    let mut out = String::with_capacity(input.len() + 8);
    for &b in input.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Compresses `bytes` with zlib and encodes the result using URL-safe Base64 without padding (`M3.2`).
pub fn encode_compressed_base64url(bytes: &[u8]) -> Result<String, ViewerError> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(bytes)?;
    let compressed = encoder.finish()?;
    Ok(URL_SAFE_NO_PAD.encode(compressed))
}

/// Decodes a URL-safe Base64 string and decompresses the zlib stream (up to [`MAX_SHAREABLE_DECOMPRESSED_BYTES`]) (`M3.2`).
pub fn decode_compressed_base64url(encoded: &str) -> Result<Vec<u8>, ViewerError> {
    let trimmed = encoded.trim();
    let compressed = URL_SAFE_NO_PAD
        .decode(trimmed)
        .or_else(|_| URL_SAFE.decode(trimmed))
        .map_err(|e| ViewerError::Validation(format!("invalid base64url payload: {e}")))?;

    let decoder = ZlibDecoder::new(compressed.as_slice());
    let mut limited = decoder.take((MAX_SHAREABLE_DECOMPRESSED_BYTES + 1) as u64);
    let mut out = Vec::new();
    limited.read_to_end(&mut out).map_err(|e| {
        ViewerError::Validation(format!("failed to decompress shareable payload: {e}"))
    })?;
    if out.len() > MAX_SHAREABLE_DECOMPRESSED_BYTES {
        return Err(ViewerError::Validation(format!(
            "decompressed shareable payload exceeds limit of {MAX_SHAREABLE_DECOMPRESSED_BYTES} bytes"
        )));
    }
    Ok(out)
}

/// Optional view-state overrides carried by a shareable link (`M3.2`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShareableViewOverrides {
    /// Camera azimuth angle in degrees (`?az=<deg>`).
    pub azimuth_deg: Option<f64>,
    /// Camera elevation angle in degrees (`?el=<deg>`).
    pub elevation_deg: Option<f64>,
    /// Camera roll angle in degrees (`?roll=<deg>`).
    pub roll_deg: Option<f64>,
    /// Camera zoom factor (`?zoom=<factor>`).
    pub zoom: Option<f64>,
    /// Zero-based time-step index (`?step=<idx>`).
    pub step: Option<usize>,
    /// Primary representation `color_by` array reference (`?color_by=point:linear` or `solid`).
    pub color_by: Option<String>,
    /// Primary representation colormap (`?colormap=viridis`).
    pub colormap: Option<ColormapName>,
    /// Logarithmic scalar mapping (`?log=true` / `?log_scale=true`).
    pub log_scale: Option<bool>,
    /// Colorbar visibility (`?colorbar=true`).
    pub colorbar: Option<bool>,
    /// Orientation axes triad visibility (`?axes=true`).
    pub axes: Option<bool>,
    /// Primary representation surface opacity in `[0.0, 1.0]` (`?opacity=0.6`).
    pub opacity: Option<f64>,
}

impl ShareableViewOverrides {
    /// Returns `true` if no overrides are set.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.azimuth_deg.is_none()
            && self.elevation_deg.is_none()
            && self.roll_deg.is_none()
            && self.zoom.is_none()
            && self.step.is_none()
            && self.color_by.is_none()
            && self.colormap.is_none()
            && self.log_scale.is_none()
            && self.colorbar.is_none()
            && self.axes.is_none()
            && self.opacity.is_none()
    }

    /// Appends non-empty overrides as `key=value` query parameters to `params`.
    pub fn append_query_params(&self, params: &mut Vec<String>) {
        if let Some(az) = self.azimuth_deg {
            params.push(format!("az={az:.2}"));
        }
        if let Some(el) = self.elevation_deg {
            params.push(format!("el={el:.2}"));
        }
        if let Some(roll) = self.roll_deg
            && roll.abs() > 1e-4
        {
            params.push(format!("roll={roll:.2}"));
        }
        if let Some(zoom) = self.zoom {
            params.push(format!("zoom={zoom:.3}"));
        }
        if let Some(step) = self.step {
            params.push(format!("step={step}"));
        }
        if let Some(ref color_by) = self.color_by {
            params.push(format!(
                "color_by={}",
                percent_encode_query_component(color_by)
            ));
        }
        if let Some(cmap) = self.colormap {
            params.push(format!("colormap={}", format_colormap_query(cmap)));
        }
        if let Some(log) = self.log_scale {
            params.push(format!("log={log}"));
        }
        if let Some(cb) = self.colorbar {
            params.push(format!("colorbar={cb}"));
        }
        if let Some(axes) = self.axes {
            params.push(format!("axes={axes}"));
        }
        if let Some(opacity) = self.opacity
            && (opacity - 1.0).abs() > 1e-4
        {
            params.push(format!("opacity={opacity:.3}"));
        }
    }
}

/// Self-contained scene + companion mesh data payload encoded in `?bundle_b64=<deflate+base64url>` (`M3.2`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineShareableBundle {
    /// Display label / filename of the primary entry (for example `"slice.json"` or `"tet.vtu"`).
    pub label: String,
    /// Raw bytes of the primary `.json` scene or mesh file.
    pub primary_bytes: Vec<u8>,
    /// Companion mesh files (source `.vtu`/`.msh`/`.pvd` and step `.vtu` files) keyed by relative path.
    pub companion_files: BTreeMap<String, Vec<u8>>,
}

impl InlineShareableBundle {
    /// Encodes this bundle into a compressed URL-safe Base64 string suitable for `?bundle_b64=`.
    pub fn encode_base64url(&self) -> Result<String, ViewerError> {
        let json_bytes = serde_json::to_vec(self)
            .map_err(|e| ViewerError::Validation(format!("bundle serialize error: {e}")))?;
        encode_compressed_base64url(&json_bytes)
    }

    /// Decodes an [`InlineShareableBundle`] from a `?bundle_b64=` compressed Base64url string.
    pub fn decode_base64url(encoded: &str) -> Result<Self, ViewerError> {
        let decompressed = decode_compressed_base64url(encoded)?;
        serde_json::from_slice(&decompressed)
            .map_err(|e| ViewerError::Validation(format!("invalid bundle_b64 JSON: {e}")))
    }
}

/// Parsed shareable link specification combining a scene/mesh target, companion data URLs,
/// or an inline compressed scene/bundle with optional view-state overrides (`M3.2`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShareableLinkSpec {
    /// Remote URL to the primary `.json` scene or mesh file (`?scene=`, `?url=`, or `?file=`).
    pub primary_url: Option<String>,
    /// Inline [`Scene`] decoded from `?scene_b64=<deflate+base64url>` or `?scene_json=<json>`.
    pub inline_scene: Option<Scene>,
    /// Self-contained scene + companion data bundle decoded from `?bundle_b64=<deflate+base64url>`.
    pub inline_bundle: Option<InlineShareableBundle>,
    /// Companion data URLs mapped by source name or filename (`?data=<name>:<url>` or `?data=<url>`).
    pub data_urls: BTreeMap<String, String>,
    /// Optional camera, time-step, and coloring overrides.
    pub view_overrides: ShareableViewOverrides,
}

impl ShareableLinkSpec {
    /// Parses a [`ShareableLinkSpec`] from a query string (`?scene=...`), hash fragment (`#scene=...`),
    /// full viewer URL (`https://.../viewer/?scene=...&data=...`), or direct file URL (`https://.../slice.json`).
    pub fn from_url_or_query(input: &str) -> Result<Option<Self>, ViewerError> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }

        // Merge query parameters from `?query` and `#fragment` if present.
        let mut combined_map: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let is_explicit_query_or_hash = trimmed.starts_with('?') || trimmed.starts_with('#');

        if is_explicit_query_or_hash {
            let (q_part, h_part) = match trimmed.split_once('#') {
                Some((q, h)) => (q, Some(h)),
                None => (trimmed, None),
            };
            merge_into_query_map(
                &mut combined_map,
                q_part.strip_prefix('#').unwrap_or(q_part),
            );
            if let Some(h) = h_part {
                merge_into_query_map(&mut combined_map, h);
            }
        } else {
            let (_, after_q) = match trimmed.split_once('?') {
                Some((base, rest)) => (base, Some(rest)),
                None => (trimmed, None),
            };
            if let Some(rest) = after_q {
                let (q_part, h_part) = match rest.split_once('#') {
                    Some((q, h)) => (q, Some(h)),
                    None => (rest, None),
                };
                merge_into_query_map(&mut combined_map, q_part);
                if let Some(h) = h_part {
                    merge_into_query_map(&mut combined_map, h);
                }
            } else if let Some((_, h_part)) = trimmed.split_once('#')
                && h_part.contains('=')
            {
                merge_into_query_map(&mut combined_map, h_part);
            }
        }

        if has_shareable_target_keys(&combined_map) {
            return Self::from_query_map(&combined_map);
        }

        if is_explicit_query_or_hash {
            return Ok(None);
        }

        // Treat as a direct file URL (`https://.../slice.json`), optionally carrying view overrides in `#fragment`.
        let mut spec = Self {
            primary_url: Some(trimmed.split('#').next().unwrap_or(trimmed).to_string()),
            ..Default::default()
        };
        if !combined_map.is_empty() {
            spec.view_overrides = parse_view_overrides(&combined_map)?;
        }
        Ok(Some(spec))
    }

    /// Parses a [`ShareableLinkSpec`] from a pre-parsed query parameter map (`M3.2`).
    pub fn from_query_map(
        query_map: &BTreeMap<String, Vec<String>>,
    ) -> Result<Option<Self>, ViewerError> {
        let primary_url = extract_url_from_query_map(query_map);

        let inline_scene = if let Some(b64) = first_non_empty(query_map, "scene_b64") {
            let bytes = decode_compressed_base64url(b64)?;
            Some(Scene::from_json_bytes(&bytes)?)
        } else if let Some(json_str) = first_non_empty(query_map, "scene_json") {
            Some(Scene::from_json_str(json_str)?)
        } else {
            None
        };

        let inline_bundle = if let Some(b64) = first_non_empty(query_map, "bundle_b64") {
            Some(InlineShareableBundle::decode_base64url(b64)?)
        } else {
            None
        };

        if primary_url.is_none() && inline_scene.is_none() && inline_bundle.is_none() {
            return Ok(None);
        }

        let mut data_urls = BTreeMap::new();
        if let Some(values) = query_map.get("data") {
            for raw in values {
                let val = raw.trim();
                if val.is_empty() {
                    continue;
                }
                if let Some((source_name, target_url)) = split_named_data_param(val) {
                    let filename = url_display_label(target_url);
                    data_urls.insert(source_name.to_string(), target_url.to_string());
                    data_urls.insert(filename, target_url.to_string());
                } else {
                    let filename = url_display_label(val);
                    data_urls
                        .entry("*".to_string())
                        .or_insert_with(|| val.to_string());
                    data_urls.insert(filename, val.to_string());
                }
            }
        }
        for (key, values) in query_map {
            let named = key
                .strip_prefix("data.")
                .or_else(|| key.strip_prefix("source."));
            if let Some(source_name) = named
                && !source_name.is_empty()
            {
                for val in values {
                    let trimmed = val.trim();
                    if !trimmed.is_empty() {
                        let filename = url_display_label(trimmed);
                        data_urls.insert(source_name.to_string(), trimmed.to_string());
                        data_urls.insert(filename, trimmed.to_string());
                    }
                }
            }
        }

        let view_overrides = parse_view_overrides(query_map)?;

        Ok(Some(Self {
            primary_url,
            inline_scene,
            inline_bundle,
            data_urls,
            view_overrides,
        }))
    }

    /// Resolves the remote URL for a scene source (`source_name`, `source_file`) against
    /// `self.data_urls` first, falling back to `base_url` if available (`M3.2`).
    #[must_use]
    pub fn resolve_source_url(
        &self,
        source_name: &str,
        source_file: &str,
        base_url: Option<&str>,
        total_sources: usize,
    ) -> Option<String> {
        if let Some(u) = self.data_urls.get(source_name) {
            return Some(u.clone());
        }
        if let Some(u) = self.data_urls.get(source_file) {
            return Some(u.clone());
        }
        let base_name = url_display_label(source_file);
        if let Some(u) = self.data_urls.get(&base_name) {
            return Some(u.clone());
        }
        if total_sources == 1
            && let Some(u) = self.data_urls.get("*")
        {
            return Some(u.clone());
        }
        if source_file.starts_with("http://")
            || source_file.starts_with("https://")
            || source_file.starts_with("file://")
            || source_file.starts_with("data:")
            || source_file.starts_with("blob:")
        {
            return Some(source_file.to_string());
        }
        base_url.map(|base| resolve_relative_url(base, source_file))
    }

    /// Formats this [`ShareableLinkSpec`] as a complete shareable viewer URL based on `base_viewer_url` (`M3.2`).
    pub fn to_viewer_url(&self, base_viewer_url: &str) -> Result<String, ViewerError> {
        let mut params = Vec::new();
        if let Some(ref bundle) = self.inline_bundle {
            let encoded = bundle.encode_base64url()?;
            params.push(format!("bundle_b64={encoded}"));
        } else if let Some(ref scene) = self.inline_scene {
            let json = scene.to_json_string()?;
            let encoded = encode_compressed_base64url(json.as_bytes())?;
            params.push(format!("scene_b64={encoded}"));
        } else if let Some(ref url) = self.primary_url {
            params.push(format!("scene={}", percent_encode_query_component(url)));
        } else {
            return Err(ViewerError::Validation(
                "ShareableLinkSpec has no primary_url, inline_scene, or inline_bundle".to_string(),
            ));
        }

        // Emit explicit `data.<name>=<url>` or `data=<url>` parameters (deduplicating basename aliases).
        let mut emitted_urls = std::collections::BTreeSet::new();
        if let Some(wildcard) = self.data_urls.get("*") {
            params.push(format!("data={}", percent_encode_query_component(wildcard)));
            emitted_urls.insert(wildcard.clone());
        }
        for (k, v) in &self.data_urls {
            if k == "*" || emitted_urls.contains(v) {
                continue;
            }
            if k == &url_display_label(v) {
                continue;
            }
            params.push(format!(
                "data.{}={}",
                percent_encode_query_component(k),
                percent_encode_query_component(v)
            ));
            emitted_urls.insert(v.clone());
        }

        self.view_overrides.append_query_params(&mut params);

        let clean_base = base_viewer_url
            .split(['?', '#'])
            .next()
            .unwrap_or(base_viewer_url)
            .trim();
        Ok(format!("{clean_base}?{}", params.join("&")))
    }
}

fn merge_into_query_map(target: &mut BTreeMap<String, Vec<String>>, raw_query: &str) {
    for (k, vals) in parse_query_map(raw_query) {
        target.entry(k).or_default().extend(vals);
    }
}

fn has_shareable_target_keys(map: &BTreeMap<String, Vec<String>>) -> bool {
    for k in [
        "url",
        "scene",
        "file",
        "scene_b64",
        "scene_json",
        "bundle_b64",
    ] {
        if map
            .get(k)
            .is_some_and(|v| v.iter().any(|s| !s.trim().is_empty()))
        {
            return true;
        }
    }
    false
}

fn first_non_empty<'a>(map: &'a BTreeMap<String, Vec<String>>, key: &str) -> Option<&'a str> {
    map.get(key)?
        .iter()
        .map(|s| s.trim())
        .find(|s| !s.is_empty())
}

fn split_named_data_param(val: &str) -> Option<(&str, &str)> {
    if val.starts_with("http://")
        || val.starts_with("https://")
        || val.starts_with("file://")
        || val.starts_with("data:")
        || val.starts_with("blob:")
        || val.starts_with('/')
    {
        return None;
    }
    let (prefix, rest) = val.split_once(':')?;
    if prefix.is_empty() || prefix.contains('/') || rest.trim().is_empty() {
        return None;
    }
    Some((prefix.trim(), rest.trim()))
}

fn parse_view_overrides(
    map: &BTreeMap<String, Vec<String>>,
) -> Result<ShareableViewOverrides, ViewerError> {
    let parse_f64 = |key: &str| -> Result<Option<f64>, ViewerError> {
        let Some(raw) = first_non_empty(map, key) else {
            return Ok(None);
        };
        let val = raw.parse::<f64>().map_err(|_| {
            ViewerError::Validation(format!("invalid numeric query parameter `{key}={raw}`"))
        })?;
        if !val.is_finite() {
            return Err(ViewerError::Validation(format!(
                "non-finite query parameter `{key}={raw}`"
            )));
        }
        Ok(Some(val))
    };

    let parse_bool = |key: &str| -> Result<Option<bool>, ViewerError> {
        let Some(raw) = first_non_empty(map, key) else {
            return Ok(None);
        };
        match raw.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(Some(true)),
            "0" | "false" | "no" | "off" => Ok(Some(false)),
            _ => Err(ViewerError::Validation(format!(
                "invalid boolean query parameter `{key}={raw}`"
            ))),
        }
    };

    let step = if let Some(raw) = first_non_empty(map, "step") {
        Some(
            raw.parse::<usize>()
                .map_err(|_| ViewerError::Validation(format!("invalid step index `step={raw}`")))?,
        )
    } else {
        None
    };

    let colormap = if let Some(raw) = first_non_empty(map, "colormap") {
        Some(parse_colormap_query(raw)?)
    } else {
        None
    };

    let log_scale = match parse_bool("log")? {
        Some(v) => Some(v),
        None => parse_bool("log_scale")?,
    };

    let opacity = parse_f64("opacity")?.or(parse_f64("alpha")?);
    if let Some(op) = opacity
        && !(0.0..=1.0).contains(&op)
    {
        return Err(ViewerError::Validation(format!(
            "opacity query parameter `{op}` must be in [0.0, 1.0]"
        )));
    }

    Ok(ShareableViewOverrides {
        azimuth_deg: parse_f64("az")?.or(parse_f64("azimuth")?),
        elevation_deg: parse_f64("el")?.or(parse_f64("elevation")?),
        roll_deg: parse_f64("roll")?,
        zoom: parse_f64("zoom")?,
        step,
        color_by: first_non_empty(map, "color_by").map(ToString::to_string),
        colormap,
        log_scale,
        colorbar: parse_bool("colorbar")?,
        axes: parse_bool("axes")?,
        opacity,
    })
}

const fn format_colormap_query(cmap: ColormapName) -> &'static str {
    match cmap {
        ColormapName::Viridis => "viridis",
        ColormapName::Cividis => "cividis",
        ColormapName::CoolWarm => "cool_warm",
        ColormapName::Turbo => "turbo",
        ColormapName::Grayscale => "grayscale",
    }
}

fn parse_colormap_query(raw: &str) -> Result<ColormapName, ViewerError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "viridis" => Ok(ColormapName::Viridis),
        "cividis" => Ok(ColormapName::Cividis),
        "cool_warm" | "coolwarm" | "cool-warm" => Ok(ColormapName::CoolWarm),
        "turbo" => Ok(ColormapName::Turbo),
        "grayscale" | "greyscale" | "gray" => Ok(ColormapName::Grayscale),
        other => Err(ViewerError::Validation(format!(
            "unknown colormap `{other}` in shareable link (expected viridis, cividis, cool_warm, turbo, or grayscale)"
        ))),
    }
}

/// A primary file and any companion source/step files fetched from a URL or shareable link (`M3.1`, `M3.2`).
#[derive(Debug, Clone)]
pub struct FetchedUrlBundle {
    /// The requested URL (or synthetic shareable link identifier).
    pub url: String,
    /// Short display label derived from `url` or embedded bundle.
    pub label: String,
    /// Raw bytes of the primary file (a `.json` scene, `.pvd`, `.vtu`, or `.msh`).
    pub primary_bytes: Vec<u8>,
    /// Companion source meshes and `.pvd` step files resolved relative to `url` or data overrides.
    pub companion_files: InMemoryFileStore,
    /// Optional view-state overrides parsed from the shareable link (`M3.2`).
    pub view_overrides: ShareableViewOverrides,
}

/// Fetches a scene or mesh at `url` (or resolves a full shareable viewer link / query string)
/// using a caller-supplied synchronous `fetcher` closure (`M3.1`, `M3.2`).
pub fn fetch_url_bundle_with<F>(url: &str, fetcher: F) -> Result<FetchedUrlBundle, ViewerError>
where
    F: FnMut(&str) -> Result<Vec<u8>, String>,
{
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return Err(ViewerError::Validation("URL must not be empty".to_string()));
    }
    let spec = ShareableLinkSpec::from_url_or_query(trimmed)?.ok_or_else(|| {
        ViewerError::Validation(format!("no scene or mesh target found in `{trimmed}`"))
    })?;
    fetch_shareable_link_with(&spec, fetcher)
}

/// Resolves and fetches all resources required by a [`ShareableLinkSpec`] using `fetcher` (`M3.2`).
pub fn fetch_shareable_link_with<F>(
    spec: &ShareableLinkSpec,
    mut fetcher: F,
) -> Result<FetchedUrlBundle, ViewerError>
where
    F: FnMut(&str) -> Result<Vec<u8>, String>,
{
    if let Some(ref inline_bundle) = spec.inline_bundle {
        let mut companion_files = InMemoryFileStore::new();
        for (k, v) in &inline_bundle.companion_files {
            companion_files.insert(k, v.clone());
        }
        return Ok(FetchedUrlBundle {
            url: spec
                .primary_url
                .clone()
                .unwrap_or_else(|| format!("inline:{}", inline_bundle.label)),
            label: inline_bundle.label.clone(),
            primary_bytes: inline_bundle.primary_bytes.clone(),
            companion_files,
            view_overrides: spec.view_overrides.clone(),
        });
    }

    let (base_url, label, primary_bytes) = if let Some(ref scene) = spec.inline_scene {
        let json_bytes = scene.to_json_string()?.into_bytes();
        let base = spec.primary_url.clone();
        let lbl = base
            .as_deref()
            .map_or_else(|| "shared_scene.json".to_string(), url_display_label);
        (base, lbl, json_bytes)
    } else if let Some(ref primary_url) = spec.primary_url {
        let trimmed_url = primary_url.trim();
        let bytes = fetcher(trimmed_url).map_err(|msg| {
            ViewerError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("failed to load `{trimmed_url}`: {msg}"),
            ))
        })?;
        (
            Some(trimmed_url.to_string()),
            url_display_label(trimmed_url),
            bytes,
        )
    } else {
        return Err(ViewerError::Validation(
            "shareable link has no scene, mesh URL, or inline bundle".to_string(),
        ));
    };

    let label_path = Path::new(&label);
    let mut companion_files = InMemoryFileStore::new();

    if is_json_label_or_bytes(label_path, &primary_bytes) {
        let scene = Scene::from_json_bytes(&primary_bytes)?;
        let total_sources = scene.sources.len();
        for (source_name, source) in &scene.sources {
            let source_url = spec
                .resolve_source_url(
                    source_name,
                    &source.file,
                    base_url.as_deref(),
                    total_sources,
                )
                .ok_or_else(|| {
                    ViewerError::Validation(format!(
                        "cannot resolve relative source `{}` (`{}`) without a base URL or `?data=` parameter",
                        source_name, source.file
                    ))
                })?;
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
            let detected =
                detect_mesh_format(Path::new(&source.file), &source_bytes, explicit_format);
            if detected == Some(MeshFormat::Pvd) {
                let entries = parse_pvd_entries(&source_bytes)?;
                for entry in entries {
                    let step_url = spec
                        .data_urls
                        .get(&entry.file)
                        .or_else(|| spec.data_urls.get(&url_display_label(&entry.file)))
                        .cloned()
                        .unwrap_or_else(|| resolve_relative_url(&source_url, &entry.file));
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
            } else if detected == Some(MeshFormat::Pvtu)
                || detect_format_from_bytes(&source_bytes) == Some(MeshFormat::Pvtu)
            {
                let manifest = parse_pvtu_manifest(&source_bytes)?;
                for piece in manifest.pieces {
                    let piece_url = spec
                        .data_urls
                        .get(&piece.source)
                        .or_else(|| spec.data_urls.get(&url_display_label(&piece.source)))
                        .cloned()
                        .unwrap_or_else(|| resolve_relative_url(&source_url, &piece.source));
                    let piece_bytes = fetcher(&piece_url).map_err(|msg| {
                        ViewerError::Io(std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            format!(
                                "failed to load PVTU piece `{}` from `{piece_url}`: {msg}",
                                piece.source
                            ),
                        ))
                    })?;
                    companion_files.insert(&piece.source, piece_bytes.clone());
                    companion_files.insert(&piece_url, piece_bytes);
                }
            }
            companion_files.insert(&source.file, source_bytes.clone());
            companion_files.insert(&source_url, source_bytes);
        }
    } else if detect_mesh_format(label_path, &primary_bytes, None) == Some(MeshFormat::Pvd) {
        let base_str = base_url.as_deref().unwrap_or("");
        let entries = parse_pvd_entries(&primary_bytes)?;
        for entry in entries {
            let step_url = spec
                .data_urls
                .get(&entry.file)
                .or_else(|| spec.data_urls.get(&url_display_label(&entry.file)))
                .cloned()
                .unwrap_or_else(|| resolve_relative_url(base_str, &entry.file));
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
    } else if detect_mesh_format(label_path, &primary_bytes, None) == Some(MeshFormat::Pvtu) {
        let base_str = base_url.as_deref().unwrap_or("");
        let manifest = parse_pvtu_manifest(&primary_bytes)?;
        for piece in manifest.pieces {
            let piece_url = spec
                .data_urls
                .get(&piece.source)
                .or_else(|| spec.data_urls.get(&url_display_label(&piece.source)))
                .cloned()
                .unwrap_or_else(|| resolve_relative_url(base_str, &piece.source));
            let piece_bytes = fetcher(&piece_url).map_err(|msg| {
                ViewerError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "failed to load PVTU piece `{}` from `{piece_url}`: {msg}",
                        piece.source
                    ),
                ))
            })?;
            companion_files.insert(&piece.source, piece_bytes.clone());
            companion_files.insert(&piece_url, piece_bytes);
        }
    } else if detect_mesh_format(label_path, &primary_bytes, None).is_none() {
        return Err(ViewerError::MeshIo(IoError::UnsupportedFeature {
            offset: 0,
            feature: format!(
                "unknown mesh or scene format for `{}`",
                base_url.as_deref().unwrap_or(&label)
            ),
        }));
    }

    Ok(FetchedUrlBundle {
        url: base_url.unwrap_or_else(|| label.clone()),
        label,
        primary_bytes,
        companion_files,
        view_overrides: spec.view_overrides.clone(),
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

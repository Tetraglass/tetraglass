//! Browser (`wasm32-unknown-unknown`) entry points, WebGPU capability checks, DOM error
//! banner rendering, and asynchronous URL fetching (`M3.1`).

use std::path::Path;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use tetraglass_io::{MeshFormat, parse_pvd_entries};
use tetraglass_scene::Scene;

use crate::web_support::{
    FetchedUrlBundle, InMemoryFileStore, WebGpuStatus, detect_mesh_format,
    extract_url_from_query_map, extract_url_from_query_str, is_json_label_or_bytes,
    resolve_relative_url, url_display_label, webgpu_only_wgpu_configuration,
};
use crate::{ViewerApp, ViewerDocument};

/// Default HTML `<canvas>` element ID used by `index.html` and [`start_web_viewer_default`].
pub const DEFAULT_CANVAS_ID: &str = "tetraglass_canvas";

/// Optional DOM container element ID where WebGPU diagnostic banners are rendered when present.
pub const WEBGPU_STATUS_CONTAINER_ID: &str = "tetraglass-webgpu-status";

/// Performs a synchronous browser check for a secure context (`https:` / `localhost`) and
/// the presence of `navigator.gpu`.
///
/// Returns `Some(WebGpuStatus)` if WebGPU is definitely unavailable, or `None` if `navigator.gpu`
/// is present and an async `requestAdapter()` probe should be run via [`detect_webgpu_status`].
#[must_use]
pub fn check_webgpu_status_sync() -> Option<WebGpuStatus> {
    let Some(window) = web_sys::window() else {
        return Some(WebGpuStatus::NoAdapterAvailable {
            details: Some("Browser Window object is not available.".to_string()),
        });
    };

    if !window.is_secure_context() {
        let origin = window
            .location()
            .origin()
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "non-secure origin".to_string());
        return Some(WebGpuStatus::NotSecureContext { origin });
    }

    let navigator = window.navigator();
    let user_agent = navigator.user_agent().ok();
    let gpu = js_sys::Reflect::get(navigator.as_ref(), &JsValue::from_str("gpu")).ok();
    match gpu {
        Some(val) if !val.is_undefined() && !val.is_null() => None,
        _ => Some(WebGpuStatus::MissingNavigatorGpu { user_agent }),
    }
}

/// Probes the browser's WebGPU support (`window.isSecureContext`, `navigator.gpu`, and
/// `navigator.gpu.requestAdapter()`) and returns the resulting [`WebGpuStatus`] (`M3.1`, `D-007`).
pub async fn detect_webgpu_status() -> WebGpuStatus {
    if let Some(early_failure) = check_webgpu_status_sync() {
        return early_failure;
    }

    let Some(window) = web_sys::window() else {
        return WebGpuStatus::NoAdapterAvailable {
            details: Some("Browser Window object is not available.".to_string()),
        };
    };
    let navigator = window.navigator();
    let Ok(gpu) = js_sys::Reflect::get(navigator.as_ref(), &JsValue::from_str("gpu")) else {
        return WebGpuStatus::MissingNavigatorGpu {
            user_agent: navigator.user_agent().ok(),
        };
    };

    let Ok(req_fn_val) = js_sys::Reflect::get(&gpu, &JsValue::from_str("requestAdapter")) else {
        return WebGpuStatus::NoAdapterAvailable {
            details: Some("navigator.gpu.requestAdapter is not a function.".to_string()),
        };
    };
    let Ok(req_fn) = req_fn_val.dyn_into::<js_sys::Function>() else {
        return WebGpuStatus::NoAdapterAvailable {
            details: Some("navigator.gpu.requestAdapter is not callable.".to_string()),
        };
    };

    let promise_val = match req_fn.call0(&gpu) {
        Ok(val) => val,
        Err(err) => {
            return WebGpuStatus::NoAdapterAvailable {
                details: Some(js_value_to_string(&err)),
            };
        }
    };
    let Ok(promise) = promise_val.dyn_into::<js_sys::Promise>() else {
        return WebGpuStatus::NoAdapterAvailable {
            details: Some("navigator.gpu.requestAdapter did not return a Promise.".to_string()),
        };
    };

    match JsFuture::from(promise).await {
        Ok(adapter) if !adapter.is_null() && !adapter.is_undefined() => WebGpuStatus::Supported,
        Ok(_) => WebGpuStatus::NoAdapterAvailable {
            details: Some("navigator.gpu.requestAdapter() resolved to null.".to_string()),
        },
        Err(err) => WebGpuStatus::NoAdapterAvailable {
            details: Some(js_value_to_string(&err)),
        },
    }
}

/// Renders a clear user-facing diagnostic card into the page DOM when WebGPU is unavailable (`M3.1`).
pub fn render_webgpu_error_banner(canvas_id: Option<&str>, status: &WebGpuStatus) {
    let Some(msg) = status.error_message() else {
        return;
    };
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };

    if let Some(cid) = canvas_id
        && let Some(canvas_el) = document
            .get_element_by_id(cid)
            .and_then(|e| e.dyn_into::<web_sys::HtmlElement>().ok())
    {
        let _ = canvas_el.style().set_property("display", "none");
    }

    let html = msg.to_html_fragment();
    if let Some(container) = document.get_element_by_id(WEBGPU_STATUS_CONTAINER_ID) {
        container.set_inner_html(&html);
        if let Ok(el) = container.dyn_into::<web_sys::HtmlElement>() {
            let _ = el.style().set_property("display", "block");
        }
        return;
    }

    if let Some(body) = document.body()
        && let Ok(wrapper) = document.create_element("div")
    {
        let _ = wrapper.set_attribute("id", WEBGPU_STATUS_CONTAINER_ID);
        wrapper.set_inner_html(&html);
        let _ = body.append_child(&wrapper);
    }
}

/// Fetches raw bytes from `url` using the browser's `fetch()` API.
pub async fn fetch_bytes_from_url(url: &str) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or_else(|| "no global browser Window".to_string())?;
    let fetch_fn = js_sys::Reflect::get(window.as_ref(), &JsValue::from_str("fetch"))
        .map_err(|e| js_value_to_string(&e))?
        .dyn_into::<js_sys::Function>()
        .map_err(|_| "window.fetch is not a function".to_string())?;

    let resp_promise = fetch_fn
        .call1(window.as_ref(), &JsValue::from_str(url))
        .map_err(|e| js_value_to_string(&e))?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| "window.fetch did not return a Promise".to_string())?;

    let resp_val = JsFuture::from(resp_promise)
        .await
        .map_err(|e| format!("network error fetching `{url}`: {}", js_value_to_string(&e)))?;

    let ok = js_sys::Reflect::get(&resp_val, &JsValue::from_str("ok"))
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !ok {
        let status = js_sys::Reflect::get(&resp_val, &JsValue::from_str("status"))
            .ok()
            .and_then(|v| v.as_f64())
            .map_or(0, |n| n as u16);
        let status_text = js_sys::Reflect::get(&resp_val, &JsValue::from_str("statusText"))
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        return Err(format!("HTTP {status} {status_text} for `{url}`"));
    }

    let ab_fn = js_sys::Reflect::get(&resp_val, &JsValue::from_str("arrayBuffer"))
        .map_err(|e| js_value_to_string(&e))?
        .dyn_into::<js_sys::Function>()
        .map_err(|_| "Response.arrayBuffer is not a function".to_string())?;

    let ab_promise = ab_fn
        .call0(&resp_val)
        .map_err(|e| js_value_to_string(&e))?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| "Response.arrayBuffer did not return a Promise".to_string())?;

    let ab_val = JsFuture::from(ab_promise).await.map_err(|e| {
        format!(
            "failed reading response body for `{url}`: {}",
            js_value_to_string(&e)
        )
    })?;

    Ok(js_sys::Uint8Array::new(&ab_val).to_vec())
}

/// Asynchronously fetches a scene JSON, `.pvd` collection, or raw `.vtu`/`.msh` mesh at `url`
/// along with any referenced companion files over browser `fetch()`.
pub async fn fetch_url_bundle_async(url: &str) -> Result<FetchedUrlBundle, String> {
    let trimmed_url = url.trim();
    if trimmed_url.is_empty() {
        return Err("URL must not be empty".to_string());
    }

    let primary_bytes = fetch_bytes_from_url(trimmed_url).await?;
    let label = url_display_label(trimmed_url);
    let label_path = Path::new(&label);
    let mut companion_files = InMemoryFileStore::new();

    if is_json_label_or_bytes(label_path, &primary_bytes) {
        let scene = Scene::from_json_bytes(&primary_bytes).map_err(|e| e.to_string())?;
        for source in scene.sources.values() {
            let source_url = resolve_relative_url(trimmed_url, &source.file);
            let source_bytes = fetch_bytes_from_url(&source_url).await?;
            let explicit_format = source.format.map(MeshFormat::from);
            let is_pvd =
                detect_mesh_format(Path::new(&source.file), &source_bytes, explicit_format)
                    == Some(MeshFormat::Pvd);
            if is_pvd {
                let entries = parse_pvd_entries(&source_bytes).map_err(|e| e.to_string())?;
                for entry in entries {
                    let step_url = resolve_relative_url(&source_url, &entry.file);
                    let step_bytes = fetch_bytes_from_url(&step_url).await?;
                    companion_files.insert(&entry.file, step_bytes.clone());
                    companion_files.insert(&step_url, step_bytes);
                }
            }
            companion_files.insert(&source.file, source_bytes.clone());
            companion_files.insert(&source_url, source_bytes);
        }
    } else if detect_mesh_format(label_path, &primary_bytes, None) == Some(MeshFormat::Pvd) {
        let entries = parse_pvd_entries(&primary_bytes).map_err(|e| e.to_string())?;
        for entry in entries {
            let step_url = resolve_relative_url(trimmed_url, &entry.file);
            let step_bytes = fetch_bytes_from_url(&step_url).await?;
            companion_files.insert(&entry.file, step_bytes.clone());
            companion_files.insert(&step_url, step_bytes);
        }
    } else if detect_mesh_format(label_path, &primary_bytes, None).is_none() {
        return Err(format!("unknown mesh or scene format for `{trimmed_url}`"));
    }

    Ok(FetchedUrlBundle {
        url: trimmed_url.to_string(),
        label,
        primary_bytes,
        companion_files,
    })
}

/// JavaScript handle for controlling the Tetraglass WebGPU viewer (`M3.1`).
#[derive(Clone)]
#[wasm_bindgen]
pub struct WebViewerHandle {
    runner: eframe::WebRunner,
}

#[wasm_bindgen]
impl WebViewerHandle {
    /// Initializes console logging and creates a new [`WebViewerHandle`].
    #[allow(clippy::new_without_default)] // Reason: #[wasm_bindgen(constructor)] entry point
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        let _ = eframe::WebLogger::init(log::LevelFilter::Info);
        Self {
            runner: eframe::WebRunner::new(),
        }
    }

    /// Starts the Tetraglass WebGPU viewer inside the `<canvas>` element with `canvas_id`.
    ///
    /// If the browser does not support WebGPU (or is not in a secure context), renders a clear
    /// diagnostic banner into the page DOM and returns an error instead of falling back to WebGL2.
    #[wasm_bindgen]
    pub async fn start_by_id(&self, canvas_id: &str) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("no browser Window"))?;
        let document = window
            .document()
            .ok_or_else(|| JsValue::from_str("no browser Document"))?;
        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or_else(|| JsValue::from_str(&format!("canvas element `#{canvas_id}` not found")))?
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_| JsValue::from_str(&format!("element `#{canvas_id}` is not a <canvas>")))?;

        self.start(canvas).await
    }

    /// Starts the Tetraglass WebGPU viewer on `canvas`.
    #[wasm_bindgen]
    pub async fn start(&self, canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
        let canvas_id = {
            let id = canvas.id();
            (!id.is_empty()).then_some(id)
        };

        let status = detect_webgpu_status().await;
        if let Some(msg) = status.error_message() {
            render_webgpu_error_banner(canvas_id.as_deref(), &status);
            let plain = msg.to_plain_text();
            log::error!("{plain}");
            return Err(JsValue::from_str(&plain));
        }

        let web_options = eframe::WebOptions {
            wgpu_options: webgpu_only_wgpu_configuration(),
            ..Default::default()
        };

        let browser_query_url = web_sys::window()
            .and_then(|w| w.location().search().ok())
            .and_then(|q| extract_url_from_query_str(&q));

        let start_res = self
            .runner
            .start(
                canvas,
                web_options,
                Box::new(move |cc| {
                    let initial_url = extract_url_from_query_map(
                        &cc.integration_info.web_info.location.query_map,
                    )
                    .or(browser_query_url);
                    let doc = ViewerDocument::default_demo()?;
                    let mut app = ViewerApp::new(cc, doc, None)?;
                    if let Some(url) = initial_url {
                        app.session_mut()
                            .request_load_url(&url, Some(cc.egui_ctx.clone()));
                    }
                    Ok(Box::new(app))
                }),
            )
            .await;

        if let Err(ref err) = start_res {
            let fallback_status = WebGpuStatus::NoAdapterAvailable {
                details: Some(js_value_to_string(err)),
            };
            render_webgpu_error_banner(canvas_id.as_deref(), &fallback_status);
        }

        start_res
    }

    /// Triggers an asynchronous load of a `.vtu`, `.msh`, `.pvd`, or `.json` scene from `url`.
    #[wasm_bindgen]
    pub fn load_url(&self, url: &str) {
        if let Some(mut app) = self.runner.app_mut::<ViewerApp>() {
            app.session_mut().request_load_url(url, None);
        }
    }

    /// Shuts down the viewer and releases WebGPU resources.
    #[wasm_bindgen]
    pub fn destroy(&self) {
        self.runner.destroy();
    }

    /// Returns `true` if the viewer panicked.
    #[wasm_bindgen]
    #[must_use]
    pub fn has_panicked(&self) -> bool {
        self.runner.has_panicked()
    }

    /// Returns the panic summary message if the viewer panicked.
    #[wasm_bindgen]
    #[must_use]
    pub fn panic_message(&self) -> Option<String> {
        self.runner.panic_summary().map(|s| s.message())
    }
}

/// Starts the Tetraglass WebGPU viewer on [`DEFAULT_CANVAS_ID`] (`"tetraglass_canvas"`).
#[wasm_bindgen]
pub async fn start_web_viewer_default() -> Result<WebViewerHandle, JsValue> {
    let handle = WebViewerHandle::new();
    handle.start_by_id(DEFAULT_CANVAS_ID).await?;
    Ok(handle)
}

fn js_value_to_string(val: &JsValue) -> String {
    val.as_string().unwrap_or_else(|| format!("{val:?}"))
}

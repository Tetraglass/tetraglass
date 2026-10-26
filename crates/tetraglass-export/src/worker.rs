//! Persistent Blender Cycles socket worker protocol, client, and process manager (`M5.4`).
//!
//! Communicates with `blender/render.py --serve <host:port|ipc:dir>` using line-delimited JSON
//! over a loopback TCP socket or an atomic local IPC directory (`D-011`, `.agents/rules/blender-gpl.md`).

#[cfg(not(target_arch = "wasm32"))]
use std::env;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{BufRead, BufReader, Write};
#[cfg(not(target_arch = "wasm32"))]
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::process::{Child, Command, Stdio};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use tetraglass_core::BoundingBox;
#[cfg(not(target_arch = "wasm32"))]
use tetraglass_scene::Scene;

#[cfg(not(target_arch = "wasm32"))]
use crate::ExportError;
use crate::bundle::{BundleBackground, BundleCamera};
#[cfg(not(target_arch = "wasm32"))]
use crate::bundle::{resolve_bundle_background, resolve_bundle_camera_and_lights};

/// Default timeout for a single request to the persistent Blender worker (`300` seconds).
pub const DEFAULT_WORKER_TIMEOUT_SECS: u64 = 300;

#[cfg(not(target_arch = "wasm32"))]
static WORKER_REQ_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Line-delimited JSON request sent to `blender/render.py --serve <host:port>` (`M5.4`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum BlenderWorkerRequest {
    /// Health check returning the worker's Blender version and currently loaded bundle path.
    Ping,

    /// Imports (or replaces) a Tetraglass bundle directory in the warm Blender session.
    LoadBundle {
        /// Filesystem path to the bundle directory or its `manifest.json`.
        bundle: String,
    },

    /// Updates the active camera, headlight, background, resolution, and frame step in-place
    /// without re-importing mesh geometry (`M5.4` live link).
    UpdateCamera {
        /// Resolved camera configuration.
        camera: BundleCamera,
        /// Optional updated background specification.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background: Option<BundleBackground>,
        /// Optional updated render resolution `[width, height]`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image_size: Option<[u32; 2]>,
        /// Optional active time-step frame index.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        step: Option<usize>,
    },

    /// Renders one or more frames from the loaded (or newly supplied) bundle using Cycles.
    Render(BlenderWorkerRenderRequest),

    /// Cleanly terminates the persistent Blender worker loop.
    Shutdown,
}

/// Parameters for a [`BlenderWorkerRequest::Render`] command (`M5.4`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlenderWorkerRenderRequest {
    /// Optional bundle directory path. If omitted or identical to the already-loaded bundle
    /// (and `force_reload` is `false`), the warm Blender scene is reused without re-importing meshes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle: Option<String>,
    /// Forces a fresh bundle import even if `bundle` matches the currently loaded path.
    #[serde(default)]
    pub force_reload: bool,
    /// Output PNG path or frame pattern (`frame_%04d.png`).
    pub out: String,
    /// Cycles path-tracing sample count per pixel.
    pub samples: u32,
    /// Cycles compute device (`"auto"`, `"cpu"`, `"gpu"`, `"metal"`, `"optix"`, `"cuda"`, `"hip"`).
    pub device: String,
    /// Optional single time-step index to render.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<usize>,
    /// Whether to render all time steps in the bundle.
    #[serde(default)]
    pub all_steps: bool,
    /// Optional inclusive start step index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_step: Option<usize>,
    /// Optional inclusive end step index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_step: Option<usize>,
    /// Optional width override in pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Optional height override in pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    /// Optional path to save a `.blend` file before rendering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub save_blend: Option<String>,
    /// Optional in-place camera override applied before rendering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<BundleCamera>,
    /// Optional in-place background override applied before rendering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<BundleBackground>,
}

impl BlenderWorkerRenderRequest {
    /// Creates a new [`BlenderWorkerRenderRequest`] writing to `out` with `samples` and `device`.
    #[must_use]
    pub fn new(out: impl Into<String>, samples: u32, device: impl Into<String>) -> Self {
        Self {
            bundle: None,
            force_reload: false,
            out: out.into(),
            samples: samples.max(1),
            device: device.into(),
            step: None,
            all_steps: false,
            from_step: None,
            to_step: None,
            width: None,
            height: None,
            save_blend: None,
            camera: None,
            background: None,
        }
    }
}

/// Line-delimited JSON response returned by `blender/render.py --serve <host:port>` (`M5.4`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlenderWorkerResponse {
    /// `"ok"` or `"error"`.
    pub status: String,
    /// Error message when `status == "error"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Blender version string reported by `bpy.app.version_string`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blender_version: Option<String>,
    /// Currently loaded bundle directory path in the worker, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loaded_bundle: Option<String>,
    /// Output image files written by a `render` request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<String>>,
    /// Active Cycles compute backend used by a `render` request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// Cycles sample count used by a `render` request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<u32>,
    /// Set to `true` in response to `shutdown`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shutting_down: Option<bool>,
    /// Live camera/view summary returned by `update_camera`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view: Option<serde_json::Value>,
    /// Bundle import summary returned by `load_bundle`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<serde_json::Value>,
}

impl BlenderWorkerResponse {
    /// Returns `true` if `status == "ok"`.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.status == "ok"
    }
}

/// Client for a persistent Blender worker (`blender/render.py --serve <host:port|ipc:dir>`) (`M5.4`).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub struct BlenderWorkerClient {
    addr: String,
    timeout: Duration,
}

#[cfg(not(target_arch = "wasm32"))]
impl BlenderWorkerClient {
    /// Creates a new [`BlenderWorkerClient`] targeting `addr` (for example `"127.0.0.1:55123"` or `"ipc:/tmp/..."`).
    #[must_use]
    pub fn new(addr: impl Into<String>) -> Self {
        Self {
            addr: addr.into(),
            timeout: Duration::from_secs(DEFAULT_WORKER_TIMEOUT_SECS),
        }
    }

    /// Sets the request timeout.
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Returns the target endpoint string (`host:port` or `ipc:<dir>`).
    #[must_use]
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Sends a [`BlenderWorkerRequest`] and awaits the parsed [`BlenderWorkerResponse`].
    ///
    /// Returns [`ExportError::Worker`] if the connection fails or the worker reports
    /// `status == "error"`.
    pub fn send_request(
        &self,
        request: &BlenderWorkerRequest,
    ) -> Result<BlenderWorkerResponse, ExportError> {
        let trimmed = self.addr.trim();
        let line = if let Some(ipc_path) = trimmed.strip_prefix("ipc:") {
            self.send_ipc_dir_request(Path::new(ipc_path), request)?
        } else {
            self.send_tcp_request(trimmed, request)?
        };

        let resp: BlenderWorkerResponse = serde_json::from_str(line.trim())?;
        if !resp.is_ok() {
            let msg = resp
                .error
                .unwrap_or_else(|| "unknown Blender worker error".to_string());
            return Err(ExportError::Worker(msg));
        }
        Ok(resp)
    }

    fn send_tcp_request(
        &self,
        addr_str: &str,
        request: &BlenderWorkerRequest,
    ) -> Result<String, ExportError> {
        let sock_addr = resolve_first_socket_addr(addr_str)?;
        let mut stream = TcpStream::connect_timeout(&sock_addr, self.timeout).map_err(|e| {
            ExportError::Worker(format!(
                "failed to connect to Blender worker at `{addr_str}`: {e}"
            ))
        })?;
        let _ = stream.set_read_timeout(Some(self.timeout));
        let _ = stream.set_write_timeout(Some(self.timeout));

        let mut payload = serde_json::to_string(request)?;
        payload.push('\n');
        stream.write_all(payload.as_bytes()).map_err(|e| {
            ExportError::Worker(format!(
                "failed to write request to Blender worker at `{addr_str}`: {e}"
            ))
        })?;
        stream.flush().map_err(|e| {
            ExportError::Worker(format!(
                "failed to flush request to Blender worker at `{addr_str}`: {e}"
            ))
        })?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        let n = reader.read_line(&mut line).map_err(|e| {
            ExportError::Worker(format!(
                "failed to read response from Blender worker at `{addr_str}`: {e}"
            ))
        })?;
        if n == 0 {
            return Err(ExportError::Worker(format!(
                "Blender worker at `{addr_str}` closed connection without replying"
            )));
        }
        Ok(line)
    }

    fn send_ipc_dir_request(
        &self,
        ipc_dir: &Path,
        request: &BlenderWorkerRequest,
    ) -> Result<String, ExportError> {
        fs::create_dir_all(ipc_dir)?;
        let seq = WORKER_REQ_COUNTER.fetch_add(1, Ordering::Relaxed);
        let req_id = format!("{}_{seq:08}", std::process::id());
        let tmp_req = ipc_dir.join(format!(".req_{req_id}.json.tmp"));
        let final_req = ipc_dir.join(format!("req_{req_id}.json"));
        let final_resp = ipc_dir.join(format!("resp_{req_id}.json"));

        let mut payload = serde_json::to_string(request)?;
        payload.push('\n');
        fs::write(&tmp_req, payload.as_bytes())?;
        fs::rename(&tmp_req, &final_req)?;

        let deadline = Instant::now() + self.timeout;
        loop {
            if final_resp.is_file() {
                let text = fs::read_to_string(&final_resp)?;
                let _ = fs::remove_file(&final_resp);
                return Ok(text);
            }
            if Instant::now() >= deadline {
                let _ = fs::remove_file(&final_req);
                return Err(ExportError::Worker(format!(
                    "timed out waiting for Blender worker response in `{}`",
                    ipc_dir.display()
                )));
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Sends a `ping` request to verify the worker is alive.
    pub fn ping(&self) -> Result<BlenderWorkerResponse, ExportError> {
        self.send_request(&BlenderWorkerRequest::Ping)
    }

    /// Instructs the worker to import `bundle_dir` into its Blender session.
    pub fn load_bundle(
        &self,
        bundle_dir: impl AsRef<Path>,
    ) -> Result<BlenderWorkerResponse, ExportError> {
        self.send_request(&BlenderWorkerRequest::LoadBundle {
            bundle: bundle_dir.as_ref().display().to_string(),
        })
    }

    /// Pushes an in-place camera, background, resolution, and step update to the worker (`M5.4`).
    pub fn update_camera(
        &self,
        camera: BundleCamera,
        background: Option<BundleBackground>,
        image_size: Option<[u32; 2]>,
        step: Option<usize>,
    ) -> Result<BlenderWorkerResponse, ExportError> {
        self.send_request(&BlenderWorkerRequest::UpdateCamera {
            camera,
            background,
            image_size,
            step,
        })
    }

    /// Resolves the live camera and background from `scene` and `bounds` and pushes them to the
    /// worker in-place (`M5.4` live link).
    pub fn push_live_scene_view(
        &self,
        scene: &Scene,
        bounds: BoundingBox,
        step: Option<usize>,
    ) -> Result<BlenderWorkerResponse, ExportError> {
        let dataset_center = bounds.center();
        let (camera, _lights) = resolve_bundle_camera_and_lights(scene, bounds, dataset_center);
        let background = resolve_bundle_background(&scene.view.background);
        let image_size = scene.output.as_ref().map(|o| o.size);
        self.update_camera(camera, Some(background), image_size, step)
    }

    /// Triggers a Cycles render on the persistent worker.
    pub fn render(
        &self,
        request: &BlenderWorkerRenderRequest,
    ) -> Result<BlenderWorkerResponse, ExportError> {
        self.send_request(&BlenderWorkerRequest::Render(request.clone()))
    }

    /// Requests a clean shutdown of the persistent worker.
    pub fn shutdown(&self) -> Result<BlenderWorkerResponse, ExportError> {
        self.send_request(&BlenderWorkerRequest::Shutdown)
    }
}

/// Managed child process running `blender --background --factory-startup --python blender/render.py -- --serve ...` (`M5.4`).
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub struct BlenderWorkerProcess {
    child: Child,
    addr: String,
    blender_version: String,
    ipc_cleanup_dir: Option<PathBuf>,
}

#[cfg(not(target_arch = "wasm32"))]
impl BlenderWorkerProcess {
    /// Spawns a persistent headless Blender worker and waits up to `startup_timeout` for its
    /// `PROGRESS stage=worker_listening addr=<endpoint>` readiness line.
    ///
    /// Pass `"127.0.0.1:0"` (or `"auto"`) as `bind_addr` to let the OS assign an available
    /// loopback port (or automatically fall back to an `ipc:<temp_dir>` endpoint when running
    /// inside a network-isolated sandbox).
    pub fn spawn(
        blender_bin: impl AsRef<Path>,
        render_script: impl AsRef<Path>,
        bind_addr: &str,
        startup_timeout: Duration,
    ) -> Result<Self, ExportError> {
        let (effective_bind_addr, ipc_cleanup_dir) = resolve_effective_bind_addr(bind_addr)?;

        let mut cmd = Command::new(blender_bin.as_ref());
        cmd.arg("--background")
            .arg("--factory-startup")
            .arg("--python")
            .arg(render_script.as_ref())
            .arg("--")
            .arg("--serve")
            .arg(&effective_bind_addr)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            ExportError::Worker(format!(
                "failed to spawn Blender worker `{}`: {e}",
                blender_bin.as_ref().display()
            ))
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            ExportError::Worker("failed to capture Blender worker stdout".to_string())
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            ExportError::Worker("failed to capture Blender worker stderr".to_string())
        })?;

        let (ready_tx, ready_rx) = mpsc::channel::<Result<(String, String), String>>();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            let mut announced = false;
            for line in reader.lines().map_while(Result::ok) {
                if !announced && line.contains("PROGRESS stage=worker_listening") {
                    let addr = extract_progress_field(&line, "addr")
                        .unwrap_or_else(|| "127.0.0.1:0".to_string());
                    let ver = extract_progress_field(&line, "blender_version")
                        .unwrap_or_else(|| "unknown".to_string());
                    let _ = ready_tx.send(Ok((addr, ver)));
                    announced = true;
                }
            }
            if !announced {
                let _ = ready_tx.send(Err(
                    "Blender process exited before announcing `worker_listening`".to_string(),
                ));
            }
        });

        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for _line in reader.lines().map_while(Result::ok) {}
        });

        match ready_rx.recv_timeout(startup_timeout) {
            Ok(Ok((addr, blender_version))) => Ok(Self {
                child,
                addr,
                blender_version,
                ipc_cleanup_dir,
            }),
            Ok(Err(err)) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(dir) = ipc_cleanup_dir {
                    let _ = fs::remove_dir_all(dir);
                }
                Err(ExportError::Worker(err))
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(dir) = ipc_cleanup_dir {
                    let _ = fs::remove_dir_all(dir);
                }
                Err(ExportError::Worker(format!(
                    "timed out after {:.1}s waiting for Blender worker to listen",
                    startup_timeout.as_secs_f64()
                )))
            }
        }
    }

    /// Returns the bound endpoint (`host:port` or `ipc:<dir>`) of the running worker.
    #[must_use]
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Returns the Blender version reported by the running worker.
    #[must_use]
    pub fn blender_version(&self) -> &str {
        &self.blender_version
    }

    /// Returns a [`BlenderWorkerClient`] connected to this worker process.
    #[must_use]
    pub fn client(&self) -> BlenderWorkerClient {
        BlenderWorkerClient::new(self.addr.clone())
    }

    /// Sends a graceful `shutdown` command and waits for the Blender child process to exit.
    pub fn shutdown(&mut self) -> Result<(), ExportError> {
        let _ = self
            .client()
            .with_timeout(Duration::from_secs(5))
            .shutdown();
        let _ = self.child.wait();
        if let Some(dir) = self.ipc_cleanup_dir.take() {
            let _ = fs::remove_dir_all(dir);
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for BlenderWorkerProcess {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            let _ = self
                .client()
                .with_timeout(Duration::from_secs(2))
                .shutdown();
            if let Ok(None) = self.child.try_wait() {
                let _ = self.child.kill();
            }
            let _ = self.child.wait();
        }
        if let Some(dir) = self.ipc_cleanup_dir.take() {
            let _ = fs::remove_dir_all(dir);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn loopback_tcp_connect_permitted() -> bool {
    let Ok(listener) = TcpListener::bind("127.0.0.1:0") else {
        return false;
    };
    let Ok(addr) = listener.local_addr() else {
        return false;
    };
    TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_effective_bind_addr(bind_addr: &str) -> Result<(String, Option<PathBuf>), ExportError> {
    let trimmed = bind_addr.trim();
    if let Some(stripped) = trimmed.strip_prefix("ipc:") {
        let dir = PathBuf::from(stripped);
        fs::create_dir_all(&dir)?;
        return Ok((trimmed.to_string(), None));
    }
    if trimmed == "auto"
        || ((trimmed == "127.0.0.1:0" || trimmed == "localhost:0")
            && !loopback_tcp_connect_permitted())
    {
        let nonce = WORKER_REQ_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = env::temp_dir().join(format!(
            "tetraglass-blender-worker-{}-{nanos}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&dir)?;
        return Ok((format!("ipc:{}", dir.display()), Some(dir)));
    }
    Ok((trimmed.to_string(), None))
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_first_socket_addr(addr: &str) -> Result<SocketAddr, ExportError> {
    addr.to_socket_addrs()
        .map_err(|e| ExportError::Worker(format!("invalid worker address `{addr}`: {e}")))?
        .next()
        .ok_or_else(|| ExportError::Worker(format!("could not resolve worker address `{addr}`")))
}

#[cfg(not(target_arch = "wasm32"))]
fn extract_progress_field(line: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    let start = line.find(&prefix)? + prefix.len();
    let rest = &line[start..];
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        Some(stripped[..end].to_string())
    } else {
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        Some(rest[..end].to_string())
    }
}

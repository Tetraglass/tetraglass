//! Desktop and WebGPU viewer binary (`tetraglass-viewer`) for opening scenes and meshes (`M1.8`, `M3.1`).
//!
//! Run natively with:
//! ```sh
//! cargo run -p tetraglass-viewer -- path/to/scene.json
//! ```
//!
//! Or compile for WebGPU (`wasm32-unknown-unknown`) and host with `crates/tetraglass-viewer/web/index.html`.

#[cfg(not(target_arch = "wasm32"))]
use std::env;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::process::ExitCode;

#[cfg(not(target_arch = "wasm32"))]
use tetraglass_viewer::{ViewerOptions, run_viewer};

#[cfg(not(target_arch = "wasm32"))]
fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match parse_viewer_args(&args) {
        Ok(Some(options)) => match run_viewer(options) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("tetraglass-viewer: {err}");
                ExitCode::FAILURE
            }
        },
        Ok(None) => {
            println!("{}", help_text());
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("tetraglass-viewer: {msg}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    wasm_bindgen_futures::spawn_local(async {
        if let Err(err) = tetraglass_viewer::start_web_viewer_default().await {
            log::error!("Failed to launch Tetraglass WebGPU viewer: {err:?}");
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn help_text() -> String {
    format!(
        "tetraglass-viewer {}\n\n\
         Usage:\n  \
         tetraglass-viewer <scene.json|mesh> [--frames <n>]",
        env!("CARGO_PKG_VERSION")
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_viewer_args(args: &[String]) -> Result<Option<ViewerOptions>, String> {
    let mut path: Option<PathBuf> = None;
    let mut max_frames: Option<u64> = None;
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" | "help" => return Ok(None),
            "--frames" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    return Err("missing value for `--frames <n>`".to_string());
                };
                let parsed = val
                    .parse::<u64>()
                    .map_err(|_| format!("invalid `--frames` count `{val}`"))?;
                max_frames = Some(parsed);
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option `{other}`"));
            }
            arg => {
                if path.is_some() {
                    return Err("expected a single `<scene.json|mesh>` path argument".to_string());
                }
                path = Some(PathBuf::from(arg));
            }
        }
        i += 1;
    }

    let Some(path) = path else {
        return Err(
            "missing `<scene.json|mesh>` argument; run `tetraglass-viewer --help` for usage"
                .to_string(),
        );
    };

    let mut options = ViewerOptions::new(path);
    options.max_frames = max_frames;
    Ok(Some(options))
}

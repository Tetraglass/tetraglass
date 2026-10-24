//! Developer tasks for this workspace. Run with `cargo xtask <task>`.
//!
//! Uses only the standard library so it builds instantly and never needs updating
//! when dependencies change.

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

/// Crates whose directory names end with these suffixes must build for wasm32.
const WASM_CRATE_SUFFIXES: [&str; 7] = [
    "-core", "-io", "-filters", "-scene", "-render", "-egui", "-viewer",
];
const WASM_TARGET: &str = "wasm32-unknown-unknown";

fn main() -> ExitCode {
    let task = env::args().nth(1).unwrap_or_default();
    let result = match task.as_str() {
        "verify" => verify(),
        "wasm-check" => wasm_check(),
        "fmt" => run("cargo", &["fmt", "--all"]),
        "" | "help" | "--help" | "-h" => {
            print_help();
            return ExitCode::SUCCESS;
        }
        other => Err(format!("unknown task `{other}`; run `cargo xtask help`")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("xtask: {message}");
            ExitCode::FAILURE
        }
    }
}

fn print_help() {
    println!(
        "Usage: cargo xtask <task>\n\n\
         Tasks:\n  \
         verify      format check, clippy, tests, wasm32 check, cargo-deny\n  \
         wasm-check  build the web-compatible crates for {WASM_TARGET}\n  \
         fmt         format the whole workspace"
    );
}

/// Runs every check `/verify` relies on, stopping at the first failure.
fn verify() -> Result<(), String> {
    step("format", || run("cargo", &["fmt", "--all", "--check"]))?;
    step("clippy", || {
        run(
            "cargo",
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        )
    })?;
    step("tests", || run("cargo", &["test", "--workspace"]))?;
    step("wasm32", wasm_check)?;
    if cargo_has_subcommand("deny") {
        step("cargo-deny", || run("cargo", &["deny", "check"]))?;
    } else {
        println!(
            "xtask: warning: cargo-deny is not installed, license checks skipped \
             (install with `cargo install cargo-deny --locked`)"
        );
    }
    println!("xtask: verify passed");
    Ok(())
}

/// Checks that the platform-independent crates compile for the browser.
fn wasm_check() -> Result<(), String> {
    let crates = wasm_crates()?;
    if crates.is_empty() {
        println!("xtask: no web-compatible crates found yet, skipping wasm32 check");
        return Ok(());
    }
    let mut args = vec!["check", "--target", WASM_TARGET];
    for name in &crates {
        args.push("-p");
        args.push(name);
    }
    run("cargo", &args).map_err(|error| {
        format!("{error}\nhint: install the target with `rustup target add {WASM_TARGET}`")
    })
}

/// Finds crate directories under `crates/` that must stay wasm32-compatible.
fn wasm_crates() -> Result<Vec<String>, String> {
    let crates_dir = workspace_root().join("crates");
    if !crates_dir.is_dir() {
        return Ok(Vec::new());
    }
    let entries = std::fs::read_dir(&crates_dir)
        .map_err(|error| format!("cannot read {}: {error}", crates_dir.display()))?;
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read crates directory: {error}"))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let is_crate = entry.path().join("Cargo.toml").is_file();
        if is_crate
            && WASM_CRATE_SUFFIXES
                .iter()
                .any(|suffix| name.ends_with(suffix))
        {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

fn step(name: &str, check: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
    println!("xtask: == {name} ==");
    check().map_err(|error| format!("{name} failed: {error}"))
}

fn run(program: &str, args: &[&str]) -> Result<(), String> {
    let status = Command::new(program)
        .args(args)
        .current_dir(workspace_root())
        .status()
        .map_err(|error| format!("cannot start `{program}`: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "`{program} {}` exited with {status}",
            args.join(" ")
        ))
    }
}

fn cargo_has_subcommand(name: &str) -> bool {
    Command::new("cargo")
        .args([name, "--version"])
        .output()
        .is_ok_and(|output| output.status.success())
}

fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .map_or_else(|| manifest_dir.to_path_buf(), Path::to_path_buf)
}

---
trigger: glob
globs: "**/*.rs"
description: "Rust coding conventions for this workspace: errors, panics, docs, numeric types, parallelism, logging and lints."
---

# Rust code conventions

**Toolchain.** Stable Rust, edition 2024. `rustfmt` defaults. `cargo clippy --all-targets -- -D warnings` must pass.

**Errors.**
- Each library crate defines its own error enum with `thiserror`. `anyhow` is allowed only in binaries (`cli`, `viewer`, `xtask`) and tests.
- User files are untrusted input. Malformed data must return an error, never panic. Index math on file data uses checked conversions.
- `unwrap()` and `expect()` appear only in tests, or with a comment proving the invariant that makes them safe.

**Safety.** `unsafe_code` is forbidden workspace-wide. If a crate ever truly needs `unsafe`, stop and ask; the exception and a `// SAFETY:` comment must be justified in `docs/decisions.md`.

**Docs.** Every public item has rustdoc. Public functions with non-obvious use get a short example. Crate-level docs say what the crate is for and what it must not depend on.

**Numeric types.**
- Positions and field values are `f64` on the CPU. Conversion to `f32` happens only when building GPU buffers, relative to the dataset center.
- Connectivity uses `u32` indices, converted with checks on load (see D-013 in `docs/decisions.md`).
- Array components are a small enum or `u8`, never a bare `usize` with magic values.

**Parallelism.** Use `rayon` only behind the crate's `parallel` feature (on by default for native, off for wasm32). Every algorithm must also work single-threaded. Parallel output must be deterministic: merge per-chunk results in chunk order, never in completion order. Image regression tests depend on this.

**State and logging.** No global mutable state and no lazy singletons. Libraries log through the `log` facade; only binaries install a logger.

**Structure.** Prefer plain functions over trait hierarchies: filters are `fn(&Input, &Params) -> Result<Output, Error>`. Keep modules small; a file above roughly 600 lines is a hint to split.

**Tests.** Unit tests sit in `#[cfg(test)] mod tests` beside the code. Cross-crate and file-based tests go in the crate's `tests/` folder. See `testing.md`.

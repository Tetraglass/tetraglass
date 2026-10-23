---
name: verify
description: Runs the full local verification suite (rustfmt, clippy with warnings as errors, all tests, a wasm32 build of the web-compatible crates, and cargo-deny license checks) and fixes what fails. Use after any code change, before marking a task done, before committing, and whenever the user asks to check, test, lint, validate or "make it green".
---

# Verify

## Run

```
cargo xtask verify
```

It runs, in order, stopping at the first failure:

1. `cargo fmt --all --check`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo test --workspace`
4. `cargo check --target wasm32-unknown-unknown` for every crate ending in `-core`, `-io`, `-filters` or `-scene`
5. `cargo deny check`, if cargo-deny is installed

If the `xtask` crate does not exist yet, run the same commands by hand.

## Fix

| Failure | What to do |
| --- | --- |
| Format | Run `cargo xtask fmt`, then rerun verify. |
| Clippy | Fix the code. An `#[allow(...)]` needs a comment explaining why the lint is wrong here. |
| Tests | Read the failure, fix the cause. Never delete, ignore or loosen a test to pass; ask if a test itself looks wrong. |
| wasm32 | Follow `.agents/rules/web-compat.md`: usually a filesystem, thread or C dependency leaked into a web-compatible crate. If the error is "can't find crate for std", run `rustup target add wasm32-unknown-unknown`. |
| cargo-deny | A license or source outside `.agents/rules/dependencies.md`. Replace the dependency or ask the user; do not widen the allowlist on your own. |
| Image regression | Use `/render-check`. |

## Report

Summarize in a few lines: which steps ran, what failed and how it was fixed, and whether anything was skipped (for example cargo-deny missing). A task is not done while any step fails or is skipped without the user knowing.

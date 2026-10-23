---
trigger: glob
globs: "crates/*-core/**, crates/*-io/**, crates/*-filters/**, crates/*-scene/**"
description: "Keeps the platform-independent crates compiling for wasm32-unknown-unknown and safe to run in a browser."
---

# Web compatibility rules

These crates must build with `cargo check --target wasm32-unknown-unknown`. `/verify` checks this.

- **No filesystem in core logic.** Readers take `&[u8]` or `Read + Seek`. Helpers that open paths live behind `#[cfg(not(target_arch = "wasm32"))]`.
- **No threads by default.** No `std::thread::spawn`. `rayon` only behind the `parallel` feature, which is off on wasm32.
- **No blocking.** Never block on a future. Keep these crates synchronous; async belongs in the app layer.
- **No C or C++.** No `build.rs` that compiles native code; no `-sys` crates. If a format seems to need one (for example HDF5), stop and ask.
- **Time.** `std::time::Instant` panics on wasm32. Avoid timing in these crates; if needed, use a wasm-safe crate behind a cfg.
- **Memory.** Browsers give wasm32 at most 4 GiB. Share large arrays through `Arc` instead of cloning; avoid holding two full copies of a mesh.

---
trigger: glob
globs: "**/Cargo.toml, deny.toml"
description: "Dependency and license policy: allowed licenses, workspace dependencies, wgpu and egui version coupling."
---

# Dependency and license policy

**License allowlist for Rust dependencies:** MIT, Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, Unicode-3.0, Unicode-DFS-2016, BSL-1.0, CC0-1.0, MPL-2.0. Fonts bundled by egui may also need OFL-1.1 and Ubuntu-font-1.0. Anything else (especially GPL, LGPL or AGPL) needs the human's approval first. `cargo deny check` enforces this.

**Adding a dependency.**
1. Check it is maintained, widely used and has a license on the list.
2. Check it builds for wasm32 if the target crate must (see `web-compat.md`).
3. Add the version once in the root `[workspace.dependencies]`; crates use `name.workspace = true`.
4. Note in `docs/progress.md` why it was added.

Prefer few dependencies. Small utilities (a 30-line function) are written in-house.

**Version coupling.** `egui`, `egui-wgpu`, `eframe` and `wgpu` move together. wgpu's version is whatever egui-wgpu depends on, never newer. Use `/upgrade-wgpu-egui`.

**Workspace hygiene.** Crate manifests inherit `version`, `edition`, `license`, `repository`, `rust-version` and `publish` from the workspace, and set `[lints] workspace = true`.

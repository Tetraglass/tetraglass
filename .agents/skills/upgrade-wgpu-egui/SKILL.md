---
name: upgrade-wgpu-egui
description: Lockstep upgrade procedure for egui, egui-wgpu, eframe and wgpu, keeping exactly one wgpu version in the dependency tree and catching rendering regressions. Use when the user asks to update egui or wgpu, when a new egui release is out, or when cargo tree shows more than one wgpu version.
---

# Upgrade wgpu and egui together

egui-wgpu decides the wgpu version. The project follows it and never runs ahead.

## Steps

1. **Read before changing.** Check the latest egui release and its changelog, and the changelogs of every wgpu version between the current and the new one. Note breaking changes that touch: device and queue creation, render pass and texture APIs, WGSL validation, limits and features, and egui-wgpu's callback or texture registration API.
2. **Upgrade together.** In the root `Cargo.toml`, bump `egui`, `egui-wgpu`, `eframe` (if present) to the new release, then set `wgpu` to the version egui-wgpu now uses:

```
cargo update -p egui -p egui-wgpu
cargo tree -i wgpu
```

`cargo tree -i wgpu` must show exactly one version.

3. **Fix compile errors only where wgpu lives:** the render crate, plus texture bridging in the egui crate. If another crate breaks, the boundary was violated; fix the boundary, not just the error.
4. **Check MSRV.** If the new egui raises its minimum Rust version, raise `rust-version` in `[workspace.package]` to match.
5. **Verify.** Run `/verify`, then `/render-check` on the whole gallery. Look closely at sRGB handling, MSAA resolve and depth formats; these change quietly between wgpu versions.
6. **Web.** Build the viewer for wasm32 (once it exists) and open it in a WebGPU browser.
7. **Record.** In `docs/progress.md`: old and new versions, breaking changes handled, and any baseline images that changed with the reason.

Do the upgrade on its own branch and in its own commit (`chore(deps): upgrade egui to X and wgpu to Y`), never mixed with feature work.

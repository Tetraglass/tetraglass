---
trigger: glob
globs: "crates/*-render/**, crates/*-egui/**, crates/*-viewer/**, **/*.wgsl"
description: "Rules for wgpu rendering code, WGSL shaders and the egui viewport widget."
---

# Rendering rules

**Ownership.** The renderer receives `&wgpu::Device`, `&wgpu::Queue` and a target format from its host. It never calls `Instance::new`, requests an adapter or creates a surface, except inside the `headless` module, which exists for the CLI and tests.

**Targets.** Rendering always goes into offscreen color and depth textures owned by the caller (the egui widget or the headless helper). The egui widget renders in egui-wgpu's `prepare` step and shows the result as a registered texture. Do not draw scene geometry inside egui's render pass.

**Where wgpu lives.** All pipelines, buffers and shaders are in the render crate. The egui crate converts input into camera changes and registers textures; it contains no pipelines.

**Version.** wgpu's version must match the one egui-wgpu uses. `cargo tree -i wgpu` must show exactly one version. Upgrades go through `/upgrade-wgpu-egui`.

**Portability.**
- Use only features and limits that WebGPU guarantees by default. A native-only feature (for example ray queries) must be behind a cargo feature and a runtime capability check, with a working fallback.
- No WebGL2-specific workarounds; the web target is WebGPU only (D-007).

**Shaders.** WGSL files live in `src/shaders/` and are loaded with `include_str!`. No runtime shader loading from disk. Keep one shader per pipeline family, with shared code in a common include assembled at build time by simple string concatenation.

**Data flow.**
- Scalars upload once as an f32 vertex attribute. Colormaps are 256-entry 1D textures. Changing colormap, range or log scale updates a uniform only.
- Cell data renders flat through a per-triangle cell index into a storage buffer.
- Positions are f32 relative to the dataset center; the camera works in the same space.

**Color.** Shade in linear space; write to an sRGB target; PNG export is sRGB. Default colormaps: viridis and cividis (sequential), cool-warm (diverging).

**Annotations.** Colorbars, axes triad and text are drawn by the renderer, not by egui, so headless PNGs contain them.

**Determinism.** No time-based or random values in shaders or draw order. Tests use a fixed MSAA sample count and fixed image size. Any change that alters pixels goes through `/render-check`.

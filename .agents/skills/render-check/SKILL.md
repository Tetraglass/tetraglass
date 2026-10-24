---
name: render-check
description: Visual regression and image inspection for anything that changes rendered pixels. It renders the gallery and test scenes headlessly, compares them with the baseline images, looks at the results, and updates baselines only for intended changes. Use for every change in the render, egui or viewer crates, shaders, colormaps, camera, lighting, annotations, or any filter whose output is shown in the gallery.
---

# Render check

## 1. Render

- **Before the CLI exists (M0):** run the render crate's image tests, `cargo test -p tetraglass-render`. They render test meshes through the headless helper and write PNGs to `target/render-tests/`.
- **From M1 on:** render each gallery scene with `cargo run -p tetraglass-cli -- render gallery/<name>.json -o target/gallery/<name>.png`, or with the xtask gallery task once it exists.

Fixed image size, fixed MSAA sample count and a fixed camera are part of every test scene. Without them, comparisons are meaningless.

## 2. Compare

Compare each image with `tests/baselines/<name>.png` using the policy in `.agents/rules/testing.md` (`D-015`): a pixel differs if any channel differs by more than 2 (of 255), and a test fails above 0.75% differing pixels. On failure, write a diff image to `target/render-diffs/<name>.png` that highlights differing pixels.

## 3. Look

Open the new image and, if it failed, the baseline and the diff. Check for:

- missing or extra geometry, holes, cracks between cells;
- inverted normals (dark faces), z-fighting, clipped geometry near the camera;
- wrong colors: colormap reversed, range wrong, cell data smeared like point data;
- the colorbar, its labels and the axes triad present and readable;
- transparency order artifacts.

Describe what you see in one or two sentences per image. Pixels passing the threshold is not enough if something is visibly wrong.

## 4. Decide

- **Unintended change:** fix the code, rerun from step 1.
- **Intended change:** replace the baseline, and say in `docs/progress.md` and the commit message which images changed and why.
- Baselines are authoritative only when produced on Mesa lavapipe, the driver CI uses. On a real GPU, compare locally, but leave baseline replacement to CI artifacts or a lavapipe run.

Never replace a baseline just to make a failing test pass without that explanation.

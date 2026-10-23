# Agent guide

This repository builds **Tetraglass**, a Rust-native scientific visualization library for FEM/CFD
results on unstructured meshes. It is built on wgpu, embeds in Rust and egui apps
first, and can be driven by coding agents through JSON scene files and a CLI.
Long term it aims at VTK-class capability; the first year targets a narrow,
polished core.

- **Project name:** Tetraglass
- **Repository:** https://github.com/Tetraglass/tetraglass
- **Crate prefix:** `tetraglass` (crates are named `tetraglass-core`, `tetraglass-io`, ...; the umbrella crate and the CLI binary are both `tetraglass`)
- **Developer:** one person, evenings and weekends. Small, finished steps beat big, half-done ones.

## Where things are

| What | Where | Read it when |
| --- | --- | --- |
| Full design and its reasons | @docs/design-plan.md | Before any design-level change |
| Binding decisions | @docs/decisions.md | Before proposing anything that might contradict one |
| Milestones, tasks, gates | @docs/roadmap.md | At the start of every session |
| What has been done | @docs/progress.md | At the start of every session; append after every task |
| Scene file format (draft) | @docs/scene-format.md | When touching the scene crate, CLI or gallery |
| Detailed rules | `.agents/rules/` | Loaded automatically by file type |
| Repeatable procedures | `.agents/skills/` | Invoke with `/<skill-name>` |

## Skills (slash commands)

- `/bootstrap-workspace`: one-time setup of the Cargo workspace, xtask, CI and license policy.
- `/next-task`: the default development loop. Use it whenever the user says "continue" or "next".
- `/verify`: format, lint, test, wasm32 build and license checks. Run before calling anything done.
- `/add-filter`, `/add-reader`: procedures for the two most common kinds of feature work.
- `/author-scene`: how to write, validate and render scene files (also for gallery entries).
- `/render-check`: visual regression and image inspection for rendering changes.
- `/make-test-data`: generates the small meshes the tests depend on.
- `/milestone-gate`: checks a milestone's gate and runs the release checklist.
- `/upgrade-wgpu-egui`: the lockstep upgrade procedure for wgpu, egui and eframe.
- `/blender-bridge`: Blender export and headless Cycles rendering (milestone M3).

## Invariants

These hold everywhere. The rules in `.agents/rules/` explain each in detail.

1. **Dependency direction:** `core` <- `io`, `filters` <- `scene` <- `render` <- `egui` <- `viewer`; `cli` and `export` sit on top. `core` knows nothing about GPUs, files, egui or Blender.
2. **The renderer never owns a window.** It takes the host's wgpu `Device` and `Queue` and draws into textures it is given. Only a `headless` helper may create its own device (for the CLI and tests).
3. **All wgpu code lives in the render crate.** The egui crate only bridges textures and input.
4. **Readers take bytes** (`&[u8]` or `Read + Seek`). Path helpers exist only behind `cfg(not(target_arch = "wasm32"))`.
5. **`core`, `io`, `filters` and `scene` compile for `wasm32-unknown-unknown`** with no C dependencies.
6. **Filters work only on the linearized tetrahedral view.** Higher-order cells and polyhedra are stored natively but never special-cased inside a filter.
7. **The scene model is plain serde data.** Any scene change updates the JSON Schema, `docs/scene-format.md` and at least one gallery scene.
8. **Licensing boundary:** Rust crates are `MIT OR Apache-2.0`. Anything that imports Blender's `bpy` lives in `blender/` under GPL-3.0-or-later. Rust code never embeds those scripts.
9. **Web means WebGPU only.** No WebGL2 code paths.
10. **Precision:** f64 on the CPU; f32 relative to the dataset center on the GPU.

## Definition of done

A task is done when all of these are true:

- `/verify` passes with no new warnings.
- New behavior has tests; every bug fix has a regression test.
- Public items have rustdoc; changed user-facing behavior is reflected in `docs/`.
- The task is ticked in `docs/roadmap.md` and an entry is appended to `docs/progress.md`.
- Changes are committed locally with a Conventional Commits message (for example `feat(filters): add slice filter`).

## Stop and ask the human when

- A change would contradict `docs/decisions.md`.
- A new dependency has a license outside the policy in `.agents/rules/dependencies.md`.
- A public API or the scene schema would break after release 0.1.
- A test or image baseline would have to be weakened or replaced to make it pass.
- Work from a later milestone seems necessary before the current gate passes.

## Do not

- Bump wgpu on its own; it moves in lockstep with egui-wgpu (`/upgrade-wgpu-egui`).
- Silence clippy with `allow` attributes unless a comment explains why.
- Commit generated large files; test data stays small and reproducible (`/make-test-data`).
- Start the next milestone's tasks without being asked.
- Push, force-push, rewrite git history or change remotes without the human's explicit go-ahead. Local commits are fine.

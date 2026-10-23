# Tetraglass

A Rust-native library for visualizing FEM/CFD results on unstructured meshes, built on wgpu. It embeds in Rust and egui apps, renders headlessly to PNG, runs in the browser via WebGPU, and can be driven by coding agents through JSON scene files and a CLI. Final-quality images come from Blender's Cycles, driven automatically.

This repository currently holds the **plan and the agent setup**; code starts with milestone M0.

## Getting started with Google Antigravity

1. In the `Tetraglass` organization on GitHub, create an empty repository named `tetraglass` (no README, license or .gitignore; this kit has them).
2. Set up git locally and push the kit:

   ```
   git init -b main
   unzip ~/Downloads/tetraglass-starter-kit.zip   # wherever you saved it; keep the zip outside the repo
   git add -A
   git commit -m "docs: add design plan and agent setup"
   git remote add origin git@github.com:Tetraglass/tetraglass.git
   git push -u origin main
   ```

3. Open the folder in Antigravity. In the agent panel, open **Customizations** and check that the rules (`.agents/rules/`) and skills (`.agents/skills/`) are listed. `AGENTS.md` is loaded automatically.
4. Run `/bootstrap-workspace`. The agent checks git, confirms the `tetraglass-*` crate names are still free, creates and verifies the Cargo workspace, fills in the repository URL from `origin`, and commits. It asks before pushing; the push starts the first CI run.
5. From then on, run `/next-task` (or just say "continue") to work through `docs/roadmap.md` one task at a time.

The procedures are **Agent Skills** (`.agents/skills/<name>/SKILL.md`), invoked as `/<name>`. Antigravity retired its older single-file workflows on 2026-10-19, so none are used here.

## Prerequisites

| Tool | Needed from | Purpose |
| --- | --- | --- |
| Rust (stable, via rustup) | M0 | Everything |
| `wasm32-unknown-unknown` target | M0 | Web-compatibility checks |
| `cargo-deny` | M0 | License and dependency checks |
| Python 3 with `vtk`, `meshio` and `gmsh` | M0 | Generating test meshes |
| Mesa lavapipe (optional locally) | M0 | Reproducing CI's render baselines |
| Blender | M3 | Ray-traced output through Cycles |

## What is in here

```
AGENTS.md                  Always-on agent context: invariants, definition of done
docs/
  design-plan.md           Full design and its reasons
  decisions.md             Binding decisions and open questions
  roadmap.md               Milestones M0 to M5 with numbered tasks and gates
  scene-format.md          Draft JSON scene format
  progress.md              Running log, appended after every task
.agents/
  rules/                   Constraints loaded by file type (Rust, rendering, web, scene, tests, dependencies, Blender)
  skills/                  Procedures: bootstrap-workspace, next-task, verify, add-filter, add-reader,
                           author-scene, render-check, make-test-data, milestone-gate,
                           upgrade-wgpu-egui, blender-bridge
blender/                   GPL-3.0-or-later Blender scripts (empty until M3)
LICENSE-MIT, LICENSE-APACHE
```

## License

The library is dual-licensed under MIT or Apache-2.0, at your option. The contents of `blender/` are licensed separately under GPL-3.0-or-later, because they use Blender's Python API; see `blender/README.md`.

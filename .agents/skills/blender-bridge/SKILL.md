---
name: blender-bridge
description: Implements and maintains the Blender integration for milestone M3. It covers the renderer-neutral bundle format, the export crate, the GPL importer and render scripts in blender/, and the CLI's render --engine cycles that drives a headless Blender. Use for any task about Blender, Cycles, ray-traced output, glTF or bundle export, .blend files, or the Blender add-on.
---

# Blender bridge

Read `.agents/rules/blender-gpl.md` first. The licensing boundary shapes every step.

## Pieces and where they live

| Piece | Location | License |
| --- | --- | --- |
| Bundle format spec | `docs/bundle-format.md` | Project docs |
| Bundle and glTF writers | `crates/tetraglass-export` | MIT OR Apache-2.0 |
| `render --engine cycles` | `crates/tetraglass-cli` | MIT OR Apache-2.0 |
| Importer add-on and render driver | `blender/` | GPL-3.0-or-later |

## 1. Bundle format (write the spec first)

`docs/bundle-format.md` defines a folder with:

- `manifest.json`: format version, units, camera (position, focal point, up, field of view or ortho scale), lights, background, image size, time steps, and per object its mesh files, colormap (as 256 RGB values plus range and scale) and material hints (opacity, edge display).
- One little-endian binary file per object and time step: positions (f32 relative to the dataset center, with the center stored in the manifest), triangle indices (u32), normals, and the scalar used for coloring, normalized to [0, 1].

The format carries data only: no code and no Blender-specific structures. Version it like the scene schema.

## 2. Export crate

Turns an evaluated scene into the bundle, and separately into glTF (baked vertex colors, normalized scalar in a second texture coordinate, one .glb per time step). Test both against small gallery scenes; check a round trip of the bundle with a reader written in the test.

## 3. Blender scripts (`blender/`)

- `blender/importer.py`: builds Blender objects from a bundle: meshes with the scalar as an attribute, a colormap shader node fed by it, camera, lights, and one frame per time step.
- `blender/render.py`: entry point for `blender --background --factory-startup --python blender/render.py -- --bundle <dir> --out <png or frame pattern> --samples <n> --device <cpu|optix|cuda|hip|metal> [--save-blend <path>]`. It parses arguments after `--`, calls the importer, configures Cycles, renders, and prints progress lines prefixed with `PROGRESS ` so the CLI can stream them.
- Also packaged as an installable add-on so users can import bundles interactively.

## 4. CLI integration

- Locate Blender: scene or config `blender_path`, then an environment variable, then `PATH`. Report clearly when it is missing; never download it.
- Locate the scripts at runtime (installed add-on, config path or a path relative to the CLI binary). Never embed them.
- Write the bundle to a temporary folder, run Blender as a subprocess with a timeout, stream `PROGRESS` lines, and map a non-zero exit code to an error that includes the last lines of Blender's output.
- Print the same JSON summary as the rasterizer, plus the Blender version used.

## 5. Test

- Rust side: bundle and glTF writers are covered by unit tests without Blender.
- Blender side: an optional CI job installs a supported Blender, renders one gallery scene with few samples, and checks the image exists and is not blank. Run it for each version listed in `blender/README.md`.

## Later (not M3)

A persistent Blender worker with a socket server, and a live link from the viewer, reuse the same bundle format. Do not start them without a roadmap task.

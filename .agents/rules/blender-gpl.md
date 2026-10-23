---
trigger: glob
globs: "blender/**, crates/*-export/**"
description: "Licensing boundary and conventions for Blender integration code (GPL scripts in blender/, data-only interface from Rust)."
---

# Blender boundary rules

Blender's FAQ requires published scripts that call its Python API to be GPL. Software that runs outside Blender, produces data for it and executes Blender may use its own license. This project relies on that split (D-011).

- **Everything in `blender/` is GPL-3.0-or-later.** Each Python file starts with `# SPDX-License-Identifier: GPL-3.0-or-later`.
- **No Rust in `blender/`, and no `bpy` outside it.** Do not copy code between `blender/` and `crates/`.
- **Never embed the scripts in Rust binaries** (no `include_str!` of anything in `blender/`). The CLI locates the scripts at runtime: a config setting, an environment variable, or the installed add-on.
- **Never bundle or download Blender.** The CLI uses an installed Blender found through config, an environment variable or `PATH`.
- **The interface is data and command-line arguments only:** the bundle format documented in `docs/bundle-format.md` (written in M3) plus arguments passed after `--`.
- **Python style:** standard library and `numpy` only (both ship with Blender). Scripts must run with `blender --background --factory-startup`.
- **Supported Blender versions** are listed in `blender/README.md`; CI renders one test scene per supported version when Blender jobs exist.

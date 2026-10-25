# Blender integration (GPL-3.0-or-later)

Everything in this folder is licensed under the GNU General Public License, version 3 or later (see `LICENSE` here), not under the MIT/Apache license of the rest of the repository.

**Why.** Blender's FAQ requires published scripts that use Blender's Python API to be GPL. Software that runs outside Blender, produces data for it and executes Blender may keep its own license (`D-011`). The Rust crates therefore talk to these scripts only through data files (the bundle format, `docs/bundle-format.md`) and command-line arguments, and never embed them.

## Contents

- `importer.py`: builds a Blender scene from a bundle (`manifest.json` + `meshes/*.tgbin`) using only the Python standard library and `numpy` — meshes with `tetraglass_scalar` and `tetraglass_raw_scalar` attributes, a 256-entry LUT texture + 32-stop ColorRamp shader node tree, original element wireframe edge curves, camera, sun lights, solid/gradient background, and per-step timeline frames.
- `render.py`: headless Cycles entry point, run as:
  ```sh
  blender --background --factory-startup --python blender/render.py -- \
    --bundle <bundle_dir> \
    --out <output.png_or_pattern> \
    --samples 128 \
    --device auto \
    [--step 0 | --all-steps] \
    [--save-blend scene.blend]
  ```
- `__init__.py` & `blender_manifest.toml`: add-on / extension packaging so bundles can also be imported interactively via `File > Import > Tetraglass Bundle (manifest.json)`.

## Installing as an interactive Blender add-on

Zip the `blender/` directory (or run `zip -r tetraglass_blender_addon.zip blender/`), open Blender (`Edit > Preferences > Add-ons` or `Get Extensions > Install from Disk...`), select the zip file, and enable **Tetraglass Bundle Importer**.

## Supported Blender versions

- **Blender 4.2 LTS** (`4.2+`)
- **Blender 5.x LTS** (verified against `Blender 5.2 LTS`)

Every Python file here starts with `# SPDX-License-Identifier: GPL-3.0-or-later`.

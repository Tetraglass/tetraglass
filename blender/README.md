# Blender integration (GPL-3.0-or-later)

Everything in this folder is licensed under the GNU General Public License, version 3 or later (see `LICENSE` here), not under the MIT/Apache license of the rest of the repository.

**Why.** Blender's FAQ requires published scripts that use Blender's Python API to be GPL. Software that runs outside Blender, produces data for it and executes Blender may keep its own license. The Rust crates therefore talk to these scripts only through data files (the bundle format, `docs/bundle-format.md`) and command-line arguments, and never embed them.

**Contents (from milestone M3):**

- `importer.py`: builds a Blender scene from a bundle (meshes, attributes, colormap shader, camera, lights, time steps).
- `render.py`: headless entry point, run as `blender --background --factory-startup --python render.py -- <args>`.
- Add-on packaging so bundles can also be imported interactively.

**Supported Blender versions:** to be decided in M3 (see the open question in `docs/decisions.md`).

Every Python file here starts with `# SPDX-License-Identifier: GPL-3.0-or-later`.

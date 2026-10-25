# SPDX-License-Identifier: GPL-3.0-or-later
"""Tetraglass Bundle Importer add-on for Blender 4.2 LTS and 5.x LTS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

bl_info = {
    "name": "Tetraglass Bundle Importer",
    "author": "Tetraglass Contributors",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "File > Import > Tetraglass Bundle (manifest.json)",
    "description": "Import Tetraglass FEM/CFD visualization bundles (manifest.json + .tgbin)",
    "category": "Import-Export",
    "doc_url": "https://github.com/Tetraglass/tetraglass",
    "license": "GPL-3.0-or-later",
}

from . import importer  # noqa: E402
from .importer import import_bundle, load_bundle_manifest, read_tgbin  # noqa: E402,F401

try:
    import bpy
    from bpy.props import BoolProperty, StringProperty
    from bpy_extras.io_utils import ImportHelper

    class IMPORT_SCENE_OT_tetraglass_bundle(bpy.types.Operator, ImportHelper):
        """Import a Tetraglass bundle directory or manifest.json"""

        bl_idname = "import_scene.tetraglass_bundle"
        bl_label = "Import Tetraglass Bundle"
        bl_options = {"REGISTER", "UNDO"}

        filename_ext = ".json"
        filter_glob: StringProperty(
            default="manifest.json;*.json",
            options={"HIDDEN"},
            maxlen=255,
        )
        clear_existing: BoolProperty(
            name="Replace Existing Scene Objects",
            description="Remove default or existing objects before importing the bundle",
            default=False,
        )

        def execute(self, context: Any) -> set[str]:
            target = Path(self.filepath)
            if target.is_file():
                bundle_dir = target.parent
            else:
                bundle_dir = target
            try:
                summary = importer.import_bundle(
                    bundle_dir,
                    context=context,
                    clear_existing=bool(self.clear_existing),
                )
            except Exception as exc:
                self.report({"ERROR"}, f"Tetraglass bundle import failed: {exc}")
                return {"CANCELLED"}

            self.report(
                {"INFO"},
                f"Imported {len(summary['objects'])} object(s) across {summary['num_steps']} step(s)",
            )
            return {"FINISHED"}

    def _menu_func_import(self: Any, _context: Any) -> None:
        self.layout.operator(
            IMPORT_SCENE_OT_tetraglass_bundle.bl_idname,
            text="Tetraglass Bundle (manifest.json)",
        )

    def register() -> None:
        bpy.utils.register_class(IMPORT_SCENE_OT_tetraglass_bundle)
        bpy.types.TOPBAR_MT_file_import.append(_menu_func_import)

    def unregister() -> None:
        bpy.types.TOPBAR_MT_file_import.remove(_menu_func_import)
        bpy.utils.unregister_class(IMPORT_SCENE_OT_tetraglass_bundle)

except ImportError:
    # Imported outside Blender (e.g., standalone Python unit tests for .tgbin / manifest I/O)
    def register() -> None:
        pass

    def unregister() -> None:
        pass

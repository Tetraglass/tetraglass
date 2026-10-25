# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless Blender Cycles render driver for Tetraglass bundles (`M3.6` / `M3.7`).

Usage:
    blender --background --factory-startup --python blender/render.py -- \\
        --bundle <bundle_dir> \\
        --out <output.png_or_pattern> \\
        [--samples 128] \\
        [--device auto|cpu|gpu|metal|optix|cuda|hip] \\
        [--step 0 | --all-steps] \\
        [--width <px>] [--height <px>] \\
        [--save-blend <scene.blend>]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

# Ensure sibling `importer.py` can be imported when invoked via `blender --python blender/render.py`
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import importer  # noqa: E402


def emit_progress(**fields: Any) -> None:
    """Prints a machine-readable `PROGRESS key=value ...` line and flushes stdout."""
    parts = ["PROGRESS"]
    for key, val in fields.items():
        text = str(val).replace("\n", " ").strip()
        if " " in text or "\t" in text or text == "":
            parts.append(f'{key}="{text}"')
        else:
            parts.append(f"{key}={text}")
    print(" ".join(parts), flush=True)


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Extracts arguments after `--` from `sys.argv` (or `argv`) and parses them."""
    raw = list(sys.argv if argv is None else argv)
    if "--" in raw:
        script_args = raw[raw.index("--") + 1 :]
    else:
        script_args = []

    parser = argparse.ArgumentParser(
        prog="blender/render.py",
        description="Render a Tetraglass bundle headlessly with Blender Cycles.",
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="Path to the Tetraglass bundle directory or its manifest.json.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG file path (or frame pattern such as frame_%%04d.png when --all-steps is set).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Number of Cycles path-tracing samples per pixel (default: 128).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Cycles compute device: auto, cpu, gpu, metal, optix, cuda, or hip (default: auto).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Zero-based time step index to render (default: 0).",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Render every time step in the bundle.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional override for output image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional override for output image height in pixels.",
    )
    parser.add_argument(
        "--save-blend",
        default=None,
        help="Optional path to save the constructed .blend file before rendering.",
    )
    return parser.parse_args(script_args)


def configure_cycles_device(scene: Any, requested_device: str) -> str:
    """Configures Cycles compute backend and returns the active backend name."""
    import bpy

    scene.render.engine = "CYCLES"
    req = requested_device.strip().lower()
    if req == "cpu":
        scene.cycles.device = "CPU"
        return "CPU"

    prefs = bpy.context.preferences
    cycles_addon = prefs.addons.get("cycles")
    if cycles_addon is None:
        scene.cycles.device = "CPU"
        return "CPU"

    cprefs = cycles_addon.preferences
    if req in ("auto", "gpu"):
        candidate_backends = ["OPTIX", "METAL", "CUDA", "HIP", "ONEAPI"]
    else:
        candidate_backends = [req.upper()]

    for backend in candidate_backends:
        try:
            cprefs.compute_device_type = backend
            if hasattr(cprefs, "refresh_devices"):
                cprefs.refresh_devices()
            elif hasattr(cprefs, "get_devices"):
                cprefs.get_devices()

            gpu_devices = [
                d for d in cprefs.devices if getattr(d, "type", "CPU") != "CPU"
            ]
            if gpu_devices:
                for d in cprefs.devices:
                    d.use = getattr(d, "type", "CPU") != "CPU"
                scene.cycles.device = "GPU"
                return backend
        except Exception:
            continue

    scene.cycles.device = "CPU"
    return "CPU"


def format_frame_output_path(pattern: str, step_idx: int, all_steps: bool) -> Path:
    """Resolves the output PNG path for `step_idx`."""
    if not all_steps:
        return Path(pattern).expanduser().resolve()

    if "%" in pattern and re.search(r"%0?\d*d", pattern):
        formatted = pattern % step_idx
        return Path(formatted).expanduser().resolve()

    if "{step" in pattern or "{frame" in pattern:
        formatted = pattern.format(step=step_idx, frame=step_idx)
        return Path(formatted).expanduser().resolve()

    base = Path(pattern).expanduser().resolve()
    stem = base.stem
    suffix = base.suffix or ".png"
    return base.with_name(f"{stem}_{step_idx:04d}{suffix}")


def run_headless_render(argv: list[str] | None = None) -> dict[str, Any]:
    """Executes the full import, Cycles setup, optional `.blend` save, and render."""
    import bpy

    args = parse_cli_args(argv)
    bundle_path = Path(args.bundle).expanduser().resolve()
    emit_progress(stage="import", status="starting", bundle=bundle_path)

    summary = importer.import_bundle(bundle_path, clear_existing=True)
    num_steps = int(summary["num_steps"])
    emit_progress(
        stage="import",
        status="done",
        objects=len(summary["objects"]),
        steps=num_steps,
    )

    scene = bpy.context.scene
    if args.width is not None and args.width > 0:
        scene.render.resolution_x = int(args.width)
    if args.height is not None and args.height > 0:
        scene.render.resolution_y = int(args.height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    active_backend = configure_cycles_device(scene, args.device)
    samples = max(1, int(args.samples))
    scene.cycles.samples = samples
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = True
    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = False

    emit_progress(
        stage="cycles",
        device=active_backend,
        samples=samples,
        resolution=f"{scene.render.resolution_x}x{scene.render.resolution_y}",
    )

    if args.save_blend:
        blend_path = Path(args.save_blend).expanduser().resolve()
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        emit_progress(stage="save_blend", path=blend_path)

    def _on_render_stats(stats_arg: Any = None, *_extra: Any) -> None:
        info = stats_arg if isinstance(stats_arg, str) else str(stats_arg)
        if info:
            emit_progress(stage="render_stats", info=info)

    stats_handlers = bpy.app.handlers.render_stats
    stats_handlers.append(_on_render_stats)

    rendered_files: list[str] = []
    try:
        if args.all_steps:
            steps_to_render = list(range(num_steps))
        else:
            chosen_step = 0 if args.step is None else max(0, min(num_steps - 1, int(args.step)))
            steps_to_render = [chosen_step]

        total_frames = len(steps_to_render)
        for idx, step_idx in enumerate(steps_to_render):
            out_path = format_frame_output_path(
                args.out, step_idx, all_steps=args.all_steps and num_steps > 1
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            scene.frame_set(step_idx)
            if num_steps > 1:
                for obj in scene.objects:
                    if "tetraglass_step" in obj:
                        hidden = int(obj["tetraglass_step"]) != step_idx
                        obj.hide_render = hidden
                        obj.hide_viewport = hidden
            scene.render.filepath = str(out_path)

            emit_progress(
                stage="render",
                frame=step_idx,
                index=idx + 1,
                total_frames=total_frames,
                status="starting",
            )
            bpy.ops.render.render(write_still=True)
            rendered_files.append(str(out_path))
            emit_progress(
                stage="render",
                frame=step_idx,
                index=idx + 1,
                total_frames=total_frames,
                status="done",
                output=out_path,
            )
    finally:
        if _on_render_stats in stats_handlers:
            stats_handlers.remove(_on_render_stats)

    blender_ver = getattr(bpy.app, "version_string", "unknown")
    emit_progress(
        stage="complete",
        frames=len(rendered_files),
        blender_version=blender_ver,
    )
    return {
        "files": rendered_files,
        "device": active_backend,
        "samples": samples,
        "blender_version": blender_ver,
    }


if __name__ == "__main__":
    run_headless_render()

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
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
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
        description="Render a Tetraglass bundle headlessly with Blender Cycles or run a persistent socket worker.",
    )
    parser.add_argument(
        "--serve",
        default=None,
        help="Start a persistent TCP socket worker bound to HOST:PORT (for example 127.0.0.1:0).",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Path to the Tetraglass bundle directory or its manifest.json.",
    )
    parser.add_argument(
        "--out",
        default=None,
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
        "--from-step",
        type=int,
        default=None,
        help="Inclusive start time-step index to render.",
    )
    parser.add_argument(
        "--to-step",
        type=int,
        default=None,
        help="Inclusive end time-step index to render.",
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
    args = parser.parse_args(script_args)
    if not args.serve and (not args.bundle or not args.out):
        parser.error("--bundle and --out are required unless --serve is specified")
    return args


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


def _execute_cycles_render_on_active_scene(
    summary: dict[str, Any],
    out_pattern: str,
    samples: int = 128,
    device: str = "auto",
    step: int | None = None,
    all_steps: bool = False,
    from_step: int | None = None,
    to_step: int | None = None,
    width: int | None = None,
    height: int | None = None,
    save_blend: str | None = None,
) -> dict[str, Any]:
    """Configures Cycles and renders the requested frame(s) from the already-imported Blender scene."""
    import bpy

    scene = bpy.context.scene
    if width is not None and int(width) > 0:
        scene.render.resolution_x = int(width)
    if height is not None and int(height) > 0:
        scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    active_backend = configure_cycles_device(scene, device)
    samples_clamped = max(1, int(samples))
    scene.cycles.samples = samples_clamped
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = True
    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = False

    emit_progress(
        stage="cycles",
        device=active_backend,
        samples=samples_clamped,
        resolution=f"{scene.render.resolution_x}x{scene.render.resolution_y}",
    )

    if save_blend:
        blend_path = Path(save_blend).expanduser().resolve()
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        emit_progress(stage="save_blend", path=blend_path)

    def _on_render_stats(stats_arg: Any = None, *_extra: Any) -> None:
        info = stats_arg if isinstance(stats_arg, str) else str(stats_arg)
        if info:
            emit_progress(stage="render_stats", info=info)

    stats_handlers = bpy.app.handlers.render_stats
    stats_handlers.append(_on_render_stats)

    num_steps = int(summary.get("num_steps", 1))
    rendered_files: list[str] = []
    try:
        available_steps = [
            int(s) for s in summary.get("steps", list(range(num_steps)))
        ] or [0]
        is_range_request = (
            bool(all_steps) or from_step is not None or to_step is not None
        )
        if is_range_request:
            lo = (
                min(available_steps)
                if from_step is None
                else int(from_step)
            )
            hi = (
                max(available_steps)
                if to_step is None
                else int(to_step)
            )
            steps_to_render = [s for s in available_steps if lo <= s <= hi]
            if not steps_to_render:
                steps_to_render = [available_steps[0]]
        else:
            if step is None:
                steps_to_render = [available_steps[0]]
            elif int(step) in available_steps:
                steps_to_render = [int(step)]
            else:
                clamped_idx = max(0, min(len(available_steps) - 1, int(step)))
                steps_to_render = [available_steps[clamped_idx]]

        total_frames = len(steps_to_render)
        for idx, step_idx in enumerate(steps_to_render):
            out_path = format_frame_output_path(
                out_pattern, step_idx, all_steps=is_range_request
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            scene.frame_set(step_idx)
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
        "samples": samples_clamped,
        "blender_version": blender_ver,
    }


def _parse_bind_address(bind_str: str) -> tuple[str, int]:
    text = bind_str.strip()
    if ":" not in text:
        return ("127.0.0.1", int(text))
    host, port_s = text.rsplit(":", 1)
    return (host or "127.0.0.1", int(port_s))


def _dispatch_worker_request(
    req: dict[str, Any],
    state: dict[str, Any],
    blender_ver: str,
) -> tuple[dict[str, Any], bool]:
    """Dispatches a single worker request and returns `(response_dict, shutting_down)`."""
    shutting_down = False
    try:
        cmd = req.get("command", "")
        if cmd == "ping":
            resp = {
                "status": "ok",
                "blender_version": blender_ver,
                "loaded_bundle": state.get("active_bundle_dir"),
            }
        elif cmd == "load_bundle":
            bundle_path = Path(req["bundle"]).expanduser().resolve()
            emit_progress(stage="import", status="starting", bundle=bundle_path)
            active_summary = importer.import_bundle(bundle_path, clear_existing=True)
            active_bundle_dir = str(active_summary["bundle_dir"])
            state["active_summary"] = active_summary
            state["active_bundle_dir"] = active_bundle_dir
            emit_progress(
                stage="import",
                status="done",
                objects=len(active_summary["objects"]),
                steps=int(active_summary["num_steps"]),
            )
            resp = {
                "status": "ok",
                "blender_version": blender_ver,
                "loaded_bundle": active_bundle_dir,
                "summary": active_summary,
            }
        elif cmd == "update_camera":
            view_info = importer.update_camera_and_view(req)
            emit_progress(
                stage="update_camera",
                status="done",
                camera=view_info["camera"],
            )
            resp = {
                "status": "ok",
                "blender_version": blender_ver,
                "view": view_info,
            }
        elif cmd == "render":
            req_bundle = req.get("bundle")
            force_reload = bool(req.get("force_reload", False))
            active_summary = state.get("active_summary")
            active_bundle_dir = state.get("active_bundle_dir")
            if req_bundle:
                resolved_dir, _ = importer.resolve_bundle_dir_and_manifest_path(
                    req_bundle
                )
                if (
                    force_reload
                    or active_summary is None
                    or active_bundle_dir != str(resolved_dir)
                ):
                    emit_progress(
                        stage="import",
                        status="starting",
                        bundle=resolved_dir,
                    )
                    active_summary = importer.import_bundle(
                        resolved_dir, clear_existing=True
                    )
                    active_bundle_dir = str(active_summary["bundle_dir"])
                    state["active_summary"] = active_summary
                    state["active_bundle_dir"] = active_bundle_dir
                    emit_progress(
                        stage="import",
                        status="done",
                        objects=len(active_summary["objects"]),
                        steps=int(active_summary["num_steps"]),
                    )
            if active_summary is None:
                raise ValueError(
                    "No bundle is loaded in the Blender worker; pass 'bundle' or call 'load_bundle' first"
                )
            if isinstance(req.get("camera"), dict) or isinstance(
                req.get("background"), dict
            ):
                importer.update_camera_and_view(req)
            out_pattern = req.get("out")
            if not out_pattern:
                raise ValueError("'render' command requires an 'out' path")
            render_res = _execute_cycles_render_on_active_scene(
                summary=active_summary,
                out_pattern=str(out_pattern),
                samples=int(req.get("samples", 128)),
                device=str(req.get("device", "auto")),
                step=req.get("step"),
                all_steps=bool(req.get("all_steps", False)),
                from_step=req.get("from_step"),
                to_step=req.get("to_step"),
                width=req.get("width"),
                height=req.get("height"),
                save_blend=req.get("save_blend"),
            )
            resp = {
                "status": "ok",
                "files": render_res["files"],
                "device": render_res["device"],
                "samples": render_res["samples"],
                "blender_version": render_res["blender_version"],
                "loaded_bundle": active_bundle_dir,
            }
        elif cmd == "shutdown":
            resp = {
                "status": "ok",
                "blender_version": blender_ver,
                "shutting_down": True,
            }
            shutting_down = True
        else:
            resp = {
                "status": "error",
                "error": f"Unknown worker command {cmd!r}",
            }
    except Exception as exc:
        resp = {
            "status": "error",
            "error": str(exc),
        }
    return resp, shutting_down


def _run_ipc_dir_worker(ipc_dir: Path, blender_ver: str) -> None:
    """Runs the persistent worker over an atomic directory queue (`ipc:<dir>`)."""
    ipc_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(
        stage="worker_listening",
        addr=f"ipc:{ipc_dir}",
        blender_version=blender_ver,
    )
    state: dict[str, Any] = {"active_summary": None, "active_bundle_dir": None}
    shutting_down = False
    while not shutting_down:
        req_files = sorted(ipc_dir.glob("req_*.json"))
        if not req_files:
            time.sleep(0.01)
            continue
        for req_path in req_files:
            req_id = req_path.stem[4:]
            try:
                raw_text = req_path.read_text(encoding="utf-8")
                req_path.unlink(missing_ok=True)
                req = json.loads(raw_text)
                resp, shutting_down = _dispatch_worker_request(req, state, blender_ver)
            except Exception as exc:
                req_path.unlink(missing_ok=True)
                resp = {"status": "error", "error": str(exc)}
            tmp_resp = ipc_dir / f".resp_{req_id}.json.tmp"
            final_resp = ipc_dir / f"resp_{req_id}.json"
            tmp_resp.write_text(json.dumps(resp) + "\n", encoding="utf-8")
            os.replace(tmp_resp, final_resp)
            if shutting_down:
                break


def run_socket_worker(bind_addr: str) -> None:
    """Runs a persistent line-delimited JSON TCP (or `ipc:<dir>`) worker inside Blender (`M5.4`)."""
    import bpy

    blender_ver = getattr(bpy.app, "version_string", "unknown")
    trimmed = bind_addr.strip()
    if trimmed.startswith("ipc:"):
        ipc_dir = Path(trimmed[4:]).expanduser().resolve()
        _run_ipc_dir_worker(ipc_dir, blender_ver)
        return

    host, port = _parse_bind_address(trimmed)
    state: dict[str, Any] = {"active_summary": None, "active_bundle_dir": None}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(8)
        bound_host, bound_port = srv.getsockname()[:2]
        emit_progress(
            stage="worker_listening",
            addr=f"{bound_host}:{bound_port}",
            blender_version=blender_ver,
        )

        shutting_down = False
        while not shutting_down:
            conn, _peer = srv.accept()
            with conn:
                reader = conn.makefile("r", encoding="utf-8")
                writer = conn.makefile("w", encoding="utf-8")
                for raw_line in reader:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        req = json.loads(line)
                    except Exception as exc:
                        resp = {"status": "error", "error": str(exc)}
                        shutting_down = False
                    else:
                        resp, shutting_down = _dispatch_worker_request(
                            req, state, blender_ver
                        )
                    writer.write(json.dumps(resp) + "\n")
                    writer.flush()
                    if shutting_down:
                        break


def run_headless_render(argv: list[str] | None = None) -> dict[str, Any]:
    """Executes the full import, Cycles setup, optional `.blend` save, and render (or starts `--serve`)."""
    args = parse_cli_args(argv)
    if args.serve:
        run_socket_worker(args.serve)
        return {"status": "worker_stopped"}

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

    return _execute_cycles_render_on_active_scene(
        summary=summary,
        out_pattern=args.out,
        samples=args.samples,
        device=args.device,
        step=args.step,
        all_steps=args.all_steps,
        from_step=args.from_step,
        to_step=args.to_step,
        width=args.width,
        height=args.height,
        save_blend=args.save_blend,
    )


if __name__ == "__main__":
    run_headless_render()

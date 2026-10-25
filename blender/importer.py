# SPDX-License-Identifier: GPL-3.0-or-later
"""Tetraglass bundle importer for Blender 4.2 LTS and 5.x LTS.

Reads a renderer-neutral Tetraglass bundle directory (`manifest.json` + `meshes/*.tgbin`,
specified in `docs/bundle-format.md`) using only the Python standard library and `numpy`,
and populates a Blender scene with meshes, scalar attributes, colormap shader nodes,
wireframe edge curves, camera, lighting, background, and per-step timeline frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any

import numpy as np

BUNDLE_FORMAT_VERSION = 1
BUNDLE_FORMAT_ID = "tetraglass-bundle"
TGBIN_MAGIC = b"TGMB"
TGBIN_VERSION = 1
TGBIN_HEADER_STRUCT = struct.Struct("<4sIIIII")


@dataclass
class TgbinMesh:
    """Decoded contents of a `.tgbin` binary mesh buffer."""

    positions: np.ndarray  # shape (N, 3), float32
    triangles: np.ndarray  # shape (M, 3), uint32
    normals: np.ndarray  # shape (N, 3), float32
    scalars: np.ndarray  # shape (N,), float32 in [0, 1]
    raw_scalars: np.ndarray  # shape (N,), float32
    edge_positions: np.ndarray  # shape (E, 2, 3), float32


def srgb_channel_to_linear(c: float) -> float:
    """Converts a single sRGB channel in [0, 1] to linear light (IEC 61966-2-1)."""
    x = max(0.0, min(1.0, float(c)))
    if x <= 0.04045:
        return x / 12.92
    return ((x + 0.055) / 1.055) ** 2.4


def srgb_rgb_to_linear_rgba(rgb: list[float] | tuple[float, float, float], alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Converts an sRGB [r, g, b] triple to a linear-light (r, g, b, a) tuple."""
    return (
        srgb_channel_to_linear(rgb[0]),
        srgb_channel_to_linear(rgb[1]),
        srgb_channel_to_linear(rgb[2]),
        float(alpha),
    )


def resolve_bundle_dir_and_manifest_path(bundle_path: str | Path) -> tuple[Path, Path]:
    """Resolves either a bundle directory or a `manifest.json` file path."""
    path = Path(bundle_path).expanduser().resolve()
    if path.is_file() and path.name == "manifest.json":
        return path.parent, path
    manifest_path = path / "manifest.json"
    if manifest_path.is_file():
        return path, manifest_path
    raise ValueError(f"Could not find manifest.json in bundle path: {path}")


def load_bundle_manifest(bundle_path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Loads and validates `manifest.json` from `bundle_path`."""
    bundle_dir, manifest_path = resolve_bundle_dir_and_manifest_path(bundle_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError(f"manifest.json root must be a JSON object in {manifest_path}")

    version = manifest.get("version")
    if version != BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported bundle manifest version {version!r} (expected {BUNDLE_FORMAT_VERSION})"
        )

    fmt = manifest.get("format")
    if fmt != BUNDLE_FORMAT_ID:
        raise ValueError(
            f"Unsupported bundle format {fmt!r} (expected {BUNDLE_FORMAT_ID!r})"
        )

    return bundle_dir, manifest


def read_tgbin(path: str | Path) -> TgbinMesh:
    """Reads and validates a little-endian `.tgbin` binary mesh buffer."""
    file_path = Path(path)
    data = file_path.read_bytes()
    if len(data) < TGBIN_HEADER_STRUCT.size:
        raise ValueError(
            f".tgbin file {file_path} is truncated ({len(data)} bytes < 24-byte header)"
        )

    magic, version, num_verts, num_tris, num_edges, _flags = TGBIN_HEADER_STRUCT.unpack_from(
        data, 0
    )
    if magic != TGBIN_MAGIC:
        raise ValueError(
            f"Invalid .tgbin magic {magic!r} in {file_path} (expected {TGBIN_MAGIC!r})"
        )
    if version != TGBIN_VERSION:
        raise ValueError(
            f"Unsupported .tgbin version {version} in {file_path} (expected {TGBIN_VERSION})"
        )

    pos_bytes = num_verts * 3 * 4
    tri_bytes = num_tris * 3 * 4
    norm_bytes = num_verts * 3 * 4
    scal_bytes = num_verts * 4
    raw_bytes = num_verts * 4
    edge_bytes = num_edges * 2 * 3 * 4
    expected_size = (
        TGBIN_HEADER_STRUCT.size
        + pos_bytes
        + tri_bytes
        + norm_bytes
        + scal_bytes
        + raw_bytes
        + edge_bytes
    )
    if len(data) != expected_size:
        raise ValueError(
            f".tgbin size mismatch in {file_path}: got {len(data)} bytes, expected {expected_size}"
        )

    offset = TGBIN_HEADER_STRUCT.size
    positions = np.frombuffer(
        data, dtype="<f4", count=num_verts * 3, offset=offset
    ).reshape((num_verts, 3)).copy()
    offset += pos_bytes

    triangles = np.frombuffer(
        data, dtype="<u4", count=num_tris * 3, offset=offset
    ).reshape((num_tris, 3)).copy()
    offset += tri_bytes

    if num_tris > 0 and num_verts > 0:
        max_idx = int(triangles.max())
        if max_idx >= num_verts:
            raise ValueError(
                f".tgbin triangle index {max_idx} out of bounds for {num_verts} vertices in {file_path}"
            )
    elif num_tris > 0 and num_verts == 0:
        raise ValueError(f".tgbin has {num_tris} triangles but 0 vertices in {file_path}")

    normals = np.frombuffer(
        data, dtype="<f4", count=num_verts * 3, offset=offset
    ).reshape((num_verts, 3)).copy()
    offset += norm_bytes

    scalars = np.frombuffer(
        data, dtype="<f4", count=num_verts, offset=offset
    ).copy()
    offset += scal_bytes

    raw_scalars = np.frombuffer(
        data, dtype="<f4", count=num_verts, offset=offset
    ).copy()
    offset += raw_bytes

    edge_positions = np.frombuffer(
        data, dtype="<f4", count=num_edges * 6, offset=offset
    ).reshape((num_edges, 2, 3)).copy()

    return TgbinMesh(
        positions=positions,
        triangles=triangles,
        normals=normals,
        scalars=scalars,
        raw_scalars=raw_scalars,
        edge_positions=edge_positions,
    )


def _enable_node_tree(id_block: Any) -> Any:
    """Returns `id_block.node_tree`, enabling `use_nodes` on Blender 4.x if needed."""
    if getattr(id_block, "node_tree", None) is None:
        id_block.use_nodes = True
    return id_block.node_tree


def _normalize3(v: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _cross3(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _matrix_from_columns(
    right: tuple[float, float, float],
    up: tuple[float, float, float],
    back: tuple[float, float, float],
    translation: tuple[float, float, float],
):
    from mathutils import Matrix

    return Matrix(
        (
            (right[0], up[0], back[0], translation[0]),
            (right[1], up[1], back[1], translation[1]),
            (right[2], up[2], back[2], translation[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def _direction_to_sun_matrix(to_light: list[float] | tuple[float, float, float]):
    """Builds a world matrix whose local +Z axis points toward `to_light`."""
    back = _normalize3((float(to_light[0]), float(to_light[1]), float(to_light[2])))
    ref_up = (0.0, 1.0, 0.0) if abs(back[1]) < 0.9 else (1.0, 0.0, 0.0)
    right = _normalize3(_cross3(ref_up, back))
    up = _normalize3(_cross3(back, right))
    return _matrix_from_columns(right, up, back, (0.0, 0.0, 0.0))


def _clear_default_scene(scene: Any) -> None:
    import bpy

    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for curve in list(bpy.data.curves):
        if curve.users == 0:
            bpy.data.curves.remove(curve)


def _configure_world_background(scene: Any, bg_spec: dict[str, Any]) -> None:
    import bpy

    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("TetraglassWorld")
        scene.world = world

    nt = _enable_node_tree(world)
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputWorld")
    out_node.location = (520, 0)

    cam_bg_node = nodes.new("ShaderNodeBackground")
    cam_bg_node.location = (180, 80)
    cam_bg_node.inputs["Strength"].default_value = 1.0

    env_bg_node = nodes.new("ShaderNodeBackground")
    env_bg_node.location = (180, -100)
    env_bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    env_bg_node.inputs["Strength"].default_value = 0.22

    light_path = nodes.new("ShaderNodeLightPath")
    light_path.location = (180, 260)

    mix_shader = nodes.new("ShaderNodeMixShader")
    mix_shader.location = (360, 0)
    links.new(light_path.outputs["Is Camera Ray"], mix_shader.inputs["Fac"])
    links.new(env_bg_node.outputs["Background"], mix_shader.inputs[1])
    links.new(cam_bg_node.outputs["Background"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], out_node.inputs["Surface"])

    kind = bg_spec.get("kind", "solid")
    if kind == "gradient" and bg_spec.get("top_srgb") and bg_spec.get("bottom_srgb"):
        tex_coord = nodes.new("ShaderNodeTexCoord")
        tex_coord.location = (-460, 80)

        sep_xyz = nodes.new("ShaderNodeSeparateXYZ")
        sep_xyz.location = (-260, 80)
        links.new(tex_coord.outputs["Window"], sep_xyz.inputs["Vector"])

        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.location = (-60, 80)
        elements = ramp.color_ramp.elements
        elements[0].position = 0.0
        elements[0].color = srgb_rgb_to_linear_rgba(bg_spec["bottom_srgb"], 1.0)
        elements[1].position = 1.0
        elements[1].color = srgb_rgb_to_linear_rgba(bg_spec["top_srgb"], 1.0)
        links.new(sep_xyz.outputs["Y"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], cam_bg_node.inputs["Color"])
    else:
        color_srgb = bg_spec.get("color_srgb", [1.0, 1.0, 1.0])
        cam_bg_node.inputs["Color"].default_value = srgb_rgb_to_linear_rgba(color_srgb, 1.0)


def _configure_camera(
    scene: Any, collection: Any, cam_spec: dict[str, Any]
) -> Any:
    import bpy

    cam_data = bpy.data.cameras.new("TetraglassCamera")
    projection = cam_spec.get("projection", "perspective")
    if projection == "orthographic":
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = float(cam_spec.get("ortho_scale", 2.0))
    else:
        cam_data.type = "PERSP"
        cam_data.sensor_fit = "VERTICAL"
        view_angle_deg = float(cam_spec.get("view_angle_deg", 30.0))
        cam_data.angle = math.radians(max(1.0, min(170.0, view_angle_deg)))

    cam_data.clip_start = max(1e-5, float(cam_spec.get("clip_start", 0.01)))
    cam_data.clip_end = max(
        cam_data.clip_start + 1.0, float(cam_spec.get("clip_end", 1000.0))
    )

    pos = tuple(float(x) for x in cam_spec.get("position", [0.0, 0.0, 3.0]))
    focal = tuple(float(x) for x in cam_spec.get("focal_point", [0.0, 0.0, 0.0]))
    up_hint = tuple(float(x) for x in cam_spec.get("up", [0.0, 1.0, 0.0]))

    back = _normalize3((pos[0] - focal[0], pos[1] - focal[1], pos[2] - focal[2]))
    right = _normalize3(_cross3(up_hint, back))
    up = _normalize3(_cross3(back, right))

    cam_obj = bpy.data.objects.new("TetraglassCamera", cam_data)
    cam_obj.matrix_world = _matrix_from_columns(right, up, back, pos)
    collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def _configure_lights(
    collection: Any, lights_spec: dict[str, Any]
) -> list[Any]:
    import bpy

    preset = lights_spec.get("preset", "default")
    created_lights = []

    def add_sun(name: str, direction: list[float], energy: float) -> Any:
        light_data = bpy.data.lights.new(name=name, type="SUN")
        light_data.energy = energy
        # Soften sun shadows slightly to match scientific visualization shading
        if hasattr(light_data, "angle"):
            light_data.angle = math.radians(5.0)
        light_obj = bpy.data.objects.new(name, light_data)
        light_obj.matrix_world = _direction_to_sun_matrix(direction)
        collection.objects.link(light_obj)
        created_lights.append(light_obj)
        return light_obj

    headlight_dir = lights_spec.get("headlight_direction", [0.0, 0.0, 1.0])
    if preset == "headlight":
        add_sun("TetraglassHeadlight", headlight_dir, energy=3.0)
    else:
        key_dir = lights_spec.get("key_direction", [0.55, 0.65, 0.52])
        fill_dir = lights_spec.get("fill_direction", [-0.65, -0.25, 0.71])
        add_sun("TetraglassKeyLight", key_dir, energy=2.4)
        add_sun("TetraglassFillLight", fill_dir, energy=0.9)
        add_sun("TetraglassHeadlight", headlight_dir, energy=0.7)

    return created_lights


def _build_surface_material(obj_spec: dict[str, Any]) -> Any:
    import bpy

    obj_name = obj_spec.get("name", "Object")
    mat_spec = obj_spec.get("material", {})
    colormap_spec = obj_spec.get("colormap")
    opacity = max(0.0, min(1.0, float(mat_spec.get("opacity", 1.0))))

    mat = bpy.data.materials.new(name=f"TetraglassMat_{obj_name}")
    nt = _enable_node_tree(mat)
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (520, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (220, 0)
    bsdf.inputs["Roughness"].default_value = 0.45
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.25
    bsdf.inputs["Alpha"].default_value = opacity
    links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    if opacity < 0.9999 and hasattr(mat, "blend_method"):
        try:
            mat.blend_method = "HASHED"
        except Exception:
            pass

    if colormap_spec and colormap_spec.get("lut_srgb"):
        lut_srgb = colormap_spec["lut_srgb"]
        num_entries = len(lut_srgb)

        attr_node = nodes.new("ShaderNodeAttribute")
        attr_node.location = (-640, 0)
        attr_node.attribute_name = "tetraglass_scalar"

        # 1. Exact 256x1 linear-light LUT texture for faithful scientific colormap reproduction.
        lut_img = bpy.data.images.new(
            name=f"TetraglassLUT_{obj_name}",
            width=num_entries,
            height=1,
            alpha=True,
            float_buffer=True,
        )
        if hasattr(lut_img, "colorspace_settings"):
            for cs_name in ("Linear Rec.709", "Linear", "Non-Color"):
                try:
                    lut_img.colorspace_settings.name = cs_name
                    break
                except Exception:
                    continue

        pixels = np.empty((num_entries, 4), dtype=np.float32)
        for idx, rgb in enumerate(lut_srgb):
            pixels[idx] = srgb_rgb_to_linear_rgba(rgb, 1.0)
        lut_img.pixels.foreach_set(pixels.ravel())
        lut_img.pack()

        # Map normalized scalar t in [0, 1] to texel centers [(0.5)/256 .. (255.5)/256]
        combine_xyz = nodes.new("ShaderNodeCombineXYZ")
        combine_xyz.location = (-420, 80)
        combine_xyz.inputs["Y"].default_value = 0.5
        combine_xyz.inputs["Z"].default_value = 0.0
        links.new(attr_node.outputs["Fac"], combine_xyz.inputs["X"])

        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.location = (-140, 120)
        tex_node.image = lut_img
        tex_node.extension = "EXTEND"
        tex_node.interpolation = "Linear"
        links.new(combine_xyz.outputs["Vector"], tex_node.inputs["Vector"])
        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

        # 2. Also populate an editable 32-stop ColorRamp node in the shader tree so
        # interactive Blender users can inspect or tweak the ramp directly.
        ramp_node = nodes.new("ShaderNodeValToRGB")
        ramp_node.location = (-140, -180)
        ramp_node.label = f"Colormap ({colormap_spec.get('name', 'viridis')})"
        ramp = ramp_node.color_ramp
        num_stops = min(32, num_entries)
        while len(ramp.elements) < num_stops:
            ramp.elements.new(0.5)
        for stop_i in range(num_stops):
            t = stop_i / max(1, num_stops - 1)
            lut_idx = min(num_entries - 1, int(round(t * (num_entries - 1))))
            elem = ramp.elements[stop_i]
            elem.position = t
            elem.color = srgb_rgb_to_linear_rgba(lut_srgb[lut_idx], 1.0)
        links.new(attr_node.outputs["Fac"], ramp_node.inputs["Fac"])
    else:
        solid_srgb = mat_spec.get("solid_color_srgb", [0.82, 0.84, 0.88])
        bsdf.inputs["Base Color"].default_value = srgb_rgb_to_linear_rgba(
            solid_srgb, 1.0
        )

    return mat


def _build_edge_material(obj_name: str, edge_color_srgb: list[float]) -> Any:
    import bpy

    mat = bpy.data.materials.new(name=f"TetraglassEdgeMat_{obj_name}")
    nt = _enable_node_tree(mat)
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    out_node.location = (300, 0)

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (40, 0)
    emission.inputs["Color"].default_value = srgb_rgb_to_linear_rgba(
        edge_color_srgb, 1.0
    )
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], out_node.inputs["Surface"])
    return mat


def _create_blender_mesh_object(
    name: str,
    tg_mesh: TgbinMesh,
    surface_mat: Any,
    field_name: str | None,
    collection: Any,
) -> Any:
    import bpy

    mesh = bpy.data.meshes.new(name)
    num_verts = int(tg_mesh.positions.shape[0])
    num_tris = int(tg_mesh.triangles.shape[0])

    if num_verts > 0 and num_tris > 0:
        mesh.vertices.add(num_verts)
        mesh.vertices.foreach_set("co", tg_mesh.positions.ravel())

        mesh.loops.add(num_tris * 3)
        mesh.loops.foreach_set("vertex_index", tg_mesh.triangles.ravel())

        mesh.polygons.add(num_tris)
        loop_start = np.arange(0, num_tris * 3, 3, dtype=np.int32)
        loop_total = np.full(num_tris, 3, dtype=np.int32)
        mesh.polygons.foreach_set("loop_start", loop_start)
        mesh.polygons.foreach_set("loop_total", loop_total)
        mesh.polygons.foreach_set(
            "use_smooth", np.ones(num_tris, dtype=bool)
        )
        mesh.update()

        # Apply exact per-vertex normals from the bundle
        normals_list = [
            (float(n[0]), float(n[1]), float(n[2])) for n in tg_mesh.normals
        ]
        try:
            mesh.normals_split_custom_set_from_vertices(normals_list)
        except Exception:
            pass

        attr_norm = mesh.attributes.new(
            name="tetraglass_scalar", type="FLOAT", domain="POINT"
        )
        attr_norm.data.foreach_set("value", tg_mesh.scalars)

        attr_raw = mesh.attributes.new(
            name="tetraglass_raw_scalar", type="FLOAT", domain="POINT"
        )
        attr_raw.data.foreach_set("value", tg_mesh.raw_scalars)

        if field_name and field_name not in (
            "tetraglass_scalar",
            "tetraglass_raw_scalar",
        ):
            attr_named = mesh.attributes.new(
                name=field_name, type="FLOAT", domain="POINT"
            )
            attr_named.data.foreach_set("value", tg_mesh.raw_scalars)

    mesh.materials.append(surface_mat)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def _create_blender_edge_object(
    name: str,
    tg_mesh: TgbinMesh,
    edge_mat: Any,
    bevel_depth: float,
    collection: Any,
) -> Any | None:
    import bpy

    num_edges = int(tg_mesh.edge_positions.shape[0])
    if num_edges == 0:
        return None

    curve = bpy.data.curves.new(name=name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = max(1e-5, float(bevel_depth))
    curve.bevel_resolution = 1
    curve.use_fill_caps = True

    for edge_idx in range(num_edges):
        p0 = tg_mesh.edge_positions[edge_idx, 0]
        p1 = tg_mesh.edge_positions[edge_idx, 1]
        spline = curve.splines.new(type="POLY")
        spline.points.add(1)  # starts with 1 point, so add(1) gives 2 points
        spline.points[0].co = (float(p0[0]), float(p0[1]), float(p0[2]), 1.0)
        spline.points[1].co = (float(p1[0]), float(p1[1]), float(p1[2]), 1.0)

    curve.materials.append(edge_mat)
    edge_obj = bpy.data.objects.new(name, curve)
    for vis_attr in ("visible_shadow", "visible_diffuse", "visible_glossy"):
        if hasattr(edge_obj, vis_attr):
            try:
                setattr(edge_obj, vis_attr, False)
            except Exception:
                pass
    collection.objects.link(edge_obj)
    return edge_obj


def _keyframe_step_visibility(
    objs: list[Any], active_step: int, num_steps: int
) -> None:
    if num_steps <= 1:
        return
    for step_idx in range(num_steps):
        hidden = step_idx != active_step
        for obj in objs:
            obj.hide_viewport = hidden
            obj.hide_render = hidden
            obj.keyframe_insert(data_path="hide_viewport", frame=step_idx)
            obj.keyframe_insert(data_path="hide_render", frame=step_idx)
    initial_hidden = active_step != 0
    for obj in objs:
        obj.hide_viewport = initial_hidden
        obj.hide_render = initial_hidden


def import_bundle(
    bundle_path: str | Path,
    context: Any = None,
    clear_existing: bool = True,
) -> dict[str, Any]:
    """Imports a Tetraglass bundle directory into the active Blender scene.

    Returns a dictionary summarizing the imported objects, time steps, camera, and lights.
    """
    import bpy

    bundle_dir, manifest = load_bundle_manifest(bundle_path)
    scene = context.scene if (context is not None and getattr(context, "scene", None)) else bpy.context.scene

    if clear_existing:
        _clear_default_scene(scene)

    collection = bpy.data.collections.get("Tetraglass")
    if collection is None:
        collection = bpy.data.collections.new("Tetraglass")
        scene.collection.children.link(collection)

    dataset_center = manifest.get("dataset_center", [0.0, 0.0, 0.0])
    scene["tetraglass_dataset_center"] = [float(x) for x in dataset_center]
    scene["tetraglass_bundle_version"] = int(manifest["version"])

    image_size = manifest.get("image_size", [1200, 800])
    scene.render.resolution_x = int(image_size[0])
    scene.render.resolution_y = int(image_size[1])
    scene.render.resolution_percentage = 100
    if hasattr(scene, "view_settings"):
        try:
            scene.view_settings.view_transform = "Standard"
        except Exception:
            pass

    _configure_world_background(scene, manifest.get("background", {"kind": "solid", "color_srgb": [1.0, 1.0, 1.0]}))
    camera_obj = _configure_camera(scene, collection, manifest.get("camera", {}))
    light_objs = _configure_lights(collection, manifest.get("lights", {}))

    time_steps = manifest.get("time_steps", [{"step": 0, "time": None}])
    num_steps = max(1, len(time_steps))
    scene.frame_start = 0
    scene.frame_end = num_steps - 1
    scene.frame_set(0)

    # Estimate characteristic length from camera clip range for wireframe edge thickness
    cam_spec = manifest.get("camera", {})
    focal = cam_spec.get("focal_point", [0.0, 0.0, 0.0])
    pos = cam_spec.get("position", [0.0, 0.0, 3.0])
    cam_dist = math.sqrt(
        (pos[0] - focal[0]) ** 2 + (pos[1] - focal[1]) ** 2 + (pos[2] - focal[2]) ** 2
    )
    edge_bevel_depth = max(1e-4, cam_dist * 0.0009)

    imported_objects: list[str] = []
    for obj_spec in manifest.get("objects", []):
        obj_name = obj_spec.get("name", "Object")
        style = obj_spec.get("style", "surface")
        mat_spec = obj_spec.get("material", {})
        draw_edges = bool(mat_spec.get("draw_edges", False)) or style == "wireframe"
        edge_color_srgb = mat_spec.get("edge_color_srgb", [0.0, 0.0, 0.0])
        colormap_spec = obj_spec.get("colormap")
        field_name = colormap_spec.get("field") if isinstance(colormap_spec, dict) else None

        surface_mat = _build_surface_material(obj_spec)
        edge_mat = _build_edge_material(obj_name, edge_color_srgb) if draw_edges else None

        mesh_refs = obj_spec.get("meshes", [])
        for mesh_ref in mesh_refs:
            step_idx = int(mesh_ref.get("step", 0))
            rel_file = mesh_ref.get("file", "")
            tg_path = bundle_dir / rel_file
            tg_mesh = read_tgbin(tg_path)

            step_suffix = f"_step_{step_idx:04d}" if num_steps > 1 else ""
            mesh_obj_name = f"{obj_name}{step_suffix}"
            step_objs = []

            if style != "wireframe" and tg_mesh.triangles.shape[0] > 0:
                mesh_obj = _create_blender_mesh_object(
                    name=mesh_obj_name,
                    tg_mesh=tg_mesh,
                    surface_mat=surface_mat,
                    field_name=field_name,
                    collection=collection,
                )
                step_objs.append(mesh_obj)
                imported_objects.append(mesh_obj.name)

            if draw_edges and edge_mat is not None and tg_mesh.edge_positions.shape[0] > 0:
                edge_obj = _create_blender_edge_object(
                    name=f"{mesh_obj_name}_edges",
                    tg_mesh=tg_mesh,
                    edge_mat=edge_mat,
                    bevel_depth=edge_bevel_depth,
                    collection=collection,
                )
                if edge_obj is not None:
                    step_objs.append(edge_obj)
                    imported_objects.append(edge_obj.name)

            for created_obj in step_objs:
                created_obj["tetraglass_step"] = step_idx
            _keyframe_step_visibility(step_objs, active_step=step_idx, num_steps=num_steps)

    scene.frame_set(0)
    return {
        "bundle_dir": str(bundle_dir),
        "num_steps": num_steps,
        "objects": imported_objects,
        "camera": camera_obj.name,
        "lights": [l.name for l in light_objs],
    }

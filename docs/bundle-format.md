# Bundle format (version 1)

Status: **v1** (implemented in M3.5).

The **Tetraglass bundle format** is a renderer-neutral directory container that bridges the Rust scene evaluation pipeline (`tetraglass-export` / `tetraglass-cli`) and external offline renderers such as Blender Cycles (`blender/importer.py` and `blender/render.py`).

In accordance with `D-011` and `.agents/rules/blender-gpl.md`, a bundle carries **data only**: no executable code, no Python snippets, and no Blender-specific data structures.

---

## Directory layout

```text
<bundle_dir>/
├── manifest.json
└── meshes/
    ├── obj_0_<input>_step_0000.tgbin
    ├── obj_0_<input>_step_0001.tgbin
    └── ...
```

- `manifest.json`: UTF-8 JSON document describing the format version, coordinate system and dataset center (`D-014`), image size, camera, lighting, background, time steps, and per-object mesh files, 256-entry colormap lookup tables, and material hints.
- `meshes/*.tgbin`: one little-endian binary mesh file per visible representation and exported time step.

---

## `manifest.json` schema

```json
{
  "version": 1,
  "format": "tetraglass-bundle",
  "units": "meters",
  "dataset_center": [1.0, 1.0, 1.0],
  "image_size": [1200, 800],
  "camera": {
    "position": [3.5, 2.3, 4.8],
    "world_position": [4.5, 3.3, 5.8],
    "focal_point": [0.0, 0.0, 0.0],
    "world_focal_point": [1.0, 1.0, 1.0],
    "up": [-0.17, 0.94, -0.29],
    "projection": "perspective",
    "view_angle_deg": 30.0,
    "ortho_scale": 3.464,
    "clip_start": 0.06,
    "clip_end": 16.5
  },
  "lights": {
    "preset": "default",
    "key_direction": [0.55, 0.65, 0.52],
    "fill_direction": [-0.65, -0.25, 0.71],
    "headlight_direction": [0.50, 0.34, 0.79]
  },
  "background": {
    "kind": "solid",
    "color_srgb": [1.0, 1.0, 1.0]
  },
  "time_steps": [
    { "step": 0, "time": 0.0 }
  ],
  "objects": [
    {
      "id": "obj_0_cut",
      "input": "cut",
      "representation_index": 0,
      "style": "surface_with_edges",
      "color_by": "linear",
      "association": "point",
      "solid_color_srgb": [0.827, 0.827, 0.827],
      "colormap": {
        "name": "viridis",
        "lut_srgb": [[0.267, 0.005, 0.329], "... 256 RGB triples in [0, 1] ..."],
        "data_range": [0.0, 12.0],
        "effective_range": [0.0, 12.0],
        "log_scale": false,
        "center": null
      },
      "material": {
        "opacity": 1.0,
        "draw_surface": true,
        "draw_edges": true,
        "edge_color_srgb": [0.0, 0.0, 0.0],
        "point_size": 3.0
      },
      "meshes": [
        {
          "step": 0,
          "time": 0.0,
          "file": "meshes/obj_0_cut_step_0000.tgbin",
          "num_vertices": 12,
          "num_triangles": 4,
          "num_edge_segments": 4
        }
      ]
    }
  ]
}
```

### Top-level fields

| Field | Type | Meaning |
| --- | --- | --- |
| `version` | `u32` | Bundle format version (`1`). Breaking changes increment this number. |
| `format` | `string` | Always `"tetraglass-bundle"`. |
| `units` | `string` | Spatial unit label (`"meters"` by default). |
| `dataset_center` | `[f64; 3]` | World-space center subtracted from all `f32` vertex positions in `.tgbin` files (`D-014`). |
| `image_size` | `[u32; 2]` | Target render resolution `[width, height]` in pixels. |
| `camera` | object | Resolved camera parameters (both dataset-center-relative and world-space coordinates). |
| `lights` | object | Lighting preset and unit light direction vectors (`to_light` in world space). |
| `background` | object | Solid or vertical gradient background colors in normalized sRGB `[0.0, 1.0]`. |
| `time_steps` | list | Ordered list of `{ "step": usize, "time": f64 }` exported in this bundle. |
| `objects` | list | Ordered list of visible scene representations and their per-step mesh file references. |

### `objects[i].colormap`

When an object is colored by a scalar field (`color_by` is present), `colormap` contains:

- `name`: `"viridis"`, `"cividis"`, `"cool_warm"`, `"turbo"`, or `"grayscale"`.
- `lut_srgb`: array of exactly **256** `[r, g, b]` triples in normalized sRGB `[0.0, 1.0]`, uniformly sampled across `t in [0.0, 1.0]`.
- `data_range`: intrinsic `[min, max]` of the scalar component on the surface.
- `effective_range`: `[min, max]` mapped to `[0.0, 1.0]`.
- `log_scale`: whether logarithmic normalization was applied when computing the normalized scalar buffer.
- `center`: optional diverging center value.

When `color_by` is omitted (solid coloring), `colormap` is `null` and `solid_color_srgb` gives the surface's sRGB color.

---

## Binary mesh file format (`.tgbin`)

Each file referenced in `objects[i].meshes[j].file` is a tightly packed **little-endian** binary buffer with a 24-byte header followed by six 4-byte-aligned typed array sections.

### Header (24 bytes)

| Byte offset | Type | Name | Value / Meaning |
| --- | --- | --- | --- |
| `0..4` | `u32` | `magic` | `0x424D4754` (ASCII `"TGMB"` in little-endian) |
| `4..8` | `u32` | `version` | `1` |
| `8..12` | `u32` | `num_vertices` (`V`) | Number of vertices (`3 * num_triangles` for triangle meshes) |
| `12..16` | `u32` | `num_triangles` (`T`) | Number of triangles |
| `16..20` | `u32` | `num_edge_segments` (`E`) | Number of wireframe element-edge segments |
| `20..24` | `u32` | `flags` | Bitmask (`0x3`: bit 0 = normalized scalars present, bit 1 = raw scalars present) |

### Payload sections (immediately following byte 24)

| Section | Element type | Count | Byte length | Meaning |
| --- | --- | --- | --- | --- |
| `positions` | `f32` | `V * 3` | `12 * V` | Vertex `[x, y, z]` relative to `manifest.dataset_center` (`world = pos + dataset_center`). |
| `triangles` | `u32` | `T * 3` | `12 * T` | Triangle vertex index triples `[i0, i1, i2]` in `0..V`. |
| `normals` | `f32` | `V * 3` | `12 * V` | Unit vertex normals `[nx, ny, nz]`. |
| `scalars` | `f32` | `V` | `4 * V` | Scalar value normalized to `[0.0, 1.0]` (ready to feed a 256-stop colormap ramp). |
| `raw_scalars` | `f32` | `V` | `4 * V` | Unnormalized scalar value (`0.0` for solid-colored meshes). |
| `edge_positions` | `f32` | `E * 6` | `24 * E` | Pairs of 3D endpoints `[x0, y0, z0, x1, y1, z1]` (`f32` relative to `dataset_center`) for original element wireframe edges (`surface.wireframe_edges()`). |

The total byte length of a `.tgbin` file is:

```text
24 + 24 * V + 12 * T + 8 * V + 24 * E = 24 + 32 * V + 12 * T + 24 * E
```

Storing `edge_positions` directly as endpoint coordinate pairs `[p0, p1]` allows any importer (including `blender/importer.py`) to construct an independent wireframe curve or edge mesh in a single `numpy.frombuffer` call without re-indexing triangle corners.

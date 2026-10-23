---
name: author-scene
description: How to write, validate, render and iterate on JSON scene files for this library, including gallery entries and example scenes for users. Use whenever the task is to visualize a dataset, create or fix a scene file, add a gallery example, demonstrate a filter, or produce a figure (rasterized or ray-traced with Cycles).
---

# Author a scene

The scene format is specified in `docs/scene-format.md`. The CLI arrives in milestone M1; before that, scenes can only be written, not rendered.

## Loop

1. **Inspect the data first.** `tetraglass inspect <file> --json` lists arrays, components, value ranges, cell types, bounds and time steps. Use the names exactly as reported; never guess an array name.
2. **Start minimal.** One source, the external surface colored by one field, a fitted camera, one output:

```json
{
  "version": 1,
  "sources": { "mesh": { "file": "data/result.vtu" } },
  "view": {
    "show": [{ "input": "mesh", "color_by": "temperature", "colorbar": true }],
    "camera": { "fit": "all", "azimuth": 30, "elevation": 20 }
  },
  "output": { "image": "out.png", "size": [1200, 800] }
}
```

3. **Validate.** `tetraglass validate scene.json`. Fix errors using the JSON path and suggestion each one carries.
4. **Render.** `tetraglass render scene.json -o out.png`. Read the JSON summary it prints (arrays used, value ranges, camera).
5. **Look at the image.** Check that the intended feature is visible and not hidden behind other geometry, colors span the range, the colorbar is readable, and the camera frames the region of interest.
6. **Iterate** one change at a time: add a filter, adjust the camera, change the colormap or range.
7. **Final quality (M3 onward):** `tetraglass render scene.json -o final.png --engine cycles` produces a ray-traced image of the same scene through Blender.

## Conventions

- Names of sources, filters and views are short, lowercase and descriptive: `flow`, `mid_slice`, `p_iso`.
- Prefer explicit ranges (`"range": [0, 350]`) for figures that will be compared; `auto` for exploration.
- Diverging data (signed values around a reference) uses a diverging colormap centered on the reference value.
- Paths in scenes are relative to the scene file.

## Gallery entries

- One file per scene: `gallery/<topic>-<what>.json`, using only small data from `testdata/`.
- Fixed `size`, fixed camera (no `fit` changes after the baseline exists) and a fixed time step.
- Add one line to `gallery/README.md` describing what the scene shows.
- Produce its baseline with `/render-check`.

## Common mistakes

- Coloring by a cell array while expecting smooth shading: cell data renders flat by design.
- Slicing or contouring a source and also showing the full surface opaque, which hides the result. Lower the surface opacity or show its edges only.
- Forgetting `"time"` on a series: the default is the first step.
- Using a component name the array lacks: scalars have no `x`, `y` or `z`; vectors also offer `magnitude`.

# Gallery

Every scene in [`gallery/`](../../../gallery/) is validated and rendered in CI against its visual regression baseline in `tests/baselines/`. You can render any gallery scene locally with `tetraglass render` or explore it interactively with `tetraglass view`.

| Scene file | Filter / Feature | Command |
| --- | --- | --- |
| `gallery/surface-mixed-linear.json` | Implicit external surface of `mixed_linear.vtu` (`Hexahedron`, `Wedge`, `Pyramid`, `Tetra`) colored by point scalar `linear` (`viridis`) with right colorbar and axes triad. | `tetraglass render gallery/surface-mixed-linear.json -o out.png` |
| `gallery/surface-cell-id.json` | Explicit `"type": "surface"` filter colored by flat cell scalar `cell:cell_id` (`cividis`). | `tetraglass render gallery/surface-cell-id.json -o out.png` |
| `gallery/surface-velocity-magnitude.json` | External surface of `hex_linear.vtu` colored by vector magnitude `velocity:magnitude` (`coolwarm`) with horizontal bottom colorbar. | `tetraglass render gallery/surface-velocity-magnitude.json -o out.png` |
| `gallery/gmsh-quadratic-tetra.json` | Curved higher-order external surface of Gmsh MSH 4.1 10-node quadratic tetrahedra (`tet2_v41.msh`) colored by `linear` (`cividis`). | `tetraglass render gallery/gmsh-quadratic-tetra.json -o out.png` |
| `gallery/series-transient-last.json` | ParaView `.pvd` 3-step transient collection (`series.pvd`) evaluated at `"time": "last"` with `"range": "auto_all_steps"`. | `tetraglass render gallery/series-transient-last.json -o out.png` |
| `gallery/slice.json` | Plane `"type": "slice"` (`normal: [1.0, -0.8, 0.9]`) with `surface_with_edges` inside a `silver` outer element wireframe. | `tetraglass render gallery/slice.json -o out.png` |
| `gallery/contour.json` | Multi-isovalue `"type": "contour"` (`values: [2.5, 5.0, 7.5, 10.0]`) colored with `turbo` inside a `silver` outer element wireframe. | `tetraglass render gallery/contour.json -o out.png` |
| `gallery/clip.json` | Exact half-space `"type": "clip"` (`normal: [1.0, -0.8, 0.9]`, `keep: "back"`, `mode: "exact"`) with `surface_with_edges` inside a `silver` outer element wireframe. | `tetraglass render gallery/clip.json -o out.png` |
| `gallery/threshold.json` | Whole-cell `"type": "threshold"` (`field: "linear"`, `min: 0.0`, `max: 9.0`) keeping 5 of 8 hexahedra with `surface_with_edges` inside a `silver` outer element wireframe. | `tetraglass render gallery/threshold.json -o out.png` |
| `gallery/streamlines.json` | Bidirectional RK4(5) `"type": "streamlines"` (`vectors: "velocity"`, `4x4` plane seed grid) rendered with `"style": "tubes"` colored by `velocity:magnitude` (`viridis`) inside a `silver` outer element wireframe. | `tetraglass render gallery/streamlines.json -o out.png` |
| `gallery/glyphs.json` | 3D `"type": "glyphs"` (`vectors: "velocity"`, `shape: "arrow"`, `scale_by: "magnitude"`) colored by `velocity:magnitude` (`viridis`) inside a `silver` outer element wireframe. | `tetraglass render gallery/glyphs.json -o out.png` |
| `gallery/volume.json` | GPU ray-marched `"type": "resample"` (`dimensions: [32, 32, 32]`, `"style": "volume"`) colored by `linear` (`viridis`) with a piecewise-linear opacity `transfer_function` inside a `silver` outer element wireframe. | `tetraglass render gallery/volume.json -o out.png` |

## Filter gallery highlights

### 1. Plane Slice (`gallery/slice.json`)

```json
{
  "filters": {
    "outer": { "type": "surface", "input": "hex" },
    "cut": {
      "type": "slice",
      "input": "hex",
      "origin": [1.0, 1.0, 1.0],
      "normal": [1.0, -0.8, 0.9]
    }
  }
}
```

### 2. Multi-Isovalue Contour (`gallery/contour.json`)

```json
{
  "filters": {
    "outer": { "type": "surface", "input": "hex" },
    "iso": {
      "type": "contour",
      "input": "hex",
      "field": "linear",
      "values": [2.5, 5.0, 7.5, 10.0]
    }
  }
}
```

### 3. Exact Half-Space Clip (`gallery/clip.json`)

```json
{
  "filters": {
    "outer": { "type": "surface", "input": "hex" },
    "clipped": {
      "type": "clip",
      "input": "hex",
      "origin": [1.0, 1.0, 1.0],
      "normal": [1.0, -0.8, 0.9],
      "keep": "back",
      "mode": "exact"
    }
  }
}
```

### 4. Whole-Cell Scalar Threshold (`gallery/threshold.json`)

```json
{
  "filters": {
    "outer": { "type": "surface", "input": "hex" },
    "threshed": {
      "type": "threshold",
      "input": "hex",
      "field": "linear",
      "min": 0.0,
      "max": 9.0
    }
  }
}
```

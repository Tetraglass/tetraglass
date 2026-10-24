# Scene Format & Agent Workflow

Tetraglass scenes are plain, serializable JSON documents (`version: 1`) backed by a `schemars`-generated JSON Schema (`tetraglass schema`). Every frontend—the CLI (`render`, `validate`, `view`), the desktop viewer, and the Rust `SceneEvaluator`—consumes the exact same `tetraglass_scene::Scene` data structure (`D-004`).

For the complete field-by-field specification, see [`docs/scene-format.md`](../../scene-format.md).

## 1. The four-step agent loop

Coding agents and scripts can go from an unfamiliar mesh file to a verified figure without compiling any Rust code:

1. **Inspect the dataset:**
   ```sh
   tetraglass inspect --json path/to/mesh.vtu
   ```
   Read the exact point and cell array names, component counts, value ranges, bounding box, and time steps from the JSON output.
2. **Write a scene file (`scene.json`):**
   ```json
   {
     "version": 1,
     "sources": {
       "flow": { "file": "testdata/vtk/hex_linear.vtu" }
     },
     "filters": {
       "outer": { "type": "surface", "input": "flow" },
       "cut": {
         "type": "slice",
         "input": "flow",
         "origin": [1.0, 1.0, 1.0],
         "normal": [1.0, -0.8, 0.9]
       }
     },
     "view": {
       "show": [
         {
           "input": "cut",
           "style": "surface_with_edges",
           "color_by": "linear",
           "colormap": "viridis",
           "range": [0.0, 12.0],
           "colorbar": { "title": "linear", "position": "right" }
         },
         {
           "input": "outer",
           "style": "wireframe",
           "color": "silver"
         }
       ],
       "camera": {
         "fit": "all",
         "azimuth": 35.0,
         "elevation": 25.0
       },
       "axes": true,
       "title": "Diagonal Plane Slice"
     },
     "output": {
       "image": "slice.png",
       "size": [800, 600],
       "msaa": 4
     }
   }
   ```
3. **Validate before rendering:**
   ```sh
   tetraglass validate scene.json
   ```
   If an array name or filter input is misspelled, `validate` reports the exact JSON path and a Damerau-Levenshtein suggestion:
   ```text
   view.show[0].color_by: no array "linaer" on input "cut"; did you mean "linear"?
   ```
4. **Render and inspect:**
   ```sh
   tetraglass render scene.json -o slice.png
   ```

## 2. Array reference syntax (`color_by` and filter `field`)

Both `view.show[i].color_by` and filter `field` properties (`contour`, `threshold`) use a single compact grammar:

```text
[point:|cell:]<array_name>[:<component>]
```

- **Association prefix (optional):** `point:` or `cell:`. When omitted, `point_data` is searched first, then `cell_data`.
- **Component selector (optional):**
  - Scalars (`1` component): omit the suffix (`"pressure"` or `"cell:cell_id"`).
  - Vectors (`3` components): `:magnitude` (default when omitted in `color_by`), `:x`, `:y`, `:z`, or `:0`, `:1`, `:2`.
  - Symmetric 6-tensors (`6` components): `:xx`, `:yy`, `:zz`, `:xy`, `:yz`, `:xz`, `:magnitude`, or `:0`..`:5`.
  - Full 9-tensors (`9` components): `:0`..`:8` or `:magnitude`.

## 3. Evaluating a `Scene` in Rust

Use `SceneEvaluator` to evaluate a `Scene` in Rust with automatic input-hash caching (`PipelineHash`):

```rust,no_run
use std::path::Path;
use tetraglass_scene::{Scene, SceneEvaluator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scene = Scene::from_json_path("gallery/slice.json")?;
    let mut evaluator = SceneEvaluator::new();

    // First evaluation loads the source, linearizes, and runs the slice filter.
    let evaluated = evaluator.evaluate(&scene, Path::new("gallery"))?;
    println!("Rendered {} representations", evaluated.representations.len());

    // Changing a colormap or camera and re-evaluating reuses the cached filter outputs!
    Ok(())
}
```

---
name: add-filter
description: Step-by-step procedure for adding or changing a visualization filter (slice, clip, contour, threshold, streamlines, glyphs, resampling and others) in the filters crate, including the tetrahedral kernel, scene integration, tests and gallery scene. Use whenever a task adds, extends or fixes a filter or anything in the filters crate.
---

# Add or change a filter

## 1. Specify

Write down, in the implementation plan:

- inputs (which dataset types), parameters with units and defaults, output type (surface, unstructured grid, polylines, instances or a 3D texture);
- which point and cell arrays the output carries, and how they are interpolated;
- the milestone and roadmap id.

## 2. Implement on the linearized view

- The filter is a plain function in the filters crate: `fn slice(input: &TetView, params: &SliceParams) -> Result<Surface, FilterError>`. Names follow what the core crate actually defines.
- It consumes only the linearized tetrahedral view (D-005). Never branch on higher-order or polyhedral cell types inside a filter; if the view lacks something, extend the linearization instead.
- Slice, clip and contour share the marching-tetrahedra kernel (D-006). Extend the kernel rather than writing a second one.
- Every output cell records its source cell id, so picking and cell data refer to the original mesh.

**Kernel details that prevent subtle bugs:**

- Interpolate on edges with `t = (iso - a) / (b - a)`, clamped to [0, 1].
- Treat a value exactly equal to the isovalue as "above". One consistent rule avoids cracks and duplicate triangles when a plane passes through vertices.
- Deduplicate output vertices by the edge key `(min(p, q), max(p, q))` of source point ids, so outputs are watertight and arrays interpolate once per edge.
- In parallel code, give each chunk its own edge map and merge in chunk order. The output must be identical across runs and thread counts.

## 3. Integrate into the scene

- Add a variant to the scene's filter enum (`#[serde(tag = "type")]`), with doc comments on every field and a `JsonSchema` derive.
- Validate parameters against the data (array exists, component valid, plane normal non-zero) and give errors with a JSON path and a suggestion.
- Add the matching method to the Rust builder API.
- Follow `.agents/rules/scene-schema.md` (docs, schema snapshot, gallery).

## 4. Test

Cover, at minimum:

- an analytic case with a known answer (the plane slice of a unit cube has area 1; a sphere's distance field contours at its radius);
- every cell family in the test data: each linear type, the quadratic tet, the Lagrange hex and the polyhedral mesh;
- an empty result and a degenerate case (plane through vertices, or isovalue equal to a vertex value);
- array interpolation, using a linear field that must be reproduced exactly;
- determinism: the same output with one thread and with many.

## 5. Show it

Add a gallery scene and baseline through `/author-scene` and `/render-check`. Add a rustdoc example on the public function and a section in `docs/scene-format.md`.

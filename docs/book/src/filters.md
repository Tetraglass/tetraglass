# Linearization & Filters

Tetraglass stores higher-order Lagrange/serendipity cells and arbitrary polyhedra natively in `UnstructuredGrid`, and runs every visualization filter on a linearized tetrahedral view (`TetView`, `D-005`). Furthermore, `slice`, `contour`, and `clip` share a single deterministic marching-tetrahedra kernel (`D-006`).

## 1. The linearization layer (`TetView`)

Calling `linearize(&grid, &LinearizeParams { subdivisions })` (or `linearize_default(&grid)` with `subdivisions = 2`) produces a `TetView`:

- **Linear 3D cells (`Tetra`, `Hexahedron`, `Wedge`, `Pyramid`):** Decomposed into `1`, `6`, `3`, or `2` tetrahedra using face-consistent vertex-ID diagonal rules so shared internal faces are always watertight across mixed cell types.
- **Higher-order 3D cells (`Quadratic*`, `Lagrange*`):** Subdivided in reference coordinates `(r, s, t)` to `subdivisions` levels per edge; positions and point attributes are evaluated using the cell's exact Lagrange or serendipity basis functions.
- **Polyhedral cells (`Polyhedron`):** Decomposed by inserting the cell's volume-weighted centroid (with averaged point attributes) and fan-triangulating each polygonal face from its minimum-index vertex.
- **Source-cell provenance (`tet_source_cells`):** Every tetrahedron records the `u32` index of its originating cell in `UnstructuredGrid`, so `cell_data`, crinkle clipping, thresholding, and GPU click-to-probe picking always report original element IDs.

## 2. Filter reference

| Filter | Rust entry point | Scene JSON `"type"` | Input -> Output | Description |
| --- | --- | --- | --- | --- |
| External surface | `extract_external_surface` / `extract_tet_view_surface` | `"surface"` | `UnstructuredGrid` or `TetView` -> `SurfaceMesh` | Extracts faces belonging to exactly one 3D cell, tessellates curved higher-order faces, and records original element perimeter edges for wireframe rendering. |
| Plane slice | `slice` / `slice_grid` | `"slice"` | `TetView` -> `SurfaceMesh` | Evaluates signed distance to `(origin, normal)` and extracts the `0.0` isosurface via marching tetrahedra, interpolating all point arrays and mapping all cell arrays. |
| Isosurface contour | `contour` / `contour_grid` | `"contour"` | `TetView` -> `SurfaceMesh` | Extracts one or more isosurfaces `val == isovalues[k]` of a point scalar or vector component/magnitude (`ContourComponent`), interpolating every point array onto cut edges. |
| Half-space clip | `clip` / `clip_grid` | `"clip"` | `TetView` -> `TetView` | Clips a tetrahedral volume to a half-space (`keep: "back"` or `"front"`). In `"exact"` mode, cut tetrahedra are split via conformal 1-tet / 3-tet prism decomposition; in `"crinkle"` mode, whole original source cells (`tet_source_cells`) overlapping the kept half-space are retained intact. |
| Scalar threshold | `threshold` / `threshold_grid` | `"threshold"` | `TetView` -> `TetView` | Keeps whole original source cells (`tet_source_cells`) whose cell scalar (or all vertex scalars for point fields) satisfies `min <= val <= max` (supports one-sided and two-sided bounds). |
| Streamlines | `streamlines` / `streamlines_grid` | `"streamlines"` | `TetView` -> `PolylineSet` (`SurfaceMesh` via `"tubes"` / `"lines"`) | Integrates streamlines through a 3-component vector field using adaptive RK4(5) or fixed-step RK4 with BVH cell location (`TetLocator`), point/line/plane seeds, interpolated point/cell attributes plus `"arc_length"`, `"integration_time"`, and `"seed_id"`, and parallel-transported tube geometry (`PolylineSet::to_tube_surface`). |
| Vector glyphs | `glyphs` / `glyphs_grid` | `"glyphs"` | `TetView` -> `SurfaceMesh` | Samples a 3-component vector field at mesh vertices (or source-cell volume centroids) with configurable stride (`every`) and instantiates 3D `"arrow"`, `"cone"`, or `"sphere"` meshes oriented along the vector and scaled by `"magnitude"` or a fixed size, replicating all point and cell attributes plus `"glyph_magnitude"`. |

## 3. Chaining filters in JSON and Rust

Volume filters (`clip` and `threshold`) output a `TetView`, so they can feed into subsequent volume filters (`clip`, `threshold`) or surface-producing filters (`slice`, `contour`, `surface`), or be referenced directly in `view.show[i].input` (which automatically extracts the boundary surface via `extract_tet_view_surface`).

### Example: chaining `threshold` -> `clip` -> `slice` in JSON

```json
{
  "filters": {
    "high_temp": {
      "type": "threshold",
      "input": "mesh",
      "field": "linear",
      "min": 2.0,
      "max": 10.0
    },
    "half": {
      "type": "clip",
      "input": "high_temp",
      "normal": [1.0, 0.0, 0.0],
      "keep": "back",
      "mode": "exact"
    },
    "cut": {
      "type": "slice",
      "input": "half",
      "normal": [0.0, 0.0, 1.0]
    }
  }
}
```

## 4. Parallelism & determinism

Every filter in `tetraglass-filters` processes tetrahedra in fixed-size chunks (`DEFAULT_TET_CHUNK_SIZE = 8192`) in parallel on native targets and merges chunk-local vertex/edge tables in strict chunk order `0..num_chunks`. As a result, output vertices, triangles, tetrahedra, and interpolated attributes are **bit-for-bit identical** regardless of chunk size or CPU thread count.

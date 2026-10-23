# Decisions

Binding decisions for this project. Agents must not contradict an **Accepted** decision without the human's approval. **Proposed** decisions are defaults to confirm in the named task. Add new decisions at the end with the next number; never renumber.

| ID | Date | Status | Decision | Why |
| --- | --- | --- | --- | --- |
| D-001 | 2026-10-22 | Accepted | Scope is FEM/CFD unstructured meshes whose data fits in GPU memory. No out-of-core, distributed or MPI rendering. | Keeps a solo project feasible; removes most of VTK's complexity. |
| D-002 | 2026-10-22 | Accepted | The renderer takes the host's wgpu device and queue and draws into given textures; it never owns a window or event loop. | Embedding in Rust apps is the top-priority workflow. |
| D-003 | 2026-10-22 | Accepted | egui is the GUI for the viewer and the first host; the widget renders offscreen in egui-wgpu's `prepare` step and shows a texture. | Runs on desktop and web from one codebase; owning passes keeps depth, MSAA and transparency independent. |
| D-004 | 2026-10-22 | Accepted | The scene model is pure serde data with a generated JSON Schema. Agents use JSON scene files plus a CLI (`inspect`, `render`, `validate`, `schema`, `view`, `export`). | No compile step in the agent loop; one model for every frontend. |
| D-005 | 2026-10-22 | Accepted | Higher-order cells and polyhedra are stored natively; filters and rendering use a linearized tetrahedral view with a map back to source cells. | One implementation per filter. |
| D-006 | 2026-10-22 | Accepted | Slice, clip and contour share one marching-tetrahedra kernel. | Six filter families become four algorithms. |
| D-007 | 2026-10-22 | Accepted | The web target is WebGPU only; no WebGL2 fallback. | WebGL2 lacks compute and storage buffers; a second backend doubles testing. |
| D-008 | 2026-10-22 | Accepted | Readers take bytes; `core`, `io`, `filters` and `scene` build for wasm32 with no C dependencies. | Same readers and filters in the browser. |
| D-009 | 2026-10-22 | Accepted | Input formats: VTK (.vtu, .vtk, .pvd) and Gmsh (.msh). VTK node ordering is canonical in core. | Covers the target users without HDF5. |
| D-010 | 2026-10-22 | Accepted | Ray tracing: Blender first (export plus headless Cycles renders driven by the CLI), a built-in compute path tracer later, wgpu ray queries optional after that. | Publication quality early with no renderer work. |
| D-011 | 2026-10-22 | Accepted | Rust crates are MIT OR Apache-2.0. Code using Blender's Python API lives in `blender/` under GPL-3.0-or-later; the interface is data files and CLI arguments only; Blender is never bundled. | Matches Blender's licensing FAQ while keeping the library permissive. |
| D-012 | 2026-10-22 | Accepted | vtk-pure-rs is out of scope: not used, evaluated or depended on. | The developer's decision. |
| D-013 | 2026-10-22 | Proposed (confirm in M0.4) | Connectivity uses u32 indices, checked on load. | Halves index memory; consistent with D-001 and GPU index buffers. |
| D-014 | 2026-10-22 | Accepted | f64 positions and fields on the CPU; f32 relative to the dataset center on the GPU. | Precision on large-coordinate CFD domains. |
| D-015 | 2026-10-22 | Proposed (confirm in M0.9) | Image baselines are produced on Mesa lavapipe, the CI driver; comparison tolerance per `.agents/rules/testing.md`. | Stable, GPU-independent visual regression tests. |
| D-016 | 2026-10-22 | Accepted | Antigravity procedures are Agent Skills in `.agents/skills/`, not legacy workflows. | Antigravity retired legacy workflows on 2026-10-19. |
| D-017 | 2026-10-22 | Accepted | The project is named Tetraglass. Crates are `tetraglass-core`, `tetraglass-io` and so on; the umbrella crate and CLI binary are `tetraglass`. The repository is `github.com/Tetraglass/tetraglass`. | Tetra (every cell becomes tetrahedra) plus looking glass. All crate names were free on crates.io on 2026-10-22, and the developer secured the Tetraglass GitHub organization. Replaces the earlier choice Tetrascope, whose GitHub name was taken. |

## Open questions

- [x] Project name, crate prefix and repository: Tetraglass, `github.com/Tetraglass/tetraglass` (D-017).
- [ ] Reuse vtkio and mshio, or write focused readers (M0.3).
- [ ] Which public FEM/CFD datasets become the reference test set, and under what licenses.
- [ ] Supported Blender versions, and whether `blender/` moves to its own repository.
- [ ] When to add Python bindings (on demand, after 0.2).

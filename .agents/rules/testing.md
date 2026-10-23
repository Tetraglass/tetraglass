---
trigger: model_decision
description: "Testing strategy for this workspace. Apply when writing or changing tests, adding a feature, fixing a bug, adding test data, or touching image regression baselines."
---

# Testing rules

**What every change needs.**
- New behavior: unit tests for the logic, plus an integration test if it crosses a crate boundary.
- Bug fix: a regression test that fails before the fix.
- Rendering change: a gallery scene and a baseline image (`/render-check`).

**Filters.** Analytic cases first: a plane slice of a unit cube has area 1; contouring the distance field of a sphere gives points at the radius; a linear field is reproduced exactly by interpolation. Then cover every cell family: each linear type, a quadratic tet, a third-order Lagrange hex and a polyhedron. Include empty results and degenerate cases (a plane through vertices or along a face).

**Linearization.** Property tests with `proptest`: total volume is preserved; linear fields interpolate exactly; every output tet maps back to a valid source cell.

**Readers.** Round-trip tests through the `.vtu` writer; one node-ordering test per higher-order cell type; truncated and corrupted inputs must return errors without panicking.

**Image regression.**
- Baselines live in `tests/baselines/` and are produced on Mesa lavapipe (software Vulkan), the same driver CI uses.
- Initial comparison policy: a pixel differs if any channel differs by more than 2 (of 255); a test fails if more than 0.5% of pixels differ. Adjust only with a decision entry.
- Never replace a baseline to make a test pass without explaining why in `docs/progress.md` and the commit message.
- Local runs on a real GPU may differ slightly; CI is authoritative.

**Test data.** Small (under about 1 MB per file), reproducible and committed with the script that made it, under `testdata/` (`/make-test-data`). Larger datasets are downloaded by a script and never committed.

**Performance.** Benchmarks use `criterion` in a `benches/` folder and a synthetic mesh generated in code. They are not part of `/verify`.

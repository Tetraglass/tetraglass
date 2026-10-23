---
name: milestone-gate
description: Checks whether the current milestone's gate in docs/roadmap.md is met, demonstrates it with evidence, and for release milestones (M2 = 0.1, M4 = 0.2) runs the release checklist up to, but not including, publishing. Use when all tasks of a milestone are ticked, or when the user asks whether a milestone is done, wants a demo, a release, a changelog or a retrospective.
---

# Milestone gate

A gate is a demo someone could watch, not a feature count. This skill proves the demo works and closes the milestone cleanly.

## 1. Check

- Every task of the milestone in `docs/roadmap.md` is ticked. If not, list what is open and stop.
- `/verify` passes on a clean checkout (`cargo clean` first if in doubt).
- `/render-check` passes for the whole gallery.

## 2. Demonstrate

Run the gate exactly as `docs/roadmap.md` words it, from a fresh terminal, and capture evidence: the commands, their output, and screenshots or rendered images. Save the evidence under `docs/milestones/<id>-gate.md` with images next to it.

For M1 the gate involves an agent: start a fresh session that only has the CLI and `docs/scene-format.md`, give it a dataset from `testdata/`, and record whether it reaches a correct PNG without compiling anything.

## 3. Release checklist (M2 and M4 only)

1. Pick the version (0.1.0 or 0.2.0) and set it in `[workspace.package]` and in every internal entry of `[workspace.dependencies]`.
2. Write `CHANGELOG.md` from `docs/progress.md` entries since the last release, grouped as Added, Changed and Fixed.
3. Check every published crate has a description, README, license, repository and documentation link; set `publish = true` where it applies.
4. Run `cargo publish --dry-run -p <crate>` in dependency order: core, io, filters, scene, render, egui, export, cli.
5. Check rustdoc builds without warnings: `cargo doc --workspace --no-deps`.
6. **Stop and ask the user** before the real `cargo publish`, creating the git tag and announcing anything.

## 4. Close

- Mark the milestone done in `docs/roadmap.md` with the date.
- Retrospective in `docs/progress.md`: planned vs. actual time, what was cut or moved, and what to change in the next milestone's plan. Update the next milestone's estimates if they now look wrong.
- Suggest the first task of the next milestone.

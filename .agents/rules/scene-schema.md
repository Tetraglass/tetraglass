---
trigger: glob
globs: "crates/*-scene/**, crates/*-cli/**, docs/scene-format.md, gallery/**"
description: "Rules for the serializable scene model, its JSON Schema, versioning, validation messages and gallery scenes."
---

# Scene model and schema rules

The scene is the contract between every frontend: the Rust builder, JSON files, the viewer, the web viewer and the Blender exporter. Treat it like a public API.

**Serde conventions.**
- Structs use `#[serde(deny_unknown_fields, rename_all = "snake_case")]`.
- Filters and representations are internally tagged enums: `#[serde(tag = "type", rename_all = "snake_case")]`.
- Every type derives `schemars::JsonSchema`. Every field has a doc comment; it becomes the schema description agents read.
- Defaults are explicit (`#[serde(default = "...")]`) and documented in `docs/scene-format.md`.

**Versioning.** The top-level `version` field is required. A breaking change bumps it and adds a migration from the previous version, with a test that loads an old file. Until release 0.2 the schema may still change, but never silently.

**Validation.** Validation runs after parsing and against the loaded data. Every error carries a JSON path and, where possible, a fix, for example: `view.show[0].color_by: no array "presure"; did you mean "pressure"?`. Suggestions use edit distance over the names that actually exist.

**Every scene change must also:**
1. update `docs/scene-format.md`,
2. update the snapshot test of the `schema` command output,
3. add or adjust at least one scene in `gallery/`, rendered through `/render-check`.

**Gallery.** One scene per file in `gallery/<topic>-<what>.json`, small data only, each listed in `gallery/README.md` with one line saying what it shows. Gallery scenes double as tests and as agent examples, so keep them minimal and correct.

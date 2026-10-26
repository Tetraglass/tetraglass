//! Draft-to-v1 scene migration and schema freeze helpers (`M4.6`).
//!
//! Version 1 (`SCENE_VERSION = 1`) of the Tetraglass scene schema is frozen as of milestone `M4.6`.
//! Strict parsing via [`Scene::from_json_str`] enforces `version == 1` and `deny_unknown_fields`
//! everywhere. Callers with earlier pre-freeze draft scenes (from M1–M4 prototyping) can use
//! [`migrate_scene_value`], [`migrate_scene_json_str`], or [`Scene::from_draft_json_str`] to
//! normalize legacy draft keys into a canonical v1 [`Scene`].

use serde_json::{Value, json};

use crate::{SCENE_VERSION, Scene, SceneError, ValidationIssue, ValidationReport};

/// Summary of draft-to-v1 migration transformations applied to a scene JSON document (`M4.6`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationReport {
    /// Original `version` field in the input JSON (`None` if omitted in an early draft).
    pub from_version: Option<u32>,
    /// Target schema version after migration ([`SCENE_VERSION`], `1`).
    pub to_version: u32,
    /// Ordered list of human-readable migration actions performed (`empty` when already canonical v1).
    pub changes: Vec<String>,
}

impl MigrationReport {
    /// Returns `true` if at least one draft field or version tag was migrated.
    #[must_use]
    pub fn was_modified(&self) -> bool {
        !self.changes.is_empty()
    }
}

/// Normalizes an in-memory JSON [`Value`] from earlier M1–M4 scene drafts into canonical
/// version 1 (`SCENE_VERSION = 1`) schema form (`M4.6`).
///
/// Canonical v1 scenes are left untouched with an empty `changes` list in the returned
/// [`MigrationReport`]. Unsupported future schema versions (`version >= 2`) return a
/// [`SceneError::Validation`] error on `"version"`.
pub fn migrate_scene_value(value: &mut Value) -> Result<MigrationReport, SceneError> {
    let Some(root) = value.as_object_mut() else {
        return Err(SceneError::Validation(ValidationReport {
            issues: vec![ValidationIssue::new(
                "",
                "scene root must be a JSON object",
                None,
            )],
        }));
    };

    let mut changes = Vec::new();
    let from_version = match root.get("version") {
        None => {
            root.insert("version".to_string(), json!(SCENE_VERSION));
            changes.push(format!("version: inserted missing version {SCENE_VERSION}"));
            None
        }
        Some(v) => match v.as_u64() {
            Some(0) => {
                root.insert("version".to_string(), json!(SCENE_VERSION));
                changes.push(format!(
                    "version: upgraded draft version 0 to {SCENE_VERSION}"
                ));
                Some(0)
            }
            Some(1) => Some(1),
            Some(other) => {
                return Err(SceneError::Validation(ValidationReport {
                    issues: vec![ValidationIssue::new(
                        "version",
                        format!(
                            "unsupported scene version {other}; supported versions are draft 0 and final {SCENE_VERSION}"
                        ),
                        Some(format!("set \"version\": {SCENE_VERSION}")),
                    )],
                }));
            }
            None => {
                return Err(SceneError::Validation(ValidationReport {
                    issues: vec![ValidationIssue::new(
                        "version",
                        "scene version must be a non-negative integer",
                        Some(format!("set \"version\": {SCENE_VERSION}")),
                    )],
                }));
            }
        },
    };

    if let Some(Value::Object(filters)) = root.get_mut("filters") {
        for (fname, fval) in filters.iter_mut() {
            let Value::Object(fobj) = fval else {
                continue;
            };
            if fobj.get("type").and_then(Value::as_str) == Some("volume") {
                fobj.insert("type".to_string(), json!("resample"));
                changes.push(format!(
                    "filters.{fname}.type: renamed draft filter type \"volume\" to \"resample\""
                ));
            }

            match fobj.get("type").and_then(Value::as_str) {
                Some("contour") => {
                    if !fobj.contains_key("field")
                        && let Some(arr) = fobj.remove("array")
                    {
                        fobj.insert("field".to_string(), arr);
                        changes.push(format!(
                            "filters.{fname}: renamed draft key \"array\" to \"field\""
                        ));
                    }
                    if !fobj.contains_key("values")
                        && let Some(val) = fobj.remove("value")
                    {
                        let wrapped = if val.is_array() {
                            val
                        } else {
                            Value::Array(vec![val])
                        };
                        fobj.insert("values".to_string(), wrapped);
                        changes.push(format!(
                            "filters.{fname}: migrated single \"value\" to \"values\" array"
                        ));
                    }
                }
                Some("threshold") => {
                    if !fobj.contains_key("field")
                        && let Some(arr) = fobj.remove("array")
                    {
                        fobj.insert("field".to_string(), arr);
                        changes.push(format!(
                            "filters.{fname}: renamed draft key \"array\" to \"field\""
                        ));
                    }
                    if let Some(Value::Array(bounds)) = fobj.get("range").cloned()
                        && bounds.len() == 2
                    {
                        fobj.remove("range");
                        if !fobj.contains_key("min") && !bounds[0].is_null() {
                            fobj.insert("min".to_string(), bounds[0].clone());
                        }
                        if !fobj.contains_key("max") && !bounds[1].is_null() {
                            fobj.insert("max".to_string(), bounds[1].clone());
                        }
                        changes.push(format!(
                            "filters.{fname}: split draft \"range\" into \"min\" and \"max\""
                        ));
                    }
                }
                Some("streamlines") | Some("glyphs") => {
                    for legacy_key in ["vector", "field"] {
                        if !fobj.contains_key("vectors")
                            && let Some(v) = fobj.remove(legacy_key)
                        {
                            fobj.insert("vectors".to_string(), v);
                            changes.push(format!(
                                "filters.{fname}: renamed draft key \"{legacy_key}\" to \"vectors\""
                            ));
                        }
                    }
                }
                Some("resample") => {
                    if !fobj.contains_key("dimensions")
                        && let Some(dims) = fobj.remove("dims")
                    {
                        fobj.insert("dimensions".to_string(), dims);
                        changes.push(format!(
                            "filters.{fname}: renamed draft key \"dims\" to \"dimensions\""
                        ));
                    }
                }
                _ => {}
            }
        }
    }

    if let Some(Value::Object(view)) = root.get_mut("view")
        && let Some(Value::Array(show)) = view.get_mut("show")
    {
        for (idx, rep_val) in show.iter_mut().enumerate() {
            let Value::Object(rep) = rep_val else {
                continue;
            };
            if !rep.contains_key("input")
                && let Some(ds) = rep.remove("dataset")
            {
                rep.insert("input".to_string(), ds);
                changes.push(format!(
                    "view.show[{idx}]: renamed draft key \"dataset\" to \"input\""
                ));
            }
            if !rep.contains_key("color_by")
                && let Some(field) = rep.remove("field")
            {
                rep.insert("color_by".to_string(), field);
                changes.push(format!(
                    "view.show[{idx}]: renamed draft key \"field\" to \"color_by\""
                ));
            }
        }
    }

    if let Some(Value::Object(output)) = root.get_mut("output")
        && let Some(Value::Object(frames)) = output.get_mut("frames")
    {
        if !frames.contains_key("from")
            && let Some(start) = frames.remove("start")
        {
            frames.insert("from".to_string(), start);
            changes.push("output.frames: renamed draft key \"start\" to \"from\"".to_string());
        }
        if !frames.contains_key("to")
            && let Some(end) = frames.remove("end")
        {
            frames.insert("to".to_string(), end);
            changes.push("output.frames: renamed draft key \"end\" to \"to\"".to_string());
        }
    }

    Ok(MigrationReport {
        from_version,
        to_version: SCENE_VERSION,
        changes,
    })
}

/// Parses a JSON string that may use earlier M1–M4 draft conventions, migrates it to
/// canonical schema version 1, and deserializes the resulting [`Scene`] (`M4.6`).
pub fn migrate_scene_json_str(json: &str) -> Result<(Scene, MigrationReport), SceneError> {
    let mut value: Value = serde_json::from_str(json)?;
    let report = migrate_scene_value(&mut value)?;
    let scene: Scene = serde_json::from_value(value)?;
    Ok((scene, report))
}

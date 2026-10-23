//! Serializable scene model, JSON Schema and cached pipeline evaluation (`docs/scene-format.md`).
//!
//! `tetraglass-scene` defines the declarative scene contract shared by JSON scene files,
//! the CLI (`tetraglass inspect`, `render`, `validate`, `schema`), the desktop and web viewers,
//! and the Rust API (`D-004`). It compiles for `wasm32-unknown-unknown` with no GPU or C
//! dependencies (`D-008`).
//!
//! # Examples
//!
//! ```
//! use tetraglass_scene::{ColormapName, Filter, RangePreset, RangeSpec, RepresentationStyle, Scene, SCENE_VERSION};
//!
//! let json = r#"{
//!   "version": 1,
//!   "sources": {
//!     "mesh": { "file": "testdata/vtk/mixed_linear.vtu" }
//!   },
//!   "filters": {
//!     "outer": { "type": "surface", "input": "mesh" }
//!   },
//!   "view": {
//!     "show": [
//!       { "input": "outer", "color_by": "linear", "colorbar": true }
//!     ],
//!     "camera": { "fit": "all", "azimuth": 30, "elevation": 20 }
//!   },
//!   "output": { "image": "out.png", "size": [1200, 800] }
//! }"#;
//!
//! let scene = Scene::from_json_str(json).expect("valid v1 scene");
//! assert_eq!(scene.version, SCENE_VERSION);
//! assert_eq!(scene.sources["mesh"].linearize_level, 2);
//! assert_eq!(scene.filters["outer"], Filter::surface("mesh"));
//! assert_eq!(scene.view.show[0].style, RepresentationStyle::Surface);
//! assert_eq!(scene.view.show[0].colormap, ColormapName::Viridis);
//! assert_eq!(scene.view.show[0].range, RangeSpec::Preset(RangePreset::Auto));
//! ```

mod array_ref;
mod error;
mod types;
mod validate;

pub use array_ref::{
    ArrayAssociation, ComponentSelector, ParsedArrayRef, ResolvedComponent, TensorComponent,
    VectorAxis,
};
pub use error::SceneError;
pub use types::{
    BackgroundGradient, BackgroundSpec, CameraConfig, ColorSpec, ColorbarOptions, ColorbarPosition,
    ColorbarSpec, ColormapName, CyclesConfig, DEFAULT_CYCLES_SAMPLES, DEFAULT_LINEARIZE_LEVEL,
    DEFAULT_MSAA_SAMPLES, DEFAULT_OUTPUT_SIZE, DEFAULT_POINT_SIZE, DEFAULT_VIEW_ANGLE_DEGREES,
    Filter, FramesConfig, LightingPreset, Output, ProjectionKind, RangePreset, RangeSpec,
    RenderEngine, Representation, RepresentationStyle, SCENE_VERSION, Scene, Source, SourceFormat,
    TimePreset, TimeSelector, TimeStepIndex, View,
};
pub use validate::{
    ArrayMetadata, DatasetMetadata, ValidationIssue, ValidationReport, closest_match,
    damerau_levenshtein, validate_scene, validate_scene_structure, validate_scene_with_metadata,
};

/// Generates the authoritative JSON Schema for [`Scene`] (used by `tetraglass schema`).
#[must_use]
pub fn scene_json_schema() -> schemars::Schema {
    schemars::schema_for!(Scene)
}

/// Generates the pretty-printed JSON Schema string for [`Scene`].
pub fn scene_json_schema_string() -> Result<String, SceneError> {
    let schema = scene_json_schema();
    Ok(serde_json::to_string_pretty(&schema)?)
}

#[cfg(test)]
mod tests {
    use tetraglass_core::ComponentCount;
    use tetraglass_io::MeshFormat;

    use super::*;

    #[test]
    fn parses_minimal_scene_and_populates_all_documented_defaults() {
        let json = r#"{
            "version": 1,
            "sources": {
                "flow": { "file": "results.pvd" }
            },
            "view": {
                "show": [
                    { "input": "flow" }
                ]
            }
        }"#;

        let scene = Scene::from_json_str(json).expect("minimal scene should parse");
        assert_eq!(scene.version, SCENE_VERSION);
        assert_eq!(scene.sources.len(), 1);
        assert!(scene.filters.is_empty());
        assert!(scene.output.is_none());

        let src = &scene.sources["flow"];
        assert_eq!(src.file, "results.pvd");
        assert_eq!(src.format, None);
        assert_eq!(src.linearize_level, 2);

        assert_eq!(scene.view.show.len(), 1);
        let rep = &scene.view.show[0];
        assert_eq!(rep.input, "flow");
        assert_eq!(rep.style, RepresentationStyle::Surface);
        assert_eq!(rep.color_by, None);
        assert_eq!(rep.color, ColorSpec::Named("lightgray".to_string()));
        assert_eq!(rep.colormap, ColormapName::Viridis);
        assert_eq!(rep.range, RangeSpec::AUTO);
        assert!(!rep.log_scale);
        assert_eq!(rep.center, None);
        assert_eq!(rep.opacity, 1.0);
        assert_eq!(rep.colorbar, ColorbarSpec::Enabled(false));
        assert!(!rep.colorbar.is_enabled());
        assert_eq!(rep.edge_color, ColorSpec::Named("black".to_string()));
        assert_eq!(rep.point_size, 3.0);
        assert!(rep.visible);

        let cam = &scene.view.camera;
        assert_eq!(cam.fit, "all");
        assert_eq!(cam.azimuth, 0.0);
        assert_eq!(cam.elevation, 0.0);
        assert_eq!(cam.roll, 0.0);
        assert_eq!(cam.zoom, 1.0);
        assert_eq!(cam.position, None);
        assert_eq!(cam.focal_point, None);
        assert_eq!(cam.view_up, None);
        assert_eq!(cam.projection, ProjectionKind::Perspective);
        assert_eq!(cam.view_angle, 30.0);

        assert_eq!(scene.view.lights, LightingPreset::Default);
        assert_eq!(
            scene.view.background,
            BackgroundSpec::Solid(ColorSpec::Named("white".to_string()))
        );
        assert!(scene.view.axes);
        assert_eq!(scene.view.title, None);
    }

    #[test]
    fn parses_full_scene_and_roundtrips_json() {
        let json = r##"{
            "version": 1,
            "sources": {
                "mesh": {
                    "file": "testdata/vtk/mixed_linear.vtu",
                    "format": "vtu",
                    "linearize_level": 3
                }
            },
            "filters": {
                "hull": {
                    "type": "surface",
                    "input": "mesh"
                }
            },
            "view": {
                "show": [
                    {
                        "input": "hull",
                        "style": "surface_with_edges",
                        "color_by": "velocity:magnitude",
                        "color": [0.2, 0.4, 0.8],
                        "colormap": "cool_warm",
                        "range": [-10.0, 10.0],
                        "log_scale": false,
                        "center": 0.0,
                        "opacity": 0.85,
                        "colorbar": {
                            "title": "Speed [m/s]",
                            "position": "left"
                        },
                        "edge_color": "#223344",
                        "point_size": 5.0,
                        "visible": true
                    }
                ],
                "camera": {
                    "fit": "hull",
                    "azimuth": 45.0,
                    "elevation": 25.0,
                    "roll": 5.0,
                    "zoom": 1.25,
                    "position": [1.0, 2.0, 3.0],
                    "focal_point": [0.5, 0.5, 0.5],
                    "view_up": [0.0, 0.0, 1.0],
                    "projection": "orthographic",
                    "view_angle": 40.0
                },
                "lights": "headlight",
                "background": {
                    "top": "skyblue",
                    "bottom": [1.0, 1.0, 1.0]
                },
                "axes": false,
                "title": "Mixed Linear Mesh"
            },
            "output": {
                "image": "out.png",
                "size": [1600, 900],
                "time": "last",
                "frames": {
                    "from": "first",
                    "to": { "step": 10 },
                    "pattern": "frame_%04d.png"
                },
                "msaa": 4,
                "engine": "cycles",
                "cycles": {
                    "samples": 256,
                    "device": "gpu",
                    "save_blend": "scene.blend"
                }
            }
        }"##;

        let scene = Scene::from_json_str(json).expect("full scene should parse");
        assert_eq!(scene.sources["mesh"].format, Some(SourceFormat::Vtu));
        assert_eq!(
            MeshFormat::from(scene.sources["mesh"].format.unwrap()),
            MeshFormat::Vtu
        );
        assert_eq!(SourceFormat::from(MeshFormat::VtkLegacy), SourceFormat::Vtk);
        assert_eq!(scene.filters["hull"].input(), "mesh");

        let rep = &scene.view.show[0];
        assert_eq!(rep.style, RepresentationStyle::SurfaceWithEdges);
        assert_eq!(rep.colormap, ColormapName::CoolWarm);
        assert_eq!(rep.range, RangeSpec::bounds(-10.0, 10.0));
        assert_eq!(rep.center, Some(0.0));
        assert!(rep.colorbar.is_enabled());
        assert_eq!(
            rep.colorbar,
            ColorbarSpec::Options(ColorbarOptions {
                title: Some("Speed [m/s]".to_string()),
                position: ColorbarPosition::Left,
            })
        );

        let out = scene.output.as_ref().expect("output present");
        assert_eq!(out.size, [1600, 900]);
        assert_eq!(out.time, TimeSelector::LAST);
        assert_eq!(out.engine, RenderEngine::Cycles);
        let frames = out.frames.as_ref().expect("frames present");
        assert_eq!(frames.from, TimeSelector::FIRST);
        assert_eq!(frames.to, TimeSelector::step(10));
        assert_eq!(frames.pattern, "frame_%04d.png");
        let cycles = out.cycles.as_ref().expect("cycles present");
        assert_eq!(cycles.samples, 256);
        assert_eq!(cycles.device, "gpu");
        assert_eq!(cycles.save_blend.as_deref(), Some("scene.blend"));

        // Round-trip through pretty and compact JSON
        let pretty = scene.to_json_string_pretty().expect("serialize pretty");
        let roundtripped = Scene::from_json_str(&pretty).expect("re-parse pretty JSON");
        assert_eq!(roundtripped, scene);

        let compact = scene.to_json_string().expect("serialize compact");
        let from_bytes = Scene::from_json_bytes(compact.as_bytes()).expect("parse bytes");
        assert_eq!(from_bytes, scene);
    }

    #[test]
    fn output_and_cycles_defaults_match_spec() {
        let json = r#"{
            "version": 1,
            "sources": { "s": { "file": "a.vtu" } },
            "view": { "show": [{ "input": "s" }] },
            "output": {
                "image": "fig.png",
                "cycles": {}
            }
        }"#;
        let scene = Scene::from_json_str(json).expect("parse output defaults");
        let out = scene.output.expect("output section");
        assert_eq!(out.image, "fig.png");
        assert_eq!(out.size, [1200, 800]);
        assert_eq!(out.time, TimeSelector::FIRST);
        assert_eq!(out.frames, None);
        assert_eq!(out.msaa, 4);
        assert_eq!(out.engine, RenderEngine::Raster);
        let cycles = out.cycles.expect("cycles section");
        assert_eq!(cycles.samples, 128);
        assert_eq!(cycles.device, "auto");
        assert_eq!(cycles.save_blend, None);
    }

    #[test]
    fn time_and_range_selectors_parse_all_variants() {
        let auto: RangeSpec = serde_json::from_str(r#""auto""#).unwrap();
        assert_eq!(auto, RangeSpec::AUTO);
        let all_steps: RangeSpec = serde_json::from_str(r#""auto_all_steps""#).unwrap();
        assert_eq!(all_steps, RangeSpec::AUTO_ALL_STEPS);
        let bounds: RangeSpec = serde_json::from_str("[0.0, 100.0]").unwrap();
        assert_eq!(bounds, RangeSpec::bounds(0.0, 100.0));
        assert!(serde_json::from_str::<RangeSpec>(r#""invalid_mode""#).is_err());
        assert!(serde_json::from_str::<RangeSpec>("[1.0, 2.0, 3.0]").is_err());

        let t_first: TimeSelector = serde_json::from_str(r#""first""#).unwrap();
        assert_eq!(t_first, TimeSelector::FIRST);
        let t_last: TimeSelector = serde_json::from_str(r#""last""#).unwrap();
        assert_eq!(t_last, TimeSelector::LAST);
        let t_step: TimeSelector = serde_json::from_str(r#"{ "step": 3 }"#).unwrap();
        assert_eq!(t_step, TimeSelector::step(3));
        let t_val: TimeSelector = serde_json::from_str("0.25").unwrap();
        assert_eq!(t_val, TimeSelector::Value(0.25));
        assert!(serde_json::from_str::<TimeSelector>(r#"{ "step": 1, "extra": 2 }"#).is_err());
    }

    #[test]
    fn color_spec_resolves_css_names_hex_and_rgb() {
        assert_eq!(
            ColorSpec::Named("white".to_string()).resolve_rgb(),
            Some([1.0, 1.0, 1.0])
        );
        assert_eq!(
            ColorSpec::Named("BLACK".to_string()).resolve_rgb(),
            Some([0.0, 0.0, 0.0])
        );
        assert!(
            ColorSpec::Named("lightgray".to_string())
                .resolve_rgb()
                .is_some()
        );
        assert_eq!(
            ColorSpec::Named("#ff0000".to_string()).resolve_rgb(),
            Some([1.0, 0.0, 0.0])
        );
        assert_eq!(
            ColorSpec::Rgb([0.25, 0.5, 0.75]).resolve_rgb(),
            Some([0.25, 0.5, 0.75])
        );

        // Out-of-range RGB and unknown color names return None
        assert_eq!(ColorSpec::Rgb([-0.1, 0.5, 0.5]).resolve_rgb(), None);
        assert_eq!(ColorSpec::Rgb([0.0, 1.2, 0.5]).resolve_rgb(), None);
        assert_eq!(
            ColorSpec::Named("not_a_real_color".to_string()).resolve_rgb(),
            None
        );
        assert!(!ColorSpec::known_color_names().is_empty());
    }

    #[test]
    fn denies_unknown_fields_everywhere() {
        // 1. Unknown top-level key
        assert!(
            Scene::from_json_str(
                r#"{
                "version": 1,
                "sources": { "s": { "file": "a.vtu" } },
                "view": { "show": [{ "input": "s" }] },
                "unknown_top": true
            }"#
            )
            .is_err()
        );

        // 2. Unknown key in source
        assert!(
            Scene::from_json_str(
                r#"{
                "version": 1,
                "sources": { "s": { "file": "a.vtu", "path": "b.vtu" } },
                "view": { "show": [{ "input": "s" }] }
            }"#
            )
            .is_err()
        );

        // 3. Unknown key in surface filter
        assert!(
            Scene::from_json_str(
                r#"{
                "version": 1,
                "sources": { "s": { "file": "a.vtu" } },
                "filters": { "f": { "type": "surface", "input": "s", "extra": 1 } },
                "view": { "show": [{ "input": "f" }] }
            }"#
            )
            .is_err()
        );

        // 4. Unknown filter type in M1
        assert!(
            Scene::from_json_str(
                r#"{
                "version": 1,
                "sources": { "s": { "file": "a.vtu" } },
                "filters": { "f": { "type": "nonexistent_filter", "input": "s" } },
                "view": { "show": [{ "input": "f" }] }
            }"#
            )
            .is_err()
        );

        // 5. Unknown key in representation
        assert!(
            Scene::from_json_str(
                r#"{
                "version": 1,
                "sources": { "s": { "file": "a.vtu" } },
                "view": { "show": [{ "input": "s", "colour_by": "p" }] }
            }"#
            )
            .is_err()
        );

        // 6. Unknown key in camera
        assert!(
            Scene::from_json_str(
                r#"{
                "version": 1,
                "sources": { "s": { "file": "a.vtu" } },
                "view": { "show": [{ "input": "s" }], "camera": { "fov": 45 } }
            }"#
            )
            .is_err()
        );

        // 7. Unknown key in output
        assert!(
            Scene::from_json_str(
                r#"{
                "version": 1,
                "sources": { "s": { "file": "a.vtu" } },
                "view": { "show": [{ "input": "s" }] },
                "output": { "image": "out.png", "width": 800 }
            }"#
            )
            .is_err()
        );
    }

    #[test]
    fn parses_and_resolves_array_references() {
        let simple = ParsedArrayRef::parse("pressure").unwrap();
        assert_eq!(simple.association, None);
        assert_eq!(simple.name, "pressure");
        assert_eq!(simple.component, None);
        assert_eq!(simple.to_string(), "pressure");
        assert_eq!(
            simple.resolve_component(ComponentCount::Scalar),
            Some(ResolvedComponent::Component(0))
        );
        assert_eq!(
            simple.resolve_component(ComponentCount::Vector3),
            Some(ResolvedComponent::Magnitude)
        );

        let vec_mag = ParsedArrayRef::parse("velocity:magnitude").unwrap();
        assert_eq!(vec_mag.association, None);
        assert_eq!(vec_mag.name, "velocity");
        assert_eq!(vec_mag.component, Some(ComponentSelector::Magnitude));
        assert_eq!(vec_mag.to_string(), "velocity:magnitude");
        assert_eq!(
            vec_mag.resolve_component(ComponentCount::Vector3),
            Some(ResolvedComponent::Magnitude)
        );
        // Scalars do not have a magnitude selector
        assert_eq!(vec_mag.resolve_component(ComponentCount::Scalar), None);

        let point_y = ParsedArrayRef::parse("point:velocity:y").unwrap();
        assert_eq!(point_y.association, Some(ArrayAssociation::Point));
        assert_eq!(point_y.name, "velocity");
        assert_eq!(
            point_y.component,
            Some(ComponentSelector::Vector(VectorAxis::Y))
        );
        assert_eq!(point_y.to_string(), "point:velocity:y");
        assert_eq!(
            point_y.resolve_component(ComponentCount::Vector3),
            Some(ResolvedComponent::Component(1))
        );
        assert_eq!(point_y.resolve_component(ComponentCount::Scalar), None);

        let cell_idx = ParsedArrayRef::parse("cell:stress:3").unwrap();
        assert_eq!(cell_idx.association, Some(ArrayAssociation::Cell));
        assert_eq!(cell_idx.name, "stress");
        assert_eq!(cell_idx.component, Some(ComponentSelector::Index(3)));
        assert_eq!(cell_idx.to_string(), "cell:stress:3");
        assert_eq!(
            cell_idx.resolve_component(ComponentCount::SymmetricTensor6),
            Some(ResolvedComponent::Component(3))
        );
        assert_eq!(cell_idx.resolve_component(ComponentCount::Vector3), None);

        let tensor_xy = ParsedArrayRef::parse("stress:xy").unwrap();
        assert_eq!(
            tensor_xy.component,
            Some(ComponentSelector::Tensor(TensorComponent::Xy))
        );
        assert_eq!(
            tensor_xy.resolve_component(ComponentCount::SymmetricTensor6),
            Some(ResolvedComponent::Component(3))
        );
        assert_eq!(
            tensor_xy.resolve_component(ComponentCount::Tensor9),
            Some(ResolvedComponent::Component(1))
        );

        // Invalid references
        assert!(ParsedArrayRef::parse("").is_err());
        assert!(ParsedArrayRef::parse("point:").is_err());
        assert!(ParsedArrayRef::parse("velocity:").is_err());
        assert!(ParsedArrayRef::parse(":x").is_err());
        assert!(ParsedArrayRef::parse("velocity:x:y").is_err());
        assert!(ParsedArrayRef::parse("velocity:unknown_comp").is_err());
    }

    #[test]
    fn generates_valid_json_schema_with_descriptions() {
        let schema_str = scene_json_schema_string().expect("generate JSON schema string");
        let value: serde_json::Value =
            serde_json::from_str(&schema_str).expect("schema is valid JSON");

        let obj = value.as_object().expect("schema root is an object");
        assert_eq!(obj.get("title").and_then(|v| v.as_str()), Some("Scene"));
        assert!(
            obj.get("description")
                .and_then(|v| v.as_str())
                .is_some_and(|d| !d.is_empty()),
            "root Scene schema must have a description"
        );

        let props = obj
            .get("properties")
            .and_then(|v| v.as_object())
            .expect("Scene properties");
        for key in ["version", "sources", "filters", "view", "output"] {
            let prop = props
                .get(key)
                .and_then(|v| v.as_object())
                .unwrap_or_else(|| panic!("missing property `{key}` in Scene schema"));
            assert!(
                prop.get("description")
                    .and_then(|v| v.as_str())
                    .is_some_and(|d| !d.is_empty()),
                "property `{key}` must carry a non-empty doc description in the schema"
            );
        }

        // Top-level Scene disallows unknown fields (`additionalProperties: false`)
        assert_eq!(
            obj.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );
    }
}

//! Scene exporters for Binary glTF 2.0 (`.glb`) and the renderer-neutral Blender bundle format (`M3.4`, `M3.5`).
//!
//! # Examples
//! ```
//! use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
//! use tetraglass_export::{export_evaluated_scene_to_glb, parse_glb};
//! use tetraglass_scene::{Scene, SceneEvaluator};
//!
//! let grid = SyntheticGridSpec::new(2, 2, 2, SyntheticCellLayout::Hexahedra)
//!     .generate()
//!     .expect("synthetic grid");
//! let scene = Scene::from_json_str(
//!     r#"{
//!       "version": 1,
//!       "sources": { "mesh": { "file": "hex.vtu" } },
//!       "view": { "show": [{ "input": "mesh", "color_by": "linear" }] }
//!     }"#,
//! )
//! .expect("parse scene");
//! let mut evaluator = SceneEvaluator::new();
//! evaluator.insert_source("mesh", grid);
//! let evaluated = evaluator.evaluate_scene(&scene).expect("evaluate scene");
//!
//! let glb_bytes = export_evaluated_scene_to_glb(&scene, &evaluated).expect("export glb");
//! let parsed = parse_glb(&glb_bytes).expect("parse glb");
//! assert_eq!(parsed.meshes.len(), 1);
//! assert!(!parsed.meshes[0].positions.is_empty());
//! assert_eq!(parsed.meshes[0].positions.len(), parsed.meshes[0].colors_rgba.len());
//! assert_eq!(parsed.meshes[0].positions.len(), parsed.meshes[0].texcoord_1.len());
//! ```

pub mod bundle;
pub mod colormap;
pub mod gltf;
pub mod worker;

pub use bundle::{
    BUNDLE_FORMAT_ID, BUNDLE_FORMAT_VERSION, BundleArchive, BundleBackground, BundleCamera,
    BundleColormap, BundleLights, BundleManifest, BundleMaterialHints, BundleMeshData,
    BundleMeshFileRef, BundleObject, BundleTimeStep, TGBIN_FLAGS_DEFAULT, TGBIN_HEADER_BYTES,
    TGBIN_MAGIC, TGBIN_VERSION, decode_tgbin, encode_tgbin, export_evaluated_scene_to_bundle,
    export_evaluated_steps_to_bundle, export_scene_to_bundle, resolve_bundle_background,
    resolve_bundle_camera_and_lights,
};
#[cfg(not(target_arch = "wasm32"))]
pub use bundle::{
    BundleWriteSummary, read_bundle_dir, write_bundle_dir, write_scene_bundle_output,
};
pub use colormap::{
    colormap_lut_srgb_256, linear_to_srgb, normalize_scalar, sample_colormap_linear,
    sample_colormap_srgb, srgb_to_linear,
};
pub use gltf::{
    GLB_CHUNK_BIN, GLB_CHUNK_JSON, GLB_MAGIC, GLB_VERSION, GLTF_COMPONENT_TYPE_F32,
    GLTF_COMPONENT_TYPE_U32, GLTF_MODE_TRIANGLES, GLTF_TARGET_ARRAY_BUFFER,
    GLTF_TARGET_ELEMENT_ARRAY_BUFFER, GlbAccessor, GlbAsset, GlbBuffer, GlbBufferView, GlbMaterial,
    GlbMesh, GlbMeshExtras, GlbNode, GlbPbrMetallicRoughness, GlbPrimitive, GlbRoot, GlbSceneDef,
    GlbSceneExtras, GltfSeriesManifest, GltfSeriesStepEntry, GltfStepFrame, GltfTimeSeriesExport,
    ParsedGlb, ParsedGlbMesh, export_evaluated_scene_to_glb, export_time_series_to_glb_frames,
    parse_glb,
};
#[cfg(not(target_arch = "wasm32"))]
pub use gltf::{GltfWriteSummary, write_evaluated_scene_glb, write_scene_glb_output};
#[cfg(not(target_arch = "wasm32"))]
pub use worker::{BlenderWorkerClient, BlenderWorkerProcess};
pub use worker::{
    BlenderWorkerRenderRequest, BlenderWorkerRequest, BlenderWorkerResponse,
    DEFAULT_WORKER_TIMEOUT_SECS,
};

use thiserror::Error;

/// Errors produced during scene export to `.glb` or bundle formats (`M3.4`, `M3.5`).
#[derive(Debug, Error)]
pub enum ExportError {
    /// Underlying scene evaluation error.
    #[error("scene evaluation failed: {0}")]
    Scene(#[from] tetraglass_scene::SceneError),

    /// Referenced attribute array was not found on the evaluated surface mesh.
    #[error("attribute array `{name}` is missing on the evaluated surface mesh")]
    MissingAttributeArray {
        /// Attribute array name.
        name: String,
    },

    /// JSON serialization or deserialization failed.
    #[error("JSON serialization failed: {0}")]
    Json(#[from] serde_json::Error),

    /// Malformed or unsupported Binary glTF 2.0 (`.glb`) data.
    #[error("invalid glTF 2.0 binary (.glb): {0}")]
    InvalidGlb(String),

    /// Malformed or unsupported Tetraglass bundle (`manifest.json` or `.tgbin`) data.
    #[error("invalid Tetraglass bundle: {0}")]
    InvalidBundle(String),

    /// Persistent Blender worker communication or runtime error (`M5.4`).
    #[error("Blender worker error: {0}")]
    Worker(String),

    /// Filesystem I/O error on native targets.
    #[cfg(not(target_arch = "wasm32"))]
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

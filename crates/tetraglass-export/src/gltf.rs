//! Binary glTF 2.0 (`.glb`) exporter and parser (`M3.4`).
//!
//! Exports an [`EvaluatedScene`] (or every step of a time series via [`SceneEvaluator`]) into
//! Binary glTF 2.0 (`.glb`) with:
//! - `POSITION` (`VEC3` `f32`) stored relative to `dataset_center` (`D-014`), with the node's
//!   `translation` set to `dataset_center` so world-space coordinates are preserved,
//! - `NORMAL` (`VEC3` `f32`) unit face normals,
//! - `COLOR_0` (`VEC4` `f32`) baked linear-light RGBA vertex colors from the active colormap
//!   (or solid color) and representation `opacity`,
//! - `TEXCOORD_0` (`VEC2` `f32`) `[normalized_scalar, 0.0]`,
//! - `TEXCOORD_1` (`VEC2` `f32`) `[normalized_scalar, raw_scalar]` so a Blender user can feed
//!   the second UV map (`UVMap.001`) directly into a ColorRamp shader node,
//! - `indices` (`SCALAR` `u32`, `ELEMENT_ARRAY_BUFFER`).

use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tetraglass_core::{BoundingBox, DataArray, SurfaceMesh};
use tetraglass_scene::{
    ArrayAssociation, EvaluatedColoring, EvaluatedRepresentation, EvaluatedScene,
    ResolvedComponent, Scene, SceneEvaluator, TimeSelector,
};

use crate::ExportError;
use crate::colormap::{normalize_scalar, sample_colormap_linear, srgb_to_linear};

/// Magic bytes `"glTF"` (`0x46546C67` in little-endian `u32`) at byte offset 0 of a `.glb` file.
pub const GLB_MAGIC: u32 = 0x4654_6C67;

/// glTF binary container version (`2`).
pub const GLB_VERSION: u32 = 2;

/// Chunk type for the JSON chunk (`"JSON"` / `0x4E4F534A` in little-endian `u32`).
pub const GLB_CHUNK_JSON: u32 = 0x4E4F_534A;

/// Chunk type for the binary buffer chunk (`"BIN\0"` / `0x004E4942` in little-endian `u32`).
pub const GLB_CHUNK_BIN: u32 = 0x004E_4942;

/// glTF component type for `f32` (`FLOAT`).
pub const GLTF_COMPONENT_TYPE_F32: u32 = 5126;

/// glTF component type for `u32` (`UNSIGNED_INT`).
pub const GLTF_COMPONENT_TYPE_U32: u32 = 5125;

/// glTF bufferView target for vertex attributes (`ARRAY_BUFFER`).
pub const GLTF_TARGET_ARRAY_BUFFER: u32 = 34962;

/// glTF bufferView target for index buffers (`ELEMENT_ARRAY_BUFFER`).
pub const GLTF_TARGET_ELEMENT_ARRAY_BUFFER: u32 = 34963;

/// glTF primitive topology for triangle lists (`TRIANGLES`).
pub const GLTF_MODE_TRIANGLES: u32 = 4;

/// Root JSON structure of a glTF 2.0 asset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlbRoot {
    /// glTF asset header (`version: "2.0"`).
    pub asset: GlbAsset,
    /// Index of the default scene (`0`).
    pub scene: usize,
    /// Scene list.
    pub scenes: Vec<GlbSceneDef>,
    /// Node list.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<GlbNode>,
    /// Mesh list.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub meshes: Vec<GlbMesh>,
    /// Material list.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<GlbMaterial>,
    /// Typed buffer accessors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accessors: Vec<GlbAccessor>,
    /// Contiguous slices into the binary buffer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub buffer_views: Vec<GlbBufferView>,
    /// Binary buffers (always 1 buffer in a `.glb` file when geometry is non-empty).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub buffers: Vec<GlbBuffer>,
}

// Serialize `buffer_views` as `"bufferViews"` per glTF 2.0 camelCase naming.
impl GlbRoot {
    fn to_gltf_json_bytes(&self) -> Result<Vec<u8>, ExportError> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct WireGlbRoot<'a> {
            asset: &'a GlbAsset,
            scene: usize,
            scenes: &'a [GlbSceneDef],
            #[serde(skip_serializing_if = " slice_is_empty")]
            nodes: &'a [GlbNode],
            #[serde(skip_serializing_if = "slice_is_empty")]
            meshes: &'a [GlbMesh],
            #[serde(skip_serializing_if = "slice_is_empty")]
            materials: &'a [GlbMaterial],
            #[serde(skip_serializing_if = "slice_is_empty")]
            accessors: &'a [GlbAccessor],
            #[serde(skip_serializing_if = "slice_is_empty")]
            buffer_views: &'a [GlbBufferView],
            #[serde(skip_serializing_if = "slice_is_empty")]
            buffers: &'a [GlbBuffer],
        }

        const fn slice_is_empty<T>(s: &[T]) -> bool {
            s.is_empty()
        }

        let wire = WireGlbRoot {
            asset: &self.asset,
            scene: self.scene,
            scenes: &self.scenes,
            nodes: &self.nodes,
            meshes: &self.meshes,
            materials: &self.materials,
            accessors: &self.accessors,
            buffer_views: &self.buffer_views,
            buffers: &self.buffers,
        };
        Ok(serde_json::to_vec(&wire)?)
    }

    fn from_gltf_json_bytes(bytes: &[u8]) -> Result<Self, ExportError> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct WireGlbRootOwned {
            asset: GlbAsset,
            #[serde(default)]
            scene: usize,
            #[serde(default)]
            scenes: Vec<GlbSceneDef>,
            #[serde(default)]
            nodes: Vec<GlbNode>,
            #[serde(default)]
            meshes: Vec<GlbMesh>,
            #[serde(default)]
            materials: Vec<GlbMaterial>,
            #[serde(default)]
            accessors: Vec<GlbAccessor>,
            #[serde(default)]
            buffer_views: Vec<GlbBufferView>,
            #[serde(default)]
            buffers: Vec<GlbBuffer>,
        }

        let w: WireGlbRootOwned = serde_json::from_slice(bytes)?;
        Ok(Self {
            asset: w.asset,
            scene: w.scene,
            scenes: w.scenes,
            nodes: w.nodes,
            meshes: w.meshes,
            materials: w.materials,
            accessors: w.accessors,
            buffer_views: w.buffer_views,
            buffers: w.buffers,
        })
    }
}

/// glTF `asset` metadata block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlbAsset {
    /// glTF specification version (`"2.0"`).
    pub version: String,
    /// Generator string (`"tetraglass-export"`).
    pub generator: String,
}

/// glTF `scene` entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlbSceneDef {
    /// Optional scene name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Root node indices in this scene.
    pub nodes: Vec<usize>,
    /// Tetraglass scene-level metadata (`time_step`, `time_value`, `dataset_center`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<GlbSceneExtras>,
}

/// Extra metadata stored on the glTF scene entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlbSceneExtras {
    /// Zero-based time-step index.
    pub time_step: usize,
    /// Simulation time value at this step.
    pub time_value: f64,
    /// Reference dataset center `[cx, cy, cz]` (`f64`) subtracted from vertex positions (`D-014`).
    pub dataset_center: [f64; 3],
}

/// glTF `node` entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlbNode {
    /// Node name (matches the representation's `input` dataset name).
    pub name: String,
    /// Index into `meshes`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh: Option<usize>,
    /// World-space translation `[cx, cy, cz]` restoring the subtracted `dataset_center`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translation: Option<[f64; 3]>,
}

/// glTF `mesh` entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlbMesh {
    /// Mesh name.
    pub name: String,
    /// Mesh primitives.
    pub primitives: Vec<GlbPrimitive>,
    /// Tetraglass representation coloring and scalar metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<GlbMeshExtras>,
}

/// Extra metadata attached to an exported [`GlbMesh`] describing its scalar mapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlbMeshExtras {
    /// Source or filter input name.
    pub input: String,
    /// Index of the representation in `scene.view.show`.
    pub representation_index: usize,
    /// Optional `color_by` array reference string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_by: Option<String>,
    /// Active colormap name (`"viridis"`, `"cividis"`, `"cool_warm"`, `"turbo"`, `"grayscale"`), if scalar-colored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colormap: Option<String>,
    /// Effective scalar range `[min, max]` mapped to `[0.0, 1.0]` in `TEXCOORD_1`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_range: Option<[f64; 2]>,
    /// Whether logarithmic scalar normalization was applied.
    pub log_scale: bool,
    /// Surface opacity in `[0.0, 1.0]`.
    pub opacity: f64,
}

/// glTF mesh primitive entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlbPrimitive {
    /// Map of attribute semantic (`"POSITION"`, `"NORMAL"`, `"COLOR_0"`, `"TEXCOORD_0"`, `"TEXCOORD_1"`) to accessor index.
    pub attributes: BTreeMap<String, usize>,
    /// Accessor index for triangle indices.
    pub indices: usize,
    /// Material index.
    pub material: usize,
    /// Primitive topology (`4` = `TRIANGLES`).
    pub mode: u32,
}

/// glTF PBR material entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlbMaterial {
    /// Material name.
    pub name: String,
    /// Metallic-roughness PBR parameters.
    pub pbr_metallic_roughness: GlbPbrMetallicRoughness,
    /// Double-sided rendering (`true` for scientific surfaces).
    pub double_sided: bool,
    /// Alpha mode (`"OPAQUE"` or `"BLEND"` when `opacity < 1.0`).
    pub alpha_mode: String,
}

/// glTF PBR metallic-roughness block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlbPbrMetallicRoughness {
    /// Base color factor (white `[1, 1, 1, 1]` because vertex `COLOR_0` carries baked color & alpha).
    pub base_color_factor: [f32; 4],
    /// Metallic factor (`0.0` for dielectric scientific surfaces).
    pub metallic_factor: f32,
    /// Roughness factor (`0.55`).
    pub roughness_factor: f32,
}

/// glTF typed accessor entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlbAccessor {
    /// Index into `bufferViews`.
    pub buffer_view: usize,
    /// Byte offset within the bufferView (`0`).
    pub byte_offset: usize,
    /// Component type (`5126` for `f32`, `5125` for `u32`).
    pub component_type: u32,
    /// Number of elements.
    pub count: usize,
    /// Element type (`"SCALAR"`, `"VEC2"`, `"VEC3"`, `"VEC4"`).
    #[serde(rename = "type")]
    pub accessor_type: String,
    /// Minimum per-component value (required by glTF 2.0 on `POSITION`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<Vec<f64>>,
    /// Maximum per-component value (required by glTF 2.0 on `POSITION`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<Vec<f64>>,
}

/// glTF `bufferView` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlbBufferView {
    /// Buffer index (`0`).
    pub buffer: usize,
    /// Byte offset from the start of the BIN chunk.
    pub byte_offset: usize,
    /// Length in bytes of this slice.
    pub byte_length: usize,
    /// GPU buffer target (`34962` = `ARRAY_BUFFER`, `34963` = `ELEMENT_ARRAY_BUFFER`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<u32>,
}

/// glTF `buffer` entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlbBuffer {
    /// Total byte length of the binary buffer.
    pub byte_length: usize,
}

/// One exported `.glb` frame in a time-series export (`M3.4`).
#[derive(Debug, Clone, PartialEq)]
pub struct GltfStepFrame {
    /// Zero-based time-step index.
    pub step: usize,
    /// Simulation time value at this step.
    pub time: f64,
    /// Recommended relative filename for this step (for example `"series_step_0000.glb"`).
    pub file_name: String,
    /// Complete `.glb` binary payload for this time step.
    pub glb_bytes: Vec<u8>,
}

/// Manifest entry for a single time step in [`GltfSeriesManifest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GltfSeriesStepEntry {
    /// Zero-based time-step index.
    pub step: usize,
    /// Simulation time value.
    pub time: f64,
    /// Relative `.glb` file name for this step.
    pub file: String,
    /// Total triangles across all visible representations in this step.
    pub total_triangles: usize,
}

/// JSON manifest accompanying a multi-step `.glb` time-series export (`M3.4`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GltfSeriesManifest {
    /// Manifest format version (`1`).
    pub version: u32,
    /// Export format identifier (`"gltf-series"`).
    pub format: String,
    /// Ordered list of exported per-step `.glb` files.
    pub steps: Vec<GltfSeriesStepEntry>,
}

/// Result of exporting a scene across one or more time steps (`M3.4`).
#[derive(Debug, Clone, PartialEq)]
pub struct GltfTimeSeriesExport {
    /// JSON manifest describing all exported `.glb` steps.
    pub manifest: GltfSeriesManifest,
    /// Ordered `.glb` step frames.
    pub frames: Vec<GltfStepFrame>,
}

/// Reconstructed mesh from [`parse_glb`], used to verify `.glb` exports.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedGlbMesh {
    /// Mesh name.
    pub name: String,
    /// Node translation `[cx, cy, cz]` (if present).
    pub translation: [f64; 3],
    /// Vertex positions (`POSITION`, relative to `translation`).
    pub positions: Vec<[f32; 3]>,
    /// Vertex normals (`NORMAL`).
    pub normals: Vec<[f32; 3]>,
    /// Baked linear RGBA vertex colors (`COLOR_0`).
    pub colors_rgba: Vec<[f32; 4]>,
    /// First texture coordinate (`TEXCOORD_0`, `[normalized_scalar, 0.0]`).
    pub texcoord_0: Vec<[f32; 2]>,
    /// Second texture coordinate (`TEXCOORD_1`, `[normalized_scalar, raw_scalar]`).
    pub texcoord_1: Vec<[f32; 2]>,
    /// Triangle vertex indices (`0..3*num_triangles`).
    pub indices: Vec<u32>,
    /// Optional Tetraglass mesh metadata.
    pub extras: Option<GlbMeshExtras>,
}

/// Parsed Binary glTF 2.0 (`.glb`) container returned by [`parse_glb`].
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedGlb {
    /// Parsed JSON root structure.
    pub root: GlbRoot,
    /// Reconstructed meshes with their unpacked vertex and index buffers.
    pub meshes: Vec<ParsedGlbMesh>,
}

/// Exports an already-evaluated [`EvaluatedScene`] to a Binary glTF 2.0 (`.glb`) byte vector (`M3.4`).
pub fn export_evaluated_scene_to_glb(
    _scene: &Scene,
    evaluated: &EvaluatedScene,
) -> Result<Vec<u8>, ExportError> {
    let dataset_center = resolve_scene_center(evaluated);

    let mut nodes = Vec::new();
    let mut meshes = Vec::new();
    let mut materials = Vec::new();
    let mut accessors = Vec::new();
    let mut buffer_views = Vec::new();
    let mut bin_data: Vec<u8> = Vec::new();

    for rep in &evaluated.representations {
        let num_tris = rep.surface.num_triangles();
        if num_tris == 0 {
            continue;
        }

        let extracted = build_representation_buffers(rep, dataset_center)?;
        let num_verts = extracted.positions.len();

        // Append slices to `bin_data` (each already 4-byte aligned because elements are f32/u32).
        let pos_view_idx = append_buffer_view(
            &mut bin_data,
            &mut buffer_views,
            bytemuck::cast_slice(&extracted.positions),
            Some(GLTF_TARGET_ARRAY_BUFFER),
        );
        let pos_acc_idx = push_accessor(
            &mut accessors,
            pos_view_idx,
            GLTF_COMPONENT_TYPE_F32,
            num_verts,
            "VEC3",
            Some(extracted.pos_min.to_vec()),
            Some(extracted.pos_max.to_vec()),
        );

        let norm_view_idx = append_buffer_view(
            &mut bin_data,
            &mut buffer_views,
            bytemuck::cast_slice(&extracted.normals),
            Some(GLTF_TARGET_ARRAY_BUFFER),
        );
        let norm_acc_idx = push_accessor(
            &mut accessors,
            norm_view_idx,
            GLTF_COMPONENT_TYPE_F32,
            num_verts,
            "VEC3",
            None,
            None,
        );

        let col_view_idx = append_buffer_view(
            &mut bin_data,
            &mut buffer_views,
            bytemuck::cast_slice(&extracted.colors_rgba),
            Some(GLTF_TARGET_ARRAY_BUFFER),
        );
        let col_acc_idx = push_accessor(
            &mut accessors,
            col_view_idx,
            GLTF_COMPONENT_TYPE_F32,
            num_verts,
            "VEC4",
            None,
            None,
        );

        let uv0_view_idx = append_buffer_view(
            &mut bin_data,
            &mut buffer_views,
            bytemuck::cast_slice(&extracted.texcoord_0),
            Some(GLTF_TARGET_ARRAY_BUFFER),
        );
        let uv0_acc_idx = push_accessor(
            &mut accessors,
            uv0_view_idx,
            GLTF_COMPONENT_TYPE_F32,
            num_verts,
            "VEC2",
            None,
            None,
        );

        let uv1_view_idx = append_buffer_view(
            &mut bin_data,
            &mut buffer_views,
            bytemuck::cast_slice(&extracted.texcoord_1),
            Some(GLTF_TARGET_ARRAY_BUFFER),
        );
        let uv1_acc_idx = push_accessor(
            &mut accessors,
            uv1_view_idx,
            GLTF_COMPONENT_TYPE_F32,
            num_verts,
            "VEC2",
            None,
            None,
        );

        let idx_view_idx = append_buffer_view(
            &mut bin_data,
            &mut buffer_views,
            bytemuck::cast_slice(&extracted.indices),
            Some(GLTF_TARGET_ELEMENT_ARRAY_BUFFER),
        );
        let idx_acc_idx = push_accessor(
            &mut accessors,
            idx_view_idx,
            GLTF_COMPONENT_TYPE_U32,
            extracted.indices.len(),
            "SCALAR",
            None,
            None,
        );

        let mat_idx = materials.len();
        let alpha_mode = if rep.opacity < 0.9999 {
            "BLEND".to_string()
        } else {
            "OPAQUE".to_string()
        };
        materials.push(GlbMaterial {
            name: format!("{}_material_{}", rep.input, rep.representation_index),
            pbr_metallic_roughness: GlbPbrMetallicRoughness {
                base_color_factor: [1.0, 1.0, 1.0, 1.0],
                metallic_factor: 0.0,
                roughness_factor: 0.55,
            },
            double_sided: true,
            alpha_mode,
        });

        let mut attributes = BTreeMap::new();
        attributes.insert("POSITION".to_string(), pos_acc_idx);
        attributes.insert("NORMAL".to_string(), norm_acc_idx);
        attributes.insert("COLOR_0".to_string(), col_acc_idx);
        attributes.insert("TEXCOORD_0".to_string(), uv0_acc_idx);
        attributes.insert("TEXCOORD_1".to_string(), uv1_acc_idx);

        let mesh_idx = meshes.len();
        let extras = GlbMeshExtras {
            input: rep.input.clone(),
            representation_index: rep.representation_index,
            color_by: rep.coloring.as_ref().map(|c| c.raw_ref.clone()),
            colormap: rep
                .coloring
                .as_ref()
                .map(|c| colormap_name_str(c.colormap).to_string()),
            effective_range: rep.coloring.as_ref().map(|c| c.effective_range),
            log_scale: rep.coloring.as_ref().is_some_and(|c| c.log_scale),
            opacity: rep.opacity,
        };
        meshes.push(GlbMesh {
            name: rep.input.clone(),
            primitives: vec![GlbPrimitive {
                attributes,
                indices: idx_acc_idx,
                material: mat_idx,
                mode: GLTF_MODE_TRIANGLES,
            }],
            extras: Some(extras),
        });

        let translation = if dataset_center[0].abs() > 0.0
            || dataset_center[1].abs() > 0.0
            || dataset_center[2].abs() > 0.0
        {
            Some(dataset_center)
        } else {
            None
        };
        nodes.push(GlbNode {
            name: rep.input.clone(),
            mesh: Some(mesh_idx),
            translation,
        });
    }

    let buffers = if bin_data.is_empty() {
        Vec::new()
    } else {
        vec![GlbBuffer {
            byte_length: bin_data.len(),
        }]
    };
    let node_indices: Vec<usize> = (0..nodes.len()).collect();
    let root = GlbRoot {
        asset: GlbAsset {
            version: "2.0".to_string(),
            generator: "tetraglass-export".to_string(),
        },
        scene: 0,
        scenes: vec![GlbSceneDef {
            name: Some("tetraglass_scene".to_string()),
            nodes: node_indices,
            extras: Some(GlbSceneExtras {
                time_step: evaluated.time_step,
                time_value: evaluated.time_value,
                dataset_center,
            }),
        }],
        nodes,
        meshes,
        materials,
        accessors,
        buffer_views,
        buffers,
    };

    pack_glb_container(&root, &bin_data)
}

/// Evaluates `scene` across time steps using `evaluator` and exports one `.glb` per time step
/// together with a [`GltfSeriesManifest`] (`M3.4`).
///
/// - If `steps` is `Some(indices)`, exports those zero-based step indices in order.
/// - If `steps` is `None`, exports all available time steps (`0..num_steps`).
pub fn export_time_series_to_glb_frames(
    scene: &Scene,
    evaluator: &mut SceneEvaluator,
    file_stem: &str,
    steps: Option<&[usize]>,
) -> Result<GltfTimeSeriesExport, ExportError> {
    let total_steps = infer_evaluator_step_count(scene, evaluator);
    let step_list: Vec<usize> = match steps {
        Some(explicit) => explicit.to_vec(),
        None => (0..total_steps).collect(),
    };

    let clean_stem = if file_stem.trim().is_empty() {
        "scene"
    } else {
        file_stem.trim()
    };

    let mut frames = Vec::with_capacity(step_list.len());
    let mut manifest_steps = Vec::with_capacity(step_list.len());

    for step_idx in step_list {
        let evaluated = evaluator.evaluate_scene_at_time(scene, TimeSelector::step(step_idx))?;
        let glb_bytes = export_evaluated_scene_to_glb(scene, &evaluated)?;
        let total_triangles: usize = evaluated
            .representations
            .iter()
            .map(|r| r.surface.num_triangles())
            .sum();
        let file_name = format!("{clean_stem}_step_{step_idx:04}.glb");
        manifest_steps.push(GltfSeriesStepEntry {
            step: evaluated.time_step,
            time: evaluated.time_value,
            file: file_name.clone(),
            total_triangles,
        });
        frames.push(GltfStepFrame {
            step: evaluated.time_step,
            time: evaluated.time_value,
            file_name,
            glb_bytes,
        });
    }

    Ok(GltfTimeSeriesExport {
        manifest: GltfSeriesManifest {
            version: 1,
            format: "gltf-series".to_string(),
            steps: manifest_steps,
        },
        frames,
    })
}

/// Writes an [`EvaluatedScene`] to `path` as a `.glb` file on native targets (`M3.4`).
#[cfg(not(target_arch = "wasm32"))]
pub fn write_evaluated_scene_glb(
    path: impl AsRef<Path>,
    scene: &Scene,
    evaluated: &EvaluatedScene,
) -> Result<(), ExportError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let bytes = export_evaluated_scene_to_glb(scene, evaluated)?;
    fs::write(path, bytes)?;
    Ok(())
}

/// Summary of files written by [`write_scene_glb_output`].
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GltfWriteSummary {
    /// Paths of all `.glb` files written, in step order.
    pub glb_files: Vec<PathBuf>,
    /// Optional path to the JSON time-series manifest (written when exporting multiple steps or a directory).
    pub manifest_file: Option<PathBuf>,
    /// Number of time steps exported.
    pub num_steps: usize,
    /// Total triangles exported across all steps.
    pub total_triangles: usize,
}

/// Exports `scene` to disk as `.glb` (`M3.4`).
///
/// - When `all_steps` is `false` and the scene has a single time step (or `output_path` ends with
///   `.glb` and `!all_steps`), evaluates the scene's configured time step (`scene.output.time` or
///   `TimeSelector::LAST`) and writes a single `.glb` file when `total_steps == 1` and `!all_steps`.
/// - When `all_steps` is `true` or `total_steps > 1`: writes one `.glb` per time step plus a
///   `<stem>_manifest.json` (or `manifest.json` inside `output_path` if `output_path` has no `.glb` extension).
#[cfg(not(target_arch = "wasm32"))]
pub fn write_scene_glb_output(
    output_path: impl AsRef<Path>,
    scene: &Scene,
    evaluator: &mut SceneEvaluator,
    all_steps: bool,
) -> Result<GltfWriteSummary, ExportError> {
    let output_path = output_path.as_ref();
    let total_steps = infer_evaluator_step_count(scene, evaluator);
    let is_glb_file = output_path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("glb"));

    if !all_steps && total_steps <= 1 && is_glb_file {
        let selector = scene.output.as_ref().map_or(TimeSelector::LAST, |o| o.time);
        let evaluated = evaluator.evaluate_scene_at_time(scene, selector)?;
        let total_triangles: usize = evaluated
            .representations
            .iter()
            .map(|r| r.surface.num_triangles())
            .sum();
        write_evaluated_scene_glb(output_path, scene, &evaluated)?;
        return Ok(GltfWriteSummary {
            glb_files: vec![output_path.to_path_buf()],
            manifest_file: None,
            num_steps: 1,
            total_triangles,
        });
    }

    let (target_dir, stem, manifest_name) = if is_glb_file {
        let parent = output_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let file_stem = output_path
            .file_stem()
            .map_or_else(|| "scene".to_string(), |s| s.to_string_lossy().into_owned());
        let manifest_name = format!("{file_stem}_manifest.json");
        (parent, file_stem, manifest_name)
    } else {
        let dir = output_path.to_path_buf();
        let file_stem = output_path
            .file_name()
            .map_or_else(|| "scene".to_string(), |s| s.to_string_lossy().into_owned());
        (dir, file_stem, "manifest.json".to_string())
    };

    fs::create_dir_all(&target_dir)?;

    let steps_slice: Option<Vec<usize>> = if all_steps || total_steps > 1 {
        None
    } else {
        let selector = scene.output.as_ref().map_or(TimeSelector::LAST, |o| o.time);
        let evaluated = evaluator.evaluate_scene_at_time(scene, selector)?;
        Some(vec![evaluated.time_step])
    };

    let exported =
        export_time_series_to_glb_frames(scene, evaluator, &stem, steps_slice.as_deref())?;
    let mut glb_files = Vec::with_capacity(exported.frames.len());
    let mut total_triangles = 0usize;

    for (frame, entry) in exported.frames.iter().zip(exported.manifest.steps.iter()) {
        let frame_path = target_dir.join(&frame.file_name);
        fs::write(&frame_path, &frame.glb_bytes)?;
        glb_files.push(frame_path);
        total_triangles += entry.total_triangles;
    }

    let manifest_path = target_dir.join(manifest_name);
    let manifest_json = serde_json::to_string_pretty(&exported.manifest)?;
    fs::write(&manifest_path, manifest_json)?;

    Ok(GltfWriteSummary {
        glb_files,
        manifest_file: Some(manifest_path),
        num_steps: exported.frames.len(),
        total_triangles,
    })
}

/// Parses and validates a Binary glTF 2.0 (`.glb`) byte slice, unpacking all mesh vertex and
/// index attributes (`M3.4`).
pub fn parse_glb(bytes: &[u8]) -> Result<ParsedGlb, ExportError> {
    if bytes.len() < 20 {
        return Err(ExportError::InvalidGlb(format!(
            "GLB buffer too short ({} bytes, minimum 20)",
            bytes.len()
        )));
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != GLB_MAGIC {
        return Err(ExportError::InvalidGlb(format!(
            "invalid GLB magic 0x{magic:08X} (expected 0x{GLB_MAGIC:08X})"
        )));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != GLB_VERSION {
        return Err(ExportError::InvalidGlb(format!(
            "unsupported GLB version {version} (expected {GLB_VERSION})"
        )));
    }
    let declared_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if declared_len != bytes.len() {
        return Err(ExportError::InvalidGlb(format!(
            "GLB header length {declared_len} does not match actual byte length {}",
            bytes.len()
        )));
    }

    // Chunk 0: JSON
    let json_chunk_len = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let json_chunk_type = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    if json_chunk_type != GLB_CHUNK_JSON {
        return Err(ExportError::InvalidGlb(format!(
            "first GLB chunk must be JSON (0x{GLB_CHUNK_JSON:08X}), got 0x{json_chunk_type:08X}"
        )));
    }
    let json_end = 20usize.saturating_add(json_chunk_len);
    if json_end > bytes.len() || !json_chunk_len.is_multiple_of(4) {
        return Err(ExportError::InvalidGlb(
            "invalid or unaligned JSON chunk length".to_string(),
        ));
    }
    let root = GlbRoot::from_gltf_json_bytes(&bytes[20..json_end])?;

    // Optional Chunk 1: BIN
    let bin_slice: &[u8] = if json_end < bytes.len() {
        if json_end + 8 > bytes.len() {
            return Err(ExportError::InvalidGlb(
                "truncated BIN chunk header".to_string(),
            ));
        }
        let bin_len = u32::from_le_bytes([
            bytes[json_end],
            bytes[json_end + 1],
            bytes[json_end + 2],
            bytes[json_end + 3],
        ]) as usize;
        let bin_type = u32::from_le_bytes([
            bytes[json_end + 4],
            bytes[json_end + 5],
            bytes[json_end + 6],
            bytes[json_end + 7],
        ]);
        if bin_type != GLB_CHUNK_BIN {
            return Err(ExportError::InvalidGlb(format!(
                "second GLB chunk must be BIN (0x{GLB_CHUNK_BIN:08X}), got 0x{bin_type:08X}"
            )));
        }
        let bin_start = json_end + 8;
        let bin_end = bin_start.saturating_add(bin_len);
        if bin_end != bytes.len() || !bin_len.is_multiple_of(4) {
            return Err(ExportError::InvalidGlb(
                "invalid or unaligned BIN chunk length".to_string(),
            ));
        }
        &bytes[bin_start..bin_end]
    } else {
        &[]
    };

    let mut parsed_meshes = Vec::with_capacity(root.meshes.len());
    for (mesh_idx, mesh) in root.meshes.iter().enumerate() {
        let translation = root
            .nodes
            .iter()
            .find(|n| n.mesh == Some(mesh_idx))
            .and_then(|n| n.translation)
            .unwrap_or([0.0, 0.0, 0.0]);

        for prim in &mesh.primitives {
            let positions = read_vec3_f32_attr(&root, bin_slice, &prim.attributes, "POSITION")?;
            let normals = read_vec3_f32_attr(&root, bin_slice, &prim.attributes, "NORMAL")?;
            let colors_rgba = read_vec4_f32_attr(&root, bin_slice, &prim.attributes, "COLOR_0")?;
            let texcoord_0 = read_vec2_f32_attr(&root, bin_slice, &prim.attributes, "TEXCOORD_0")?;
            let texcoord_1 = read_vec2_f32_attr(&root, bin_slice, &prim.attributes, "TEXCOORD_1")?;
            let indices = read_scalar_u32_accessor(&root, bin_slice, prim.indices)?;

            parsed_meshes.push(ParsedGlbMesh {
                name: mesh.name.clone(),
                translation,
                positions,
                normals,
                colors_rgba,
                texcoord_0,
                texcoord_1,
                indices,
                extras: mesh.extras.clone(),
            });
        }
    }

    Ok(ParsedGlb {
        root,
        meshes: parsed_meshes,
    })
}

fn infer_evaluator_step_count(scene: &Scene, evaluator: &SceneEvaluator) -> usize {
    let mut max_steps = 1usize;
    for source_name in scene.sources.keys() {
        if let Some(series) = evaluator.series_source(source_name) {
            max_steps = max_steps.max(series.len());
        }
    }
    max_steps
}

fn resolve_scene_center(evaluated: &EvaluatedScene) -> [f64; 3] {
    if let Some(bounds) = evaluated.camera_bounds {
        return bounds.center();
    }
    let mut combined: Option<BoundingBox> = None;
    for rep in &evaluated.representations {
        if let Some(b) = rep.surface.bounds() {
            combined = Some(match combined {
                Some(acc) => BoundingBox {
                    min: [
                        acc.min[0].min(b.min[0]),
                        acc.min[1].min(b.min[1]),
                        acc.min[2].min(b.min[2]),
                    ],
                    max: [
                        acc.max[0].max(b.max[0]),
                        acc.max[1].max(b.max[1]),
                        acc.max[2].max(b.max[2]),
                    ],
                },
                None => b,
            });
        }
    }
    combined.map_or([0.0, 0.0, 0.0], |b| b.center())
}

const fn colormap_name_str(cmap: tetraglass_scene::ColormapName) -> &'static str {
    match cmap {
        tetraglass_scene::ColormapName::Viridis => "viridis",
        tetraglass_scene::ColormapName::Cividis => "cividis",
        tetraglass_scene::ColormapName::CoolWarm => "cool_warm",
        tetraglass_scene::ColormapName::Turbo => "turbo",
        tetraglass_scene::ColormapName::Grayscale => "grayscale",
    }
}

struct ExtractedPrimitiveBuffers {
    positions: Vec<[f32; 3]>,
    pos_min: [f64; 3],
    pos_max: [f64; 3],
    normals: Vec<[f32; 3]>,
    colors_rgba: Vec<[f32; 4]>,
    texcoord_0: Vec<[f32; 2]>,
    texcoord_1: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

fn build_representation_buffers(
    rep: &EvaluatedRepresentation,
    dataset_center: [f64; 3],
) -> Result<ExtractedPrimitiveBuffers, ExportError> {
    let surface = rep.surface.as_ref();
    let points = surface.points();
    let triangles = surface.triangles();
    let num_verts = triangles.len() * 3;
    let opacity = (rep.opacity as f32).clamp(0.0, 1.0);

    let mut positions = Vec::with_capacity(num_verts);
    let mut normals = Vec::with_capacity(num_verts);
    let mut colors_rgba = Vec::with_capacity(num_verts);
    let mut texcoord_0 = Vec::with_capacity(num_verts);
    let mut texcoord_1 = Vec::with_capacity(num_verts);
    let mut indices = Vec::with_capacity(num_verts);

    let mut pos_min = [f64::INFINITY; 3];
    let mut pos_max = [f64::NEG_INFINITY; 3];

    let resolved_scalars = match &rep.coloring {
        Some(coloring) => Some(extract_representation_scalars(surface, coloring)?),
        None => None,
    };

    let solid_linear = [
        srgb_to_linear(rep.color[0]),
        srgb_to_linear(rep.color[1]),
        srgb_to_linear(rep.color[2]),
        opacity,
    ];

    for (tri_idx, &[i0, i1, i2]) in triangles.iter().enumerate() {
        let p0 = points[i0 as usize];
        let p1 = points[i1 as usize];
        let p2 = points[i2 as usize];
        let face_normal = compute_triangle_normal(p0, p1, p2);

        for (corner_offset, &pt_idx) in [i0, i1, i2].iter().enumerate() {
            let p = points[pt_idx as usize];
            let rel = [
                (p[0] - dataset_center[0]) as f32,
                (p[1] - dataset_center[1]) as f32,
                (p[2] - dataset_center[2]) as f32,
            ];
            for k in 0..3 {
                let v = f64::from(rel[k]);
                if v < pos_min[k] {
                    pos_min[k] = v;
                }
                if v > pos_max[k] {
                    pos_max[k] = v;
                }
            }
            positions.push(rel);
            normals.push(face_normal);

            if let (Some(coloring), Some(scalars)) = (&rep.coloring, &resolved_scalars) {
                let raw_val = match coloring.association {
                    ArrayAssociation::Point => scalars[pt_idx as usize],
                    ArrayAssociation::Cell => scalars[tri_idx],
                };
                let norm_t = normalize_scalar(
                    raw_val,
                    coloring.effective_range,
                    coloring.log_scale,
                    coloring.center,
                );
                let [r_lin, g_lin, b_lin] = sample_colormap_linear(coloring.colormap, norm_t);
                colors_rgba.push([r_lin, g_lin, b_lin, opacity]);
                let raw_f32 = if raw_val.is_finite() {
                    raw_val as f32
                } else {
                    0.0
                };
                texcoord_0.push([norm_t, 0.0]);
                texcoord_1.push([norm_t, raw_f32]);
            } else {
                colors_rgba.push(solid_linear);
                texcoord_0.push([0.0, 0.0]);
                texcoord_1.push([0.0, 0.0]);
            }

            let vert_idx = (tri_idx * 3 + corner_offset) as u32;
            indices.push(vert_idx);
        }
    }

    if positions.is_empty() {
        pos_min = [0.0, 0.0, 0.0];
        pos_max = [0.0, 0.0, 0.0];
    }

    Ok(ExtractedPrimitiveBuffers {
        positions,
        pos_min,
        pos_max,
        normals,
        colors_rgba,
        texcoord_0,
        texcoord_1,
        indices,
    })
}

fn extract_representation_scalars(
    surface: &SurfaceMesh,
    coloring: &EvaluatedColoring,
) -> Result<Vec<f64>, ExportError> {
    let attrs = match coloring.association {
        ArrayAssociation::Point => surface.point_data(),
        ArrayAssociation::Cell => surface.cell_data(),
    };
    let array =
        attrs
            .get(&coloring.array_name)
            .ok_or_else(|| ExportError::MissingAttributeArray {
                name: coloring.array_name.clone(),
            })?;
    Ok(extract_component_values(array, coloring.component))
}

fn extract_component_values(array: &DataArray, component: ResolvedComponent) -> Vec<f64> {
    let n = array.num_tuples();
    let k = array.components().as_usize();
    let mut out = Vec::with_capacity(n);
    for tuple_idx in 0..n {
        let v = match component {
            ResolvedComponent::Component(c) => array.value_as_f64(tuple_idx, c).unwrap_or(0.0),
            ResolvedComponent::Magnitude => {
                let mut sum_sq = 0.0;
                for c in 0..k {
                    let val = array.value_as_f64(tuple_idx, c).unwrap_or(0.0);
                    sum_sq += val * val;
                }
                sum_sq.sqrt()
            }
        };
        out.push(v);
    }
    out
}

fn compute_triangle_normal(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> [f32; 3] {
    let ux = p1[0] - p0[0];
    let uy = p1[1] - p0[1];
    let uz = p1[2] - p0[2];
    let vx = p2[0] - p0[0];
    let vy = p2[1] - p0[1];
    let vz = p2[2] - p0[2];
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-20 {
        [(nx / len) as f32, (ny / len) as f32, (nz / len) as f32]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn append_buffer_view(
    bin_data: &mut Vec<u8>,
    buffer_views: &mut Vec<GlbBufferView>,
    raw_bytes: &[u8],
    target: Option<u32>,
) -> usize {
    while !bin_data.len().is_multiple_of(4) {
        bin_data.push(0);
    }
    let byte_offset = bin_data.len();
    let byte_length = raw_bytes.len();
    bin_data.extend_from_slice(raw_bytes);
    while !bin_data.len().is_multiple_of(4) {
        bin_data.push(0);
    }
    let idx = buffer_views.len();
    buffer_views.push(GlbBufferView {
        buffer: 0,
        byte_offset,
        byte_length,
        target,
    });
    idx
}

fn push_accessor(
    accessors: &mut Vec<GlbAccessor>,
    buffer_view: usize,
    component_type: u32,
    count: usize,
    accessor_type: &str,
    min: Option<Vec<f64>>,
    max: Option<Vec<f64>>,
) -> usize {
    let idx = accessors.len();
    accessors.push(GlbAccessor {
        buffer_view,
        byte_offset: 0,
        component_type,
        count,
        accessor_type: accessor_type.to_string(),
        min,
        max,
    });
    idx
}

fn pack_glb_container(root: &GlbRoot, bin_data: &[u8]) -> Result<Vec<u8>, ExportError> {
    let mut json_bytes = root.to_gltf_json_bytes()?;
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }

    let mut padded_bin = bin_data.to_vec();
    while !padded_bin.len().is_multiple_of(4) {
        padded_bin.push(0);
    }

    let bin_chunk_total = if padded_bin.is_empty() {
        0
    } else {
        8 + padded_bin.len()
    };
    let total_length = 12 + 8 + json_bytes.len() + bin_chunk_total;
    let total_u32 = u32::try_from(total_length).map_err(|_| {
        ExportError::InvalidGlb(format!("GLB file size {total_length} exceeds 4 GiB limit"))
    })?;

    let mut out = Vec::with_capacity(total_length);
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&GLB_VERSION.to_le_bytes());
    out.extend_from_slice(&total_u32.to_le_bytes());

    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&GLB_CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&json_bytes);

    if !padded_bin.is_empty() {
        out.extend_from_slice(&(padded_bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&GLB_CHUNK_BIN.to_le_bytes());
        out.extend_from_slice(&padded_bin);
    }

    Ok(out)
}

fn accessor_slice<'a>(
    root: &GlbRoot,
    bin_slice: &'a [u8],
    accessor_idx: usize,
    expected_component_type: u32,
    expected_type: &str,
    bytes_per_elem: usize,
) -> Result<(&'a [u8], usize), ExportError> {
    let acc = root.accessors.get(accessor_idx).ok_or_else(|| {
        ExportError::InvalidGlb(format!("accessor index {accessor_idx} out of bounds"))
    })?;
    if acc.component_type != expected_component_type || acc.accessor_type != expected_type {
        return Err(ExportError::InvalidGlb(format!(
            "accessor {accessor_idx} type mismatch: got ({}, {}), expected ({expected_component_type}, {expected_type})",
            acc.component_type, acc.accessor_type
        )));
    }
    let bv = root.buffer_views.get(acc.buffer_view).ok_or_else(|| {
        ExportError::InvalidGlb(format!(
            "bufferView index {} out of bounds",
            acc.buffer_view
        ))
    })?;
    let start = bv.byte_offset.saturating_add(acc.byte_offset);
    let len = acc.count.saturating_mul(bytes_per_elem);
    let end = start.saturating_add(len);
    if end > bin_slice.len() || acc.byte_offset.saturating_add(len) > bv.byte_length {
        return Err(ExportError::InvalidGlb(format!(
            "accessor {accessor_idx} range [{start}..{end}] exceeds bufferView or BIN chunk"
        )));
    }
    Ok((&bin_slice[start..end], acc.count))
}

fn read_vec3_f32_attr(
    root: &GlbRoot,
    bin_slice: &[u8],
    attrs: &BTreeMap<String, usize>,
    name: &str,
) -> Result<Vec<[f32; 3]>, ExportError> {
    let &acc_idx = attrs
        .get(name)
        .ok_or_else(|| ExportError::InvalidGlb(format!("missing attribute `{name}`")))?;
    let (raw, count) = accessor_slice(
        root,
        bin_slice,
        acc_idx,
        GLTF_COMPONENT_TYPE_F32,
        "VEC3",
        12,
    )?;
    let mut out = Vec::with_capacity(count);
    for chunk in raw.as_chunks::<12>().0 {
        out.push([
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
            f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
        ]);
    }
    Ok(out)
}

fn read_vec4_f32_attr(
    root: &GlbRoot,
    bin_slice: &[u8],
    attrs: &BTreeMap<String, usize>,
    name: &str,
) -> Result<Vec<[f32; 4]>, ExportError> {
    let &acc_idx = attrs
        .get(name)
        .ok_or_else(|| ExportError::InvalidGlb(format!("missing attribute `{name}`")))?;
    let (raw, count) = accessor_slice(
        root,
        bin_slice,
        acc_idx,
        GLTF_COMPONENT_TYPE_F32,
        "VEC4",
        16,
    )?;
    let mut out = Vec::with_capacity(count);
    for chunk in raw.as_chunks::<16>().0 {
        out.push([
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
            f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
            f32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]),
        ]);
    }
    Ok(out)
}

fn read_vec2_f32_attr(
    root: &GlbRoot,
    bin_slice: &[u8],
    attrs: &BTreeMap<String, usize>,
    name: &str,
) -> Result<Vec<[f32; 2]>, ExportError> {
    let &acc_idx = attrs
        .get(name)
        .ok_or_else(|| ExportError::InvalidGlb(format!("missing attribute `{name}`")))?;
    let (raw, count) =
        accessor_slice(root, bin_slice, acc_idx, GLTF_COMPONENT_TYPE_F32, "VEC2", 8)?;
    let mut out = Vec::with_capacity(count);
    for chunk in raw.as_chunks::<8>().0 {
        out.push([
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
        ]);
    }
    Ok(out)
}

fn read_scalar_u32_accessor(
    root: &GlbRoot,
    bin_slice: &[u8],
    acc_idx: usize,
) -> Result<Vec<u32>, ExportError> {
    let (raw, count) = accessor_slice(
        root,
        bin_slice,
        acc_idx,
        GLTF_COMPONENT_TYPE_U32,
        "SCALAR",
        4,
    )?;
    let mut out = Vec::with_capacity(count);
    for &chunk in raw.as_chunks::<4>().0 {
        out.push(u32::from_le_bytes(chunk));
    }
    Ok(out)
}

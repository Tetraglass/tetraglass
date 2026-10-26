//! Renderer-neutral scene bundle exporter and reader (`M3.5`, `docs/bundle-format.md`).
//!
//! Produces a directory (or in-memory [`BundleArchive`]) containing `manifest.json` and
//! one little-endian `.tgbin` binary mesh buffer per object and time step.
//! Carries pure data only (`D-011`, `.agents/rules/blender-gpl.md`).

use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tetraglass_core::{BoundingBox, DataArray, SurfaceMesh};
use tetraglass_scene::{
    ArrayAssociation, BackgroundSpec, ColormapName, DEFAULT_OUTPUT_SIZE, EvaluatedColoring,
    EvaluatedRepresentation, EvaluatedScene, LightingPreset, ProjectionKind, RepresentationStyle,
    ResolvedComponent, Scene, SceneEvaluator, TimeSelector,
};

use crate::ExportError;
use crate::colormap::{colormap_lut_srgb_256, normalize_scalar};

/// Current bundle manifest format version (`1`).
pub const BUNDLE_FORMAT_VERSION: u32 = 1;

/// Canonical `"format"` identifier in `manifest.json`.
pub const BUNDLE_FORMAT_ID: &str = "tetraglass-bundle";

/// Little-endian magic bytes `"TGMB"` (`0x424D4754`) at offset 0 of every `.tgbin` file.
pub const TGBIN_MAGIC: u32 = 0x424D_4754;

/// Binary `.tgbin` format version (`1`).
pub const TGBIN_VERSION: u32 = 1;

/// Fixed byte length of the `.tgbin` header (`24` bytes = `6 * u32`).
pub const TGBIN_HEADER_BYTES: usize = 24;

/// Default `.tgbin` header flags (`0x3`: bit 0 = normalized scalars, bit 1 = raw scalars).
pub const TGBIN_FLAGS_DEFAULT: u32 = 0x3;

/// Root structure of `manifest.json` in a Tetraglass bundle (`docs/bundle-format.md`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleManifest {
    /// Bundle format version (`1`).
    pub version: u32,
    /// Format identifier (`"tetraglass-bundle"`).
    pub format: String,
    /// Spatial unit label (`"meters"`).
    pub units: String,
    /// World-space center `[cx, cy, cz]` (`f64`) subtracted from all `f32` vertex positions (`D-014`).
    pub dataset_center: [f64; 3],
    /// Target image dimensions `[width, height]` in pixels.
    pub image_size: [u32; 2],
    /// Resolved camera configuration.
    pub camera: BundleCamera,
    /// Resolved directional and headlight configuration.
    pub lights: BundleLights,
    /// Resolved viewport background color or vertical gradient.
    pub background: BundleBackground,
    /// Ordered simulation time steps included in this bundle.
    pub time_steps: Vec<BundleTimeStep>,
    /// Ordered visible scene objects and their per-step binary mesh references.
    pub objects: Vec<BundleObject>,
}

/// Resolved camera stored in [`BundleManifest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleCamera {
    /// Camera eye position `[x, y, z]` relative to `dataset_center`.
    pub position: [f64; 3],
    /// Camera eye position `[x, y, z]` in world coordinates (`position + dataset_center`).
    pub world_position: [f64; 3],
    /// Camera focal point `[x, y, z]` relative to `dataset_center`.
    pub focal_point: [f64; 3],
    /// Camera focal point `[x, y, z]` in world coordinates (`focal_point + dataset_center`).
    pub world_focal_point: [f64; 3],
    /// Unit camera up vector `[ux, uy, uz]`.
    pub up: [f64; 3],
    /// Camera projection mode (`Perspective` or `Orthographic`).
    pub projection: ProjectionKind,
    /// Vertical field-of-view angle in degrees (used when `projection` is `Perspective`).
    pub view_angle_deg: f64,
    /// Orthographic vertical view height in world units (used when `projection` is `Orthographic`).
    pub ortho_scale: f64,
    /// Near clipping plane distance.
    pub clip_start: f64,
    /// Far clipping plane distance.
    pub clip_end: f64,
}

/// Resolved lighting vectors (`to_light` unit vectors in world space) stored in [`BundleManifest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleLights {
    /// Lighting preset (`Default` or `Headlight`).
    pub preset: LightingPreset,
    /// Unit vector pointing toward the key light.
    pub key_direction: [f64; 3],
    /// Unit vector pointing toward the fill light.
    pub fill_direction: [f64; 3],
    /// Unit vector pointing from `focal_point` toward the camera eye (`headlight`).
    pub headlight_direction: [f64; 3],
}

/// Resolved background specification stored in [`BundleManifest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleBackground {
    /// `"solid"` or `"gradient"`.
    pub kind: String,
    /// Primary sRGB color `[r, g, b]` in `[0.0, 1.0]` (solid color, or midpoint of gradient).
    pub color_srgb: [f32; 3],
    /// Top sRGB color `[r, g, b]` when `kind == "gradient"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_srgb: Option<[f32; 3]>,
    /// Bottom sRGB color `[r, g, b]` when `kind == "gradient"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bottom_srgb: Option<[f32; 3]>,
}

/// Simulation time step entry in [`BundleManifest::time_steps`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleTimeStep {
    /// Zero-based step index.
    pub step: usize,
    /// Simulation time value.
    pub time: f64,
}

/// Object entry in [`BundleManifest::objects`] corresponding to a visible representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleObject {
    /// Unique object identifier within the bundle (for example `"obj_0_cut"`).
    pub id: String,
    /// Source or filter dataset name (`representation.input`).
    pub input: String,
    /// Index of this representation in `scene.view.show`.
    pub representation_index: usize,
    /// Visual representation style (`Surface`, `SurfaceWithEdges`, `Wireframe`, `Points`).
    pub style: RepresentationStyle,
    /// Optional scalar array reference string (`representation.color_by`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_by: Option<String>,
    /// Resolved scalar association (`Point` or `Cell`), if scalar-colored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub association: Option<ArrayAssociation>,
    /// Solid sRGB color `[r, g, b]` in `[0.0, 1.0]` (used when `colormap` is `None`).
    pub solid_color_srgb: [f32; 3],
    /// 256-entry colormap lookup table and scalar range, when scalar-colored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colormap: Option<BundleColormap>,
    /// Material and edge display hints.
    pub material: BundleMaterialHints,
    /// Per-step `.tgbin` binary mesh file references.
    pub meshes: Vec<BundleMeshFileRef>,
}

/// 256-entry RGB colormap table and scalar range stored on [`BundleObject`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleColormap {
    /// Colormap name (`"viridis"`, `"cividis"`, `"cool_warm"`, `"turbo"`, `"grayscale"`).
    pub name: ColormapName,
    /// 256 sRGB triples `[r, g, b]` in `[0.0, 1.0]`, sampled uniformly for `t in [0.0, 1.0]`.
    pub lut_srgb: Vec<[f32; 3]>,
    /// Intrinsic `[min, max]` of the scalar component on the surface.
    pub data_range: [f64; 2],
    /// Effective `[min, max]` mapped to `[0.0, 1.0]`.
    pub effective_range: [f64; 2],
    /// Whether logarithmic scalar mapping is enabled.
    pub log_scale: bool,
    /// Optional diverging center value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub center: Option<f64>,
}

/// Material and edge display hints stored on [`BundleObject`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleMaterialHints {
    /// Surface opacity in `[0.0, 1.0]`.
    pub opacity: f64,
    /// Whether the triangulated surface is drawn.
    pub draw_surface: bool,
    /// Whether element wireframe edges are drawn.
    pub draw_edges: bool,
    /// Element edge sRGB color `[r, g, b]` in `[0.0, 1.0]`.
    pub edge_color_srgb: [f32; 3],
    /// Point size in pixels (for `RepresentationStyle::Points`).
    pub point_size: f64,
}

/// Reference to a single `.tgbin` binary mesh file for a specific object and time step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleMeshFileRef {
    /// Zero-based time-step index.
    pub step: usize,
    /// Simulation time value.
    pub time: f64,
    /// Relative path inside the bundle directory (for example `"meshes/obj_0_cut_step_0000.tgbin"`).
    pub file: String,
    /// Number of vertices (`V`) in the `.tgbin` file.
    pub num_vertices: u32,
    /// Number of triangles (`T`) in the `.tgbin` file.
    pub num_triangles: u32,
    /// Number of wireframe edge segments (`E`) in the `.tgbin` file.
    pub num_edge_segments: u32,
}

/// Unpacked mesh data for a single `.tgbin` binary file (`docs/bundle-format.md`).
#[derive(Debug, Clone, PartialEq)]
pub struct BundleMeshData {
    /// Vertex positions `[x, y, z]` (`f32`) relative to `manifest.dataset_center` (`V` elements).
    pub positions: Vec<[f32; 3]>,
    /// Triangle vertex index triples `[i0, i1, i2]` in `0..V` (`T` elements).
    pub triangles: Vec<[u32; 3]>,
    /// Unit vertex normals `[nx, ny, nz]` (`V` elements).
    pub normals: Vec<[f32; 3]>,
    /// Scalar values normalized to `[0.0, 1.0]` (`V` elements).
    pub scalars: Vec<f32>,
    /// Unnormalized raw scalar values (`V` elements).
    pub raw_scalars: Vec<f32>,
    /// Pairs of 3D endpoints `[[x0, y0, z0], [x1, y1, z1]]` relative to `dataset_center` for
    /// original element wireframe edges (`E` elements).
    pub edge_positions: Vec<[[f32; 3]; 2]>,
}

/// Complete in-memory representation of a Tetraglass bundle (`manifest.json` + `.tgbin` mesh files).
#[derive(Debug, Clone, PartialEq)]
pub struct BundleArchive {
    /// Parsed `manifest.json` metadata.
    pub manifest: BundleManifest,
    /// Map of relative path (such as `"meshes/obj_0_cut_step_0000.tgbin"`) to raw `.tgbin` bytes.
    pub mesh_files: BTreeMap<String, Vec<u8>>,
}

impl BundleArchive {
    /// Decodes and validates the `.tgbin` mesh file for `file_ref`.
    pub fn decode_mesh(&self, file_ref: &BundleMeshFileRef) -> Result<BundleMeshData, ExportError> {
        let bytes = self.mesh_files.get(&file_ref.file).ok_or_else(|| {
            ExportError::InvalidBundle(format!(
                "missing mesh file `{}` in bundle archive",
                file_ref.file
            ))
        })?;
        let data = decode_tgbin(bytes)?;
        if data.positions.len() != file_ref.num_vertices as usize {
            return Err(ExportError::InvalidBundle(format!(
                "vertex count mismatch in `{}`: manifest has {}, binary has {}",
                file_ref.file,
                file_ref.num_vertices,
                data.positions.len()
            )));
        }
        if data.triangles.len() != file_ref.num_triangles as usize {
            return Err(ExportError::InvalidBundle(format!(
                "triangle count mismatch in `{}`: manifest has {}, binary has {}",
                file_ref.file,
                file_ref.num_triangles,
                data.triangles.len()
            )));
        }
        if data.edge_positions.len() != file_ref.num_edge_segments as usize {
            return Err(ExportError::InvalidBundle(format!(
                "edge segment count mismatch in `{}`: manifest has {}, binary has {}",
                file_ref.file,
                file_ref.num_edge_segments,
                data.edge_positions.len()
            )));
        }
        Ok(data)
    }
}

/// Encodes a [`BundleMeshData`] into the little-endian `.tgbin` binary format (`docs/bundle-format.md`).
pub fn encode_tgbin(data: &BundleMeshData) -> Result<Vec<u8>, ExportError> {
    let num_verts = data.positions.len();
    if data.normals.len() != num_verts
        || data.scalars.len() != num_verts
        || data.raw_scalars.len() != num_verts
    {
        return Err(ExportError::InvalidBundle(format!(
            "BundleMeshData attribute length mismatch: positions={num_verts}, normals={}, scalars={}, raw_scalars={}",
            data.normals.len(),
            data.scalars.len(),
            data.raw_scalars.len()
        )));
    }
    let v_u32 = u32::try_from(num_verts).map_err(|_| {
        ExportError::InvalidBundle(format!("vertex count {num_verts} overflows u32"))
    })?;
    let t_u32 = u32::try_from(data.triangles.len()).map_err(|_| {
        ExportError::InvalidBundle(format!(
            "triangle count {} overflows u32",
            data.triangles.len()
        ))
    })?;
    let e_u32 = u32::try_from(data.edge_positions.len()).map_err(|_| {
        ExportError::InvalidBundle(format!(
            "edge segment count {} overflows u32",
            data.edge_positions.len()
        ))
    })?;

    for &[i0, i1, i2] in &data.triangles {
        if i0 >= v_u32 || i1 >= v_u32 || i2 >= v_u32 {
            return Err(ExportError::InvalidBundle(format!(
                "triangle index [{i0}, {i1}, {i2}] out of bounds for {v_u32} vertices"
            )));
        }
    }

    let total_bytes = TGBIN_HEADER_BYTES
        + 32 * num_verts
        + 12 * data.triangles.len()
        + 24 * data.edge_positions.len();
    let mut out = Vec::with_capacity(total_bytes);

    // 24-byte header
    out.extend_from_slice(&TGBIN_MAGIC.to_le_bytes());
    out.extend_from_slice(&TGBIN_VERSION.to_le_bytes());
    out.extend_from_slice(&v_u32.to_le_bytes());
    out.extend_from_slice(&t_u32.to_le_bytes());
    out.extend_from_slice(&e_u32.to_le_bytes());
    out.extend_from_slice(&TGBIN_FLAGS_DEFAULT.to_le_bytes());

    // Payload sections (all little-endian f32/u32 on supported targets; write explicitly in LE)
    for &[x, y, z] in &data.positions {
        out.extend_from_slice(&x.to_le_bytes());
        out.extend_from_slice(&y.to_le_bytes());
        out.extend_from_slice(&z.to_le_bytes());
    }
    for &[i0, i1, i2] in &data.triangles {
        out.extend_from_slice(&i0.to_le_bytes());
        out.extend_from_slice(&i1.to_le_bytes());
        out.extend_from_slice(&i2.to_le_bytes());
    }
    for &[nx, ny, nz] in &data.normals {
        out.extend_from_slice(&nx.to_le_bytes());
        out.extend_from_slice(&ny.to_le_bytes());
        out.extend_from_slice(&nz.to_le_bytes());
    }
    for &s in &data.scalars {
        out.extend_from_slice(&s.to_le_bytes());
    }
    for &rs in &data.raw_scalars {
        out.extend_from_slice(&rs.to_le_bytes());
    }
    for &[p0, p1] in &data.edge_positions {
        for &c in &p0 {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for &c in &p1 {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }

    debug_assert_eq!(out.len(), total_bytes);
    Ok(out)
}

/// Decodes and validates a little-endian `.tgbin` binary mesh buffer (`docs/bundle-format.md`).
pub fn decode_tgbin(bytes: &[u8]) -> Result<BundleMeshData, ExportError> {
    if bytes.len() < TGBIN_HEADER_BYTES {
        return Err(ExportError::InvalidBundle(format!(
            "tgbin buffer too short ({} bytes, minimum {TGBIN_HEADER_BYTES})",
            bytes.len()
        )));
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != TGBIN_MAGIC {
        return Err(ExportError::InvalidBundle(format!(
            "invalid tgbin magic 0x{magic:08X} (expected 0x{TGBIN_MAGIC:08X})"
        )));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != TGBIN_VERSION {
        return Err(ExportError::InvalidBundle(format!(
            "unsupported tgbin version {version} (expected {TGBIN_VERSION})"
        )));
    }
    let num_verts = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let num_tris = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    let num_edges = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
    let _flags = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);

    let expected_bytes =
        TGBIN_HEADER_BYTES
            .checked_add(num_verts.checked_mul(32).ok_or_else(|| {
                ExportError::InvalidBundle("vertex byte count overflow".to_string())
            })?)
            .and_then(|n| n.checked_add(num_tris.checked_mul(12)?))
            .and_then(|n| n.checked_add(num_edges.checked_mul(24)?))
            .ok_or_else(|| ExportError::InvalidBundle("tgbin size overflow".to_string()))?;

    if bytes.len() != expected_bytes {
        return Err(ExportError::InvalidBundle(format!(
            "tgbin byte length mismatch: expected {expected_bytes} bytes for V={num_verts}, T={num_tris}, E={num_edges}, got {}",
            bytes.len()
        )));
    }

    let mut offset = TGBIN_HEADER_BYTES;

    let pos_end = offset + num_verts * 12;
    let mut positions = Vec::with_capacity(num_verts);
    for chunk in bytes[offset..pos_end].as_chunks::<12>().0 {
        positions.push([
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
            f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
        ]);
    }
    offset = pos_end;

    let tri_end = offset + num_tris * 12;
    let mut triangles = Vec::with_capacity(num_tris);
    for chunk in bytes[offset..tri_end].as_chunks::<12>().0 {
        let i0 = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let i1 = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let i2 = u32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
        if (i0 as usize) >= num_verts || (i1 as usize) >= num_verts || (i2 as usize) >= num_verts {
            return Err(ExportError::InvalidBundle(format!(
                "triangle index [{i0}, {i1}, {i2}] out of bounds for V={num_verts}"
            )));
        }
        triangles.push([i0, i1, i2]);
    }
    offset = tri_end;

    let norm_end = offset + num_verts * 12;
    let mut normals = Vec::with_capacity(num_verts);
    for chunk in bytes[offset..norm_end].as_chunks::<12>().0 {
        normals.push([
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
            f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
        ]);
    }
    offset = norm_end;

    let scal_end = offset + num_verts * 4;
    let mut scalars = Vec::with_capacity(num_verts);
    for &chunk in bytes[offset..scal_end].as_chunks::<4>().0 {
        scalars.push(f32::from_le_bytes(chunk));
    }
    offset = scal_end;

    let raw_scal_end = offset + num_verts * 4;
    let mut raw_scalars = Vec::with_capacity(num_verts);
    for &chunk in bytes[offset..raw_scal_end].as_chunks::<4>().0 {
        raw_scalars.push(f32::from_le_bytes(chunk));
    }
    offset = raw_scal_end;

    let edge_end = offset + num_edges * 24;
    let mut edge_positions = Vec::with_capacity(num_edges);
    for chunk in bytes[offset..edge_end].as_chunks::<24>().0 {
        let p0 = [
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
            f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
        ];
        let p1 = [
            f32::from_le_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]),
            f32::from_le_bytes([chunk[16], chunk[17], chunk[18], chunk[19]]),
            f32::from_le_bytes([chunk[20], chunk[21], chunk[22], chunk[23]]),
        ];
        edge_positions.push([p0, p1]);
    }

    Ok(BundleMeshData {
        positions,
        triangles,
        normals,
        scalars,
        raw_scalars,
        edge_positions,
    })
}

/// Exports a single [`EvaluatedScene`] into an in-memory [`BundleArchive`] (`M3.5`).
pub fn export_evaluated_scene_to_bundle(
    scene: &Scene,
    evaluated: &EvaluatedScene,
) -> Result<BundleArchive, ExportError> {
    build_bundle_from_evaluated_steps(scene, std::slice::from_ref(evaluated))
}

/// Exports a sequence of pre-evaluated [`EvaluatedScene`] steps into an in-memory
/// [`BundleArchive`] (`M4.5`).
pub fn export_evaluated_steps_to_bundle(
    scene: &Scene,
    evaluated_steps: &[EvaluatedScene],
) -> Result<BundleArchive, ExportError> {
    build_bundle_from_evaluated_steps(scene, evaluated_steps)
}

/// Evaluates `scene` across time steps using `evaluator` and exports all requested steps into an
/// in-memory [`BundleArchive`] (`M3.5`).
///
/// - If `steps` is `Some(indices)`, exports those zero-based step indices in order.
/// - If `steps` is `None`, exports all available time steps (`0..num_steps`).
pub fn export_scene_to_bundle(
    scene: &Scene,
    evaluator: &mut SceneEvaluator,
    steps: Option<&[usize]>,
) -> Result<BundleArchive, ExportError> {
    let total_steps = infer_evaluator_step_count(scene, evaluator);
    let step_list: Vec<usize> = match steps {
        Some(explicit) => explicit.to_vec(),
        None => (0..total_steps).collect(),
    };

    let mut evaluated_steps = Vec::with_capacity(step_list.len().max(1));
    if step_list.is_empty() {
        let selector = scene.output.as_ref().map_or(TimeSelector::LAST, |o| o.time);
        evaluated_steps.push(evaluator.evaluate_scene_at_time(scene, selector)?);
    } else {
        for step_idx in step_list {
            evaluated_steps
                .push(evaluator.evaluate_scene_at_time(scene, TimeSelector::step(step_idx))?);
        }
    }

    build_bundle_from_evaluated_steps(scene, &evaluated_steps)
}

/// Summary of files written by [`write_bundle_dir`] or [`write_scene_bundle_output`].
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleWriteSummary {
    /// Root bundle directory path.
    pub bundle_dir: PathBuf,
    /// Path to `<bundle_dir>/manifest.json`.
    pub manifest_file: PathBuf,
    /// Paths to all written `.tgbin` mesh files.
    pub mesh_files: Vec<PathBuf>,
    /// Number of time steps exported.
    pub num_steps: usize,
    /// Total triangles across all objects and steps.
    pub total_triangles: usize,
}

/// Writes an in-memory [`BundleArchive`] to `bundle_dir` on disk (`M3.5`).
#[cfg(not(target_arch = "wasm32"))]
pub fn write_bundle_dir(
    bundle_dir: impl AsRef<Path>,
    archive: &BundleArchive,
) -> Result<BundleWriteSummary, ExportError> {
    let bundle_dir = bundle_dir.as_ref();
    fs::create_dir_all(bundle_dir)?;
    let meshes_dir = bundle_dir.join("meshes");
    fs::create_dir_all(&meshes_dir)?;

    let mut written_meshes = Vec::with_capacity(archive.mesh_files.len());
    for (rel_path, bytes) in &archive.mesh_files {
        let full_path = bundle_dir.join(rel_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&full_path, bytes)?;
        written_meshes.push(full_path);
    }

    let manifest_path = bundle_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&archive.manifest)?;
    fs::write(&manifest_path, manifest_json)?;

    let total_triangles: usize = archive
        .manifest
        .objects
        .iter()
        .flat_map(|o| o.meshes.iter())
        .map(|m| m.num_triangles as usize)
        .sum();

    Ok(BundleWriteSummary {
        bundle_dir: bundle_dir.to_path_buf(),
        manifest_file: manifest_path,
        mesh_files: written_meshes,
        num_steps: archive.manifest.time_steps.len(),
        total_triangles,
    })
}

/// Evaluates `scene` and writes a Tetraglass bundle directory to `bundle_dir` (`M3.5`).
///
/// When `all_steps` is `false`, exports only the active step selected by `scene.output.time`
/// (or `TimeSelector::LAST`); when `all_steps` is `true`, exports every step in the series.
#[cfg(not(target_arch = "wasm32"))]
pub fn write_scene_bundle_output(
    bundle_dir: impl AsRef<Path>,
    scene: &Scene,
    evaluator: &mut SceneEvaluator,
    all_steps: bool,
) -> Result<BundleWriteSummary, ExportError> {
    let archive = if all_steps {
        export_scene_to_bundle(scene, evaluator, None)?
    } else {
        let selector = scene.output.as_ref().map_or(TimeSelector::LAST, |o| o.time);
        let evaluated = evaluator.evaluate_scene_at_time(scene, selector)?;
        export_evaluated_scene_to_bundle(scene, &evaluated)?
    };
    write_bundle_dir(bundle_dir, &archive)
}

/// Reads and validates a Tetraglass bundle directory (`manifest.json` and all referenced `.tgbin`
/// files) from disk (`M3.5`).
#[cfg(not(target_arch = "wasm32"))]
pub fn read_bundle_dir(bundle_dir: impl AsRef<Path>) -> Result<BundleArchive, ExportError> {
    let bundle_dir = bundle_dir.as_ref();
    let manifest_path = bundle_dir.join("manifest.json");
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes)?;

    if manifest.version != BUNDLE_FORMAT_VERSION {
        return Err(ExportError::InvalidBundle(format!(
            "unsupported bundle manifest version {} (expected {BUNDLE_FORMAT_VERSION})",
            manifest.version
        )));
    }
    if manifest.format != BUNDLE_FORMAT_ID {
        return Err(ExportError::InvalidBundle(format!(
            "invalid bundle format identifier `{}` (expected `{BUNDLE_FORMAT_ID}`)",
            manifest.format
        )));
    }

    let mut mesh_files = BTreeMap::new();
    for obj in &manifest.objects {
        if let Some(ref cmap) = obj.colormap
            && cmap.lut_srgb.len() != 256
        {
            return Err(ExportError::InvalidBundle(format!(
                "object `{}` colormap LUT has {} entries (expected 256)",
                obj.id,
                cmap.lut_srgb.len()
            )));
        }
        for mesh_ref in &obj.meshes {
            let full_path = bundle_dir.join(&mesh_ref.file);
            let bytes = fs::read(&full_path)?;
            mesh_files.insert(mesh_ref.file.clone(), bytes);
        }
    }

    let archive = BundleArchive {
        manifest,
        mesh_files,
    };
    // Validate every referenced mesh file header and counts.
    for obj in &archive.manifest.objects {
        for mesh_ref in &obj.meshes {
            let _ = archive.decode_mesh(mesh_ref)?;
        }
    }
    Ok(archive)
}

fn build_bundle_from_evaluated_steps(
    scene: &Scene,
    evaluated_steps: &[EvaluatedScene],
) -> Result<BundleArchive, ExportError> {
    let first_eval = evaluated_steps.first().ok_or_else(|| {
        ExportError::InvalidBundle("cannot export bundle with zero evaluated steps".to_string())
    })?;

    let bounds = resolve_combined_bounds(evaluated_steps);
    let dataset_center = bounds.center();
    let image_size = scene
        .output
        .as_ref()
        .map_or(DEFAULT_OUTPUT_SIZE, |o| o.size);
    let (camera, lights) = resolve_bundle_camera_and_lights(scene, bounds, dataset_center);
    let background = resolve_bundle_background(&scene.view.background);

    let time_steps: Vec<BundleTimeStep> = evaluated_steps
        .iter()
        .map(|ev| BundleTimeStep {
            step: ev.time_step,
            time: ev.time_value,
        })
        .collect();

    let num_objects = first_eval.representations.len();
    let mut objects = Vec::with_capacity(num_objects);
    let mut mesh_files = BTreeMap::new();

    for obj_slot in 0..num_objects {
        let first_rep = &first_eval.representations[obj_slot];
        let obj_id = format!(
            "obj_{}_{}",
            first_rep.representation_index,
            sanitize_identifier(&first_rep.input)
        );
        let (draw_surface, draw_edges) = match first_rep.style {
            RepresentationStyle::Wireframe => (false, true),
            RepresentationStyle::SurfaceWithEdges => (true, true),
            _ => (true, false),
        };

        let colormap = first_rep.coloring.as_ref().map(|c| BundleColormap {
            name: c.colormap,
            lut_srgb: colormap_lut_srgb_256(c.colormap).to_vec(),
            data_range: [c.data_range.min, c.data_range.max],
            effective_range: c.effective_range,
            log_scale: c.log_scale,
            center: c.center,
        });

        let mut obj_meshes = Vec::with_capacity(evaluated_steps.len());
        for ev in evaluated_steps {
            let Some(rep) = ev.representations.get(obj_slot) else {
                continue;
            };
            let mesh_data = build_bundle_mesh_data(rep, dataset_center, draw_edges)?;
            let encoded = encode_tgbin(&mesh_data)?;
            let rel_file = format!("meshes/{obj_id}_step_{:04}.tgbin", ev.time_step);
            obj_meshes.push(BundleMeshFileRef {
                step: ev.time_step,
                time: ev.time_value,
                file: rel_file.clone(),
                num_vertices: mesh_data.positions.len() as u32,
                num_triangles: mesh_data.triangles.len() as u32,
                num_edge_segments: mesh_data.edge_positions.len() as u32,
            });
            mesh_files.insert(rel_file, encoded);
        }

        objects.push(BundleObject {
            id: obj_id,
            input: first_rep.input.clone(),
            representation_index: first_rep.representation_index,
            style: first_rep.style,
            color_by: first_rep.coloring.as_ref().map(|c| c.raw_ref.clone()),
            association: first_rep.coloring.as_ref().map(|c| c.association),
            solid_color_srgb: first_rep.color,
            colormap,
            material: BundleMaterialHints {
                opacity: first_rep.opacity,
                draw_surface,
                draw_edges,
                edge_color_srgb: first_rep.edge_color,
                point_size: first_rep.point_size,
            },
            meshes: obj_meshes,
        });
    }

    let manifest = BundleManifest {
        version: BUNDLE_FORMAT_VERSION,
        format: BUNDLE_FORMAT_ID.to_string(),
        units: "meters".to_string(),
        dataset_center,
        image_size,
        camera,
        lights,
        background,
        time_steps,
        objects,
    };

    Ok(BundleArchive {
        manifest,
        mesh_files,
    })
}

fn build_bundle_mesh_data(
    rep: &EvaluatedRepresentation,
    dataset_center: [f64; 3],
    include_edges: bool,
) -> Result<BundleMeshData, ExportError> {
    let surface = rep.surface.as_ref();
    let points = surface.points();
    let src_triangles = surface.triangles();
    let num_verts = src_triangles.len() * 3;

    let resolved_scalars = match &rep.coloring {
        Some(coloring) => Some(extract_representation_scalars(surface, coloring)?),
        None => None,
    };

    let mut positions = Vec::with_capacity(num_verts);
    let mut triangles = Vec::with_capacity(src_triangles.len());
    let mut normals = Vec::with_capacity(num_verts);
    let mut scalars = Vec::with_capacity(num_verts);
    let mut raw_scalars = Vec::with_capacity(num_verts);

    for (tri_idx, &[i0, i1, i2]) in src_triangles.iter().enumerate() {
        let p0 = points[i0 as usize];
        let p1 = points[i1 as usize];
        let p2 = points[i2 as usize];
        let face_normal = compute_triangle_normal(p0, p1, p2);
        let base_v = (tri_idx * 3) as u32;
        triangles.push([base_v, base_v + 1, base_v + 2]);

        for &pt_idx in &[i0, i1, i2] {
            let p = points[pt_idx as usize];
            positions.push([
                (p[0] - dataset_center[0]) as f32,
                (p[1] - dataset_center[1]) as f32,
                (p[2] - dataset_center[2]) as f32,
            ]);
            normals.push(face_normal);

            if let (Some(coloring), Some(sc)) = (&rep.coloring, &resolved_scalars) {
                let raw_val = match coloring.association {
                    ArrayAssociation::Point => sc[pt_idx as usize],
                    ArrayAssociation::Cell => sc[tri_idx],
                };
                let norm_t = normalize_scalar(
                    raw_val,
                    coloring.effective_range,
                    coloring.log_scale,
                    coloring.center,
                );
                let raw_f32 = if raw_val.is_finite() {
                    raw_val as f32
                } else {
                    0.0
                };
                scalars.push(norm_t);
                raw_scalars.push(raw_f32);
            } else {
                scalars.push(0.0);
                raw_scalars.push(0.0);
            }
        }
    }

    let edge_positions = if include_edges {
        surface
            .wireframe_edges()
            .iter()
            .map(|&[e0, e1]| {
                let a = points[e0 as usize];
                let b = points[e1 as usize];
                [
                    [
                        (a[0] - dataset_center[0]) as f32,
                        (a[1] - dataset_center[1]) as f32,
                        (a[2] - dataset_center[2]) as f32,
                    ],
                    [
                        (b[0] - dataset_center[0]) as f32,
                        (b[1] - dataset_center[1]) as f32,
                        (b[2] - dataset_center[2]) as f32,
                    ],
                ]
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(BundleMeshData {
        positions,
        triangles,
        normals,
        scalars,
        raw_scalars,
        edge_positions,
    })
}

fn resolve_combined_bounds(evaluated_steps: &[EvaluatedScene]) -> BoundingBox {
    for ev in evaluated_steps {
        if let Some(b) = ev.camera_bounds {
            return b;
        }
    }
    let mut combined: Option<BoundingBox> = None;
    for ev in evaluated_steps {
        for rep in &ev.representations {
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
    }
    combined.unwrap_or(BoundingBox {
        min: [-1.0, -1.0, -1.0],
        max: [1.0, 1.0, 1.0],
    })
}

fn resolve_bundle_camera_and_lights(
    scene: &Scene,
    bounds: BoundingBox,
    dataset_center: [f64; 3],
) -> (BundleCamera, BundleLights) {
    let cam_cfg = &scene.view.camera;
    let radius = (bounds.diagonal() * 0.5).max(1e-3);
    let view_angle_deg = cam_cfg.view_angle.clamp(1.0, 170.0);
    let half_fov = 0.5 * view_angle_deg.to_radians();
    let base_distance = (radius * 1.25) / half_fov.tan();

    let mut focal_point = [0.0, 0.0, 0.0];
    if let Some(fp) = cam_cfg.focal_point {
        focal_point = [
            fp[0] - dataset_center[0],
            fp[1] - dataset_center[1],
            fp[2] - dataset_center[2],
        ];
    }

    let (position, azimuth_deg, elevation_deg) = if let Some(pos) = cam_cfg.position {
        let rel_pos = [
            pos[0] - dataset_center[0],
            pos[1] - dataset_center[1],
            pos[2] - dataset_center[2],
        ];
        let dx = rel_pos[0] - focal_point[0];
        let dy = rel_pos[1] - focal_point[1];
        let dz = rel_pos[2] - focal_point[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-5);
        let el = (dy / dist).clamp(-1.0, 1.0).asin().to_degrees();
        let az = dx.atan2(dz).to_degrees();
        (rel_pos, az, el.clamp(-89.0, 89.0))
    } else {
        let zoom = cam_cfg.zoom.max(1e-3);
        let dist = (base_distance / zoom).max(1e-5);
        let az = cam_cfg.azimuth;
        let el = cam_cfg.elevation.clamp(-89.0, 89.0);
        let az_rad = az.to_radians();
        let el_rad = el.to_radians();
        let pos = [
            focal_point[0] + dist * el_rad.cos() * az_rad.sin(),
            focal_point[1] + dist * el_rad.sin(),
            focal_point[2] + dist * el_rad.cos() * az_rad.cos(),
        ];
        (pos, az, el)
    };
    let _ = (azimuth_deg, elevation_deg);

    let back = normalize3([
        position[0] - focal_point[0],
        position[1] - focal_point[1],
        position[2] - focal_point[2],
    ]);
    let world_up = cam_cfg.view_up.unwrap_or([0.0, 1.0, 0.0]);
    let base_right = normalize3(cross3(world_up, back));
    let base_up = cross3(back, base_right);
    let roll_rad = cam_cfg.roll.to_radians();
    let (c, s) = (roll_rad.cos(), roll_rad.sin());
    let right = [
        c * base_right[0] + s * base_up[0],
        c * base_right[1] + s * base_up[1],
        c * base_right[2] + s * base_up[2],
    ];
    let up = normalize3([
        -s * base_right[0] + c * base_up[0],
        -s * base_right[1] + c * base_up[1],
        -s * base_right[2] + c * base_up[2],
    ]);

    let dx = position[0] - focal_point[0];
    let dy = position[1] - focal_point[1];
    let dz = position[2] - focal_point[2];
    let effective_dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-5);
    let ortho_scale = 2.0 * effective_dist * half_fov.tan();
    let clip_start = (effective_dist - radius * 2.5)
        .max(effective_dist * 0.01)
        .max(1e-4);
    let clip_end = (effective_dist + radius * 3.5).max(clip_start + 1.0);

    let key_direction = normalize3([
        0.55 * right[0] + 0.65 * up[0] + 0.52 * back[0],
        0.55 * right[1] + 0.65 * up[1] + 0.52 * back[1],
        0.55 * right[2] + 0.65 * up[2] + 0.52 * back[2],
    ]);
    let fill_direction = normalize3([
        -0.65 * right[0] - 0.25 * up[0] + 0.71 * back[0],
        -0.65 * right[1] - 0.25 * up[1] + 0.71 * back[1],
        -0.65 * right[2] - 0.25 * up[2] + 0.71 * back[2],
    ]);

    let camera = BundleCamera {
        position,
        world_position: [
            position[0] + dataset_center[0],
            position[1] + dataset_center[1],
            position[2] + dataset_center[2],
        ],
        focal_point,
        world_focal_point: [
            focal_point[0] + dataset_center[0],
            focal_point[1] + dataset_center[1],
            focal_point[2] + dataset_center[2],
        ],
        up,
        projection: cam_cfg.projection,
        view_angle_deg,
        ortho_scale,
        clip_start,
        clip_end,
    };

    let lights = BundleLights {
        preset: scene.view.lights,
        key_direction,
        fill_direction,
        headlight_direction: back,
    };

    (camera, lights)
}

fn resolve_bundle_background(bg: &BackgroundSpec) -> BundleBackground {
    match bg {
        BackgroundSpec::Solid(color) => {
            let rgb = color.resolve_rgb().unwrap_or([1.0, 1.0, 1.0]);
            BundleBackground {
                kind: "solid".to_string(),
                color_srgb: rgb,
                top_srgb: None,
                bottom_srgb: None,
            }
        }
        BackgroundSpec::Gradient(grad) => {
            let top = grad.top.resolve_rgb().unwrap_or([1.0, 1.0, 1.0]);
            let bottom = grad.bottom.resolve_rgb().unwrap_or([1.0, 1.0, 1.0]);
            BundleBackground {
                kind: "gradient".to_string(),
                color_srgb: [
                    0.5 * (top[0] + bottom[0]),
                    0.5 * (top[1] + bottom[1]),
                    0.5 * (top[2] + bottom[2]),
                ],
                top_srgb: Some(top),
                bottom_srgb: Some(bottom),
            }
        }
    }
}

fn sanitize_identifier(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "surface".to_string()
    } else {
        out
    }
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

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let len = dot3(v, v).sqrt();
    if len > 1e-12 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

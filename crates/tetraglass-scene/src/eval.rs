//! Lazy pipeline evaluation and input-hash caching (`M1.3`).
//!
//! [`SceneEvaluator`] evaluates the directed acyclic graph of sources and filters in a
//! [`Scene`], caching filter outputs and extracted surfaces by a deterministic 64-bit
//! input hash ([`PipelineHash`]).
//!
//! Changing view-only properties (`colormap`, `color_by`, `range`, `log_scale`, `opacity`,
//! `colorbar`, `camera`, `lights`, `background`, or `output`) never invalidates or re-runs
//! upstream filters.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use tetraglass_core::{
    ArrayBuffer, Attributes, BoundingBox, SurfaceMesh, UnstructuredGrid, ValueRange,
};
use tetraglass_filters::extract_external_surface;

use crate::{
    ArrayAssociation, ColorbarSpec, ColormapName, DatasetMetadata, Filter, ParsedArrayRef,
    RangeSpec, Representation, RepresentationStyle, ResolvedComponent, Scene, SceneError,
    validate_scene_with_metadata,
};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Deterministic 64-bit FNV-1a hasher used for content and pipeline-node hashing.
#[derive(Debug, Clone, Copy)]
struct FnvHasher(u64);

impl FnvHasher {
    const fn new() -> Self {
        Self(FNV_OFFSET_BASIS)
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
        self.0 = h;
    }

    fn write_u8(&mut self, v: u8) {
        self.write_bytes(&[v]);
    }

    fn write_u32(&mut self, v: u32) {
        self.write_bytes(&v.to_le_bytes());
    }

    fn write_u64(&mut self, v: u64) {
        self.write_bytes(&v.to_le_bytes());
    }

    fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64);
    }

    fn write_f64(&mut self, v: f64) {
        // Canonicalize -0.0 and +0.0 to the same bit pattern.
        let bits = if v == 0.0 { 0u64 } else { v.to_bits() };
        self.write_u64(bits);
    }

    fn write_str(&mut self, s: &str) {
        self.write_usize(s.len());
        self.write_bytes(s.as_bytes());
    }

    const fn finish(self) -> u64 {
        self.0
    }
}

/// Deterministic 64-bit content/input hash identifying a pipeline dataset node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineHash(pub u64);

/// Computes a deterministic 64-bit content hash for an [`UnstructuredGrid`].
#[must_use]
pub fn hash_unstructured_grid(grid: &UnstructuredGrid) -> PipelineHash {
    let mut h = FnvHasher::new();
    h.write_str("UnstructuredGrid:v1");

    h.write_usize(grid.num_points());
    for &[x, y, z] in grid.points() {
        h.write_f64(x);
        h.write_f64(y);
        h.write_f64(z);
    }

    h.write_usize(grid.num_cells());
    for ct in grid.cell_types() {
        h.write_u8(ct.vtk_type_id());
        h.write_usize(ct.expected_node_count().unwrap_or(0));
    }
    for &off in grid.offsets() {
        h.write_u32(off);
    }
    for &idx in grid.connectivity() {
        h.write_u32(idx);
    }

    if let Some(pf) = grid.polyhedron_faces() {
        h.write_u8(1);
        h.write_usize(pf.face_offsets().len());
        for &off in pf.face_offsets() {
            h.write_u32(off);
        }
        h.write_usize(pf.faces().len());
        for &val in pf.faces() {
            h.write_u32(val);
        }
    } else {
        h.write_u8(0);
    }

    hash_attributes(&mut h, grid.point_data());
    hash_attributes(&mut h, grid.cell_data());

    PipelineHash(h.finish())
}

fn hash_attributes(h: &mut FnvHasher, attrs: &Attributes) {
    h.write_usize(attrs.len());
    for (name, arr) in attrs.iter() {
        h.write_str(name);
        h.write_u8(arr.components().as_u8());
        h.write_usize(arr.num_tuples());
        match arr.buffer() {
            ArrayBuffer::F32(buf) => {
                h.write_u8(1);
                for &v in buf.iter() {
                    h.write_f64(f64::from(v));
                }
            }
            ArrayBuffer::F64(buf) => {
                h.write_u8(2);
                for &v in buf.iter() {
                    h.write_f64(v);
                }
            }
            ArrayBuffer::I32(buf) => {
                h.write_u8(3);
                for &v in buf.iter() {
                    h.write_u32(v as u32);
                }
            }
            ArrayBuffer::I64(buf) => {
                h.write_u8(4);
                for &v in buf.iter() {
                    h.write_u64(v as u64);
                }
            }
            ArrayBuffer::U8(buf) => {
                h.write_u8(5);
                h.write_bytes(buf);
            }
        }
    }
}

fn hash_source_node(content_hash: PipelineHash, linearize_level: u32) -> PipelineHash {
    let mut h = FnvHasher::new();
    h.write_str("source");
    h.write_u64(content_hash.0);
    h.write_u32(linearize_level);
    PipelineHash(h.finish())
}

fn hash_surface_extraction(upstream: PipelineHash) -> PipelineHash {
    let mut h = FnvHasher::new();
    h.write_str("filter:surface");
    h.write_u64(upstream.0);
    PipelineHash(h.finish())
}

/// Output of evaluating a source or filter node in the scene pipeline.
#[derive(Debug, Clone)]
pub enum EvaluatedDataset {
    /// Volume or general unstructured grid dataset.
    Grid(Arc<UnstructuredGrid>),
    /// Extracted or generated triangulated surface mesh.
    Surface(Arc<SurfaceMesh>),
}

impl EvaluatedDataset {
    /// Returns the axis-aligned bounding box of this dataset, if non-empty.
    #[must_use]
    pub fn bounds(&self) -> Option<BoundingBox> {
        match self {
            Self::Grid(grid) => grid.bounds(),
            Self::Surface(surface) => surface.bounds(),
        }
    }
}

/// Resolved scalar coloring parameters for an [`EvaluatedRepresentation`].
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatedColoring {
    /// Original array reference string from `representation.color_by`.
    pub raw_ref: String,
    /// Resolved association (`Point` or `Cell`; point data wins when present in both).
    pub association: ArrayAssociation,
    /// Attribute array name on the dataset.
    pub array_name: String,
    /// Resolved component or magnitude selector.
    pub component: ResolvedComponent,
    /// Intrinsic `[min, max]` range of the array component on the dataset.
    pub data_range: ValueRange,
    /// Effective `[min, max]` range mapped to the colormap (`"auto"` or explicit `[min, max]`).
    pub effective_range: [f64; 2],
    /// Selected colormap.
    pub colormap: ColormapName,
    /// Whether logarithmic scaling is enabled.
    pub log_scale: bool,
    /// Optional diverging center value.
    pub center: Option<f64>,
}

/// Evaluated representation ready for GPU upload or headless rendering.
#[derive(Debug, Clone)]
pub struct EvaluatedRepresentation {
    /// Index of this representation in `scene.view.show`.
    pub representation_index: usize,
    /// Name of the input source or filter dataset.
    pub input: String,
    /// Deterministic pipeline hash of the evaluated [`SurfaceMesh`].
    pub surface_hash: PipelineHash,
    /// Triangulated surface mesh produced by the pipeline.
    pub surface: Arc<SurfaceMesh>,
    /// Visual drawing style (`Surface`, `SurfaceWithEdges`, `Wireframe`, `Points`, ...).
    pub style: RepresentationStyle,
    /// Resolved solid RGB color in `0.0..=1.0`.
    pub color: [f32; 3],
    /// Resolved element-edge RGB color in `0.0..=1.0`.
    pub edge_color: [f32; 3],
    /// Resolved scalar coloring configuration, or `None` for solid coloring.
    pub coloring: Option<EvaluatedColoring>,
    /// Opacity in `0.0..=1.0`.
    pub opacity: f64,
    /// Point diameter in pixels for `RepresentationStyle::Points`.
    pub point_size: f64,
    /// Colorbar legend configuration.
    pub colorbar: ColorbarSpec,
}

/// Result of evaluating a [`Scene`] via [`SceneEvaluator::evaluate_scene`].
#[derive(Debug, Clone)]
pub struct EvaluatedScene {
    /// Ordered list of visible evaluated representations from `scene.view.show`.
    pub representations: Vec<EvaluatedRepresentation>,
    /// Combined world-space bounding box framed by `scene.view.camera.fit`.
    pub camera_bounds: Option<BoundingBox>,
}

#[derive(Debug, Clone)]
struct LoadedSource {
    content_hash: PipelineHash,
    grid: Arc<UnstructuredGrid>,
    metadata: DatasetMetadata,
}

/// Lazy scene pipeline evaluator with input-hash caching (`M1.3`).
///
/// Tracks `filter_executions` and `cache_hits` so callers and tests can verify that
/// changing view-only settings (`colormap`, `camera`, `range`, `background`, `output`)
/// never re-runs filters.
#[derive(Debug, Default)]
pub struct SceneEvaluator {
    sources: BTreeMap<String, LoadedSource>,
    surface_cache: HashMap<PipelineHash, Arc<SurfaceMesh>>,
    filter_executions: usize,
    cache_hits: usize,
}

impl SceneEvaluator {
    /// Creates an empty [`SceneEvaluator`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a [`SceneEvaluator`] pre-populated with named source [`UnstructuredGrid`]s.
    #[must_use]
    pub fn from_sources(sources: impl IntoIterator<Item = (String, UnstructuredGrid)>) -> Self {
        let mut eval = Self::new();
        for (name, grid) in sources {
            eval.insert_source(name, grid);
        }
        eval
    }

    /// Inserts or replaces a named source [`UnstructuredGrid`].
    pub fn insert_source(&mut self, name: impl Into<String>, grid: UnstructuredGrid) {
        self.insert_shared_source(name, Arc::new(grid));
    }

    /// Inserts or replaces a named source [`Arc<UnstructuredGrid>`].
    pub fn insert_shared_source(&mut self, name: impl Into<String>, grid: Arc<UnstructuredGrid>) {
        let name = name.into();
        // Fast path: if the exact same Arc is already stored for this source name, keep its hash.
        if let Some(existing) = self.sources.get(&name)
            && Arc::ptr_eq(&existing.grid, &grid)
        {
            return;
        }
        let content_hash = hash_unstructured_grid(&grid);
        let metadata = DatasetMetadata::from_grid(&grid);
        self.sources.insert(
            name,
            LoadedSource {
                content_hash,
                grid,
                metadata,
            },
        );
    }

    /// Removes a loaded source by name, returning its [`Arc<UnstructuredGrid>`] if present.
    pub fn remove_source(&mut self, name: &str) -> Option<Arc<UnstructuredGrid>> {
        self.sources.remove(name).map(|s| s.grid)
    }

    /// Returns a reference to a loaded source [`Arc<UnstructuredGrid>`] by name.
    #[must_use]
    pub fn source_grid(&self, name: &str) -> Option<&Arc<UnstructuredGrid>> {
        self.sources.get(name).map(|s| &s.grid)
    }

    /// Clears all cached filter outputs without removing loaded sources.
    pub fn clear_cache(&mut self) {
        self.surface_cache.clear();
    }

    /// Resets the [`Self::filter_executions`] and [`Self::cache_hits`] counters to zero.
    pub fn reset_counters(&mut self) {
        self.filter_executions = 0;
        self.cache_hits = 0;
    }

    /// Returns the total number of filter executions (cache misses) since creation or [`Self::reset_counters`].
    #[must_use]
    pub const fn filter_executions(&self) -> usize {
        self.filter_executions
    }

    /// Returns the total number of filter cache hits since creation or [`Self::reset_counters`].
    #[must_use]
    pub const fn cache_hits(&self) -> usize {
        self.cache_hits
    }

    /// Returns the number of [`SurfaceMesh`] entries currently stored in the filter cache.
    #[must_use]
    pub fn cached_surface_count(&self) -> usize {
        self.surface_cache.len()
    }

    /// Evaluates a single named source or filter dataset in `scene`.
    ///
    /// Filter results are cached by their upstream [`PipelineHash`].
    pub fn evaluate_dataset(
        &mut self,
        scene: &Scene,
        dataset_name: &str,
    ) -> Result<(PipelineHash, EvaluatedDataset), SceneError> {
        if let Some(source_cfg) = scene.sources.get(dataset_name) {
            let loaded =
                self.sources
                    .get(dataset_name)
                    .ok_or_else(|| SceneError::MissingDataset {
                        name: dataset_name.to_string(),
                    })?;
            let node_hash = hash_source_node(loaded.content_hash, source_cfg.linearize_level);
            return Ok((node_hash, EvaluatedDataset::Grid(Arc::clone(&loaded.grid))));
        }

        if let Some(filter) = scene.filters.get(dataset_name) {
            match filter {
                Filter::Surface { input } => {
                    let (upstream_hash, upstream_data) = self.evaluate_dataset(scene, input)?;
                    let surface_hash = hash_surface_extraction(upstream_hash);
                    let surface = self.get_or_extract_surface(surface_hash, upstream_data)?;
                    return Ok((surface_hash, EvaluatedDataset::Surface(surface)));
                }
            }
        }

        Err(SceneError::MissingDataset {
            name: dataset_name.to_string(),
        })
    }

    /// Evaluates the displayable [`SurfaceMesh`] for a named source or filter in `scene`.
    ///
    /// Per `docs/scene-format.md`, showing a volume mesh source directly implies external
    /// surface extraction, which is cached under the same input hash as an explicit
    /// `Filter::Surface`.
    pub fn evaluate_surface(
        &mut self,
        scene: &Scene,
        dataset_name: &str,
    ) -> Result<(PipelineHash, Arc<SurfaceMesh>), SceneError> {
        let (node_hash, data) = self.evaluate_dataset(scene, dataset_name)?;
        match data {
            EvaluatedDataset::Surface(surface) => Ok((node_hash, surface)),
            other @ EvaluatedDataset::Grid(_) => {
                let surface_hash = hash_surface_extraction(node_hash);
                let surface = self.get_or_extract_surface(surface_hash, other)?;
                Ok((surface_hash, surface))
            }
        }
    }

    fn get_or_extract_surface(
        &mut self,
        surface_hash: PipelineHash,
        upstream_data: EvaluatedDataset,
    ) -> Result<Arc<SurfaceMesh>, SceneError> {
        if let Some(cached) = self.surface_cache.get(&surface_hash) {
            self.cache_hits += 1;
            return Ok(Arc::clone(cached));
        }

        self.filter_executions += 1;
        let surface = match upstream_data {
            EvaluatedDataset::Grid(grid) => Arc::new(extract_external_surface(&grid)?),
            EvaluatedDataset::Surface(surf) => surf,
        };
        self.surface_cache
            .insert(surface_hash, Arc::clone(&surface));
        Ok(surface)
    }

    /// Validates `scene` against the loaded sources and lazily evaluates all visible
    /// representations and camera bounds.
    ///
    /// Filters in `scene.filters` that are not referenced by any visible representation
    /// in `scene.view.show` (or by `scene.view.camera.fit`) are never executed.
    pub fn evaluate_scene(&mut self, scene: &Scene) -> Result<EvaluatedScene, SceneError> {
        let metadata_map: BTreeMap<String, DatasetMetadata> = self
            .sources
            .iter()
            .map(|(k, v)| (k.clone(), v.metadata.clone()))
            .collect();
        validate_scene_with_metadata(scene, &metadata_map)?;

        let mut representations = Vec::new();
        let mut visible_bounds: Option<BoundingBox> = None;

        for (idx, rep) in scene.view.show.iter().enumerate() {
            if !rep.visible {
                continue;
            }
            let (surface_hash, surface) = self.evaluate_surface(scene, &rep.input)?;
            if let Some(b) = surface.bounds() {
                visible_bounds = Some(match visible_bounds {
                    None => b,
                    Some(acc) => union_bounds(acc, b),
                });
            }

            let color = rep.color.resolve_rgb().unwrap_or([0.827, 0.827, 0.827]);
            let edge_color = rep.edge_color.resolve_rgb().unwrap_or([0.0, 0.0, 0.0]);
            let coloring = resolve_representation_coloring(rep, &surface)?;

            representations.push(EvaluatedRepresentation {
                representation_index: idx,
                input: rep.input.clone(),
                surface_hash,
                surface,
                style: rep.style,
                color,
                edge_color,
                coloring,
                opacity: rep.opacity,
                point_size: rep.point_size,
                colorbar: rep.colorbar.clone(),
            });
        }

        // Resolve camera framing bounds lazily.
        let camera_bounds = if scene.view.camera.position.is_some() {
            visible_bounds
        } else if scene.view.camera.fit == "all" {
            visible_bounds.or_else(|| {
                // Fallback if all representations were hidden or empty: union of loaded sources.
                let mut acc: Option<BoundingBox> = None;
                for name in scene.sources.keys() {
                    if let Some(src) = self.sources.get(name)
                        && let Some(b) = src.grid.bounds()
                    {
                        acc = Some(match acc {
                            None => b,
                            Some(prev) => union_bounds(prev, b),
                        });
                    }
                }
                acc
            })
        } else {
            // Frame the specific dataset named in `camera.fit`.
            let (_, ds) = self.evaluate_dataset(scene, &scene.view.camera.fit)?;
            ds.bounds()
        };

        Ok(EvaluatedScene {
            representations,
            camera_bounds,
        })
    }
}

fn union_bounds(a: BoundingBox, b: BoundingBox) -> BoundingBox {
    BoundingBox {
        min: [
            a.min[0].min(b.min[0]),
            a.min[1].min(b.min[1]),
            a.min[2].min(b.min[2]),
        ],
        max: [
            a.max[0].max(b.max[0]),
            a.max[1].max(b.max[1]),
            a.max[2].max(b.max[2]),
        ],
    }
}

fn resolve_representation_coloring(
    rep: &Representation,
    surface: &SurfaceMesh,
) -> Result<Option<EvaluatedColoring>, SceneError> {
    let Some(raw_ref) = &rep.color_by else {
        return Ok(None);
    };
    let parsed = ParsedArrayRef::parse(raw_ref)?;

    let (association, array) = match parsed.association {
        Some(ArrayAssociation::Point) => {
            let arr = surface.point_data().get(&parsed.name).ok_or_else(|| {
                SceneError::InvalidArrayRef {
                    reference: raw_ref.clone(),
                    reason: format!("missing point array `{}`", parsed.name),
                }
            })?;
            (ArrayAssociation::Point, arr)
        }
        Some(ArrayAssociation::Cell) => {
            let arr = surface.cell_data().get(&parsed.name).ok_or_else(|| {
                SceneError::InvalidArrayRef {
                    reference: raw_ref.clone(),
                    reason: format!("missing cell array `{}`", parsed.name),
                }
            })?;
            (ArrayAssociation::Cell, arr)
        }
        None => {
            // Point data wins if the name exists on both point and cell data (`docs/scene-format.md`).
            if let Some(arr) = surface.point_data().get(&parsed.name) {
                (ArrayAssociation::Point, arr)
            } else if let Some(arr) = surface.cell_data().get(&parsed.name) {
                (ArrayAssociation::Cell, arr)
            } else {
                return Err(SceneError::InvalidArrayRef {
                    reference: raw_ref.clone(),
                    reason: format!("missing array `{}`", parsed.name),
                });
            }
        }
    };

    let component = parsed
        .resolve_component(array.components())
        .ok_or_else(|| SceneError::InvalidArrayRef {
            reference: raw_ref.clone(),
            reason: format!(
                "incompatible component selector for {}-component array `{}`",
                array.components().as_usize(),
                parsed.name
            ),
        })?;

    let data_range = match component {
        ResolvedComponent::Magnitude => array.magnitude_range(),
        ResolvedComponent::Component(idx) => array.component_range(idx),
    }
    .unwrap_or(ValueRange { min: 0.0, max: 0.0 });

    let effective_range = match rep.range {
        RangeSpec::Preset(_) => [data_range.min, data_range.max],
        RangeSpec::Bounds(bounds) => bounds,
    };

    Ok(Some(EvaluatedColoring {
        raw_ref: raw_ref.clone(),
        association,
        array_name: parsed.name,
        component,
        data_range,
        effective_range,
        colormap: rep.colormap,
        log_scale: rep.log_scale,
        center: rep.center,
    }))
}

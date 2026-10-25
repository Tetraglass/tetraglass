//! Vector glyph filter (`M4.3`): oriented and scaled arrows, cones, and spheres on [`TetView`].
//!
//! [`glyphs`] samples a 3-component vector attribute array (`ComponentCount::Vector3`) at
//! either mesh vertices (for point arrays) or source-cell volume centroids (for cell arrays)
//! with configurable stride (`every`), and instantiates oriented 3D `"arrow"`, `"cone"`, or
//! `"sphere"` meshes scaled by vector magnitude or a fixed world-space factor.
//!
//! All point and cell attributes on the input [`TetView`] are replicated onto each generated
//! glyph instance (alongside `"glyph_magnitude"`), and [`SurfaceMesh::triangle_source_cells`]
//! records the originating source cell ID so scalar colormapping, GPU picking, glTF export,
//! and headless Blender Cycles rendering work out of the box.

use std::sync::Arc;

use tetraglass_core::{
    ArrayBuffer, Attributes, ComponentCount, CoreError, DataArray, SurfaceMesh, TetView,
    UnstructuredGrid, try_u32_from_usize,
};

use crate::linearize::{DEFAULT_SUBDIVISIONS, LinearizeParams, linearize};
use crate::surface::FilterError;

/// Default nominal glyph length as a fraction of the dataset bounding-box diagonal (`0.14`).
pub const DEFAULT_GLYPH_LENGTH_FRACTION: f64 = 0.14;

/// Default number of glyph instances processed per parallel chunk (`256`).
pub const DEFAULT_GLYPH_CHUNK_SIZE: usize = 256;

const GLYPH_RADIAL_SEGMENTS: usize = 12;
const SPHERE_LAT_RINGS: usize = 8;
const SPHERE_LON_SEGMENTS: usize = 12;

/// 3D geometric shape for vector glyphs (`M4.3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GlyphShape {
    /// Cylindrical shaft with a conical tip oriented along the vector (default).
    #[default]
    Arrow,
    /// Circular cone with base at the sample point and apex oriented along the vector.
    Cone,
    /// UV sphere centered at the sample point.
    Sphere,
}

/// How each glyph's world-space size is determined (`M4.3`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum GlyphScaleMode {
    /// Scale glyph size proportionally to the local vector magnitude `|v|` (default).
    #[default]
    Magnitude,
    /// Use a uniform base size `s` (`> 0.0`) for all glyphs regardless of vector magnitude.
    Fixed(f64),
}

/// Which attribute table to search for the glyph vector array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GlyphAssociation {
    /// Search `point_data` first, then `cell_data` (default).
    #[default]
    Auto,
    /// Require the vector array in `point_data`.
    Point,
    /// Require the vector array in `cell_data`.
    Cell,
}

/// Parameters for the [`glyphs`] filter (`M4.3`).
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphParams {
    /// Name of the 3-component vector array (`point_data` or `cell_data`).
    pub vectors: String,
    /// Attribute association (`Auto`, `Point`, or `Cell`).
    pub association: GlyphAssociation,
    /// Glyph shape (`Arrow`, `Cone`, or `Sphere`).
    pub shape: GlyphShape,
    /// Whether glyphs scale by `"magnitude"` or a fixed size (`GlyphScaleMode`).
    pub scale_by: GlyphScaleMode,
    /// Optional explicit multiplier (`> 0.0`) applied to the glyph size.
    ///
    /// When `None` and `scale_by == GlyphScaleMode::Magnitude`, glyphs are normalized so the
    /// maximum vector magnitude on the dataset has world-space length
    /// [`DEFAULT_GLYPH_LENGTH_FRACTION`] `* bounds.diagonal()`.
    pub scale_factor: Option<f64>,
    /// Stride (`>= 1`) for subsampling points or source cells (default: `1`).
    pub every: usize,
}

impl GlyphParams {
    /// Creates [`GlyphParams`] for the 3-component vector array `vectors` with default settings
    /// (`shape: Arrow`, `scale_by: Magnitude`, `scale_factor: None`, `every: 1`).
    #[must_use]
    pub fn new(vectors: impl Into<String>) -> Self {
        Self {
            vectors: vectors.into(),
            association: GlyphAssociation::Auto,
            shape: GlyphShape::Arrow,
            scale_by: GlyphScaleMode::Magnitude,
            scale_factor: None,
            every: 1,
        }
    }

    /// Sets the attribute association (`Auto`, `Point`, or `Cell`).
    #[must_use]
    pub const fn with_association(mut self, association: GlyphAssociation) -> Self {
        self.association = association;
        self
    }

    /// Sets the 3D glyph shape (`Arrow`, `Cone`, or `Sphere`).
    #[must_use]
    pub const fn shape(mut self, shape: GlyphShape) -> Self {
        self.shape = shape;
        self
    }

    /// Sets the scaling mode (`Magnitude` or `Fixed(size)`).
    #[must_use]
    pub const fn scale_by(mut self, scale_by: GlyphScaleMode) -> Self {
        self.scale_by = scale_by;
        self
    }

    /// Sets an explicit positive multiplier applied to the glyph size.
    #[must_use]
    pub const fn scale_factor(mut self, factor: f64) -> Self {
        self.scale_factor = Some(factor);
        self
    }

    /// Sets the subsampling stride (`>= 1`).
    #[must_use]
    pub const fn every(mut self, every: usize) -> Self {
        self.every = every;
        self
    }
}

/// Resolved placement, orientation, magnitude, and world-space scale for a single glyph instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphInstance {
    /// World-space anchor position `[x, y, z]` in `f64`.
    pub position: [f64; 3],
    /// Unit orientation vector `[dx, dy, dz]` in `f64`.
    pub direction: [f64; 3],
    /// Euclidean magnitude `|v|` of the vector at this sample site.
    pub magnitude: f64,
    /// World-space length (for `Arrow`/`Cone`) or diameter (for `Sphere`) of this glyph.
    pub scale: f64,
    /// Originating vertex index in `TetView::points` when sampled from a point array, or `None` for cell arrays.
    pub sample_point_index: Option<usize>,
    /// Originating source cell ID in the parent `UnstructuredGrid`.
    pub source_cell_id: u32,
}

/// Computes the oriented and scaled [`GlyphInstance`] placements on `view` for `params`.
pub fn glyph_instances(
    view: &TetView,
    params: &GlyphParams,
) -> Result<Vec<GlyphInstance>, FilterError> {
    validate_glyph_params(params)?;
    let (is_point_assoc, vec_arr) = resolve_glyph_vector_array(view, params)?;
    let raw_sites = collect_sample_sites(view, is_point_assoc, vec_arr, params.every);
    Ok(resolve_instances_from_sites(&raw_sites, view, params))
}

/// Generates an oriented and scaled 3D glyph [`SurfaceMesh`] from `view` using [`DEFAULT_GLYPH_CHUNK_SIZE`].
///
/// # Examples
///
/// ```
/// use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
/// use tetraglass_filters::{GlyphParams, GlyphShape, glyphs, linearize_default};
///
/// let grid = SyntheticGridSpec::new(3, 3, 3, SyntheticCellLayout::Hexahedra).generate()?;
/// let view = linearize_default(&grid)?;
///
/// let params = GlyphParams::new("velocity")
///     .shape(GlyphShape::Arrow)
///     .every(4);
/// let mesh = glyphs(&view, &params)?;
/// assert!(mesh.num_triangles() > 0);
/// assert!(mesh.point_data().get("velocity").is_some());
/// assert!(mesh.point_data().get("glyph_magnitude").is_some());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn glyphs(view: &TetView, params: &GlyphParams) -> Result<SurfaceMesh, FilterError> {
    glyphs_with_chunk_size(view, params, DEFAULT_GLYPH_CHUNK_SIZE)
}

/// Generates an oriented and scaled 3D glyph [`SurfaceMesh`] from `view` using an explicit
/// parallel `chunk_size`.
///
/// Output vertices, triangles, and attributes are bit-for-bit identical across all `chunk_size` values.
pub fn glyphs_with_chunk_size(
    view: &TetView,
    params: &GlyphParams,
    chunk_size: usize,
) -> Result<SurfaceMesh, FilterError> {
    let chunk_size = chunk_size.max(1);
    validate_glyph_params(params)?;
    let (is_point_assoc, vec_arr) = resolve_glyph_vector_array(view, params)?;
    let raw_sites = collect_sample_sites(view, is_point_assoc, vec_arr, params.every);
    let instances = resolve_instances_from_sites(&raw_sites, view, params);

    if instances.is_empty() {
        return Ok(SurfaceMesh::new(
            Vec::<[f64; 3]>::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Attributes::new(0),
            Attributes::new(0),
        )?);
    }

    let template = build_template_mesh(params.shape);
    let verts_per_glyph = template.positions.len();
    let tris_per_glyph = template.triangles.len();
    let faces_per_glyph = template.num_logical_faces;

    let total_verts = instances.len().saturating_mul(verts_per_glyph);
    let _ = try_u32_from_usize(total_verts)?;
    let total_tris = instances.len().saturating_mul(tris_per_glyph);
    let total_faces = instances.len().saturating_mul(faces_per_glyph);

    #[cfg(not(target_arch = "wasm32"))]
    let chunk_meshes: Vec<GlyphChunkGeom> = if instances.len() > chunk_size {
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (chunk_idx, slice) in instances.chunks(chunk_size).enumerate() {
                let base_instance = chunk_idx * chunk_size;
                let tpl = &template;
                handles.push(s.spawn(move || build_glyph_chunk(slice, base_instance, tpl)));
            }
            handles
                .into_iter()
                .map(|h| h.join().expect("glyph chunk worker panicked"))
                .collect()
        })
    } else {
        instances
            .chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, slice)| {
                let base_instance = chunk_idx * chunk_size;
                build_glyph_chunk(slice, base_instance, &template)
            })
            .collect()
    };

    #[cfg(target_arch = "wasm32")]
    let chunk_meshes: Vec<GlyphChunkGeom> = instances
        .chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, slice)| {
            let base_instance = chunk_idx * chunk_size;
            build_glyph_chunk(slice, base_instance, &template)
        })
        .collect();

    let mut points = Vec::with_capacity(total_verts);
    let mut triangles = Vec::with_capacity(total_tris);
    let mut triangle_source_cells = Vec::with_capacity(total_tris);
    let mut face_source_cells = Vec::with_capacity(total_faces);
    let mut vertex_point_indices = Vec::with_capacity(total_verts);
    let mut vertex_cell_indices = Vec::with_capacity(total_verts);
    let mut vertex_magnitudes = Vec::with_capacity(total_verts);

    for chunk in chunk_meshes {
        points.extend(chunk.points);
        triangles.extend(chunk.triangles);
        triangle_source_cells.extend(chunk.triangle_source_cells);
        face_source_cells.extend(chunk.face_source_cells);
        vertex_point_indices.extend(chunk.vertex_point_indices);
        vertex_cell_indices.extend(chunk.vertex_cell_indices);
        vertex_magnitudes.extend(chunk.vertex_magnitudes);
    }

    let mut point_data = Attributes::new(total_verts);
    if is_point_assoc {
        for (name, arr) in view.point_data().iter() {
            let gathered = gather_array_by_indices(arr, &vertex_point_indices)?;
            point_data.insert(name.to_string(), gathered)?;
        }
    } else {
        // When glyphs are placed at source-cell centroids, interpolate point arrays to each
        // cell centroid and replicate onto the glyph vertices.
        let cell_centroid_point_data =
            compute_cell_centroid_point_data(view, &instances, verts_per_glyph)?;
        for (name, arr) in cell_centroid_point_data {
            point_data.insert(name, arr)?;
        }
    }
    // Also promote cell arrays into `point_data` if not already present so vertex coloring works seamlessly.
    for (name, arr) in view.cell_data().iter() {
        if point_data.get(name).is_none() {
            let gathered = gather_array_by_u32_indices(arr, &vertex_cell_indices)?;
            point_data.insert(name.to_string(), gathered)?;
        }
    }
    if point_data.get("glyph_magnitude").is_none() {
        point_data.insert(
            "glyph_magnitude",
            DataArray::from_f64(ComponentCount::Scalar, vertex_magnitudes)?,
        )?;
    }

    let mut cell_data = Attributes::new(total_tris);
    for (name, arr) in view.cell_data().iter() {
        let gathered = gather_array_by_u32_indices(arr, &triangle_source_cells)?;
        cell_data.insert(name.to_string(), gathered)?;
    }

    Ok(SurfaceMesh::new(
        points,
        triangles,
        triangle_source_cells,
        face_source_cells,
        point_data,
        cell_data,
    )?)
}

/// Linearizes `grid` with [`DEFAULT_SUBDIVISIONS`] and runs [`glyphs`].
pub fn glyphs_grid(
    grid: &UnstructuredGrid,
    params: &GlyphParams,
) -> Result<SurfaceMesh, FilterError> {
    let view = linearize(grid, &LinearizeParams::new(DEFAULT_SUBDIVISIONS))?;
    glyphs(&view, params)
}

fn validate_glyph_params(params: &GlyphParams) -> Result<(), FilterError> {
    if params.every == 0 {
        return Err(FilterError::InvalidGlyphParams {
            reason: "glyph `every` stride must be >= 1",
        });
    }
    if let GlyphScaleMode::Fixed(s) = params.scale_by
        && (!s.is_finite() || s <= 0.0)
    {
        return Err(FilterError::InvalidGlyphParams {
            reason: "glyph fixed `scale_by` value must be finite and > 0.0",
        });
    }
    if let Some(factor) = params.scale_factor
        && (!factor.is_finite() || factor <= 0.0)
    {
        return Err(FilterError::InvalidGlyphParams {
            reason: "glyph `scale_factor` must be finite and > 0.0",
        });
    }
    Ok(())
}

fn resolve_glyph_vector_array<'a>(
    view: &'a TetView,
    params: &GlyphParams,
) -> Result<(bool, &'a DataArray), FilterError> {
    let (is_point, arr) = match params.association {
        GlyphAssociation::Point => {
            let arr = view.point_data().get(&params.vectors).ok_or_else(|| {
                FilterError::MissingPointArray {
                    name: params.vectors.clone(),
                }
            })?;
            (true, arr)
        }
        GlyphAssociation::Cell => {
            let arr = view.cell_data().get(&params.vectors).ok_or_else(|| {
                FilterError::MissingCellArray {
                    name: params.vectors.clone(),
                }
            })?;
            (false, arr)
        }
        GlyphAssociation::Auto => {
            if let Some(arr) = view.point_data().get(&params.vectors) {
                (true, arr)
            } else if let Some(arr) = view.cell_data().get(&params.vectors) {
                (false, arr)
            } else {
                return Err(FilterError::MissingAttributeArray {
                    name: params.vectors.clone(),
                });
            }
        }
    };

    if arr.components() != ComponentCount::Vector3 {
        return Err(FilterError::NonVectorArray {
            array: params.vectors.clone(),
            num_components: arr.components().as_usize(),
        });
    }
    Ok((is_point, arr))
}

#[derive(Debug, Clone)]
struct RawSampleSite {
    position: [f64; 3],
    vector: [f64; 3],
    sample_point_index: Option<usize>,
    source_cell_id: u32,
}

fn collect_sample_sites(
    view: &TetView,
    is_point_assoc: bool,
    vec_arr: &DataArray,
    every: usize,
) -> Vec<RawSampleSite> {
    let points = view.points();
    let tets = view.tets();
    let tet_source_cells = view.tet_source_cells();

    if is_point_assoc {
        // Map each vertex to the lowest source_cell_id of any incident tetrahedron.
        let mut point_source_cell = vec![0u32; points.len()];
        let mut seen = vec![false; points.len()];
        for (tet, &src_cell) in tets.iter().zip(tet_source_cells.iter()) {
            for &v in tet {
                let vi = v as usize;
                if !seen[vi] || src_cell < point_source_cell[vi] {
                    point_source_cell[vi] = src_cell;
                    seen[vi] = true;
                }
            }
        }

        // Deduplicate vertices that share identical world-space coordinates (from linearized interfaces)
        // in deterministic vertex index order before applying the `every` stride.
        let mut unique_point_indices = Vec::with_capacity(points.len());
        let mut visited_coords: std::collections::HashSet<[u64; 3]> =
            std::collections::HashSet::with_capacity(points.len());
        for (p_idx, &pt) in points.iter().enumerate() {
            let key = [
                canonical_f64_bits(pt[0]),
                canonical_f64_bits(pt[1]),
                canonical_f64_bits(pt[2]),
            ];
            if visited_coords.insert(key) {
                unique_point_indices.push(p_idx);
            }
        }

        unique_point_indices
            .into_iter()
            .step_by(every)
            .map(|p_idx| RawSampleSite {
                position: points[p_idx],
                vector: read_vec3(vec_arr, p_idx),
                sample_point_index: Some(p_idx),
                source_cell_id: point_source_cell[p_idx],
            })
            .collect()
    } else {
        // Group tetrahedra by original `source_cell_id` in deterministic ascending order of `source_cell_id`
        // and compute each source cell's volume-weighted centroid.
        let mut cell_accum: std::collections::BTreeMap<u32, ([f64; 3], f64)> =
            std::collections::BTreeMap::new();
        for (tet_idx, (tet, &src_cell)) in tets.iter().zip(tet_source_cells.iter()).enumerate() {
            let p0 = points[tet[0] as usize];
            let p1 = points[tet[1] as usize];
            let p2 = points[tet[2] as usize];
            let p3 = points[tet[3] as usize];
            let center = [
                0.25 * (p0[0] + p1[0] + p2[0] + p3[0]),
                0.25 * (p0[1] + p1[1] + p2[1] + p3[1]),
                0.25 * (p0[2] + p1[2] + p2[2] + p3[2]),
            ];
            let vol = view.tet_volume(tet_idx).unwrap_or(0.0).max(1e-18);
            let entry = cell_accum.entry(src_cell).or_insert(([0.0, 0.0, 0.0], 0.0));
            entry.0[0] += center[0] * vol;
            entry.0[1] += center[1] * vol;
            entry.0[2] += center[2] * vol;
            entry.1 += vol;
        }

        cell_accum
            .into_iter()
            .step_by(every)
            .filter_map(|(src_cell, (weighted_sum, total_vol))| {
                let cell_idx = src_cell as usize;
                if cell_idx >= vec_arr.num_tuples() {
                    return None;
                }
                let inv_vol = 1.0 / total_vol.max(1e-18);
                Some(RawSampleSite {
                    position: [
                        weighted_sum[0] * inv_vol,
                        weighted_sum[1] * inv_vol,
                        weighted_sum[2] * inv_vol,
                    ],
                    vector: read_vec3(vec_arr, cell_idx),
                    sample_point_index: None,
                    source_cell_id: src_cell,
                })
            })
            .collect()
    }
}

fn resolve_instances_from_sites(
    sites: &[RawSampleSite],
    view: &TetView,
    params: &GlyphParams,
) -> Vec<GlyphInstance> {
    let diag = view.bounds().map_or(1.0, |b| b.diagonal().max(1e-6));
    let default_len = DEFAULT_GLYPH_LENGTH_FRACTION * diag;

    let mut max_mag = 0.0_f64;
    for site in sites {
        let m = vec3_norm(site.vector);
        if m.is_finite() && m > max_mag {
            max_mag = m;
        }
    }

    let mut instances = Vec::with_capacity(sites.len());
    for site in sites {
        if !site.position.iter().all(|c| c.is_finite())
            || !site.vector.iter().all(|c| c.is_finite())
        {
            continue;
        }
        let mag = vec3_norm(site.vector);
        let direction = if mag > 1e-14 {
            [
                site.vector[0] / mag,
                site.vector[1] / mag,
                site.vector[2] / mag,
            ]
        } else {
            [1.0, 0.0, 0.0]
        };

        let scale = match params.scale_by {
            GlyphScaleMode::Magnitude => match params.scale_factor {
                Some(factor) => mag * factor,
                None => {
                    if max_mag > 1e-14 {
                        mag * (default_len / max_mag)
                    } else {
                        0.0
                    }
                }
            },
            GlyphScaleMode::Fixed(base) => base * params.scale_factor.unwrap_or(1.0),
        };

        // Skip zero-scale glyphs (such as zero-magnitude vectors under `Magnitude` scaling).
        if !scale.is_finite() || scale <= 1e-14 {
            continue;
        }

        instances.push(GlyphInstance {
            position: site.position,
            direction,
            magnitude: mag,
            scale,
            sample_point_index: site.sample_point_index,
            source_cell_id: site.source_cell_id,
        });
    }
    instances
}

#[derive(Debug, Clone)]
struct TemplateMesh {
    positions: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    num_logical_faces: usize,
}

fn build_template_mesh(shape: GlyphShape) -> TemplateMesh {
    match shape {
        GlyphShape::Arrow => build_arrow_template(GLYPH_RADIAL_SEGMENTS),
        GlyphShape::Cone => build_cone_template(GLYPH_RADIAL_SEGMENTS),
        GlyphShape::Sphere => build_sphere_template(SPHERE_LAT_RINGS, SPHERE_LON_SEGMENTS),
    }
}

fn build_arrow_template(segments: usize) -> TemplateMesh {
    let n = segments.max(3);
    let r_shaft = 0.065;
    let r_head = 0.16;
    let z_head_base = 0.65;
    let z_tip = 1.0;

    let mut positions = Vec::with_capacity(2 + 3 * n);
    let mut triangles = Vec::with_capacity(5 * n);

    // 0: shaft base center [0, 0, 0]
    positions.push([0.0, 0.0, 0.0]);
    // 1..=n: shaft base ring at z = 0
    for i in 0..n {
        let theta = (i as f64) * std::f64::consts::TAU / (n as f64);
        positions.push([r_shaft * theta.cos(), r_shaft * theta.sin(), 0.0]);
    }
    // (1 + n)..=(2*n): shaft top ring at z = z_head_base
    let shaft_top_start = 1 + n;
    for i in 0..n {
        let theta = (i as f64) * std::f64::consts::TAU / (n as f64);
        positions.push([r_shaft * theta.cos(), r_shaft * theta.sin(), z_head_base]);
    }
    // (1 + 2*n)..=(3*n): head outer base ring at z = z_head_base
    let head_base_start = 1 + 2 * n;
    for i in 0..n {
        let theta = (i as f64) * std::f64::consts::TAU / (n as f64);
        positions.push([r_head * theta.cos(), r_head * theta.sin(), z_head_base]);
    }
    // tip index = 1 + 3*n
    let tip_idx = (1 + 3 * n) as u32;
    positions.push([0.0, 0.0, z_tip]);

    for i in 0..n {
        let next = (i + 1) % n;
        let b0 = (1 + i) as u32;
        let b1 = (1 + next) as u32;
        let t0 = (shaft_top_start + i) as u32;
        let t1 = (shaft_top_start + next) as u32;
        let h0 = (head_base_start + i) as u32;
        let h1 = (head_base_start + next) as u32;

        // Shaft bottom cap (outward normal -Z)
        triangles.push([0, b1, b0]);
        // Shaft cylinder quad (2 triangles, outward radial normal)
        triangles.push([b0, b1, t1]);
        triangles.push([b0, t1, t0]);
        // Head base annulus quad between shaft top and head rim (outward normal -Z)
        triangles.push([t0, h1, h0]);
        triangles.push([t0, t1, h1]);
        // Cone side triangle (outward normal)
        triangles.push([h0, h1, tip_idx]);
    }

    TemplateMesh {
        positions,
        triangles,
        num_logical_faces: 4 * n,
    }
}

fn build_cone_template(segments: usize) -> TemplateMesh {
    let n = segments.max(3);
    let r_base = 0.20;
    let mut positions = Vec::with_capacity(2 + n);
    let mut triangles = Vec::with_capacity(2 * n);

    // 0: base center [0, 0, 0]
    positions.push([0.0, 0.0, 0.0]);
    for i in 0..n {
        let theta = (i as f64) * std::f64::consts::TAU / (n as f64);
        positions.push([r_base * theta.cos(), r_base * theta.sin(), 0.0]);
    }
    let tip_idx = (1 + n) as u32;
    positions.push([0.0, 0.0, 1.0]);

    for i in 0..n {
        let next = (i + 1) % n;
        let b0 = (1 + i) as u32;
        let b1 = (1 + next) as u32;
        // Base cap (-Z)
        triangles.push([0, b1, b0]);
        // Cone side
        triangles.push([b0, b1, tip_idx]);
    }

    TemplateMesh {
        positions,
        triangles,
        num_logical_faces: 2 * n,
    }
}

fn build_sphere_template(lat_rings: usize, lon_segments: usize) -> TemplateMesh {
    let n_lat = lat_rings.max(3);
    let n_lon = lon_segments.max(3);
    let radius = 0.5;

    let mut positions = Vec::with_capacity(2 + (n_lat - 1) * n_lon);
    let mut triangles = Vec::with_capacity(2 * n_lon + 2 * (n_lat - 2) * n_lon);

    // South pole (0) at z = -0.5
    positions.push([0.0, 0.0, -radius]);
    for lat in 1..n_lat {
        let phi =
            -std::f64::consts::FRAC_PI_2 + (lat as f64) * std::f64::consts::PI / (n_lat as f64);
        let z = radius * phi.sin();
        let r_ring = radius * phi.cos();
        for lon in 0..n_lon {
            let theta = (lon as f64) * std::f64::consts::TAU / (n_lon as f64);
            positions.push([r_ring * theta.cos(), r_ring * theta.sin(), z]);
        }
    }
    let north_pole_idx = positions.len() as u32;
    positions.push([0.0, 0.0, radius]);

    // South polar fan
    for lon in 0..n_lon {
        let next = (lon + 1) % n_lon;
        triangles.push([0, (1 + next) as u32, (1 + lon) as u32]);
    }
    // Middle latitude quads
    for lat in 0..(n_lat - 2) {
        let row0 = 1 + lat * n_lon;
        let row1 = row0 + n_lon;
        for lon in 0..n_lon {
            let next = (lon + 1) % n_lon;
            let v00 = (row0 + lon) as u32;
            let v10 = (row0 + next) as u32;
            let v01 = (row1 + lon) as u32;
            let v11 = (row1 + next) as u32;
            triangles.push([v00, v10, v11]);
            triangles.push([v00, v11, v01]);
        }
    }
    // North polar fan
    let last_row = 1 + (n_lat - 2) * n_lon;
    for lon in 0..n_lon {
        let next = (lon + 1) % n_lon;
        triangles.push([
            (last_row + lon) as u32,
            (last_row + next) as u32,
            north_pole_idx,
        ]);
    }

    TemplateMesh {
        positions,
        triangles,
        num_logical_faces: 2 * n_lon + (n_lat - 2) * n_lon,
    }
}

struct GlyphChunkGeom {
    points: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    triangle_source_cells: Vec<u32>,
    face_source_cells: Vec<u32>,
    vertex_point_indices: Vec<usize>,
    vertex_cell_indices: Vec<u32>,
    vertex_magnitudes: Vec<f64>,
}

fn build_glyph_chunk(
    slice: &[GlyphInstance],
    base_instance: usize,
    template: &TemplateMesh,
) -> GlyphChunkGeom {
    let v_per = template.positions.len();
    let t_per = template.triangles.len();
    let f_per = template.num_logical_faces;

    let mut points = Vec::with_capacity(slice.len() * v_per);
    let mut triangles = Vec::with_capacity(slice.len() * t_per);
    let mut triangle_source_cells = Vec::with_capacity(slice.len() * t_per);
    let mut face_source_cells = Vec::with_capacity(slice.len() * f_per);
    let mut vertex_point_indices = Vec::with_capacity(slice.len() * v_per);
    let mut vertex_cell_indices = Vec::with_capacity(slice.len() * v_per);
    let mut vertex_magnitudes = Vec::with_capacity(slice.len() * v_per);

    for (local_i, inst) in slice.iter().enumerate() {
        let global_inst = base_instance + local_i;
        let base_v = (global_inst * v_per) as u32;

        let d = inst.direction;
        let u = orthogonal_unit_vector(d);
        let v = cross3(d, u);
        let s = inst.scale;
        let p0 = inst.position;

        for &[lx, ly, lz] in &template.positions {
            let wx = p0[0] + s * (lx * u[0] + ly * v[0] + lz * d[0]);
            let wy = p0[1] + s * (lx * u[1] + ly * v[1] + lz * d[1]);
            let wz = p0[2] + s * (lx * u[2] + ly * v[2] + lz * d[2]);
            points.push([wx, wy, wz]);
            vertex_point_indices.push(inst.sample_point_index.unwrap_or(0));
            vertex_cell_indices.push(inst.source_cell_id);
            vertex_magnitudes.push(inst.magnitude);
        }

        for &[i0, i1, i2] in &template.triangles {
            triangles.push([base_v + i0, base_v + i1, base_v + i2]);
            triangle_source_cells.push(inst.source_cell_id);
        }
        face_source_cells.extend(std::iter::repeat_n(inst.source_cell_id, f_per));
    }

    GlyphChunkGeom {
        points,
        triangles,
        triangle_source_cells,
        face_source_cells,
        vertex_point_indices,
        vertex_cell_indices,
        vertex_magnitudes,
    }
}

fn compute_cell_centroid_point_data(
    view: &TetView,
    instances: &[GlyphInstance],
    verts_per_glyph: usize,
) -> Result<Vec<(String, DataArray)>, CoreError> {
    if view.point_data().is_empty() {
        return Ok(Vec::new());
    }

    // Build map from `source_cell_id` to unique vertex indices in `view`.
    let mut cell_verts: std::collections::BTreeMap<u32, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (tet, &src_cell) in view.tets().iter().zip(view.tet_source_cells().iter()) {
        let entry = cell_verts.entry(src_cell).or_default();
        for &v in tet {
            entry.push(v as usize);
        }
    }
    for verts in cell_verts.values_mut() {
        verts.sort_unstable();
        verts.dedup();
    }

    let total_verts = instances.len() * verts_per_glyph;
    let mut out = Vec::with_capacity(view.point_data().len());
    for (name, arr) in view.point_data().iter() {
        let ncomp = arr.components().as_usize();
        let mut values = Vec::with_capacity(total_verts * ncomp);
        let mut avg = vec![0.0_f64; ncomp];
        for inst in instances {
            avg.fill(0.0);
            if let Some(verts) = cell_verts.get(&inst.source_cell_id)
                && !verts.is_empty()
            {
                let inv = 1.0 / (verts.len() as f64);
                for &vi in verts {
                    for (c, slot) in avg.iter_mut().enumerate() {
                        *slot += read_component_f64(arr, vi, c) * inv;
                    }
                }
            }
            for _ in 0..verts_per_glyph {
                values.extend_from_slice(&avg);
            }
        }
        out.push((
            name.to_string(),
            DataArray::from_f64(arr.components(), values)?,
        ));
    }
    Ok(out)
}

fn gather_array_by_indices(array: &DataArray, indices: &[usize]) -> Result<DataArray, CoreError> {
    let ncomp = array.components().as_usize();
    let out_len = indices.len().saturating_mul(ncomp);
    let buffer = match array.buffer() {
        ArrayBuffer::F32(src) => {
            let mut out = Vec::with_capacity(out_len);
            for &idx in indices {
                let start = idx * ncomp;
                out.extend_from_slice(&src[start..start + ncomp]);
            }
            ArrayBuffer::F32(Arc::from(out))
        }
        ArrayBuffer::F64(src) => {
            let mut out = Vec::with_capacity(out_len);
            for &idx in indices {
                let start = idx * ncomp;
                out.extend_from_slice(&src[start..start + ncomp]);
            }
            ArrayBuffer::F64(Arc::from(out))
        }
        ArrayBuffer::I32(src) => {
            let mut out = Vec::with_capacity(out_len);
            for &idx in indices {
                let start = idx * ncomp;
                out.extend_from_slice(&src[start..start + ncomp]);
            }
            ArrayBuffer::I32(Arc::from(out))
        }
        ArrayBuffer::I64(src) => {
            let mut out = Vec::with_capacity(out_len);
            for &idx in indices {
                let start = idx * ncomp;
                out.extend_from_slice(&src[start..start + ncomp]);
            }
            ArrayBuffer::I64(Arc::from(out))
        }
        ArrayBuffer::U8(src) => {
            let mut out = Vec::with_capacity(out_len);
            for &idx in indices {
                let start = idx * ncomp;
                out.extend_from_slice(&src[start..start + ncomp]);
            }
            ArrayBuffer::U8(Arc::from(out))
        }
    };
    DataArray::new(array.components(), buffer)
}

fn gather_array_by_u32_indices(array: &DataArray, indices: &[u32]) -> Result<DataArray, CoreError> {
    let usize_indices: Vec<usize> = indices.iter().map(|&i| i as usize).collect();
    gather_array_by_indices(array, &usize_indices)
}

#[inline]
fn read_vec3(array: &DataArray, tuple_idx: usize) -> [f64; 3] {
    [
        read_component_f64(array, tuple_idx, 0),
        read_component_f64(array, tuple_idx, 1),
        read_component_f64(array, tuple_idx, 2),
    ]
}

#[inline]
fn read_component_f64(array: &DataArray, tuple_idx: usize, comp: usize) -> f64 {
    let idx = tuple_idx * array.components().as_usize() + comp;
    match array.buffer() {
        ArrayBuffer::F32(b) => f64::from(b[idx]),
        ArrayBuffer::F64(b) => b[idx],
        ArrayBuffer::I32(b) => f64::from(b[idx]),
        ArrayBuffer::I64(b) => b[idx] as f64,
        ArrayBuffer::U8(b) => f64::from(b[idx]),
    }
}

#[inline]
fn canonical_f64_bits(v: f64) -> u64 {
    if v == 0.0 { 0 } else { v.to_bits() }
}

#[inline]
fn vec3_norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn orthogonal_unit_vector(d: [f64; 3]) -> [f64; 3] {
    let axis = if d[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let c = cross3(d, axis);
    let n = vec3_norm(c);
    if n > 1e-14 {
        [c[0] / n, c[1] / n, c[2] / n]
    } else {
        [0.0, 0.0, 1.0]
    }
}

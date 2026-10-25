//! Streamline integration filter (`M4.2a`) over linearized tetrahedral views (`D-005`).
//!
//! Integrates 3D vector fields (`point_data` or `cell_data`) from point, line, or plane
//! [`StreamlineSeeds`] using either fixed-step 4th-order Runge-Kutta ([`StreamlineIntegrator::Rk4`])
//! or adaptive Dormand-Prince 5(4) ([`StreamlineIntegrator::Rk45`]), accelerated by
//! [`TetLocator::locate_with_hint`].
//!
//! All point and cell [`DataArray`]s on the input [`TetView`] are interpolated onto the output
//! [`PolylineSet`], alongside `"arc_length"`, `"integration_time"`, and per-polyline `"seed_id"`.

use tetraglass_core::{
    Attributes, ComponentCount, DataArray, PolylineSet, TetView, UnstructuredGrid,
};

use crate::linearize::linearize_default;
use crate::locator::{TetLocation, TetLocator};
use crate::surface::FilterError;

/// Default geometric step length as a fraction of the dataset bounding-box diagonal (`1%`).
pub const DEFAULT_STREAMLINE_STEP_FRACTION: f64 = 0.01;

/// Default maximum streamline arc length per direction as a multiple of the dataset bounding-box diagonal (`4x`).
pub const DEFAULT_STREAMLINE_MAX_LENGTH_FACTOR: f64 = 4.0;

/// Default maximum number of integration steps per direction.
pub const DEFAULT_STREAMLINE_MAX_STEPS: usize = 2000;

/// Default stagnation speed threshold $\|\mathbf{v}\|_2$ below which integration stops.
pub const DEFAULT_STREAMLINE_TERMINAL_SPEED: f64 = 1e-12;

/// Default local truncation error tolerance (in world length units) for adaptive [`StreamlineIntegrator::Rk45`].
pub const DEFAULT_STREAMLINE_ERROR_TOLERANCE: f64 = 1e-6;

/// Default number of seeds processed per parallel worker chunk on native targets.
pub const DEFAULT_STREAMLINE_CHUNK_SIZE: usize = 16;

/// Seed point generator for [`StreamlineParams`].
#[derive(Debug, Clone, PartialEq)]
pub enum StreamlineSeeds {
    /// A single 3D seed point `[x, y, z]`.
    Point([f64; 3]),
    /// An explicit list of 3D seed points.
    Points(Vec<[f64; 3]>),
    /// `count` evenly spaced seed points along the line segment from `from` to `to`.
    Line {
        /// Starting endpoint `[x, y, z]`.
        from: [f64; 3],
        /// Ending endpoint `[x, y, z]`.
        to: [f64; 3],
        /// Number of seed points (`>= 1`).
        count: usize,
    },
    /// A regular `count[0] × count[1]` grid of seed points centered at `origin` on the plane
    /// orthogonal to `normal` with total dimensions `extent = [width, height]`.
    Plane {
        /// Center point `[x, y, z]` of the seed plane.
        origin: [f64; 3],
        /// Plane normal vector `[nx, ny, nz]` (must be finite and non-zero).
        normal: [f64; 3],
        /// Total planar extents `[width, height]` (`>= 0.0`) along the two orthonormal plane axes.
        extent: [f64; 2],
        /// Grid sample counts `[nu, nv]` (`>= 1` along each axis).
        count: [usize; 2],
    },
}

impl StreamlineSeeds {
    /// Creates a single-point seed specification.
    #[must_use]
    pub const fn point(point: [f64; 3]) -> Self {
        Self::Point(point)
    }

    /// Creates an explicit multi-point seed specification.
    #[must_use]
    pub const fn points(points: Vec<[f64; 3]>) -> Self {
        Self::Points(points)
    }

    /// Creates a line-segment seed specification with `count` evenly spaced points.
    #[must_use]
    pub const fn line(from: [f64; 3], to: [f64; 3], count: usize) -> Self {
        Self::Line { from, to, count }
    }

    /// Creates a planar grid seed specification centered at `origin`.
    #[must_use]
    pub const fn plane(
        origin: [f64; 3],
        normal: [f64; 3],
        extent: [f64; 2],
        count: [usize; 2],
    ) -> Self {
        Self::Plane {
            origin,
            normal,
            extent,
            count,
        }
    }

    /// Validates this seed specification and generates the ordered list of 3D seed points.
    pub fn generate_points(&self) -> Result<Vec<[f64; 3]>, FilterError> {
        match self {
            Self::Point(p) => {
                if !is_finite_vec3(*p) {
                    return Err(FilterError::InvalidStreamlineSeeds {
                        reason: "seed point coordinates must be finite",
                    });
                }
                Ok(vec![*p])
            }
            Self::Points(pts) => {
                if pts.is_empty() {
                    return Err(FilterError::InvalidStreamlineSeeds {
                        reason: "seed points list must not be empty",
                    });
                }
                for p in pts {
                    if !is_finite_vec3(*p) {
                        return Err(FilterError::InvalidStreamlineSeeds {
                            reason: "seed point coordinates must be finite",
                        });
                    }
                }
                Ok(pts.clone())
            }
            Self::Line { from, to, count } => {
                if !is_finite_vec3(*from) || !is_finite_vec3(*to) {
                    return Err(FilterError::InvalidStreamlineSeeds {
                        reason: "line seed endpoints must be finite",
                    });
                }
                if *count == 0 {
                    return Err(FilterError::InvalidStreamlineSeeds {
                        reason: "line seed count must be >= 1",
                    });
                }
                if *count == 1 {
                    return Ok(vec![[
                        0.5 * (from[0] + to[0]),
                        0.5 * (from[1] + to[1]),
                        0.5 * (from[2] + to[2]),
                    ]]);
                }
                let mut out = Vec::with_capacity(*count);
                let denom = (*count - 1) as f64;
                for i in 0..*count {
                    let t = (i as f64) / denom;
                    out.push([
                        from[0] + t * (to[0] - from[0]),
                        from[1] + t * (to[1] - from[1]),
                        from[2] + t * (to[2] - from[2]),
                    ]);
                }
                Ok(out)
            }
            Self::Plane {
                origin,
                normal,
                extent,
                count,
            } => {
                if !is_finite_vec3(*origin) {
                    return Err(FilterError::InvalidStreamlineSeeds {
                        reason: "plane seed origin must be finite",
                    });
                }
                if !is_finite_vec3(*normal) {
                    return Err(FilterError::InvalidPlaneNormal { normal: *normal });
                }
                let n_len = dot3(*normal, *normal).sqrt();
                if !n_len.is_finite() || n_len <= 1e-15 {
                    return Err(FilterError::InvalidPlaneNormal { normal: *normal });
                }
                if !extent[0].is_finite()
                    || !extent[1].is_finite()
                    || extent[0] < 0.0
                    || extent[1] < 0.0
                {
                    return Err(FilterError::InvalidStreamlineSeeds {
                        reason: "plane seed extent must be finite and non-negative",
                    });
                }
                if count[0] == 0 || count[1] == 0 {
                    return Err(FilterError::InvalidStreamlineSeeds {
                        reason: "plane seed counts must both be >= 1",
                    });
                }

                let unit_n = [normal[0] / n_len, normal[1] / n_len, normal[2] / n_len];
                let (u_axis, v_axis) = orthonormal_plane_basis(unit_n);
                let total = count[0].saturating_mul(count[1]);
                let mut out = Vec::with_capacity(total);

                for j in 0..count[1] {
                    let v_frac = if count[1] == 1 {
                        0.0
                    } else {
                        (j as f64) / ((count[1] - 1) as f64) - 0.5
                    };
                    let dv = v_frac * extent[1];
                    for i in 0..count[0] {
                        let u_frac = if count[0] == 1 {
                            0.0
                        } else {
                            (i as f64) / ((count[0] - 1) as f64) - 0.5
                        };
                        let du = u_frac * extent[0];
                        out.push([
                            origin[0] + du * u_axis[0] + dv * v_axis[0],
                            origin[1] + du * u_axis[1] + dv * v_axis[1],
                            origin[2] + du * u_axis[2] + dv * v_axis[2],
                        ]);
                    }
                }
                Ok(out)
            }
        }
    }
}

/// Integration direction along the vector field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StreamlineDirection {
    /// Integrate forward along $+\mathbf{v}(\mathbf{x})$.
    #[default]
    Forward,
    /// Integrate backward along $-\mathbf{v}(\mathbf{x})$.
    Backward,
    /// Integrate both backward and forward from each seed and join into a single polyline.
    Both,
}

/// ODE integration scheme for streamline tracing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StreamlineIntegrator {
    /// Fixed-step classical 4th-order Runge-Kutta (`RK4`).
    Rk4,
    /// Adaptive-step Dormand-Prince 5(4) embedded Runge-Kutta (`RK45`).
    #[default]
    Rk45,
}

/// Parameters for the [`streamlines`] filter.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamlineParams {
    /// Name of the 3-component vector array (`point_data` or `cell_data`, preferring `point_data`).
    pub vectors: String,
    /// Seed points generator (`Point`, `Points`, `Line`, or `Plane`).
    pub seeds: StreamlineSeeds,
    /// Integration direction (`Forward`, `Backward`, or `Both`).
    pub direction: StreamlineDirection,
    /// ODE integrator (`Rk4` or adaptive `Rk45`).
    pub integrator: StreamlineIntegrator,
    /// Maximum arc length integrated per direction (`None` defaults to `4.0 * bounds.diagonal()`).
    pub max_length: Option<f64>,
    /// Initial geometric step length (`None` defaults to `0.01 * bounds.diagonal()`).
    pub step_size: Option<f64>,
    /// Minimum adaptive step length for `Rk45` (`None` defaults to `0.05 * step_size`).
    pub min_step_size: Option<f64>,
    /// Maximum adaptive step length for `Rk45` (`None` defaults to `5.0 * step_size`).
    pub max_step_size: Option<f64>,
    /// Absolute local truncation error tolerance (in world length units) for `Rk45`.
    pub error_tolerance: f64,
    /// Maximum number of integration steps per direction.
    pub max_steps: usize,
    /// Stagnation speed threshold $\|\mathbf{v}\|_2$ below which integration terminates.
    pub terminal_speed: f64,
}

impl StreamlineParams {
    /// Creates [`StreamlineParams`] for `vectors` and `seeds` with default adaptive `Rk45`
    /// forward integration.
    #[must_use]
    pub fn new(vectors: impl Into<String>, seeds: StreamlineSeeds) -> Self {
        Self {
            vectors: vectors.into(),
            seeds,
            direction: StreamlineDirection::Forward,
            integrator: StreamlineIntegrator::Rk45,
            max_length: None,
            step_size: None,
            min_step_size: None,
            max_step_size: None,
            error_tolerance: DEFAULT_STREAMLINE_ERROR_TOLERANCE,
            max_steps: DEFAULT_STREAMLINE_MAX_STEPS,
            terminal_speed: DEFAULT_STREAMLINE_TERMINAL_SPEED,
        }
    }

    /// Sets the integration direction (`Forward`, `Backward`, or `Both`).
    #[must_use]
    pub const fn direction(mut self, direction: StreamlineDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Sets the ODE integrator (`Rk4` or `Rk45`).
    #[must_use]
    pub const fn integrator(mut self, integrator: StreamlineIntegrator) -> Self {
        self.integrator = integrator;
        self
    }

    /// Sets the maximum arc length integrated per direction (`> 0.0`).
    #[must_use]
    pub const fn max_length(mut self, max_length: f64) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Sets the initial geometric step length (`> 0.0`).
    #[must_use]
    pub const fn step_size(mut self, step_size: f64) -> Self {
        self.step_size = Some(step_size);
        self
    }

    /// Sets the minimum adaptive geometric step length (`> 0.0`).
    #[must_use]
    pub const fn min_step_size(mut self, min_step_size: f64) -> Self {
        self.min_step_size = Some(min_step_size);
        self
    }

    /// Sets the maximum adaptive geometric step length (`> 0.0`).
    #[must_use]
    pub const fn max_step_size(mut self, max_step_size: f64) -> Self {
        self.max_step_size = Some(max_step_size);
        self
    }

    /// Sets the local truncation error tolerance (`> 0.0`) for adaptive `Rk45`.
    #[must_use]
    pub const fn error_tolerance(mut self, error_tolerance: f64) -> Self {
        self.error_tolerance = error_tolerance;
        self
    }

    /// Sets the maximum number of integration steps per direction (`>= 1`).
    #[must_use]
    pub const fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Sets the stagnation speed threshold (`>= 0.0`).
    #[must_use]
    pub const fn terminal_speed(mut self, terminal_speed: f64) -> Self {
        self.terminal_speed = terminal_speed;
        self
    }
}

#[derive(Debug, Clone, Copy)]
enum VectorFieldSource<'a> {
    Point(&'a DataArray),
    Cell(&'a DataArray),
}

impl VectorFieldSource<'_> {
    #[inline]
    fn sample(&self, view: &TetView, loc: &TetLocation) -> Option<[f64; 3]> {
        match self {
            Self::Point(arr) => loc.interpolate_point_vector3(view, arr),
            Self::Cell(arr) => loc.sample_cell_vector3(arr),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct IntegratedVertex {
    position: [f64; 3],
    location: TetLocation,
    arc_length: f64,
    integration_time: f64,
}

#[derive(Debug, Clone, Copy)]
struct ResolvedIntegrationConfig {
    max_length: f64,
    step_size: f64,
    min_step_size: f64,
    max_step_size: f64,
    error_tolerance: f64,
    max_steps: usize,
    terminal_speed: f64,
}

/// Integrates streamlines through `view` according to `params`, building a [`TetLocator`] on demand.
///
/// # Examples
///
/// ```
/// use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
/// use tetraglass_filters::{
///     StreamlineDirection, StreamlineIntegrator, StreamlineParams, StreamlineSeeds,
///     linearize_default, streamlines,
/// };
///
/// let grid = SyntheticGridSpec::new(6, 6, 6, SyntheticCellLayout::Hexahedra).generate()?;
/// let view = linearize_default(&grid)?;
///
/// let params = StreamlineParams::new(
///     "velocity",
///     StreamlineSeeds::line([0.2, 0.5, 0.5], [0.8, 0.5, 0.5], 4),
/// )
/// .direction(StreamlineDirection::Both)
/// .integrator(StreamlineIntegrator::Rk45)
/// .max_length(0.5);
///
/// let lines = streamlines(&view, &params)?;
/// assert_eq!(lines.num_polylines(), 4);
/// assert!(lines.total_length() > 0.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn streamlines(view: &TetView, params: &StreamlineParams) -> Result<PolylineSet, FilterError> {
    let locator = TetLocator::new(view);
    streamlines_with_locator(view, &locator, params)
}

/// Integrates streamlines through `view` using a pre-built [`TetLocator`].
pub fn streamlines_with_locator(
    view: &TetView,
    locator: &TetLocator,
    params: &StreamlineParams,
) -> Result<PolylineSet, FilterError> {
    streamlines_with_chunk_size(view, locator, params, DEFAULT_STREAMLINE_CHUNK_SIZE)
}

/// Linearizes `grid` with default subdivision settings and integrates streamlines according to `params`.
pub fn streamlines_grid(
    grid: &UnstructuredGrid,
    params: &StreamlineParams,
) -> Result<PolylineSet, FilterError> {
    let view = linearize_default(grid)?;
    streamlines(&view, params)
}

/// Integrates streamlines through `view` using a pre-built [`TetLocator`] and an explicit seed
/// batch `chunk_size` for deterministic parallel execution.
pub fn streamlines_with_chunk_size(
    view: &TetView,
    locator: &TetLocator,
    params: &StreamlineParams,
    chunk_size: usize,
) -> Result<PolylineSet, FilterError> {
    let seed_points = params.seeds.generate_points()?;
    let vector_source = resolve_vector_field(view, &params.vectors)?;
    let cfg = resolve_integration_config(view, params)?;

    if view.num_tets() == 0 {
        return empty_polyline_set(view);
    }

    let chunk_size = chunk_size.max(1);
    let per_seed_polylines: Vec<Option<(u32, Vec<IntegratedVertex>)>> = {
        #[cfg(not(target_arch = "wasm32"))]
        if seed_points.len() > chunk_size {
            let mut results = vec![None; seed_points.len()];
            std::thread::scope(|s| {
                let mut handles = Vec::new();
                for (chunk_idx, (seed_chunk, out_chunk)) in seed_points
                    .chunks(chunk_size)
                    .zip(results.chunks_mut(chunk_size))
                    .enumerate()
                {
                    let base_seed_id = chunk_idx * chunk_size;
                    handles.push(s.spawn(move || {
                        for (local_idx, (&seed_pt, slot)) in
                            seed_chunk.iter().zip(out_chunk.iter_mut()).enumerate()
                        {
                            let seed_id = (base_seed_id + local_idx) as u32;
                            *slot = trace_seed(
                                view,
                                locator,
                                vector_source,
                                &cfg,
                                params.direction,
                                params.integrator,
                                seed_id,
                                seed_pt,
                            );
                        }
                    }));
                }
                for handle in handles {
                    let _ = handle.join();
                }
            });
            results
        } else {
            seed_points
                .iter()
                .enumerate()
                .map(|(idx, &seed_pt)| {
                    trace_seed(
                        view,
                        locator,
                        vector_source,
                        &cfg,
                        params.direction,
                        params.integrator,
                        idx as u32,
                        seed_pt,
                    )
                })
                .collect()
        }

        #[cfg(target_arch = "wasm32")]
        {
            seed_points
                .iter()
                .enumerate()
                .map(|(idx, &seed_pt)| {
                    trace_seed(
                        view,
                        locator,
                        vector_source,
                        &cfg,
                        params.direction,
                        params.integrator,
                        idx as u32,
                        seed_pt,
                    )
                })
                .collect()
        }
    };

    assemble_polyline_set(view, per_seed_polylines)
}

fn resolve_vector_field<'a>(
    view: &'a TetView,
    raw_name: &str,
) -> Result<VectorFieldSource<'a>, FilterError> {
    if let Some(name) = raw_name.strip_prefix("point:") {
        let arr = view
            .point_data()
            .get(name)
            .ok_or_else(|| FilterError::MissingPointArray {
                name: name.to_string(),
            })?;
        validate_vector3_array(name, arr)?;
        return Ok(VectorFieldSource::Point(arr));
    }
    if let Some(name) = raw_name.strip_prefix("cell:") {
        let arr = view
            .cell_data()
            .get(name)
            .ok_or_else(|| FilterError::MissingCellArray {
                name: name.to_string(),
            })?;
        validate_vector3_array(name, arr)?;
        return Ok(VectorFieldSource::Cell(arr));
    }

    if let Some(arr) = view.point_data().get(raw_name) {
        validate_vector3_array(raw_name, arr)?;
        return Ok(VectorFieldSource::Point(arr));
    }
    if let Some(arr) = view.cell_data().get(raw_name) {
        validate_vector3_array(raw_name, arr)?;
        return Ok(VectorFieldSource::Cell(arr));
    }

    Err(FilterError::MissingAttributeArray {
        name: raw_name.to_string(),
    })
}

fn validate_vector3_array(name: &str, arr: &DataArray) -> Result<(), FilterError> {
    if arr.components() != ComponentCount::Vector3 {
        return Err(FilterError::NonVectorArray {
            array: name.to_string(),
            num_components: arr.components().as_usize(),
        });
    }
    Ok(())
}

fn resolve_integration_config(
    view: &TetView,
    params: &StreamlineParams,
) -> Result<ResolvedIntegrationConfig, FilterError> {
    if params.max_steps == 0 {
        return Err(FilterError::InvalidStreamlineParams {
            reason: "max_steps must be >= 1",
        });
    }
    if !params.terminal_speed.is_finite() || params.terminal_speed < 0.0 {
        return Err(FilterError::InvalidStreamlineParams {
            reason: "terminal_speed must be finite and >= 0.0",
        });
    }
    if !params.error_tolerance.is_finite() || params.error_tolerance <= 0.0 {
        return Err(FilterError::InvalidStreamlineParams {
            reason: "error_tolerance must be finite and > 0.0",
        });
    }

    let diag = view.bounds().map(|b| b.diagonal().max(1e-6)).unwrap_or(1.0);

    let max_length = params
        .max_length
        .unwrap_or(DEFAULT_STREAMLINE_MAX_LENGTH_FACTOR * diag);
    if !max_length.is_finite() || max_length <= 0.0 {
        return Err(FilterError::InvalidStreamlineParams {
            reason: "max_length must be finite and > 0.0",
        });
    }

    let step_size = params
        .step_size
        .unwrap_or((DEFAULT_STREAMLINE_STEP_FRACTION * diag).min(max_length));
    if !step_size.is_finite() || step_size <= 0.0 {
        return Err(FilterError::InvalidStreamlineParams {
            reason: "step_size must be finite and > 0.0",
        });
    }

    let min_step_size = params.min_step_size.unwrap_or(0.05 * step_size);
    let max_step_size = params
        .max_step_size
        .unwrap_or((5.0 * step_size).max(step_size));
    if !min_step_size.is_finite()
        || !max_step_size.is_finite()
        || min_step_size <= 0.0
        || max_step_size < min_step_size
    {
        return Err(FilterError::InvalidStreamlineParams {
            reason: "min_step_size and max_step_size must satisfy 0 < min_step_size <= max_step_size",
        });
    }

    Ok(ResolvedIntegrationConfig {
        max_length,
        step_size: step_size.clamp(min_step_size, max_step_size),
        min_step_size,
        max_step_size,
        error_tolerance: params.error_tolerance,
        max_steps: params.max_steps,
        terminal_speed: params.terminal_speed,
    })
}

#[allow(clippy::too_many_arguments)]
fn trace_seed(
    view: &TetView,
    locator: &TetLocator,
    vector_source: VectorFieldSource<'_>,
    cfg: &ResolvedIntegrationConfig,
    direction: StreamlineDirection,
    integrator: StreamlineIntegrator,
    seed_id: u32,
    seed_point: [f64; 3],
) -> Option<(u32, Vec<IntegratedVertex>)> {
    let seed_loc = locator.locate(seed_point)?;
    let seed_vertex = IntegratedVertex {
        position: seed_point,
        location: seed_loc,
        arc_length: 0.0,
        integration_time: 0.0,
    };

    let vertices = match direction {
        StreamlineDirection::Forward => trace_branch(
            view,
            locator,
            vector_source,
            cfg,
            integrator,
            seed_vertex,
            1.0,
        ),
        StreamlineDirection::Backward => trace_branch(
            view,
            locator,
            vector_source,
            cfg,
            integrator,
            seed_vertex,
            -1.0,
        ),
        StreamlineDirection::Both => {
            let mut backward = trace_branch(
                view,
                locator,
                vector_source,
                cfg,
                integrator,
                seed_vertex,
                -1.0,
            );
            let forward = trace_branch(
                view,
                locator,
                vector_source,
                cfg,
                integrator,
                seed_vertex,
                1.0,
            );
            backward.reverse();
            if forward.len() > 1 {
                backward.extend_from_slice(&forward[1..]);
            }
            backward
        }
    };

    if vertices.len() >= 2 {
        Some((seed_id, vertices))
    } else {
        None
    }
}

fn trace_branch(
    view: &TetView,
    locator: &TetLocator,
    vector_source: VectorFieldSource<'_>,
    cfg: &ResolvedIntegrationConfig,
    integrator: StreamlineIntegrator,
    seed_vertex: IntegratedVertex,
    dir_sign: f64,
) -> Vec<IntegratedVertex> {
    let mut out = Vec::new();
    out.push(seed_vertex);

    let mut pos = seed_vertex.position;
    let mut loc = seed_vertex.location;
    let mut arc_len = 0.0;
    let mut time = 0.0;
    let mut h = cfg.step_size;
    let min_boundary_h = (cfg.min_step_size * 0.1).max(1e-9);

    for _step in 0..cfg.max_steps {
        let remaining = cfg.max_length - arc_len;
        if remaining <= 1e-12 {
            break;
        }

        let Some(raw_v) = vector_source.sample(view, &loc) else {
            break;
        };
        let speed = norm3(raw_v);
        if !speed.is_finite() || speed <= cfg.terminal_speed {
            break;
        }
        let v0 = scale3(raw_v, dir_sign);

        let mut step_h = h.min(remaining);
        let mut accepted: Option<([f64; 3], TetLocation, f64, f64)> = None;

        // Retry with smaller step size when a step exits the mesh boundary or exceeds RK45 error tolerance
        for _attempt in 0..12 {
            let dt = step_h / speed;
            match integrator {
                StreamlineIntegrator::Rk4 => {
                    if let Some((next_pos, next_loc)) = rk4_step(
                        view,
                        locator,
                        vector_source,
                        pos,
                        loc.tet_index,
                        v0,
                        dt,
                        dir_sign,
                    ) {
                        accepted = Some((next_pos, next_loc, dt, step_h));
                        break;
                    }
                    // Step exited domain: halve step length to approach the boundary cleanly
                    step_h *= 0.5;
                    if step_h < min_boundary_h {
                        break;
                    }
                }
                StreamlineIntegrator::Rk45 => {
                    if let Some((next_pos, next_loc, err)) = rk45_step(
                        view,
                        locator,
                        vector_source,
                        pos,
                        loc.tet_index,
                        v0,
                        dt,
                        dir_sign,
                    ) {
                        if err <= cfg.error_tolerance || step_h <= cfg.min_step_size {
                            let ratio = cfg.error_tolerance / err.max(1e-16);
                            let factor = (0.9 * ratio.powf(0.2)).clamp(0.2, 4.0);
                            let next_h =
                                (step_h * factor).clamp(cfg.min_step_size, cfg.max_step_size);
                            accepted = Some((next_pos, next_loc, dt, next_h));
                            break;
                        }
                        // Truncation error exceeded tolerance: shrink step and retry
                        let ratio = cfg.error_tolerance / err;
                        let factor = (0.9 * ratio.powf(0.25)).clamp(0.2, 0.8);
                        step_h = (step_h * factor).max(cfg.min_step_size);
                    } else {
                        // Step exited domain: halve step length to approach the boundary cleanly
                        step_h *= 0.5;
                        if step_h < min_boundary_h {
                            break;
                        }
                    }
                }
            }
        }

        let Some((next_pos, next_loc, dt, next_h)) = accepted else {
            break;
        };

        let seg_len = dist3(pos, next_pos);
        if seg_len <= 1e-14 {
            break;
        }

        pos = next_pos;
        loc = next_loc;
        arc_len += seg_len;
        time += dir_sign * dt;
        h = next_h;

        out.push(IntegratedVertex {
            position: pos,
            location: loc,
            arc_length: arc_len,
            integration_time: time,
        });
    }

    out
}

#[allow(clippy::too_many_arguments)]
fn rk4_step(
    view: &TetView,
    locator: &TetLocator,
    vector_source: VectorFieldSource<'_>,
    p0: [f64; 3],
    hint_tet: usize,
    k1: [f64; 3],
    dt: f64,
    dir_sign: f64,
) -> Option<([f64; 3], TetLocation)> {
    let p2 = add_scaled3(p0, k1, 0.5 * dt);
    let loc2 = locator.locate_with_hint(p2, Some(hint_tet))?;
    let k2 = scale3(vector_source.sample(view, &loc2)?, dir_sign);

    let p3 = add_scaled3(p0, k2, 0.5 * dt);
    let loc3 = locator.locate_with_hint(p3, Some(loc2.tet_index))?;
    let k3 = scale3(vector_source.sample(view, &loc3)?, dir_sign);

    let p4 = add_scaled3(p0, k3, dt);
    let loc4 = locator.locate_with_hint(p4, Some(loc3.tet_index))?;
    let k4 = scale3(vector_source.sample(view, &loc4)?, dir_sign);

    let next_pos = [
        p0[0] + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        p0[1] + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        p0[2] + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
    ];
    let next_loc = locator.locate_with_hint(next_pos, Some(loc4.tet_index))?;
    Some((next_pos, next_loc))
}

#[allow(clippy::too_many_arguments)]
fn rk45_step(
    view: &TetView,
    locator: &TetLocator,
    vector_source: VectorFieldSource<'_>,
    p0: [f64; 3],
    hint_tet: usize,
    k1: [f64; 3],
    dt: f64,
    dir_sign: f64,
) -> Option<([f64; 3], TetLocation, f64)> {
    // Dormand-Prince 5(4) (`DOPRI5`) Butcher tableau coefficients
    const A21: f64 = 1.0 / 5.0;

    const A31: f64 = 3.0 / 40.0;
    const A32: f64 = 9.0 / 40.0;

    const A41: f64 = 44.0 / 45.0;
    const A42: f64 = -56.0 / 15.0;
    const A43: f64 = 32.0 / 9.0;

    const A51: f64 = 19372.0 / 6561.0;
    const A52: f64 = -25360.0 / 2187.0;
    const A53: f64 = 64448.0 / 6561.0;
    const A54: f64 = -212.0 / 729.0;

    const A61: f64 = 9017.0 / 3168.0;
    const A62: f64 = -355.0 / 33.0;
    const A63: f64 = 46732.0 / 5247.0;
    const A64: f64 = 49.0 / 176.0;
    const A65: f64 = -5103.0 / 18656.0;

    // 5th-order solution weights (also stage 7 coefficients)
    const B1: f64 = 35.0 / 384.0;
    const B3: f64 = 500.0 / 1113.0;
    const B4: f64 = 125.0 / 192.0;
    const B5: f64 = -2187.0 / 6784.0;
    const B6: f64 = 11.0 / 84.0;

    // Error weights `E_i = B_i - B_hat_i`
    const E1: f64 = 71.0 / 57600.0;
    const E3: f64 = -71.0 / 16695.0;
    const E4: f64 = 71.0 / 1920.0;
    const E5: f64 = -17253.0 / 339200.0;
    const E6: f64 = 22.0 / 525.0;
    const E7: f64 = -1.0 / 40.0;

    let p2 = add_scaled3(p0, k1, dt * A21);
    let loc2 = locator.locate_with_hint(p2, Some(hint_tet))?;
    let k2 = scale3(vector_source.sample(view, &loc2)?, dir_sign);

    let p3 = [
        p0[0] + dt * (A31 * k1[0] + A32 * k2[0]),
        p0[1] + dt * (A31 * k1[1] + A32 * k2[1]),
        p0[2] + dt * (A31 * k1[2] + A32 * k2[2]),
    ];
    let loc3 = locator.locate_with_hint(p3, Some(loc2.tet_index))?;
    let k3 = scale3(vector_source.sample(view, &loc3)?, dir_sign);

    let p4 = [
        p0[0] + dt * (A41 * k1[0] + A42 * k2[0] + A43 * k3[0]),
        p0[1] + dt * (A41 * k1[1] + A42 * k2[1] + A43 * k3[1]),
        p0[2] + dt * (A41 * k1[2] + A42 * k2[2] + A43 * k3[2]),
    ];
    let loc4 = locator.locate_with_hint(p4, Some(loc3.tet_index))?;
    let k4 = scale3(vector_source.sample(view, &loc4)?, dir_sign);

    let p5 = [
        p0[0] + dt * (A51 * k1[0] + A52 * k2[0] + A53 * k3[0] + A54 * k4[0]),
        p0[1] + dt * (A51 * k1[1] + A52 * k2[1] + A53 * k3[1] + A54 * k4[1]),
        p0[2] + dt * (A51 * k1[2] + A52 * k2[2] + A53 * k3[2] + A54 * k4[2]),
    ];
    let loc5 = locator.locate_with_hint(p5, Some(loc4.tet_index))?;
    let k5 = scale3(vector_source.sample(view, &loc5)?, dir_sign);

    let p6 = [
        p0[0] + dt * (A61 * k1[0] + A62 * k2[0] + A63 * k3[0] + A64 * k4[0] + A65 * k5[0]),
        p0[1] + dt * (A61 * k1[1] + A62 * k2[1] + A63 * k3[1] + A64 * k4[1] + A65 * k5[1]),
        p0[2] + dt * (A61 * k1[2] + A62 * k2[2] + A63 * k3[2] + A64 * k4[2] + A65 * k5[2]),
    ];
    let loc6 = locator.locate_with_hint(p6, Some(loc5.tet_index))?;
    let k6 = scale3(vector_source.sample(view, &loc6)?, dir_sign);

    let next_pos = [
        p0[0] + dt * (B1 * k1[0] + B3 * k3[0] + B4 * k4[0] + B5 * k5[0] + B6 * k6[0]),
        p0[1] + dt * (B1 * k1[1] + B3 * k3[1] + B4 * k4[1] + B5 * k5[1] + B6 * k6[1]),
        p0[2] + dt * (B1 * k1[2] + B3 * k3[2] + B4 * k4[2] + B5 * k5[2] + B6 * k6[2]),
    ];
    let next_loc = locator.locate_with_hint(next_pos, Some(loc6.tet_index))?;
    let k7 = scale3(vector_source.sample(view, &next_loc)?, dir_sign);

    let err_vec = [
        dt * (E1 * k1[0] + E3 * k3[0] + E4 * k4[0] + E5 * k5[0] + E6 * k6[0] + E7 * k7[0]),
        dt * (E1 * k1[1] + E3 * k3[1] + E4 * k4[1] + E5 * k5[1] + E6 * k6[1] + E7 * k7[1]),
        dt * (E1 * k1[2] + E3 * k3[2] + E4 * k4[2] + E5 * k5[2] + E6 * k6[2] + E7 * k7[2]),
    ];
    let err = norm3(err_vec);
    if !err.is_finite() {
        return None;
    }

    Some((next_pos, next_loc, err))
}

fn empty_polyline_set(view: &TetView) -> Result<PolylineSet, FilterError> {
    let mut point_data = Attributes::new(0);
    for (name, arr) in view.point_data().iter() {
        point_data.insert(
            name,
            DataArray::from_f64(arr.components(), Vec::<f64>::new())?,
        )?;
    }
    for (name, arr) in view.cell_data().iter() {
        if point_data.get(name).is_none() {
            point_data.insert(
                name,
                DataArray::from_f64(arr.components(), Vec::<f64>::new())?,
            )?;
        }
    }
    if point_data.get("arc_length").is_none() {
        point_data.insert(
            "arc_length",
            DataArray::from_f64(ComponentCount::Scalar, Vec::<f64>::new())?,
        )?;
    }
    if point_data.get("integration_time").is_none() {
        point_data.insert(
            "integration_time",
            DataArray::from_f64(ComponentCount::Scalar, Vec::<f64>::new())?,
        )?;
    }

    let mut cell_data = Attributes::new(0);
    cell_data.insert(
        "seed_id",
        DataArray::from_i32(ComponentCount::Scalar, Vec::<i32>::new())?,
    )?;

    Ok(PolylineSet::new(
        Vec::<[f64; 3]>::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        point_data,
        cell_data,
    )?)
}

fn assemble_polyline_set(
    view: &TetView,
    per_seed_polylines: Vec<Option<(u32, Vec<IntegratedVertex>)>>,
) -> Result<PolylineSet, FilterError> {
    let mut total_points = 0usize;
    let mut num_polylines = 0usize;
    for entry in per_seed_polylines.iter().flatten() {
        total_points += entry.1.len();
        num_polylines += 1;
    }

    if num_polylines == 0 || total_points == 0 {
        return empty_polyline_set(view);
    }

    let mut points = Vec::with_capacity(total_points);
    let mut offsets = Vec::with_capacity(num_polylines);
    let mut point_source_cells = Vec::with_capacity(total_points);
    let mut polyline_seed_ids = Vec::with_capacity(num_polylines);
    let mut locations = Vec::with_capacity(total_points);
    let mut arc_lengths = Vec::with_capacity(total_points);
    let mut integration_times = Vec::with_capacity(total_points);
    let mut seed_ids_i32 = Vec::with_capacity(num_polylines);

    for (seed_id, verts) in per_seed_polylines.into_iter().flatten() {
        for v in verts {
            points.push(v.position);
            point_source_cells.push(v.location.source_cell_id);
            locations.push(v.location);
            arc_lengths.push(v.arc_length);
            integration_times.push(v.integration_time);
        }
        offsets.push(points.len() as u32);
        polyline_seed_ids.push(seed_id);
        seed_ids_i32.push(seed_id as i32);
    }

    let mut point_data = Attributes::new(total_points);

    // 1. Interpolate all point arrays from `view.point_data()` using barycentric weights:
    for (name, arr) in view.point_data().iter() {
        let ncomp = arr.components().as_usize();
        let mut values = vec![0.0_f64; total_points * ncomp];
        for (pt_idx, loc) in locations.iter().enumerate() {
            let _ = loc.interpolate_point_tuple(
                view,
                arr,
                &mut values[pt_idx * ncomp..(pt_idx + 1) * ncomp],
            );
        }
        point_data.insert(name, DataArray::from_f64(arr.components(), values)?)?;
    }

    // 2. Sample any cell arrays from `view.cell_data()` onto streamline points if not shadowed:
    for (name, arr) in view.cell_data().iter() {
        if point_data.get(name).is_some() {
            continue;
        }
        let ncomp = arr.components().as_usize();
        let mut values = Vec::with_capacity(total_points * ncomp);
        for loc in &locations {
            for c in 0..ncomp {
                values.push(loc.sample_cell_scalar(arr, c).unwrap_or(0.0));
            }
        }
        point_data.insert(name, DataArray::from_f64(arr.components(), values)?)?;
    }

    // 3. Attach streamline arc_length and integration_time point arrays:
    if point_data.get("arc_length").is_none() {
        point_data.insert(
            "arc_length",
            DataArray::from_f64(ComponentCount::Scalar, arc_lengths)?,
        )?;
    }
    if point_data.get("integration_time").is_none() {
        point_data.insert(
            "integration_time",
            DataArray::from_f64(ComponentCount::Scalar, integration_times)?,
        )?;
    }

    let mut cell_data = Attributes::new(num_polylines);
    cell_data.insert(
        "seed_id",
        DataArray::from_i32(ComponentCount::Scalar, seed_ids_i32)?,
    )?;

    Ok(PolylineSet::new(
        points,
        offsets,
        point_source_cells,
        polyline_seed_ids,
        point_data,
        cell_data,
    )?)
}

fn orthonormal_plane_basis(unit_n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let ref_axis = if unit_n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u_raw = cross3(unit_n, ref_axis);
    let u_len = norm3(u_raw).max(1e-15);
    let u = [u_raw[0] / u_len, u_raw[1] / u_len, u_raw[2] / u_len];
    let v = cross3(unit_n, u);
    (u, v)
}

#[inline]
const fn is_finite_vec3(v: [f64; 3]) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

#[inline]
const fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

#[inline]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
const fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
const fn add_scaled3(a: [f64; 3], b: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}

#[inline]
const fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

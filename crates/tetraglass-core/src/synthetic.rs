//! Deterministic synthetic benchmark mesh generation and M2 interactive performance target (`M1.9`, D-019).
//!
//! Provides a zero-I/O generator for structured hexahedral and face-consistent Kuhn-tetrahedral
//! unstructured grids over a 3D box domain with analytic scalar and vector fields (`linear`,
//! `radius`, `velocity`, and `cell_id`), together with [`M2_PERFORMANCE_TARGET`].

use crate::array::{ComponentCount, DataArray};
use crate::cell::CellType;
use crate::error::{CoreError, try_u32_from_usize};
use crate::grid::{BoundingBox, UnstructuredGrid};

/// Interactive performance target for Milestone M2 (`D-019` in `docs/decisions.md`).
///
/// Evaluated in `--release` builds on the reference synthetic benchmark mesh
/// ([`m2_benchmark_tet_spec`] / [`m2_benchmark_hex_spec`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerformanceTarget {
    /// Grid subdivisions `[nx, ny, nz]` along $X$, $Y$, and $Z$.
    pub divisions: [u32; 3],
    /// Number of hexahedral cells in [`m2_benchmark_hex_spec`] (`nx * ny * nz`).
    pub num_hex_cells: usize,
    /// Number of tetrahedral cells in [`m2_benchmark_tet_spec`] (`6 * nx * ny * nz`).
    pub num_tet_cells: usize,
    /// Number of mesh vertices (`(nx + 1) * (ny + 1) * (nz + 1)`).
    pub num_points: usize,
    /// Reference viewport resolution `[width, height]` in physical pixels.
    pub viewport_size: [u32; 2],
    /// Reference MSAA sample count for interactive rendering.
    pub msaa_samples: u32,
    /// Target maximum frame time in milliseconds ($60\text{ FPS} \approx 16.6\text{ ms}$) for camera
    /// orbit/pan/zoom, colormap/range uniform updates, and shader clip-plane dragging (`M2.8`).
    pub max_frame_time_ms: f64,
    /// Target maximum latency in milliseconds for an interactive plane `slice` or single-isovalue
    /// `contour` on the linearized `600,000`-tetrahedron mesh (`M2.3`, `M2.4`).
    pub max_slice_contour_ms: f64,
    /// Target maximum latency in milliseconds for full-volume operations (`linearize`,
    /// `extract_external_surface`, exact `clip`, or `threshold`) on the reference mesh (`M2.1`–`M2.6`).
    pub max_volume_filter_ms: f64,
}

/// Canonical Milestone M2 interactive performance target (`D-019`).
///
/// - **Mesh size:** `50 × 50 × 40 = 100,000` hexahedra (`600,000` linearized tetrahedra) and `106,641` points.
/// - **Viewport frame time:** $\le 16.6\text{ ms}$ ($60\text{ FPS}$) at `1200 × 800` with `4x` MSAA.
/// - **Interactive slice / contour latency:** $\le 50.0\text{ ms}$ in `--release`.
/// - **Volume filter / linearization / surface extraction latency:** $\le 150.0\text{ ms}$ in `--release`.
pub const M2_PERFORMANCE_TARGET: PerformanceTarget = PerformanceTarget {
    divisions: [50, 50, 40],
    num_hex_cells: 100_000,
    num_tet_cells: 600_000,
    num_points: 51 * 51 * 41,
    viewport_size: [1200, 800],
    msaa_samples: 4,
    max_frame_time_ms: 16.6,
    max_slice_contour_ms: 50.0,
    max_volume_filter_ms: 150.0,
};

/// Cell topology emitted by [`generate_synthetic_grid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyntheticCellLayout {
    /// One 8-node linear hexahedron (`CellType::Hexahedron`) per grid sub-cell (`nx * ny * nz` cells).
    Hexahedra,
    /// Six 4-node linear tetrahedra (`CellType::Tetra`) per grid sub-cell using the face-consistent
    /// Kuhn diagonal decomposition (`6 * nx * ny * nz` cells).
    Tetrahedra6,
}

/// Specification for generating a deterministic synthetic 3D unstructured grid in code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticGridSpec {
    /// Number of cell subdivisions along the $X$, $Y$, and $Z$ axes (each must be $\ge 1$).
    pub divisions: [u32; 3],
    /// Spatial bounding box of the generated grid (defaults to `[0.0, 0.0, 0.0]..=[1.0, 1.0, 1.0]`).
    pub bounds: BoundingBox,
    /// Cell layout (`Hexahedra` or `Tetrahedra6`).
    pub layout: SyntheticCellLayout,
}

impl SyntheticGridSpec {
    /// Creates a [`SyntheticGridSpec`] over the unit cube `[0, 1]^3` with `[nx, ny, nz]` subdivisions.
    #[must_use]
    pub const fn new(nx: u32, ny: u32, nz: u32, layout: SyntheticCellLayout) -> Self {
        Self {
            divisions: [nx, ny, nz],
            bounds: BoundingBox {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            },
            layout,
        }
    }

    /// Sets a custom spatial [`BoundingBox`] for the synthetic grid.
    #[must_use]
    pub const fn with_bounds(mut self, bounds: BoundingBox) -> Self {
        self.bounds = bounds;
        self
    }

    /// Returns the total number of vertices `(nx + 1) * (ny + 1) * (nz + 1)`.
    #[must_use]
    pub const fn num_points(&self) -> usize {
        let [nx, ny, nz] = self.divisions;
        ((nx as usize) + 1) * ((ny as usize) + 1) * ((nz as usize) + 1)
    }

    /// Returns the total number of cells (`nx * ny * nz` for `Hexahedra`, `6 * nx * ny * nz` for `Tetrahedra6`).
    #[must_use]
    pub const fn num_cells(&self) -> usize {
        let [nx, ny, nz] = self.divisions;
        let sub_cubes = (nx as usize) * (ny as usize) * (nz as usize);
        match self.layout {
            SyntheticCellLayout::Hexahedra => sub_cubes,
            SyntheticCellLayout::Tetrahedra6 => sub_cubes * 6,
        }
    }

    /// Returns the expected number of outer boundary faces (`2 * (nx*ny + ny*nz + nx*nz)` for hexes,
    /// twice that for `Tetrahedra6`).
    #[must_use]
    pub const fn expected_boundary_faces(&self) -> usize {
        let [nx, ny, nz] = self.divisions;
        let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
        let quad_faces = 2 * (nx * ny + ny * nz + nx * nz);
        match self.layout {
            SyntheticCellLayout::Hexahedra => quad_faces,
            SyntheticCellLayout::Tetrahedra6 => quad_faces * 2,
        }
    }

    /// Generates the [`UnstructuredGrid`] described by this specification.
    pub fn generate(&self) -> Result<UnstructuredGrid, CoreError> {
        generate_synthetic_grid(self)
    }
}

/// Returns the canonical M2 benchmark specification for a `600,000`-tetrahedron grid (`50 × 50 × 40 × 6`).
#[must_use]
pub const fn m2_benchmark_tet_spec() -> SyntheticGridSpec {
    let [nx, ny, nz] = M2_PERFORMANCE_TARGET.divisions;
    SyntheticGridSpec::new(nx, ny, nz, SyntheticCellLayout::Tetrahedra6)
}

/// Returns the canonical M2 benchmark specification for a `100,000`-hexahedron grid (`50 × 50 × 40`).
#[must_use]
pub const fn m2_benchmark_hex_spec() -> SyntheticGridSpec {
    let [nx, ny, nz] = M2_PERFORMANCE_TARGET.divisions;
    SyntheticGridSpec::new(nx, ny, nz, SyntheticCellLayout::Hexahedra)
}

/// Generates a deterministic 3D [`UnstructuredGrid`] with analytic point and cell fields:
///
/// - Point scalar `"linear"` (`1` component, `f64`): $x + 2y + 3z$
/// - Point scalar `"radius"` (`1` component, `f64`): $\|(x, y, z) - \text{center}\|_2$
/// - Point vector `"velocity"` (`3` components, `f64`): $[-(y - c_y), x - c_x, 0.5 (z - c_z)]$
/// - Cell scalar `"cell_id"` (`1` component, `f64`): zero-based cell index `0..num_cells`
///
/// # Example
///
/// ```
/// use tetraglass_core::{SyntheticCellLayout, SyntheticGridSpec};
///
/// let spec = SyntheticGridSpec::new(4, 4, 4, SyntheticCellLayout::Tetrahedra6);
/// let grid = spec.generate().unwrap();
/// assert_eq!(grid.num_points(), 125);
/// assert_eq!(grid.num_cells(), 384);
/// assert!(grid.point_data().get("linear").is_some());
/// assert!(grid.point_data().get("radius").is_some());
/// assert!(grid.point_data().get("velocity").is_some());
/// assert!(grid.cell_data().get("cell_id").is_some());
/// ```
pub fn generate_synthetic_grid(spec: &SyntheticGridSpec) -> Result<UnstructuredGrid, CoreError> {
    let [nx, ny, nz] = spec.divisions;
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(CoreError::InvalidSyntheticDivisions {
            divisions: spec.divisions,
        });
    }

    let num_pts = spec.num_points();
    let _ = try_u32_from_usize(num_pts)?;

    let npx = (nx as usize) + 1;
    let npy = (ny as usize) + 1;
    let npz = (nz as usize) + 1;

    let min = spec.bounds.min;
    let max = spec.bounds.max;
    let center = spec.bounds.center();
    let dx = (max[0] - min[0]) / f64::from(nx);
    let dy = (max[1] - min[1]) / f64::from(ny);
    let dz = (max[2] - min[2]) / f64::from(nz);

    let mut points = Vec::with_capacity(num_pts);
    let mut linear = Vec::with_capacity(num_pts);
    let mut radius = Vec::with_capacity(num_pts);
    let mut velocity = Vec::with_capacity(num_pts * 3);

    for k in 0..npz {
        let z = if k == (nz as usize) {
            max[2]
        } else {
            min[2] + (k as f64) * dz
        };
        for j in 0..npy {
            let y = if j == (ny as usize) {
                max[1]
            } else {
                min[1] + (j as f64) * dy
            };
            for i in 0..npx {
                let x = if i == (nx as usize) {
                    max[0]
                } else {
                    min[0] + (i as f64) * dx
                };
                points.push([x, y, z]);
                linear.push(x + 2.0 * y + 3.0 * z);
                let rx = x - center[0];
                let ry = y - center[1];
                let rz = z - center[2];
                radius.push((rx * rx + ry * ry + rz * rz).sqrt());
                velocity.push(-ry);
                velocity.push(rx);
                velocity.push(0.5 * rz);
            }
        }
    }

    let point_idx = |i: usize, j: usize, k: usize| -> u32 { ((k * npy + j) * npx + i) as u32 };

    let num_cells = spec.num_cells();
    let mut cell_types = Vec::with_capacity(num_cells);
    let mut offsets = Vec::with_capacity(num_cells);

    let nodes_per_cell = match spec.layout {
        SyntheticCellLayout::Hexahedra => 8,
        SyntheticCellLayout::Tetrahedra6 => 4,
    };
    let _ = try_u32_from_usize(num_cells * nodes_per_cell)?;
    let mut connectivity = Vec::with_capacity(num_cells * nodes_per_cell);

    for k in 0..(nz as usize) {
        for j in 0..(ny as usize) {
            for i in 0..(nx as usize) {
                let v000 = point_idx(i, j, k);
                let v100 = point_idx(i + 1, j, k);
                let v110 = point_idx(i + 1, j + 1, k);
                let v010 = point_idx(i, j + 1, k);
                let v001 = point_idx(i, j, k + 1);
                let v101 = point_idx(i + 1, j, k + 1);
                let v111 = point_idx(i + 1, j + 1, k + 1);
                let v011 = point_idx(i, j + 1, k + 1);

                match spec.layout {
                    SyntheticCellLayout::Hexahedra => {
                        cell_types.push(CellType::Hexahedron);
                        connectivity
                            .extend_from_slice(&[v000, v100, v110, v010, v001, v101, v111, v011]);
                        offsets.push(connectivity.len() as u32);
                    }
                    SyntheticCellLayout::Tetrahedra6 => {
                        // Kuhn 6-tetrahedra subdivision sharing main diagonal [v000, v111].
                        let tets: [[u32; 4]; 6] = [
                            [v000, v100, v110, v111],
                            [v000, v100, v111, v101],
                            [v000, v010, v111, v110],
                            [v000, v010, v011, v111],
                            [v000, v001, v101, v111],
                            [v000, v001, v111, v011],
                        ];
                        for tet in tets {
                            cell_types.push(CellType::Tetra);
                            connectivity.extend_from_slice(&tet);
                            offsets.push(connectivity.len() as u32);
                        }
                    }
                }
            }
        }
    }

    let cell_ids: Vec<f64> = (0..num_cells).map(|idx| idx as f64).collect();

    let mut grid = UnstructuredGrid::new(points, cell_types, offsets, connectivity, None)?;
    grid.insert_point_data(
        "linear",
        DataArray::from_f64(ComponentCount::Scalar, linear)?,
    )?;
    grid.insert_point_data(
        "radius",
        DataArray::from_f64(ComponentCount::Scalar, radius)?,
    )?;
    grid.insert_point_data(
        "velocity",
        DataArray::from_f64(ComponentCount::Vector3, velocity)?,
    )?;
    grid.insert_cell_data(
        "cell_id",
        DataArray::from_f64(ComponentCount::Scalar, cell_ids)?,
    )?;

    Ok(grid)
}

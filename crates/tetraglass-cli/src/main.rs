//! Command-line interface for Tetraglass: `inspect`, `render`, `validate`, and `schema` (`M1.4`).
//!
//! See `docs/design-plan.md` and `docs/scene-format.md` for command usage and JSON outputs.

use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::{Deserialize, Serialize};
use tetraglass_core::{Attributes, BoundingBox, CellType, ScalarType, UnstructuredGrid};
use tetraglass_io::{MeshFormat, detect_format_from_bytes, detect_format_from_extension, read_vtu};
use tetraglass_render::{
    Annotations, Camera, ColorBy, ColorbarAnnotation, ColorbarPlacement, ColormapKind,
    DEFAULT_COLOR_FORMAT, DEFAULT_COLORBAR_TICKS, GpuSurface, HeadlessContext, OffscreenTarget,
    Renderer, ScalarComponent, SurfaceStyle,
};
use tetraglass_scene::{
    ArrayAssociation, BackgroundSpec, ColorbarSpec, ColormapName, DEFAULT_MSAA_SAMPLES,
    DEFAULT_OUTPUT_SIZE, DatasetMetadata, EvaluatedColoring, EvaluatedRepresentation,
    EvaluatedScene, RenderEngine, RepresentationStyle, ResolvedComponent, Scene, SceneEvaluator,
    ValidationIssue, ValidationReport, scene_json_schema_string, validate_scene_with_metadata,
};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match run_cli(&args) {
        Ok(output) => {
            if !output.is_empty() {
                println!("{output}");
            }
            ExitCode::SUCCESS
        }
        Err(CliFailure::Validation {
            output,
            is_json_stdout,
        }) => {
            if is_json_stdout {
                println!("{output}");
            } else {
                eprintln!("{output}");
            }
            ExitCode::FAILURE
        }
        Err(CliFailure::Message(msg)) => {
            eprintln!("tetraglass: {msg}");
            ExitCode::FAILURE
        }
    }
}

enum CliFailure {
    Message(String),
    Validation {
        output: String,
        is_json_stdout: bool,
    },
}

impl From<String> for CliFailure {
    fn from(s: String) -> Self {
        Self::Message(s)
    }
}

fn run_cli(args: &[String]) -> Result<String, CliFailure> {
    let Some((cmd, rest)) = args.split_first() else {
        return Ok(help_text());
    };
    match cmd.as_str() {
        "help" | "--help" | "-h" => Ok(help_text()),
        "--version" | "-V" | "version" => Ok(format!("tetraglass {}", env!("CARGO_PKG_VERSION"))),
        "schema" => cmd_schema(rest),
        "inspect" => cmd_inspect(rest),
        "validate" => cmd_validate(rest),
        "render" => cmd_render(rest),
        other => Err(CliFailure::Message(format!(
            "unknown command `{other}`; run `tetraglass --help` for usage"
        ))),
    }
}

fn help_text() -> String {
    format!(
        "tetraglass {}\n\n\
         Usage:\n  \
         tetraglass inspect <file> [--json]\n  \
         tetraglass render <scene.json> [-o <out.png>] [--engine <raster|cycles>]\n  \
         tetraglass validate <scene.json> [--json]\n  \
         tetraglass schema",
        env!("CARGO_PKG_VERSION")
    )
}

fn cmd_schema(args: &[String]) -> Result<String, CliFailure> {
    if let Some(arg) = args.first() {
        if arg == "--help" || arg == "-h" {
            return Ok("Usage: tetraglass schema".to_string());
        }
        return Err(CliFailure::Message(format!(
            "`schema` takes no arguments, got `{arg}`"
        )));
    }
    scene_json_schema_string().map_err(|e| CliFailure::Message(e.to_string()))
}

// --- `inspect` command ---

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct InspectBounds {
    min: [f64; 3],
    max: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct InspectArraySummary {
    name: String,
    scalar_type: String,
    components: usize,
    num_tuples: usize,
    component_ranges: Vec<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    magnitude_range: Option<[f64; 2]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct InspectSummary {
    file: String,
    format: String,
    num_points: usize,
    num_cells: usize,
    bounds: Option<InspectBounds>,
    cell_types: BTreeMap<String, usize>,
    point_arrays: Vec<InspectArraySummary>,
    cell_arrays: Vec<InspectArraySummary>,
    time_steps: Vec<f64>,
}

fn cmd_inspect(args: &[String]) -> Result<String, CliFailure> {
    let mut json_flag = false;
    let mut file_arg: Option<&str> = None;

    for arg in args {
        match arg.as_str() {
            "--help" | "-h" => {
                return Ok("Usage: tetraglass inspect <file> [--json]".to_string());
            }
            "--json" => json_flag = true,
            other if other.starts_with('-') => {
                return Err(CliFailure::Message(format!(
                    "unknown option `{other}` for `inspect`"
                )));
            }
            path => {
                if file_arg.is_some() {
                    return Err(CliFailure::Message(
                        "`inspect` accepts a single <file> argument".to_string(),
                    ));
                }
                file_arg = Some(path);
            }
        }
    }

    let Some(file_path_str) = file_arg else {
        return Err(CliFailure::Message(
            "missing <file> argument; usage: `tetraglass inspect <file> [--json]`".to_string(),
        ));
    };

    let path = Path::new(file_path_str);
    let (format, grid) = load_mesh_from_path(path, None)?;
    let summary = build_inspect_summary(file_path_str, format, &grid);

    if json_flag {
        serde_json::to_string_pretty(&summary).map_err(|e| CliFailure::Message(e.to_string()))
    } else {
        Ok(format_inspect_text(&summary))
    }
}

fn build_inspect_summary(
    file: &str,
    format: MeshFormat,
    grid: &UnstructuredGrid,
) -> InspectSummary {
    let mut cell_types: BTreeMap<String, usize> = BTreeMap::new();
    for &ct in grid.cell_types() {
        *cell_types.entry(cell_type_label(ct)).or_insert(0) += 1;
    }

    let bounds = grid.bounds().map(|b| InspectBounds {
        min: b.min,
        max: b.max,
    });

    InspectSummary {
        file: file.to_string(),
        format: mesh_format_label(format).to_string(),
        num_points: grid.num_points(),
        num_cells: grid.num_cells(),
        bounds,
        cell_types,
        point_arrays: summarize_attributes(grid.point_data()),
        cell_arrays: summarize_attributes(grid.cell_data()),
        time_steps: Vec::new(),
    }
}

fn summarize_attributes(attrs: &Attributes) -> Vec<InspectArraySummary> {
    attrs
        .iter()
        .map(|(name, arr)| {
            let num_comp = arr.components().as_usize();
            let component_ranges = (0..num_comp)
                .map(|c| {
                    arr.component_range(c)
                        .map_or([0.0, 0.0], |r| [r.min, r.max])
                })
                .collect();
            let magnitude_range = if num_comp > 1 {
                arr.magnitude_range().map(|r| [r.min, r.max])
            } else {
                None
            };
            InspectArraySummary {
                name: name.to_string(),
                scalar_type: scalar_type_label(arr.scalar_type()).to_string(),
                components: num_comp,
                num_tuples: arr.num_tuples(),
                component_ranges,
                magnitude_range,
            }
        })
        .collect()
}

fn format_inspect_text(summary: &InspectSummary) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "File: {} ({})", summary.file, summary.format);
    let _ = writeln!(
        out,
        "Points: {} | Cells: {}",
        summary.num_points, summary.num_cells
    );
    if let Some(b) = &summary.bounds {
        let _ = writeln!(
            out,
            "Bounds: [{:.6}, {:.6}, {:.6}] .. [{:.6}, {:.6}, {:.6}]",
            b.min[0], b.min[1], b.min[2], b.max[0], b.max[1], b.max[2]
        );
    } else {
        let _ = writeln!(out, "Bounds: empty");
    }

    if !summary.cell_types.is_empty() {
        let pairs: Vec<String> = summary
            .cell_types
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect();
        let _ = writeln!(out, "Cell types: {}", pairs.join(", "));
    }

    let _ = writeln!(out, "Point arrays ({}):", summary.point_arrays.len());
    for arr in &summary.point_arrays {
        format_array_line(&mut out, arr);
    }

    let _ = writeln!(out, "Cell arrays ({}):", summary.cell_arrays.len());
    for arr in &summary.cell_arrays {
        format_array_line(&mut out, arr);
    }

    if !summary.time_steps.is_empty() {
        let _ = writeln!(
            out,
            "Time steps ({}): {:?}",
            summary.time_steps.len(),
            summary.time_steps
        );
    }

    while out.ends_with('\n') {
        out.pop();
    }
    out
}

fn format_array_line(out: &mut String, arr: &InspectArraySummary) {
    if arr.components == 1 {
        let r = arr.component_ranges.first().copied().unwrap_or([0.0, 0.0]);
        let _ = writeln!(
            out,
            "  - {} ({}, 1 component): range [{:.6}, {:.6}]",
            arr.name, arr.scalar_type, r[0], r[1]
        );
    } else {
        let mag = arr.magnitude_range.unwrap_or([0.0, 0.0]);
        let _ = writeln!(
            out,
            "  - {} ({}, {} components): magnitude [{:.6}, {:.6}], components {:?}",
            arr.name, arr.scalar_type, arr.components, mag[0], mag[1], arr.component_ranges
        );
    }
}

fn cell_type_label(ct: CellType) -> String {
    match ct {
        CellType::Vertex => "vertex".to_string(),
        CellType::Line => "line".to_string(),
        CellType::Triangle => "triangle".to_string(),
        CellType::Quad => "quad".to_string(),
        CellType::Tetra => "tetra".to_string(),
        CellType::Hexahedron => "hexahedron".to_string(),
        CellType::Wedge => "wedge".to_string(),
        CellType::Pyramid => "pyramid".to_string(),
        CellType::QuadraticEdge => "quadratic_edge".to_string(),
        CellType::QuadraticTriangle => "quadratic_triangle".to_string(),
        CellType::QuadraticQuad => "quadratic_quad".to_string(),
        CellType::QuadraticTetra => "quadratic_tetra".to_string(),
        CellType::QuadraticHexahedron => "quadratic_hexahedron".to_string(),
        CellType::QuadraticWedge => "quadratic_wedge".to_string(),
        CellType::QuadraticPyramid => "quadratic_pyramid".to_string(),
        CellType::Polyhedron => "polyhedron".to_string(),
        CellType::LagrangeCurve { order } => format!("lagrange_curve_p{order}"),
        CellType::LagrangeTriangle { order } => format!("lagrange_triangle_p{order}"),
        CellType::LagrangeQuadrilateral { order } => format!("lagrange_quad_p{order}"),
        CellType::LagrangeTetrahedron { order } => format!("lagrange_tetra_p{order}"),
        CellType::LagrangeHexahedron { order } => format!("lagrange_hexahedron_p{order}"),
        CellType::LagrangeWedge { order } => format!("lagrange_wedge_p{order}"),
    }
}

const fn mesh_format_label(format: MeshFormat) -> &'static str {
    match format {
        MeshFormat::Vtu => "vtu",
        MeshFormat::Pvd => "pvd",
        MeshFormat::VtkLegacy => "vtk",
        MeshFormat::Gmsh => "msh",
    }
}

const fn scalar_type_label(st: ScalarType) -> &'static str {
    match st {
        ScalarType::F32 => "f32",
        ScalarType::F64 => "f64",
        ScalarType::I32 => "i32",
        ScalarType::I64 => "i64",
        ScalarType::U8 => "u8",
    }
}

// --- `validate` command ---

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ValidateJsonOutput {
    valid: bool,
    issues: Vec<ValidationIssue>,
}

fn cmd_validate(args: &[String]) -> Result<String, CliFailure> {
    let mut json_flag = false;
    let mut scene_arg: Option<&str> = None;

    for arg in args {
        match arg.as_str() {
            "--help" | "-h" => {
                return Ok("Usage: tetraglass validate <scene.json> [--json]".to_string());
            }
            "--json" => json_flag = true,
            other if other.starts_with('-') => {
                return Err(CliFailure::Message(format!(
                    "unknown option `{other}` for `validate`"
                )));
            }
            path => {
                if scene_arg.is_some() {
                    return Err(CliFailure::Message(
                        "`validate` accepts a single <scene.json> argument".to_string(),
                    ));
                }
                scene_arg = Some(path);
            }
        }
    }

    let Some(scene_path_str) = scene_arg else {
        return Err(CliFailure::Message(
            "missing <scene.json> argument; usage: `tetraglass validate <scene.json> [--json]`"
                .to_string(),
        ));
    };

    let scene_path = Path::new(scene_path_str);
    let scene_bytes = std::fs::read(scene_path).map_err(|e| {
        CliFailure::Message(format!("cannot read scene `{}`: {e}", scene_path.display()))
    })?;

    let scene = match Scene::from_json_bytes(&scene_bytes) {
        Ok(s) => s,
        Err(err) => {
            if json_flag {
                let out = ValidateJsonOutput {
                    valid: false,
                    issues: vec![ValidationIssue::new("<root>", err.to_string(), None)],
                };
                return Err(CliFailure::Validation {
                    output: serde_json::to_string_pretty(&out).unwrap_or_default(),
                    is_json_stdout: true,
                });
            }
            return Err(CliFailure::Validation {
                output: err.to_string(),
                is_json_stdout: false,
            });
        }
    };

    let base_dir = scene_path.parent().unwrap_or_else(|| Path::new("."));
    let mut metadata_map: BTreeMap<String, DatasetMetadata> = BTreeMap::new();
    let mut io_issues: Vec<ValidationIssue> = Vec::new();

    for (name, source) in &scene.sources {
        let resolved_path = resolve_relative_path(base_dir, &source.file);
        let explicit_format = source.format.map(MeshFormat::from);
        match load_mesh_from_path(&resolved_path, explicit_format) {
            Ok((_, grid)) => {
                metadata_map.insert(name.clone(), DatasetMetadata::from_grid(&grid));
            }
            Err(CliFailure::Message(msg)) => {
                io_issues.push(ValidationIssue::new(
                    format!("sources.{name}.file"),
                    msg,
                    None,
                ));
            }
            Err(CliFailure::Validation { output, .. }) => {
                io_issues.push(ValidationIssue::new(
                    format!("sources.{name}.file"),
                    output,
                    None,
                ));
            }
        }
    }

    // Run semantic validation: if all sources loaded, validate with metadata;
    // otherwise run structural validation and combine with file-loading issues.
    let mut all_issues = io_issues;
    if all_issues.is_empty() {
        if let Err(report) = validate_scene_with_metadata(&scene, &metadata_map) {
            all_issues.extend(report.issues);
        }
    } else if let Err(report) = scene.validate_structure() {
        all_issues.extend(report.issues);
    }

    if all_issues.is_empty() {
        if json_flag {
            let out = ValidateJsonOutput {
                valid: true,
                issues: Vec::new(),
            };
            serde_json::to_string_pretty(&out).map_err(|e| CliFailure::Message(e.to_string()))
        } else {
            Ok(format!("ok: {scene_path_str} is valid"))
        }
    } else {
        let report = ValidationReport { issues: all_issues };
        if json_flag {
            let out = ValidateJsonOutput {
                valid: false,
                issues: report.issues,
            };
            Err(CliFailure::Validation {
                output: serde_json::to_string_pretty(&out).unwrap_or_default(),
                is_json_stdout: true,
            })
        } else {
            Err(CliFailure::Validation {
                output: report.to_string(),
                is_json_stdout: false,
            })
        }
    }
}

// --- `render` command ---

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RenderRepresentationSummary {
    input: String,
    style: RepresentationStyle,
    num_points: usize,
    num_faces: usize,
    num_triangles: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    color_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    association: Option<ArrayAssociation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    colormap: Option<ColormapName>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data_range: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effective_range: Option<[f64; 2]>,
    log_scale: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RenderCameraSummary {
    fit: String,
    bounds: Option<InspectBounds>,
    azimuth: f64,
    elevation: f64,
    roll: f64,
    zoom: f64,
    view_angle: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RenderSummary {
    image: String,
    size: [u32; 2],
    msaa: u32,
    engine: RenderEngine,
    representations: Vec<RenderRepresentationSummary>,
    camera: RenderCameraSummary,
}

fn cmd_render(args: &[String]) -> Result<String, CliFailure> {
    let mut scene_arg: Option<&str> = None;
    let mut output_override: Option<&str> = None;
    let mut engine_override: Option<RenderEngine> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                return Ok(
                    "Usage: tetraglass render <scene.json> [-o <out.png>] [--engine <raster|cycles>]"
                        .to_string(),
                );
            }
            "-o" | "--output" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    return Err(CliFailure::Message(
                        "missing output path after `-o` / `--output`".to_string(),
                    ));
                };
                output_override = Some(val.as_str());
            }
            "--engine" => {
                i += 1;
                let Some(val) = args.get(i) else {
                    return Err(CliFailure::Message(
                        "missing engine name after `--engine` (expected `raster` or `cycles`)"
                            .to_string(),
                    ));
                };
                engine_override = match val.as_str() {
                    "raster" => Some(RenderEngine::Raster),
                    "cycles" => Some(RenderEngine::Cycles),
                    other => {
                        return Err(CliFailure::Message(format!(
                            "unknown render engine `{other}`; expected `raster` or `cycles`"
                        )));
                    }
                };
            }
            other if other.starts_with('-') => {
                return Err(CliFailure::Message(format!(
                    "unknown option `{other}` for `render`"
                )));
            }
            path => {
                if scene_arg.is_some() {
                    return Err(CliFailure::Message(
                        "`render` accepts a single <scene.json> argument".to_string(),
                    ));
                }
                scene_arg = Some(path);
            }
        }
        i += 1;
    }

    let Some(scene_path_str) = scene_arg else {
        return Err(CliFailure::Message(
            "missing <scene.json> argument; usage: `tetraglass render <scene.json> [-o <out.png>]`"
                .to_string(),
        ));
    };

    let scene_path = Path::new(scene_path_str);
    let scene = Scene::from_path(scene_path).map_err(|e| {
        CliFailure::Message(format!(
            "cannot parse scene `{}`: {e}",
            scene_path.display()
        ))
    })?;

    let base_dir = scene_path.parent().unwrap_or_else(|| Path::new("."));

    // Determine output path, dimensions, MSAA, and engine.
    let output_path: PathBuf = if let Some(override_str) = output_override {
        PathBuf::from(override_str)
    } else if let Some(out_cfg) = &scene.output {
        resolve_relative_path(base_dir, &out_cfg.image)
    } else {
        return Err(CliFailure::Message(
            "scene has no `output` section and `-o <out.png>` was not provided".to_string(),
        ));
    };

    let size = scene
        .output
        .as_ref()
        .map_or(DEFAULT_OUTPUT_SIZE, |o| o.size);
    let msaa = scene
        .output
        .as_ref()
        .map_or(DEFAULT_MSAA_SAMPLES, |o| o.msaa);
    let engine = engine_override.unwrap_or_else(|| {
        scene
            .output
            .as_ref()
            .map_or(RenderEngine::Raster, |o| o.engine)
    });

    if engine == RenderEngine::Cycles {
        return Err(CliFailure::Message(
            "Cycles rendering arrives in milestone M3 (use `--engine raster`)".to_string(),
        ));
    }

    // Load all sources referenced in the scene.
    let mut evaluator = SceneEvaluator::new();
    for (name, source) in &scene.sources {
        let resolved_path = resolve_relative_path(base_dir, &source.file);
        let explicit_format = source.format.map(MeshFormat::from);
        let (_, grid) = load_mesh_from_path(&resolved_path, explicit_format)?;
        evaluator.insert_source(name.clone(), grid);
    }

    let evaluated = evaluator
        .evaluate_scene(&scene)
        .map_err(|e| CliFailure::Message(e.to_string()))?;

    let bounds = evaluated.camera_bounds.unwrap_or(BoundingBox {
        min: [-1.0, -1.0, -1.0],
        max: [1.0, 1.0, 1.0],
    });
    let camera = build_camera_from_scene(&scene, bounds);
    let background_srgb = resolve_background_srgb(&scene.view.background);

    // Render offscreen via wgpu HeadlessContext.
    let ctx = HeadlessContext::new().map_err(|e| CliFailure::Message(e.to_string()))?;
    let renderer = Renderer::new(ctx.device(), ctx.queue(), DEFAULT_COLOR_FORMAT, msaa)
        .map_err(|e| CliFailure::Message(e.to_string()))?;
    let target = OffscreenTarget::new(ctx.device(), size[0], size[1], DEFAULT_COLOR_FORMAT, msaa)
        .map_err(|e| CliFailure::Message(e.to_string()))?;

    let dataset_center = bounds.center();
    let mut gpu_surfaces: Vec<GpuSurface> = Vec::with_capacity(evaluated.representations.len());
    for rep in &evaluated.representations {
        let style = build_surface_style(rep);
        let gpu_surf = renderer
            .upload_surface_with_center(ctx.device(), &rep.surface, dataset_center, &style)
            .map_err(|e| CliFailure::Message(e.to_string()))?;
        gpu_surfaces.push(gpu_surf);
    }

    let gpu_refs: Vec<&GpuSurface> = gpu_surfaces.iter().collect();
    let annotations = build_annotations_from_scene(&scene, &evaluated);
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    renderer
        .render_with_annotations(
            ctx.device(),
            &mut encoder,
            ctx.queue(),
            &target,
            &camera,
            background_srgb,
            &gpu_refs,
            &annotations,
        )
        .map_err(|e| CliFailure::Message(e.to_string()))?;
    ctx.queue().submit(std::iter::once(encoder.finish()));

    let image = ctx
        .readback_target(&target)
        .map_err(|e| CliFailure::Message(e.to_string()))?;
    image
        .save_png(&output_path)
        .map_err(|e| CliFailure::Message(e.to_string()))?;

    let rep_summaries: Vec<RenderRepresentationSummary> = evaluated
        .representations
        .iter()
        .map(|rep| RenderRepresentationSummary {
            input: rep.input.clone(),
            style: rep.style,
            num_points: rep.surface.num_points(),
            num_faces: rep.surface.num_faces(),
            num_triangles: rep.surface.num_triangles(),
            color_by: rep.coloring.as_ref().map(|c| c.raw_ref.clone()),
            association: rep.coloring.as_ref().map(|c| c.association),
            colormap: rep.coloring.as_ref().map(|c| c.colormap),
            data_range: rep
                .coloring
                .as_ref()
                .map(|c| [c.data_range.min, c.data_range.max]),
            effective_range: rep.coloring.as_ref().map(|c| c.effective_range),
            log_scale: rep.coloring.as_ref().is_some_and(|c| c.log_scale),
        })
        .collect();

    let summary = RenderSummary {
        image: output_path.display().to_string(),
        size,
        msaa,
        engine,
        representations: rep_summaries,
        camera: RenderCameraSummary {
            fit: scene.view.camera.fit.clone(),
            bounds: evaluated.camera_bounds.map(|b| InspectBounds {
                min: b.min,
                max: b.max,
            }),
            azimuth: camera.azimuth_deg,
            elevation: camera.elevation_deg,
            roll: camera.roll_deg,
            zoom: camera.zoom,
            view_angle: camera.view_angle_deg,
        },
    };

    serde_json::to_string_pretty(&summary).map_err(|e| CliFailure::Message(e.to_string()))
}

fn build_annotations_from_scene(scene: &Scene, evaluated: &EvaluatedScene) -> Annotations {
    let mut colorbars = Vec::new();
    for rep in &evaluated.representations {
        if !rep.colorbar.is_enabled() {
            continue;
        }
        let Some(coloring) = &rep.coloring else {
            continue;
        };
        let (title, position) = match &rep.colorbar {
            ColorbarSpec::Enabled(_) => (Some(coloring.raw_ref.clone()), ColorbarPlacement::Right),
            ColorbarSpec::Options(opts) => (
                opts.title
                    .clone()
                    .or_else(|| Some(coloring.raw_ref.clone())),
                ColorbarPlacement::from(opts.position),
            ),
        };
        colorbars.push(ColorbarAnnotation {
            colormap: map_colormap_kind(coloring.colormap),
            range: coloring.effective_range,
            use_log_scale: coloring.log_scale,
            title,
            position,
            tick_count: DEFAULT_COLORBAR_TICKS,
        });
    }

    Annotations {
        title: scene.view.title.clone(),
        axes: scene.view.axes,
        colorbars,
    }
}

fn build_camera_from_scene(scene: &Scene, bounds: BoundingBox) -> Camera {
    let cam_cfg = &scene.view.camera;
    let mut camera = Camera::fit_bounds(bounds, cam_cfg.azimuth, cam_cfg.elevation);
    camera.roll_deg = cam_cfg.roll;
    camera.zoom = cam_cfg.zoom;
    camera.view_angle_deg = cam_cfg.view_angle;

    let center = bounds.center();
    if let Some(fp) = cam_cfg.focal_point {
        camera.focal_point = [fp[0] - center[0], fp[1] - center[1], fp[2] - center[2]];
    }
    if let Some(pos) = cam_cfg.position {
        let rel_pos = [pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]];
        let dx = rel_pos[0] - camera.focal_point[0];
        let dy = rel_pos[1] - camera.focal_point[1];
        let dz = rel_pos[2] - camera.focal_point[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-5);
        let el = (dy / dist).clamp(-1.0, 1.0).asin().to_degrees();
        let az = dx.atan2(dz).to_degrees();
        camera.distance = dist;
        camera.zoom = 1.0;
        camera.azimuth_deg = az;
        camera.elevation_deg = el.clamp(-89.0, 89.0);
    }
    camera
}

fn resolve_background_srgb(bg: &BackgroundSpec) -> [f32; 4] {
    match bg {
        BackgroundSpec::Solid(color) => {
            let [r, g, b] = color.resolve_rgb().unwrap_or([1.0, 1.0, 1.0]);
            [r, g, b, 1.0]
        }
        BackgroundSpec::Gradient(grad) => {
            let top = grad.top.resolve_rgb().unwrap_or([1.0, 1.0, 1.0]);
            let bottom = grad.bottom.resolve_rgb().unwrap_or([1.0, 1.0, 1.0]);
            [
                0.5 * (top[0] + bottom[0]),
                0.5 * (top[1] + bottom[1]),
                0.5 * (top[2] + bottom[2]),
                1.0,
            ]
        }
    }
}

fn build_surface_style(rep: &EvaluatedRepresentation) -> SurfaceStyle {
    let solid_color_srgb = [rep.color[0], rep.color[1], rep.color[2], rep.opacity as f32];
    match &rep.coloring {
        None => SurfaceStyle {
            color_by: ColorBy::Solid,
            colormap: ColormapKind::Viridis,
            range: None,
            use_log_scale: false,
            solid_color_srgb,
        },
        Some(c) => SurfaceStyle {
            color_by: build_color_by(c),
            colormap: map_colormap_kind(c.colormap),
            range: Some(c.effective_range),
            use_log_scale: c.log_scale,
            solid_color_srgb,
        },
    }
}

fn build_color_by(coloring: &EvaluatedColoring) -> ColorBy {
    let component = match coloring.component {
        ResolvedComponent::Magnitude => ScalarComponent::Magnitude,
        ResolvedComponent::Component(0) => ScalarComponent::First,
        ResolvedComponent::Component(idx) => ScalarComponent::Index(idx),
    };
    match coloring.association {
        ArrayAssociation::Point => ColorBy::Point {
            name: coloring.array_name.clone(),
            component,
        },
        ArrayAssociation::Cell => ColorBy::Cell {
            name: coloring.array_name.clone(),
            component,
        },
    }
}

const fn map_colormap_kind(name: ColormapName) -> ColormapKind {
    match name {
        ColormapName::Viridis | ColormapName::Turbo => ColormapKind::Viridis,
        ColormapName::Cividis | ColormapName::Grayscale => ColormapKind::Cividis,
        ColormapName::CoolWarm => ColormapKind::CoolWarm,
    }
}

fn resolve_relative_path(base_dir: &Path, file: &str) -> PathBuf {
    let p = Path::new(file);
    if p.is_absolute() || base_dir.as_os_str().is_empty() {
        p.to_path_buf()
    } else {
        base_dir.join(p)
    }
}

fn load_mesh_from_path(
    path: &Path,
    explicit_format: Option<MeshFormat>,
) -> Result<(MeshFormat, UnstructuredGrid), CliFailure> {
    let bytes = std::fs::read(path)
        .map_err(|e| CliFailure::Message(format!("cannot read `{}`: {e}", path.display())))?;
    let format = explicit_format
        .or_else(|| {
            path.extension()
                .and_then(|e| e.to_str())
                .and_then(detect_format_from_extension)
        })
        .or_else(|| detect_format_from_bytes(&bytes))
        .ok_or_else(|| {
            CliFailure::Message(format!(
                "cannot determine mesh format for `{}`",
                path.display()
            ))
        })?;

    match format {
        MeshFormat::Vtu => {
            let grid = read_vtu(&bytes).map_err(|e| {
                CliFailure::Message(format!("failed to parse `{}`: {e}", path.display()))
            })?;
            Ok((format, grid))
        }
        other => Err(CliFailure::Message(format!(
            "format `{}` is not yet supported in this milestone (`{}`)",
            mesh_format_label(other),
            path.display()
        ))),
    }
}

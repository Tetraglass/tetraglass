//! Umbrella crate for **Tetraglass**, a Rust-native scientific visualization library for
//! FEM/CFD results on unstructured meshes (`D-017`).
//!
//! `tetraglass` re-exports the workspace sub-crates behind feature flags and provides a unified
//! [`prelude`] and top-level access to core data types, file readers, visualization filters,
//! the declarative [`Scene`] model and [`SceneBuilder`], `wgpu` rendering, and the `egui`
//! viewport widget.
//!
//! # Feature flags
//!
//! | Feature | Default | Re-exported module | Description |
//! | --- | --- | --- | --- |
//! | *(always on)* | Yes | [`core`] | [`UnstructuredGrid`], [`SurfaceMesh`], [`TetView`], [`RegularGrid`], [`DataArray`], [`TimeSeries`] |
//! | `io` | Yes | [`io`] | Pure-Rust byte-buffer readers for `.vtu`, `.vtk`, `.pvd`, and `.msh` plus `.vtu` writer |
//! | `filters` | Yes | [`filters`] | `linearize`, `extract_external_surface`, `slice`, `contour`, `clip`, `threshold`, `streamlines`, `glyphs`, `resample` |
//! | `scene` | Yes | [`scene`] | Declarative [`Scene`] model, [`SceneBuilder`], JSON Schema, validation, and cached [`SceneEvaluator`] |
//! | `export` | Yes | [`export`] | Binary glTF 2.0 (`.glb`) and renderer-neutral Blender bundle (`.tgbin`) exporters |
//! | `render` | Yes | [`render`] | Embeddable `wgpu` renderer, colormaps, annotations, SSAO, progressive path tracer, and `headless` helper |
//! | `egui` | Yes | [`egui`] | Embeddable `ViewportState` and `ViewportWidget` for `egui-wgpu` applications |
//! | `viewer` | No | `viewer` | Full `eframe` desktop and WebGPU browser viewer application |
//! | `full` | No | all of the above | Enables every optional feature including `viewer` |
//!
//! # Examples
//!
//! Constructing a visualization scene with the fluent Rust builder API (`Scene::builder()`)
//! and evaluating it against a synthetic hexahedral mesh:
//!
//! ```
//! use tetraglass::prelude::*;
//!
//! // 1. Generate a small test mesh in memory
//! let grid = SyntheticGridSpec::new(4, 4, 4, SyntheticCellLayout::Hexahedra).generate()?;
//!
//! // 2. Build a declarative scene with a plane slice and an isosurface contour
//! let mut builder = Scene::builder();
//! let flow = builder.source("flow", Source::file("synthetic.vtu"));
//! let cut = builder.filter("cut", SliceFilter::new(&flow, [0.0, 0.0, 1.0]));
//! let iso = builder.filter("iso", ContourFilter::new(&flow, "linear", [3.0]));
//!
//! builder
//!     .show(&cut)
//!     .color_by("velocity")
//!     .magnitude()
//!     .colormap(ColormapName::Viridis)
//!     .colorbar(true);
//! builder.show(&iso).opacity(0.4);
//! builder.camera().fit("all").orbit(30.0, 20.0);
//!
//! let scene = builder.build_validated()?;
//!
//! // 3. Evaluate the scene lazily with input-hash caching
//! let mut evaluator = SceneEvaluator::new();
//! evaluator.insert_source("flow", grid);
//! let evaluated = evaluator.evaluate_scene(&scene)?;
//! assert_eq!(evaluated.representations.len(), 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub use tetraglass_core as core;

#[cfg(feature = "io")]
pub use tetraglass_io as io;

#[cfg(feature = "filters")]
pub use tetraglass_filters as filters;

#[cfg(feature = "scene")]
pub use tetraglass_scene as scene;

#[cfg(feature = "export")]
pub use tetraglass_export as export;

#[cfg(feature = "render")]
pub use tetraglass_render as render;

#[cfg(feature = "egui")]
pub use tetraglass_egui as egui;

#[cfg(feature = "viewer")]
pub use tetraglass_viewer as viewer;

pub use tetraglass_core::{
    Attributes, BoundingBox, CellType, ComponentCount, CoreError, DataArray, PolyhedronFaces,
    PolylineSet, RegularGrid, SurfaceMesh, SyntheticCellLayout, SyntheticGridSpec, TetView,
    TimeSeries, UnstructuredGrid, ValueRange, generate_synthetic_grid,
};

#[cfg(feature = "io")]
pub use tetraglass_io::{
    IoError, MeshFormat, detect_format_from_bytes, detect_format_from_extension, read_msh,
    read_vtk, read_vtu, write_vtu_to_bytes,
};

#[cfg(feature = "filters")]
pub use tetraglass_filters::{
    ClipParams, ContourParams, FilterError, GlyphParams, LinearizeParams, ResampleParams,
    SliceParams, StreamlineParams, TetLocator, ThresholdParams, clip, contour,
    extract_external_surface, glyphs, linearize, linearize_default, resample, slice, streamlines,
    threshold,
};

#[cfg(feature = "scene")]
pub use tetraglass_scene::{
    BackgroundSpec, CameraBuilder, CameraConfig, ClipFilter, ClipKeep, ClipMode, ColorSpec,
    ColorbarPosition, ColorbarSpec, ColormapName, ContourFilter, DatasetRef, EvaluatedScene,
    Filter, GlyphsFilter, LightingPreset, Output, OutputBuilder, ProjectionKind, RangeSpec,
    RenderEngine, Representation, RepresentationBuilder, RepresentationStyle, ResampleFilter,
    SCENE_VERSION, Scene, SceneBuilder, SceneError, SceneEvaluator, SliceFilter, Source,
    SourceFormat, StreamlineDirectionSpec, StreamlineIntegratorSpec, StreamlineSeedsSpec,
    StreamlinesFilter, SurfaceFilter, ThresholdFilter, TimeSelector, View, scene_json_schema,
    scene_json_schema_string,
};

#[cfg(feature = "export")]
pub use tetraglass_export::{
    BundleArchive, BundleManifest, ExportError, export_evaluated_scene_to_bundle,
    export_evaluated_scene_to_glb,
};

#[cfg(feature = "render")]
pub use tetraglass_render::{
    Annotations, Camera, ColorBy, ColorbarAnnotation, ColormapKind, GpuSurface, HeadlessContext,
    OffscreenTarget, PathTraceConfig, PathTracer, PickTarget, RenderError, Renderer,
    ShaderClipKeep, ShaderClipPlane, SsaoConfig, SurfaceStyle,
};

#[cfg(feature = "egui")]
pub use tetraglass_egui::{ViewportResponse, ViewportState, ViewportWidget};

/// Convenience prelude re-exporting the most commonly used Tetraglass types and builders.
pub mod prelude {
    pub use crate::{
        Attributes, BoundingBox, CellType, ComponentCount, DataArray, PolylineSet, RegularGrid,
        SurfaceMesh, SyntheticCellLayout, SyntheticGridSpec, TetView, TimeSeries, UnstructuredGrid,
        ValueRange,
    };

    #[cfg(feature = "io")]
    pub use crate::{MeshFormat, read_msh, read_vtk, read_vtu, write_vtu_to_bytes};

    #[cfg(feature = "filters")]
    pub use crate::{
        TetLocator, clip, contour, extract_external_surface, glyphs, linearize, linearize_default,
        resample, slice, streamlines, threshold,
    };

    #[cfg(feature = "scene")]
    pub use crate::{
        CameraBuilder, ClipFilter, ClipKeep, ClipMode, ColorSpec, ColorbarPosition, ColorbarSpec,
        ColormapName, ContourFilter, DatasetRef, Filter, GlyphsFilter, LightingPreset,
        OutputBuilder, ProjectionKind, RangeSpec, RenderEngine, RepresentationBuilder,
        RepresentationStyle, ResampleFilter, Scene, SceneBuilder, SceneEvaluator, SliceFilter,
        Source, SourceFormat, StreamlineDirectionSpec, StreamlineIntegratorSpec,
        StreamlineSeedsSpec, StreamlinesFilter, SurfaceFilter, ThresholdFilter, TimeSelector,
    };

    #[cfg(feature = "render")]
    pub use crate::{
        Annotations, Camera, ColorBy, ColorbarAnnotation, ColormapKind, HeadlessContext,
        OffscreenTarget, PathTraceConfig, PathTracer, Renderer, ShaderClipKeep, ShaderClipPlane,
        SsaoConfig, SurfaceStyle,
    };

    #[cfg(feature = "egui")]
    pub use crate::{ViewportState, ViewportWidget};
}

//! Error types for `tetraglass-render`.

use tetraglass_core::CoreError;
use thiserror::Error;

/// Which mesh attribute dictionary a field lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldAssociation {
    /// Per-vertex (`point_data`) attribute.
    Point,
    /// Per-cell / per-triangle (`cell_data`) attribute.
    Cell,
}

impl std::fmt::Display for FieldAssociation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Point => f.write_str("point"),
            Self::Cell => f.write_str("cell"),
        }
    }
}

/// Errors returned by the `tetraglass-render` crate.
#[derive(Debug, Error)]
pub enum RenderError {
    /// Error from `tetraglass-core`.
    #[error(transparent)]
    Core(#[from] CoreError),

    /// Failed to find a compatible `wgpu` adapter.
    #[error("failed to request a compatible wgpu adapter: {0}")]
    RequestAdapter(#[from] wgpu::RequestAdapterError),

    /// Failed to create a `wgpu` device and queue.
    #[error("failed to request wgpu device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),

    /// Failed to map a GPU staging buffer for CPU readback.
    #[error("failed to map GPU buffer: {0}")]
    BufferAsync(#[from] wgpu::BufferAsyncError),

    /// Failed to obtain a mapped range view from a GPU buffer.
    #[error("failed to read mapped GPU buffer range: {0}")]
    MapRange(#[from] wgpu::MapRangeError),

    /// Polling the `wgpu` device failed during readback.
    #[error("failed to poll wgpu device: {0}")]
    Poll(#[from] wgpu::PollError),

    /// Requested attribute array does not exist on the surface mesh.
    #[error("{association} data array `{name}` not found on surface mesh")]
    MissingArray {
        /// Name of the requested attribute array.
        name: String,
        /// Whether point or cell data was requested.
        association: FieldAssociation,
    },

    /// Requested component index exceeds the array's component count.
    #[error(
        "component index {component} out of bounds for array `{array}` with {available} components"
    )]
    InvalidComponent {
        /// Name of the attribute array.
        array: String,
        /// Requested zero-based component index.
        component: usize,
        /// Number of components available in the array.
        available: usize,
    },

    /// Render target dimensions must be non-zero.
    #[error("invalid render target dimensions {width}x{height} (both must be > 0)")]
    InvalidDimensions {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
    },

    /// Unsupported MSAA sample count (WebGPU guarantees 1 and 4).
    #[error("unsupported MSAA sample count {sample_count} (expected 1 or 4)")]
    UnsupportedSampleCount {
        /// Requested MSAA sample count.
        sample_count: u32,
    },

    /// Target texture format does not match the renderer's configured format.
    #[error("render target format {actual:?} does not match renderer format {expected:?}")]
    FormatMismatch {
        /// Texture format the renderer pipeline was compiled for.
        expected: wgpu::TextureFormat,
        /// Texture format of the provided target.
        actual: wgpu::TextureFormat,
    },

    /// PNG encoding or decoding error.
    #[error("PNG error: {0}")]
    Png(String),

    /// Image dimensions did not match during visual comparison.
    #[error(
        "image dimensions mismatch: actual {actual_width}x{actual_height} vs baseline {baseline_width}x{baseline_height}"
    )]
    DimensionMismatch {
        /// Width of rendered image.
        actual_width: u32,
        /// Height of rendered image.
        actual_height: u32,
        /// Width of baseline image.
        baseline_width: u32,
        /// Height of baseline image.
        baseline_height: u32,
    },

    /// Standard I/O error when reading or writing PNG files.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

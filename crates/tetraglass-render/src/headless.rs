//! Headless `wgpu` context, offscreen PNG readback, and visual regression comparison.
//!
//! Per `D-002` and `.agents/rules/rendering.md`, this module is the only place in
//! `tetraglass-render` that creates a `wgpu::Instance`, `Adapter`, `Device`, and `Queue`.

use std::future::Future;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

use tetraglass_core::{BoundingBox, SurfaceMesh};

use crate::annotations::Annotations;
use crate::camera::Camera;
use crate::error::RenderError;
use crate::pick::{PickTarget, SurfacePickResult, probe_surface_triangle};
use crate::pipeline::{DEFAULT_COLOR_FORMAT, OffscreenTarget, Renderer, SurfaceStyle};
use crate::png::{decode_rgba8_png, encode_rgba8_png};

/// Per-channel tolerance for image regression tests (`.agents/rules/testing.md`: `<= 2` of `255`).
pub const DEFAULT_CHANNEL_TOLERANCE: u8 = 2;

/// Maximum fraction of differing pixels before an image regression test fails (`0.75%`, `D-015`).
pub const DEFAULT_MAX_DIFF_FRACTION: f64 = 0.0075;

/// Options for one-shot headless surface rendering.
#[derive(Debug, Clone)]
pub struct HeadlessRenderOptions {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// MSAA sample count (`1` or `4`, default `4`).
    pub sample_count: u32,
    /// Background clear color in sRGB `[r, g, b, a]` (default white `[1.0, 1.0, 1.0, 1.0]`).
    pub background_srgb: [f32; 4],
    /// Surface coloring and colormap settings.
    pub style: SurfaceStyle,
    /// Explicit camera, or `None` to fit the surface bounds at `azimuth = 30°, elevation = 20°`.
    pub camera: Option<Camera>,
    /// Viewport overlay annotations (title text, colorbars, and orientation axes triad).
    pub annotations: Annotations,
}

impl Default for HeadlessRenderOptions {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            sample_count: 4,
            background_srgb: [1.0, 1.0, 1.0, 1.0],
            style: SurfaceStyle::default(),
            camera: None,
            annotations: Annotations::default(),
        }
    }
}

/// An 8-bit sRGB `RGBA` image in CPU memory read back from an [`OffscreenTarget`] or decoded from PNG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedImage {
    width: u32,
    height: u32,
    rgba8: Vec<u8>,
}

impl RenderedImage {
    /// Constructs a [`RenderedImage`] from raw tightly-packed `RGBA8` bytes.
    pub fn new(width: u32, height: u32, rgba8: Vec<u8>) -> Result<Self, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidDimensions { width, height });
        }
        let expected = (width as usize) * (height as usize) * 4;
        if rgba8.len() != expected {
            return Err(RenderError::Png(format!(
                "expected {expected} RGBA8 bytes for {width}x{height} image, got {}",
                rgba8.len()
            )));
        }
        Ok(Self {
            width,
            height,
            rgba8,
        })
    }

    /// Returns the image width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns the image height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Returns the tightly-packed sRGB `RGBA8` byte slice (`width * height * 4` bytes).
    #[must_use]
    pub fn as_rgba8(&self) -> &[u8] {
        &self.rgba8
    }

    /// Returns the `[r, g, b, a]` pixel at `(x, y)` (`(0, 0)` is top-left).
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let idx = ((y * self.width + x) as usize) * 4;
        [
            self.rgba8[idx],
            self.rgba8[idx + 1],
            self.rgba8[idx + 2],
            self.rgba8[idx + 3],
        ]
    }

    /// Encodes the image as an 8-bit `RGBA` PNG in memory.
    pub fn encode_png(&self) -> Result<Vec<u8>, RenderError> {
        encode_rgba8_png(self.width, self.height, &self.rgba8)
    }

    /// Decodes a [`RenderedImage`] from PNG bytes in memory.
    pub fn decode_png(bytes: &[u8]) -> Result<Self, RenderError> {
        let (width, height, rgba8) = decode_rgba8_png(bytes)?;
        Ok(Self {
            width,
            height,
            rgba8,
        })
    }

    /// Writes the image to `path` as a PNG file, creating parent directories if needed.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), RenderError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let png_bytes = self.encode_png()?;
        std::fs::write(path, png_bytes)?;
        Ok(())
    }

    /// Loads a [`RenderedImage`] from a PNG file on disk.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_png(path: impl AsRef<Path>) -> Result<Self, RenderError> {
        let bytes = std::fs::read(path)?;
        Self::decode_png(&bytes)
    }

    /// Compares `self` against `baseline` using the default `.agents/rules/testing.md` policy
    /// (`channel_tolerance = 2`, `max_diff_fraction = 0.0075`).
    pub fn compare_with_baseline(&self, baseline: &Self) -> Result<ImageComparison, RenderError> {
        self.compare(
            baseline,
            DEFAULT_CHANNEL_TOLERANCE,
            DEFAULT_MAX_DIFF_FRACTION,
        )
    }

    /// Compares `self` against `baseline` with explicit tolerances and builds a diff image.
    pub fn compare(
        &self,
        baseline: &Self,
        channel_tolerance: u8,
        max_diff_fraction: f64,
    ) -> Result<ImageComparison, RenderError> {
        if self.width != baseline.width || self.height != baseline.height {
            return Err(RenderError::DimensionMismatch {
                actual_width: self.width,
                actual_height: self.height,
                baseline_width: baseline.width,
                baseline_height: baseline.height,
            });
        }

        let total_pixels = (self.width as usize) * (self.height as usize);
        let mut differing_pixels = 0usize;
        let mut max_channel_diff = 0u8;
        let mut diff_rgba = Vec::with_capacity(self.rgba8.len());

        for (actual_px, base_px) in self
            .rgba8
            .as_chunks::<4>()
            .0
            .iter()
            .zip(baseline.rgba8.as_chunks::<4>().0.iter())
        {
            let dr = actual_px[0].abs_diff(base_px[0]);
            let dg = actual_px[1].abs_diff(base_px[1]);
            let db = actual_px[2].abs_diff(base_px[2]);
            let da = actual_px[3].abs_diff(base_px[3]);
            let px_diff = dr.max(dg).max(db).max(da);
            max_channel_diff = max_channel_diff.max(px_diff);

            if px_diff > channel_tolerance {
                differing_pixels += 1;
                // Highlight differing pixel in bright magenta
                diff_rgba.extend_from_slice(&[255, 0, 255, 255]);
            } else {
                // Dimmed grayscale context for matching pixels
                let luma = ((u16::from(actual_px[0]) * 77
                    + u16::from(actual_px[1]) * 150
                    + u16::from(actual_px[2]) * 29)
                    >> 9) as u8;
                diff_rgba.extend_from_slice(&[luma, luma, luma, 255]);
            }
        }

        let diff_fraction = if total_pixels > 0 {
            (differing_pixels as f64) / (total_pixels as f64)
        } else {
            0.0
        };
        let passed = diff_fraction <= max_diff_fraction;

        Ok(ImageComparison {
            passed,
            differing_pixels,
            total_pixels,
            diff_fraction,
            max_channel_diff,
            diff_image: Self {
                width: self.width,
                height: self.height,
                rgba8: diff_rgba,
            },
        })
    }
}

/// Result of comparing a rendered image against a baseline image (`/render-check`).
#[derive(Debug, Clone)]
pub struct ImageComparison {
    /// Whether `diff_fraction <= max_diff_fraction`.
    pub passed: bool,
    /// Number of pixels where any channel differed by more than `channel_tolerance`.
    pub differing_pixels: usize,
    /// Total number of pixels in the image.
    pub total_pixels: usize,
    /// Fraction of pixels that differed (`differing_pixels / total_pixels`).
    pub diff_fraction: f64,
    /// Maximum absolute single-channel difference observed across all pixels.
    pub max_channel_diff: u8,
    /// Visualization image with differing pixels highlighted in bright magenta.
    pub diff_image: RenderedImage,
}

/// Headless `wgpu` device and queue context for CLI rendering and integration tests.
#[derive(Debug)]
pub struct HeadlessContext {
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl HeadlessContext {
    /// Initializes a headless `wgpu` adapter, device, and queue using WebGPU default limits.
    ///
    /// Respects standard `wgpu` environment variables (`WGPU_BACKEND`, `WGPU_ADAPTER_NAME`)
    /// as well as `TETRAGLASS_FORCE_FALLBACK_ADAPTER=1` to explicitly select software adapters
    /// such as Mesa `lavapipe`.
    pub fn new() -> Result<Self, RenderError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());

        let force_fallback = std::env::var("TETRAGLASS_FORCE_FALLBACK_ADAPTER")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));

        let adapter = if force_fallback {
            block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                force_fallback_adapter: true,
                compatible_surface: None,
                ..Default::default()
            }))?
        } else {
            match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })) {
                Ok(adapter) => adapter,
                Err(_) => block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::None,
                    force_fallback_adapter: true,
                    compatible_surface: None,
                    ..Default::default()
                }))?,
            }
        };

        let adapter_info = adapter.get_info();
        log::debug!(
            "HeadlessContext using adapter `{}` ({:?}, {:?})",
            adapter_info.name,
            adapter_info.backend,
            adapter_info.device_type
        );

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tetraglass_headless_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))?;

        Ok(Self {
            adapter_info,
            device,
            queue,
        })
    }

    /// Returns information about the selected `wgpu` adapter.
    #[must_use]
    pub const fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Returns the headless `wgpu::Device`.
    #[must_use]
    pub const fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Returns the headless `wgpu::Queue`.
    #[must_use]
    pub const fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Renders `surface` offscreen with `options` and reads back the resulting [`RenderedImage`].
    pub fn render_surface(
        &self,
        surface: &SurfaceMesh,
        options: &HeadlessRenderOptions,
    ) -> Result<RenderedImage, RenderError> {
        let renderer = Renderer::new(
            &self.device,
            &self.queue,
            DEFAULT_COLOR_FORMAT,
            options.sample_count,
        )?;
        let target = OffscreenTarget::new(
            &self.device,
            options.width,
            options.height,
            DEFAULT_COLOR_FORMAT,
            options.sample_count,
        )?;
        let gpu_surface = renderer.upload_surface(&self.device, surface, &options.style)?;

        let camera = options.camera.clone().unwrap_or_else(|| {
            let bounds = surface.bounds().unwrap_or(BoundingBox {
                min: [-1.0, -1.0, -1.0],
                max: [1.0, 1.0, 1.0],
            });
            Camera::fit_bounds(bounds, 30.0, 20.0)
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tetraglass_headless_encoder"),
            });
        renderer.render_with_annotations(
            &self.device,
            &mut encoder,
            &self.queue,
            &target,
            &camera,
            options.background_srgb,
            &[&gpu_surface],
            &options.annotations,
        )?;
        self.queue.submit(std::iter::once(encoder.finish()));

        self.readback_target(&target)
    }

    /// Renders `surface` into a GPU ID buffer and probes pixel `(x, y)` (`(0, 0)` top-left).
    pub fn pick_surface(
        &self,
        surface: &SurfaceMesh,
        options: &HeadlessRenderOptions,
        x: u32,
        y: u32,
    ) -> Result<Option<SurfacePickResult>, RenderError> {
        let renderer = Renderer::new(
            &self.device,
            &self.queue,
            DEFAULT_COLOR_FORMAT,
            options.sample_count,
        )?;
        let pick_target = PickTarget::new(&self.device, options.width, options.height)?;
        let gpu_surface = renderer.upload_surface(&self.device, surface, &options.style)?;

        let camera = options.camera.clone().unwrap_or_else(|| {
            let bounds = surface.bounds().unwrap_or(BoundingBox {
                min: [-1.0, -1.0, -1.0],
                max: [1.0, 1.0, 1.0],
            });
            Camera::fit_bounds(bounds, 30.0, 20.0)
        });

        let Some(raw) = renderer.pick_pixel(
            &self.device,
            &self.queue,
            &pick_target,
            &camera,
            &[&gpu_surface],
            x,
            y,
        )?
        else {
            return Ok(None);
        };

        Ok(probe_surface_triangle(
            surface,
            raw.representation_index,
            [x, y],
            [options.width, options.height],
            &camera,
            gpu_surface.dataset_center(),
            raw,
            &options.style.color_by,
        ))
    }

    /// Copies the resolved color attachment of `target` to a CPU [`RenderedImage`].
    pub fn readback_target(&self, target: &OffscreenTarget) -> Result<RenderedImage, RenderError> {
        let width = target.width();
        let height = target.height();
        let unpadded_bytes_per_row = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
        let buffer_size = u64::from(padded_bytes_per_row) * u64::from(height);

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tetraglass_readback_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tetraglass_readback_copy_encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target.color_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let map_result: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> =
            Arc::new(Mutex::new(None));
        let map_result_clone = Arc::clone(&map_result);
        staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| {
                if let Ok(mut slot) = map_result_clone.lock() {
                    *slot = Some(res);
                }
            });

        self.device.poll(wgpu::PollType::wait_indefinitely())?;

        let status = map_result
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
            .unwrap_or(Ok(()));
        status?;

        let mapped = staging_buffer.get_mapped_range(..)?;
        let mut rgba8 = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
        for row in 0..(height as usize) {
            let start = row * (padded_bytes_per_row as usize);
            let end = start + (unpadded_bytes_per_row as usize);
            rgba8.extend_from_slice(&mapped[start..end]);
        }
        drop(mapped);
        staging_buffer.unmap();

        RenderedImage::new(width, height, rgba8)
    }
}

struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(fut: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => thread::park(),
        }
    }
}

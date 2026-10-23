//! Interactive desktop example for the `tetraglass-egui` viewport widget (M0.8).
//!
//! Run with:
//! ```sh
//! cargo run -p tetraglass-egui --example egui_viewport -- testdata/vtk/mixed_linear.vtu
//! ```

use std::path::PathBuf;

use eframe::egui;
use tetraglass_core::{SurfaceMesh, UnstructuredGrid};
use tetraglass_egui::{
    FieldOption, ViewportColorBy, ViewportColormap, ViewportState, ViewportSurfaceStyle,
    ViewportWidget, available_field_options,
};
use tetraglass_filters::extract_external_surface;
use tetraglass_io::read_vtu_from_path;

fn parse_args() -> (PathBuf, Option<u64>) {
    let mut args = std::env::args().skip(1);
    let mut path: Option<PathBuf> = None;
    let mut max_frames: Option<u64> = None;

    while let Some(arg) = args.next() {
        if arg == "--frames" {
            if let Some(val) = args.next() {
                max_frames = val.parse::<u64>().ok();
            }
        } else if path.is_none() {
            path = Some(PathBuf::from(arg));
        }
    }

    let vtu_path = path.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root")
            .join("testdata/vtk/mixed_linear.vtu")
    });

    (vtu_path, max_frames)
}

struct ViewportExampleApp {
    file_label: String,
    grid: UnstructuredGrid,
    surface: SurfaceMesh,
    field_options: Vec<FieldOption>,
    selected_field_idx: usize,
    colormap: ViewportColormap,
    use_log_scale: bool,
    viewport: ViewportState,
    render_state: egui_wgpu::RenderState,
    frame_count: u64,
    max_frames: Option<u64>,
}

impl ViewportExampleApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        vtu_path: PathBuf,
        max_frames: Option<u64>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let render_state = cc
            .wgpu_render_state
            .clone()
            .ok_or("eframe was not initialized with the wgpu renderer")?;

        let grid = read_vtu_from_path(&vtu_path)?;
        let surface = extract_external_surface(&grid)?;
        let field_options = available_field_options(&surface);

        // Default to "Point: linear" when available, otherwise the first non-solid field if any.
        let selected_field_idx = field_options
            .iter()
            .position(|opt| opt.color_by == ViewportColorBy::point("linear"))
            .or_else(|| (field_options.len() > 1).then_some(1))
            .unwrap_or(0);

        let colormap = ViewportColormap::Viridis;
        let use_log_scale = false;
        let style = ViewportSurfaceStyle {
            color_by: field_options[selected_field_idx].color_by.clone(),
            colormap,
            range: None,
            use_log_scale,
            solid_color_srgb: [0.82, 0.83, 0.86, 1.0],
        };

        let mut viewport = ViewportState::from_render_state(&render_state, 4)?;
        viewport.set_surface(&render_state.device, &surface, style, true)?;

        let file_label = vtu_path.file_name().map_or_else(
            || vtu_path.display().to_string(),
            |s| s.to_string_lossy().into_owned(),
        );

        Ok(Self {
            file_label,
            grid,
            surface,
            field_options,
            selected_field_idx,
            colormap,
            use_log_scale,
            viewport,
            render_state,
            frame_count: 0,
            max_frames,
        })
    }

    fn sync_style(&mut self) {
        let style = ViewportSurfaceStyle {
            color_by: self.field_options[self.selected_field_idx].color_by.clone(),
            colormap: self.colormap,
            range: None,
            use_log_scale: self.use_log_scale,
            solid_color_srgb: [0.82, 0.83, 0.86, 1.0],
        };
        if let Err(err) = self
            .viewport
            .update_style(&self.render_state.queue, &self.surface, style)
        {
            log::error!("Failed to update surface style: {err}");
        }
    }
}

impl eframe::App for ViewportExampleApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("tetraglass_controls")
            .resizable(true)
            .default_size(260.0)
            .show(ui, |ui| {
                ui.heading("Tetraglass Viewport");
                ui.separator();

                ui.label(format!("File: {}", self.file_label));
                ui.label(format!(
                    "Volume: {} points, {} cells",
                    self.grid.num_points(),
                    self.grid.num_cells()
                ));
                ui.label(format!(
                    "Surface: {} faces ({} triangles)",
                    self.surface.num_faces(),
                    self.surface.num_triangles()
                ));

                ui.separator();
                ui.label("Color by:");
                let mut style_changed = false;
                let current_label = self.field_options[self.selected_field_idx].label.clone();
                egui::ComboBox::from_id_salt("color_by_picker")
                    .selected_text(current_label)
                    .width(220.0)
                    .show_ui(ui, |ui| {
                        for (idx, opt) in self.field_options.iter().enumerate() {
                            if ui
                                .selectable_value(&mut self.selected_field_idx, idx, &opt.label)
                                .changed()
                            {
                                style_changed = true;
                            }
                        }
                    });

                ui.add_space(6.0);
                ui.label("Colormap:");
                egui::ComboBox::from_id_salt("colormap_picker")
                    .selected_text(self.colormap.as_str())
                    .width(220.0)
                    .show_ui(ui, |ui| {
                        for kind in ViewportColormap::ALL {
                            if ui
                                .selectable_value(&mut self.colormap, kind, kind.as_str())
                                .changed()
                            {
                                style_changed = true;
                            }
                        }
                    });

                ui.add_space(6.0);
                if ui
                    .checkbox(&mut self.use_log_scale, "Logarithmic scale")
                    .changed()
                {
                    style_changed = true;
                }

                if style_changed {
                    self.sync_style();
                }

                if let Some([lo, hi]) = self.viewport.effective_range() {
                    ui.label(format!("Range: [{lo:.4}, {hi:.4}]"));
                }

                ui.separator();
                let cam = self.viewport.camera();
                ui.label(format!(
                    "Camera: az {:.1}°, el {:.1}°, zoom {:.2}x",
                    cam.azimuth_deg, cam.elevation_deg, cam.zoom
                ));
                if ui.button("Reset camera").clicked() {
                    self.viewport.reset_camera();
                }

                ui.separator();
                ui.small("Left drag: orbit");
                ui.small("Right / middle / Shift+drag: pan");
                ui.small("Scroll / pinch: zoom");
                ui.small("Double-click: reset camera");
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                if let Err(err) = ViewportWidget::new(&mut self.viewport)
                    .show_with_render_state(ui, &self.render_state)
                {
                    ui.colored_label(egui::Color32::RED, format!("Viewport error: {err}"));
                }
            });

        self.frame_count += 1;
        if let Some(limit) = self.max_frames {
            if self.frame_count >= limit {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            } else {
                ui.ctx().request_repaint();
            }
        }
    }

    fn on_exit(&mut self) {
        let mut renderer = self.render_state.renderer.write();
        self.viewport.destroy(&mut renderer);
    }
}

fn main() -> eframe::Result {
    let (vtu_path, max_frames) = parse_args();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 720.0])
            .with_title("Tetraglass — egui Viewport Example"),
        ..Default::default()
    };

    eframe::run_native(
        "tetraglass_egui_viewport",
        native_options,
        Box::new(move |cc| Ok(Box::new(ViewportExampleApp::new(cc, vtu_path, max_frames)?))),
    )
}

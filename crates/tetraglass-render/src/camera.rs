//! Orbit/pan/zoom camera working in dataset-center-relative coordinates (`D-014`).

use tetraglass_core::BoundingBox;

/// 3D perspective camera operating in coordinates relative to the dataset center.
///
/// By default (`azimuth_deg = 0`, `elevation_deg = 0`), the camera sits on the `+Z` axis
/// looking toward `focal_point` (`[0, 0, 0]`) with `+Y` up and `+X` to the right.
/// Positive `azimuth_deg` rotates toward `+X`, and positive `elevation_deg` raises toward `+Y`.
#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    /// Focal point relative to the dataset center (default `[0.0, 0.0, 0.0]`).
    pub focal_point: [f64; 3],
    /// Base camera distance from `focal_point` when `zoom == 1.0`.
    pub distance: f64,
    /// Horizontal orbit angle in degrees around `+Y` (`0.0` is along `+Z`).
    pub azimuth_deg: f64,
    /// Vertical elevation angle in degrees above the `XZ` plane (`[-89.0, 89.0]`).
    pub elevation_deg: f64,
    /// Roll angle in degrees around the viewing axis.
    pub roll_deg: f64,
    /// Zoom factor (`1.0` default; values `> 1.0` move the camera closer).
    pub zoom: f64,
    /// Vertical field-of-view angle in degrees (default `30.0`).
    pub view_angle_deg: f64,
    /// Bounding sphere radius of the framed dataset (used to compute tight near/far planes).
    pub scene_radius: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            focal_point: [0.0, 0.0, 0.0],
            distance: 4.0,
            azimuth_deg: 30.0,
            elevation_deg: 20.0,
            roll_deg: 0.0,
            zoom: 1.0,
            view_angle_deg: 30.0,
            scene_radius: 1.0,
        }
    }
}

impl Camera {
    /// Frames the given [`BoundingBox`] at the specified `azimuth_deg` and `elevation_deg`.
    ///
    /// Because GPU positions are uploaded relative to `bounds.center()`, `focal_point` starts at
    /// `[0.0, 0.0, 0.0]`.
    #[must_use]
    pub fn fit_bounds(bounds: BoundingBox, azimuth_deg: f64, elevation_deg: f64) -> Self {
        let radius = (bounds.diagonal() * 0.5).max(1e-3);
        let view_angle_deg = 30.0_f64;
        let half_fov = 0.5 * view_angle_deg.to_radians();
        let distance = (radius * 1.25) / half_fov.tan();
        Self {
            focal_point: [0.0, 0.0, 0.0],
            distance,
            azimuth_deg,
            elevation_deg: elevation_deg.clamp(-89.0, 89.0),
            roll_deg: 0.0,
            zoom: 1.0,
            view_angle_deg,
            scene_radius: radius,
        }
    }

    /// Returns the effective distance from `focal_point` (`distance / zoom`).
    #[must_use]
    pub fn effective_distance(&self) -> f64 {
        (self.distance / self.zoom.max(1e-3)).max(1e-5)
    }

    /// Orbits the camera by `delta_azimuth_deg` and `delta_elevation_deg`.
    pub fn orbit(&mut self, delta_azimuth_deg: f64, delta_elevation_deg: f64) {
        self.azimuth_deg = (self.azimuth_deg + delta_azimuth_deg).rem_euclid(360.0);
        self.elevation_deg = (self.elevation_deg + delta_elevation_deg).clamp(-89.0, 89.0);
    }

    /// Pans the focal point in the camera's local right/up plane by normalized viewport deltas.
    pub fn pan(&mut self, delta_right: f64, delta_up: f64) {
        let (right, up, _) = self.basis_f64();
        let scale =
            self.effective_distance() * (0.5 * self.view_angle_deg.to_radians()).tan() * 2.0;
        for i in 0..3 {
            self.focal_point[i] += (-delta_right * right[i] + delta_up * up[i]) * scale;
        }
    }

    /// Scales the camera zoom by `factor` (`> 1.0` zooms in, `< 1.0` zooms out).
    pub fn zoom_by(&mut self, factor: f64) {
        if factor.is_finite() && factor > 0.0 {
            self.zoom = (self.zoom * factor).clamp(0.05, 100.0);
        }
    }

    /// Returns the camera eye position in dataset-center-relative `f32` coordinates.
    #[must_use]
    pub fn eye_position(&self) -> [f32; 3] {
        let p = self.eye_position_f64();
        [p[0] as f32, p[1] as f32, p[2] as f32]
    }

    /// Returns the orthonormal camera basis `(right, up, back)` in `f32`, where `back` points
    /// from `focal_point` toward the eye (`-back` is the forward viewing direction).
    #[must_use]
    pub fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let (r, u, b) = self.basis_f64();
        (
            [r[0] as f32, r[1] as f32, r[2] as f32],
            [u[0] as f32, u[1] as f32, u[2] as f32],
            [b[0] as f32, b[1] as f32, b[2] as f32],
        )
    }

    /// Returns the right-handed column-major 4x4 view matrix.
    #[must_use]
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let eye = self.eye_position_f64();
        let (right, up, back) = self.basis_f64();
        let tx = -dot3(right, eye);
        let ty = -dot3(up, eye);
        let tz = -dot3(back, eye);
        [
            [right[0] as f32, up[0] as f32, back[0] as f32, 0.0],
            [right[1] as f32, up[1] as f32, back[1] as f32, 0.0],
            [right[2] as f32, up[2] as f32, back[2] as f32, 0.0],
            [tx as f32, ty as f32, tz as f32, 1.0],
        ]
    }

    /// Returns the near and far clipping plane distances `(near, far)` in view space.
    #[must_use]
    pub fn near_far_f64(&self) -> (f64, f64) {
        let dist = self.effective_distance();
        let radius = self.scene_radius.max(1e-3);
        let near = (dist - radius * 2.5).max(dist * 0.01).max(1e-4);
        let far = (dist + radius * 3.5).max(near + 1.0);
        (near, far)
    }

    /// Returns the right-handed column-major 4x4 perspective matrix for WebGPU (`z in [0, 1]`).
    #[must_use]
    pub fn proj_matrix(&self, aspect: f32) -> [[f32; 4]; 4] {
        let aspect = (aspect as f64).max(1e-4);
        let f = 1.0 / (0.5 * self.view_angle_deg.clamp(1.0, 170.0).to_radians()).tan();
        let (near, far) = self.near_far_f64();
        let nf = 1.0 / (near - far);

        [
            [(f / aspect) as f32, 0.0, 0.0, 0.0],
            [0.0, f as f32, 0.0, 0.0],
            [0.0, 0.0, (far * nf) as f32, -1.0],
            [0.0, 0.0, (near * far * nf) as f32, 0.0],
        ]
    }

    /// Returns `proj_matrix(aspect) * view_matrix()` in column-major order for WGSL.
    #[must_use]
    pub fn view_proj_matrix(&self, aspect: f32) -> [[f32; 4]; 4] {
        mul_mat4(self.proj_matrix(aspect), self.view_matrix())
    }

    /// Returns key and fill directional light unit vectors in world space (`to_light` directions).
    ///
    /// Anchored relative to the camera basis so every viewing angle has consistent three-point lighting.
    #[must_use]
    pub fn light_directions(&self) -> ([f32; 3], [f32; 3]) {
        let (right, up, back) = self.basis_f64();
        // Key light: upper-right front of camera
        let key = normalize3([
            0.55 * right[0] + 0.65 * up[0] + 0.52 * back[0],
            0.55 * right[1] + 0.65 * up[1] + 0.52 * back[1],
            0.55 * right[2] + 0.65 * up[2] + 0.52 * back[2],
        ]);
        // Fill light: lower-left front of camera
        let fill = normalize3([
            -0.65 * right[0] - 0.25 * up[0] + 0.71 * back[0],
            -0.65 * right[1] - 0.25 * up[1] + 0.71 * back[1],
            -0.65 * right[2] - 0.25 * up[2] + 0.71 * back[2],
        ]);
        (
            [key[0] as f32, key[1] as f32, key[2] as f32],
            [fill[0] as f32, fill[1] as f32, fill[2] as f32],
        )
    }

    /// Computes a world-space ray `(origin, unit_direction)` in dataset-center-relative `f64`
    /// coordinates passing through pixel center `(x + 0.5, y + 0.5)` in a `width x height` viewport.
    #[must_use]
    pub fn pixel_ray_f64(&self, x: f64, y: f64, width: f64, height: f64) -> ([f64; 3], [f64; 3]) {
        let w = width.max(1.0);
        let h = height.max(1.0);
        let ndc_x = ((x + 0.5) / w) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((y + 0.5) / h) * 2.0;
        let aspect = (w / h).max(1e-4);
        let tan_half_fov = (0.5 * self.view_angle_deg.clamp(1.0, 170.0).to_radians()).tan();
        let cam_x = ndc_x * aspect * tan_half_fov;
        let cam_y = ndc_y * tan_half_fov;

        let eye = self.eye_position_f64();
        let (right, up, back) = self.basis_f64();
        let dir = normalize3([
            cam_x * right[0] + cam_y * up[0] - back[0],
            cam_x * right[1] + cam_y * up[1] - back[1],
            cam_x * right[2] + cam_y * up[2] - back[2],
        ]);
        (eye, dir)
    }

    /// Unprojects a pixel center `(x + 0.5, y + 0.5)` and WebGPU normalized depth `ndc_z in [0, 1]`
    /// into dataset-center-relative `f64` coordinates.
    #[must_use]
    pub fn unproject_pixel_depth_f64(
        &self,
        x: f64,
        y: f64,
        ndc_z: f64,
        width: f64,
        height: f64,
    ) -> [f64; 3] {
        let w = width.max(1.0);
        let h = height.max(1.0);
        let ndc_x = ((x + 0.5) / w) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((y + 0.5) / h) * 2.0;
        let aspect = (w / h).max(1e-4);
        let tan_half_fov = (0.5 * self.view_angle_deg.clamp(1.0, 170.0).to_radians()).tan();
        let cam_x = ndc_x * aspect * tan_half_fov;
        let cam_y = ndc_y * tan_half_fov;

        let dist = self.effective_distance();
        let radius = self.scene_radius.max(1e-3);
        let near = (dist - radius * 2.5).max(dist * 0.01).max(1e-4);
        let far = (dist + radius * 3.5).max(near + 1.0);
        let z_clamped = ndc_z.clamp(0.0, 1.0);
        let denom = (far - z_clamped * (far - near)).max(1e-12);
        let forward_depth = (near * far) / denom;

        let eye = self.eye_position_f64();
        let (right, up, back) = self.basis_f64();
        [
            eye[0] + forward_depth * (cam_x * right[0] + cam_y * up[0] - back[0]),
            eye[1] + forward_depth * (cam_x * right[1] + cam_y * up[1] - back[1]),
            eye[2] + forward_depth * (cam_x * right[2] + cam_y * up[2] - back[2]),
        ]
    }

    fn eye_position_f64(&self) -> [f64; 3] {
        let dist = self.effective_distance();
        let az = self.azimuth_deg.to_radians();
        let el = self.elevation_deg.clamp(-89.0, 89.0).to_radians();
        let x = dist * el.cos() * az.sin();
        let y = dist * el.sin();
        let z = dist * el.cos() * az.cos();
        [
            self.focal_point[0] + x,
            self.focal_point[1] + y,
            self.focal_point[2] + z,
        ]
    }

    fn basis_f64(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let eye = self.eye_position_f64();
        let back = normalize3([
            eye[0] - self.focal_point[0],
            eye[1] - self.focal_point[1],
            eye[2] - self.focal_point[2],
        ]);
        let world_up = [0.0, 1.0, 0.0];
        let base_right = normalize3(cross3(world_up, back));
        let base_up = cross3(back, base_right);
        let roll = self.roll_deg.to_radians();
        let (c, s) = (roll.cos(), roll.sin());
        let right = [
            c * base_right[0] + s * base_up[0],
            c * base_right[1] + s * base_up[1],
            c * base_right[2] + s * base_up[2],
        ];
        let up = [
            -s * base_right[0] + c * base_up[0],
            -s * base_right[1] + c * base_up[1],
            -s * base_right[2] + c * base_up[2],
        ];
        (right, up, back)
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

fn mul_mat4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = a[0][row] * b[col][0]
                + a[1][row] * b[col][1]
                + a[2][row] * b[col][2]
                + a[3][row] * b[col][3];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_fits_unit_bounds_and_projects_origin_to_center() {
        let bounds = BoundingBox {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 1.0, 1.0],
        };
        let cam = Camera::fit_bounds(bounds, 30.0, 20.0);
        let vp = cam.view_proj_matrix(1.5);
        // Origin [0, 0, 0, 1] is column 3 of view_proj
        let clip_x = vp[3][0];
        let clip_y = vp[3][1];
        let clip_z = vp[3][2];
        let clip_w = vp[3][3];
        assert!(clip_w > 0.0);
        assert!((clip_x / clip_w).abs() < 1e-5);
        assert!((clip_y / clip_w).abs() < 1e-5);
        let ndc_z = clip_z / clip_w;
        assert!(ndc_z > 0.0 && ndc_z < 1.0, "ndc_z={ndc_z}");

        // Unproject the center pixel at `ndc_z` and verify it recovers the origin `[0, 0, 0]`.
        let recovered = cam.unproject_pixel_depth_f64(399.5, 299.5, f64::from(ndc_z), 800.0, 600.0);
        for c in recovered {
            assert!(c.abs() < 1e-4, "recovered={recovered:?}");
        }
    }
}

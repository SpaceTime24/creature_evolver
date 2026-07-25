use std::collections::HashSet;

use glam::{Mat4, Vec3};
use winit::keyboard::KeyCode;

/// A free-flying ("fly") camera described by a position and a look direction
/// expressed as yaw/pitch. It produces the combined view-projection matrix that
/// the shader uses to transform world-space vertices into clip space.
pub struct Camera {
    pub position: Vec3,
    /// Rotation around the world Y axis, in radians. 0 looks toward +X.
    pub yaw: f32,
    /// Up/down rotation, in radians. Clamped so we never flip over the poles.
    pub pitch: f32,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            position: Vec3::new(-8.0, 6.0, 8.0),
            // Look roughly toward the origin.
            yaw: -0.6,
            pitch: -0.5,
            aspect,
            fovy: 60f32.to_radians(),
            znear: 0.1,
            zfar: 1000.0,
        }
    }

    /// Unit vector pointing where the camera is looking.
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        )
        .normalize()
    }

    /// Unit vector pointing parallel to xz axis towards the front of camera.
    pub fn front(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin())
    }

    /// Unit vector pointing to the camera's right (for strafing).
    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    pub fn view_proj(&self) -> Mat4 {
        let view = glam::camera::rh::view::look_to_mat4(self.position, self.forward(), Vec3::Y);
        // The DirectX-style projection targets the wgpu depth range of 0..1.
        let proj = glam::camera::rh::proj::directx::perspective(
            self.fovy,
            self.aspect.max(0.001),
            self.znear,
            self.zfar,
        );
        proj * view
    }
}

/// GPU-visible camera data. Kept 16-byte aligned so it maps cleanly to a WGSL
/// uniform struct.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
}

impl CameraUniform {
    pub fn from_camera(camera: &Camera) -> Self {
        Self {
            view_proj: camera.view_proj().to_cols_array_2d(),
            camera_pos: camera.position.extend(1.0).to_array(),
        }
    }
}

/// Translates keyboard/mouse input into camera motion each frame.
pub struct CameraController {
    pressed: HashSet<KeyCode>,
    /// Accumulated mouse motion (dx, dy) waiting to be applied on the next update.
    mouse_delta: (f32, f32),
    /// Whether the "look" mouse button is currently held.
    pub looking: bool,
    move_speed: f32,
    look_sensitivity: f32,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            pressed: HashSet::new(),
            mouse_delta: (0.0, 0.0),
            looking: false,
            move_speed: 30.0,
            look_sensitivity: 0.0025,
        }
    }

    pub fn key_changed(&mut self, key: KeyCode, pressed: bool) {
        if pressed {
            self.pressed.insert(key);
        } else {
            self.pressed.remove(&key);
        }
    }

    /// Feed raw mouse motion (from a device event). Only used while `looking`.
    pub fn mouse_motion(&mut self, dx: f32, dy: f32) {
        if self.looking {
            self.mouse_delta.0 += dx;
            self.mouse_delta.1 += dy;
        }
    }

    /// Apply accumulated input to the camera. `dt` is the frame time in seconds.
    pub fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        // --- Look ---
        let (dx, dy) = std::mem::take(&mut self.mouse_delta);
        camera.yaw += dx * self.look_sensitivity;
        camera.pitch -= dy * self.look_sensitivity;
        let limit = 89f32.to_radians();
        camera.pitch = camera.pitch.clamp(-limit, limit);

        // --- Move ---
        let front = camera.front();
        let right = camera.right();
        let mut dir = Vec3::ZERO;
        if self.pressed.contains(&KeyCode::KeyW) {
            dir += front;
        }
        if self.pressed.contains(&KeyCode::KeyS) {
            dir -= front;
        }
        if self.pressed.contains(&KeyCode::KeyD) {
            dir += right;
        }
        if self.pressed.contains(&KeyCode::KeyA) {
            dir -= right;
        }
        if self.pressed.contains(&KeyCode::Space) {
            dir += Vec3::Y;
        }
        if self.pressed.contains(&KeyCode::ShiftLeft) {
            dir -= Vec3::Y;
        }

        if dir.length_squared() > 0.0 {
            camera.position += dir.normalize() * self.move_speed * dt;
        }
    }
}

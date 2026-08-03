use std::{
    println,
    sync::{Arc, RwLock},
};

use wgpu::CurrentSurfaceTexture;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::CursorGrabMode,
};

use crate::graphical_app::renderer::Renderer;
use crate::{
    creature_environment::static_environment::CreatureParty, graphical_app::wgpu_state::WgpuState,
};

enum AppState {
    Uninitialized,
    Loading,
    Running(Renderer),
}

pub struct App {
    state: AppState,
    creature_party: Arc<RwLock<CreatureParty>>,
}

impl App {
    pub fn new(party: Arc<RwLock<CreatureParty>>) -> Self {
        App {
            state: AppState::Uninitialized,
            creature_party: party,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let AppState::Uninitialized = self.state {
            self.state = AppState::Loading;
            let wgpu_state = pollster::block_on(WgpuState::new(event_loop));
            let renderer = Renderer::new(wgpu_state);
            renderer.gpu.window.request_redraw();
            self.state = AppState::Running(renderer);
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        let AppState::Running(ref mut renderer) = self.state else {
            return;
        };
        // Raw mouse motion drives the free-look camera (only applied while the
        // look button is held; see the controller).
        if let DeviceEvent::MouseMotion { delta } = event {
            renderer.mouse_motion(delta.0 as f32, delta.1 as f32);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let AppState::Running(ref mut renderer) = self.state else {
            return;
        };

        match event {
            WindowEvent::Resized(new_size) => {
                renderer.resize(new_size);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if code == KeyCode::Escape && event.state.is_pressed() {
                        // Release the mouse look grab.
                        renderer.set_looking(false);
                        let _ = renderer.gpu.window.set_cursor_grab(CursorGrabMode::None);
                        renderer.gpu.window.set_cursor_visible(true);
                    }
                    renderer.key_changed(code, event.state.is_pressed());
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Right {
                    let looking = state == ElementState::Pressed;
                    renderer.set_looking(looking);
                    let window = &renderer.gpu.window;
                    if looking {
                        // Prefer Locked (cursor recentered) and fall back to Confined.
                        if window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                            let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                        }
                        window.set_cursor_visible(false);
                    } else {
                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                        window.set_cursor_visible(true);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                renderer.update();

                let frame = match renderer.gpu.surface.get_current_texture() {
                    CurrentSurfaceTexture::Success(frame) => frame,
                    CurrentSurfaceTexture::Timeout | CurrentSurfaceTexture::Occluded => {
                        renderer.gpu.window.request_redraw();
                        return;
                    }
                    CurrentSurfaceTexture::Suboptimal(texture) => {
                        println!("redraw-suboptimal");
                        drop(texture);
                        renderer
                            .gpu
                            .surface
                            .configure(&renderer.gpu.device, &renderer.gpu.config);
                        renderer.gpu.window.request_redraw();
                        return;
                    }
                    CurrentSurfaceTexture::Outdated => {
                        println!("redraw-outdated");
                        renderer
                            .gpu
                            .surface
                            .configure(&renderer.gpu.device, &renderer.gpu.config);
                        renderer.gpu.window.request_redraw();
                        return;
                    }
                    CurrentSurfaceTexture::Validation => {
                        unreachable!("No error scope registered, so validation errors will panic");
                    }
                    CurrentSurfaceTexture::Lost => {
                        println!("redraw-lost");
                        renderer.gpu.remake_surface();
                        renderer
                            .gpu
                            .surface
                            .configure(&renderer.gpu.device, &renderer.gpu.config);
                        renderer.gpu.window.request_redraw();
                        return;
                    }
                };

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                renderer.render(&view, &self.creature_party);

                renderer.gpu.window.pre_present_notify();
                renderer.gpu.queue.present(frame);

                // Keep the simulation and camera animating continuously.
                renderer.gpu.window.request_redraw();
            }
            WindowEvent::Occluded(is_occluded) => {
                if !is_occluded {
                    renderer.gpu.window.request_redraw();
                }
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        }
    }
}

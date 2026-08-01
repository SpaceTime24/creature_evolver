use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use wgpu::{Surface, SurfaceTargetUnsafe};
use winit::{event_loop::ActiveEventLoop, window::Window};

pub struct WgpuState {
    pub inner: Box<WgpuStateInner<'static>>,
}

impl Deref for WgpuState {
    type Target = WgpuStateInner<'static>;

    fn deref(&self) -> &Self::Target {
        self.inner.deref()
    }
}

impl DerefMut for WgpuState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.deref_mut()
    }
}

pub struct WgpuStateInner<'a> {
    pub instance: wgpu::Instance,
    pub surface: Surface<'a>,
    pub window: Window,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
}

impl WgpuState {
    pub fn remake_surface(&mut self) {
        let new_surface = unsafe {
            self.instance
                .create_surface_unsafe(
                    SurfaceTargetUnsafe::from_window(&self.window)
                        .expect("failed to create surface target from window (remake)"),
                )
                .expect("failed to create surface with generated surface target (remake)")
        };
        self.surface = new_surface;
    }

    pub fn resize_surface(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        // Reconfigure the surface with the new size
        self.config.width = new_size.width.max(1);
        self.config.height = new_size.height.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    pub async fn new(event_loop: &ActiveEventLoop) -> Self {
        let inner: Box<WgpuStateInner> = unsafe {
            let mut uninit_inner: Box<MaybeUninit<WgpuStateInner>> = Box::new_uninit();
            let uninit_wgpu_state_ptr = uninit_inner.as_mut_ptr();
            {
                let attributes = Window::default_attributes();
                let window = event_loop
                    .create_window(attributes)
                    .expect("Failed to create window");

                let window_ptr: *mut _ = &mut (*uninit_wgpu_state_ptr).window;
                window_ptr.write(window);
            }

            let mut size = (*uninit_wgpu_state_ptr).window.inner_size();
            size.width = size.width.max(1);
            size.height = size.height.max(1);

            {
                let display_handle = event_loop.owned_display_handle();

                let instance = wgpu::Instance::new(
                    wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(
                        display_handle,
                    )),
                );

                let instance_ptr: *mut _ = &mut (*uninit_wgpu_state_ptr).instance;
                instance_ptr.write(instance);
            }

            {
                let surface = (*uninit_wgpu_state_ptr)
                    .instance
                    .create_surface_unsafe(
                        SurfaceTargetUnsafe::from_window(&(*uninit_wgpu_state_ptr).window)
                            .expect("Couldn't create surface target from window"),
                    )
                    .expect("Couldn't create surface from surface target");

                let surface_ptr: *mut _ = &mut (*uninit_wgpu_state_ptr).surface;
                surface_ptr.write(surface);
            }

            {
                let adapter = (*uninit_wgpu_state_ptr)
                    .instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        force_fallback_adapter: false,
                        compatible_surface: Some(&(*uninit_wgpu_state_ptr).surface),
                        apply_limit_buckets: false,
                    })
                    .await
                    .expect("Failed to get adapter");

                {
                    let (device, queue) = adapter
                        .request_device(&wgpu::DeviceDescriptor {
                            label: None,
                            required_features: wgpu::Features::empty(),
                            // Make sure we use the texture resolution limits from the adapter,
                            // so we can support images the size of the swapchain.
                            required_limits: wgpu::Limits {
                                ..wgpu::Limits::default()
                            },
                            experimental_features: wgpu::ExperimentalFeatures::disabled(),
                            memory_hints: wgpu::MemoryHints::Performance,
                            trace: wgpu::Trace::Off,
                        })
                        .await
                        .expect("Failed to create device");

                    let device_ptr: *mut _ = &mut (*uninit_wgpu_state_ptr).device;
                    let queue_ptr: *mut _ = &mut (*uninit_wgpu_state_ptr).queue;
                    device_ptr.write(device);
                    queue_ptr.write(queue);
                }

                {
                    let config = (*uninit_wgpu_state_ptr)
                        .surface
                        .get_default_config(&adapter, size.width, size.height)
                        .unwrap();

                    let config_ptr: *mut _ = &mut (*uninit_wgpu_state_ptr).config;
                    config_ptr.write(config);
                }
            }
            uninit_inner.assume_init()
        };

        WgpuState { inner }
    }
}

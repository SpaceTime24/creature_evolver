use std::sync::{Arc, RwLock};
use std::time::Instant;

use wgpu::util::DeviceExt;
use winit::keyboard::KeyCode;

use crate::creature_environment::static_environment::CreatureParty;
use crate::graphical_app::camera::{Camera, CameraController, CameraUniform};
use crate::graphical_app::mesh::InstanceRaw;
use crate::graphical_app::mesh::{Mesh, MeshId, unit_cube, unit_cylinder, unit_sphere};
use crate::graphical_app::pipeline::{DEPTH_FORMAT, SimplePipelineManager};
use crate::graphical_app::wgpu_state::WgpuState;
use crate::simulation_running::threading::SimulationThreadHandle;

/// Owns everything needed to simulate and draw the world: the GPU core, the
/// render pipeline, the camera, the depth buffer, the mesh library, and the scene.
pub struct Renderer {
    pub gpu: WgpuState,
    pipeline_manager: SimplePipelineManager,

    pub camera: Camera,
    pub controller: CameraController,
    camera_uniform_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    depth_view: wgpu::TextureView,

    meshes: Vec<Option<Mesh>>,

    last_frame: Instant,
}

impl Renderer {
    pub fn new(gpu: WgpuState) -> Self {
        // The surface hasn't been configured yet (WgpuState::new only builds the
        // resources); configure it now so the first frame has a valid swapchain.
        gpu.surface.configure(&gpu.device, &gpu.config);

        let pipeline_manager = SimplePipelineManager::new(&gpu.device, gpu.config.format);

        let camera = Camera::new(gpu.config.width as f32 / gpu.config.height as f32);
        let controller = CameraController::new();

        let camera_uniform = CameraUniform::from_camera(&camera);
        let camera_uniform_buffer =
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Camera Uniform Buffer"),
                    contents: bytemuck::bytes_of(&camera_uniform),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
        let camera_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &pipeline_manager.camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_uniform_buffer.as_entire_binding(),
            }],
        });

        let depth_view = create_depth_view(&gpu.device, &gpu.config);

        // Build the unit mesh library. Objects reuse these, scaled per-instance.
        let mut meshes = Vec::with_capacity(MeshId::MeshCount as usize);
        for _ in 0..MeshId::MeshCount as usize {
            meshes.push(None);
        }

        let (cv, ci) = unit_cube();
        meshes[MeshId::Cube as usize] = Some(Mesh::new(&gpu.device, "Cube", &cv, &ci));

        let (sv, si) = unit_sphere(16, 12);
        meshes[MeshId::Sphere as usize] = Some(Mesh::new(&gpu.device, "Sphere", &sv, &si));

        let (cyv, cyi) = unit_cylinder(24);
        meshes[MeshId::Cylinder as usize] = Some(Mesh::new(&gpu.device, "Cylinder", &cyv, &cyi));

        Self {
            gpu,
            pipeline_manager,
            camera,
            controller,
            camera_uniform_buffer,
            camera_bind_group,
            depth_view,
            meshes,
            last_frame: Instant::now(),
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.gpu.resize_surface(new_size);
        self.depth_view = create_depth_view(&self.gpu.device, &self.gpu.config);
        self.camera.aspect = self.gpu.config.width as f32 / self.gpu.config.height as f32;
        self.gpu.window.request_redraw();
    }

    pub fn key_changed(&mut self, key: KeyCode, pressed: bool) {
        self.controller.key_changed(key, pressed);
    }

    pub fn set_looking(&mut self, looking: bool) {
        self.controller.looking = looking;
    }

    pub fn mouse_motion(&mut self, dx: f32, dy: f32) {
        self.controller.mouse_motion(dx, dy);
    }

    /// Advance one frame of simulation and camera motion, then upload the camera
    /// uniform. Call once per frame before [`render`](Self::render).
    pub fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        self.controller.update_camera(&mut self.camera, dt);

        let camera_uniform = CameraUniform::from_camera(&self.camera);
        self.gpu.queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(&camera_uniform),
        );
    }

    fn add_mesh_raw_instances_to_instance_holders(
        mesh_instance_pairs: &Vec<(MeshId, Vec<InstanceRaw>)>,
        instance_holders: &mut [Vec<InstanceRaw>; MeshId::MeshCount as usize],
        mesh_type_count: &mut usize,
    ) {
        for (mesh_id, instances) in mesh_instance_pairs {
            if instance_holders[*mesh_id as usize].is_empty() {
                *mesh_type_count += 1;
            }
            for instance in instances {
                instance_holders[*mesh_id as usize].push(*instance);
            }
        }
    }

    /// Draw the current scene into the given swapchain view.
    pub fn render(&self, view: &wgpu::TextureView, party: &Arc<RwLock<CreatureParty>>) {
        let mut instance_holders: [Vec<InstanceRaw>; MeshId::MeshCount as usize] =
            [const { Vec::new() }; MeshId::MeshCount as usize];

        let mut mesh_type_count = 0;

        for thread_handle in party.write().unwrap().thread_handles() {
            if let Some(mesh_instance_pairs) = thread_handle.get_new_instances() {
                Renderer::add_mesh_raw_instances_to_instance_holders(
                    mesh_instance_pairs,
                    &mut instance_holders,
                    &mut mesh_type_count,
                );
            }
        }

        {
            Renderer::add_mesh_raw_instances_to_instance_holders(
                party
                    .read()
                    .unwrap()
                    .static_environment
                    .get_mesh_instances(),
                &mut instance_holders,
                &mut mesh_type_count,
            )
        }

        // Build one instance buffer per mesh type from the current body poses.
        let mut draws: Vec<(&Mesh, wgpu::Buffer, u32)> = Vec::with_capacity(mesh_type_count);
        for mesh_id in MeshId::all_mesh_ids() {
            if let Some(mesh) = self.meshes[mesh_id as usize].as_ref() {
                let instances = &instance_holders[mesh_id as usize];
                if instances.is_empty() {
                    //I think this check can be removed
                    continue;
                }
                let buffer =
                    self.gpu
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Instance Buffer"),
                            contents: bytemuck::cast_slice(&instances),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                draws.push((mesh, buffer, instances.len() as u32));
            }
        }

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Encoder"),
            });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.06,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(&self.pipeline_manager.render_pipeline);
            rpass.set_bind_group(0, &self.camera_bind_group, &[]);

            for (mesh, instance_buffer, instance_count) in &draws {
                rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                rpass.set_vertex_buffer(1, instance_buffer.slice(..));
                rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..mesh.num_indices, 0, 0..*instance_count);
            }
        }

        self.gpu.queue.submit(Some(encoder.finish()));
    }
}

fn create_depth_view(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size: wgpu::Extent3d {
            width: config.width.max(1),
            height: config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

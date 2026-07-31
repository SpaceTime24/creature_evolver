use std::path::Iter;

use glam::Vec3;
use rapier3d::geometry::{ShapeType, SharedShape};
use wgpu::util::DeviceExt;

/// A single mesh vertex: position plus a normal for lighting.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl Vertex {
    /// Vertex buffer layout: consumed per-vertex at shader locations 0 and 1.
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
    };
}

/// GPU buffers for one drawable mesh.
pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

impl Mesh {
    pub fn new(device: &wgpu::Device, label: &str, vertices: &[Vertex], indices: &[u32]) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} Vertex Buffer")),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} Index Buffer")),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
        }
    }
}

/// Identifies which unit mesh a scene object is drawn with. Objects scale the
/// unit mesh via their model matrix, so a single mesh serves every box/ball.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum MeshId {
    /// Unit cube spanning -1..1 on each axis (half-extents of 1).
    Cube,
    /// Unit sphere of radius 1.
    Sphere,

    //Cylinder radius 1, height 1
    Cylinder,

    MeshCount,

    NotSupported,
}

impl MeshId {
    pub fn all_mesh_ids() -> impl Iterator<Item = MeshId> {
        MeshIdIter {
            cur_id: unsafe { std::mem::transmute(0u8) },
        }
    }
}

impl TryFrom<ShapeType> for MeshId {
    type Error = ();
    fn try_from(value: ShapeType) -> std::result::Result<MeshId, ()> {
        Ok(match value {
            ShapeType::Ball => MeshId::Sphere,
            ShapeType::Cuboid => MeshId::Cube,
            ShapeType::Cylinder => MeshId::Cylinder,
            _ => return Err(()),
        })
    }
}

pub fn scale_from_shape(shape: &SharedShape) -> Result<Vec3, ()> {
    match shape.shape_type() {
        rapier3d::geometry::ShapeType::Ball => {
            let radius = shape.as_ball().unwrap().radius;
            Ok(Vec3::new(radius, radius, radius))
        }
        rapier3d::geometry::ShapeType::Cuboid => Ok(shape.as_cuboid().unwrap().half_extents),
        rapier3d::geometry::ShapeType::Cylinder => {
            let cylinder = shape.as_cylinder().unwrap();
            let half_height = cylinder.half_height;
            let radius = cylinder.radius;
            Ok(Vec3::new(radius, half_height, radius))
        }
        _ => return Err(()),
    }
}

struct MeshIdIter {
    cur_id: MeshId,
}

impl Iterator for MeshIdIter {
    type Item = MeshId;

    fn next(&mut self) -> Option<Self::Item> {
        if let MeshId::MeshCount = self.cur_id {
            return None;
        } else {
            let return_id = self.cur_id;
            self.cur_id = unsafe { std::mem::transmute(self.cur_id as u8 + 1) };
            return Some(return_id);
        }
    }
}

/// Unit cube with per-face normals, spanning -1..=1 so that scaling by a
/// collider's half-extents reproduces the physics box exactly.
pub fn unit_cube() -> (Vec<Vertex>, Vec<u32>) {
    // (normal, four corners in CCW order when viewed from outside)
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +X
        (
            [1.0, 0.0, 0.0],
            [
                [1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
        ),
        // -X
        (
            [-1.0, 0.0, 0.0],
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
            ],
        ),
        // +Y
        (
            [0.0, 1.0, 0.0],
            [
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
            ],
        ),
        // -Y
        (
            [0.0, -1.0, 0.0],
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0],
            ],
        ),
        // +Z
        (
            [0.0, 0.0, 1.0],
            [
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
        ),
        // -Z
        (
            [0.0, 0.0, -1.0],
            [
                [1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
            ],
        ),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, corners) in faces {
        let base = vertices.len() as u32;
        for position in corners {
            vertices.push(Vertex { position, normal });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}

/// A UV sphere of radius 1. Positions double as normals since it is centered at
/// the origin with unit radius.
pub fn unit_sphere(sectors: u32, stacks: u32) -> (Vec<Vertex>, Vec<u32>) {
    use std::f32::consts::PI;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=stacks {
        // stack_angle goes from +pi/2 (top) to -pi/2 (bottom)
        let stack_angle = PI / 2.0 - (i as f32 / stacks as f32) * PI;
        let xy = stack_angle.cos();
        let z = stack_angle.sin();
        for j in 0..=sectors {
            let sector_angle = (j as f32 / sectors as f32) * 2.0 * PI;
            let x = xy * sector_angle.cos();
            let y = z;
            let w = xy * sector_angle.sin();
            let p = [x, y, w];
            vertices.push(Vertex {
                position: p,
                normal: p,
            });
        }
    }

    let ring = sectors + 1;
    for i in 0..stacks {
        for j in 0..sectors {
            let a = i * ring + j;
            let b = a + ring;
            // Two triangles per quad, skipping degenerate ones at the poles.
            if i != 0 {
                indices.extend_from_slice(&[a + 1, b, a]);
            }
            if i != stacks - 1 {
                indices.extend_from_slice(&[a + 1, b + 1, b]);
            }
        }
    }
    (vertices, indices)
}

pub fn unit_cylinder(sectors: u32) -> (Vec<Vertex>, Vec<u32>) {
    use std::f32::consts::PI;
    //(sectors + 1) * 4 sector vertices, + 1 top, 1 bottom
    let sector_vtx_count = (sectors) * 4;
    let top_center_idx = sector_vtx_count;
    let bot_center_idx = sector_vtx_count + 1;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..sectors {
        let sector_angle = (i as f32 / sectors as f32) * 2.0 * PI;
        let x = sector_angle.cos();
        let z = sector_angle.sin();
        let out_normal = [x, 0.0, z];

        vertices.push(Vertex {
            position: [x, 0.5, z],
            normal: out_normal,
        });
        vertices.push(Vertex {
            position: [x, -0.5, z],
            normal: out_normal,
        });
        vertices.push(Vertex {
            position: [x, 0.5, z],
            normal: [0.0, 1.0, 0.0],
        });
        vertices.push(Vertex {
            position: [x, -0.5, z],
            normal: [0.0, -1.0, 0.0],
        });
    }

    vertices.push(Vertex {
        position: [0.0, 0.5, 0.0],
        normal: [0.0, 1.0, 0.0],
    });
    vertices.push(Vertex {
        position: [0.0, -0.5, 0.0],
        normal: [0.0, -1.0, 0.0],
    });

    for i in 0..sectors {
        let top_out_idx_a = i * 4;
        let bot_out_idx_a = top_out_idx_a + 1;
        let top_up_idx_a = top_out_idx_a + 2;
        let bot_up_idx_a = top_out_idx_a + 3;

        let top_out_idx_b = ((i + 1) % sectors) * 4;
        let bot_out_idx_b = top_out_idx_b + 1;
        let top_up_idx_b = top_out_idx_b + 2;
        let bot_up_idx_b = top_out_idx_b + 3;

        indices.extend_from_slice(&[top_out_idx_a, top_out_idx_b, bot_out_idx_a]);
        indices.extend_from_slice(&[bot_out_idx_a, top_out_idx_b, bot_out_idx_b]);

        indices.extend_from_slice(&[top_up_idx_a, top_center_idx, top_up_idx_b]);
        indices.extend_from_slice(&[bot_up_idx_b, bot_center_idx, bot_up_idx_a]);
    }

    (vertices, indices)
}

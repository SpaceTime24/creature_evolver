use std::{array, todo};

use glam::{Mat4, Vec3};
use rand::random_range;
use rapier3d::prelude::*;

use crate::{creature_environment::creature::Creature, graphical_app::mesh::MeshId};

/// Per-instance data uploaded to the GPU: a full model matrix plus a color.
/// One of these is produced for every scene object each frame.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
}

impl InstanceRaw {
    /// Instance buffer layout. Uses shader locations 2..=6 so it does not collide
    /// with the per-vertex attributes (locations 0 and 1). A `mat4x4` is passed
    /// as four consecutive `vec4` slots.
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &wgpu::vertex_attr_array![
            2 => Float32x4,
            3 => Float32x4,
            4 => Float32x4,
            5 => Float32x4,
            6 => Float32x4,
        ],
    };
}

/// A visible object in the world: a rigid body paired with how to draw it.
pub struct Object {
    pub body: RigidBodyHandle,
    pub mesh: MeshId,
    /// Non-uniform scale applied to the unit mesh (the collider half-extents for a
    /// box, or the radius for a sphere).
    pub scale: Vec3,
    pub color: Vec3,
}

/// The simulated world plus the list of objects to render. Wraps rapier's
/// [`PhysicsWorld`] convenience type and keeps a parallel list of render data.
pub struct Scene {
    pub physics: PhysicsWorld,
    pub objects: [Vec<Object>; MeshId::MeshCount as usize],
}

impl Scene {
    pub fn new() -> Self {
        Self {
            physics: PhysicsWorld::new(),
            objects: [const { Vec::new() }; MeshId::MeshCount as usize],
        }
    }

    /// Add a stationary (fixed) box — part of the environment. `half_extents` are
    /// the box half-sizes, matching rapier's cuboid convention.
    pub fn spawn_fixed_box(&mut self, center: Vec3, half_extents: Vec3, color: Vec3) {
        let body = RigidBodyBuilder::fixed().translation(center);
        let collider = ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z);
        let (handle, _) = self.physics.insert(body, collider);
        self.objects[MeshId::Cube as usize].push(Object {
            body: handle,
            mesh: MeshId::Cube,
            scale: half_extents,
            color,
        });
    }

    /// Add a dynamic box — the kind of part a creature body will be built from.
    pub fn spawn_dynamic_box(&mut self, center: Vec3, half_extents: Vec3, color: Vec3) {
        let body = RigidBodyBuilder::dynamic().translation(center);
        let collider =
            ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z).density(1.0);
        let (handle, _) = self.physics.insert(body, collider);
        self.objects[MeshId::Cube as usize].push(Object {
            body: handle,
            mesh: MeshId::Cube,
            scale: half_extents,
            color,
        });
    }

    /// Add a dynamic ball.
    pub fn spawn_dynamic_ball(&mut self, center: Vec3, radius: f32, color: Vec3) {
        let body = RigidBodyBuilder::dynamic().translation(center);
        let collider = ColliderBuilder::ball(radius).density(1.0).restitution(0.8);
        let (handle, _) = self.physics.insert(body, collider);
        self.objects[MeshId::Sphere as usize].push(Object {
            body: handle,
            mesh: MeshId::Sphere,
            scale: Vec3::splat(radius),
            color,
        });
    }

    /// Add a dynamic cylinder.
    pub fn spawn_dynamic_cylinder(&mut self, center: Vec3, radius: f32, height: f32, color: Vec3) {
        let body = RigidBodyBuilder::dynamic().translation(center);
        let collider = ColliderBuilder::cylinder(height * 0.5, radius)
            .density(1.0)
            .restitution(1.2);
        let (handle, _) = self.physics.insert(body, collider);
        self.objects[MeshId::Cylinder as usize].push(Object {
            body: handle,
            mesh: MeshId::Cylinder,
            scale: Vec3::new(radius, height, radius),
            color,
        });
    }

    /// Advance the simulation by one step.
    pub fn step(&mut self) {
        self.physics.step();
    }

    /// Produce the render instances for every object with the given mesh, reading
    /// the current pose from each rigid body.
    pub fn instances_for(&self, mesh: MeshId) -> Vec<InstanceRaw> {
        self.objects[mesh as usize]
            .iter()
            .map(|obj| {
                let body = &self.physics.bodies[obj.body];
                let translation = body.translation();
                let rotation = *body.rotation();
                let model = Mat4::from_scale_rotation_translation(obj.scale, rotation, translation);
                InstanceRaw {
                    model: model.to_cols_array_2d(),
                    color: obj.color.extend(1.0).to_array(),
                }
            })
            .collect()
    }

    /// A small starter world: a ground plane (fixed box) plus a few dynamic bodies
    /// that fall and settle. Replace/extend this as the creature body takes shape.
    pub fn demo_scene() -> Scene {
        let mut scene = Scene::new();

        let box_edge = 100.0;
        let box_thickness = 1.0;

        scene.spawn_fixed_box(
            Vec3::new(box_edge, 0.0, 0.0),
            Vec3::new(box_thickness, box_edge, box_edge),
            Vec3::new(0.8, 0.2, 0.2),
        );

        scene.spawn_fixed_box(
            Vec3::new(0.0, 0.0, box_edge),
            Vec3::new(box_edge, box_edge, box_thickness),
            Vec3::new(0.2, 0.2, 0.8),
        );

        scene.spawn_fixed_box(
            Vec3::new(-box_edge, 0.0, 0.0),
            Vec3::new(box_thickness, box_edge, box_edge),
            Vec3::new(0.35, 0.37, 0.4),
        );

        scene.spawn_fixed_box(
            Vec3::new(0.0, 0.0, -box_edge),
            Vec3::new(box_edge, box_edge, box_thickness),
            Vec3::new(0.35, 0.37, 0.4),
        );

        scene.spawn_fixed_box(
            Vec3::new(0.0, -box_edge, 0.0),
            Vec3::new(box_edge, box_thickness, box_edge),
            Vec3::new(0.35, 0.37, 0.4),
        );

        scene.spawn_fixed_box(
            Vec3::new(-5.0, 0.5, 0.0),
            Vec3::new(2.0, 1.0, 2.0),
            Vec3::new(0.3, 0.45, 0.5),
        );

        // Dynamic bodies (future creature parts) dropped from a height.
        scene.spawn_dynamic_box(
            Vec3::new(0.0, 6.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.85, 0.4, 0.35),
        );
        scene.spawn_dynamic_box(
            Vec3::new(0.4, 9.0, 0.2),
            Vec3::new(0.5, 0.25, 0.75),
            Vec3::new(0.4, 0.75, 0.5),
        );

        scene.spawn_dynamic_cylinder(
            Vec3::new(0.4, 9.0, 0.2),
            1.0,
            5.0,
            Vec3::new(0.6, 0.25, 0.3),
        );

        for i in 0..70 {
            let random_pos =
                Vec3::from_array(array::from_fn(|_| random_range(-box_edge..box_edge)));
            let random_color = Vec3::from_array(array::from_fn(|_| random_range(0.0..1.0)));

            scene.spawn_dynamic_ball(random_pos, random_color.z * 5.0, random_color);
        }

        for i in 0..70 {
            let random_pos =
                Vec3::from_array(array::from_fn(|_| random_range(-box_edge..box_edge)));
            let random_color = Vec3::from_array(array::from_fn(|_| random_range(0.0..1.0)));

            scene.spawn_dynamic_cylinder(
                random_pos,
                random_color.x * 5.0,
                random_color.y * 8.0,
                random_color,
            );
        }

        for i in 0..70 {
            let random_pos =
                Vec3::from_array(array::from_fn(|_| random_range(-box_edge..box_edge)));
            let random_color = Vec3::from_array(array::from_fn(|_| random_range(0.0..1.0)));

            scene.spawn_dynamic_box(
                random_pos,
                Vec3::new(
                    random_color.x * 5.0,
                    random_color.y * 5.0,
                    random_color.z * 5.0,
                ),
                random_color,
            );
        }

        scene
    }
}

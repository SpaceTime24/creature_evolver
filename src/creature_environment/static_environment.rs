use glam::Vec3;
use rapier3d::{
    dynamics::{RigidBodyBuilder, RigidBodyHandle},
    geometry::{ColliderBuilder, ColliderHandle},
    pipeline::PhysicsWorld,
};

pub struct StaticEnvironmentBuilder<'a> {
    rigid_environment: RigidBodyHandle,
    physics_world: &'a mut PhysicsWorld,
    colliders_and_colors: Vec<(ColliderHandle, Vec3)>,
}

impl<'a> StaticEnvironmentBuilder<'a> {
    pub fn new(physics_world: &'a mut PhysicsWorld) -> StaticEnvironmentBuilder<'a> {
        let rigid_builder = RigidBodyBuilder::fixed();
        let world_rigid_body_handle = physics_world.bodies.insert(rigid_builder);
        StaticEnvironmentBuilder {
            rigid_environment: world_rigid_body_handle,
            physics_world,
            colliders_and_colors: Vec::new(),
        }
    }

    pub fn spawn_fixed_box(&mut self, center: Vec3, half_extents: Vec3, color: Vec3) {
        let collider = ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z)
            .translation(center);
        let collider_handle = self
            .physics_world
            .insert_collider(collider, Some(self.rigid_environment));
        self.colliders_and_colors.push((collider_handle, color));
    }
}

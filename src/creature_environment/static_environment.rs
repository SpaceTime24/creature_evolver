use std::{iter::zip, todo};

use glam::Vec3;
use rapier3d::{
    dynamics::{RigidBodyBuilder, RigidBodyHandle},
    geometry::{Collider, ColliderBuilder, ColliderHandle, ColliderSet},
    pipeline::PhysicsWorld,
};

use crate::{
    creature_environment::{creature::Creature, creature_world::CreatureWorld},
    graphical_app::{
        mesh::{MeshId, instance_from_collider},
        scene::InstanceRaw,
    },
    neural_net::neural_placeholder::NeuralPlaceholder,
    simulation_running::threading::SimulationThreadHandle,
};

pub struct StaticEnvironment {
    template_collider_set: ColliderSet,
    instances_to_render: Vec<(MeshId, Vec<InstanceRaw>)>,
}

impl StaticEnvironment {
    pub fn new(
        colliders_generator: impl IntoIterator<Item = (Collider, Vec3)>,
    ) -> StaticEnvironment {
        let mut collider_set = ColliderSet::new();

        let mut instance_holders: [Vec<InstanceRaw>; MeshId::MeshCount as usize] =
            [const { Vec::new() }; MeshId::MeshCount as usize];
        let mut mesh_type_count = 0;

        for (collider, color) in colliders_generator {
            if let Ok((mesh_id, instance)) = instance_from_collider(&collider, color) {
                collider_set.insert(collider);
                if instance_holders[mesh_id as usize].is_empty() {
                    mesh_type_count += 1;
                }
                instance_holders[mesh_id as usize].push(instance);
            }
        }

        let mut instances_to_render = Vec::with_capacity(mesh_type_count);
        for (mesh_id, instance_list) in zip(MeshId::all_mesh_ids(), instance_holders) {
            if !instance_list.is_empty() {
                instances_to_render.push((mesh_id, instance_list));
            }
        }
        StaticEnvironment {
            template_collider_set: collider_set,
            instances_to_render,
        }
    }

    pub fn get_mesh_instances(&self) -> &Vec<(MeshId, Vec<InstanceRaw>)> {
        &self.instances_to_render
    }

    pub fn add_environment_to_creature_world(&self, creature_world: &mut CreatureWorld) {
        let world_rigid_body_builder = RigidBodyBuilder::fixed();
        let world_rigid_handle = creature_world.physics.insert_body(world_rigid_body_builder);
        for (_template_collider_handle, template_collider) in self.template_collider_set.iter() {
            let new_collider = template_collider.clone();
            creature_world
                .physics
                .insert_collider(new_collider, Some(world_rigid_handle));
        }
    }
}

pub struct CreatureParty {
    simulation_threads: Vec<SimulationThreadHandle>,
    pub static_environment: StaticEnvironment,
}

impl CreatureParty {
    pub fn new(static_environment: StaticEnvironment) -> CreatureParty {
        CreatureParty {
            simulation_threads: Vec::new(),
            static_environment,
        }
    }

    pub fn thread_handles(&mut self) -> &mut Vec<SimulationThreadHandle> {
        &mut self.simulation_threads
    }

    pub fn simulation_count(&self) -> usize {
        return self.simulation_threads.len();
    }

    pub fn run_generation_of_creatures(
        &mut self,
        creature_generator: for<'a> fn(&'a mut PhysicsWorld) -> Creature<'a>,
        neural_networks: impl IntoIterator<Item = NeuralPlaceholder>,
    ) {
        let mut world_idx = 0;

        for neural_net in neural_networks {
            if self.simulation_threads.get(world_idx).is_none() {
                self.create_world();
            }
            let thread_handle = self.simulation_threads.get_mut(world_idx).unwrap();
            thread_handle.add_creature(creature_generator);
            thread_handle.start_simulation();
            world_idx += 1;
        }
    }

    fn create_world(&mut self) {
        let mut new_thread = SimulationThreadHandle::new();
        let mut new_world = CreatureWorld::new_empty();
        self.static_environment
            .add_environment_to_creature_world(&mut new_world);
        new_thread.activate(new_world);
        self.simulation_threads.push(new_thread);
    }

    fn delete_world(&mut self) {
        if let Some(mut thread) = self.simulation_threads.pop() {
            thread.deactivate();
        }
    }
}

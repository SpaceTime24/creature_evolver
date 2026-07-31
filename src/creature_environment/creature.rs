use std::{
    any::Any,
    f32::consts::PI,
    iter::{Zip, zip},
    mem::MaybeUninit,
    todo,
};

use glam::{Mat4, Vec3};
use rapier3d::{
    dynamics::{
        ImpulseJointHandle, RevoluteJointBuilder, RigidBodyBuilder, RigidBodyHandle,
        SphericalJointBuilder,
    },
    geometry::{ColliderBuilder, ColliderHandle, ColliderType, SharedShape},
    pipeline::PhysicsWorld,
};

use crate::graphical_app::{
    mesh::{MeshId, scale_from_shape},
    scene::InstanceRaw,
};

//Creature local coordinates: +x front, +z right, +y up

pub struct Creature<'world_life> {
    base: CreatureBodyLink,
    world: &'world_life PhysicsWorld,
}

pub struct CreatureBodyLink {
    rigid_body: RigidBodyHandle,
    child_links: Vec<CreatureBodyLink>,
    joints: Vec<ImpulseJointHandle>,
    label: String,
    color: Vec3,
}

impl CreatureBodyLink {
    pub fn new(rigid_body: RigidBodyHandle, label: String, color: Vec3) -> CreatureBodyLink {
        CreatureBodyLink {
            rigid_body,
            child_links: Vec::new(),
            joints: Vec::new(),
            label,
            color,
        }
    }

    pub fn add_child_link(&mut self, child: CreatureBodyLink, joint_handle: ImpulseJointHandle) {
        self.joints.push(joint_handle);
        self.child_links.push(child);
    }

    pub fn all_child_links(&self) -> &Vec<CreatureBodyLink> {
        &self.child_links
    }
}

impl<'world_life> Creature<'world_life> {
    //pub fn sample_creature<'b>(physics_world: &'b PhysicsWorld) -> Creature<'b> {
    // let base_body:rigidbody = make rigid body with colliders and put into world
    // let base_link:link = Link(base_body)
    //
    // let fr_femur_body: rigidbody = make rigid body with colliders and put into world
    // let fr_femur_link:link = Link(fr_leg)
    // let fr_tibia: rigidbody = ...
    // let fr_tibia_link = Link(fr_tibia)
    // let joint = make a joint
    // chain_links(parent = fr_femur_link, child = fr_tibia_link, joint = joint, joint_name = "knee")
    //}

    pub fn get_raw_instances(&self) -> Vec<(MeshId, Vec<InstanceRaw>)> {
        let mut instance_holders: [Option<Vec<InstanceRaw>>; MeshId::MeshCount as usize] =
            [const { None }; MeshId::MeshCount as usize];

        let mut mesh_type_count = 0;

        for cur_link in self.all_links() {
            let link_color = cur_link.color;
            for collider_handle in self
                .world
                .bodies
                .get(cur_link.rigid_body)
                .unwrap()
                .colliders()
            {
                let collider = self.world.colliders.get(*collider_handle).unwrap();
                let shared_shape = collider.shared_shape();
                if let Ok(mesh_id) = MeshId::try_from(shared_shape.shape_type()) {
                    let scale = scale_from_shape(shared_shape).unwrap();
                    let translation = collider.translation();
                    let rotation = collider.rotation();

                    let model = Mat4::from_scale_rotation_translation(scale, rotation, translation);

                    let new_instance = InstanceRaw {
                        model: model.to_cols_array_2d(),
                        color: link_color.extend(1.0).to_array(),
                    };
                    if instance_holders[mesh_id as usize].is_none() {
                        instance_holders[mesh_id as usize] = Some(Vec::new());
                        mesh_type_count += 1;
                    }
                    instance_holders[mesh_id as usize]
                        .as_mut()
                        .unwrap()
                        .push(new_instance);
                }
            }
        }

        let mut out_vec = Vec::with_capacity(mesh_type_count);

        for (mesh_id, instance_list) in zip(MeshId::all_mesh_ids(), instance_holders) {
            if let Some(instances) = instance_list {
                out_vec.push((mesh_id, instances));
            }
        }

        return out_vec;
    }

    pub fn sample_creature<'a, 'b>(
        physics_world: &'a mut PhysicsWorld,
        position: Vec3,
    ) -> Creature<'b>
    where
        'a: 'b,
    {
        let base_body = RigidBodyBuilder::dynamic().translation(position);
        let torso_collider =
            ColliderBuilder::cylinder(3.0, 2.25).rotation(Vec3::new(0.0, 0.0, PI * 0.5));
        let head_collider = ColliderBuilder::ball(1.25).translation(Vec3::new(2.0, 0.5, 0.0));
        let (base_body_handle, _) = physics_world.insert(base_body, torso_collider);
        physics_world.insert_collider(head_collider, Some(base_body_handle));
        let mut body_link = CreatureBodyLink::new(
            base_body_handle,
            String::from("torso_and_head"),
            Vec3::new(0.7, 0.2, 0.2),
        );

        let femur_length = 3.0;
        let fr_femur_base = RigidBodyBuilder::dynamic();
        let fr_femur_collider = ColliderBuilder::cylinder(femur_length * 0.5, 0.75)
            .rotation(Vec3::new(0.0, 0.0, PI * 0.5))
            .translation(Vec3::new(femur_length * 0.5, 0.0, 0.0));
        let (femur_body_handle, _) = physics_world.insert(fr_femur_base, fr_femur_collider);
        let femur_link = CreatureBodyLink::new(
            femur_body_handle,
            String::from("fr_femur"),
            Vec3 {
                x: 0.2,
                y: 0.8,
                z: 0.1,
            },
        );

        let hip_joint_builder = SphericalJointBuilder::new()
            .local_anchor1(Vec3::new(1.5, -0.2, 1.0))
            .local_anchor2(Vec3::new(0.0, 0.0, 0.0));
        let hip_joint_handle = physics_world.impulse_joints.insert(
            base_body_handle,
            femur_body_handle,
            hip_joint_builder,
            true,
        );

        let joint = physics_world
            .impulse_joints
            .get_mut(hip_joint_handle, true)
            .unwrap()
            .data
            .set_motor_velocity(rapier3d::dynamics::JointAxis::AngY, 1.0, 0.5);

        body_link.add_child_link(femur_link, hip_joint_handle);

        let new_creature = Creature {
            base: body_link,
            world: physics_world,
        };

        new_creature
    }

    pub fn all_colliders(&'world_life self) -> CreatureColliderIter<'world_life> {
        CreatureColliderIter::new(&self.base, &self.world)
    }

    pub fn all_links(&'world_life self) -> CreatureLinkIter<'world_life> {
        CreatureLinkIter::new(&self.base, &self.world)
    }
}

pub struct CreatureLinkIter<'a> {
    dfs_stack: Vec<&'a CreatureBodyLink>,
    world: &'a PhysicsWorld,
}

impl<'a> CreatureLinkIter<'a> {
    pub fn new(
        creature_link: &'a CreatureBodyLink,
        world: &'a PhysicsWorld,
    ) -> CreatureLinkIter<'a> {
        let mut dfs_stack = Vec::new();
        dfs_stack.push(creature_link);

        CreatureLinkIter { dfs_stack, world }
    }
}

impl<'a> Iterator for CreatureLinkIter<'a> {
    type Item = &'a CreatureBodyLink;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(current_link) = self.dfs_stack.pop() {
            for child_link in current_link.all_child_links() {
                self.dfs_stack.push(child_link);
            }
            Some(current_link)
        } else {
            None
        }
    }
}

pub struct CreatureColliderIter<'a> {
    dfs_stack: Vec<&'a CreatureBodyLink>,
    current_link_iter: Option<std::slice::Iter<'a, ColliderHandle>>,

    world: &'a PhysicsWorld,
}

impl<'a> CreatureColliderIter<'a> {
    pub fn new(
        creature_link: &'a CreatureBodyLink,
        world: &'a PhysicsWorld,
    ) -> CreatureColliderIter<'a> {
        let mut dfs_stack = Vec::new();
        dfs_stack.push(creature_link);

        CreatureColliderIter {
            dfs_stack,
            current_link_iter: None,
            world,
        }
    }
}

impl<'a> Iterator for CreatureColliderIter<'a> {
    type Item = ColliderHandle;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(in_progress_link) = &mut self.current_link_iter {
            if let Some(collider_handle) = in_progress_link.next() {
                Some(*collider_handle)
            } else {
                self.current_link_iter = None;
                self.next()
            }
        } else {
            if let Some(visiting_link) = self.dfs_stack.pop() {
                for child_link in visiting_link.all_child_links() {
                    self.dfs_stack.push(child_link);
                }
                self.current_link_iter = Some(
                    self.world
                        .bodies
                        .get(visiting_link.rigid_body)
                        .unwrap()
                        .colliders()
                        .iter(),
                );
                self.next()
            } else {
                None
            }
        }
    }
}

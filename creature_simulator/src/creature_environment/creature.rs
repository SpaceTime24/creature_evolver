use glam::Vec3;
use rapier3d::{
    dynamics::{ImpulseJointHandle, JointAxis, RigidBodyBuilder, RigidBodyHandle},
    pipeline::PhysicsWorld,
};

use crate::{
    creature_environment::blueprint::{CreatureBlueprint, JointKind},
    graphical_app::mesh::{InstanceRaw, MeshId, ModelFrame, instance_from_collider},
    neural_net::network::NeuralNet,
};

/// One rigid body of the creature (may own several colliders).
pub struct CreatureBodyLink {
    rigid_body: RigidBodyHandle,
    label: String,
    color: Vec3,
}

impl CreatureBodyLink {
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// One controllable degree of freedom.
struct JointDof {
    joint_handle: ImpulseJointHandle,
    ///  rapier joint-frame axis.
    drive_axis: JointAxis,
    target_velocity_scale: f32,
    /// Motor strength.
    gain: f32,
    parent_body: RigidBodyHandle,
    child_body: RigidBodyHandle,
    /// Physical rotation axis (parent-local)
    sense_axis: Vec3,
}

//Creature local coordinates: +x front, +z right, +y up
pub struct Creature {
    /// Flat list of body links; index 0 is always the root.
    links: Vec<CreatureBodyLink>,
    joint_dofs: Vec<JointDof>,
    observation_size: usize,
    neural_net: Option<NeuralNet>,
}

impl Creature {
    pub fn from_blueprint(
        blueprint: &CreatureBlueprint,
        world: &mut PhysicsWorld,
        spawn: Vec3,
    ) -> Creature {
        let mut links = Vec::with_capacity(blueprint.links.len());
        let mut handles = Vec::with_capacity(blueprint.links.len());

        for link_spec in &blueprint.links {
            let body = RigidBodyBuilder::dynamic()
                .translation(spawn + Vec3::from_array(link_spec.translation));
            let body_handle = world.insert_body(body);
            for collider_spec in &link_spec.colliders {
                world.insert_collider(collider_spec.build(), Some(body_handle));
            }
            links.push(CreatureBodyLink {
                rigid_body: body_handle,
                label: link_spec.label.clone(),
                color: Vec3::from_array(link_spec.color),
            });
            handles.push(body_handle);
        }

        let mut joint_dofs = Vec::new();
        for joint_spec in &blueprint.joints {
            let (Some(&parent), Some(&child)) = (
                handles.get(joint_spec.parent),
                handles.get(joint_spec.child),
            ) else {
                eprintln!(
                    "creature blueprint: joint references out-of-range link index ({} -> {}); skipping",
                    joint_spec.parent, joint_spec.child
                );
                continue;
            };
            let joint_handle = world.insert_impulse_joint(parent, child, joint_spec.build());
            for motor in &joint_spec.motors {
                // A revolute joint's free DOF is always the joint-frame AngX,
                let (drive_axis, sense_axis) = match &joint_spec.kind {
                    JointKind::Revolute { axis } => {
                        (JointAxis::AngX, Vec3::from_array(*axis).normalize_or_zero())
                    }
                    _ => (motor.axis.joint_axis(), motor.axis.unit_vec()),
                };
                joint_dofs.push(JointDof {
                    joint_handle,
                    drive_axis,
                    target_velocity_scale: motor.target_velocity_scale,
                    gain: motor.gain,
                    parent_body: parent,
                    child_body: child,
                    sense_axis,
                });
            }
        }

        Creature {
            links,
            joint_dofs,
            observation_size: blueprint.observation_size(),
            neural_net: None,
        }
    }

    pub fn add_brain(&mut self, neural_net: NeuralNet) {
        self.neural_net = Some(neural_net);
    }

    pub fn action_size(&self) -> usize {
        self.joint_dofs.len()
    }

    pub fn observation_size(&self) -> usize {
        self.observation_size
    }

    /// Root always index 0
    pub fn root_body(&self) -> RigidBodyHandle {
        self.links[0].rigid_body
    }

    pub fn root_position(&self, world: &PhysicsWorld) -> Vec3 {
        world
            .bodies
            .get(self.root_body())
            .map(|b| b.translation())
            .unwrap_or(Vec3::ZERO)
    }

    /// How level the torso is: the alignment of the torso's local up-axis with
    /// world-up. +1 fully upright, 0 on its side, -1 flipped.
    pub fn torso_uprightness(&self, world: &PhysicsWorld) -> f32 {
        let world_up = (-world.gravity).normalize_or_zero();
        match world.bodies.get(self.root_body()) {
            Some(body) => (*body.rotation() * Vec3::Y).dot(world_up),
            None => 0.0,
        }
    }

    /// Read the creature's sensors into a flat observation vector. Layout:
    ///   [0..3)                    gravity direction in the root body's local frame
    ///   next 2 per motor DOF      [joint angle, joint angular velocity]
    ///   final 1 per link          contact impulse magnitude on that link
    pub fn sense(&self, world: &PhysicsWorld) -> Vec<f32> {
        let mut obs = Vec::with_capacity(self.observation_size);

        // Orientation relative to gravity
        let gravity_dir = world.gravity.normalize_or_zero();
        if let Some(root) = world.bodies.get(self.root_body()) {
            let local_gravity = root.rotation().conjugate() * gravity_dir;
            obs.extend_from_slice(&[local_gravity.x, local_gravity.y, local_gravity.z]);
        } else {
            obs.extend_from_slice(&[0.0, 0.0, 0.0]);
        }

        // Sense motor rotation speed and position for all controllable joints
        for dof in &self.joint_dofs {
            let (angle, velocity) = match (
                world.bodies.get(dof.parent_body),
                world.bodies.get(dof.child_body),
            ) {
                (Some(parent), Some(child)) => {
                    let parent_rot = *parent.rotation();
                    // Child orientation in parents frame.
                    let relative_rot = parent_rot.conjugate() * *child.rotation();
                    let twist = Vec3::new(relative_rot.x, relative_rot.y, relative_rot.z)
                        .dot(dof.sense_axis);
                    let angle = 2.0 * twist.atan2(relative_rot.w);

                    let relative_vel = child.angvel() - parent.angvel();
                    let velocity = relative_vel.dot(parent_rot * dof.sense_axis);
                    (angle, velocity)
                }
                _ => (0.0, 0.0),
            };
            obs.push(angle);
            obs.push(velocity);
        }

        // Per link: total contact impulse magnitude across the link's colliders. normal force
        // but not exactly
        for link in &self.links {
            let mut contact_force = 0.0;
            if let Some(body) = world.bodies.get(link.rigid_body) {
                for collider_handle in body.colliders() {
                    for pair in world.contact_pairs_with(*collider_handle) {
                        contact_force += pair.total_impulse_magnitude();
                    }
                }
            }
            obs.push(contact_force);
        }

        obs
    }

    /// Apply output to motors
    pub fn actuate(&self, world: &mut PhysicsWorld, actions: &[f32]) {
        for (dof, action) in self.joint_dofs.iter().zip(actions) {
            let target_velocity = action.clamp(-1.0, 1.0) * dof.target_velocity_scale;
            if let Some(joint) = world.impulse_joints.get_mut(dof.joint_handle, true) {
                joint
                    .data
                    .set_motor_velocity(dof.drive_axis, target_velocity, dof.gain);
            }
        }
    }

    /// One control tick: sense -> network -> actuate
    pub fn control_step(&self, world: &mut PhysicsWorld) {
        if let Some(net) = &self.neural_net {
            let observation = self.sense(world);
            let actions = net.apply_network(&observation);
            self.actuate(world, &actions);
        }
    }

    /// Build the per-mesh instance lists used to render this creature.
    pub fn get_raw_instances(&self, world: &PhysicsWorld) -> ModelFrame {
        let mut instance_holders: [Vec<InstanceRaw>; MeshId::MeshCount as usize] =
            [const { Vec::new() }; MeshId::MeshCount as usize];

        for link in &self.links {
            let Some(body) = world.bodies.get(link.rigid_body) else {
                continue;
            };
            for collider_handle in body.colliders() {
                let Some(collider) = world.colliders.get(*collider_handle) else {
                    continue;
                };
                if let Ok((mesh_id, new_instance)) = instance_from_collider(collider, link.color) {
                    instance_holders[mesh_id as usize].push(new_instance);
                }
            }
        }

        let mut out_vec = Vec::new();
        for (mesh_id, instance_list) in std::iter::zip(MeshId::all_mesh_ids(), instance_holders) {
            if !instance_list.is_empty() {
                out_vec.push((mesh_id, instance_list));
            }
        }
        out_vec
    }
}

#[cfg(test)]
mod motor_tests {
    use super::*;
    use crate::creature_environment::blueprint::{
        ColliderSpec, CreatureBlueprint, JointSpec, LinkSpec, MotorAxis, MotorSpec, ShapeSpec,
    };

    fn two_link_hinge(gain: f32, limit: Option<[f32; 2]>) -> CreatureBlueprint {
        let cuboid = |hx: f32, hy: f32, hz: f32| ColliderSpec {
            shape: ShapeSpec::Cuboid {
                half_extents: [hx, hy, hz],
            },
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
        };
        CreatureBlueprint {
            hidden_size: 4,
            links: vec![
                LinkSpec {
                    label: "torso".into(),
                    color: [0.7, 0.2, 0.2],
                    translation: [0.0, 0.0, 0.0],
                    colliders: vec![cuboid(1.0, 0.3, 0.6)],
                },
                LinkSpec {
                    label: "leg".into(),
                    color: [0.2, 0.8, 0.3],
                    translation: [0.0, -1.0, 0.0],
                    colliders: vec![cuboid(0.2, 0.6, 0.2)],
                },
            ],
            joints: vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Revolute {
                    axis: [0.0, 0.0, 1.0],
                },
                parent_anchor: [0.0, -0.5, 0.0],
                child_anchor: [0.0, 0.5, 0.0],
                motors: vec![MotorSpec {
                    axis: MotorAxis::AngX,
                    max_force: 1.0e4,
                    target_velocity_scale: 3.0,
                    gain,
                    limit,
                }],
            }],
        }
    }

    fn spun_rate(gain: f32) -> f32 {
        let mut world = PhysicsWorld::new();
        world.gravity = Vec3::ZERO;
        let creature =
            Creature::from_blueprint(&two_link_hinge(gain, None), &mut world, Vec3::ZERO);
        for _ in 0..120 {
            creature.actuate(&mut world, &[1.0]);
            world.step();
        }
        let parent = world.bodies.get(creature.links[0].rigid_body).unwrap();
        let child = world.bodies.get(creature.links[1].rigid_body).unwrap();
        (child.angvel() - parent.angvel()).z.abs()
    }

    fn limited_angle() -> f32 {
        let limit = [-0.5f32, 0.5f32];
        let mut world = PhysicsWorld::new();
        world.gravity = Vec3::ZERO;
        let creature =
            Creature::from_blueprint(&two_link_hinge(1000.0, Some(limit)), &mut world, Vec3::ZERO);
        // Drive continuously toward +limit, should stop at the stop, not spin past.
        for _ in 0..180 {
            creature.actuate(&mut world, &[1.0]);
            world.step();
        }
        // sense() reports [angle, velocity] per DOF right after the 3 gravity slots.
        creature.sense(&world)[3]
    }

    #[test]
    fn motor_respects_range_limit() {
        let angle = limited_angle();
        assert!(
            angle.abs() <= 0.6,
            "expected joint to stop near the 0.5 rad limit, got {angle}"
        );
    }

    #[test]
    fn strong_gain_tracks_target_velocity() {
        let rate = spun_rate(1000.0);
        assert!(rate > 1.0, "expected strong motor to spin up, got {rate}");
    }

    #[test]
    fn weak_gain_ragdolls() {
        let rate = spun_rate(1.0);
        assert!(rate < 0.5, "expected weak motor to stay limp, got {rate}");
    }
}

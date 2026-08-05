use glam::Vec3;
use rapier3d::{
    dynamics::{JointAxis, RevoluteJointBuilder, SphericalJointBuilder, SpringCoefficients},
    geometry::{Collider, ColliderBuilder},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum ShapeSpec {
    Cuboid { half_extents: [f32; 3] },
    Ball { radius: f32 },
    Cylinder { half_height: f32, radius: f32 },
}

impl ShapeSpec {
    fn builder(&self) -> ColliderBuilder {
        match *self {
            ShapeSpec::Cuboid { half_extents } => {
                ColliderBuilder::cuboid(half_extents[0], half_extents[1], half_extents[2])
            }
            ShapeSpec::Ball { radius } => ColliderBuilder::ball(radius),
            ShapeSpec::Cylinder {
                half_height,
                radius,
            } => ColliderBuilder::cylinder(half_height, radius),
        }
    }
}

fn zero3() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}

fn default_color() -> [f32; 3] {
    [0.7, 0.2, 0.2]
}

/// collision shape placed relative to its owning body link.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColliderSpec {
    #[serde(flatten)]
    pub shape: ShapeSpec,
    /// Offset from the body's origin.
    #[serde(default = "zero3")]
    pub translation: [f32; 3],
    #[serde(default = "zero3")]
    pub rotation: [f32; 3],
}

impl ColliderSpec {
    pub fn build(&self) -> Collider {
        self.shape
            .builder()
            .translation(Vec3::from_array(self.translation))
            .rotation(Vec3::from_array(self.rotation))
            .friction(1.0)
            .build()
    }
}

/// One rigid body of the creature. May carry several colliders.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkSpec {
    pub label: String,
    #[serde(default = "default_color")]
    pub color: [f32; 3],
    #[serde(default = "zero3")]
    pub translation: [f32; 3],
    pub colliders: Vec<ColliderSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JointKind {
    /// Use axis_x to drive
    Revolute {
        axis: [f32; 3],
    },
    Spherical,
    Fixed,
}

/// Which local rotational axis a motor drives.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotorAxis {
    AngX,
    AngY,
    AngZ,
}

impl MotorAxis {
    pub fn joint_axis(self) -> JointAxis {
        match self {
            MotorAxis::AngX => JointAxis::AngX,
            MotorAxis::AngY => JointAxis::AngY,
            MotorAxis::AngZ => JointAxis::AngZ,
        }
    }

    pub fn unit_vec(self) -> Vec3 {
        match self {
            MotorAxis::AngX => Vec3::X,
            MotorAxis::AngY => Vec3::Y,
            MotorAxis::AngZ => Vec3::Z,
        }
    }
}

/// a single controllable degree of freedom on a joint
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotorSpec {
    pub axis: MotorAxis,

    pub max_force: f32,
    /// scale the network's [-1, 1] output into a target angular velocity.
    #[serde(default = "one")]
    pub target_velocity_scale: f32,

    #[serde(default = "default_motor_gain")]
    pub gain: f32,
    ///motor angular limits
    #[serde(default)]
    pub limit: Option<[f32; 2]>,
}

fn one() -> f32 {
    1.0
}

fn default_motor_gain() -> f32 {
    4000.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointSpec {
    pub parent: usize,
    pub child: usize,
    #[serde(flatten)]
    pub kind: JointKind,
    /// Anchor on the parent body (parent-local).
    #[serde(default = "zero3")]
    pub parent_anchor: [f32; 3],
    /// Anchor on the child body (child-local).
    #[serde(default = "zero3")]
    pub child_anchor: [f32; 3],
    /// Controllable DOFs on this joint
    #[serde(default)]
    pub motors: Vec<MotorSpec>,
}

impl JointSpec {
    pub fn build(&self) -> rapier3d::dynamics::GenericJoint {
        let anchor1 = Vec3::from_array(self.parent_anchor);
        let anchor2 = Vec3::from_array(self.child_anchor);
        match &self.kind {
            JointKind::Revolute { axis } => {
                let mut builder = RevoluteJointBuilder::new(Vec3::from_array(*axis))
                    .local_anchor1(anchor1)
                    .local_anchor2(anchor2)
                    //.softness(SpringCoefficients::new(1.0e6, 10000.0))
                    ;
                //Revolute alway Use `ang_x`  to drive it.
                for motor in &self.motors {
                    builder = builder.motor_max_force(motor.max_force);
                    if let Some(limit) = motor.limit {
                        builder = builder.limits(limit);
                    }
                }
                builder.build().into()
            }
            JointKind::Spherical => {
                let mut builder = SphericalJointBuilder::new()
                    .local_anchor1(anchor1)
                    .local_anchor2(anchor2);
                for motor in &self.motors {
                    builder = builder.motor_max_force(motor.axis.joint_axis(), motor.max_force);
                    if let Some(limit) = motor.limit {
                        builder = builder.limits(motor.axis.joint_axis(), limit);
                    }
                }
                builder.build().into()
            }
            JointKind::Fixed => rapier3d::dynamics::FixedJointBuilder::new()
                .local_anchor1(anchor1)
                .local_anchor2(anchor2)
                .build()
                .into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreatureBlueprint {
    pub links: Vec<LinkSpec>,
    #[serde(default)]
    pub joints: Vec<JointSpec>,
    #[serde(default = "default_hidden")]
    pub hidden_size: usize,
}

fn default_hidden() -> usize {
    16
}

impl CreatureBlueprint {
    /// One controllable value for every dof of all joints.
    pub fn action_size(&self) -> usize {
        self.joints.iter().map(|j| j.motors.len()).sum()
    }

    /// Length of the observation vector fed to the controller:
    ///   3 (gravity direction rel to creature)
    /// + 2 per motor DOF (joint angle and angular velocity)
    /// + 1 per link (contact force magnitude on that link).
    pub fn observation_size(&self) -> usize {
        3 + 2 * self.action_size() + self.links.len()
    }

    pub fn from_json(json: &str) -> Result<CreatureBlueprint, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvColliderSpec {
    #[serde(flatten)]
    pub collider: ColliderSpec,
    #[serde(default = "default_color")]
    pub color: [f32; 3],
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FitnessSpec {
    /// Reward per unit of net forward (+X) displacement of the torso.
    #[serde(default = "one")]
    pub forward_weight: f32,
    /// Reward for keeping the torso level.
    #[serde(default = "default_upright_weight")]
    pub upright_weight: f32,
    /// Reward for keeping the torso near target height
    #[serde(default = "default_height_weight")]
    pub height_weight: f32,
    /// Torso height at which the height reward saturates.
    #[serde(default = "default_target_clearance")]
    pub target_clearance: f32,
    /// World Y treated as ground level when measuring torso clearance.
    #[serde(default)]
    pub ground_height: f32,
}

impl Default for FitnessSpec {
    fn default() -> FitnessSpec {
        FitnessSpec {
            forward_weight: 1.0,
            upright_weight: default_upright_weight(),
            height_weight: default_height_weight(),
            target_clearance: default_target_clearance(),
            ground_height: 0.0,
        }
    }
}

fn default_upright_weight() -> f32 {
    2.0
}

fn default_height_weight() -> f32 {
    1.5
}

fn default_target_clearance() -> f32 {
    1.5
}

/// Full description of the fixed world the creatures live in.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvironmentBlueprint {
    pub colliders: Vec<EnvColliderSpec>,
    #[serde(default = "default_gravity")]
    pub gravity: [f32; 3],

    #[serde(default)]
    pub fitness: FitnessSpec,
}

fn default_gravity() -> [f32; 3] {
    [0.0, -9.81, 0.0]
}

impl EnvironmentBlueprint {
    pub fn from_json(json: &str) -> Result<EnvironmentBlueprint, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CREATURE_JSON: &str = r#"{
        "hidden_size": 8,
        "links": [
            { "label": "torso", "colliders": [ { "shape": "cuboid", "half_extents": [1.5, 0.4, 0.8] } ] },
            { "label": "leg", "translation": [1.2, -0.9, 0.0],
              "colliders": [ { "shape": "cuboid", "half_extents": [0.2, 0.8, 0.2], "translation": [0.0, 0.0, 0.0] } ] }
        ],
        "joints": [
            { "kind": "revolute", "axis": [0.0, 0.0, 1.0], "parent": 0, "child": 1,
              "parent_anchor": [1.2, -0.4, 0.0], "child_anchor": [0.0, 0.8, 0.0],
              "motors": [ { "axis": "ang_x", "max_force": 60.0 } ] }
        ]
    }"#;

    const ENV_JSON: &str = r#"{
        "colliders": [
            { "shape": "cuboid", "half_extents": [50.0, 0.5, 50.0], "translation": [0.0, -1.0, 0.0], "color": [0.3, 0.3, 0.35] }
        ]
    }"#;

    #[test]
    fn parses_creature_and_reports_io_sizes() {
        let bp = CreatureBlueprint::from_json(CREATURE_JSON).expect("creature parses");
        assert_eq!(bp.links.len(), 2);
        assert_eq!(bp.joints.len(), 1);
        assert_eq!(bp.action_size(), 1); // one motorized DOF
        // 3 (gravity) + 2 (angle+velocity per DOF) + 2 (links) = 7
        assert_eq!(bp.observation_size(), 7);
    }

    #[test]
    fn parses_environment_with_defaults() {
        let env = EnvironmentBlueprint::from_json(ENV_JSON).expect("env parses");
        assert_eq!(env.colliders.len(), 1);
        assert_eq!(env.gravity, [0.0, -9.81, 0.0]);
    }
}

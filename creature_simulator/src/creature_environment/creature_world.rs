use glam::Vec3;
use rapier3d::pipeline::PhysicsWorld;

use crate::{
    creature_environment::{blueprint::CreatureBlueprint, creature::Creature},
    neural_net::network::NeuralNet,
};

pub struct CreatureWorld {
    pub creature: Option<Creature>,
    pub physics: PhysicsWorld,
}

impl CreatureWorld {
    /// A fresh, empty world with the given gravity.
    pub fn new(gravity: Vec3) -> CreatureWorld {
        let mut physics = PhysicsWorld::new();
        physics.gravity = gravity;
        CreatureWorld {
            physics,
            creature: None,
        }
    }

    pub fn add_creature(
        &mut self,
        blueprint: &CreatureBlueprint,
        genome: &[f32],
        spawn: Vec3,
    ) -> Result<(), String> {
        let mut creature = Creature::from_blueprint(blueprint, &mut self.physics, spawn);
        let net = NeuralNet::from_genome(
            genome,
            creature.observation_size(),
            blueprint.hidden_size,
            creature.action_size(),
        )?;
        creature.add_brain(net);
        self.creature = Some(creature);
        Ok(())
    }

    /// Advance the simulation one tick: run the controller, then step physics.
    pub fn step(&mut self) {
        if let Some(creature) = &self.creature {
            creature.control_step(&mut self.physics);
        }
        self.physics.step();
    }
}

use std::iter::zip;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use glam::Vec3;
use rapier3d::{dynamics::RigidBodyBuilder, geometry::Collider, pipeline::PhysicsWorld};

use crate::graphical_app::mesh::ModelFrame;
use crate::{
    creature_environment::blueprint::{CreatureBlueprint, EnvironmentBlueprint},
    graphical_app::mesh::{InstanceRaw, MeshId, instance_from_collider},
    neural_net::network::NeuralNet,
    simulation_running::threading::{SimConfig, SimJob, SimResult, SimulationThreadHandle},
};

/// fixed geometry every creature world is made with. Holds
/// template colliders and precomutee renderable instances
pub struct StaticEnvironment {
    template_colliders: Vec<Collider>,
    instances_to_render: ModelFrame,
}

impl StaticEnvironment {
    pub fn from_blueprint(blueprint: &EnvironmentBlueprint) -> StaticEnvironment {
        let mut template_colliders = Vec::with_capacity(blueprint.colliders.len());
        let mut instance_holders: [Vec<InstanceRaw>; MeshId::MeshCount as usize] =
            [const { Vec::new() }; MeshId::MeshCount as usize];

        for spec in &blueprint.colliders {
            let collider = spec.collider.build();
            if let Ok((mesh_id, instance)) =
                instance_from_collider(&collider, Vec3::from_array(spec.color))
            {
                instance_holders[mesh_id as usize].push(instance);
            }
            template_colliders.push(collider);
        }

        let mut instances_to_render = Vec::new();
        for (mesh_id, instance_list) in zip(MeshId::all_mesh_ids(), instance_holders) {
            if !instance_list.is_empty() {
                instances_to_render.push((mesh_id, instance_list));
            }
        }

        StaticEnvironment {
            template_colliders,
            instances_to_render,
        }
    }

    pub fn get_mesh_instances(&self) -> &ModelFrame {
        &self.instances_to_render
    }

    /// Insert this environment's fixed geometry into a fresh physics world.
    pub fn populate(&self, world: &mut PhysicsWorld) {
        let fixed_body = world.insert_body(RigidBodyBuilder::fixed());
        for collider in &self.template_colliders {
            world.insert_collider(collider.clone(), Some(fixed_body));
        }
    }
}

/// Owns the pool of simulation worker threads and the channels used to feed them
/// genomes and collect fitnesses.
pub struct CreatureParty {
    pub static_environment: Arc<StaticEnvironment>,
    simulation_threads: Vec<SimulationThreadHandle>,
    job_sender: Option<crossbeam::channel::Sender<SimJob>>,
    result_receiver: crossbeam::channel::Receiver<SimResult>,
    terminate: Arc<AtomicBool>,
    config: Arc<SimConfig>,
}

impl CreatureParty {
    pub fn new(
        environment: EnvironmentBlueprint,
        creature: CreatureBlueprint,
        worker_count: usize,
        max_steps: u32,
        spawn: Vec3,
    ) -> CreatureParty {
        let static_environment = Arc::new(StaticEnvironment::from_blueprint(&environment));
        let gravity = Vec3::from_array(environment.gravity);
        let fitness = environment.fitness;

        let config = Arc::new(SimConfig {
            environment: Arc::clone(&static_environment),
            creature,
            gravity,
            spawn,
            max_steps,
            fitness,
        });

        let (job_sender, job_receiver) = crossbeam::channel::unbounded();
        let (result_sender, result_receiver) = crossbeam::channel::unbounded();
        let terminate = Arc::new(AtomicBool::new(false));

        let mut simulation_threads = Vec::with_capacity(worker_count.max(1));
        for _ in 0..worker_count.max(1) {
            let mut handle = SimulationThreadHandle::new();
            handle.spawn(
                Arc::clone(&config),
                job_receiver.clone(),
                result_sender.clone(),
                Arc::clone(&terminate),
            );
            simulation_threads.push(handle);
        }

        drop(result_sender);

        CreatureParty {
            static_environment,
            simulation_threads,
            job_sender: Some(job_sender),
            result_receiver,
            terminate,
            config,
        }
    }

    /// Number of `f32` weights each genome must contain for this creature.
    pub fn genome_size(&self) -> usize {
        NeuralNet::genome_len(
            self.config.creature.observation_size(),
            self.config.creature.hidden_size,
            self.config.creature.action_size(),
        )
    }

    pub fn observation_size(&self) -> usize {
        self.config.creature.observation_size()
    }

    pub fn action_size(&self) -> usize {
        self.config.creature.action_size()
    }

    /// Evaluate a whole generation.
    pub fn evaluate(&self, genomes: Vec<Vec<f32>>) -> Vec<f32> {
        let count = genomes.len();
        let mut fitness = vec![f32::NAN; count];

        let Some(sender) = self.job_sender.as_ref() else {
            return fitness;
        };
        for (index, genome) in genomes.into_iter().enumerate() {
            if sender.send(SimJob { index, genome }).is_err() {
                return fitness;
            }
        }

        for _ in 0..count {
            match self.result_receiver.recv() {
                Ok(result) if result.index < count => fitness[result.index] = result.fitness,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        fitness
    }

    pub fn channels(
        &self,
    ) -> Option<(
        crossbeam::channel::Sender<SimJob>,
        crossbeam::channel::Receiver<SimResult>,
    )> {
        self.job_sender
            .as_ref()
            .map(|sender| (sender.clone(), self.result_receiver.clone()))
    }

    pub fn thread_handles(&mut self) -> &mut Vec<SimulationThreadHandle> {
        &mut self.simulation_threads
    }

    pub fn simulation_count(&self) -> usize {
        self.simulation_threads.len()
    }

    /// Make worker `index` the one whose simulation is drawn on screen.
    pub fn make_renderable(&mut self, index: usize) {
        if let Some(handle) = self.simulation_threads.get_mut(index) {
            handle.make_renderable();
        }
    }
}

impl Drop for CreatureParty {
    fn drop(&mut self) {
        self.terminate.store(true, Ordering::Relaxed);
        // Dropping the job sender disconnects the channel so idle workers exit.
        self.job_sender.take();
        for handle in &mut self.simulation_threads {
            handle.join();
        }
    }
}

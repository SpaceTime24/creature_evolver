use std::{
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use crossbeam::{
    channel::{Receiver, RecvTimeoutError, Sender},
    queue::ArrayQueue,
};
use glam::Vec3;

use crate::{
    creature_environment::{
        blueprint::{CreatureBlueprint, FitnessSpec},
        creature_world::CreatureWorld,
        static_environment::StaticEnvironment,
    },
    graphical_app::mesh::ModelFrame,
};

const THREAD_PUBLISH_AHEAD: usize = 2;
const RENDER_STEP: Duration = Duration::from_millis(16);

pub struct SimConfig {
    pub environment: Arc<StaticEnvironment>,
    pub creature: CreatureBlueprint,
    pub gravity: Vec3,
    pub spawn: Vec3,
    pub max_steps: u32,
    pub fitness: FitnessSpec,
}

pub struct SimJob {
    pub index: usize,
    pub genome: Vec<f32>,
}

/// The outcome of one evaluation.
pub struct SimResult {
    pub index: usize,
    pub fitness: f32,
}

pub struct SimulationThreadHandle {
    join_handle: Option<JoinHandle<()>>,
    /// Present only for a renderable worker
    models_to_draw: Arc<OnceLock<ArrayQueue<ModelFrame>>>,
    last_drawn_models: Option<ModelFrame>,
}

impl SimulationThreadHandle {
    pub fn new() -> SimulationThreadHandle {
        SimulationThreadHandle {
            join_handle: None,
            models_to_draw: Arc::new(OnceLock::new()),
            last_drawn_models: None,
        }
    }

    pub fn get_new_instances(&mut self) -> Option<&ModelFrame> {
        if let Some(queue) = self.models_to_draw.get() {
            if let Some(new_instances) = queue.pop() {
                self.last_drawn_models.replace(new_instances);
            }
            self.last_drawn_models.as_ref()
        } else {
            None
        }
    }

    /// Mark this worker as one whose simulation is drawn on screen.
    pub fn make_renderable(&mut self) {
        self.models_to_draw
            .get_or_init(|| ArrayQueue::new(THREAD_PUBLISH_AHEAD));
    }

    /// Start the worker loop, run until job sender disconnects or terminated
    pub fn spawn(
        &mut self,
        config: Arc<SimConfig>,
        jobs: Receiver<SimJob>,
        results: Sender<SimResult>,
        terminate: Arc<AtomicBool>,
    ) {
        let models_to_draw = Arc::clone(&self.models_to_draw);
        self.join_handle = Some(thread::spawn(move || {
            worker_loop(config, jobs, results, terminate, models_to_draw);
        }));
    }

    pub fn join(&mut self) {
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

fn worker_loop(
    config: Arc<SimConfig>,
    jobs: Receiver<SimJob>,
    results: Sender<SimResult>,
    terminate: Arc<AtomicBool>,
    models_to_draw: Arc<OnceLock<ArrayQueue<ModelFrame>>>,
) {
    loop {
        // Time out so that we can detect a terminate command.
        let job = match jobs.recv_timeout(Duration::from_millis(50)) {
            Ok(job) => job,
            Err(RecvTimeoutError::Timeout) => {
                if terminate.load(Ordering::Relaxed) {
                    break;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };

        let fitness = run_evaluation(&config, &job, &models_to_draw, &terminate);
        if results
            .send(SimResult {
                index: job.index,
                fitness,
            })
            .is_err()
        {
            break;
        }
    }
}

/// Build a fresh world for one genome, simulate it, and return its fitness.
fn run_evaluation(
    config: &SimConfig,
    job: &SimJob,
    models_to_draw: &OnceLock<ArrayQueue<ModelFrame>>,
    terminate: &AtomicBool,
) -> f32 {
    let mut world = CreatureWorld::new(config.gravity);
    config.environment.populate(&mut world.physics);

    if let Err(err) = world.add_creature(&config.creature, &job.genome, config.spawn) {
        eprintln!("simulation: failed to build creature: {err}");
        return f32::NEG_INFINITY;
    }

    let mut upright_sum = 0.0f32;
    let mut clearance_sum = 0.0f32;
    let mut samples = 0u32;

    let renderable = models_to_draw.get();
    for _ in 0..config.max_steps {
        if terminate.load(Ordering::Relaxed) {
            break;
        }
        world.step();

        if let Some(creature) = &world.creature {
            upright_sum += creature.torso_uprightness(&world.physics);
            let clearance = creature.root_position(&world.physics).y - config.fitness.ground_height;
            let clearance_factor = (clearance / config.fitness.target_clearance).clamp(0.0, 1.0);
            clearance_sum += clearance_factor;
            samples += 1;
        }

        if let Some(queue) = renderable {
            if let Some(creature) = &world.creature {
                if !queue.is_full() {
                    let _ = queue.push(creature.get_raw_instances(&world.physics));
                }
            }
            // Only the on-screen worker paces itself; headless workers run flat out.
            thread::sleep(RENDER_STEP);
        }
    }

    shaped_fitness(&world, config, upright_sum, clearance_sum, samples)
}

fn shaped_fitness(
    world: &CreatureWorld,
    config: &SimConfig,
    upright_sum: f32,
    clearance_sum: f32,
    samples: u32,
) -> f32 {
    let Some(creature) = &world.creature else {
        return 0.0;
    };
    let f = &config.fitness;

    let forward = creature.root_position(&world.physics).x - config.spawn.x;
    let mean_upright = if samples > 0 {
        upright_sum / samples as f32
    } else {
        0.0
    };
    let mean_clearance = if samples > 0 {
        clearance_sum / samples as f32
    } else {
        0.0
    };

    f.forward_weight * forward + f.upright_weight * mean_upright + f.height_weight * mean_clearance
}

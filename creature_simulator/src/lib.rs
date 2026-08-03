pub mod creature_environment;
pub mod graphical_app;
pub mod neural_net;
pub mod simulation_running;

use std::fs;
use std::sync::{Arc, RwLock};
use std::thread;

use glam::Vec3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use winit::event_loop::EventLoopBuilder;
use winit::platform::wayland::EventLoopBuilderExtWayland;

use crate::{
    creature_environment::{
        blueprint::{CreatureBlueprint, EnvironmentBlueprint},
        static_environment::CreatureParty,
    },
    graphical_app::app_handler::App,
    simulation_running::threading::{SimJob, SimResult},
};

/// A set of parallel creature simulations sharing one environment and creature
/// body plan. This is the object a Python genetic-algorithm driver interacts
/// with: build it from config files, ask it for the required genome size, then
/// repeatedly `evaluate(...)` whole generations of genomes to get fitnesses.
///
/// Evaluations run on a background thread pool, and an optional graphical view
/// runs on its own thread, so Python stays free to run GA logic in between.
#[pyclass]
struct SimulationSet {
    party: Arc<RwLock<CreatureParty>>,
    job_sender: crossbeam::channel::Sender<SimJob>,
    result_receiver: crossbeam::channel::Receiver<SimResult>,
    genome_size: usize,
    observation_size: usize,
    action_size: usize,
    window_active: bool,
}

#[pymethods]
impl SimulationSet {
    /// Build a simulation set from a creature config and an environment config,
    /// both JSON files. `workers` background threads run evaluations in parallel;
    /// each rollout runs for `max_steps` physics steps.
    #[staticmethod]
    #[pyo3(signature = (creature_config, environment_config, workers=8, max_steps=600, spawn_height=2.0))]
    fn from_config_files(
        creature_config: &str,
        environment_config: &str,
        workers: usize,
        max_steps: u32,
        spawn_height: f32,
    ) -> PyResult<SimulationSet> {
        let creature_json = fs::read_to_string(creature_config)
            .map_err(|e| PyValueError::new_err(format!("reading {creature_config}: {e}")))?;
        let environment_json = fs::read_to_string(environment_config)
            .map_err(|e| PyValueError::new_err(format!("reading {environment_config}: {e}")))?;
        Self::from_config_strings(
            &creature_json,
            &environment_json,
            workers,
            max_steps,
            spawn_height,
        )
    }

    /// Same as [`from_config_files`], but the configs are passed as JSON strings.
    #[staticmethod]
    #[pyo3(signature = (creature_json, environment_json, workers=8, max_steps=600, spawn_height=2.0))]
    fn from_config_strings(
        creature_json: &str,
        environment_json: &str,
        workers: usize,
        max_steps: u32,
        spawn_height: f32,
    ) -> PyResult<SimulationSet> {
        let creature = CreatureBlueprint::from_json(creature_json)
            .map_err(|e| PyValueError::new_err(format!("parsing creature config: {e}")))?;
        let environment = EnvironmentBlueprint::from_json(environment_json)
            .map_err(|e| PyValueError::new_err(format!("parsing environment config: {e}")))?;

        let spawn = Vec3::new(0.0, spawn_height, 0.0);
        let party = CreatureParty::new(environment, creature, workers, max_steps, spawn);

        let (job_sender, result_receiver) = party
            .channels()
            .ok_or_else(|| PyValueError::new_err("simulation party has no active workers"))?;
        let genome_size = party.genome_size();
        let observation_size = party.observation_size();
        let action_size = party.action_size();

        Ok(SimulationSet {
            party: Arc::new(RwLock::new(party)),
            job_sender,
            result_receiver,
            genome_size,
            observation_size,
            action_size,
            window_active: false,
        })
    }

    /// Number of `f32` weights each genome must contain. The GA sizes its
    /// individuals to this.
    #[getter]
    fn genome_size(&self) -> usize {
        self.genome_size
    }

    /// Length of the observation vector each creature senses.
    #[getter]
    fn observation_size(&self) -> usize {
        self.observation_size
    }

    /// Length of the action vector each creature's controller outputs.
    #[getter]
    fn action_size(&self) -> usize {
        self.action_size
    }

    /// Evaluate a whole generation. `genomes` is a list of flat weight vectors,
    /// each of length `genome_size`. Returns their fitnesses in the same order.
    ///
    /// Blocks until all rollouts finish, but releases the GIL while waiting so
    /// other Python threads can run. A genome of the wrong length yields a
    /// fitness of negative infinity rather than raising.
    fn evaluate(&self, py: Python<'_>, genomes: Vec<Vec<f32>>) -> Vec<f32> {
        let sender = self.job_sender.clone();
        let receiver = self.result_receiver.clone();
        py.detach(move || {
            let count = genomes.len();
            let mut fitness = vec![f32::NEG_INFINITY; count];

            for (index, genome) in genomes.into_iter().enumerate() {
                if sender.send(SimJob { index, genome }).is_err() {
                    return fitness;
                }
            }
            for _ in 0..count {
                match receiver.recv() {
                    Ok(result) if result.index < count => fitness[result.index] = result.fitness,
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            fitness
        })
    }

    /// Open a window showing one of the worker simulations. The chosen worker is
    /// paced to real time; the others keep running flat out. Whatever genomes
    /// that worker happens to pick up during `evaluate` calls are what you see.
    #[pyo3(signature = (index=0))]
    fn render(&mut self, index: usize) {
        self.party.write().unwrap().make_renderable(index);
        let party = Arc::clone(&self.party);
        if !self.window_active {
            thread::spawn(move || {
                run_render_app(party);
            });
            self.window_active = true;
        }
    }
}

fn run_render_app(party: Arc<RwLock<CreatureParty>>) {
    let _ = env_logger::try_init();

    let event_loop = EventLoopBuilder::default()
        .with_any_thread(true)
        .build()
        .expect("failed to build event loop");

    let mut app = App::new(party);
    event_loop.run_app(&mut app).expect("event loop error");
}

/// The Python module. Exposes [`SimulationSet`].
#[pymodule]
fn creature_simulator(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SimulationSet>()?;
    Ok(())
}

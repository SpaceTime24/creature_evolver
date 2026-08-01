use core::time;
use std::{
    matches, println,
    sync::{Arc, OnceLock},
    thread::{self, JoinHandle},
    time::Duration,
    todo,
};

use crossbeam::{atomic::AtomicCell, queue::ArrayQueue};
use rapier3d::pipeline::PhysicsWorld;
use wgpu::naga::compact::KeepUnused::No;

use crate::{
    creature_environment::{
        creature::{Creature, CreatureGenerator},
        creature_world::CreatureWorld,
    },
    graphical_app::{mesh::InstanceRaw, mesh::MeshId},
};

const THREAD_PUBLISH_AHEAD: u8 = 2;

#[derive(Clone, Copy)]
enum ThreadCommand {
    NoCommand,
    Received,
    Pause,
    Start,
    AddCreature,
    Terminate,
}

#[derive(Clone, Copy)]
enum ThreadState {
    Idle,
    Simulating,
    Paused,
    SimulationComplete,
}

pub struct SimulationThreadHandle {
    join_handle: Option<JoinHandle<()>>,
    // This is initialized only for renderable simulations.  The queue itself is
    // shared directly, so pushing and popping remain lock-free.
    models_to_draw: Arc<OnceLock<ArrayQueue<Vec<(MeshId, Vec<InstanceRaw>)>>>>,
    last_drawn_models: Option<Vec<(MeshId, Vec<InstanceRaw>)>>,
    command: Arc<AtomicCell<ThreadCommand>>,
    state: Arc<AtomicCell<ThreadState>>,
    creature_generator_passthrough: Arc<AtomicCell<Option<fn(&mut PhysicsWorld) -> Creature>>>,
}

impl SimulationThreadHandle {
    pub fn new() -> SimulationThreadHandle {
        SimulationThreadHandle {
            join_handle: None,
            models_to_draw: Arc::new(OnceLock::new()),
            command: Arc::from(AtomicCell::from(ThreadCommand::NoCommand)),
            state: Arc::from(AtomicCell::from(ThreadState::Idle)),
            last_drawn_models: None,
            creature_generator_passthrough: Arc::from(AtomicCell::from(None)),
        }
    }

    pub fn get_new_instances(&mut self) -> Option<&Vec<(MeshId, Vec<InstanceRaw>)>> {
        if let Some(queue) = self.models_to_draw.get() {
            if let Some(new_instances) = queue.pop() {
                self.last_drawn_models.replace(new_instances);
            }
            self.last_drawn_models.as_ref()
        } else {
            None
        }
    }

    fn send_command(&self, command: ThreadCommand) {
        while !matches!(self.command.load(), ThreadCommand::Received) {
            thread::sleep(Duration::from_millis(1));
        }
        self.command.store(command);
    }

    pub fn add_creature(&mut self, generator: fn(&mut PhysicsWorld) -> Creature) {
        self.creature_generator_passthrough.store(Some(generator));
        self.send_command(ThreadCommand::AddCreature);
    }

    pub fn make_renderable(&mut self) {
        self.models_to_draw
            .get_or_init(|| ArrayQueue::new(THREAD_PUBLISH_AHEAD as usize));
        println!("I'm Renderable now!");
    }

    pub fn activate(&mut self, creature_world: CreatureWorld) {
        let thread = SimulationThread {
            creature_world,
            models_to_draw: Arc::clone(&self.models_to_draw),
            command: Arc::clone(&self.command),
            state: Arc::clone(&self.state),
            creature_generator_passthrough: Arc::clone(&self.creature_generator_passthrough),
        };
        self.join_handle = Some(thread::spawn(|| {
            thread.run();
        }));
    }

    pub fn start_simulation(&self) {
        self.send_command(ThreadCommand::Start);
    }

    pub fn deactivate(&mut self) {
        if let Some(join_handle) = self.join_handle.take() {
            self.command.store(ThreadCommand::Terminate);
            join_handle.join().unwrap();
            println!("Terminated ")
        }
    }
}

impl Drop for SimulationThreadHandle {
    fn drop(&mut self) {
        self.deactivate();
    }
}

struct SimulationThread {
    creature_world: CreatureWorld,
    models_to_draw: Arc<OnceLock<ArrayQueue<Vec<(MeshId, Vec<InstanceRaw>)>>>>,
    command: Arc<AtomicCell<ThreadCommand>>,
    state: Arc<AtomicCell<ThreadState>>,
    creature_generator_passthrough: Arc<AtomicCell<Option<fn(&mut PhysicsWorld) -> Creature>>>,
}

impl SimulationThread {
    fn run(mut self) {
        loop {
            let command = self.command.swap(ThreadCommand::Received);
            match command {
                ThreadCommand::NoCommand | ThreadCommand::Received => {}
                ThreadCommand::Terminate => break,
                ThreadCommand::Pause => match self.state.load() {
                    ThreadState::Simulating => self.state.store(ThreadState::Paused),
                    ThreadState::Idle | ThreadState::SimulationComplete | ThreadState::Paused => {}
                },
                ThreadCommand::Start => match self.state.load() {
                    ThreadState::Idle => self.state.store(ThreadState::Simulating), //Maybe reset?
                    ThreadState::Simulating => {}
                    ThreadState::Paused => self.state.store(ThreadState::Simulating),
                    ThreadState::SimulationComplete => {} //Maybe reset?
                },
                ThreadCommand::AddCreature => {
                    let generator = self
                        .creature_generator_passthrough
                        .load()
                        .expect("No generator found when commanded to add crature");
                    println!("Adding creature into my little thread world");
                    self.creature_world.add_creature(generator);
                }
            }
            match self.state.load() {
                ThreadState::Idle => {}
                ThreadState::Simulating => {
                    self.update_creature_world();
                }
                ThreadState::Paused => {}
                ThreadState::SimulationComplete => {}
            }
            thread::sleep(time::Duration::from_millis(100));
        }
        println!("Simulation thread end");
    }

    fn update_creature_world(&mut self) {
        self.creature_world.physics.step();
        if let Some(model_queue) = self.models_to_draw.get() {
            if !model_queue.is_full() {
                if let Some(creature) = self.creature_world.creature.as_ref() {
                    let _ = model_queue.push(creature.get_raw_instances());
                }
            }
        }
    }
}

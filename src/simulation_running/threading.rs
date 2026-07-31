use core::time;
use std::{
    println,
    sync::{Arc, OnceLock},
    thread::{self, JoinHandle},
};

use crossbeam::{atomic::AtomicCell, queue::ArrayQueue};

use crate::{
    creature_environment::creature_world::CreatureWorld,
    graphical_app::{mesh::MeshId, scene::InstanceRaw},
};

const THREAD_PUBLISH_AHEAD: u8 = 4;

enum ThreadCommand {
    NoCommand,
    Received,
    Terminate,
}

pub struct SimulationThreadHandle {
    join_handle: Option<JoinHandle<()>>,
    // This is initialized only for renderable simulations.  The queue itself is
    // shared directly, so pushing and popping remain lock-free.
    models_to_draw: Arc<OnceLock<ArrayQueue<Vec<(MeshId, Vec<InstanceRaw>)>>>>,
    command: Arc<AtomicCell<ThreadCommand>>,
}

impl SimulationThreadHandle {
    pub fn new() -> SimulationThreadHandle {
        SimulationThreadHandle {
            join_handle: None,
            models_to_draw: Arc::new(OnceLock::new()),
            command: Arc::from(AtomicCell::from(ThreadCommand::NoCommand)),
        }
    }

    pub fn make_renderable(&mut self) {
        self.models_to_draw
            .get_or_init(|| ArrayQueue::new(THREAD_PUBLISH_AHEAD as usize));
    }

    pub fn activate(&mut self, creature_world: CreatureWorld) {
        let thread = SimulationThread {
            creature_world,
            models_to_draw: Arc::clone(&self.models_to_draw),
            command: Arc::clone(&self.command),
        };
        self.join_handle = Some(thread::spawn(|| {
            thread.run();
        }));
    }

    pub fn deactivate(&mut self) {
        if let Some(join_handle) = self.join_handle.take() {
            self.command.store(ThreadCommand::Terminate);
            join_handle.join().unwrap();
            println!("Terminated ")
        }
    }
}

struct SimulationThread {
    creature_world: CreatureWorld,
    models_to_draw: Arc<OnceLock<ArrayQueue<Vec<(MeshId, Vec<InstanceRaw>)>>>>,
    command: Arc<AtomicCell<ThreadCommand>>,
}

impl SimulationThread {
    fn run(mut self) {
        loop {
            let command = self.command.swap(ThreadCommand::Received);
            match command {
                ThreadCommand::NoCommand | ThreadCommand::Received => {
                    self.update_creature_world();
                }
                ThreadCommand::Terminate => break,
            }
            thread::sleep(time::Duration::from_millis(10));
        }
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

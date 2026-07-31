pub mod creature_environment;
pub mod graphical_app;
pub mod simulation_running;

use std::println;

use glam::Vec3;
use rapier3d::pipeline::PhysicsWorld;
use winit::event_loop::EventLoop;

use crate::creature_environment::creature::Creature;
use crate::creature_environment::creature_world::CreatureWorld;
use crate::graphical_app::app_handler::App;
use crate::simulation_running::simulator::SimulationRunner;
use crate::simulation_running::threading::SimulationThreadHandle;

fn make_event_loop_and_run(thread_handles: &Vec<SimulationThreadHandle>) {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(thread_handles);

    event_loop.run_app(&mut app).expect("TODO: panic message");
}

fn creature_generator<'a>(physics_world: &'a mut PhysicsWorld) -> Creature<'a> {
    Creature::sample_creature(physics_world, Vec3::ZERO)
}

fn main() {
    // let creature_generator = |physics_world: &mut PhysicsWorld| {
    //     Creature::sample_creature(physics_world, Vec3::new(0.0, 0.0, 0.0))
    // };

    let environment_generator = |pworld: &mut PhysicsWorld| {};
    let mut creature_world = CreatureWorld::new_unpopulated(environment_generator);

    creature_world.add_creature(creature_generator);

    let mut thread_handle = SimulationThreadHandle::new();
    thread_handle.activate(creature_world);
    thread_handle.make_renderable();

    let mut thread_handles = Vec::new();
    thread_handles.push(thread_handle);

    make_event_loop_and_run(&thread_handles);

    for mut thread in thread_handles {
        thread.deactivate();
    }
}

pub mod creature_environment;
pub mod graphical_app;
pub mod neural_net;
pub mod simulation_running;

use std::println;

use glam::{Vec3, vec3};
use rapier3d::geometry::{Collider, ColliderBuilder, ColliderSet};
use rapier3d::pipeline::PhysicsWorld;
use winit::event_loop::EventLoop;

use crate::creature_environment::creature::Creature;
use crate::creature_environment::creature_world::CreatureWorld;
use crate::creature_environment::static_environment::{self, CreatureParty, StaticEnvironment};
use crate::graphical_app::app_handler::App;
use crate::simulation_running::simulator::SimulationRunner;
use crate::simulation_running::threading::SimulationThreadHandle;

fn make_event_loop_and_run(thread_handles: &mut Vec<SimulationThreadHandle>) {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new(thread_handles);

    event_loop.run_app(&mut app).expect("TODO: panic message");
}

fn creature_generator<'a>(physics_world: &'a mut PhysicsWorld) -> Creature<'a> {
    Creature::sample_creature(physics_world, Vec3::ZERO)
}

fn simple_world() -> Vec<(Collider, Vec3)> {
    let mut colliders = Vec::new();
    let mut collider_set = ColliderSet::new();

    let ground = ColliderBuilder::cuboid(100.0, 1.0, 100.0)
        .translation(vec3(0.0, -20.0, 0.0))
        .build();
    colliders.push((ground, Vec3::new(0.2, 0.2, 0.7)));

    return colliders;
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
    thread_handle.start_simulation();

    let mut thread_handles = Vec::new();
    thread_handles.push(thread_handle);

    let static_environment = StaticEnvironment::new(simple_world());
    let party = CreatureParty::new(static_environment);

    make_event_loop_and_run(&mut thread_handles);

    for mut thread in thread_handles {
        thread.deactivate();
    }
}

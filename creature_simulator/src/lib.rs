pub mod creature_environment;
pub mod graphical_app;
pub mod neural_net;
pub mod simulation_running;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod creature_simulator {

    use pyo3::prelude::*;

    use crate::creature_environment::static_environment::{CreatureParty, StaticEnvironment};

    use glam::{Vec3, vec3};

    use rapier3d::{
        geometry::{Collider, ColliderBuilder, ColliderSet},
        pipeline::PhysicsWorld,
    };
    use std::{
        sync::{Arc, RwLock},
        thread,
        time::Duration,
    };
    use winit::event_loop::EventLoopBuilder;
    use winit::platform::wayland::EventLoopBuilderExtWayland;

    use crate::{
        creature_environment::creature::Creature, graphical_app::app_handler::App,
        neural_net::neural_placeholder::NeuralPlaceholder,
    };

    fn make_event_loop_and_run(party: Arc<RwLock<CreatureParty>>) {
        env_logger::init();

        // let event_loop = EventLoop::new().unwrap();

        let event_loop = EventLoopBuilder::default()
            .with_any_thread(true)
            .with_wayland()
            .build()
            .unwrap();

        let mut app = App::new(party);

        event_loop.run_app(&mut app).expect("TODO: panic message");
    }

    fn creature_generator<'a>(physics_world: &'a mut PhysicsWorld) -> Creature<'a> {
        Creature::single_obj_creature(physics_world, Vec3::ZERO)
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

    fn do_render_app(locked_party: Arc<RwLock<CreatureParty>>) {
        make_event_loop_and_run(locked_party);
    }

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn run_thing() {
        // let creature_generator = |physics_world: &mut PhysicsWorld| {
        //     Creature::sample_creature(physics_world, Vec3::new(0.0, 0.0, 0.0))
        // };

        let static_environment = StaticEnvironment::new(simple_world());
        let mut party = CreatureParty::new(static_environment);

        let locked_party: Arc<RwLock<CreatureParty>> = Arc::new(RwLock::new(party));

        let party_clone = locked_party.clone();

        let nets = vec![NeuralPlaceholder::default(), NeuralPlaceholder::default()];

        locked_party
            .write()
            .unwrap()
            .run_generation_of_creatures(creature_generator, nets);

        thread::spawn(|| {
            do_render_app(party_clone);
        });

        locked_party
            .write()
            .unwrap()
            .thread_handles()
            .get_mut(0)
            .unwrap()
            .make_renderable();

        thread::sleep(Duration::from_secs(5));

        //do_render_app(party_clone);
    }
}

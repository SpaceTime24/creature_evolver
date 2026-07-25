pub mod graphical_app;

use std::println;

use winit::event_loop::EventLoop;

use crate::graphical_app::app_handler::App;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();

    let mut app = App::new();

    event_loop.run_app(&mut app).expect("TODO: panic message");
}

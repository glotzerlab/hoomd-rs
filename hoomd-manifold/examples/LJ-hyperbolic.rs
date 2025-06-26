#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_mc::{Sweep, Trial, Zero};
use std::array;
use libm::{cosh, sinh, acosh};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, property::Point, boundary::Open};
use hoomd_manifold::{Minkowski, HyperbolicTranslate, EightEight, Hyperboloid};

use ratatui::{
    crossterm::event::{self, Event, poll},
    layout::{Flex, Layout},
    style::Color,
    symbols::Marker,
    widgets::{
        Block,
        canvas::{Canvas, Circle},
    },
    {DefaultTerminal, Frame},
};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let terminal = ratatui::init();
    let result = run(terminal);
    ratatui::restore();
    result
}

/// Run the simulation
fn run(mut terminal: DefaultTerminal) -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = MicrostateBuilder::with_boundary(Open)
    .bodies([Body::point(Minkowski::from([1.0, 1.0, 3.0_f64.sqrt()]))])
    .try_build()?;

    let kt = 1.0;
    let hamiltonian = Zero;
    let d = 0.1;

    let translate = HyperbolicTranslate {
        maximum_distance: d.try_into()?,
        skirt: 1.0
    };
    let translate_sweep = Sweep { local: translate };

    loop {
        terminal.draw(|frame| render(frame, &microstate))?;

        if poll(Duration::from_millis(0))? && matches!(event::read()?, Event::Key(_)) {
            break Ok(());
        }

        translate_sweep.apply(&mut microstate, &hamiltonian, &kt);
        microstate.increment_step();
    }
}

/// Render the system state.
fn render(
    frame: &mut Frame,
    microstate: &Microstate<Point<Minkowski<3>>, Point<Minkowski<3>>, Open>,
) {
    let properties = &microstate.bodies()[0].item.properties;
    let proj = properties.position.to_poincare(1.0);
    let rad_sq = 0.25*0.25;
    let v = acosh((rad_sq + 1.0)/(1.0-rad_sq));
    let eta = properties.position[2].acosh();
    let edge_proj = (sinh(eta+v))/(1.0 + cosh(eta+v));
    let rad_proj = (sinh(eta))/(1.0 + cosh(eta)) - edge_proj;

    let canvas = Canvas::default()
        .block(Block::bordered().title("Random walk in Hyperbolic Space"))
        .marker(Marker::Braille)
        .paint(|ctx| {
            ctx.draw(&Circle {
                x: proj[0],
                y: proj[1],
                radius: rad_proj,
                color: Color::Yellow,
            });
            ctx.draw(&Circle {
                x: 0.0,
                y: 0.0,
                radius: 1.0,
                color: Color::Blue,
            })
        })
        .x_bounds([-1.0, 1.0])
        .y_bounds([-1.0, 1.0]);

    let horizontal = Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
    let [area] = horizontal.areas(frame.area());

    frame.render_widget(canvas, area);
}
#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_mc::{Sweep, Trial, Zero};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, property::Point};
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
    let mut microstate = MicrostateBuilder::with_boundary(EightEight {
        skirt: 1.0.try_into()?,
    })
    .bodies([Body::point(Minkowski::from([0.0, 0.0, 1.0]))])
    .try_build()?;

    let kt = 1.0;
    let hamiltonian = Zero;
    let d = 0.05;

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
    microstate: &Microstate<Point<Minkowski<3>>, Point<Minkowski<3>>, EightEight>,
) {
    let properties = &microstate.bodies()[0].item.properties;

    let canvas = Canvas::default()
        .block(Block::bordered().title("Bounded random walk in Hyperbolic Space"))
        .marker(Marker::Braille)
        .paint(|ctx| {
            ctx.draw(&Circle {
                x: properties.position.to_poincare(1.0)[0],
                y: properties.position.to_poincare(1.0)[1],
                radius: 0.1,
                color: Color::Yellow,
            });
        })
        .x_bounds([-2.5, 2.5])
        .y_bounds([-2.5, 2.5]);

    let horizontal = Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
    let [area] = horizontal.areas(frame.area());

    frame.render_widget(canvas, area);
}
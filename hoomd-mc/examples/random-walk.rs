#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_mc::{Sweep, Translate, Trial, Zero};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, boundary::Square, property::Point};
use hoomd_vector::Cartesian;

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
    let mut microstate = MicrostateBuilder::with_boundary(Square {
        l: 10.0.try_into()?,
    })
    .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
    .try_build()?;

    let kt = 1.0;
    let hamiltonian = Zero;
    let d = 0.1;

    let translate = Translate { maximum_distance: d.try_into()? };
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
    microstate: &Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Square>,
) {
    let properties = &microstate.bodies()[0].item.properties;

    let l = microstate.boundary().l.get();

    let canvas = Canvas::default()
        .block(Block::bordered().title("Bounded random walk"))
        .marker(Marker::Braille)
        .paint(|ctx| {
            ctx.draw(&Circle {
                x: properties.position[0],
                y: properties.position[1],
                radius: 0.5,
                color: Color::Yellow,
            });
        })
        .x_bounds([-l / 2.0, l / 2.0])
        .y_bounds([-l / 2.0, l / 2.0]);

    let horizontal = Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
    let [area] = horizontal.areas(frame.area());

    frame.render_widget(canvas, area);
}

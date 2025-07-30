#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_mc::{Sweep, Trial, Zero};
use libm::{cos, sin, acos, sqrt};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, property::Point, boundary::Open};
use hoomd_manifold::{Sphere, SphericalTranslate};
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

const RADIUS : f64 = 1.0;

/// Run the simulation
fn run(mut terminal: DefaultTerminal) -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = MicrostateBuilder::with_boundary(Open)
    .bodies([Body::point(Cartesian::from([0.01,0.01,-sqrt(RADIUS.powi(2)-2.0*(0.01_f64).powi(2))]))])
    .try_build()?;

    let kt = 1.0;
    let hamiltonian = Zero;
    let d = 0.1 * RADIUS;

    let translate = SphericalTranslate {
        maximum_distance: d.try_into()?,
        radius: RADIUS,
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

const RAD_SQ : f64 = 0.1;

/// stereographic projection
fn stereographic(point: &Cartesian<3>, radius: f64) -> [f64;3] {
    let pt = Sphere::from(point);
    let proj = pt.stereographic_projection();
    let theta = acos(point.coordinates[2]/radius);
    let v = acos((radius.powi(2) - RAD_SQ)/(radius.powi(2)+RAD_SQ));
    let edge_proj = (RADIUS * sin(theta+v))/(1.0 - cos(theta+v));
    let rad_proj =  edge_proj - (RADIUS * sin(theta))/(1.0 - cos(theta));
    [proj[0], proj[1], rad_proj]
}
/// Render the system state.
fn render(
    frame: &mut Frame,
    microstate: &Microstate<Point<Cartesian<3>>, Point<Cartesian<3>>, Open>,
) {
    let properties = &microstate.bodies()[0].item.properties;
    let canvas = Canvas::default()
        .block(Block::bordered().title("Random walk on a Sphere"))
        .marker(Marker::Braille)
        .paint(|ctx| {
            let coords = stereographic(&properties.position, RADIUS);
            ctx.draw(&Circle {
                x: coords[0],
                y: coords[1],
                radius: coords[2],
                color: Color::Yellow,
            });
        })
        .x_bounds([-RADIUS*10.0, RADIUS*10.0])
        .y_bounds([-RADIUS*10.0, RADIUS*10.0]);

    let horizontal = Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
    let [area] = horizontal.areas(frame.area());

    frame.render_widget(canvas, area);
}
#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_manifold::{EightEight, FundamentalDomain, HyperbolicTranslate, Hyperboloid, Minkowski};
use hoomd_mc::{Sweep, Trial, Zero};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, property::Point};
use libm::{acosh, cosh, sinh, sqrt};
use std::array;

use ratatui::{
    crossterm::event::{self, Event, poll},
    layout::{Flex, Layout},
    style::Color,
    symbols::Marker,
    widgets::{
        Block,
        canvas::{Canvas, Circle, Points},
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

const RHO: f64 = 0.6;

/// Run the simulation
fn run(mut terminal: DefaultTerminal) -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = MicrostateBuilder::with_boundary(EightEight { skirt: RHO })
        .bodies([Body::point(Minkowski::from([
            0.00001,
            0.00001,
            sqrt(2.0 * (0.00001_f64).powi(2) + RHO.powi(2)),
        ]))])
        .try_build()?;

    let kt = 1.0;
    let hamiltonian = Zero;
    let d = 0.05 * RHO;

    let translate = HyperbolicTranslate {
        maximum_distance: d.try_into()?,
        skirt: RHO,
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

const RAD_SQ: f64 = 0.01;
const BOUNDARY_NUMBER: usize = 1000;

/// Project coordinates to Poincare disk
fn poincare(point: &Minkowski<3>, skirt: f64) -> [f64; 3] {
    let pt = Hyperboloid::from(point);
    let proj = pt.to_poincare();
    let v = acosh((RAD_SQ + RHO.powi(2)) / (RHO.powi(2) - RAD_SQ));
    let eta = acosh(point.coordinates[2] / RHO);
    let edge_proj = (RHO * sinh(eta + v)) / (1.0 + cosh(eta + v));
    let rad_proj = (RHO * sinh(eta)) / (1.0 + cosh(eta)) - edge_proj;
    [proj[0], proj[1], rad_proj]
}
/// Render the system state.
fn render(
    frame: &mut Frame,
    microstate: &Microstate<Point<Minkowski<3>>, Point<Minkowski<3>>, EightEight>,
) {
    let v = Hyperboloid::<3>::boundary_points(BOUNDARY_NUMBER, RHO);
    let (a, b): (Vec<_>, Vec<_>) = v.into_iter().unzip();
    let boundary_particles: [(f64, f64); BOUNDARY_NUMBER] = array::from_fn(|i| (a[i], b[i]));
    let properties = &microstate.bodies()[0].item.properties;
    let canvas = Canvas::default()
        .block(Block::bordered().title("Random walk in Hyperbolic Space"))
        .marker(Marker::Braille)
        .paint(|ctx| {
            let coords = poincare(&properties.position, RHO);
            ctx.draw(&Circle {
                x: coords[0],
                y: coords[1],
                radius: coords[2],
                color: Color::Yellow,
            });
            ctx.draw(&Circle {
                x: 0.0,
                y: 0.0,
                radius: RHO,
                color: Color::Blue,
            });
            ctx.draw(&Points {
                coords: &boundary_particles,
                color: Color::Blue,
            })
        })
        .x_bounds([-RHO, RHO])
        .y_bounds([-RHO, RHO]);

    let horizontal = Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
    let [area] = horizontal.areas(frame.area());

    frame.render_widget(canvas, area);
}

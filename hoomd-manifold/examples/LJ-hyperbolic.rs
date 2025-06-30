#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_interaction::{
    CutoffPair, Single,
    pairwise::{LennardJones, Isotropic, IsotropicEnergy},
};
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
    .bodies([Body::point(Minkowski::from([1.0, 1.0, 3.0_f64.sqrt()])),
            Body::point(Minkowski::from([-1.0, 1.0, 3.0_f64.sqrt()])),
            Body::point(Minkowski::from([1.0, -1.0, 3.0_f64.sqrt()])),
            Body::point(Minkowski::from([-1.0, -1.0, 3.0_f64.sqrt()]))])
    .try_build()?;

    let lj : LennardJones = LennardJones {
        epsilon: 10.0,
        sigma: 0.5,
    };

//// TODO: Isotropic looks at the distance of the metric space, which defaults to Minkowski.
/// reconfigure to pass hyperbolic distance instead 

    let evaluator = Isotropic(lj);
    let cutoff_pair = CutoffPair {
        r_cut: 1.0,
        evaluator,
    };

    let kt = 1.0;
    let hamiltonian = cutoff_pair;
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

const RAD_SQ : f64 = 0.03;

/// Project coordinates to Poincare disk 
fn poincare(point: &Minkowski<3>, skirt: f64) -> [f64;3] {
    let proj = point.to_poincare(skirt);
    let v = acosh((RAD_SQ + 1.0)/(1.0-RAD_SQ));
    let eta = point.coordinates[2].acosh();
    let edge_proj = (sinh(eta+v))/(1.0 + cosh(eta+v));
    let rad_proj = (sinh(eta))/(1.0 + cosh(eta)) - edge_proj;
    [proj[0], proj[1], rad_proj]
}

/// Render the system state.
fn render(
    frame: &mut Frame,
    microstate: &Microstate<Point<Minkowski<3>>, Point<Minkowski<3>>, Open>,
) {
    let canvas = Canvas::default()
        .block(Block::bordered().title("Lennard Jones Gas in Hyperbolic Space"))
        .marker(Marker::Braille)
        .paint(|ctx| {
            for site in microstate.sites() {
                let mut coords = poincare(&site.properties.position, 1.0);
                ctx.draw(&Circle {
                x: coords[0],
                y: coords[1],
                radius: coords[2],
                color: Color::Yellow,
            });
            }
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
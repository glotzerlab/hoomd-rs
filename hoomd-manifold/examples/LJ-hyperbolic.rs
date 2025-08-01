#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_interaction::{CutoffPair, pairwise::LennardJones};
use hoomd_manifold::{
    CurvedIsotropic, HyperbolicDisk, HyperbolicTranslate, Hyperboloid, Minkowski,
};
use hoomd_mc::{Sweep, Trial};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, boundary::Open, property::Point};
use libm::{acosh, cosh, sinh, sqrt};
use rand::distr::Distribution;
use rand::{SeedableRng, rngs::StdRng};

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

/// number of particles
const PARTICLE_NUMBER: usize = 500;
/// skirt width of hyperboloid
const RHO: f64 = 10.0;

/// Run the simulation
fn run(mut terminal: DefaultTerminal) -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = MicrostateBuilder::with_boundary(Open)
        //.bodies([Body::point(Minkowski::from([1.0, -2.0, sqrt(5.0)])),
        //    Body::point(Minkowski::from([1.0, -1.0, sqrt(3.0)])),
        //    Body::point(Minkowski::from([-1.0, -2.0, sqrt(5.0)])),
        //    Body::point(Minkowski::from([-1.0, -1.0, sqrt(3.0)]))])
        .try_build()?;

    let initial_spacing = 1.0;
    let mut rng = StdRng::seed_from_u64(23);
    let sample_disk = HyperbolicDisk {
        r: initial_spacing.try_into()?,
        point: Minkowski::from([
            0.00001,
            0.00001,
            sqrt(2.0 * (0.00001_f64).powi(2) + RHO.powi(2)),
        ]),
        skirt: RHO,
    };
    for _n in 0..PARTICLE_NUMBER {
        let new_point: Minkowski<3> = sample_disk.sample(&mut rng).point;
        microstate.add_body(Body::point(new_point))?;
    }

    let lj: LennardJones = LennardJones {
        epsilon: 10.0,
        sigma: 0.065_507,
    };

    let evaluator = CurvedIsotropic {
        isotropic: lj,
        manifold: Hyperboloid::from(&Minkowski::from([0.0, 0.0, RHO])),
    };
    let cutoff_pair = CutoffPair {
        r_cut: 10.0,
        evaluator,
    };

    let kt = 1.0;
    let hamiltonian = cutoff_pair;
    let d = 0.001;

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

/// squared radius of disk in render
const RAD_SQ: f64 = 0.0001;

/// Project coordinates to Poincare disk
fn poincare(point: &Minkowski<3>) -> [f64; 3] {
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
    microstate: &Microstate<Point<Minkowski<3>>, Point<Minkowski<3>>, Open>,
) {
    let canvas = Canvas::default()
        .block(Block::bordered().title("Lennard Jones Gas in Hyperbolic Space"))
        .marker(Marker::Braille)
        .paint(|ctx| {
            for site in microstate.sites() {
                let coords = poincare(&site.properties.position);
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
                radius: RHO,
                color: Color::Blue,
            });
        })
        .x_bounds([-0.7, 0.7]) //([-RHO, RHO])
        .y_bounds([-0.7, 0.7]); //([-RHO, RHO]);

    let horizontal = Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
    let [area] = horizontal.areas(frame.area());

    frame.render_widget(canvas, area);
}

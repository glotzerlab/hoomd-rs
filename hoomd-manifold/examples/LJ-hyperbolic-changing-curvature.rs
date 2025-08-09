#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_interaction::{
    CutoffPair,
    pairwise::{Isotropic, LennardJones},
};
use hoomd_manifold::{HyperbolicDisk, HyperbolicTranslate, Hyperboloid, Minkowski};
use hoomd_mc::{Sweep, Trial};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, boundary::Open, property::Point};
use libm::{acosh, cosh, exp, sinh, sqrt};
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
const PARTICLE_NUMBER: usize = 100;

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
        point: Minkowski::from([0.00001, 0.00001, sqrt(2.0 * (0.00001_f64).powi(2) + 1.0)]),
        skirt: 1.0,
    };
    for _n in 0..PARTICLE_NUMBER {
        let new_point: Minkowski<3> = sample_disk.sample(&mut rng).point;
        let hyp_point = Hyperboloid::from(&new_point);
        microstate.add_body(Body::point(hyp_point))?;
    }

    let lj: LennardJones = LennardJones {
        epsilon: 10.0,
        sigma: 0.5,
    };

    loop {
        let time = microstate.step();
        terminal.draw(|frame| render(frame, &microstate, time))?;

        if poll(Duration::from_millis(0))? && matches!(event::read()?, Event::Key(_)) {
            break Ok(());
        }

        let evaluator = Isotropic(lj);

        let cutoff_pair = CutoffPair {
            r_cut: 10.0,
            evaluator,
        };
        let kt = 1.0;
        let d = 0.05;

        let hamiltonian = cutoff_pair;

        let hyp_translate = HyperbolicTranslate {
            maximum_distance: d.try_into()?,
            skirt: skirt_size(time),
        };
        let translate_sweep = Sweep(hyp_translate);

        translate_sweep.apply(&mut microstate, &hamiltonian, &kt);
        microstate.increment_step();
    }
}

/// squared radius of disk in render
const RAD_SQ: f64 = 0.01;

/// Project coordinates to Poincare disk
fn poincare(point: &Minkowski<3>, skirt: f64) -> [f64; 3] {
    let pt = Hyperboloid::from(point);
    let proj = pt.to_poincare();
    let v = acosh((RAD_SQ + skirt.powi(2)) / (skirt.powi(2) - RAD_SQ));
    let eta = acosh(point.coordinates[2] / skirt);
    let edge_proj = (skirt * sinh(eta + v)) / (1.0 + cosh(eta + v));
    let rad_proj = (skirt * sinh(eta)) / (1.0 + cosh(eta)) - edge_proj;
    [proj[0], proj[1], rad_proj]
}
/// time before starting to change curvature
const WAIT_TIME: u64 = 200;
/// speed with which curvature changes
const SLOPE: f64 = 0.000_005;

/// function to tune skirt size
fn skirt_size(time: u64) -> f64 {
    if time < WAIT_TIME {
        1.0
    } else {
        //SLOPE * ((time as f64) - (WAIT_TIME as f64)) + 1.0
        exp(SLOPE * ((WAIT_TIME as f64) - (time as f64)))
    }
}

/// Render the system state.
fn render(
    frame: &mut Frame,
    microstate: &Microstate<Point<Hyperboloid<3>>, Point<Hyperboloid<3>>, Open>,
    time: u64,
) {
    let canvas = Canvas::default()
        .block(Block::bordered().title("Lennard Jones Gas in Hyperbolic Space"))
        .marker(Marker::Braille)
        .paint(|ctx| {
            for site in microstate.sites() {
                let coords = poincare(&site.properties.position.point, skirt_size(time));
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
                radius: skirt_size(time),
                color: Color::Blue,
            });
        })
        .x_bounds([-skirt_size(time), skirt_size(time)])
        .y_bounds([-skirt_size(time), skirt_size(time)]);

    let horizontal = Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
    let [area] = horizontal.areas(frame.area());

    frame.render_widget(canvas, area);
}

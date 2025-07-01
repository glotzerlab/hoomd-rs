#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_interaction::{
    CutoffPair, pairwise::LennardJonesGauss};
use hoomd_mc::{Sweep, Trial, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand::distr::Distribution;
use libm::{cosh, sinh, acosh, sqrt};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, property::Point, boundary::Open};
use hoomd_manifold::{Minkowski, HyperbolicTranslate, EightEight, Hyperboloid, CurvedIsotropic, 
                    CurvedManifold, HyperbolicDisk};

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

const PARTICLE_NUMBER : usize = 100;
const RHO : f64 = 1.0;

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
    let sample_disk = HyperbolicDisk{
        r: initial_spacing.try_into()?, 
        point: Minkowski::from([0.00001,0.00001,sqrt(2.0*(0.00001_f64).powi(2) + RHO.powi(2))]),
        skirt: RHO,}; 
    for _n in 0..PARTICLE_NUMBER {
        let new_point: Minkowski<3> = sample_disk.sample(&mut rng);
        microstate.add_body(Body::point(new_point))?;
    }
    
    let ljg : LennardJonesGauss = LennardJonesGauss {
        epsilon: 1.1,
        sigma_squared: 0.02,
        r_0: 1.6,
    };

    let evaluator = CurvedIsotropic(ljg, RHO);
    let cutoff_pair = CutoffPair {
        r_cut: 10.0 * RHO,
        evaluator,
    };

    let kt = 1.0;
    let hamiltonian = cutoff_pair;
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

const RAD_SQ : f64 = 0.025;

/// Project coordinates to Poincare disk 
fn poincare(point: &Minkowski<3>, skirt: f64) -> [f64;3] {
    let proj = point.to_poincare(skirt);
    let v = acosh((RAD_SQ + RHO.powi(2))/(RHO.powi(2)-RAD_SQ));
    let eta = acosh(point.coordinates[2]/RHO);
    let edge_proj = (RHO * sinh(eta+v))/(1.0 + cosh(eta+v));
    let rad_proj = (RHO * sinh(eta))/(1.0 + cosh(eta)) - edge_proj;
    [proj[0], proj[1], rad_proj]
}

/// Render the system state.
fn render(
    frame: &mut Frame,
    microstate: &Microstate<Point<Minkowski<3>>, Point<Minkowski<3>>, Open>,
) {
    let canvas = Canvas::default()
        .block(Block::bordered().title("Lennard-Jones-Gauss Gas in Hyperbolic Space"))
        .marker(Marker::Braille)
        .paint(|ctx| {
            for site in microstate.sites() {
                let coords = poincare(&site.properties.position, RHO);
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
            })
        })
        .x_bounds([-RHO, RHO])
        .y_bounds([-RHO, RHO]);

    let horizontal = Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
    let [area] = horizontal.areas(frame.area());

    frame.render_widget(canvas, area);
}
#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_interaction::{CutoffPair, pairwise::LennardJones};
use hoomd_manifold::{CurvedIsotropic, Sphere, SphericalDisk, SphericalTranslate};
use hoomd_mc::{Sweep, Trial};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, boundary::Open, property::Point};
use hoomd_vector::Cartesian;
use libm::{acos, cos, sin, sqrt};
use rand::distr::Distribution;
use rand::{SeedableRng, rngs::StdRng};
use std::f64::consts::PI;

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

const PARTICLE_NUMBER: usize = 12;
const RADIUS: f64 = 0.5;

/// Run the simulation
fn run(mut terminal: DefaultTerminal) -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = MicrostateBuilder::with_boundary(Open).try_build()?;

    let initial_spacing = RADIUS * PI / 4.0;
    let mut rng = StdRng::seed_from_u64(23);
    let sample_disk = SphericalDisk {
        r: initial_spacing.try_into()?,
        point: Cartesian::from([0.01, 0.01, -sqrt(RADIUS.powi(2) - 2.0 * (0.01_f64).powi(2))]),
        radius: RADIUS,
    };
    for _n in 0..PARTICLE_NUMBER {
        let new_point: Cartesian<3> = sample_disk.sample(&mut rng).point;
        microstate.add_body(Body::point(new_point))?;
    }

    let lj: LennardJones = LennardJones {
        epsilon: 10.0,
        sigma: 0.5,
    };

    let evaluator = CurvedIsotropic {
        isotropic: lj,
        manifold: Sphere::from(&Cartesian::from([0.0, 0.0, RADIUS])),
    };
    let cutoff_pair = CutoffPair {
        r_cut: 10.0,
        evaluator,
    };

    let kt = 1.0;
    let hamiltonian = cutoff_pair;
    let d = 0.05;

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

const RAD_SQ: f64 = 0.01;

/// stereographic projection
fn stereographic(point: &Cartesian<3>, radius: f64) -> [f64; 3] {
    let pt = Sphere::from(point);
    let proj = pt.stereographic_projection();
    let theta = acos(point.coordinates[2] / radius);
    let v = acos((radius.powi(2) - RAD_SQ) / (radius.powi(2) + RAD_SQ));
    let edge_proj = (RADIUS * sin(theta + v)) / (1.0 - cos(theta + v));
    let rad_proj = edge_proj - (RADIUS * sin(theta)) / (1.0 - cos(theta));
    [proj[0], proj[1], rad_proj]
}

/// Render the system state.
fn render(
    frame: &mut Frame,
    microstate: &Microstate<Point<Cartesian<3>>, Point<Cartesian<3>>, Open>,
) {
    let canvas = Canvas::default()
        .block(Block::bordered().title("Lennard Jones Gas in Hyperbolic Space"))
        .marker(Marker::Braille)
        .paint(|ctx| {
            for site in microstate.sites() {
                let coords = stereographic(&site.properties.position, RADIUS);
                ctx.draw(&Circle {
                    x: coords[0],
                    y: coords[1],
                    radius: coords[2],
                    color: Color::Yellow,
                });
            }
        })
        .x_bounds([-RADIUS * 2.0, RADIUS * 2.0])
        .y_bounds([-RADIUS * 2.0, RADIUS * 2.0]);

    let horizontal = Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
    let [area] = horizontal.areas(frame.area());

    frame.render_widget(canvas, area);
}

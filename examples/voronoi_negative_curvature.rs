// testing ground

extern crate glam;
extern crate hoomd_order;
extern crate rand;

use glam::DVec3;
use hoomd_order::meshless_voronoi::{Voronoi, VoronoiCell, VoronoiFace};
use hoomd_manifold::{Minkowski, Hyperboloid, HyperbolicDisk};
use rand::{prelude::*, Rng};
use rand::prelude::Distribution;
use std::convert::TryInto;
use libm::{cosh, sinh, acosh, sqrt};
use std::env;
use std::time::Duration;
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

const RHO: f64 = 1.0; 
const PARTICLE_NUMBER : usize = 100;
const RAD_SQ : f64 = 0.0005;

fn initial_distribution() -> (Vec<Vec<f64>>,Vec<[f64;3]>) {
    let initial_spacing = 2.0;
    let mut rng = StdRng::seed_from_u64(23);
    let sample_disk = HyperbolicDisk{
        r: initial_spacing.try_into().expect("hard-coded value should be valid"), 
        point: Minkowski::from([0.00001,0.00001,sqrt(2.0*(0.00001_f64).powi(2) + RHO.powi(2))]),
        skirt: RHO,}; 
    let mut poincare_coordinates = vec![];
    let mut generators = vec![];
    for _n in 0..PARTICLE_NUMBER {
            let new_point: Minkowski<3> = sample_disk.sample(&mut rng);
            let new_point_poincare = poincare(&new_point, RHO);
            generators.push(vec![new_point_poincare[0],new_point_poincare[1],0.0]);
            poincare_coordinates.push(new_point_poincare);
    }
    (generators, poincare_coordinates)
}

fn poincare(point: &Minkowski<3>, skirt: f64) -> [f64;3] {
    let proj = point.to_poincare(skirt);
    let v = acosh((RAD_SQ + RHO.powi(2))/(RHO.powi(2)-RAD_SQ));
    let eta = acosh(point.coordinates[2]/RHO);
    let edge_proj = (RHO * sinh(eta+v))/(1.0 + cosh(eta+v));
    let rad_proj = (RHO * sinh(eta))/(1.0 + cosh(eta)) - edge_proj;
    [proj[0], proj[1], rad_proj]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let terminal = ratatui::init();
    let mut args = env::args().skip(1);
    let count = match args.next() {
        Some(n) => n.parse::<usize>().expect(
            "The first argument should be an integer denoting the grid size along one dimension!",
        ),
        None => 20,
    };
    let pert = match args.next() {
        Some(p) => p.parse::<f64>().expect(
            "The second argument should be a number between 0 and 1 denoting the size of the grid perturbations!"
        ),
        None => 0.8,
    };

    let anchor = DVec3::splat(-100.);
    let width = DVec3::splat(200.);
    let (generators, poincare_coords) = initial_distribution();
    let _voronoi = Voronoi::build_hyperbolic(&generators, RHO, anchor, width, 2.try_into().unwrap(), false);
    let special_guy: usize = rand::thread_rng().gen_range(0..PARTICLE_NUMBER);
    let nlist = _voronoi.cells()[special_guy].neighbour_ids(&_voronoi);
    let mut nlist_vec = Vec::new();
    for n in nlist {
        nlist_vec.push(n);
    }
    let result = draw(terminal, &poincare_coords, special_guy, &nlist_vec);
    ratatui::restore();
    result
}

fn render(
    frame: &mut Frame,
    poincare_coords: &Vec<[f64;3]>,
    guy: usize,
    neighbors: &Vec<usize>,
) {
    let canvas = Canvas::default()
        .block(Block::bordered().title("2D Voronoi"))
        .marker(Marker::Braille)
        .paint(|ctx| {
            for n in 0..PARTICLE_NUMBER {
                let poin = poincare_coords[n];
                ctx.draw(&Circle {
                    x: poin[0],
                    y: poin[1],
                    radius: poin[2],
                    //coords: &[(coords[0],coords[1])],
                    color: 
                        if n==guy {
                            Color::Red
                        } else if neighbors.contains(&n) {
                            Color::Yellow
                        } else {
                            Color::Blue
                        },
                });
            }
        })
        .x_bounds([-1.0, 1.0])
        .y_bounds([-1.0, 1.0]);

    let horizontal = Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
    let [area] = horizontal.areas(frame.area());

    frame.render_widget(canvas, area);
}

fn draw(
    mut terminal: DefaultTerminal,
    poincare_coords: &Vec<[f64;3]>, 
    guy: usize,
    neighbors: &Vec<usize>
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        terminal.draw(|frame| render(frame, poincare_coords, guy, &neighbors))?;

        if poll(Duration::from_millis(0))? && matches!(event::read()?, Event::Key(_)) {
            break Ok(());
        }
    }
}

// testing ground

#![allow(clippy::all)]
#![allow(clippy::pedantic)]

extern crate glam;
extern crate hoomd_order;
extern crate rand;

use hoomd_geometry::shape::EightEight;
use hoomd_manifold::{HyperbolicDisk, Hyperboloid, Minkowski};
use hoomd_meshless_voronoi::{GenerateNeighborList, NeighborList};
use hoomd_microstate::{Body, MicrostateBuilder, boundary::Periodic};
use rand::prelude::Distribution;
use rand::{Rng, prelude::*};
#[cfg(not(target_arch = "wasm32"))]
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
use std::convert::TryInto;
use std::time::Duration;

const RHO: f64 = 1.0;
const PARTICLE_NUMBER: usize = 100;
const RAD_SQ: f64 = 0.0005;

#[cfg(not(target_arch = "wasm32"))]
fn poincare(point: &Hyperboloid<3>) -> [f64; 3] {
    let proj = point.to_poincare();
    let v = ((RAD_SQ + RHO.powi(2)) / (RHO.powi(2) - RAD_SQ)).acosh();
    let eta = (point.point.coordinates[2] / RHO).acosh();
    let edge_proj = (RHO * (eta + v).sinh()) / (1.0 + (eta + v).cosh());
    let rad_proj = (RHO * (eta.sinh())) / (1.0 + eta.cosh()) - edge_proj;
    [proj[0], proj[1], rad_proj]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let terminal = ratatui::init();
        let boundary = Periodic::new(0.5, EightEight { skirt: 1.0_f64 })?;
        let mut microstate =
            MicrostateBuilder::with_boundary(boundary).try_build()?;

        let initial_spacing = 1.3;
        let mut rng = rand::rng();
        let special_guy = rng.random_range(0..PARTICLE_NUMBER);

        let mut rng_2 = StdRng::seed_from_u64(23);
        let sample_disk = HyperbolicDisk {
            r: initial_spacing.try_into()?,
            point: Minkowski::from([
                0.00001,
                0.00001,
                (2.0 * (0.00001_f64).powi(2) + RHO.powi(2)).sqrt(),
            ]),
            skirt: RHO,
        };
        let mut poincare_coords = vec![];
        for _n in 0..PARTICLE_NUMBER {
            let new_point: Minkowski<3> = sample_disk.sample(&mut rng_2).point;
            let hyp_point = Hyperboloid::from(&new_point);
            microstate.add_body(Body::point(hyp_point))?;
            poincare_coords.push(poincare(&hyp_point));
        }

        let nlist = NeighborList::from_microstate(&microstate);
        let mut nlist_vec = Vec::new();
        for (a, b) in nlist.neighbors {
            if a == special_guy {
                nlist_vec.push(b);
            } else if b == special_guy {
                nlist_vec.push(a);
            }
        }
        let result = draw(terminal, &poincare_coords, special_guy, &nlist_vec);
        ratatui::restore();
        result
    }
    #[cfg(target_arch = "wasm32")]
    {
        Result((()))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn render(
    frame: &mut Frame,
    poincare_coords: &Vec<[f64; 3]>,
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
                    color: if n == guy {
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

    let horizontal =
        Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
    let [area] = horizontal.areas(frame.area());

    frame.render_widget(canvas, area);
}

#[cfg(not(target_arch = "wasm32"))]
fn draw(
    mut terminal: DefaultTerminal,
    poincare_coords: &Vec<[f64; 3]>,
    guy: usize,
    neighbors: &Vec<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        terminal
            .draw(|frame| render(frame, poincare_coords, guy, &neighbors))?;

        if poll(Duration::from_millis(0))?
            && matches!(event::read()?, Event::Key(_))
        {
            break Ok(());
        }
    }
}

/*! Test code for voronoi from hoomd microstates
*/

extern crate glam;
extern crate hoomd_order;
extern crate rand;

use glam::DVec3;
use hoomd_order::meshless_voronoi::Voronoi;
use rand::{Rng, distr::StandardUniform};
use ratatui::{
    crossterm::event::{self, Event, poll},
    layout::{Flex, Layout},
    style::Color,
    symbols::Marker,
    widgets::{
        Block,
        canvas::{Canvas, Points},
    },
    {DefaultTerminal, Frame},
};
use std::convert::TryInto;
use std::env;
use std::time::Duration;

fn perturbed_grid(anchor: DVec3, width: DVec3, count: usize, pert: f64) -> Vec<DVec3> {
    let mut generators = vec![];
    for n in 0..count {
        for m in 0..count {
            let pos = DVec3 {
                x: n as f64
                    + 0.5
                    + pert * (rand::rng().sample::<f64, StandardUniform>(StandardUniform) as f64),
                y: m as f64
                    + 0.5
                    + pert * (rand::rng().sample::<f64, StandardUniform>(StandardUniform) as f64),
                z: 0.0,
            } * width
                / count as f64
                + anchor;
            generators.push(pos.clamp(anchor, anchor + width));
        }
    }

    generators
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

    let anchor = DVec3::splat(0.);
    let width = DVec3::splat(1.);
    let generators = perturbed_grid(anchor, width, count, pert);
    let _voronoi = Voronoi::build(&generators, anchor, width, 2.try_into().unwrap(), false);
    let special_guy: usize = rand::rng().random_range(0..count.pow(2));
    let nlist = _voronoi.cells()[special_guy].neighbour_ids(&_voronoi);
    let mut nlist_vec = Vec::new();
    for n in nlist {
        nlist_vec.push(n);
    }
    let result = draw(terminal, &_voronoi, special_guy, &nlist_vec);
    ratatui::restore();
    result
}

fn render(frame: &mut Frame, voro: &Voronoi, guy: usize, neighbors: &Vec<usize>) {
    let canvas = Canvas::default()
        .block(Block::bordered().title("2D Voronoi"))
        .marker(Marker::Braille)
        .paint(|ctx| {
            for n in 0..400 {
                let coords = voro.cells()[n].loc();
                ctx.draw(&Points {
                    coords: &[(coords[0], coords[1])],
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
        .x_bounds([0.0, 1.0])
        .y_bounds([0.0, 1.0]);

    let horizontal = Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
    let [area] = horizontal.areas(frame.area());

    frame.render_widget(canvas, area);
}

fn draw(
    mut terminal: DefaultTerminal,
    voro: &Voronoi,
    guy: usize,
    neighbors: &Vec<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        terminal.draw(|frame| render(frame, &voro, guy, &neighbors))?;

        if poll(Duration::from_millis(0))? && matches!(event::read()?, Event::Key(_)) {
            break Ok(());
        }
    }
}

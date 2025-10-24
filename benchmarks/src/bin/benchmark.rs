// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::collections::HashMap;

use log::info;
use clap::Parser;
use clap_verbosity_flag::{Verbosity, InfoLevel};
use clap_verbosity_flag::log::LevelFilter;
use serde::Serialize;
use serde_json::Result;

use hoomd_microstate::property::{OrientedPoint, Point};
use hoomd_vector::{Angle, Cartesian, Versor};

use benchmarks::{Benchmark, mc};
use wildmatch::WildMatch;

#[derive(Serialize)]
struct Performance {
    units: String,
    n: Vec<usize>,
    performance: Vec<f64>,
}

impl Performance {
    fn with_units(units: String) -> Self {
        Self {
            units,
            n: Vec::new(),
            performance: Vec::new(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Options {
    /// Execute benchmarks that match a wildcard pattern.
    #[arg(short, long, value_name = "pattern", default_value_t=String::from("*"), display_order=0)]
    benchmarks: String,
    
    #[command(flatten)]
    pub verbose: Verbosity<InfoLevel>,
}

fn main() -> anyhow::Result<()> {
    let options = Options::parse();

    let log_level = match options.verbose.log_level_filter() {
        LevelFilter::Off => "off",
        LevelFilter::Error => "error",
        LevelFilter::Warn => "warn",

        LevelFilter::Info => "info",
        LevelFilter::Debug => "debug",
        LevelFilter::Trace => "trace",
    };

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
       .format_timestamp(None)
       .init();

    let benchmark_matcher = WildMatch::new(&options.benchmarks);

    let mut results: HashMap<&str, Performance> = HashMap::new();

    let number_density = 0.8;
    let benchmark = Benchmark::default();

    for n in [256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072] {
    
    let microstate_2d = benchmarks::place_hard_hyperspheres::<Point<Cartesian<2>>, Point<Cartesian<2>>, 2>(n, number_density)?;
    let microstate_oriented_2d = benchmarks::place_hard_hyperspheres::<OrientedPoint<Cartesian<2>, Angle>, OrientedPoint<Cartesian<2>, Angle>, 2>(n, number_density)?;

    if benchmark_matcher.matches("mc_hard_sphere_2d") {
        let mut mc_hard_sphere_2d = mc::HardSphere::with_microstate(&microstate_2d)?;
        info!("mc_hard_sphere_2d: {} disks at number density {}", n, number_density);
        let performance = benchmark.benchmark_one(&mut mc_hard_sphere_2d)?;

        let entry = results.entry("mc_hard_sphere_2d").or_insert_with(|| Performance::with_units("ms / step".to_string()));
        entry.n.push(n);
        entry.performance.push(performance);
    }

    if benchmark_matcher.matches("mc_lennard_jones_2d") {
        let mut mc_lennard_jones_2d = mc::LennardJones::with_microstate(&microstate_2d)?;
        info!("mc_lennard_jones_2d: {} disks at number density {}", n, number_density);
        let performance = benchmark.benchmark_one(&mut mc_lennard_jones_2d)?;

        let entry = results.entry("mc_lennard_jones_2d").or_insert_with(|| Performance::with_units("ms / step".to_string()));
        entry.n.push(n);
        entry.performance.push(performance);
    }

    if benchmark_matcher.matches("mc_hexagon_2d") {
        let mut mc_hexagon_2d = mc::RegularPolygon::with_microstate(&microstate_oriented_2d)?;
        info!("mc_hexagon_2d: {} hexagons at number density {}", n, number_density);
        let performance = benchmark.benchmark_one(&mut mc_hexagon_2d)?;

        let entry = results.entry("mc_hexagon_2d").or_insert_with(|| Performance::with_units("ms / step".to_string()));
        entry.n.push(n);
        entry.performance.push(performance);
    }

    let microstate_3d = benchmarks::place_hard_hyperspheres::<Point<Cartesian<3>>, Point<Cartesian<3>>, 3>(n, number_density)?;
    let microstate_oriented_3d = benchmarks::place_hard_hyperspheres::<OrientedPoint<Cartesian<3>, Versor>, OrientedPoint<Cartesian<3>, Versor>, 3>(n, number_density)?;

    if benchmark_matcher.matches("mc_hard_sphere_3d") {
        let mut mc_hard_sphere_3d = mc::HardSphere::with_microstate(&microstate_3d)?;
        info!("mc_hard_sphere_3d: {} disks at number density {}", n, number_density);
        let performance = benchmark.benchmark_one(&mut mc_hard_sphere_3d)?;

        let entry = results.entry("mc_hard_sphere_3d").or_insert_with(|| Performance::with_units("ms / step".to_string()));
        entry.n.push(n);
        entry.performance.push(performance);
    }

    if benchmark_matcher.matches("mc_lennard_jones_3d") {
        let mut mc_lennard_jones_3d = mc::LennardJones::with_microstate(&microstate_3d)?;
        info!("mc_lennard_jones_3d: {} spheres at number density {}", n, number_density);
        let performance = benchmark.benchmark_one(&mut mc_lennard_jones_3d)?;

        let entry = results.entry("mc_lennard_jones_3d").or_insert_with(|| Performance::with_units("ms / step".to_string()));
        entry.n.push(n);
        entry.performance.push(performance);
    }

    if benchmark_matcher.matches("mc_octahedron_3d") {
        let mut mc_octahedron_3d = mc::Octahedron::with_microstate(&microstate_oriented_3d)?;
        info!("mc_octahedron_3d: {} octahedra at number density {}", n, number_density);
        let performance = benchmark.benchmark_one(&mut mc_octahedron_3d)?;

        let entry = results.entry("mc_octahedron_3d").or_insert_with(|| Performance::with_units("ms / step".to_string()));
        entry.n.push(n);
        entry.performance.push(performance);
    }

    }
    let results_json = serde_json::to_string(&results)?;
    println!("{results_json}");

    Ok(())
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::collections::HashMap;

use hoomd_microstate::SiteKey;
use hoomd_spatial::{AllPairs, HashCell, VecCell};
use log::info;
use clap::Parser;
use clap_verbosity_flag::{Verbosity, InfoLevel};
use clap_verbosity_flag::log::LevelFilter;
use serde::Serialize;

use hoomd_microstate::property::OrientedPoint;
use hoomd_vector::{Angle, Cartesian, Versor};

use benchmarks::{Benchmark, mc};
use wildmatch::WildMatch;

#[derive(Serialize)]
struct Performance {
    units: String,
    n: Vec<usize>,
    hash_cell_performance: Vec<f64>,
    vec_cell_performance: Vec<f64>,
    all_pairs_performance: Vec<f64>,
}

impl Performance {
    fn with_units(units: String) -> Self {
        Self {
            units,
            n: Vec::new(),
            hash_cell_performance: Vec::new(),
            vec_cell_performance: Vec::new(),
            all_pairs_performance: Vec::new(),
            
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Options {
    /// Execute benchmarks that match a wildcard pattern.
    #[arg(short, long, value_name = "pattern", default_value_t=String::from("*"), display_order=0)]
    benchmarks: String,

    /// Smallest system size to benchmark.
    #[arg(short, long, default_value_t=4096, display_order=0)]
    n_min: usize,

    /// Largest system size to benchmark (defaults to the smallest).
    #[arg(long, display_order=0)]
    n_max: Option<usize>,
        
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

    let mut n = options.n_min;

    loop {
    
    let microstate_2d = benchmarks::place_hard_hyperspheres::<OrientedPoint<Cartesian<2>, Angle>, OrientedPoint<Cartesian<2>, Angle>, 2>(n, number_density)?;

    if benchmark_matcher.matches("mc_hard_sphere_2d") {
        info!("mc_hard_sphere_2d: {} disks at number density {}", n, number_density);
        let mut mc_hard_sphere_2d = mc::HardSphere::<2, VecCell<SiteKey, 2>>::with_microstate(&microstate_2d)?;
        let performance = benchmark.benchmark_one(&mut mc_hard_sphere_2d)?;

        let entry = results.entry("mc_hard_sphere_2d").or_insert_with(|| Performance::with_units("ms / step".to_string()));
        entry.n.push(n);
        entry.vec_cell_performance.push(performance);

        let mut mc_hard_sphere_2d = mc::HardSphere::<2, AllPairs<SiteKey>>::with_microstate(&microstate_2d)?;
        let performance = benchmark.benchmark_one(&mut mc_hard_sphere_2d)?;
        entry.all_pairs_performance.push(performance);

        let mut mc_hard_sphere_2d = mc::HardSphere::<2, HashCell<SiteKey, 2>>::with_microstate(&microstate_2d)?;
        let performance = benchmark.benchmark_one(&mut mc_hard_sphere_2d)?;
        entry.hash_cell_performance.push(performance);
    }

    if benchmark_matcher.matches("mc_lennard_jones_2d") {
        info!("mc_lennard_jones_2d: {} disks at number density {}", n, number_density);
        let mut mc_lennard_jones_2d = mc::LennardJones::<2, VecCell<SiteKey, 2>>::with_microstate(&microstate_2d)?;
        let performance = benchmark.benchmark_one(&mut mc_lennard_jones_2d)?;

        let entry = results.entry("mc_lennard_jones_2d").or_insert_with(|| Performance::with_units("ms / step".to_string()));
        entry.n.push(n);
        entry.vec_cell_performance.push(performance);

        let mut mc_lennard_jones_2d = mc::LennardJones::<2, AllPairs<SiteKey>>::with_microstate(&microstate_2d)?;
        let performance = benchmark.benchmark_one(&mut mc_lennard_jones_2d)?;
        entry.all_pairs_performance.push(performance);

        let mut mc_lennard_jones_2d = mc::LennardJones::<2, HashCell<SiteKey, 2>>::with_microstate(&microstate_2d)?;
        let performance = benchmark.benchmark_one(&mut mc_lennard_jones_2d)?;
        entry.hash_cell_performance.push(performance);
    }

    if benchmark_matcher.matches("mc_hexagon_2d") {
        info!("mc_hexagon_2d: {} hexagons at number density {}", n, number_density);
        let mut mc_hexagon_2d = mc::RegularPolygon::<VecCell<SiteKey, 2>>::with_microstate(&microstate_2d)?;
        let performance = benchmark.benchmark_one(&mut mc_hexagon_2d)?;

        let entry = results.entry("mc_hexagon_2d").or_insert_with(|| Performance::with_units("ms / step".to_string()));
        entry.n.push(n);
        entry.vec_cell_performance.push(performance);

        let mut mc_hexagon_2d = mc::RegularPolygon::<AllPairs<SiteKey>>::with_microstate(&microstate_2d)?;
        let performance = benchmark.benchmark_one(&mut mc_hexagon_2d)?;
        entry.all_pairs_performance.push(performance);

        let mut mc_hexagon_2d = mc::RegularPolygon::<HashCell<SiteKey, 2>>::with_microstate(&microstate_2d)?;
        let performance = benchmark.benchmark_one(&mut mc_hexagon_2d)?;
        entry.hash_cell_performance.push(performance);
    }

    let microstate_3d = benchmarks::place_hard_hyperspheres::<OrientedPoint<Cartesian<3>, Versor>, OrientedPoint<Cartesian<3>, Versor>, 3>(n, number_density)?;

    if benchmark_matcher.matches("mc_hard_sphere_3d") {
        info!("mc_hard_sphere_3d: {} disks at number density {}", n, number_density);
        let mut mc_hard_sphere_3d = mc::HardSphere::<3, VecCell<SiteKey, 3>>::with_microstate(&microstate_3d)?;
        let performance = benchmark.benchmark_one(&mut mc_hard_sphere_3d)?;

        let entry = results.entry("mc_hard_sphere_3d").or_insert_with(|| Performance::with_units("ms / step".to_string()));
        entry.n.push(n);
        entry.vec_cell_performance.push(performance);

        let mut mc_hard_sphere_3d = mc::HardSphere::<3, AllPairs<SiteKey>>::with_microstate(&microstate_3d)?;
        let performance = benchmark.benchmark_one(&mut mc_hard_sphere_3d)?;
        entry.all_pairs_performance.push(performance);

        let mut mc_hard_sphere_3d = mc::HardSphere::<3, HashCell<SiteKey, 3>>::with_microstate(&microstate_3d)?;
        let performance = benchmark.benchmark_one(&mut mc_hard_sphere_3d)?;
        entry.hash_cell_performance.push(performance);
    }

    if benchmark_matcher.matches("mc_lennard_jones_3d") {
        info!("mc_lennard_jones_3d: {} spheres at number density {}", n, number_density);
        let mut mc_lennard_jones_3d = mc::LennardJones::<3, VecCell<SiteKey, 3>>::with_microstate(&microstate_3d)?;
        let performance = benchmark.benchmark_one(&mut mc_lennard_jones_3d)?;

        let entry = results.entry("mc_lennard_jones_3d").or_insert_with(|| Performance::with_units("ms / step".to_string()));
        entry.n.push(n);
        entry.vec_cell_performance.push(performance);

        let mut mc_lennard_jones_3d = mc::LennardJones::<3, AllPairs<SiteKey>>::with_microstate(&microstate_3d)?;
        let performance = benchmark.benchmark_one(&mut mc_lennard_jones_3d)?;
        entry.all_pairs_performance.push(performance);

        let mut mc_lennard_jones_3d = mc::LennardJones::<3, HashCell<SiteKey, 3>>::with_microstate(&microstate_3d)?;
        let performance = benchmark.benchmark_one(&mut mc_lennard_jones_3d)?;
        entry.hash_cell_performance.push(performance);
    }

    if benchmark_matcher.matches("mc_octahedron_3d") {
        info!("mc_octahedron_3d: {} octahedra at number density {}", n, number_density);
        let mut mc_octahedron_3d = mc::Octahedron::<VecCell<SiteKey, 3>>::with_microstate(&microstate_3d)?;
        let performance = benchmark.benchmark_one(&mut mc_octahedron_3d)?;

        let entry = results.entry("mc_octahedron_3d").or_insert_with(|| Performance::with_units("ms / step".to_string()));
        entry.n.push(n);
        entry.vec_cell_performance.push(performance);

        let mut mc_octahedron_3d = mc::Octahedron::<AllPairs<SiteKey>>::with_microstate(&microstate_3d)?;
        let performance = benchmark.benchmark_one(&mut mc_octahedron_3d)?;
        entry.all_pairs_performance.push(performance);

        let mut mc_octahedron_3d = mc::Octahedron::<HashCell<SiteKey, 3>>::with_microstate(&microstate_3d)?;
        let performance = benchmark.benchmark_one(&mut mc_octahedron_3d)?;
        entry.hash_cell_performance.push(performance);
    }

    n *= 2;
    if n > options.n_max.unwrap_or(options.n_min) {
        break;
    }
    }
    let results_json = serde_json::to_string(&results)?;
    println!("{results_json}");

    Ok(())
}

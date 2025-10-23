// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use log::info;

use clap::Parser;
use clap_verbosity_flag::{Verbosity, InfoLevel};
use clap_verbosity_flag::log::LevelFilter;

use hoomd_microstate::property::Point;
use hoomd_vector::Cartesian;

use benchmarks::{Benchmark, mc};
use wildmatch::WildMatch;

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

    let n = 4096;
    let number_density = 0.8;
    let benchmark = Benchmark::default();
    
    let microstate_2d = benchmarks::place_hard_hyperspheres::<Point<Cartesian<2>>, Point<Cartesian<2>>, 2>(n, number_density)?;

    if benchmark_matcher.matches("mc_hard_sphere_2d") {
        let mut mc_hard_sphere_2d = mc::HardSphere::with_microstate(&microstate_2d)?;
        info!("mc_hard_sphere_2d: {} disks at number density {}", n, number_density);
        benchmark.benchmark_one(&mut mc_hard_sphere_2d)?;
    }

    if benchmark_matcher.matches("mc_step_2d") {
        let mut mc_step_2d = mc::Step::with_microstate(&microstate_2d)?;
        info!("mc_step_2d: {} disks at number density {}", n, number_density);
        benchmark.benchmark_one(&mut mc_step_2d)?;
    }

    if benchmark_matcher.matches("mc_lennard_jones_2d") {
        let mut mc_lennard_jones_2d = mc::LennardJones::with_microstate(&microstate_2d)?;
        info!("mc_lennard_jones_2d: {} disks at number density {}", n, number_density);
        benchmark.benchmark_one(&mut mc_lennard_jones_2d)?;
    }

    let microstate_3d = benchmarks::place_hard_hyperspheres::<Point<Cartesian<3>>, Point<Cartesian<3>>, 3>(n, number_density)?;

    if benchmark_matcher.matches("mc_hard_sphere_3d") {
        let mut mc_hard_sphere_3d = mc::HardSphere::with_microstate(&microstate_3d)?;
        info!("mc_hard_sphere_3d: {} disks at number density {}", n, number_density);
        benchmark.benchmark_one(&mut mc_hard_sphere_3d)?;
    }

    if benchmark_matcher.matches("mc_step_3d") {
        let mut mc_step_3d = mc::Step::with_microstate(&microstate_3d)?;
        info!("mc_step_3d: {} spheres at number density {}", n, number_density);
        benchmark.benchmark_one(&mut mc_step_3d)?;
    }

    if benchmark_matcher.matches("mc_lennard_jones_3d") {
        let mut mc_lennard_jones_3d = mc::LennardJones::with_microstate(&microstate_3d)?;
        info!("mc_lennard_jones_3d: {} spheres at number density {}", n, number_density);
        benchmark.benchmark_one(&mut mc_lennard_jones_3d)?;
    }

    Ok(())
}

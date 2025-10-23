// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use log::info;

use hoomd_microstate::property::Point;
use hoomd_vector::Cartesian;

use benchmarks::{Benchmark, mc};

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let n = 4096;
    let number_density = 0.8;
    
    let microstate_2d = benchmarks::place_hard_hyperspheres::<Point<Cartesian<2>>, Point<Cartesian<2>>, 2>(n, number_density)?;

    let mut mc_hard_sphere_2d = mc::HardSphere::with_microstate(&microstate_2d)?;
    
    let benchmark = Benchmark::default();
    info!("[mc] hard disk benchmark: {} disks at number density {}", n, number_density);
    benchmark.benchmark_one(&mut mc_hard_sphere_2d)?;

    let mut mc_step_2d = mc::Step::with_microstate(&microstate_2d)?;
    info!("[mc] hard sphere + well benchmark: {} disks at number density {}", n, number_density);
    benchmark.benchmark_one(&mut mc_step_2d)?;

    let microstate_3d = benchmarks::place_hard_hyperspheres::<Point<Cartesian<3>>, Point<Cartesian<3>>, 3>(n, number_density)?;

    let mut mc_hard_sphere_3d = mc::HardSphere::with_microstate(&microstate_3d)?;
    
    info!("[mc] hard sphere benchmark: {} disks at number density {}", n, number_density);
    benchmark.benchmark_one(&mut mc_hard_sphere_3d)?;

    let mut mc_step_3d = mc::Step::with_microstate(&microstate_3d)?;
    info!("[mc] hard sphere + well benchmark: {} spheres at number density {}", n, number_density);
    benchmark.benchmark_one(&mut mc_step_3d)?;

    Ok(())
}

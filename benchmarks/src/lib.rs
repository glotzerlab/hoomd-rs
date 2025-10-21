// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Common benchmarking methods.
use std::time::Instant;

use hoomd_spatial::{VecCell, PointsInBall};
use rand::distr::Distribution;
use log::{debug, trace};

use hoomd_geometry::shape::{Hypercuboid};
use hoomd_microstate::{boundary::{GenerateGhosts, Periodic}, property::Position, Body, Microstate, MicrostateBuilder, SiteKey, Transform 
};
use hoomd_interaction::{pairwise::{Isotropic, OverlapPenalty, Expanded}, CutoffPair};
use hoomd_simulation::{macrostate::Isothermal, Simulation};
use hoomd_mc::{Sweep, Translate, Trial, UniformIn, QuickInsert};
use hoomd_vector::Cartesian;


/// Place n hard hyperspheres in a D-dimensional hypercube at the given number density
///
/// The spheres have diameter 1, are randomly placed in a non-overlapping configuration.
pub fn place_hard_hyperspheres<B, S, const D: usize>(n: usize, number_density: f64) -> anyhow::Result<Microstate<B, S, VecCell<SiteKey, D>, Periodic<Hypercuboid<D>>>> where
B: Default + Position<Position = Cartesian<D>> + Transform<S> + Copy,
S: Default + Position<Position = Cartesian<D>> + Copy,
UniformIn<S, Periodic<Hypercuboid<D>>>: Distribution<Body<B, S>>,
Periodic<Hypercuboid<D>>: GenerateGhosts<S>,
VecCell<SiteKey, D>: PointsInBall<Cartesian<D>, SiteKey>,
{
    let box_length = (n as f64 / number_density).powf(1.0 / (D as f64));
    let sigma = 1.0;
    let macrostate = Isothermal { temperature: 1.0 };

    debug!("Initializing...");

    let cell_list = VecCell::new(sigma, (box_length / sigma).ceil() as u32 + 2);
    let boundary = Periodic::new(sigma,
        Hypercuboid::<D>::with_equal_edges(
            box_length.try_into()?))?;

    let mut microstate = MicrostateBuilder::<B, S, VecCell<SiteKey, D>, Periodic<Hypercuboid<D>>>::with_spatial_data_and_boundary(cell_list, boundary)
        .try_build()?;

    let translate = Translate::with_maximum_distance(0.1.try_into()?);
    let translate_sweep = Sweep(translate);

    let distribution = UniformIn {
        boundary: microstate.boundary().clone(),
        template_sites: vec![S::default()],
    };
    let mut quick_insert = QuickInsert::new(distribution, n);    

    let f = OverlapPenalty::default();
    let overlap_penalty = Expanded { f, delta: sigma };
    let evaluator = Isotropic(overlap_penalty);
    let insert_hamiltonian = CutoffPair { r_cut: sigma,
        evaluator };

    while !quick_insert.is_complete() {
        quick_insert.apply(&mut microstate, &insert_hamiltonian);
        translate_sweep.apply(
            &mut microstate,
            &insert_hamiltonian,
            &macrostate
        );
        microstate.increment_step();

        if microstate.step().is_multiple_of(100) {
            trace!("Step {}: N = {} / {}", microstate.step(), microstate.sites().len(), n);
        }        
    }

    Ok(microstate)
    }

pub fn benchmark<S>(simulation: &mut S, warmup_steps: usize, benchmark_steps: usize, repeat: usize) -> anyhow::Result<()>
where S: Simulation
{
    debug!("Warm up for {warmup_steps} steps...");
    for _ in 0..warmup_steps {
        simulation.advance()?;
    }

    debug!("Benchmark {benchmark_steps} steps {repeat} time(s)...");
    for _ in 0..repeat {
        let time = Instant::now();
        let start_step = simulation.step();
        
        for _ in 0..benchmark_steps {
            simulation.advance()?;
        }

        let run_time = time.elapsed().as_secs_f64();
        let steps = simulation.step() - start_step;
        
        trace!("Completed {steps} steps in {run_time} seconds.");
        println!("{} steps/s", steps as f64 / run_time);
    }
    

    Ok(())
}

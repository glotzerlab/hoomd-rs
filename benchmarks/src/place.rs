// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement particle placement methods.

use hoomd_spatial::{PointsNearBall, VecCell};
use log::{debug, trace};
use rand::distr::Distribution;

use hoomd_geometry::shape::Hypercuboid;
use hoomd_interaction::{
    PairwiseCutoff,
    pairwise::{Expanded, Isotropic, OverlapPenalty},
};
use hoomd_mc::{QuickInsert, Sweep, Translate, Trial, UniformIn};
use hoomd_microstate::{
    Body, Microstate, SiteKey, Transform,
    boundary::{GenerateGhosts, Periodic},
    property::Position,
};
use hoomd_simulation::macrostate::Isothermal;
use hoomd_vector::Cartesian;

/// Place n hard hyperspheres in a D-dimensional hypercube at the given number density
///
/// The spheres have diameter 1, are randomly placed in a non-overlapping configuration.
pub fn place_hard_hyperspheres<B, S, const D: usize>(
    n: usize,
    number_density: f64,
) -> anyhow::Result<Microstate<B, S, VecCell<SiteKey, D>, Periodic<Hypercuboid<D>>>>
where
    B: Default + Position<Position = Cartesian<D>> + Transform<S> + Copy,
    S: Default + Position<Position = Cartesian<D>> + Copy,
    UniformIn<S, Periodic<Hypercuboid<D>>>: Distribution<Body<B, S>>,
    Periodic<Hypercuboid<D>>: GenerateGhosts<S>,
    VecCell<SiteKey, D>: PointsNearBall<Cartesian<D>, SiteKey>,
{
    let box_length = (n as f64 / number_density).powf(1.0 / (D as f64));
    let sigma = 1.0;
    let macrostate = Isothermal { temperature: 1.0 };

    debug!("Initializing {n} points in {D} dimensions with number density {number_density}...");

    let cell_list = VecCell::builder()
        .nominal_search_radius(1.0.try_into()?)
        .build();
    let boundary = Periodic::new(
        sigma,
        Hypercuboid::<D>::with_equal_edges(box_length.try_into()?),
    )?;

    let mut microstate = Microstate::builder()
        .spatial_data(cell_list)
        .boundary(boundary)
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
    let insert_hamiltonian = PairwiseCutoff {
        r_cut: sigma,
        evaluator,
    };

    while !quick_insert.is_complete() {
        quick_insert.apply(&mut microstate, &insert_hamiltonian);
        translate_sweep.apply(&mut microstate, &insert_hamiltonian, &macrostate);
        microstate.increment_step();

        if microstate.step().is_multiple_of(100) {
            trace!(
                "Step {}: N = {} / {}",
                microstate.step(),
                microstate.sites().len(),
                n
            );
        }
    }

    Ok(microstate)
}

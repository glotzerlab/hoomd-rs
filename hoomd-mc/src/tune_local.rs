// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement local trial tune

use log::{debug, trace};
use rand::RngExt;
use std::fmt::Display;

use super::{Adjust, Count, LocalTrial, TuneOptions, tune_by_scaling};
use hoomd_interaction::DeltaEnergyOne;
use hoomd_microstate::{
    Body, Microstate, SiteKey, Tagged, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::Position,
};
use hoomd_simulation::macrostate::Temperature;
use hoomd_spatial::PointUpdate;

/// Tune local trial moves.
pub(crate) fn tune_local_trial<P1, P2, B, S, X, C, L, H, MA, F>(
    local_trial: &mut L,
    microstate: &Microstate<B, S, X, C>,
    hamiltonian: &H,
    macrostate: &MA,
    options: &TuneOptions,
    should_move_body: F,
) where
    P1: Copy,
    P2: Copy,
    B: Copy + Default + Transform<S> + Position<Position = P1>,
    S: Copy + Default + Position<Position = P2>,
    X: PointUpdate<P2, SiteKey>,
    L: LocalTrial<B> + Adjust + Display,
    H: DeltaEnergyOne<B, S, X, C>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    MA: Temperature,
    F: Fn(&Tagged<Body<B, S>>) -> bool,
{
    let kt = macrostate.temperature();
    let mut rng = microstate.counter().make_rng();
    let mut trial = Body::<B, S>::default();

    debug!("Tuning trial move size");

    for step in 0..options.steps {
        let mut count = Count::default();
        while count.total() < options.samples as u64 {
            let body_index = rng.random_range(0..microstate.bodies().len());
            let body = &microstate.bodies()[body_index];
            if !should_move_body(body) {
                continue;
            }

            trial.clone_from(&body.item);

            match microstate
                .boundary()
                .wrap(local_trial.propose(&mut rng, trial.properties))
            {
                Ok(new_properties) => {
                    trial.properties = new_properties;

                    let delta_h = hamiltonian.delta_energy_one(microstate, body_index, &trial);
                    if delta_h != f64::INFINITY
                        && (delta_h <= 0.0 || rng.random::<f64>() < (-delta_h / kt).exp())
                    {
                        count.accepted += 1;
                    } else {
                        count.rejected += 1;
                    }
                }
                Err(_) => count.rejected += 1,
            }
        }

        if let Some(acceptance_ratio) = count.acceptance_ratio() {
            let steps = options.steps;
            trace!(
                "-- {step} / {steps}: {local_trial:.4} - {:.1}%",
                acceptance_ratio * 100.0
            );
        }

        tune_by_scaling(local_trial, options.target_acceptance, &count);
    }

    debug!("-- complete: {local_trial}");
}

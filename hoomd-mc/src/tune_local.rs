// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement local trial tune

use log::{debug, trace};
use rand::Rng;
use std::fmt::Display;

use super::{Adjust, Count, LocalTrial, tune};
use hoomd_interaction::DeltaEnergyOne;
use hoomd_microstate::{
    Body, Microstate, SiteKey, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::Position,
};
use hoomd_simulation::macrostate::Temperature;
use hoomd_spatial::PointUpdate;
use hoomd_utility::valid::OpenUnitIntervalNumber;

/// Tune local trial moves.
pub(crate) fn tune_local_trial<P, B, S, X, C, L, H, MA>(
    local_trial: &mut L,
    microstate: &Microstate<B, S, X, C>,
    hamiltonian: &H,
    macrostate: &MA,
    target_acceptance: OpenUnitIntervalNumber,
    samples: usize,
    steps: usize,
) where
    P: Copy,
    B: Copy + Default + Transform<S> + Position<Position = P>,
    S: Copy + Default + Position<Position = P>,
    X: PointUpdate<P, SiteKey>,
    L: LocalTrial<B> + Adjust + Display,
    H: DeltaEnergyOne<B, S, X, C>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    MA: Temperature,
{
    let kt = macrostate.temperature();
    let mut rng = microstate.counter().make_rng();
    let mut trial = Body::<B, S>::default();

    debug!("Tuning trial move size");

    for step in 0..steps {
        let mut count = Count::default();
        for _ in 0..samples {
            let body_index = rng.random_range(0..microstate.bodies().len());
            trial.clone_from(&microstate.bodies()[body_index].item);

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
            trace!(
                "-- {step} / {steps}: {local_trial:.4} - {:.1}%",
                acceptance_ratio * 100.0
            );
        }

        tune(local_trial, target_acceptance, &count);
    }

    debug!("-- complete: {local_trial}");
}

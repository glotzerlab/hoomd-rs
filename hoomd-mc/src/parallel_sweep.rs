// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement ParallelSweep

use std::iter;

use rand::{rngs::StdRng, seq::IndexedRandom, Rng, RngCore, SeedableRng};
use rayon::prelude::*;

use super::{Count, LocalTrial, Trial};
use hoomd_interaction::DeltaEnergyOne;
use hoomd_microstate::{
    Body, Microstate, SiteKey, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::Position,
};
use hoomd_simulation::macrostate::Temperature;
use hoomd_spatial::PointUpdate;
use hoomd_utility::valid::PositiveReal;

use crate::checkerboard::{Checkerboard, Cover};

#[derive(Default)]
struct BodyTrial<B, S> {
    body: Body::<B, S>,
    body_index: usize,
    accepted: bool,
}
    

// ParallelSweep
// 1. Make checkerboard (or update a cached one for efficiency)
// 2. Place body indices in checkerboard spaces.
// 3. Loop until a sufficient number of trial moves is performed.
// 4. Loop over all space indices by color.
// 5. Prepare trial bodies vec (same len the current space indices)
// 6. for each space index (of the current color) in parallel:
//    * Choose a body randomly.
//    * Propose a trial move.
//    * Reject if the body center leaves the current space.
//    * Accept or reject the move as in Sweep.
//    * Store the result in the trial bodies
// 7. Process the trial bodies output and apply the accepted moves.

pub struct ParallelSweep<L> {
    pub body_interaction_range: PositiveReal,
    pub local_trial: L,
}

impl<L> ParallelSweep<L> where
{
    /// TODO
    #[inline(always)]
    fn update_checkerboard<P, B, S, X, C>(
        &self,
        microstate: &Microstate<B, S, X, C>,
        ) -> (C::Checkerboard, Vec<Vec<usize>>)
        where
    B: Position<Position = P>,
    C: Cover<P>
    {
    let mut checkerboard_rng = microstate.counter().index(u64::MAX).make_rng();
    let checkerboard = microstate.boundary().cover(&mut checkerboard_rng, self.body_interaction_range);

    let mut spaces: Vec<_> = iter::repeat_n(Vec::new(), checkerboard.num_spaces()).collect();
    for (body_index, body) in microstate.bodies().iter().enumerate() {
        let space_index = checkerboard.point_to_space_index(body.item.properties.position());
        spaces[space_index.expect("body should be inside the checkerboard")].push(body_index);
    }

    (checkerboard, spaces)
    }

    /// TODO
    #[inline(always)]
    fn generate_trial_moves<P, B, S, X, C, H, K>(
        &self,
        body_trials: &mut Vec<BodyTrial<B,S>>,
        microstate: &Microstate<B, S, X, C>,
        hamiltonian: &H,
        kt: f64,
        checkerboard: &K,
        spaces: &Vec<Vec<usize>>,
        space_indices: &Vec<usize>,
        ) where
    P: Copy,
    B: Copy + Default + Transform<S> + Position<Position = P> + Send + Sync,
    S: Copy + Default + Position<Position = P> + Send + Sync,
    L: LocalTrial<B> + Sync,
    H: DeltaEnergyOne<B, S, X, C> + Sync,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S> + Sync,
    K: Checkerboard<P> + Sync,
    X: Sync,
        {
        body_trials.resize_with(space_indices.len(), Default::default);

        body_trials.par_iter_mut().zip(space_indices).for_each(|(body_trial, space_index)| {
            body_trial.accepted = false;
            let space_index = *space_index;
            let mut rng = microstate.counter().index(space_index as u64).make_rng();

            if let Some(body_index) = spaces[space_index].choose(&mut rng) {
                let body_index = *body_index;
                body_trial.body.clone_from(&microstate.bodies()[body_index].item);
                body_trial.body_index = body_index;

                match microstate
                    .boundary()
                    .wrap(self.local_trial.propose(&mut rng, body_trial.body.properties))
                {
                    Ok(new_properties) if checkerboard.point_to_space_index(new_properties.position()) != Some(space_index) => body_trial.accepted = false,
                    Ok(new_properties) => {
                        body_trial.body.properties = new_properties;

                        let delta_h = hamiltonian.delta_energy_one(microstate, body_index, &body_trial.body);
                        body_trial.accepted = delta_h != f64::INFINITY && (delta_h <= 0.0 || rng.random::<f64>() < (-delta_h / kt).exp());
                    }
                    Err(_) => body_trial.accepted = false,
                }
            }
        });
    }        

    /// TODO
    #[inline(always)]
    fn update_bodies<P, B, S, X, C>(
        &self,
        microstate: &mut Microstate<B, S, X, C>,
        body_trials: &Vec<BodyTrial<B,S>>,
        ) -> Count

        where
    P: Copy,
    B: Copy + Default + Transform<S> + Position<Position = P>,
    S: Copy + Default + Position<Position = P>,
    X: PointUpdate<P, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
        {
        let mut count = Count::default();

        for body_trial in body_trials {
            if !body_trial.accepted {
                count.rejected += 1;
                continue;
            }

            if microstate
                .update_body_properties(body_trial.body_index, body_trial.body.properties)
                .is_ok()
                {
                    count.accepted += 1;
                } else {
                    count.rejected += 1;
                }
            }
        count
    }
}

impl<P, B, S, X, C, L, H, MA> Trial<Microstate<B, S, X, C>, H, MA> for ParallelSweep<L>
where
    P: Copy,
    B: Copy + Default + Transform<S> + Position<Position = P> + Send + Sync,
    S: Copy + Default + Position<Position = P> + Send + Sync,
    X: PointUpdate<P, SiteKey> + Sync,
    L: LocalTrial<B> + Sync,
    H: DeltaEnergyOne<B, S, X, C> + Sync,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S> + Cover<P> + Sync,
    MA: Temperature,
{
    type Count = Count;

    /// TODO
    #[inline]
    fn apply(
        &self,
        microstate: &mut Microstate<B, S, X, C>,
        hamiltonian: &H,
        macrostate: &MA,
    ) -> Self::Count {
        let kt = macrostate.temperature();
        
        let (checkerboard, spaces) = self.update_checkerboard(microstate);
               
        let mut body_trials: Vec<BodyTrial<B,S>> = Vec::with_capacity(checkerboard.space_indices_by_color().first()
            .expect("checkerboard should have at least one color").len());

        let mut count = Self::Count::default();

        while count.total() < microstate.bodies().len() as u64 {
            for space_indices in checkerboard.space_indices_by_color() {           
                self.generate_trial_moves(
                        &mut body_trials,
                        microstate,
                        hamiltonian,
                        *kt,
                        &checkerboard,
                        &spaces,
                        space_indices);

                count += self.update_bodies(microstate, &body_trials);
            }
            microstate.increment_substep();
        }

        count
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Sweep
*/

use super::{Count, DeltaEnergyOne, LocalTrial, Trial};
use hoomd_microstate::{Body, Microstate, Tagged, Transform};
use rand::Rng;

/** Apply a local trial move to every body in the microsstate.
*/
pub struct Sweep<L> {
    pub local: L,
}

impl<L> Sweep<L> {
    pub fn new(local: L) -> Self {
        Self {
            local,
        }
    }
}

impl<B, S, C, L, H> Trial<Microstate<B, S, C>, H> for Sweep<L>
where
    B: Copy + Clone + Default + Transform<S>,
    S: Clone + Default,
    L: LocalTrial<B>,
    H: DeltaEnergyOne<B, S, C>,
{
    type Count = Count;
    type Macrostate = f64;

    #[inline]
    fn apply(&self, microstate: &mut Microstate<B, S, C>, hamiltonian: &H, state: &Self::Macrostate) -> Self::Count where
    {
        let kt = state;
        let mut rng = microstate.counter().make_rng();
        let mut count = Self::Count::default();
        let mut trial = Tagged::<Body<B, S>>::default();

        // For loop over a range instead of bodies().iter() as the latter holds an immutable borrow.
        // The call to `update_body_properties` makes a mutable borrow of microstate.
        for body_index in 0..microstate.bodies().len() {
            trial.clone_from(&microstate.bodies()[body_index]);
            trial.item.properties = self.local.propose(&mut rng, trial.item.properties);

            // TODO: Handle boundary conditions

            let delta_h = hamiltonian.delta_energy_one(microstate, &trial);
            if rng.random::<f64>() < (-delta_h / kt).exp() {
                microstate.update_body_properties(body_index, trial.item.properties);
                count.accepted += 1;
            } else {
                count.rejected += 1;
            }
        }

        microstate.increment_substep();
        count
    }
}

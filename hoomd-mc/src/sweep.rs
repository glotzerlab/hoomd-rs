// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Sweep
*/

use super::{Count, DeltaEnergyOne, LocalTrial, Trial};
use hoomd_microstate::{Body, Microstate, Tagged, Transform};
use hoomd_random::Counter;
use rand::Rng;

/** Apply a local trial move to every body in the microsstate.
*/
pub struct Sweep<'a, L, H> {
    pub local: L,
    pub kt: &'a f64,
    pub hamiltonian: &'a H,
}

impl<'a, L, H> Sweep<'a, L, H> {
    pub fn new(local: L, kt: &'a f64, hamiltonian: &'a H) -> Self {
        Self { local, kt, hamiltonian }
    }
}

impl<'a, B, S, C, L, H> Trial<Microstate<B, S, C>> for Sweep<'a, L, H>
where
    B: Copy + Clone + Default + Transform<S>,
    S: Copy + Clone + Default,
    L: LocalTrial<B>,
    H: DeltaEnergyOne,
{
    type Count = Count;

    #[inline]
    fn apply(&self, microstate: &mut Microstate<B, S, C>) -> Self::Count {
        // TODO: Implement a convenience to create Counter from Microstate so this code isn't repeated?
        let mut rng =
            Counter::new(microstate.step(), microstate.substep(), microstate.seed()).make_rng();
        let mut count = Self::Count::default();
        let mut trial = Tagged::<Body<B, S>>::default();

        // For loop over a range instead of bodies().iter() as the latter holds an immutable borrow.
        // The call to `update_body_properties` makes a mutable borrow of microstate.
        for body_index in 0..microstate.bodies().len() {
            trial.clone_from(&microstate.bodies()[body_index]);
            trial.item.properties = self.local.propose(&mut rng, trial.item.properties);

            // TODO: Handle boundary conditions

            let delta_h = self.hamiltonian.delta_energy_one(microstate, &trial);
            if rng.random::<f64>() < (delta_h / self.kt).exp() {
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

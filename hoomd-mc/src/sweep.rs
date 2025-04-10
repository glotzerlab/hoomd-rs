// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Sweep
*/

use super::{Count, Sweep, LocalTrial, Trial};
use hoomd_random::Counter;
use hoomd_microstate::{Body, Microstate, Transform};

impl<L> Sweep<L> {
    pub fn new(local: L) -> Self {
        Self { local }
    }
}

impl<B, S, C, L> Trial<Microstate<B, S, C>> for Sweep<L> where
B: Copy+Clone+Default+Transform<S>,
S: Copy+Clone+Default,
L: LocalTrial<B>
 {
    type Count = Count;

    #[inline]
    fn apply(&self, microstate: &mut Microstate<B, S, C>) -> Self::Count {
        // TODO: Implement a convenience to create Counter from Microstate so this code isn't repeated?
        let mut rng = Counter::new(microstate.step(), microstate.substep(), microstate.seed())
            .make_rng();
        let mut count = Self::Count::default();
        let mut trial = Body::<B, S>::default();

        // For loop over a range instead of bodies().iter() as the latter holds an immutable borrow.
        // The call to `update_body_properties` makes a mutable borrow of microstate.
        for body_index in 0..microstate.bodies().len() {
            trial.clone_from(&microstate.bodies()[body_index].item);
            trial.properties = self.local.propose(&mut rng, trial.properties);

            // TODO: Handle boundary conditions
            // TODO: Calculate delta E, apply acceptance criterion
            microstate.update_body_properties(body_index, trial.properties);
            count.accepted += 1;
        }

        microstate.increment_substep();
        count
    }
}

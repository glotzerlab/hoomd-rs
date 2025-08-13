// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `QuickInsert`
*/

use super::{Count, Trial};
use hoomd_interaction::{DeltaEnergyInsert, TotalEnergy};
use hoomd_microstate::{
    Body, Microstate, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::Position,
};

use rand::distr::Distribution;

/** Add bodies to the microstate with random configurations.

[`QuickInsert`] allows you to *quickly* insert many particles into the
microstate. It does so by *breaking detailed balance*, so you should use it
only during the initialization phase of your simulation where you prepare a
microstate for later equilibration. The [`QuickInsert`] protocol is a refinement
of the `QuickCompress` protocol implemented in HOOMD-blue with the advantage
that you can keep the boundary and any of your barriers fixed while randomly
inserting particles.

[`QuickInsert`] works only with hard particle potentials that go to infinity
when overlapping. It works best with potentials that always evaluate to
non-negative values. If your model uses soft and/or attractive interactions, you
can use an alternate Hamiltonian during the initialization phase.

When you `apply` [`QuickInsert`] to a microstate, it:
1. Checks the total energy of the given hamiltonian. If greater than zero, return
   immediately.
2. Generate a random body and attempt to insert it into the microstate. Reject
   any insertion that would result in an infinite energy. Accept in all other cases.
3. Repeat step 2 up to `attempts_per_apply` times or when `state` particles have
   been inserted, whichever comes first.

You **must** combine [`QuickInsert`] with local trial moves that relieve the
stress produced by random insertions to make room for more insertions.

TODO: Document effectiveness of basic vs advanced configurations.

The generic type names are:
* `D`: The body distribution.

TODO: Example
*/
pub struct QuickInsert<D> {
    /// Sample random bodies to insert.
    pub distribution: D,
    /// Number of insertion attempts per call to `apply`.
    pub attempts_per_apply: usize,
}

impl<V, B, S, C, D, H> Trial<Microstate<B, S, C>, H> for QuickInsert<D>
where
    B: Position<Vector = V> + Transform<S>,
    S: Position<Vector = V> + Default,
    D: Distribution<Body<B, S>>,
    H: DeltaEnergyInsert<B, S, C> + TotalEnergy<Microstate<B, S, C>>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    type Count = Count;
    type Macrostate = usize;

    #[inline]
    fn apply(
        &self,
        microstate: &mut Microstate<B, S, C>,
        hamiltonian: &H,
        state: &Self::Macrostate,
    ) -> Self::Count {
        let mut count = Self::Count::default();

        if hamiltonian.total_energy(microstate) <= 0.0 {
            let mut rng = microstate.counter().make_rng();

            for _ in 0..self.attempts_per_apply {
                let new_body = self.distribution.sample(&mut rng);

                let delta_energy = hamiltonian.delta_energy_insert(microstate, &new_body);
                if delta_energy.is_finite() && microstate.add_body(new_body).is_ok() {
                    count.accepted += 1;
                    if count.accepted >= *state as u64 {
                        break;
                    }
                } else {
                    count.rejected += 1;
                }
            }
        }

        microstate.increment_substep();
        count
    }
}

// TODO: test

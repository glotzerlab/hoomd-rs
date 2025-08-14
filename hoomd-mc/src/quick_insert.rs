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

#[derive(PartialEq)]
enum State {
    Running,
    Complete,
}

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
    distribution: D,

    /// Total number of particles to insert.
    target: usize,

    /// Maximum number of overlapping inserts allowed.
    allowed_overlaps: usize,

    /// Count of insertions completed
    inserted: usize,

    /// Current stage of the method.
    state: State
}

impl<D> QuickInsert<D>
{
    pub fn new(distribution: D, target: usize) -> Self {
        Self {
            distribution,
            target,
            allowed_overlaps: (target / 8).max(1),
            inserted: 0,
            state: State::Running,
        }
    }
    
    pub fn is_complete(&self) -> bool {
        self.state == State::Complete
    }

    #[inline]
    pub fn apply<V, B, S, C, H, T>(
        &mut self,
        microstate: &mut Microstate<B, S, C>,
        hamiltonian: &H,
        local_trial: &T,
        state: &T::Macrostate,
    ) -> Count
where
    B: Position<Vector = V> + Transform<S>,
    S: Position<Vector = V> + Default,
    D: Distribution<Body<B, S>>,
    H: DeltaEnergyInsert<B, S, C> + TotalEnergy<Microstate<B, S, C>>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    T: Trial<Microstate<B, S, C>, H>,
     {
    let mut count = Count::default();

    // Perform no work at all if already complete.
    if self.is_complete() {
        return count;
    }

    let energy = hamiltonian.total_energy(microstate);

    // The quick insert protocol is not complete until the energy has reached 0.
    if energy <= 0.0 && self.inserted >= self.target {
        self.state = State::Complete;
        return count;
    }

    // Scaling the number of insertion attempts with the target number of insertions
    // is a good way to ensure that there are sufficient attempts on each call to
    // apply. Larger boxes will naturally get more insertion attempts. At the same
    // time, we need to limit the total strain caused by the insertions. Count
    // the number of insertions that cause overlaps and exit early when there
    // are too many.
    if energy <= 0.0 {
        let mut rng = microstate.counter().make_rng();
        let mut insertions_with_overlaps = 0;

        for _ in 0..self.target{
            let new_body = self.distribution.sample(&mut rng);

            let delta_energy = hamiltonian.delta_energy_insert(microstate, &new_body);
            if delta_energy.is_finite() && microstate.add_body(new_body).is_ok() {
                count.accepted += 1;
                self.inserted += 1;

                if delta_energy > 0.0 {
                    insertions_with_overlaps += 1;
                }
                
                if self.inserted == self.target || insertions_with_overlaps >= self.allowed_overlaps {
                    break;
                }
            } else {
                count.rejected += 1;
            }
        }
    }

    microstate.increment_substep();

    // Applying local trial moves is critical to the success of the quick
    // insert protocol. Require that users pass in a local trial type and
    // apply it at the appropriate time.
    local_trial.apply(microstate, hamiltonian, state);
    
    count
}
}

// TODO: test

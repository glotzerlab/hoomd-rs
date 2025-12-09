// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `QuickCompress`
use rand::Rng;

use super::Count;
use hoomd_geometry::{Map, Scale, Volume};
use hoomd_interaction::{TotalEnergy};
use hoomd_microstate::{
    Body, Microstate, SiteKey, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::Position,
};
use hoomd_spatial::PointUpdate;

/// Track the state of a given `QuickCompress` instance.
#[derive(Default, Clone, Debug, PartialEq)]
enum State<C> {
    /// `compress` has not yet been called.
    #[default]
    Startup,
    /// Waiting for all overlaps to be removed.
    ///
    /// The first field indicates whether the compression that led to this state
    /// achieved the target volume. The second field stores a clone of the last
    /// boundary known to `QuickCompress`.
    Waiting(bool, C),
    /// Compressing the system
    Compressing,
    /// The system is at the target volume and all overlaps have been removed.
    ///
    /// The field stores a clone of the final boundary.
    Complete(C),
}

/// Quickly compress a microstate to a target volume.
///
/// TODO: Document
/// TODO: Validate `maximum_delta` is in the range (0,1).
#[derive(Clone, Debug, PartialEq)]
pub struct QuickCompress<C> {
    /// Final boundary volume after the compression algorithm completes.
    target_volume: f64,

    /// Per-site energy threshold at which to reject compression moves.
    maximum_energy_per_site: f64,

    /// Current state of the method.
    state: State<C>,

    /// Maximum value of delta.
    maximum_delta: f64,
}

impl<C> QuickCompress<C> {
    #[inline]
    pub fn with_target_volume(target_volume: f64) -> Self {
        Self {
            target_volume,
            maximum_energy_per_site: 25.0,
            state: State::default(),
            maximum_delta: 0.01,
        }
    }
    
    #[inline]
    pub fn is_complete(&self) -> bool {
        matches!(self.state, State::Complete(_))
    }

    #[inline]
    pub fn target_volume(&self) -> f64 {
        self.target_volume
    }

    #[inline]
    pub fn set_target_volume(&mut self, target_volume: f64) where C: Clone {
        // Callers will expect `QuickCompress::compress` to seamlessly move
        // toward the new target. How to do that depends on the current state.
        // We cannot always switch to `Startup` because a caller *might* call
        // `set_target_volume` every time they call `compress` and that would
        // result in no progress. 
        if self.target_volume != target_volume {
            self.state = match self.state {
                State::Startup => State::Startup,
                State::Complete(_) => State::Startup,
                State::Waiting(_, ref last_known_boundary) => State::Waiting(false, last_known_boundary.clone()),
                State::Compressing => State::Compressing,
            };
        }

        self.target_volume = target_volume;
    }

    #[inline]
    pub fn maximum_energy_per_site(&self) -> f64 {
        self.maximum_energy_per_site
    }

    #[inline]
    pub fn maximum_energy_per_site_mut(&mut self) -> &mut f64 {
        &mut self.maximum_energy_per_site
    }

    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    #[inline]
    pub fn compress<P, B, S, X, H>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        hamiltonian: &H,
    )
    where
        P: Copy,
        B: Clone + Position<Position = P> + Transform<S>,
        S: Clone + Position<Position = P> + Default,
        X: Clone + PointUpdate<P, SiteKey>,
        H: TotalEnergy<Microstate<B, S, X, C>>,
        C: Clone + Map<P> + Wrap<B> + Wrap<S> + GenerateGhosts<S> + Volume + Scale + PartialEq,
        Microstate<B, S, X, C>: Clone,
    {
        // TODO: pass through map argument
        let mut rng = microstate.counter().make_rng();
        microstate.increment_substep();

        match self.state {
            State::Startup => {
                self.state = State::Waiting(false, microstate.boundary().clone());
            }
            State::Complete(ref final_boundary) if final_boundary == microstate.boundary() => {
            }
            State::Complete(_) => {
                // While it is not intended that callers change the microstate's
                // boundary outside `QuickCompress`, it is a possibility. Should
                // a caller do so, they would be surprised that `QuickCompress`
                // failed to compress toward the target.
                self.state = State::Waiting(false, microstate.boundary().clone());
            }
            State::Compressing => {
                let delta = rng.random::<f64>() * self.maximum_delta;
                let current_volume = microstate.boundary().volume();

                let trial_volume = if self.target_volume > current_volume {
                    (current_volume * (1.0 + delta)).min(self.target_volume)
                } else {
                    (current_volume * (1.0 - delta)).max(self.target_volume)
                };

                let trial_boundary = microstate.boundary().scale((trial_volume / current_volume)
                    .try_into().expect("both volumes should be positive"));
                let Ok(trial_microstate) = microstate.clone_with_boundary(trial_boundary, |_| true) else { return };
                let delta_energy = hamiltonian.delta_energy_total(microstate, &trial_microstate); 

                if delta_energy > self.maximum_energy_per_site * microstate.sites().len() as f64 {
                    // The trial state is too strained. Remain in the `Compressing` state
                    // to try again on the next call.
                } else {
                    // The trial state is valid. Transition to the waiting state and indicate
                    // whether this trial achieved the target.
                    self.state = State::Waiting(trial_volume == self.target_volume, trial_microstate.boundary().clone());
                    *microstate = trial_microstate;
                }
            }
            State::Waiting(is_final, ref last_known_boundary) => {
                if hamiltonian.total_energy(microstate) <= 0.0 {
                    if is_final && last_known_boundary == microstate.boundary() {
                        self.state = State::Complete(microstate.boundary().clone());
                    } else {
                        self.state = State::Compressing;
                    }
                }
            }
        }
    }
}

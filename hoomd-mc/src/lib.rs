// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

//! Apply the Metropolis Monte Carlo simulation method to systems of particles.
//!
//! TODO: Expand documentation.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign};

use hoomd_microstate::Microstate;
use hoomd_utility::valid::{OpenUnitIntervalNumber, PositiveReal};

mod hypercuboid;
mod parallel_sweep;
mod quick_compress;
mod quick_insert;
mod rotate;
mod sweep;
mod translate;
pub(crate) mod tune_local;
mod uniform_in;

pub use hypercuboid::HypercuboidCheckerboard;
pub use parallel_sweep::ParallelSweep;
pub use quick_compress::QuickCompress;
pub use quick_insert::QuickInsert;
pub use rotate::Rotate;
pub use sweep::Sweep;
pub use translate::Translate;
pub use uniform_in::UniformIn;

/// Propose trial moves in the microstate, evaluate the changes in energy and accept or reject accordingly.
///
/// `Trial` describes a type that applies trial moves to microstates. Specifically,
/// the method `apply` will attempt one or more individual trial moves to the
/// microstate. For each individual move, it evaluates the change in energy with
/// the given `hamiltonian`, then accepts or rejects the trial based on the `state`
/// parameters.
///
/// Each type of trial move in *hoomd-rs* implements the `Trial` trait so that they
/// may be used as generic arguments in higher level functions.
///
/// See [`Sweep`] or any of the other implementations of `Trial` for code examples.
///
/// The generic type names are:
/// * `MI`: The [`Microstate`] type.
/// * `H`: The Hamiltonian type.
/// * `MA`: The [`Macrostate`](hoomd_simulation::macrostate) type.
pub trait Trial<MI, H, MA> {
    /// Represent the number of accepted and rejected individual trial moves.
    ///
    /// Most implementations of `Trial` will use [`crate::Count`] directly. Some
    /// may provide more granular detail broken down by move type.
    type Count;

    /// Apply the trial move(s).
    ///
    /// A given type that implements `Trial` may perform one or many trial moves
    /// in a single call to `apply`. The returned value informs the caller how many
    /// trial moves were accepted and rejected (possibly broken down by type).
    fn apply(&mut self, microstate: &mut MI, hamiltonian: &H, macrostate: &MA) -> Self::Count;
}

/// Propose a new configuration for given body properties.
///
/// A *local* trial move is one applied to a specific body in the microstate.
/// Implementations of [`Trial`], such as [`Sweep`], apply a given local move
/// to one or more bodies in the microstate.
///
/// Use one of the provided local trials to [`Translate`] and/or [`Rotate`]
/// bodies or implement your own custom [`LocalTrial`].
///
/// Local trial moves **MUST** satisfy *local detailed balance*,
/// as defined in [Manousiouthakis & Deem](https://doi.org/10.1063/1.477973).
///
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
pub trait LocalTrial<B> {
    /// Propose a new configuration for the given body properties.
    #[must_use]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B;
}

/// Accepted and rejected trial moves.
///
/// A [`Trial`] reports the number moves it accepts and rejects via `Count` (or
/// some variation on `Count`). `Count` implements [`Add`], [`AddAssign`], and
/// convenience methods that compute often used properties, like the acceptance
/// rate.
///
/// # Example
///
/// Count the total number of trial moves performed over a number of sweeps:
/// ```
/// use hoomd_interaction::Zero;
/// use hoomd_mc::{Count, Sweep, Translate, Trial};
/// use hoomd_microstate::{Body, Microstate, property::Position};
/// use hoomd_simulation::macrostate::Isothermal;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let macrostate = Isothermal { temperature: 1.0 };
/// let mut microstate = Microstate::new();
/// microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
/// let d = 0.1;
/// let translate = Translate::with_maximum_distance(d.try_into()?);
/// let mut translate_sweep = Sweep(translate);
///
/// let mut count = Count::default();
///
/// for _ in 0..1_000 {
///     count += translate_sweep.apply(&mut microstate, &Zero, &macrostate);
///     microstate.increment_step();
/// }
///
/// assert_eq!(count.total(), 1_000);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Count {
    /// The number of accepted moves.
    pub accepted: u64,
    /// The number of rejected moves.
    pub rejected: u64,
}

impl Count {
    /// The total number of trial moves.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_mc::Count;
    ///
    /// let count = Count {
    ///     accepted: 2_000,
    ///     rejected: 8_000,
    /// };
    /// let total = count.total();
    ///
    /// assert_eq!(total, 10_000);
    /// ```
    #[inline]
    #[must_use]
    pub fn total(&self) -> u64 {
        self.accepted + self.rejected
    }

    /// The fraction of moves that were accepted.
    ///
    /// The acceptance ratio is the ratio of accepted moves to total moves.
    /// `acceptance_ratio` returns `Some(ratio)` when the number of total moves is
    /// nonzero and `None` when there are 0 moves.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_mc::Count;
    ///
    /// let count = Count {
    ///     accepted: 2_000,
    ///     rejected: 8_000,
    /// };
    /// let acceptance_ratio = count.acceptance_ratio();
    ///
    /// assert_eq!(acceptance_ratio, Some(0.2));
    /// ```
    ///
    /// ```
    /// use hoomd_mc::Count;
    ///
    /// let count = Count::default();
    /// let acceptance_ratio = count.acceptance_ratio();
    ///
    /// assert_eq!(acceptance_ratio, None);
    /// ```
    #[inline]
    #[must_use]
    pub fn acceptance_ratio(&self) -> Option<f64> {
        let total = self.total();

        if total > 0 {
            Some(self.accepted as f64 / total as f64)
        } else {
            None
        }
    }
}

impl AddAssign for Count {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.accepted += rhs.accepted;
        self.rejected += rhs.rejected;
    }
}

impl Add for Count {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Count {
            accepted: self.accepted + rhs.accepted,
            rejected: self.rejected + rhs.rejected,
        }
    }
}

/// Partition space into sets of spaces where trial moves can safely be applied in parallel.
///
/// [`ParallelSweep`] uses a [`Checkerboard`] when selecting bodies for
/// parallel trial moves. A well-behaved checkerboard:
///
/// 1. Colors spaces such that any body with its position in a space cannot possibly
///    interact with any body positioned in any other space of the same color.
/// 1. Covers all points within the boundary of the simulation.
/// 3. Respects periodic boundary conditions (when present).
///
/// Given a boundary, construct a suitable [`Checkerboard`] via the [`Cover`] trait.
pub trait Checkerboard<P> {
    /// Determine the space index of a given point.
    ///
    /// Space indices must be in the range `[0,num_spaces)`. [`ParallelSweep`]
    /// uses the space index as an array index. `point_to_space_index` maps
    /// a real-valued, D-dimensional point to the linear index.
    fn point_to_space_index(&self, point: &P) -> Option<usize>;

    /// The indices of all spaces, grouped by color.
    ///
    /// In the return value, the outer slice's length is the number of colors
    /// in the checkerboard. Element of that slice contains the indices of all
    /// the spaces of that color.
    fn space_indices_by_color(&self) -> &[Vec<usize>];

    /// The total number of spaces in the checkerboard.
    fn num_spaces(&self) -> usize;
}

/// Construct a [`Checkerboard`] that covers all points in this boundary.
pub trait Cover<P> {
    /// The checkerboard type associated with this boundary.
    type Checkerboard: Checkerboard<P>;

    /// Construct a [`Checkerboard`] that covers all points in this boundary.
    ///
    /// The constructed [`Checkerboard`] must place spaces assuming that
    /// any body might interact with another body at distances less than
    /// `interaction_range`. [`ParallelSweep`] must reject trial moves from one
    /// space to another. To make simulations ergodic, `cover` must randomly
    /// place the checkerboard boundaries using the provided `rng`.
    fn cover<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        interaction_range: PositiveReal,
    ) -> Self::Checkerboard;

    /// Update a given checkerboard to match this boundary.
    ///
    /// After calling `cover_into`, `checkerboard` will have the same properties
    /// as the return value of `self.cover(rng, interaction_range)`. However,
    /// `cover_into` may be able to reuse existing dynamically allocated memory
    /// in `checkerboard` or avoid some calculations completely (e.g. when the
    /// checkerboard dimensions remain the same).
    fn cover_into<R: Rng + ?Sized>(
        &self,
        checkerboard: &mut Self::Checkerboard,
        rng: &mut R,
        interaction_range: PositiveReal,
    );
}

/// Change the maximum size of a local trial move.
pub trait Adjust {
    /// Change the maximum trial move size by the given scale factor.
    fn adjust(&mut self, factor: PositiveReal);
}

/// Tune trial move maximum sizes toward a target acceptance ratio.
///
/// [`Trial`] moves that implement [`Tune`] can automatically adjust
/// their trial move size to achieve a desired acceptance ratio.
/// The tuning is performed in the context of a given microstate,
/// Hamiltonian, and macrostate **but the microstate is not modified**.
pub trait Tune<P, B, S, X, C, L, H, MA> {
    /// Tune the trial move maximum size to achieve a given acceptance ratio.
    ///
    /// Use [`tune_default`] unless you have a specific need to adjust the
    /// tuning parameters.
    ///
    /// [`tune`] performs `samples` individual trial moves to measure the
    /// current acceptance ratio. It then adjusts the trial move size
    /// to increase or decrease the acceptance ratio as needed over
    /// `steps` iterations.
    ///
    /// [`tune_default`]: Tune::tune_default
    fn tune(
        &mut self,
        microstate: &Microstate<B, S, X, C>,
        hamiltonian: &H,
        macrostate: &MA,
        target_acceptance: OpenUnitIntervalNumber,
        samples: usize,
        steps: usize,
    );

    /// Tune the trial move maximum size with default parameters.
    ///
    /// The defaults are:
    /// - `target_acceptance`: 0.2
    /// - `samples`: 8,000
    /// - `steps`: 32
    #[inline]
    fn tune_default(
        &mut self,
        microstate: &Microstate<B, S, X, C>,
        hamiltonian: &H,
        macrostate: &MA,
    ) {
        self.tune(
            microstate,
            hamiltonian,
            macrostate,
            0.2.try_into().expect("hard-coded constant should be valid"),
            8_000,
            32,
        );
    }
}

/// Tune adjustable move sizes toward a target acceptance ratio.
///
/// Prefer [`Sweep::tune`] or [`ParallelSweep::tune`] when tuning local
/// trial move sizes.
///
/// Pass [`tune`] a target acceptance ratio and the move `count` obtained during
/// a sampling period with the current `local_trial` move size. [`tune`] will
/// scale the trial move size by the factor:
/// ```math
/// \frac{a + \gamma}{t + \gamma}
/// ```
/// where $` a `$ is the current acceptance (from `count`), $` t `$ is the target
/// and $` \gamma = 1.5 `$.
#[expect(
    clippy::missing_panics_doc,
    reason = "Panic would occur due to a bug in hoomd-rs."
)]
#[inline]
pub fn tune<L>(local_trial: &mut L, target_acceptance: OpenUnitIntervalNumber, count: &Count)
where
    L: Adjust,
{
    const GAMMA: f64 = 1.5;

    if let Some(acceptance_ratio) = count.acceptance_ratio() {
        let scale = (acceptance_ratio + GAMMA) / (target_acceptance.get() + GAMMA);
        local_trial.adjust(scale.try_into().expect("scale should always be positive"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::check;
    use hoomd_vector::Cartesian;

    #[test]
    fn test_count() {
        let default = Count::default();
        assert_eq!(default.accepted, 0);
        assert_eq!(default.rejected, 0);
        assert_eq!(default.total(), 0);
        assert_eq!(default.acceptance_ratio(), None);

        let a = Count {
            accepted: 1_500,
            rejected: 500,
        };
        check!(a.total() == 2_000);
        check!(a.acceptance_ratio() == Some(0.75));

        let mut b = Count {
            accepted: 500,
            rejected: 200,
        };
        b += a;
        check!(b.accepted == 2_000);
        check!(b.rejected == 700);
        check!(b.total() == 2_700);
    }

    #[test]
    fn test_tune() -> anyhow::Result<()> {
        let high = Count {
            accepted: 1_500,
            rejected: 500,
        };

        let mut local_trial = Translate::<Cartesian<2>>::with_maximum_distance(1.0.try_into()?);
        tune(&mut local_trial, 0.2.try_into()?, &high);
        check!(local_trial.maximum_distance().get() > 1.0);

        let low = Count {
            accepted: 100,
            rejected: 1_900,
        };

        let mut local_trial = Translate::<Cartesian<2>>::with_maximum_distance(1.0.try_into()?);
        tune(&mut local_trial, 0.2.try_into()?, &low);
        check!(local_trial.maximum_distance().get() < 1.0);

        let zero = Count {
            accepted: 0,
            rejected: 2_000,
        };

        let mut local_trial = Translate::<Cartesian<2>>::with_maximum_distance(1.0.try_into()?);
        tune(&mut local_trial, 0.2.try_into()?, &zero);
        check!(local_trial.maximum_distance().get() < 1.0);

        Ok(())
    }
}

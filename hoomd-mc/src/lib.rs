// Copyright (c) 2024-2025 The Regents of the University of Michigan.
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
use std::ops::AddAssign;

mod manifold_translate;
mod quick_insert;
mod rotate;
mod sweep;
mod translate;
mod uniform_in;

pub use manifold_translate::{HyperbolicTranslate, SphericalTranslate};
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
/// * `M`: The [`Microstate`](hoomd_microstate::Microstate) type.
/// * `H`: The Hamiltonian type.
pub trait Trial<M, H> {
    /// Represent the number of accepted and rejected individual trial moves.
    ///
    /// Most implementations of `Trial` will use [`crate::Count`] directly. Some
    /// may provide more granular detail broken down by move type.
    type Count;

    /// Represent the macrostate parameter(s) used when evaluating move acceptance.
    type Macrostate;

    /// Apply the trial move(s).
    ///
    /// A given type that implements `Trial` may perform one or many trial moves
    /// in a single call to `apply`. The returned value informs the caller how many
    /// trial moves were accepted and rejected (possibly broken down by type).
    fn apply(&self, microstate: &mut M, hamiltonian: &H, state: &Self::Macrostate) -> Self::Count;
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
/// A [`Trial`] reports the number moves it accepts and rejects via `Count`
/// (or some variation on `Count`). `Count` implements [`AddAssign`] and convenience
/// methods that compute often used properties, like the acceptance rate.
///
/// # Example
///
/// Count the total number of trial moves performed over a number of sweeps:
/// ```
/// use hoomd_interaction::Zero;
/// use hoomd_mc::{Count, Sweep, Translate, Trial};
/// use hoomd_microstate::{Body, Microstate, property::Position};
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
/// let d = 0.1;
/// let translate = Translate {
///     maximum_distance: d.try_into()?,
/// };
/// let translate_sweep = Sweep(translate);
///
/// let mut count = Count::default();
///
/// for _ in 0..1_000 {
///     count += translate_sweep.apply(&mut microstate, &Zero, &1.0);
///     microstate.increment_step();
/// }
///
/// assert_eq!(count.total(), 1_000);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count() {
        let default = Count::default();
        assert_eq!(default.accepted, 0);
        assert_eq!(default.rejected, 0);
        assert_eq!(default.total(), 0);
        assert_eq!(default.acceptance_ratio(), None);

        let a = Count {
            accepted: 1_500,
            rejected: 500,
        };
        assert_eq!(a.total(), 2_000);
        assert_eq!(a.acceptance_ratio(), Some(0.75));

        let mut b = Count {
            accepted: 500,
            rejected: 200,
        };
        b += a;
        assert_eq!(b.accepted, 2_000);
        assert_eq!(b.rejected, 700);
        assert_eq!(b.total(), 2_700);
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Interactions as a function of one variable.
//!
//! TODO: Document

mod boxcar;
mod expanded;
mod harmonic;
mod harmonic_repulsion;
mod lennard_jones;
mod lennard_jones_gauss;
mod overlap_penalty;
mod shifted;
mod weeks_chandler_anderson;
mod xplor;

pub use boxcar::Boxcar;
pub use expanded::Expanded;
pub use harmonic::Harmonic;
pub use harmonic_repulsion::HarmonicRepulsion;
pub use lennard_jones::LennardJones;
pub use lennard_jones_gauss::LennardJonesGauss;
pub use overlap_penalty::OverlapPenalty;
pub use shifted::Shifted;
pub use weeks_chandler_anderson::WeeksChandlerAnderson;
pub use xplor::Xplor;

/// Computes energy as a function of one variable.
///
/// A univariate energy is function only of one variable. Often, this is
/// distance between two points: $`U(r)`$, but it could be a distance from a
/// point to a surface, or any other single variable.
///
/// Implement [`UnivariateEnergy`] on a custom type or use one of the provided
/// potentials in [`univariate`] in MD or MC simulations.
///
/// * Use [`UnivariateEnergy`] in combination with [`Isotropic`] and
///   [`PairwiseCutoff`] to model pairwise interactions between sites.
///
/// [`univariate`]: crate::univariate
/// [`Isotropic`]: crate::pairwise::Isotropic
/// [`PairwiseCutoff`]: crate::PairwiseCutoff
///
/// # Examples
///
/// Set a custom potential using a closure:
/// ```
/// use hoomd_interaction::univariate::UnivariateEnergy;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let a = 2.0;
/// let custom = |r: f64| a / (r.powi(12));
///
/// let energy = custom.energy(1.0);
/// assert_eq!(energy, 2.0);
/// # Ok(())
/// # }
/// ```
///
/// Implement a custom potential via a type:
/// ```
/// use hoomd_interaction::univariate::UnivariateEnergy;
///
/// struct Custom {
///     a: f64,
/// }
///
/// impl UnivariateEnergy for Custom {
///     fn energy(&self, r: f64) -> f64 {
///         self.a / r.powi(12)
///     }
/// }
///
/// let custom = Custom { a: 2.0 };
///
/// let energy = custom.energy(1.0);
/// assert_eq!(energy, 2.0);
/// ```
pub trait UnivariateEnergy {
    /// Compute the energy as a function of one variable.
    /// ```math
    /// U(r)
    /// ```
    #[must_use]
    fn energy(&self, r: f64) -> f64;
}

impl<F> UnivariateEnergy for F
where
    F: Fn(f64) -> f64,
{
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self(r)
    }
}

/// Computes force as a function of one variable.
///
/// An univariate force is a function only one variable. Often, this is
/// distance between two points: $`f(r)`$, but it could be a distance from a
/// point to a surface, or any other single variable. The direction of the
/// resulting force vector depends on how the [`UnivariateForce`] is applied.
///
/// Implement [`UnivariateForce`] on a custom type or use one of the provided
/// forces in [`univariate`] in MD simulations.
///
/// [`univariate`]: crate::univariate
pub trait UnivariateForce {
    /// Compute force as a function of one variable.
    ///
    /// When the force is associated with a potential energy [`UnivariateEnergy`],
    /// it must follow:
    /// ```math
    /// -\frac{\mathrm{d} U}{\mathrm{d} r}
    /// ```
    #[must_use]
    fn force(&self, r: f64) -> f64;
}

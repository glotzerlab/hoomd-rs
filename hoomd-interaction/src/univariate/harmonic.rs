// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Harmonic`]

use super::{UnivariateEnergy, UnivariateForce};

/// Quadratic potential well centered at a given separation distance.
///
/// ```math
/// U = \frac{1}{2} k (r - r_0)^2
/// ```
///
/// Compute the harmonic potential and force as a function of `r` with
/// equilibrium spring length `r_0`.
///
/// # Examples
///
/// ```
/// use approxim::{assert_abs_diff_eq, assert_relative_eq};
/// use hoomd_interaction::univariate::{
///     Harmonic, UnivariateEnergy, UnivariateForce,
/// };
///
/// let k = 2.0;
/// let r_0 = 0.0;
///
/// let harmonic = Harmonic { k, r_0 };
/// assert_abs_diff_eq!(harmonic.energy(0.0), 0.0);
/// assert_relative_eq!(harmonic.energy(1.0), 1.0);
/// assert_abs_diff_eq!(harmonic.force(1.0), -2.0, epsilon = 1e-12);
/// ```
///
/// The parameters are public fields and may be accessed directly:
///
/// ```
/// use hoomd_interaction::univariate::Harmonic;
///
/// let mut harmonic = Harmonic { k: 1.0, r_0: 0.0 };
/// harmonic.k = 5.0;
/// harmonic.r_0 = 1.0;
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Harmonic {
    /// Spring constant $`[\mathrm{energy}] [\mathrm{length}]^{-2}`$.
    pub k: f64,
    /// Equilibrium spring length $`[\mathrm{length}]`$.
    pub r_0: f64,
}

impl UnivariateEnergy for Harmonic {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        0.5 * self.k * (r - self.r_0) * (r - self.r_0)
    }
}

impl UnivariateForce for Harmonic {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        -self.k * (r - self.r_0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use rstest::*;

    #[rstest]
    fn zero_energy_point(#[values(1.0, 2.0, 5.0, 10.0)] k: f64, #[values(0.0, 1.0, 2.0)] r_0: f64) {
        let harmonic = Harmonic { k, r_0 };

        assert_eq!(harmonic.k, k);
        assert_eq!(harmonic.r_0, r_0);

        assert_eq!(harmonic.energy(r_0), 0.0);
        assert_eq!(harmonic.force(r_0), 0.0);
    }

    #[rstest]
    fn general_case(#[values(1.0, 2.0, 5.0, 10.0)] k: f64, #[values(0.0, 1.0, 2.0)] r_0: f64) {
        let r = 5.0;
        let harmonic = Harmonic { k, r_0 };

        assert_eq!(harmonic.k, k);
        assert_eq!(harmonic.r_0, r_0);

        let expected_energy = 0.5 * k * (r - r_0) * (r - r_0);
        let expected_force = -k * (r - r_0);

        assert_relative_eq!(harmonic.energy(r), expected_energy);
        assert_relative_eq!(harmonic.force(r), expected_force);
    }
}

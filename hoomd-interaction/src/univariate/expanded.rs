// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Expanded`]

use super::{UnivariateEnergy, UnivariateForce};

/// Expand another potential.
///
/// ```math
/// U(r) = f(r - \delta)
/// ```
///
/// # Example
///
/// Expanded Lennard-Jones:
/// ```
/// use approxim::{assert_abs_diff_eq, assert_relative_eq};
/// use hoomd_interaction::univariate::{
///     Expanded, LennardJones, UnivariateEnergy,
/// };
///
/// let epsilon = 1.5;
/// let sigma = 1.0;
/// let delta = 0.75;
/// let lj: LennardJones = LennardJones { epsilon, sigma };
/// let expanded_lj = Expanded { f: lj, delta };
///
/// assert_abs_diff_eq!(
///     expanded_lj.energy(1.0),
///     expanded_lj.f.energy(1.0 - 0.75)
/// );
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Expanded<F> {
    /// The original potential.
    pub f: F,
    /// $`\delta`$ value $`[\mathrm{length}]`$.
    pub delta: f64,
}

impl<F> Default for Expanded<F>
where
    F: Default,
{
    /// Construct a shifted potential with default parameters
    ///
    /// The defaults are:
    /// `f = F::default()`
    /// `delta = 0.0`
    #[inline]
    fn default() -> Self {
        Self {
            f: F::default(),
            delta: 0.0,
        }
    }
}

impl<F: UnivariateEnergy> UnivariateEnergy for Expanded<F> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.f.energy(r - self.delta)
    }
}

impl<F: UnivariateForce> UnivariateForce for Expanded<F> {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        self.f.force(r - self.delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::univariate::LennardJones;
    use approxim::{assert_abs_diff_eq, assert_relative_eq};
    use rstest::*;

    #[rstest]
    fn special_points_12_6(
        #[values(1.0, 2.0, 12.125, 0.25)] epsilon: f64,
        #[values(1.0, 2.0, 0.5)] sigma: f64,
        #[values(0.125, 0.5, 2.0)] delta: f64,
    ) {
        let lj: LennardJones = LennardJones { epsilon, sigma };
        let expanded_lj = Expanded { f: lj, delta };

        // Zero crossing
        assert_abs_diff_eq!(expanded_lj.energy(sigma + delta), 0.0);
        assert_relative_eq!(expanded_lj.force(sigma + delta), 24.0 * epsilon / sigma);

        // Bottom of the well
        assert_relative_eq!(
            expanded_lj.energy(2.0_f64.powf(1.0 / 6.0) * sigma + delta),
            -epsilon
        );
        assert_abs_diff_eq!(
            expanded_lj.force(2.0_f64.powf(1.0 / 6.0) * sigma + delta),
            0.0,
            epsilon = 1e-12
        );

        // r = 2 sigma
        assert_relative_eq!(
            expanded_lj.energy(2.0 * sigma + delta),
            -63.0 / 1024.0 * epsilon
        );
        assert_relative_eq!(
            expanded_lj.force(2.0 * sigma + delta),
            -93.0 / 512.0 * epsilon / sigma
        );
    }
}

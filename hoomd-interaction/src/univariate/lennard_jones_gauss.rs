// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`LennardJonesGauss`]

use super::{UnivariateEnergy, UnivariateForce};

/// Double-well potential with a steep repulsive core
///
/// ```math
/// U(r) = 1[\mathrm{energy}]\cdot\left[ \left(\frac{\ell}{r}\right)^{12} - 2\left(\frac{\ell}{r}\right)^{6}\right] - \varepsilon \exp \left(-\frac{(r/\ell-r_0)^2}{2\sigma^2}\right)
/// ```
///
/// Compute the Lennard-Jones-Gauss (LJG) potential and force as a function of `r`.
///
/// # Examples
///
/// ```
/// use approxim::{assert_abs_diff_eq, assert_relative_eq};
/// use hoomd_interaction::univariate::{
///     UnivariateEnergy, UnivariateForce, LennardJonesGauss,
/// };
///
/// let epsilon = 0.5;
/// let sigma_squared = 0.5;
/// let r_0 = 0.5_f64.powf(1.0 / 6.0);
/// let scale = 1.0_f64;
///
/// let lennard_jones_gauss: LennardJonesGauss = LennardJonesGauss {
///     epsilon,
///     sigma_squared,
///     r_0,
///     scale,
/// };
/// assert_relative_eq!(
///     lennard_jones_gauss.energy(0.5_f64.powf(1.0 / 6.0)),
///     -epsilon,
///     epsilon = 1e-12
/// );
/// ```
///
/// The parameters are public fields and may be accessed directly:
///
/// ```
/// use hoomd_interaction::univariate::LennardJonesGauss;
///
/// let mut lennard_jones_gauss: LennardJonesGauss = LennardJonesGauss {
///     epsilon: 1.5,
///     sigma_squared: 0.02,
///     r_0: 3.2,
///     scale: 1.0,
/// };
/// lennard_jones_gauss.epsilon = 1.5;
/// lennard_jones_gauss.sigma_squared = 0.02;
/// lennard_jones_gauss.r_0 = 3.2;
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LennardJonesGauss {
    /// Scale of Gaussian, in units of energy
    pub epsilon: f64,
    /// Width of Gaussian, sigma^2, unitless
    pub sigma_squared: f64,
    /// Gaussian center, unitless
    pub r_0: f64,
    /// unit of the length scale
    pub scale: f64,
}

impl UnivariateEnergy for LennardJonesGauss {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        let r_inv = self.scale / r;
        let arg = -((r / self.scale) - self.r_0).powi(2) / (2.0 * self.sigma_squared);
        r_inv.powi(12) - 2.0 * r_inv.powi(6) - self.epsilon * arg.exp()
    }
}

impl UnivariateForce for LennardJonesGauss {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        let r_inv = self.scale / r;
        let arg = -((r / self.scale) - self.r_0).powi(2) / (2.0 * self.sigma_squared);
        12.0 * (r_inv.powi(13) - r_inv.powi(7))
            - (self.epsilon * ((r / self.scale) - self.r_0) / self.sigma_squared) * arg.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::{assert_abs_diff_eq, assert_relative_eq};
    use rstest::*;

    #[rstest]
    fn select_points_first_set() {
        let lj_gauss: LennardJonesGauss = LennardJonesGauss {
            epsilon: 2.0,
            sigma_squared: 0.5,
            r_0: 3.0,
            scale: 1.0,
        };

        assert_eq!(lj_gauss.epsilon, 2.0);
        assert_eq!(lj_gauss.sigma_squared, 0.5);
        assert_eq!(lj_gauss.r_0, 3.0);

        // numeric tests
        assert_relative_eq!(
            lj_gauss.energy(1.5_f64),
            -0.378_674_092_892_274,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            lj_gauss.force(1.5_f64),
            -0.008_277_841_185_963,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            lj_gauss.energy(3.2_f64),
            -1.923_440_656_092_139,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            lj_gauss.force(3.2_f64),
            -0.772_120_758_370_149,
            epsilon = 1e-12
        );
    }
    #[rstest]
    fn select_points_second_set() {
        let lj_gauss: LennardJonesGauss = LennardJonesGauss {
            epsilon: 10.0,
            sigma_squared: 0.1,
            r_0: 5.0,
            scale: 1.0,
        };

        assert_eq!(lj_gauss.epsilon, 10.0);
        assert_eq!(lj_gauss.sigma_squared, 0.1);
        assert_eq!(lj_gauss.r_0, 5.0);

        // numeric tests
        assert_relative_eq!(
            lj_gauss.energy(1.5_f64),
            -0.167_875_643_768_546,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            lj_gauss.force(1.5_f64),
            -0.640_673_188_557_149,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            lj_gauss.energy(3.2_f64),
            -0.001_862_699_147_576_425,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            lj_gauss.force(3.2_f64),
            -0.003_472_622_566_788_369,
            epsilon = 1e-12
        );
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`LennardJonesGauss`]
 */

use super::{IsotropicEnergy, IsotropicForce};

/** Double-well potential with a steep repulsive core

```math
<<<<<<< HEAD
U(r) = \frac{1}{r^{12}} - \frac{2}{r^6} - \varepsilon \exp \left(-\frac{(r-r_0)^2}{2\sigma^2}\right)
=======
U(r) = 1[\mathrm{energy}]\cdot\left[ \left(\frac{1[\mathrm{length}]}{r}\right)^{12} - 2\left(\frac{1[\mathrm{length}]}{r}\right)^{6}\right] - \varepsilon \exp \left(-\frac{(r-r_0)^2}{2\sigma^2}\right)
>>>>>>> LJ-Gauss
```

Compute the Lennard-Jones-Gauss (LJG) potential and force as a function of `r`.

# Examples

```
use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce, LennardJonesGauss};
use approx::{assert_abs_diff_eq, assert_relative_eq};

let epsilon = 0.5;
let sigma_squared = 0.5;
let r_0 = 0.5_f64.powf(1.0/6.0);

let lennard_jones_gauss: LennardJonesGauss = LennardJonesGauss { epsilon, sigma_squared, r_0 };
<<<<<<< HEAD
// lennard_jones_gauss.energy(0.5_f64.powf(1.0/6.0)) is approximately -self.epsilon
=======
assert_relative_eq!(lennard_jones_gauss.energy(0.5_f64.powf(1.0/6.0)), -epsilon, epsilon=1e-12);
>>>>>>> LJ-Gauss
```

The parameters are public fields and may be accessed directly:

```
use hoomd_interaction::pairwise::{LennardJonesGauss};

<<<<<<< HEAD
let mut lennard_jones_gauss: LennardJonesGauss = LennardJonesGauss::default();
lennard_jones_gauss.epsilon = 1.5;
lennard_jones_gauss.sigma_squared = 3.0;
=======
let mut lennard_jones_gauss: LennardJonesGauss = LennardJonesGauss{
    epsilon: 1.5,
    sigma_squared: 0.02,
    r_0:  3.2
};
lennard_jones_gauss.epsilon = 1.5;
lennard_jones_gauss.sigma_squared = 0.02;
lennard_jones_gauss.r_0 = 3.2;
>>>>>>> LJ-Gauss
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LennardJonesGauss {
<<<<<<< HEAD
    /// Energy scale *(\[energy\])*.
    pub epsilon: f64,
    /// Interaction width *(\[length\])*.
    pub sigma_squared: f64,
    //// Gaussian center
    pub r_0: f64,
}

impl Default for LennardJonesGauss {
    /** Construct a [`LennardJonesGauss`] with default parameters (epsilon=1.0, sigma_squared=1.0, r_0=0.0)

    # Example

    ```
    use hoomd_interaction::pairwise::LennardJonesGauss;

    let lennard_jones_gauss: LennardJonesGauss = LennardJonesGauss::default();
    ```
    */
    #[inline]
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            sigma_squared: 1.0,
            r_0: 0.0,
        }
    }
}

=======
    /// Scale of Gaussian, in units of energy
    pub epsilon: f64,
    /// Width of Gaussian, sigma^2 is in units of length squared
    pub sigma_squared: f64,
    /// Gaussian center, in units of length
    pub r_0: f64,
}

>>>>>>> LJ-Gauss
impl IsotropicEnergy for LennardJonesGauss {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        let r_inv = r.recip();
<<<<<<< HEAD
        let arg = -1.0 * (r-self.r_0).powi(2) / (2.0*self.sigma_squared);
=======
        let arg = -(r - self.r_0).powi(2) / (2.0 * self.sigma_squared);
>>>>>>> LJ-Gauss
        r_inv.powi(12) - 2.0 * r_inv.powi(6) - self.epsilon * arg.exp()
    }
}

impl IsotropicForce for LennardJonesGauss {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        let r_inv = r.recip();
<<<<<<< HEAD
        let arg = -1.0 * (r-self.r_0).powi(2) / (2.0*self.sigma_squared);
        12.0 * (r_inv.powi(13) - r_inv.powi(7)) - (self.epsilon*(r-self.r_0)/self.sigma_squared) * arg.exp()
    }
}
=======
        let arg = -(r - self.r_0).powi(2) / (2.0 * self.sigma_squared);
        12.0 * (r_inv.powi(13) - r_inv.powi(7))
            - (self.epsilon * (self.r_0 - r) / self.sigma_squared) * arg.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::approx::{assert_abs_diff_eq, assert_relative_eq};
    use rstest::*;

    #[rstest]
    fn select_points_first_set() {
        let lj_gauss: LennardJonesGauss = LennardJonesGauss {
            epsilon: 2.0,
            sigma_squared: 0.5,
            r_0: 3.0,
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
            -1.273_068_535_928_335,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            lj_gauss.energy(3.2_f64),
            -1.923_440_656_092_139,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            lj_gauss.force(3.2_f64),
            0.765_142_344_273_568,
            epsilon = 1e-12
        );
    }
    #[rstest]
    fn select_points_second_set() {
        let lj_gauss: LennardJonesGauss = LennardJonesGauss {
            epsilon: 10.0,
            sigma_squared: 0.1,
            r_0: 5.0,
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
            -0.003_505_791_529_792_807,
            epsilon = 1e-12
        );
    }
}
>>>>>>> LJ-Gauss

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`LennardJonesGauss`]
 */

use super::{IsotropicEnergy, IsotropicForce};

/** Double-well potential with a steep repulsive core

```math
U(r) = \frac{1}{r^{12}} - \frac{2}{r^6} - \varepsilon \exp \left(-\frac{(r-r_0)^2}{2\sigma^2}\right)
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
// lennard_jones_gauss.energy(0.5_f64.powf(1.0/6.0)) is approximately -self.epsilon
```

The parameters are public fields and may be accessed directly:

```
use hoomd_interaction::pairwise::{LennardJonesGauss};

let mut lennard_jones_gauss: LennardJonesGauss = LennardJonesGauss::default();
lennard_jones_gauss.epsilon = 1.5;
lennard_jones_gauss.sigma_squared = 3.0;
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LennardJonesGauss {
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

impl IsotropicEnergy for LennardJonesGauss {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        let r_inv = r.recip();
        let arg = -1.0 * (r-self.r_0).powi(2) / (2.0*self.sigma_squared);
        r_inv.powi(12) - 2.0 * r_inv.powi(6) - self.epsilon * arg.exp()
    }
}

impl IsotropicForce for LennardJonesGauss {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        let r_inv = r.recip();
        let arg = -1.0 * (r-self.r_0).powi(2) / (2.0*self.sigma_squared);
        12.0 * (r_inv.powi(13) - r_inv.powi(7)) - (self.epsilon*(r-self.r_0)/self.sigma_squared) * arg.exp()
    }
}
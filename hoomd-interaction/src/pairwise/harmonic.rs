// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Harmonic`]
 */

use super::{IsotropicEnergy, IsotropicForce};

/**
Harmonic  potential between pair of particles.

```math
U = \frac{1}{2} k (r - r0)^2
```

Compute the harmonic potential and force as a function of `r` with
equilibrium spring length `r0`.

# Examples

```
use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce, Harmonic};
use approx::{assert_abs_diff_eq, assert_relative_eq};

let k = 2.0;
let r0 = 0.0;

let harmonic = Harmonic{ k, r0 };
assert_abs_diff_eq!(harmonic.energy(0.0), 0.0);
assert_relative_eq!(harmonic.energy(1.0), 1.0);
assert_abs_diff_eq!(harmonic.force(1.0), -2.0, epsilon=1e-12);
```

The parameters are public fields and may be accessed directly:

```
use hoomd_interaction::pairwise::Harmonic;

let mut harmonic = Harmonic{ k: 1.0, r0: 0.0};
harmonic.k = 5.0;
harmonic.r0 = 1.0;
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Harmonic {
    /** Construct a [`Harmonic`] with the given values for `k` and `r0`.

    # Examples

    ```
    use hoomd_interaction::pairwise::Harmonic;

    let harmonic = Harmonic{ k: 1.0, r0: 0.0};
    ```
    */

    /// Spring constant *(\[energy\] \[lenght\]^{-2})*.
    pub k: f64,
    /// Equilibrium spring length *(\[lenght\])*.
    pub r0: f64,
}

impl IsotropicEnergy for Harmonic {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        0.5 * self.k * (r - self.r0) * (r - self.r0)
    }
}

impl IsotropicForce for Harmonic {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        -self.k * (r - self.r0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::approx::assert_relative_eq;
    use rstest::*;

    #[rstest]
    fn zero_energy_point(#[values(1.0, 2.0, 5.0, 10.0)] k: f64, #[values(0.0, 1.0, 2.0)] r0: f64) {
        let harmonic = Harmonic { k, r0 };

        assert_eq!(harmonic.k, k);
        assert_eq!(harmonic.r0, r0);

        assert_eq!(harmonic.energy(r0), 0.0);
        assert_eq!(harmonic.force(r0), 0.0);
    }

    #[rstest]
    fn general_case(#[values(1.0, 2.0, 5.0, 10.0)] k: f64, #[values(0.0, 1.0, 2.0)] r0: f64) {
        let r = 5.0;
        let harmonic = Harmonic { k, r0 };

        assert_eq!(harmonic.k, k);
        assert_eq!(harmonic.r0, r0);

        let expected_energy = 0.5 * k * (r - r0) * (r - r0);
        let expected_force = -k * (r - r0);

        assert_relative_eq!(harmonic.energy(r), expected_energy);
        assert_relative_eq!(harmonic.force(r), expected_force);
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Shifted`]
 */

use super::{IsotropicEnergy, IsotropicForce};

/** Shift another potential to 0 at a given `r`.

```math
U(r) = f(r) - f(r_\mathrm{shift})
```

# Example

Shifted Lennard-Jones:
```
use hoomd_interaction::pairwise::{LennardJones, IsotropicEnergy, Shifted};
use approx::{assert_abs_diff_eq, assert_relative_eq};

let epsilon = 1.5;
let sigma = 1.0;
let r_shift = 2.5;
let lj: LennardJones = LennardJones { epsilon, sigma };
let shifted_lj = Shifted { f: lj, r_shift };

assert_abs_diff_eq!(shifted_lj.energy(r_shift), 0.0);
assert_relative_eq!(shifted_lj.energy(2.0_f64.powf(1.0/6.0) * sigma), -epsilon - lj.energy(r_shift));
```

Fields can be accessed directly, including those of the original potential `f`:
```
use hoomd_interaction::pairwise::{LennardJones, Shifted};

let epsilon = 1.5;
let sigma = 1.0;
let r_shift = 2.5;
let mut shifted_lj = Shifted{ f: LennardJones::<12,6> { epsilon, sigma }, r_shift };

shifted_lj.r_shift = 3.0;
shifted_lj.f.sigma = 1.2;
```
*/
#[derive(Clone, Debug, PartialEq)]
pub struct Shifted<F> {
    /// The original potential.
    pub f: F,
    /// `r` value *(\[length\])* where the shifted potential will be 0.
    pub r_shift: f64,
}

impl<F> Default for Shifted<F>
where
    F: Default,
{
    /** Construct a shifted potential with default parameters

    The defaults are:
    * `f = F::default()`
    * `r_shift = 0.0`

    # Example

    ```
    use hoomd_interaction::pairwise::{LennardJones, Shifted};

    let shifted_lj = Shifted::<LennardJones>::default();
    ```
    */
    #[inline]
    fn default() -> Self {
        Self {
            f: F::default(),
            r_shift: 0.0,
        }
    }
}

impl<F: IsotropicEnergy> IsotropicEnergy for Shifted<F> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.f.energy(r) - self.f.energy(self.r_shift)
    }
}

impl<F: IsotropicForce> IsotropicForce for Shifted<F> {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        self.f.force(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use rstest::*;

    use crate::pairwise::LennardJones;

    #[rstest]
    fn special_points_12_6(
        #[values(1.0, 2.0, 12.125, 0.25)] epsilon: f64,
        #[values(1.0, 2.0, 0.5)] sigma: f64,
    ) {
        let lj: LennardJones = LennardJones { epsilon, sigma };
        let r_shift = 2.5 * sigma;
        let shifted_lj = Shifted { f: lj, r_shift };

        assert_eq!(shifted_lj.f.epsilon, epsilon);
        assert_eq!(shifted_lj.f.sigma, sigma);
        assert_eq!(shifted_lj.r_shift, r_shift);

        let delta_e = lj.energy(r_shift);

        // Zero crossing
        assert_abs_diff_eq!(shifted_lj.energy(sigma), 0.0 - delta_e);
        assert_relative_eq!(shifted_lj.force(sigma), 24.0 * epsilon / sigma);

        // Bottom of the well
        assert_relative_eq!(
            shifted_lj.energy(2.0_f64.powf(1.0 / 6.0) * sigma),
            -epsilon - delta_e
        );
        assert_abs_diff_eq!(
            shifted_lj.force(2.0_f64.powf(1.0 / 6.0) * sigma),
            0.0,
            epsilon = 1e-12
        );

        // r = 2 sigma
        assert_relative_eq!(
            shifted_lj.energy(2.0 * sigma),
            -63.0 / 1024.0 * epsilon - delta_e
        );
        assert_relative_eq!(
            shifted_lj.force(2.0 * sigma),
            -93.0 / 512.0 * epsilon / sigma
        );
    }
}

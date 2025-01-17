// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`LennardJones`]
*/

use super::{IsotropicEnergy, IsotropicForce};

/** Potential with a steep repulsive core and an attractive well.

<!-- U(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{N} - \left( \frac{\sigma}{r} \right)^{M} \right] -->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>=</mo><mn>4</mn><mi>ε</mi><mrow><mo fence="true" form="prefix">[</mo><msup><mrow><mo fence="true" form="prefix">(</mo><mfrac><mi>σ</mi><mi>r</mi></mfrac><mo fence="true" form="postfix">)</mo></mrow><mi>N</mi></msup><mo>−</mo><msup><mrow><mo fence="true" form="prefix">(</mo><mfrac><mi>σ</mi><mi>r</mi></mfrac><mo fence="true" form="postfix">)</mo></mrow><mi>M</mi></msup><mo fence="true" form="postfix">]</mo></mrow></mrow></math>

Compute the Lennard-Jones potential and force as a function of `r`.

# Examples

In basic usage, the exponents `N` and `M` default to 12 and 6, respectively:

```
use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce, LennardJones};
use ::approx::{assert_abs_diff_eq, assert_relative_eq};

let epsilon = 1.5;
let sigma = 2.5;

let lj: LennardJones = LennardJones::new(epsilon, sigma);
assert_abs_diff_eq!(lj.energy(sigma), 0.0);
assert_relative_eq!(lj.energy(2.0_f64.powf(1.0/6.0) * sigma), -epsilon);
assert_abs_diff_eq!(lj.force(2.0_f64.powf(1.0/6.0) * sigma), 0.0, epsilon=1e-12);
```

You can choose any values for `N` and `M` _at compile time_:

```
use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce, LennardJones};
use ::approx::{assert_abs_diff_eq, assert_relative_eq};

let epsilon = 1.5;
let sigma = 2.5;

let lj: LennardJones<8,4> = LennardJones::new(epsilon, sigma);
assert_abs_diff_eq!(lj.energy(sigma), 0.0);
assert_relative_eq!(lj.energy(2.0_f64.powf(1.0/4.0) * sigma), -epsilon);
assert_abs_diff_eq!(lj.force(2.0_f64.powf(1.0/4.0) * sigma), 0.0, epsilon=1e-12);
```

The parameters are public fields and may be set directly:

```
use hoomd_interaction::pairwise::{LennardJones};

let mut lj: LennardJones = LennardJones::new(1.0, 2.5);
lj.epsilon = 1.5;
lj.sigma = 3.0;
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LennardJones<const N: i32 = 12, const M: i32 = 6> {
    /// Energy scale `[energy]`.
    pub epsilon: f64,
    /// Interaction width `[length]`.
    pub sigma: f64,
}

impl<const N: i32, const M: i32> LennardJones<N, M> {
    /** Construct a [`LennardJones`] with the given values for `epsilon` and `sigma`.

    # Examples

    The default sets `N=12` and `M=6`:
    ```
    use hoomd_interaction::pairwise::LennardJones;

    let lj: LennardJones = LennardJones::new(2.0, 3.0);
    ```

    Choose the exponents:
    ```
    use hoomd_interaction::pairwise::LennardJones;

    let lj: LennardJones<8,4> = LennardJones::new(1.0, 2.5);
    ```
    */
    #[inline]
    #[must_use]
    pub fn new(epsilon: f64, sigma: f64) -> Self {
        Self { epsilon, sigma }
    }
}

impl<const N: i32, const M: i32> IsotropicEnergy for LennardJones<N, M> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        let sigma_r = self.sigma / r;

        4.0 * self.epsilon * (sigma_r.powi(N) - sigma_r.powi(M))
    }
}

impl<const N: i32, const M: i32> IsotropicForce for LennardJones<N, M> {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        let r_inv = r.recip();
        let sigma_r = self.sigma * r_inv;

        self.epsilon
            * r_inv
            * (4.0 * f64::from(N) * sigma_r.powi(N) - 4.0 * f64::from(M) * sigma_r.powi(M))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::approx::{assert_abs_diff_eq, assert_relative_eq};
    use rstest::*;

    #[rstest]
    fn special_points_12_6(
        #[values(1.0, 2.0, 12.125, 0.25)] epsilon: f64,
        #[values(1.0, 2.0, 0.5)] sigma: f64,
    ) {
        let lj: LennardJones = LennardJones::new(epsilon, sigma);

        assert_eq!(lj.epsilon, epsilon);
        assert_eq!(lj.sigma, sigma);

        // Zero crossing
        assert_abs_diff_eq!(lj.energy(sigma), 0.0);
        assert_relative_eq!(lj.force(sigma), 24.0 * epsilon / sigma);

        // Bottom of the well
        assert_relative_eq!(lj.energy(2.0_f64.powf(1.0 / 6.0) * sigma), -epsilon);
        assert_abs_diff_eq!(
            lj.force(2.0_f64.powf(1.0 / 6.0) * sigma),
            0.0,
            epsilon = 1e-12
        );

        // r = 2 sigma
        assert_relative_eq!(lj.energy(2.0 * sigma), -63.0 / 1024.0 * epsilon);
        assert_relative_eq!(lj.force(2.0 * sigma), -93.0 / 512.0 * epsilon / sigma);
    }

    #[rstest]
    fn special_points_8_4(
        #[values(1.0, 2.0, 12.125, 0.25)] epsilon: f64,
        #[values(1.0, 2.0, 0.5)] sigma: f64,
    ) {
        let lj: LennardJones<8, 4> = LennardJones::new(epsilon, sigma);

        assert_eq!(lj.epsilon, epsilon);
        assert_eq!(lj.sigma, sigma);

        // Zero crossing
        assert_abs_diff_eq!(lj.energy(sigma), 0.0);
        assert_relative_eq!(lj.force(sigma), 16.0 * epsilon / sigma);

        // Bottom of the well
        assert_relative_eq!(lj.energy(2.0_f64.powf(1.0 / 4.0) * sigma), -epsilon);
        assert_abs_diff_eq!(
            lj.force(2.0_f64.powf(1.0 / 4.0) * sigma),
            0.0,
            epsilon = 1e-12
        );

        // r = 2 sigma
        assert_relative_eq!(lj.energy(2.0 * sigma), -15.0 / 64.0 * epsilon);
        assert_relative_eq!(lj.force(2.0 * sigma), -7.0 / 16.0 * epsilon / sigma);
    }
}

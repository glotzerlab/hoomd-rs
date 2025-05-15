// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Harmonic`]
 */

use super::{IsotropicEnergy, IsotropicForce};

/** Harmonic potential between pair of particles. Can be used to perform 
    Frenkel-Ladd free energy calculation and simulate the covalent bonds
    between particles.

<!-- U = \frac{1}{2} k (r - r0)^2 -->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>=</mo><mn>4</mn><mi>ε</mi><mrow><mo fence="true" form="prefix">[</mo><msup><mrow><mo fence="true" form="prefix">(</mo><mfrac><mi>σ</mi><mi>r</mi></mfrac><mo fence="true" form="postfix">)</mo></mrow><mi>N</mi></msup><mo>−</mo><msup><mrow><mo fence="true" form="prefix">(</mo><mfrac><mi>σ</mi><mi>r</mi></mfrac><mo fence="true" form="postfix">)</mo></mrow><mi>M</mi></msup><mo fence="true" form="postfix">]</mo></mrow></mrow></math>

Compute the harmonic potential and force as a function of `r` with 
equilibrium spring length `r0`.

# Examples

In basic usage, the `r0` defaults to 0, respectively:

```
use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce, Harmonic};
use approx::{assert_abs_diff_eq, assert_relative_eq};

let k = 2.0;

let harmonic = Harmonic::new(k, None);
assert_abs_diff_eq!(harmonic.energy(0.0), 0.0);
assert_relative_eq!(harmonic.energy(1.0), 1.0);
assert_abs_diff_eq!(harmonic.force(1.0), -2.0, epsilon=1e-12);
```

The parameters are public fields and may be accessed directly:

```
use hoomd_interaction::pairwise::Harmonic;

let mut harmonic = Harmonic::new(2.0, None);
harmonic.k = 5.0;
harmonic.r0 = 1.0;
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Harmonic {
    /// Spring constant (`[energy] [length]^(-2)`).
    pub k: f64,
    /// Equilibrium spring length (`[length]`).
    pub r0: f64,
}

impl Harmonic {
    /** Construct a [`Harmonic`] with the given values for `k` and `r0`.

    # Examples

    The default sets `r0=0`:
    ```
    use hoomd_interaction::pairwise::Harmonic;

    let harmonic = Harmonic::new(2.0, None);
    ```
    */
    #[inline]
    #[must_use]
    pub fn new(k: f64, r0: Option<f64>) -> Self {
        Harmonic {
            k: k,
            r0: r0.unwrap_or(0.0), // Use 0.0 if r0 is None
        }
    }
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
    use ::approx::{assert_abs_diff_eq, assert_relative_eq};
    use rstest::*;

    #[rstest]
    fn user_defined_r0(
        #[values(1.0, 2.0, 5.0, 10.0)] k: f64,
        #[values(0.0, 1.0, 2.0)] r0: f64,
    ) {
        let r = 5.0;
        let harmonic = Harmonic::new(k, Some(r0));

        assert_eq!(harmonic.k, k);
        assert_eq!(harmonic.r0, r0);

        let expected_energy = 0.5 * k * (r - r0) * (r - r0);
        let expected_force = -k * (r - r0);

        assert_abs_diff_eq!(harmonic.energy(r), expected_energy);
        assert_relative_eq!(harmonic.force(r), expected_force);
    }
}

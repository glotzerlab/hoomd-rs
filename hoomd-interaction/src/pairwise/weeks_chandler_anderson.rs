// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`WeeksChandlerAnderson`]
 */

use super::{IsotropicEnergy, IsotropicForce, LennardJones};

/** Potential with a steep repulsive core.

```math
U(r) = \begin{cases}
4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6} \right] + \varepsilon & r \lt 2^{1/6} \sigma \\

0 & r \ge 2^{1/6} \sigma
\end{cases}
```

Compute the Weeks-Chandler-Anderson (WCA) potential and force as a function of `r`.

# Examples

Basic usage:

```
use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce, WeeksChandlerAnderson};
use approx::{assert_abs_diff_eq, assert_relative_eq};

let epsilon = 1.5;
let sigma = 2.5;

let wca = WeeksChandlerAnderson { epsilon, sigma };
assert_relative_eq!(wca.energy(sigma), epsilon);
assert_abs_diff_eq!(wca.energy(2.0*sigma), 0.0);
assert_relative_eq!(wca.energy(2.0_f64.powf(1.0/6.0) * sigma), 0.0);
assert_abs_diff_eq!(wca.force(2.0_f64.powf(1.0/6.0) * sigma), 0.0, epsilon=1e-12);
```

The parameters are public fields and may be accessed directly:

```
use hoomd_interaction::pairwise::WeeksChandlerAnderson;

let mut wca = WeeksChandlerAnderson::default();
wca.epsilon = 1.5;
wca.sigma = 3.0;
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeeksChandlerAnderson {
    /// Energy scale *(\[energy\])*.
    pub epsilon: f64,
    /// Interaction width *(\[length\])*.
    pub sigma: f64,
}

impl Default for WeeksChandlerAnderson {
    /** Construct a [`WeeksChandlerAnderson`] with default parameters (epsilon=1.0, sigma=1.0)

    # Example

    ```
    use hoomd_interaction::pairwise::WeeksChandlerAnderson;

    let wca = WeeksChandlerAnderson::default();
    ```
    */
    #[inline]
    fn default() -> Self {
        WeeksChandlerAnderson {
            epsilon: 1.0,
            sigma: 1.0,
        }
    }
}

impl IsotropicEnergy for WeeksChandlerAnderson {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        if r < 2.0_f64.powf(1.0 / 6.0) * self.sigma {
            let lj = LennardJones::<12, 6> {
                epsilon: self.epsilon,
                sigma: self.sigma,
            };
            lj.energy(r) + self.epsilon
        } else {
            0.0
        }
    }
}

impl IsotropicForce for WeeksChandlerAnderson {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        if r < 2.0_f64.powf(1.0 / 6.0) * self.sigma {
            let lj = LennardJones::<12, 6> {
                epsilon: self.epsilon,
                sigma: self.sigma,
            };
            lj.force(r)
        } else {
            0.0
        }
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
        let wca = WeeksChandlerAnderson { epsilon, sigma };

        assert_eq!(wca.epsilon, epsilon);
        assert_eq!(wca.sigma, sigma);

        // Zero crossing (shifted by epsilon)
        assert_relative_eq!(wca.energy(sigma), epsilon);
        assert_relative_eq!(wca.force(sigma), 24.0 * epsilon / sigma);

        // Bottom of the well
        assert_abs_diff_eq!(wca.energy(2.0_f64.powf(1.0 / 6.0) * sigma), 0.0);
        assert_abs_diff_eq!(
            wca.force(2.0_f64.powf(1.0 / 6.0) * sigma),
            0.0,
            epsilon = 1e-12
        );

        // r = 2 sigma
        assert_eq!(wca.energy(2.0 * sigma), 0.0);
        assert_eq!(wca.force(2.0 * sigma), 0.0);
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`HarmonicRepulsion`]
 */

use super::{IsotropicEnergy, IsotropicForce};

/** Repulsive half of a quadratic potential well.

[`HarmonicRepulsion`] is the conservative part of the dissipative particle
dynamics soft potential. The parameter `a` controls the strength of the
potential and `r_cut` sets the distance at which the energy goes to 0.


The potential produce the radially repulsive force between particles as:
```math
F(r) = \begin{cases}
-\frac{A}{r_\mathrm{cut}} \left( r - r_\mathrm{cut}\right) & r < r_\mathrm{cut} \\

0 & r \ge r_\mathrm{cut}
\end{cases}
```

resulting from the potential energy:
```math
U(r) = \begin{cases}
\frac{1}{2} \frac{A}{r_\mathrm{cut}} \left(r - r_\mathrm{cut}\right)^2 & r \lt r_\mathrm{cut} \\

0 & r \ge r_\mathrm{cut}
\end{cases}
```

This potential forms the left half of a harmonic well centered at
$`r = r_\mathrm{cut}`$ with a spring constant $`k = \frac{A}{r_\mathrm{cut}}`$.

# Examples


```
use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce, HarmonicRepulsion};
use approx::{assert_abs_diff_eq, assert_relative_eq};

let a = 1.0;
let r_cut = 1.0;

let h_repsulsion = HarmonicRepulsion{ a, r_cut };
assert_abs_diff_eq!(h_repsulsion.energy(1.5), 0.0);
assert_relative_eq!(h_repsulsion.energy(0.5), 0.125);
assert_abs_diff_eq!(h_repsulsion.force(0.5), 0.5, epsilon=1e-12);
```

The parameters are public fields and may be accessed directly:

```
use hoomd_interaction::pairwise::HarmonicRepulsion;

let mut h_repulsion = HarmonicRepulsion{ a: 1.0, r_cut: 1.0};
h_repulsion.a = 5.0;
h_repulsion.r_cut = 0.75;
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarmonicRepulsion {
    /// Potential strength $`[\mathrm{energy}] [\mathrm{length}]^{-1}`$.
    pub a: f64,
    /// Distance cut-off  $`[\mathrm{length}]`$.
    pub r_cut: f64,
}

impl IsotropicEnergy for HarmonicRepulsion {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        if r < self.r_cut {
            0.5 * self.a / self.r_cut * (r - self.r_cut).powi(2)
        } else {
            0.0
        }
    }
}

impl IsotropicForce for HarmonicRepulsion {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        if r < self.r_cut {
            self.a / self.r_cut * (self.r_cut - r)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::approx::assert_relative_eq;
    use rstest::*;

    #[rstest]
    fn outside_rcut(#[values(1.0, 2.0, 5.0, 10.0)] a: f64, #[values(1.0, 2.0, 3.0)] r_cut: f64) {
        let h_repulsion = HarmonicRepulsion { a, r_cut };

        assert_eq!(h_repulsion.a, a);
        assert_eq!(h_repulsion.r_cut, r_cut);

        assert_eq!(h_repulsion.energy(r_cut + 1.0), 0.0);
        assert_eq!(h_repulsion.force(r_cut + 1.0), 0.0);
    }

    #[rstest]
    fn general_case(#[values(1.0, 2.0, 5.0, 10.0)] a: f64, #[values(1.0, 2.0, 3.0)] r_cut: f64) {
        let r = 0.5;
        let h_repulsion = HarmonicRepulsion { a, r_cut };

        assert_eq!(h_repulsion.a, a);
        assert_eq!(h_repulsion.r_cut, r_cut);

        let expected_energy = a * (r_cut - r) - 0.5 * a / r_cut * (r_cut * r_cut - r * r);
        let expected_force = a * (1.0 - r / r_cut);

        assert_relative_eq!(h_repulsion.energy(r), expected_energy);
        assert_relative_eq!(h_repulsion.force(r), expected_force);
    }
}

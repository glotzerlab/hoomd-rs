// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`ChimesPenalty`]
 */

use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce};

/**
Implement the penalty potential $`f_\mathrm{p}`$ of `ChIMES`
potential, which must be used to prevent inter-particle
distances fall below inner cut-off.

The potential is defined as:

```math
f_p(r) =
\begin{cases}
&A_\mathrm{p} (r_\mathrm{in} + d_\mathrm{p} - r)^3 \text{, if } r < r_\mathrm{in} + d_\mathrm{p} \\
&0 \text{, otherwise} \\
\end{cases}
```

Where $`r_\mathrm{in}`$ is the inner distance cutoff, same as that
defined in [`Chimes2b`], $`A_\mathrm{p}`$ is the penalty strength
, and $`d_\mathrm{p}`$ is a small distance to smooth activation of
penalty potential.

# Note:
See equation 9 in <https://doi.org/10.1038/s41524-024-01497-y>.

# Example:
```
use hoomd_chimes::transformation::MorseTransformation;
use hoomd_chimes::potential::{Chimes2b, TersoffSmooth, ChimesPenalty};
use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce};

// Main body of chimes potential
let lambda = 1.5;
let r_out = 3.0;
let r_in = 1.0;
let fo = 0.75;
let coeff_2b = vec![1.0, 2.0, 3.0];

let morse_trans: MorseTransformation = MorseTransformation {
    lambda,
    r_out,
    r_in,
};

let chimes2b_cheby: Chimes2b<MorseTransformation> =
    Chimes2b::new(morse_trans, coeff_2b, r_in);

let chimes2b = TersoffSmooth {
    f: chimes2b_cheby,
    r_out,
    r_in,
    fo,
};

// ChIMES penalty. Parameters are obtain from <https://doi.org/10.1038/s41524-024-01497-y>.
let a = 1e+6;
let dt = 0.02;

let chimes_penalty = ChimesPenalty{r_in, a, dt};

let r = 1.5;
let chimes_energy = chimes_penalty.energy(r) + chimes2b.energy(r);
```
*/
#[derive(Clone, Debug, PartialEq)]
pub struct ChimesPenalty {
    /// Inner radial cut-off (`[length]`).
    pub r_in: f64,
    /// Penalty strength (`[energy]`)
    pub a: f64,
    /// Smooth kick-in distance (`[length]`).
    pub dt: f64,
}

impl IsotropicEnergy for ChimesPenalty {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        let r_penalty = self.r_in + self.dt - r;

        if r_penalty <= 0.0 {
            0.0
        } else {
            let e_penalty = self.a * r_penalty * r_penalty * r_penalty;
            /// println!(
            ///    "HOOMD Warning: Adding penalty in 2B Cheby calc, r < r_in+dt {:6} < {:6}",
            ///    r,
            ///    self.r_in + self.dt
            /// );
            /// println!("HOOMD Warning: Penalty potential = {:6}", e_penalty);
            e_penalty
        }
    }
}

impl IsotropicForce for ChimesPenalty {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        let r_penalty = self.r_in + self.dt - r;

        if r_penalty <= 0.0 {
            0.0
        } else {
            3.0 * self.a * r_penalty * r_penalty
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    use crate::potential::ChimesPenalty;

    #[rstest]
    fn beyond_inner_cutoff(
        #[values(2.0, 3.0, 4.0)] r_in: f64,
        #[values(1e+4, 1e+5, 1e+6)] a: f64,
        #[values(0.01, 0.02, 0.03)] dt: f64,
    ) {
        let r = r_in + dt;

        let chimes_penalty = ChimesPenalty { r_in, a, dt };

        assert_eq!(chimes_penalty.energy(r), 0.0);
        assert_eq!(chimes_penalty.force(r), 0.0);
    }

    #[rstest]
    fn general_cases(
        #[values(2.0, 3.0, 4.0)] r_in: f64,
        #[values(1e+4, 1e+5, 1e+6)] a: f64,
        #[values(0.01, 0.02, 0.03)] dt: f64,
    ) {
        let r = 1.0;

        let chimes_penalty = ChimesPenalty { r_in, a, dt };

        let r_penalty = r_in + dt - r;
        let expect_energy = a * r_penalty * r_penalty * r_penalty;
        let expect_force = 3.0 * a * r_penalty * r_penalty;

        assert_eq!(chimes_penalty.energy(r), expect_energy);
        assert_eq!(chimes_penalty.force(r), expect_force);
    }
}

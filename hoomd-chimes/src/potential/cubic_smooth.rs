// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`CubicSmooth`]
 */

use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce};

/**
Implement the cubic style smoothing $`f_s`$ of `ChIMES`
potential, for one plus two-body case:

```math
U(r) = f_s(r) \sum^{n}_{O=1} c_{O} T_{O}(s(r))
```
where:

```math
f_s(r) = (1 - \frac{r}{r_\mathrm{out}})^3
```

Where `r_out` is the outer distance cutoff.

# Note:
See equation 7 in <https://doi.org/10.1038/s41524-024-01497-y>.

# Example:
```
use arrayvec::ArrayVec;
use hoomd_chimes::transformation::MorseTransformation;
use hoomd_chimes::potential::{Chimes2b, CubicSmooth};
use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce};

let lambda = 1.5;
let r_out = 3.0;
let r_in = 1.0;
let fo = 0.75;
let coeff = vec![1.0, 2.0, 3.0];

let morse_trans: MorseTransformation = MorseTransformation {
    lambda,
    r_out,
    r_in,
};

let chimes2b_cheby: Chimes2b<MorseTransformation, 3> =
    Chimes2b::new(morse_trans, coeff, r_in);

let chimes2b = CubicSmooth {
    f: chimes2b_cheby,
    r_out
};
```
*/
#[derive(Clone, Debug, PartialEq)]
pub struct CubicSmooth<F> {
    /// The [`Chimes2b`] fucntion.
    pub f: F,
    /// Outer radial cut-off (`[length]`).
    pub r_out: f64,
}

impl<F> CubicSmooth<F> {
    /// The cubic smoothing function
    #[inline]
    fn fs(&self, r: f64) -> f64 {
        (1.0 - r / self.r_out).powi(3)
    }

    /// Partial derivative of the cubic smoothing function
    /// with respect to r
    #[inline]
    fn dfs_dr(&self, r: f64) -> f64 {
        (-3.0 / self.r_out) * (1.0 - r / self.r_out).powi(2)
    }
}

impl<F: IsotropicEnergy> IsotropicEnergy for CubicSmooth<F> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.fs(r) * self.f.energy(r)
    }
}

impl<F: IsotropicForce + IsotropicEnergy> IsotropicForce for CubicSmooth<F> {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        // Chain rule of -d/dr(fs(r)*U(r))
        self.fs(r) * self.f.force(r) - self.dfs_dr(r) * self.f.energy(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformation::MorseTransformation;
    use ::approx::assert_abs_diff_eq;
    use arrayvec::ArrayVec;
    use hoomd_interaction::pairwise::LennardJones;
    use rstest::*;

    use crate::potential::Chimes2b;

    #[rstest]
    fn test_construction() {
        let lambda = 1.5;
        let r_out = 3.0;
        let r_in = 1.0;
        let coeff = vec![1.0, 2.0, 3.0];

        let morse_trans: MorseTransformation = MorseTransformation {
            lambda,
            r_out,
            r_in,
        };

        let chimes2b_cheby: Chimes2b<MorseTransformation, 3> =
            Chimes2b::new(morse_trans, coeff, r_in);

        let chimes2b = CubicSmooth {
            f: chimes2b_cheby,
            r_out,
        };

        // cubic smoothing
        assert_eq!(chimes2b.r_out, r_out);
        // chimes 2b main function
        assert_eq!(chimes2b.f.coeff().as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(chimes2b.f.r_in(), &r_in);
        // transformation
        assert_eq!(chimes2b.f.trans_style().lambda, lambda);
        assert_eq!(chimes2b.f.trans_style().r_out, r_out);
        assert_eq!(chimes2b.f.trans_style().r_in, r_in);
    }

    #[rstest]
    fn fit_to_lennard_jones(#[values(0.9, 1.0, 1.5, 2.0, 2.5, 3.0)] r_test: f64) {
        const NCOEFF: usize = 30;
        let r_out = 3.0;
        let r_in = 0.9;
        let morse_lambda: f64 = 2.0_f64.powf(1.0 / 6.0);
        let coeff_2b = vec![
            4.34880025e+00,
            6.74743542e+00,
            -6.00543791e-01,
            1.0612591e+01,
            -1.06837188e+01,
            1.58120461e+01,
            -1.85113547e+01,
            2.1308595e+01,
            -2.33995804e+01,
            2.46871116e+01,
            -2.5260915e+01,
            2.51692842e+01,
            -2.42385164e+01,
            2.3015699e+01,
            -2.09620467e+01,
            1.90261118e+01,
            -1.63844403e+01,
            1.42058462e+01,
            -1.1510464e+01,
            9.49511096e+00,
            -7.16349469e+00,
            5.57621000e+00,
            -3.84320086e+00,
            2.78202787e+00,
            -1.69065245e+00,
            1.1084381e+00,
            -5.50974194e-01,
            3.10395085e-01,
            -1.01646248e-01,
            4.31317586e-02,
        ];

        let morse_trans = MorseTransformation {
            lambda: morse_lambda,
            r_out: r_out,
            r_in: r_in,
        };

        let chimes2b_cheby: Chimes2b<MorseTransformation, NCOEFF> =
            Chimes2b::new(morse_trans, coeff_2b, r_in);

        let chimes2b = CubicSmooth {
            f: chimes2b_cheby,
            r_out,
        };

        let lj: LennardJones<12, 6> = LennardJones {
            epsilon: 1.0,
            sigma: 1.0,
        };

        assert_abs_diff_eq!(chimes2b.energy(r_test), lj.energy(r_test), epsilon = 6e-3);
    }
}

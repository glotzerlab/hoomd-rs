// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`ChIMES_2B`]
 */

use super::{IsotropicEnergy, IsotropicForce};
use hoomd_utility::cheby::Chebyshev;
use hoomd_utility::chimes_transformation::Transformation;

/**
Implement the one- plus two-body
part of `ChIMES` potential.
 */
#[derive(Clone, Debug, PartialEq)]
pub struct Chimes2b<F: Transformation, T = Chebyshev> {
    /// Transformation style.
    pub trans_style: F,
    /// one plus two-body `ChIMES` coefficient (`[energy]`).
    pub coeff: Vec<f64>,
    /// Inner radial cut-off (`[length]`).
    pub r_in: f64,
    /// Buffer distance before triggering the damping.
    inner_smooth_r: f64,
    /// Chebyshev polynomials.
    cheby: T,
}

impl<F: Transformation> Chimes2b<F, Chebyshev> {
    /** Construct a [`ChIMES_2B_Cheby`] with the given values for `epsilon` and `sigma`.

    # Example

    ```
    use hoomd_interaction::pairwise::Chimes2b;
    use hoomd_utility::chimes_transformation::MorseTransformation;

    let lambda = 1.5;
    let r_out = 3.0;
    let r_in = 1.0;
    let coeff = vec![1.0, 2.0];
    let morse_trans: MorseTransformation = MorseTransformation{lambda, r_out, r_in};

    let chimes2b = Chimes2b::new(morse_trans, coeff, r_in);
    ```
    */
    #[inline]
    #[must_use]
    pub fn new(trans_style: F, coeff: Vec<f64>, r_in: f64) -> Self {
        let n = coeff.len(); // Store length before moving
        Self {
            trans_style,
            coeff,
            r_in,
            inner_smooth_r: 0.01,
            cheby: Chebyshev { n },
        }
    }
}

impl<F: Transformation> IsotropicEnergy for Chimes2b<F, Chebyshev> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        let mut value: f64 = 0.0;

        let s = self.trans_style.s(&r);
        let tn = self.cheby.eval_cheby(&s);

        if r > self.r_in {
            for (idx, c) in self.coeff.iter().enumerate() {
                value += c * tn[idx];
            }
        } else {
            let tnd = self.cheby.eval_dcheby_ds(&s);
            let damp_fac = ((r - self.r_in) / self.inner_smooth_r).exp();

            for (idx, c) in self.coeff.iter().enumerate() {
                value += c * (tn[idx] + self.inner_smooth_r * (damp_fac - 1.0) * tnd[idx]);
            }
        }
        value
    }
}

impl<F: Transformation> IsotropicForce for Chimes2b<F, Chebyshev> {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        let mut value: f64 = 0.0;

        let s = self.trans_style.s(&r);
        let ds_dr = self.trans_style.ds_dr(&r);
        let tnd = self.cheby.eval_dcheby_ds(&s);

        if r > self.r_in {
            for (idx, c) in self.coeff.iter().enumerate() {
                value -= c * tnd[idx] * ds_dr;
            }
        } else {
            let damp_fac = ((r - self.r_in) / self.inner_smooth_r).exp();

            for (idx, c) in self.coeff.iter().enumerate() {
                value -= c * tnd[idx] * ds_dr * damp_fac;
            }
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_utility::chimes_transformation::MorseTransformation;

    #[test]
    fn test_chimes2b_new() {
        let lambda = 1.5;
        let r_out = 3.0;
        let r_in = 1.0;
        let morse_trans: MorseTransformation = MorseTransformation {
            lambda,
            r_out,
            r_in,
        };

        let chimes2b = Chimes2b::new(morse_trans, vec![1.0, 2.0], r_in);
        assert_eq!(chimes2b.coeff, vec![1.0, 2.0]);
        assert_eq!(chimes2b.r_in, 1.0);
    }
}

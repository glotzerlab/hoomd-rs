// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Chimes2b`]
 */

use super::{IsotropicEnergy, IsotropicForce};
use hoomd_utility::cheby::Chebyshev;
use hoomd_utility::chimes_transformation::Transformation;

/**
Implement the main body of one- plus two-body
part of `ChIMES` potential. It performs the
sum of product between `ChIMES` coefficient
and the corresponding Chebyshev polynomials as:

```math
U = \sum^{n}_{i=1} c_{i} T_{i}(s(r))
```

Where $`c_i`$ is the `ChIMES` coefficent, $`T_i`$ is the
Chebyshev polynomials, and $`s`$ is the transformed
distance between particles, given by [`Transformation`].

# Note:
* See equation 2 in <https://doi.org/10.1038/s41524-024-01497-y>.
* Must be used with the [`TersoffSmooth`] and [`ChimesPenalty`]
to enable correct potential calculation.
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
    /** Construct a [`Chimes2b`] with the given a transformation
    fucntion `trans_style`, defined in [`Transformation`], `ChIMES`
    one- plus two-body coefficient `coeff`, and the inner
    distance cutoff `r_in`.

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
        let n = coeff.len() + 1; // Store length before moving
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
        let tn = &self.cheby.eval_cheby(&s)[1..];

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
        let tnd = &self.cheby.eval_dcheby_ds(&s)[1..];

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
    use rstest::*;

    #[rstest]
    fn test_chimes2b_new() {
        let lambda = 1.5;
        let r_out = 3.0;
        let r_in = 1.0;
        let morse_trans: MorseTransformation = MorseTransformation {
            lambda,
            r_out,
            r_in,
        };
        let coeff_2b = vec![1.0, 2.0, 3.0];

        let chimes2b = Chimes2b::new(morse_trans, coeff_2b, r_in);
        assert_eq!(chimes2b.coeff, vec![1.0, 2.0, 3.0]);
        assert_eq!(chimes2b.r_in, 1.0);
    }

    #[rstest]
    fn special_points() {
        // Test the `ChIMES` potential (without smoothing) at the
        // special points when $`r=r_{in}`$ or $`r_{out}`$.
        //
        // Here, I use the maximum order of 3, resulting in the
        // expression of potential energy as:
        //
        // ```math
        // U = c_1 * s
        //     + c_2 * (2*s^2 - 1)
        //     + c_3 * (r*s^3 - 3s)
        // ```
        //
        // At $`r=r_{in}`$ or $`r_{out}`$, the corresponding
        // Morse transformed distance s is $`s=f(r_in)=1.0`$
        // or $`s=f(r_out)=-1.0`$. Finally, the potential
        // energy at these two special point is:
        //
        // ```math
        // \begin{align*}
        // U(r_in) &= c_1 + c_2 + c_3 \\
        // U(r_out) &= -c_1 + c_2 - c_3
        // \end{align*}
        let lambda = 1.5;
        let r_out = 3.0;
        let r_in = 1.0;
        let morse_trans: MorseTransformation = MorseTransformation {
            lambda,
            r_out,
            r_in,
        };
        let coeff = vec![1.0, 2.0, 3.0];

        let chimes2b = Chimes2b::new(morse_trans, coeff, r_in);
        assert_eq!(chimes2b.energy(r_in), 1.0 + 2.0 + 3.0);
        assert_eq!(chimes2b.energy(r_out), -1.0 + 2.0 - 3.0);
        println!("{:.64}", chimes2b.force(r_in - 0.2));
    }
}

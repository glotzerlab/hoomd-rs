// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`TersoffSmooth`]

use hoomd_interaction::univariate::{UnivariateEnergy, UnivariateForce};
use std::f64::consts::PI; // Note: Becky's chimes uses hard coding value: 3.14159265359

/// Implement the Tersoff style smoothing $`f_s`$ of `ChIMES`
/// potential, for one plus two-body case:
///
/// ```math
/// U(r) = f_s(r) \sum^{n}_{O=1} c_{O} T_{O}(s(r))
/// ```
/// where:
///
/// ```math
/// f_s(r) =
/// \begin{cases}
/// 0 &\text{if } r > r_{\mathrm{out}} \\
/// 1 &\text{if } r < r_{\mathrm{in}} \\
/// \frac{1}{2} +  \frac{1}{2}
/// \sin{\left( \pi \left[ \frac{r-d_t}{r_\mathrm{out} - d_t}\right]
/// + \frac{\pi}{2}\right)} &\text{, otherwise}\\
/// \end{cases}
/// ```
///
/// ```math
/// d_t = r_{\mathrm{out}} * (1.0 - f_o)
/// ```
///
/// Where `r_in` is the inner distance cutoff, same as that
/// defined in [`ChimesChebyshevExpansion`], `r_out` is the outer distance cutoff
/// , and `f_o` is the parameter with the value between 0 and 1
/// to control the activation of smoothing.
///
/// # Note:
/// See equation 8 in <https://doi.org/10.1038/s41524-024-01497-y>.
///
/// # Example:
/// ```
/// use hoomd_chimes::{
///     potential::{ChimesChebyshevExpansion, TersoffSmooth},
///     transformation::MorseTransformation,
/// };
/// use hoomd_interaction::univariate::{UnivariateEnergy, UnivariateForce};
///
/// let lambda = 1.5;
/// let r_out = 3.0;
/// let r_in = 1.0;
/// let fo = 0.75;
/// let coeff = vec![1.0, 2.0, 3.0];
///
/// let morse_trans: MorseTransformation = MorseTransformation {
///     lambda,
///     r_out,
///     r_in,
/// };
///
/// let chimes2b_cheby: ChimesChebyshevExpansion<MorseTransformation, 3> =
///     ChimesChebyshevExpansion::new(morse_trans, coeff, r_in);
///
/// let chimes2b = TersoffSmooth {
///     f: chimes2b_cheby,
///     r_out,
///     r_in,
///     fo,
/// };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct TersoffSmooth<F> {
    /// The [`ChimesChebyshevExpansion`] fucntion.
    pub f: F,
    /// Outer radial cut-off (`[length]`).
    pub r_out: f64,
    /// Inner radial cut-off (`[length]`).
    pub r_in: f64,
    /// Parameter to control the activation of smoothing.
    pub fo: f64,
}

impl<F> TersoffSmooth<F> {
    /// The tersoff smoothing function
    #[inline]
    fn fs(&self, r: f64) -> f64 {
        let dt = self.r_out * (1.0 - self.fo);

        if r < dt {
            1.0
        } else if r > self.r_out {
            0.0
        } else {
            0.5 + 0.5 * (PI * (r - dt) / (self.r_out - dt) + 0.5 * PI).sin()
        }
    }

    /// Partial derivative of the tersoff smoothing function
    /// with respect to r
    #[inline]
    fn dfs_dr(&self, r: f64) -> f64 {
        let dt = self.r_out * (1.0 - self.fo);

        if r < dt || r > self.r_out {
            0.0
        } else {
            let pref = PI / (self.r_out - dt);
            0.5 * pref * (pref * (r - dt) + 0.5 * PI).cos()
        }
    }
}

impl<F: UnivariateEnergy> UnivariateEnergy for TersoffSmooth<F> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.fs(r) * self.f.energy(r)
    }
}

impl<F: UnivariateForce + UnivariateEnergy> UnivariateForce for TersoffSmooth<F> {
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
    use approxim::assert_abs_diff_eq;
    use hoomd_interaction::univariate::LennardJones;
    use rstest::*;

    use crate::potential::ChimesChebyshevExpansion;

    #[rstest]
    fn test_construction() {
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

        let chimes2b_cheby: ChimesChebyshevExpansion<MorseTransformation, 3> =
            ChimesChebyshevExpansion::new(morse_trans, coeff, r_in);

        let chimes2b = TersoffSmooth {
            f: chimes2b_cheby,
            r_out,
            r_in,
            fo,
        };

        // tersoff smoothing
        assert_eq!(chimes2b.r_out, r_out);
        assert_eq!(chimes2b.r_in, r_in);
        assert_eq!(chimes2b.fo, fo);
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
        let fo = 0.5;
        let coeff_2b = vec![
            2.370_076_83e+00,
            5.581_525_19e-01,
            2.393_305_27e+00,
            2.776_457_04e-02,
            1.419_708_95e+00,
            -5.755_377_59e-01,
            8.808_602_84e-01,
            -6.646_239_98e-01,
            6.643_554_78e-01,
            -5.802_035_08e-01,
            5.249_903_60e-01,
            -4.691_661_36e-01,
            4.033_253_16e-01,
            -3.641_750_18e-01,
            2.964_170_62e-01,
            -2.708_412_28e-01,
            2.061_733_97e-01,
            -1.919_486_54e-01,
            1.345_077_73e-01,
            -1.279_830_94e-01,
            8.089_841_28e-02,
            -7.903_035_36e-02,
            4.387_415_21e-02,
            -4.390_832_27e-02,
            2.051_670_15e-02,
            -2.101_972_58e-02,
            7.637_392_23e-03,
            -7.898_353_06e-03,
            1.769_846_74e-03,
            -1.880_151_13e-03,
        ];

        let morse_trans = MorseTransformation {
            lambda: morse_lambda,
            r_out,
            r_in,
        };

        let chimes2b_cheby: ChimesChebyshevExpansion<MorseTransformation, NCOEFF> =
            ChimesChebyshevExpansion::new(morse_trans, coeff_2b, r_in);

        let chimes2b = TersoffSmooth {
            f: chimes2b_cheby,
            r_out,
            r_in,
            fo,
        };

        let lj: LennardJones<12, 6> = LennardJones {
            epsilon: 1.0,
            sigma: 1.0,
        };

        assert_abs_diff_eq!(chimes2b.energy(r_test), lj.energy(r_test), epsilon = 6e-3);
    }
}

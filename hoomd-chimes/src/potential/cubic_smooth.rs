// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`CubicSmooth`]

use hoomd_interaction::univariate::{UnivariateEnergy, UnivariateForce};

/// Implement the cubic style smoothing $`f_s`$ of `ChIMES`
/// potential, for one plus two-body case:
///
/// ```math
/// U(r) = f_s(r) \sum^{n}_{O=1} c_{O} T_{O}(s(r))
/// ```
/// where:
///
/// ```math
/// f_s(r) = (1 - \frac{r}{r_\mathrm{out}})^3
/// ```
///
/// Where `r_out` is the outer distance cutoff.
///
/// # Note:
/// See equation 7 in <https://doi.org/10.1038/s41524-024-01497-y>.
///
/// # Example:
/// ```
/// use hoomd_chimes::{
///     potential::{ChimesChebyshevExpansion, CubicSmooth},
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
/// let chimes2b = CubicSmooth {
///     f: chimes2b_cheby,
///     r_out,
/// };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct CubicSmooth<F> {
    /// The [`ChimesChebyshevExpansion`] fucntion.
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

impl<F: UnivariateEnergy> UnivariateEnergy for CubicSmooth<F> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.fs(r) * self.f.energy(r)
    }
}

impl<F: UnivariateForce + UnivariateEnergy> UnivariateForce for CubicSmooth<F> {
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
        let coeff = vec![1.0, 2.0, 3.0];

        let morse_trans: MorseTransformation = MorseTransformation {
            lambda,
            r_out,
            r_in,
        };

        let chimes2b_cheby: ChimesChebyshevExpansion<MorseTransformation, 3> =
            ChimesChebyshevExpansion::new(morse_trans, coeff, r_in);

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
            4.348_800_25e+00,
            6.747_435_42e+00,
            -6.005_437_91e-01,
            1.061_259_1e+01,
            -1.068_371_88e+01,
            1.581_204_61e+01,
            -1.851_135_47e+01,
            2.130_859_5e+01,
            -2.339_958_04e+01,
            2.468_711_16e+01,
            -2.526_091_5e+01,
            2.516_928_42e+01,
            -2.423_851_64e+01,
            2.301_569_9e+01,
            -2.096_204_67e+01,
            1.902_611_18e+01,
            -1.638_444_03e+01,
            1.420_584_62e+01,
            -1.151_046_4e+01,
            9.495_110_96e+00,
            -7.163_494_69e+00,
            5.576_210_00e+00,
            -3.843_200_86e+00,
            2.782_027_87e+00,
            -1.690_652_45e+00,
            1.108_438_1e+00,
            -5.509_741_94e-01,
            3.103_950_85e-01,
            -1.016_462_48e-01,
            4.313_175_86e-02,
        ];

        let morse_trans = MorseTransformation {
            lambda: morse_lambda,
            r_out: r_out,
            r_in: r_in,
        };

        let chimes2b_cheby: ChimesChebyshevExpansion<MorseTransformation, NCOEFF> =
            ChimesChebyshevExpansion::new(morse_trans, coeff_2b, r_in);

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

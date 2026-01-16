// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`MorseTransformation`]

use super::Transformation;
/// Calculate the morse transformation style.
///
/// Given a distance $`r`$, the morse transformation are defined by:
/// morse decaying factor $`\lambda`$, outer radial cutoff
/// $`r_\mathrm{out}`$ , and inner radial cutoff $`r_\mathrm{in}`$.
///
/// It calculates a coordinate $`s(r)`$ falls between [-1, 1]
/// and its derivative $`\frac{ds(r)}{dr}`$ with respect to $`r`$.
///
/// See equation 3 to 5 in
/// <https://doi.org/10.1038/s41524-024-01497-y>.
///
/// The morse transformation can be expressed as:
///
/// ```math
/// s(r) = (x(r) - x_\mathrm{avg}) / x_\mathrm{diff}
/// ```
/// Where
/// ```math
/// \begin{align*}
/// x(r) &= \exp{(-r/\lambda)} \\
/// x_\mathrm{avg} &= 0.5(\exp{(-r_\mathrm{out}/\lambda)}
/// + \exp{(-r_\mathrm{in}/\lambda)}) \\
/// x_\mathrm{diff} &= 0.5|\exp{(-r_\mathrm{out}/\lambda)}
/// - \exp{(-r_\mathrm{in}/\lambda)}|
/// \end{align*}
/// ```
///
/// The derivative is:
/// ```math
/// \frac{ds(r)}{dr} = -\frac{x(r)}{\lambda x_\mathrm{diff}}
/// ```
///
/// # Example
/// ```
/// use hoomd_chimes::transformation::MorseTransformation;
///
/// let lambda = 1.5;
/// let r_out = 3.0;
/// let r_in = 1.0;
/// let morse_trans: MorseTransformation = MorseTransformation {
///     lambda,
///     r_out,
///     r_in,
/// };
/// ```

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MorseTransformation {
    /// Morse decaying factor $`[\mathrm{length}]`$
    pub lambda: f64,
    /// distance where the smoothed potential will be 0 $`[\mathrm{length}]`$.
    pub r_out: f64,
    /// Inner distance cutoff $`[\mathrm{length}]`$.
    pub r_in: f64,
}

impl Transformation for MorseTransformation {
    /// The morse transformation function s
    #[inline]
    fn s(&self, r: &f64) -> f64 {
        let x_out = (-self.r_out / self.lambda).exp();
        let x_in = (-self.r_in / self.lambda).exp();
        let x_avg = 0.5 * (x_out + x_in);
        let x_diff = -0.5 * (x_out - x_in); // special for morse transformation. see chimes 2.0 eq

        // set r to r_in when r < r_in.
        // r > r_out is not took care of
        // where it assume the smoothing
        // function should set potential
        // to zero.
        let exprlen = if *r < self.r_in {
            x_in
        } else {
            (-r / self.lambda).exp()
        };
        (exprlen - x_avg) / x_diff
    }
    /// Partial derivative of the s function with respect to r
    #[inline]
    fn ds_dr(&self, r: &f64) -> f64 {
        let x_out = (-self.r_out / self.lambda).exp();
        let x_in = (-self.r_in / self.lambda).exp();
        let x_diff = -0.5 * (x_out - x_in); // special for morse transformation. see chimes 2.0 eq

        // set r to r_in when r < r_in.
        // r > r_out is not took care of
        // where it assume the smoothing
        // function should set potential
        // to zero.
        let exprlen = if *r < self.r_in {
            x_in
        } else {
            (-r / self.lambda).exp()
        };
        -exprlen / self.lambda / x_diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use rstest::*;

    use crate::transformation::MorseTransformation;

    #[rstest]
    fn r_valid_range(
        #[values(1.5, 2.5, 3.5)] lambda: f64,
        #[values(4.0, 5.0, 6.0)] r_out: f64,
        #[values(1.0, 2.0, 3.0)] r_in: f64,
    ) {
        let r = 3.5;
        let morse_trans = MorseTransformation {
            lambda,
            r_out,
            r_in,
        };

        assert_eq!(morse_trans.lambda, lambda);
        assert_eq!(morse_trans.r_out, r_out);
        assert_eq!(morse_trans.r_in, r_in);

        let x_out = (-r_out / lambda).exp();
        let x_in = (-r_in / lambda).exp();
        let x_avg = 0.5 * (x_out + x_in);
        let x_diff = -0.5 * (x_out - x_in);

        let exprlen = (-r / lambda).exp();
        let expected_s = (exprlen - x_avg) / x_diff;
        let expected_ds_dr = -exprlen / lambda / x_diff;

        assert_relative_eq!(morse_trans.s(&r), expected_s);
        assert_relative_eq!(morse_trans.ds_dr(&r), expected_ds_dr);
    }

    #[rstest]
    fn r_inner(
        #[values(1.5, 2.5, 3.5)] lambda: f64,
        #[values(4.0, 5.0, 6.0)] r_out: f64,
        #[values(1.0, 2.0, 3.0)] r_in: f64,
    ) {
        let r = 0.5;
        let morse_trans = MorseTransformation {
            lambda,
            r_out,
            r_in,
        };

        assert_eq!(morse_trans.lambda, lambda);
        assert_eq!(morse_trans.r_out, r_out);
        assert_eq!(morse_trans.r_in, r_in);

        let x_out = (-r_out / lambda).exp();
        let x_in = (-r_in / lambda).exp();
        let x_avg = 0.5 * (x_out + x_in);
        let x_diff = -0.5 * (x_out - x_in);

        let exprlen = x_in;
        let expected_s = (exprlen - x_avg) / x_diff;
        let expected_ds_dr = -exprlen / lambda / x_diff;

        assert_relative_eq!(morse_trans.s(&r), expected_s);
        assert_relative_eq!(morse_trans.ds_dr(&r), expected_ds_dr);
    }
}

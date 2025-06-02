// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`MorseTransformation`]
 */

use super::Transformation;
/** Calculate the morse transformation style given
    distance `r` with morse decaying factor `lambda`,
    outer radial cutoff `r_out` , and inner radial
    cutoff `r_in`. Finally, return a variable `s(r)`
    falls between [-1, 1] and its derivative `dr_dr(r)`
    with respect to r. See equation 3 to 5 in
    <https://doi.org/10.1038/s41524-024-01497-y>.

```math
    s(r) = (x(r) - x_\mathrm{avg}) / x_\mathrm{diff}
    x(r) = \exp{(-r/\lambda)}
    x_\mathrm{avg} = 0.5(\exp{(-r_\mathrm{out}/\lambda)}
      + \exp{(-r_\mathrm{in}/\lambda)})
    x_\mathrm{diff} = 0.5|\exp{(-r_\mathrm{out}/\lambda)}
      - \exp{(-r_\mathrm{in}/\lambda)}|
```

# Example
TODO
*/

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MorseTransformation {
    /// Morse decaying factor `[length]`
    pub lambda: f64,
    /// `r` value `[length]` where the smoothed potential will be 0.
    pub r_out: f64,
    /// `r` value `[length]` where the smoothing function is enabled. Should be < `r_cut`
    pub r_in: f64, // TODO: find alternate name?
}

impl Transformation for MorseTransformation {
    /** Construct an [`MorseTransformation`] with the given
    Morse decaying factor `lambda`, outer cutoff `r_out`,
    and inner cutoff `r_on`.

    # Example

    ```
    use hoomd_utility::chimes_transformation::MorseTransformation;

    let lambda = 1.5;
    let r_out = 3.0;
    let r_in = 1.0;
    let morse_trans: MorseTransformation = MorseTransformation{lambda, r_out, r_in};
    ```
    */
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
    use ::approx::assert_relative_eq;
    use rstest::*;

    use crate::chimes_transformation::MorseTransformation;

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

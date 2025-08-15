// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`InverseTransformation`]
 */

use super::Transformation;
/**
Calculate the inverse transformation style.

Given a distance $`r`$, the inverse transformation is defined by:
outer radial cutoff $`r_\mathrm{out}`$
and inner radial cutoff $`r_\mathrm{in}`$.

It calculates a coordinate $`s(r)`$ falls between [-1, 1]
and its derivative $`\frac{ds(r)}{dr}`$ with respect to $`r`$.

See equation 3 to 5 in
<https://doi.org/10.1038/s41524-024-01497-y>.

The inverse transformation can be expressed as:

```math
    s(r) = (x(r) - x_\mathrm{avg}) / x_\mathrm{diff}
```
Where
```math
    \begin{align*}
    x(r) &= 1 / r \\
    x_\mathrm{avg} &= 0.5(1 / r_\mathrm{out} + 1 / r_\mathrm{in}) \\
    x_\mathrm{diff} &= 0.5|1 / r_\mathrm{out} - 1 / r_\mathrm{in}|
    \end{align*}
```

The derivative is:
```math
\frac{ds(r)}{dr} = -\frac{1}{r^{2} x_\mathrm{diff}}
```

# Example
```
use hoomd_chimes::transformation::InverseTransformation;

let r_out = 3.0;
let r_in = 1.0;
let inverse_trans: InverseTransformation = InverseTransformation{r_out, r_in};
```
*/

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseTransformation {
    /// distance where the smoothed potential will be 0 $`[\mathrm{length}]`$.
    pub r_out: f64,
    /// Inner distance cutoff $`[\mathrm{length}]`$.
    pub r_in: f64,
}

impl Transformation for InverseTransformation {
    /// The inverse transformation function s
    #[inline]
    fn s(&self, r: &f64) -> f64 {
        let x_out = 1.0 / self.r_out;
        let x_in = 1.0 / self.r_in;
        let x_avg = 0.5 * (x_out + x_in);
        let x_diff = -0.5 * (x_out - x_in);

        // set r to r_in when r < r_in.
        // r > r_out is not took care of
        // where it assume the smoothing
        // function should set potential
        // to zero.
        let invrlen = if *r < self.r_in { x_in } else { 1.0 / *r };
        (invrlen - x_avg) / x_diff
    }
    /// Partial derivative of the s function with respect to r
    #[inline]
    fn ds_dr(&self, r: &f64) -> f64 {
        let x_out = 1.0 / self.r_out;
        let x_in = 1.0 / self.r_in;
        let x_diff = -0.5 * (x_out - x_in);

        -1.0 / r.powf(2.0) / x_diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::approx::assert_relative_eq;
    use rstest::*;

    use crate::transformation::InverseTransformation;

    #[rstest]
    fn r_valid_range(#[values(4.0, 5.0, 6.0)] r_out: f64, #[values(1.0, 2.0, 3.0)] r_in: f64) {
        let r: f64 = 3.5;
        let inverse_trans = InverseTransformation { r_out, r_in };

        assert_eq!(inverse_trans.r_out, r_out);
        assert_eq!(inverse_trans.r_in, r_in);

        let x_out = 1.0 / r_out;
        let x_in = 1.0 / r_in;
        let x_avg = 0.5 * (x_out + x_in);
        let x_diff = -0.5 * (x_out - x_in);

        let invrlen = 1.0 / r;
        let expected_s = (invrlen - x_avg) / x_diff;
        let expected_ds_dr = -1.0 / r.powf(2.0) / x_diff;

        assert_relative_eq!(inverse_trans.s(&r), expected_s);
        assert_relative_eq!(inverse_trans.ds_dr(&r), expected_ds_dr);
    }

    #[rstest]
    fn r_inner(#[values(4.0, 5.0, 6.0)] r_out: f64, #[values(1.0, 2.0, 3.0)] r_in: f64) {
        let r: f64 = 0.5;
        let inverse_trans = InverseTransformation { r_out, r_in };

        assert_eq!(inverse_trans.r_out, r_out);
        assert_eq!(inverse_trans.r_in, r_in);

        let x_out = 1.0 / r_out;
        let x_in = 1.0 / r_in;
        let x_avg = 0.5 * (x_out + x_in);
        let x_diff = -0.5 * (x_out - x_in);

        let invrlen = x_in;
        let expected_s = (invrlen - x_avg) / x_diff;
        let expected_ds_dr = -1.0 / r.powf(2.0) / x_diff;

        assert_relative_eq!(inverse_trans.s(&r), expected_s);
        assert_relative_eq!(inverse_trans.ds_dr(&r), expected_ds_dr);
    }
}

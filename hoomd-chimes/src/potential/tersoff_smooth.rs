// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`TersoffSmooth`]
 */

use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce};
use std::f64::consts::PI; // Note: Becky's chimes uses hard coding value: 3.14159265359

/**
Implement the Tersoff style smoothing $`f_s`$ of `ChIMES`
potential, for one plus two-body case:

```math
U(r) = f_s(r) \sum^{n}_{O=1} c_{O} T_{O}(s(r))
```
where:

```math
f_s(r) =
\begin{cases}
0 &\text{if } r > r_{\mathrm{out}} \\
1 &\text{if } r < r_{\mathrm{in}} \\
\frac{1}{2} +  \frac{1}{2}
\sin{\left( \pi \left[ \frac{r-d_t}{r_\mathrm{out} - d_t}\right]
+ \frac{\pi}{2}\right)} &\text{, otherwise}\\
\end{cases}
```

```math
d_t = r_{\mathrm{out}} * (1.0 - f_o)
```

Where `r_in` is the inner distance cutoff, same as that
defined in [`Chimes2b`], `r_out` is the outer distance cutoff
, and `f_o` is the parameter with the value between 0 and 1
 to control the activation of smoothing.

# Note:
See equation 8 in <https://doi.org/10.1038/s41524-024-01497-y>.

# Example:
```
use arrayvec::ArrayVec;
use hoomd_chimes::transformation::MorseTransformation;
use hoomd_chimes::potential::{Chimes2b, TersoffSmooth};
use hoomd_interaction::pairwise::{IsotropicEnergy, IsotropicForce};

let lambda = 1.5;
let r_out = 3.0;
let r_in = 1.0;
let fo = 0.75;
let coeff: ArrayVec<f64, 3> = [1.0, 2.0, 3.0].into_iter().collect();

let morse_trans: MorseTransformation = MorseTransformation {
    lambda,
    r_out,
    r_in,
};

let chimes2b_cheby: Chimes2b<MorseTransformation, 3> =
    Chimes2b::new(morse_trans, coeff, r_in);

let chimes2b = TersoffSmooth {
    f: chimes2b_cheby,
    r_out,
    r_in,
    fo,
};
```
*/
#[derive(Clone, Debug, PartialEq)]
pub struct TersoffSmooth<F> {
    /// The [`Chimes2b`] fucntion.
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

impl<F: IsotropicEnergy> IsotropicEnergy for TersoffSmooth<F> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.fs(r) * self.f.energy(r)
    }
}

impl<F: IsotropicForce + IsotropicEnergy> IsotropicForce for TersoffSmooth<F> {
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
    use arrayvec::ArrayVec;
    use rstest::*;

    use crate::potential::Chimes2b;

    #[rstest]
    fn test_construction() {
        let lambda = 1.5;
        let r_out = 3.0;
        let r_in = 1.0;
        let fo = 0.75;
        let coeff: ArrayVec<f64, 3> = [1.0, 2.0, 3.0].into_iter().collect();

        let morse_trans: MorseTransformation = MorseTransformation {
            lambda,
            r_out,
            r_in,
        };

        let chimes2b_cheby: Chimes2b<MorseTransformation, 3> =
            Chimes2b::new(morse_trans, coeff, r_in);

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
}

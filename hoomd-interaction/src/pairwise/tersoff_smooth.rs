// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`TersoffSmooth`]
 */

use super::{Chimes2b, IsotropicEnergy, IsotropicForce};
use std::f64::consts::PI;

/**
Implement the Tersoff style smoothing `f_s` of `ChIMES`
potential, for one plus two-body case:

```math
U(r) = c_0 + f_s(r) \sum^{\mathcal{n-1}}}_{O=1} c_{O} T_{O}(s(r))
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

        if r < dt {
            0.0
        } else if r > self.r_out {
            0.0
        } else {
            let pref = PI / (self.r_out - dt);
            0.5 * pref * (pref * (r - dt) + 0.5 * PI).cos()
        }
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Expanded`]
 */

use super::{IsotropicEnergy, IsotropicForce};

/** Expand another potential.

```math
U(r) = f(r - \delta)
```

TODO: Example
*/
#[derive(Clone, Debug, PartialEq)]
pub struct Expanded<F> {
    /// The original potential.
    pub f: F,
    /// $`\delta`$ value $`[\mathrm{length}]`$.
    pub delta: f64,
}

impl<F> Default for Expanded<F>
where
    F: Default,
{
    /** Construct a shifted potential with default parameters

    The defaults are:
    * `f = F::default()`
    * `delta = 0.0`
    */
    #[inline]
    fn default() -> Self {
        Self {
            f: F::default(),
            delta: 0.0,
        }
    }
}

impl<F: IsotropicEnergy> IsotropicEnergy for Expanded<F> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.f.energy(r - self.delta)
    }
}

impl<F: IsotropicForce> IsotropicForce for Expanded<F> {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        self.f.force(r - self.delta)
    }
}

// TODO: Test

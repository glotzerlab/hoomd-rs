// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`WeeksChandlerAnderson`]
*/

use super::{IsotropicEnergy, IsotropicForce};

/** Shift another potential to 0 at a given `r`.

<!-- U(r) = f(r) - f(r_\mathrm{shift}) -->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>=</mo><mi>f</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>−</mo><mi>f</mi><mo form="prefix" stretchy="false">(</mo><msub><mi>r</mi><mrow><mtext></mtext><mi>shift</mi></mrow></msub><mo form="postfix" stretchy="false">)</mo></mrow></math>

# Example

Shifted Lennard-Jones:
```
use hoomd_interaction::pairwise::{LennardJones, Shifted};

let epsilon = 1.5;
let sigma = 1.0;
let r_shift = 2.5;
let shifted_lj = Shifted::new(LennardJones::<12,6>::new(epsilon, sigma), r_shift);
```

Fields can be accessed directly:
```
use hoomd_interaction::pairwise::{LennardJones, Shifted};

let epsilon = 1.5;
let sigma = 1.0;
let r_shift = 2.5;
let mut shifted_lj = Shifted::new(LennardJones::<12,6>::new(epsilon, sigma), r_shift);

shifted_lj.r_shift = 3.0;
shifted_lj.f.sigma = 1.2;
```
*/
#[derive(Clone, Debug, PartialEq)]
pub struct Shifted<F> {
    /// The original potential.
    pub f: F,
    /// `r` value `[length]` where the shifted potential will be 0.
    pub r_shift: f64,
}

impl<F> Shifted<F> {
    #[inline]
    #[must_use]
    pub fn new(f: F, r_shift: f64) -> Self {
        Self { f, r_shift }
    }
}

impl<F: IsotropicEnergy> IsotropicEnergy for Shifted<F> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.f.energy(r) - self.f.energy(self.r_shift)
    }
}

impl<F: IsotropicForce> IsotropicForce for Shifted<F> {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        self.f.force(r)
    }
}

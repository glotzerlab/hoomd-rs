// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Boxcar`]
*/

use super::IsotropicEnergy;

/** Constant valued potential in a given range of `r` (_not differentiable_).

<!--
U(r) = \begin{cases}
0 & r \lt a \\
\varepsilon & a \le r \lt b \\
0 & r \ge b
\end{cases}
-->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>=</mo><mrow><mo fence="true" form="prefix">{</mo><mtable><mtr><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 0em;"><mn>0</mn></mtd><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 1em;"><mrow><mi>r</mi><mo>&lt;</mo><mi>a</mi></mrow></mtd></mtr><mtr><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 0em;"><mi>ε</mi></mtd><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 1em;"><mrow><mi>a</mi><mo>≤</mo><mi>r</mi><mo>&lt;</mo><mi>b</mi></mrow></mtd></mtr><mtr><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 0em;"><mn>0</mn></mtd><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 1em;"><mrow><mi>r</mi><mo>≥</mo><mi>b</mi></mrow></mtd></mtr></mtable><mo fence="true" form="postfix"></mo></mrow></mrow></math>

Compute boxcar potential function. Some uses of this in the literature call it
the "square well" potential.

# Examples

Basic usage:

```
use hoomd_interaction::pairwise::{IsotropicEnergy, Boxcar};

let epsilon = 1.5;
let (a,b) = (1.0, 2.5);

let boxcar = Boxcar::new(epsilon, a, b);
assert_eq!(boxcar.energy(0.0), 0.0);
assert_eq!(boxcar.energy(1.0), 1.5);
assert_eq!(boxcar.energy(2.0), 1.5);
assert_eq!(boxcar.energy(2.5), 0.0);
assert_eq!(boxcar.energy(1000.0), 0.0);
```

The parameters are public fields and may be set directly:

```
use hoomd_interaction::pairwise::{IsotropicEnergy, Boxcar};

let mut boxcar = Boxcar::new(1.5, 1.0, 2.5);
boxcar.epsilon = -2.0;
boxcar.a = 0.0;
boxcar.b = 1.0;
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Boxcar {
    /// Energy scale `[energy]`.
    pub epsilon: f64,
    /// Left side of the boxcar `[length]`.
    pub a: f64,
    /// Right side of the boxcar `[length]`.
    pub b: f64
}

impl Boxcar {
    /** Construct a [`Boxcar`] with the given values for `epsilon`, `a`, and `b`.

    # Example

    ```
    use hoomd_interaction::pairwise::Boxcar;

    let boxcar = Boxcar::new(-2.0, 0.0, 1.0);
    ```
    */
    #[inline]
    #[must_use]
    pub fn new(epsilon: f64, a: f64, b: f64) -> Self {
        Self { epsilon, a, b }
        }
}

impl IsotropicEnergy for Boxcar {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        match r {
            x if x < self.a => 0.0,
            x if x < self.b => self.epsilon,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn general_case(
        #[values(1.0, -2.0, 12.125, 0.25)]
        epsilon: f64,
        #[values(1.0, 2.0, 0.5)]
        a: f64,
        #[values(0.5, 0.125)]
        w: f64) {
            let b = a + w;
            let boxcar = Boxcar::new(epsilon, a, b);

            assert_eq!(boxcar.epsilon, epsilon);
            assert_eq!(boxcar.a, a);
            assert_eq!(boxcar.b, b);
            
            // Note: These tests could be cleaner with the next_up_down feature to come in a
            // future version of Rust: https://github.com/rust-lang/rust/issues/91399

            // Left
            assert_eq!(boxcar.energy(0.0), 0.0);
            assert_eq!(boxcar.energy(a * (1.0-f64::EPSILON)), 0.0);

            // Center
            assert_eq!(boxcar.energy(a), epsilon);
            assert_eq!(boxcar.energy(a * (1.0+f64::EPSILON)), epsilon);
            assert_eq!(boxcar.energy(a + w/2.0), epsilon);
            assert_eq!(boxcar.energy(b * (1.0-f64::EPSILON)), epsilon);

            // Right
            assert_eq!(boxcar.energy(b), 0.0);
            assert_eq!(boxcar.energy(b * 10.0), 0.0);
        }
}

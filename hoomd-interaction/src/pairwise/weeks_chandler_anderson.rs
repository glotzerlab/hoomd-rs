// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`WeeksChandlerAnderson`]
*/

use super::{IsotropicEnergy, IsotropicForce};
use super::LennardJones;

/** Potential with a steep repulsive core.
    
<!--
U(r) = \begin{cases}
4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6} \right] + \varepsilon & r \lt 2^{1/6} \sigma \\

0 & r \ge 2^{1/6} \sigma
\end{cases}
-->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>=</mo><mrow><mo fence="true" form="prefix">{</mo><mtable><mtr><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 0em;"><mrow><mn>4</mn><mi>ε</mi><mrow><mo fence="true" form="prefix">[</mo><msup><mrow><mo fence="true" form="prefix">(</mo><mfrac><mi>σ</mi><mi>r</mi></mfrac><mo fence="true" form="postfix">)</mo></mrow><mn>12</mn></msup><mo>−</mo><msup><mrow><mo fence="true" form="prefix">(</mo><mfrac><mi>σ</mi><mi>r</mi></mfrac><mo fence="true" form="postfix">)</mo></mrow><mn>6</mn></msup><mo fence="true" form="postfix">]</mo></mrow><mo>+</mo><mi>ε</mi></mrow></mtd><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 1em;"><mrow><mi>r</mi><mo>&lt;</mo><msup><mn>2</mn><mrow><mn>1</mn><mi>/</mi><mn>6</mn></mrow></msup><mi>σ</mi></mrow></mtd></mtr><mtr><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 0em;"><mn>0</mn></mtd><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 1em;"><mrow><mi>r</mi><mo>≥</mo><msup><mn>2</mn><mrow><mn>1</mn><mi>/</mi><mn>6</mn></mrow></msup><mi>σ</mi></mrow></mtd></mtr></mtable><mo fence="true" form="postfix"></mo></mrow></mrow></math>
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeeksChandlerAnderson {
    /// Energy scale `[energy]`.
    pub epsilon: f64,
    /// Interaction width `[length]`.
    pub sigma: f64
}

impl WeeksChandlerAnderson {
    #[inline]
    #[must_use]
    pub fn new(epsilon: f64, sigma: f64) -> Self {
        Self { epsilon, sigma }
        }
}

impl IsotropicEnergy for WeeksChandlerAnderson {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        if r < 2.0_f64.powf(1.0/6.0) * self.sigma {
            let lj = LennardJones::<12,6>::new(self.epsilon, self.sigma);
            lj.energy(r) + self.epsilon
            }
        else
            {
            0.0
            }
        }
    }

impl IsotropicForce for WeeksChandlerAnderson {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        if r < 2.0_f64.powf(1.0/6.0) * self.sigma {
            let lj = LennardJones::<12,6>::new(self.epsilon, self.sigma);
            lj.force(r)
            }
        else
            {
            0.0
            }
        }
    }

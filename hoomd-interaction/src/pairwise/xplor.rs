// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`WeeksChandlerAnderson`]
 */

use super::{IsotropicEnergy, IsotropicForce};

/** Smoothly shift a potential (and its force) to 0 at some `r_cut`, beginning at `r_on`.

<!-- U(r) = S(r) \cdot f(r) -->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>=</mo><mi>S</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>⋅</mo><mi>f</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo></mrow></math>
where:
<!-- S(r) =
\begin{cases}
1 & r < r_{\mathrm{on}} \\
\frac{(r_{\mathrm{cut}}^2 - r^2)^2 \cdot
(r_{\mathrm{cut}}^2 + 2r^2 -
3r_{\mathrm{on}}^2)}{(r_{\mathrm{cut}}^2 -
r_{\mathrm{on}}^2)^3}
& r_{\mathrm{on}} < r \le r_{\mathrm{cut}} \\
0 & r \ge r_{\mathrm{cut}} \\
\end{cases}
-->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>S</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>=</mo><mrow><mo fence="true" form="prefix">{</mo><mtable><mtr><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 0em;"><mn>1</mn></mtd><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 1em;"><mrow><mi>r</mi><mo>&lt;</mo><msub><mi>r</mi><mrow><mtext></mtext><mi>on</mi></mrow></msub></mrow></mtd></mtr><mtr><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 0em;"><mfrac><mrow><mo form="prefix" stretchy="false" lspace="0em" rspace="0em">(</mo><msubsup><mi>r</mi><mrow><mtext></mtext><mi>cut</mi></mrow><mn>2</mn></msubsup><mo>−</mo><msup><mi>r</mi><mn>2</mn></msup><msup><mo form="postfix" stretchy="false">)</mo><mn>2</mn></msup><mo>⋅</mo><mo form="prefix" stretchy="false">(</mo><msubsup><mi>r</mi><mrow><mtext></mtext><mi>cut</mi></mrow><mn>2</mn></msubsup><mo>+</mo><mn>2</mn><msup><mi>r</mi><mn>2</mn></msup><mo>−</mo><mn>3</mn><msubsup><mi>r</mi><mrow><mtext></mtext><mi>on</mi></mrow><mn>2</mn></msubsup><mo form="postfix" stretchy="false" lspace="0em" rspace="0em">)</mo></mrow><mrow><mo form="prefix" stretchy="false" lspace="0em" rspace="0em">(</mo><msubsup><mi>r</mi><mrow><mtext></mtext><mi>cut</mi></mrow><mn>2</mn></msubsup><mo>−</mo><msubsup><mi>r</mi><mrow><mtext></mtext><mi>on</mi></mrow><mn>2</mn></msubsup><msup><mo form="postfix" stretchy="false">)</mo><mn>3</mn></msup></mrow></mfrac></mtd><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 1em;"><mrow><msub><mi>r</mi><mrow><mtext></mtext><mi>on</mi></mrow></msub><mo>≤</mo><mi>r</mi><mo>&lt;</mo><msub><mi>r</mi><mrow><mtext></mtext><mi>cut</mi></mrow></msub></mrow></mtd></mtr><mtr><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 0em;"><mn>0</mn></mtd><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 1em;"><mrow><mi>r</mi><mo>≥</mo><msub><mi>r</mi><mrow><mtext></mtext><mi>cut</mi></mrow></msub></mrow></mtd></mtr></mtable><mo fence="true" form="postfix"></mo></mrow></mrow></math>
# Example
TODO
*/

#[derive(Clone, Debug, PartialEq)]
pub struct Xplor<F> {
    /// The original potential.
    pub f: F,
    /// `r` value `[length]` where the smoothed potential will be 0.
    pub r_cut: f64,
    /// `r` value `[length]` where the smoothing function is enabled. Should be < `r_cut`
    pub r_on: f64, // TODO: find alternate name?
}

impl<F> Xplor<F> {
    /** Construct an [`Xplor`] with the given potential `f`, a cutoff `r_cut`, and a
    start point `r_on`.

    # Example

    Smooth a Lennard-Jones potential to 0 at some `r_cut`.
    ```
    use hoomd_interaction::pairwise::{LennardJones, Xplor};

    let epsilon = 1.5;
    let sigma = 1.0;
    let r_cut = 2.5 * sigma;
    let r_on = 1.5 * sigma;
    let xplor_lj = Xplor::new(
        LennardJones::<12,6>::new(epsilon, sigma), r_cut, r_on
    );
    ```
    */
    #[inline]
    #[must_use]
    pub fn new(f: F, r_cut: f64, r_on: f64) -> Self {
        Self { f, r_cut, r_on }
    }

    /// The xplor shifting function
    #[inline]
    fn s(&self, r: f64) -> f64 {
        // NOTE: r checks must be performed here to scale the forces properly
        if r < self.r_on {
            1.0
        } else if r > self.r_cut {
            0.0
        } else {
            let r_sq = r.powi(2);
            let r_cut_sq = self.r_cut.powi(2);
            let r_on_sq = self.r_on.powi(2);
            (r_cut_sq - r_sq).powi(2) * (r_cut_sq + 2.0 * r_sq - 3.0 * r_on_sq)
                / (r_cut_sq - r_on_sq).powi(3)
        }
    }
    /// Partial derivative of the xplor function with respect to r
    #[inline]
    fn ds_dr(&self, r: f64) -> f64 {
        let r_sq = r.powi(2);
        let r_cut_sq = self.r_cut.powi(2);
        let r_on_sq = self.r_on.powi(2);
        12.0 * r * ((r_cut_sq - r_sq) * (r_on_sq - r_sq)) / (r_cut_sq - r_on_sq).powi(3)
    }
}

impl<F: IsotropicEnergy> IsotropicEnergy for Xplor<F> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.s(r) * self.f.energy(r)
    }
}

impl<F: IsotropicForce + IsotropicEnergy> IsotropicForce for Xplor<F> {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        if r < self.r_on {
            self.f.force(r)
        } else if r > self.r_cut {
            0.0
        } else {
            // Chain rule of s(r)*U(r)
            self.s(r) * self.f.force(r) + self.ds_dr(r) * self.f.energy(r)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_abs_diff_eq, assert_abs_diff_ne, assert_relative_eq};
    use rstest::*;

    use crate::pairwise::LennardJones;

    #[rstest]
    fn special_points_12_6(
        #[values(1.0, 2.0, 12.125, 0.25)] epsilon: f64,
        #[values(1.0, 2.0, 0.5)] sigma: f64,
    ) {
        let lj: LennardJones = LennardJones::new(epsilon, sigma);
        let r_on = 1.0; // Provides cases where r_on <, =, > sigma and epsilon
        let r_cut = 2.5 * sigma;
        let xplor_lj = Xplor::new(lj, r_cut, r_on);

        assert_eq!(xplor_lj.f.epsilon, epsilon);
        assert_eq!(xplor_lj.f.sigma, sigma);
        assert_eq!(xplor_lj.r_on, r_on);
        assert_eq!(xplor_lj.r_cut, r_cut);

        // Values should not be shifted below r_on
        assert_abs_diff_eq!(xplor_lj.energy(r_on / 2.0), lj.energy(r_on / 2.0));
        assert_abs_diff_eq!(xplor_lj.energy(r_on - 1e-6), lj.energy(r_on - 1e-6));
        assert_abs_diff_eq!(xplor_lj.force(r_on / 2.0), lj.force(r_on / 2.0)); // TODO
        assert_abs_diff_eq!(xplor_lj.force(r_on - 1e-6), lj.force(r_on - 1e-6)); // TODO

        // Values should be zero at and above r_cut
        assert_abs_diff_eq!(xplor_lj.energy(r_cut), 0.0);
        assert_abs_diff_eq!(xplor_lj.energy(r_cut + 1e-6), 0.0);
        assert_abs_diff_eq!(xplor_lj.energy(r_cut * 2.0), 0.0);

        assert_abs_diff_eq!(xplor_lj.force(r_cut), 0.0); // TODO: should be zero!
        assert_abs_diff_eq!(xplor_lj.force(r_cut + 1e-6), 0.0);
        assert_abs_diff_eq!(xplor_lj.force(r_cut * 2.0), 0.0);

        assert_abs_diff_eq!(xplor_lj.energy(sigma), 0.0);
        // assert_relative_eq!(xplor_lj.force(sigma), 24.0 * epsilon / sigma); // TODO

        // Values should not be the same between r_on and r_cut
        assert_abs_diff_ne!(
            xplor_lj.energy((r_on + r_cut) / 2.0),
            lj.energy((r_on + r_cut) / 2.0)
        );
        assert_abs_diff_ne!(
            xplor_lj.force((r_on + r_cut) / 2.0),
            lj.force((r_on + r_cut) / 2.0)
        );

        // Zero crossing
        assert_abs_diff_eq!(xplor_lj.energy(sigma), 0.0);
        // assert_relative_eq!(xplor_lj.force(sigma), 24.0 * epsilon / sigma); // TODO

        // Bottom of the well
        let r_min = 2.0_f64.powf(1.0 / 6.0) * sigma;
        assert_relative_eq!(xplor_lj.energy(r_min), -epsilon * xplor_lj.s(r_min));
        // assert_abs_diff_eq!(xplor_lj.force(r_min), 0.0, epsilon = 1e-12); // TODO

        // r = 2 sigma
        assert_relative_eq!(
            xplor_lj.energy(2.0 * sigma),
            -63.0 / 1024.0 * epsilon * xplor_lj.s(2.0 * sigma)
        );
        // assert_relative_eq!(
        //     xplor_lj.force(2.0 * sigma),
        //     -93.0 / 512.0 * epsilon / sigma
        // ); // TODO: should be shifted
    }
}

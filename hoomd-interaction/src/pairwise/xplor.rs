// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`WeeksChandlerAnderson`]
 */

use super::{IsotropicEnergy, IsotropicForce};

/** Smoothly shift a potential (and its force) to 0 at some `r`.

# Example
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
    #[inline]
    #[must_use]
    pub fn new(f: F, r_cut: f64, r_on: f64) -> Self {
        Self { f, r_cut, r_on }
    }

    /// The xplor shifting function
    #[inline]
    fn s(&self, r: f64) -> f64 {
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
    /// The xplor shifting function
    #[inline]
    fn ds_dr(&self, r: f64) -> f64 {
        // TODO: can we share r_*sq between energy and force?
        let r_sq = r.powi(2);
        let r_cut_sq = self.r_cut.powi(2);
        let r_on_sq = self.r_on.powi(2);
        12.0 * r * ((r_cut_sq - r_sq) * (r_on_sq - r_sq)) / (r_cut_sq - r_on_sq).powi(3)
    }
}

impl<F: IsotropicEnergy> IsotropicEnergy for Xplor<F> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        // self.f.energy(r) - self.f.energy(self.r_on)
        // if r < self.r_on {self.f.energy(r)} else {self.s(r) * self.f.energy(r)}
        self.s(r) * self.f.energy(r)
    }
    // NOTE: HOOMD impl has special case where r_on > r_cut. However, because each pot
    // is separate in this version, users can just not xplor on WCA
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

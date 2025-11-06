// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`WeeksChandlerAnderson`]

use super::{UnivariateEnergy, UnivariateForce};

/// Smoothly shift a potential (and its force) to 0 at some `r_cut`, beginning at `r_smooth`.
///
/// ```math
/// U(r) = S(r) \cdot f(r)
/// ```
/// where:
/// ```math
/// S(r) =
/// \begin{cases}
/// 1 & r < r_{\mathrm{on}} \\
/// \frac{(r_{\mathrm{cut}}^2 - r^2)^2 \cdot
/// (r_{\mathrm{cut}}^2 + 2r^2 -
/// 3r_{\mathrm{on}}^2)}{(r_{\mathrm{cut}}^2 -
/// r_{\mathrm{on}}^2)^3}
/// & r_{\mathrm{on}} < r \le r_{\mathrm{cut}} \\
/// 0 & r \ge r_{\mathrm{cut}} \\
/// \end{cases}
/// ```
/// # Example
///
/// ```
/// use hoomd_interaction::pairwise::{LennardJones, Xplor};
///
/// let epsilon = 1.5;
/// let sigma = 1.0;
/// let r_cut = 2.5 * sigma;
/// let r_smooth = 1.5 * sigma;
/// let xplor_lj = Xplor { f: LennardJones::<12,6> { epsilon, sigma }, r_cut, r_smooth };
#[derive(Clone, Debug, PartialEq)]
pub struct Xplor<F> {
    /// The original potential.
    pub f: F,
    /// `r` value *(\[length\])* where the smoothed potential will be 0.
    pub r_cut: f64,
    /// `r` value *(\[length\])* where the smoothing function is enabled. Should be less than `r_cut`
    pub r_smooth: f64,
}

impl<F> Xplor<F> {
    /// The xplor shifting function
    #[inline]
    fn s(&self, r: f64) -> f64 {
        if r < self.r_smooth {
            1.0
        } else if r > self.r_cut {
            0.0
        } else {
            let r_sq = r.powi(2);
            let r_cut_sq = self.r_cut.powi(2);
            let r_smooth_sq = self.r_smooth.powi(2);
            (r_cut_sq - r_sq).powi(2) * (r_cut_sq + 2.0 * r_sq - 3.0 * r_smooth_sq)
                / (r_cut_sq - r_smooth_sq).powi(3)
        }
    }
    /// Partial derivative of the xplor function with respect to r
    #[inline]
    fn ds_dr(&self, r: f64) -> f64 {
        let r_sq = r.powi(2);
        let r_cut_sq = self.r_cut.powi(2);
        let r_smooth_sq = self.r_smooth.powi(2);
        12.0 * r * ((r_cut_sq - r_sq) * (r_smooth_sq - r_sq)) / (r_cut_sq - r_smooth_sq).powi(3)
    }
}

impl<F: UnivariateEnergy> UnivariateEnergy for Xplor<F> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.s(r) * self.f.energy(r)
    }
}

impl<F: UnivariateForce + UnivariateEnergy> UnivariateForce for Xplor<F> {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        if r < self.r_smooth {
            self.f.force(r)
        } else if r > self.r_cut {
            0.0
        } else {
            // Chain rule of -d/dr(s(r)*U(r))
            self.s(r) * self.f.force(r) - self.ds_dr(r) * self.f.energy(r)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::{assert_abs_diff_eq, assert_abs_diff_ne, assert_relative_eq};
    use rstest::*;

    use crate::univariate::LennardJones;

    #[rstest]
    fn special_points_12_6(
        #[values(1.0, 2.0, 12.125, 0.25)] epsilon: f64,
        #[values(1.0, 2.0, 0.5)] sigma: f64,
    ) {
        let lj: LennardJones = LennardJones { epsilon, sigma };
        let r_smooth = 1.0; // Provides cases where r_smooth <, =, > sigma and epsilon
        let r_cut = 2.5 * sigma;
        let xplor_lj = Xplor {
            f: lj.clone(),
            r_cut,
            r_smooth,
        };

        assert_eq!(xplor_lj.f.epsilon, epsilon);
        assert_eq!(xplor_lj.f.sigma, sigma);
        assert_eq!(xplor_lj.r_smooth, r_smooth);
        assert_eq!(xplor_lj.r_cut, r_cut);

        // Values should not be shifted below r_smooth
        assert_abs_diff_eq!(xplor_lj.energy(r_smooth / 2.0), lj.energy(r_smooth / 2.0));
        assert_abs_diff_eq!(
            xplor_lj.energy(r_smooth.next_down()),
            lj.energy(r_smooth.next_down())
        );
        assert_abs_diff_eq!(xplor_lj.force(r_smooth / 2.0), lj.force(r_smooth / 2.0));
        assert_abs_diff_eq!(
            xplor_lj.force(r_smooth.next_down()),
            lj.force(r_smooth.next_down())
        );

        if sigma < r_smooth {
            assert_abs_diff_eq!(xplor_lj.energy(sigma), 0.0);
            assert_relative_eq!(xplor_lj.force(sigma), 24.0 * epsilon / sigma);
        }

        // Values should be zero at and above r_cut
        assert_abs_diff_eq!(xplor_lj.energy(r_cut), 0.0);
        assert_abs_diff_eq!(xplor_lj.energy(r_cut.next_up()), 0.0);
        assert_abs_diff_eq!(xplor_lj.energy(r_cut * 2.0), 0.0);

        assert_abs_diff_eq!(xplor_lj.force(r_cut), 0.0);
        assert_abs_diff_eq!(xplor_lj.force(r_cut.next_up()), 0.0);
        assert_abs_diff_eq!(xplor_lj.force(r_cut * 2.0), 0.0);

        // Values should not be the same between r_smooth and r_cut
        assert_abs_diff_ne!(
            xplor_lj.energy(f64::midpoint(r_smooth, r_cut)),
            lj.energy(f64::midpoint(r_smooth, r_cut))
        );
        assert_abs_diff_ne!(
            xplor_lj.force(f64::midpoint(r_smooth, r_cut)),
            lj.force(f64::midpoint(r_smooth, r_cut))
        );

        // Bottom of the well (without the xplor modification)
        let r_min = 2.0_f64.powf(1.0 / 6.0) * sigma;
        assert_relative_eq!(xplor_lj.energy(r_min), -epsilon * xplor_lj.s(r_min));
        if sigma < r_smooth {
            assert_abs_diff_eq!(xplor_lj.force(r_min), 0.0, epsilon = 1e-12);
        }

        // r = 2 sigma
        assert_relative_eq!(
            xplor_lj.energy(2.0 * sigma),
            -63.0 / 1024.0 * epsilon * xplor_lj.s(2.0 * sigma)
        );
        assert_relative_eq!(
            xplor_lj.force(2.0 * sigma),
            -(93.0 * r_cut.powi(6) * epsilon - 279.0 * r_cut.powi(4) * r_smooth.powi(2) * epsilon
                + 1476.0 * r_cut.powi(2) * r_smooth.powi(2) * sigma.powi(2) * epsilon
                - 1440.0 * r_cut.powi(2) * sigma.powi(4) * epsilon
                - 1440.0 * r_smooth.powi(2) * sigma.powi(4) * epsilon
                - 192.0 * sigma.powi(6) * epsilon)
                / (512.0 * r_cut.powi(6) * sigma
                    - 1536.0 * r_cut.powi(4) * r_smooth.powi(2) * sigma
                    + 1536.0 * r_cut.powi(2) * r_smooth.powi(4) * sigma
                    - 512.0 * r_smooth.powi(6) * sigma)
        );
    }
}

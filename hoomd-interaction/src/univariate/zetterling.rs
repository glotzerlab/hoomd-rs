// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Zetterling`]

use super::{UnivariateEnergy, UnivariateForce};

/// `Zetterling` computes the oscillating pair potential between every pair of
/// particles in the simulation state.
/// ```math
/// U(r) = \epsilon \frac{\exp(\alpha r/\ell)\cos(2k_Fr/\ell)}{(r/\ell)^3} + \beta\left(\frac{\sigma \ell}{r}\right)^n
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Zetterling {
    /// Energy scale of the first term *(\[energy\])*.
    pub epsilon: f64,
    /// Screening factor *(\[unitless\])*.
    pub alpha: f64,
    /// Wave number to mimic the Friedel oscillations effect *(\[unitless\])*.
    pub kf: f64,
    /// Energy scale of the second term *(\[energy\])*.
    pub beta: f64,
    /// Repulsive core size *(\[unitless\])*.
    pub sigma: f64,
    /// The power to take sigma/r in the second term *(\[unitless\])*.
    pub n: f64,
    /// The length scale of the distances *(\[length\])*
    pub ell: f64,
}

impl UnivariateEnergy for Zetterling {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        let sigma_ell_r = self.sigma * self.ell / r;
        let r_ell = r / self.ell;

        self.epsilon * ((self.alpha * r_ell).exp()) * ((2.0 * self.kf * r_ell).cos())
            / (r_ell.powi(3))
            + self.beta * (sigma_ell_r.powf(self.n))
    }
}

impl UnivariateForce for Zetterling {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        let r_ell = r / self.ell;
        let r_inv = r.recip();
        let cos = (2.0 * self.kf * r_ell).cos();
        let sin = (2.0 * self.kf * r_ell).sin();
        let exp = (self.alpha * r_ell).exp();

        let first = -self.epsilon * self.alpha * (self.ell.powi(2)) * exp * cos * (r_inv.powi(3));
        let second =
            self.epsilon * 2.0 * self.kf * (self.ell.powi(2)) * exp * sin * (r_inv.powi(3));
        let third = 3.0 * self.epsilon * (self.ell.powi(3)) * exp * cos * (r_inv.powi(4));
        let fourth = self.beta
            * self.n
            * ((self.sigma * self.ell).powf(self.n))
            * (r_inv.powf(self.n + 1.0));

        first + second + third + fourth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;

    #[test]
    fn select_points() {
        let zetterling1 = Zetterling {
            epsilon: 1.58,
            alpha: -0.22,
            kf: 4.12,
            beta: 0.95533,
            sigma: 1.0,
            n: 18.0,
            ell: 1.0,
        };

        let (u_1, r_1) = (-0.742_644_124_392_870, 1.130_547_632_166_212);
        assert_relative_eq!(u_1, zetterling1.energy(r_1), epsilon = 1e-12);
        assert_relative_eq!(0.0, zetterling1.force(r_1), epsilon = 1e-6);

        let (u_2, r_2) = (-0.153_547_971_965_573, 1.879_994_132_512_336);
        assert_relative_eq!(u_2, zetterling1.energy(r_2), epsilon = 1e-12);
        assert_relative_eq!(0.0, zetterling1.force(r_2), epsilon = 1e-10);

        let zetterling2 = Zetterling {
            epsilon: 1.04,
            alpha: 0.33,
            kf: 4.139,
            beta: 0.94656,
            sigma: 1.0,
            n: 14.5,
            ell: 1.0,
        };

        let (u_1, r_1) = (-0.883_354_387_732_971, 1.132_662_993_677_647);
        assert_relative_eq!(u_1, zetterling2.energy(r_1), epsilon = 1e-12);
        assert_relative_eq!(0.0, zetterling2.force(r_1), epsilon = 1e-6);

        let (u_2, r_2) = (-0.287_906_300_605_616, 1.879_255_603_675_727);
        assert_relative_eq!(u_2, zetterling2.energy(r_2), epsilon = 1e-12);
        assert_relative_eq!(0.0, zetterling2.force(r_2), epsilon = 1e-6);
    }
}

//! ...

use num_complex::Complex64;
use std::{
    f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2},
    ops::Index,
};

/// Complex spherical harmonics `Y_L^m` for a single degree L.
///
/// Index with `[m]` to access `Y_L^m` for m = 0..L.
/// The m = 0 term is always purely real.
pub struct HarmonicOutput<const L: usize> {
    /// `Y_L^0` (zonal harmonic, always real).
    pub m0: Complex64,
    /// `Y_L^m` for m = 1..L, stored at index m − 1.
    pub mp: [Complex64; L],
}

impl<const L: usize> Index<usize> for HarmonicOutput<L> {
    type Output = Complex64;

    #[inline]
    fn index(&self, index: usize) -> &Complex64 {
        match index {
            0 => &self.m0,
            n => &self.mp[n - 1],
        }
    }
}

/// Compute complex spherical harmonics `Y_L^m(x`, y, z) for m = 0..L.
///
/// The point (x, y, z) must lie on the unit sphere.
#[must_use]
#[inline]
pub fn spherical_harmonic<const L: usize>(x: f64, y: f64, z: f64) -> HarmonicOutput<L> {
    let rxy2 = x * x + y * y;

    // Normalized seed: prefactor(L, L) * (2L-1)!! = sqrt((2L+1) * r(L) / (2π))
    // where r(L) = (2L-1)!! / (2^L * L!) stays small via r(l) = r(l-1) * (2l-1)/(2l)
    let norm_seed = {
        let mut r = 1.0;
        for k in 1..=L {
            r *= (2 * k - 1) as f64 / (2 * k) as f64;
        }
        f64::sqrt((2 * L + 1) as f64 * r / (2.0 * PI)) * FRAC_1_SQRT_2
    };

    // h[k] are the normalized polar parts of the spherical harmonic, defined as:
    // h[k] = prefactor(L, k+1) * Q_l^{k+1}, h_0 = prefactor(L, 0) * Q_l^0
    // All values of h remain small, as the prefactor and recurrence mostly cancel at
    // each step. h[k>1] includes the correct normalization for complex SHs √2.
    let h_0;
    let mut h = [0.0; L];

    if L == 0 {
        h_0 = f64::sqrt(1.0 / (4.0 * PI));
    } else {
        h[L - 1] = norm_seed;

        // sqrt(2*L): is reused as the h[L-2] prefactor and carried through recurrence
        // After the loop, carry = sqrt((L-1)(L+2)), which is the m=0 step's numerator
        let mut carry = f64::sqrt(2.0 * L as f64);

        if L > 1 {
            h[L - 2] = z * carry * h[L - 1];

            for m in (1..L - 1).rev() {
                let denom = f64::sqrt(((L - m) * (L + m + 1)) as f64);
                h[m - 1] = (2.0 * (m + 1) as f64 * z * h[m] - rxy2 * carry * h[m + 1]) / denom;
                carry = denom;
            }
        }

        // m = 0 step: carry = sqrt((L-1)(L+2)) after loop (or sqrt(2L) if L≤2; * 0 when L=1)
        let denom = f64::sqrt((2 * L * (L + 1)) as f64);
        let h1 = if L > 1 { h[1] } else { 0.0 };
        h_0 = (2.0 * z * h[0] - rxy2 * carry * h1) / denom * SQRT_2;
    }

    // Assemble complex output using both azimuthal components (cm, sm).
    let mut result = [Complex64::ZERO; L];

    if L > 0 {
        let mut cm = x;
        let mut sm = y;
        result[0] = Complex64::new(h[0] * cm, h[0] * sm);

        for m in 1..L {
            let prev_cm = cm;
            let prev_sm = sm;
            cm = prev_cm * x - prev_sm * y;
            sm = prev_cm * y + prev_sm * x;
            result[m] = Complex64::new(h[m] * cm, h[m] * sm);
        }
    }

    HarmonicOutput {
        m0: Complex64::new(h_0, 0.0),
        mp: result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::marker::PhantomData;

    type Degree<const L: usize> = PhantomData<[(); L]>;
    fn degree<const L: usize>() -> Degree<L> {
        Degree::default()
    }

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn inv3() -> f64 {
        1.0 / 3.0_f64.sqrt()
    }

    #[test]
    fn l0() {
        let sh = spherical_harmonic::<0>(0.0, 0.0, 1.0);
        let expected = 1.0 / (2.0 * f64::sqrt(PI));
        assert!(approx_eq(sh[0].re, expected, 1e-12));
        assert!(approx_eq(sh[0].im, 0.0, 1e-12));
        assert_eq!(sh.mp.len(), 0);
    }

    #[test]
    fn l1_north_pole() {
        let sh = spherical_harmonic::<1>(0.0, 0.0, 1.0);
        let c = f64::sqrt(3.0 / (4.0 * PI));
        assert!(approx_eq(sh[0].re, c, 1e-12));
        assert!(approx_eq(sh[0].im, 0.0, 1e-12));
        assert!(approx_eq(sh[1].re, 0.0, 1e-12));
        assert!(approx_eq(sh[1].im, 0.0, 1e-12));
    }

    #[test]
    fn l1_x_axis() {
        let sh = spherical_harmonic::<1>(1.0, 0.0, 0.0);
        let c = f64::sqrt(3.0 / (8.0 * PI));
        assert!(approx_eq(sh[0].re, 0.0, 1e-12));
        assert!(approx_eq(sh[0].im, 0.0, 1e-12));
        assert!(approx_eq(sh[1].re, c, 1e-12));
        assert!(approx_eq(sh[1].im, 0.0, 1e-12));
    }

    #[test]
    fn l1_y_axis() {
        let sh = spherical_harmonic::<1>(0.0, 1.0, 0.0);
        let c = f64::sqrt(3.0 / (8.0 * PI));
        assert!(approx_eq(sh[0].re, 0.0, 1e-12));
        assert!(approx_eq(sh[0].im, 0.0, 1e-12));
        assert!(approx_eq(sh[1].re, 0.0, 1e-12));
        assert!(approx_eq(sh[1].im, c, 1e-12));
    }

    #[test]
    fn l2_finite() {
        let inv3 = 1.0 / 3.0_f64.sqrt();
        let sh = spherical_harmonic::<2>(inv3, inv3, inv3);
        assert_eq!(sh.mp.len(), 2);
        assert!(sh.m0.re.is_finite());
        assert!(sh.m0.im.is_finite());
        for v in &sh.mp {
            assert!(v.re.is_finite());
            assert!(v.im.is_finite());
        }
    }

    /// Validate against sphrs via Y_l^m = (S_l^{+m} + i·S_l^{-m}) / √2.
    fn check_against_sphrs<const L: usize>(x: f64, y: f64, z: f64) {
        use approxim::assert_abs_diff_eq;
        use sphrs::{Coordinates, RealSH, SHEval};

        let sh = spherical_harmonic::<L>(x, y, z);
        let p = Coordinates::cartesian(x, y, z);

        let expected_m0: f64 = RealSH::Spherical.eval(L as i64, 0, &p);
        assert_abs_diff_eq!(sh[0].re, expected_m0, epsilon = 1e-8);
        assert_abs_diff_eq!(sh[0].im, 0.0, epsilon = 1e-8);

        for m in 1..=L {
            let s_pos: f64 = RealSH::Spherical.eval(L as i64, m as i64, &p);
            let s_neg: f64 = RealSH::Spherical.eval(L as i64, -(m as i64), &p);
            assert_abs_diff_eq!(sh[m].re, s_pos * FRAC_1_SQRT_2, epsilon = 1e-8);
            assert_abs_diff_eq!(sh[m].im, s_neg * FRAC_1_SQRT_2, epsilon = 1e-8);
        }
    }

    #[rstest]
    fn sphrs_test<const L: usize>(
        #[values(
            degree::<0>(),
            degree::<1>(),
            degree::<2>(),
            degree::<3>(),
            degree::<4>(),
            degree::<5>(),
            degree::<6>(),
            degree::<7>(),
            degree::<8>(),
            degree::<9>(),
            degree::<10>()
            // Values of L>10 overflow sphrs's factorial implementation
        )]
        _d: Degree<L>,
        #[values(
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (inv3(), inv3(), inv3()),
            (0.6_f64.sin() * 0.3_f64.cos(), 0.6_f64.sin() * 0.3_f64.sin(), 0.6_f64.cos()),
        )]
        point: (f64, f64, f64),
    ) {
        let (x, y, z) = point;
        check_against_sphrs::<L>(x, y, z);
    }

    /// Completeness: |Y_l^0|² + 2·Σ_{m=1}^l |Y_l^m|² = (2l+1) / (4π).
    fn check_completeness<const L: usize>(x: f64, y: f64, z: f64) {
        let sh = spherical_harmonic::<L>(x, y, z);
        let mut sum = sh[0].norm_sqr();
        for m in 1..=L {
            sum += 2.0 * sh[m].norm_sqr();
        }
        let expected = (2 * L + 1) as f64 / (4.0 * PI);
        let abs_err = (sum - expected).abs();
        let rel_err = abs_err / expected;
        eprintln!(
            "L={:3}  abs_err={:.3e}  rel_err={:.3e}",
            L, abs_err, rel_err
        );
        assert!(
            abs_err < 1e-5,
            "completeness violated: abs_err={:.3e}",
            abs_err
        );
    }

    #[rstest]
    fn completeness_test<const L: usize>(
        #[values(
            degree::<0>(),  degree::<1>(),  degree::<2>(),  degree::<3>(),
            degree::<4>(),  degree::<5>(),  degree::<6>(),  degree::<7>(),
            degree::<8>(),  degree::<9>(),  degree::<10>(), degree::<11>(),
            degree::<12>(), degree::<13>(), degree::<14>(), degree::<15>(),
            degree::<16>(), degree::<17>(), degree::<18>(), degree::<19>(),
            degree::<20>(), degree::<21>(), degree::<22>(), degree::<23>(),
            degree::<24>(), degree::<25>(), degree::<26>(), degree::<27>(),
            degree::<28>(), degree::<29>(), degree::<30>(), degree::<31>(),
            degree::<32>(), degree::<33>(), degree::<34>(), degree::<35>(),
            degree::<36>(), degree::<37>(), degree::<38>(), degree::<39>(),
            degree::<40>(), degree::<41>(), degree::<42>(), degree::<43>(),
            degree::<44>(), degree::<45>(), degree::<46>(), degree::<47>(),
            degree::<48>(), degree::<49>(), degree::<50>(),
        )]
        _d: Degree<L>,
    ) {
        let x = 0.7_f64.sin() * 0.3_f64.cos();
        let y = 0.7_f64.sin() * 0.3_f64.sin();
        let z = 0.7_f64.cos();
        check_completeness::<L>(x, y, z);
    }
}

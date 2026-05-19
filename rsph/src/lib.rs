//! ...

use std::f64::consts::PI;
use std::ops::Index;

/// Real spherical harmonics Y_L^m for a single degree L.
///
/// Index with `[m]` to access Y_L^m for m = 0..L.
pub struct HarmonicOutput<const L: usize> {
    /// Y_L^0 (zonal harmonic).
    pub m0: f64,
    /// Y_L^{+m} for m = 1..L, stored at index m − 1.
    pub mp: [f64; L],
}

impl<const L: usize> Index<usize> for HarmonicOutput<L> {
    type Output = f64;

    #[inline]
    fn index(&self, m: usize) -> &f64 {
        match m {
            0 => &self.m0,
            n => &self.mp[n - 1],
        }
    }
}

/// Compute real spherical harmonics Y_L^m(x, y, z) for m = 0..L.
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
        f64::sqrt((2 * L + 1) as f64 * r / (2.0 * PI))
    };

    // h[k] = prefactor(L, k+1) * Q_l^{k+1}, h_0 = prefactor(L, 0) * Q_l^0
    // All values O(1) — the prefactor is folded into the recurrence to avoid
    // the large (2L-1)!! intermediate.
    let h_0;
    let mut h = [0.0; L];

    if L == 0 {
        h_0 = f64::sqrt(1.0 / (4.0 * PI));
    } else {
        h[L - 1] = norm_seed;
        if L == 1 {
            // m=0 step: h_0 = z * h[0]
            h_0 = z * h[0];
        } else {
            // Initial step: h[L-2] = z * sqrt(2L) * h[L-1]
            h[L - 2] = z * f64::sqrt(2.0 * L as f64) * h[L - 1];

            // General recurrence from m = L-2 down to m = 1
            for m in (1..L - 1).rev() {
                let denom = f64::sqrt(((L - m) * (L + m + 1)) as f64);
                let num = f64::sqrt(((L - m - 1) * (L + m + 2)) as f64);
                h[m - 1] = (2.0 * (m + 1) as f64 * z * h[m] - rxy2 * num * h[m + 1]) / denom;
            }

            // m = 0 step
            let denom = f64::sqrt((2 * L * (L + 1)) as f64);
            let num = f64::sqrt(((L - 1) * (L + 2)) as f64);
            h_0 = (2.0 * z * h[0] - rxy2 * num * h[1]) / denom;
        }
    }

    // Assemble output with fused azimuthal recurrence
    let mut out_pos = [0.0; L];

    if L > 0 {
        let mut cm = x;
        let mut sm = y;
        out_pos[0] = h[0] * cm;

        for m in 1..L {
            let prev_cm = cm;
            let prev_sm = sm;
            cm = prev_cm * x - prev_sm * y;
            sm = prev_cm * y + prev_sm * x;
            out_pos[m] = h[m] * cm;
        }
    }

    HarmonicOutput {
        m0: h_0,
        mp: out_pos,
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
        assert!(approx_eq(sh[0], 1.0 / (2.0 * f64::sqrt(PI)), 1e-12));
        assert_eq!(sh.mp.len(), 0);
    }

    #[test]
    fn l1_north_pole() {
        let sh = spherical_harmonic::<1>(0.0, 0.0, 1.0);
        let c = f64::sqrt(3.0 / (4.0 * PI));
        assert!(approx_eq(sh[0], c, 1e-12));
        assert!(approx_eq(sh[1], 0.0, 1e-12));
    }

    #[test]
    fn l1_x_axis() {
        let sh = spherical_harmonic::<1>(1.0, 0.0, 0.0);
        let c = f64::sqrt(3.0 / (4.0 * PI));
        assert!(approx_eq(sh[0], 0.0, 1e-12));
        assert!(approx_eq(sh[1], c, 1e-12));
    }

    #[test]
    fn l1_y_axis() {
        let sh = spherical_harmonic::<1>(0.0, 1.0, 0.0);
        assert!(approx_eq(sh[0], 0.0, 1e-12));
        assert!(approx_eq(sh[1], 0.0, 1e-12));
    }

    #[test]
    fn l2_finite() {
        let inv3 = 1.0 / 3.0_f64.sqrt();
        let sh = spherical_harmonic::<2>(inv3, inv3, inv3);
        assert_eq!(sh.mp.len(), 2);
        assert!(sh.m0.is_finite());
        for &v in &sh.mp {
            assert!(v.is_finite());
        }
    }

    fn check_against_sphrs<const L: usize>(x: f64, y: f64, z: f64) {
        use approxim::assert_abs_diff_eq;
        use sphrs::{Coordinates, RealSH, SHEval};

        let sh = spherical_harmonic::<L>(x, y, z);
        let p = Coordinates::cartesian(x, y, z);

        let expected_m0: f64 = RealSH::Spherical.eval(L as i64, 0, &p);
        assert_abs_diff_eq!(sh[0], expected_m0, epsilon = 1e-8);

        for m in 1..=L {
            let expected: f64 = RealSH::Spherical.eval(L as i64, m as i64, &p);
            assert_abs_diff_eq!(sh[m], expected, epsilon = 1e-8);
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

    /// Completeness check: sum_m |Y_l^m|^2 = (2L+1) / (4π).
    fn check_completeness<const L: usize>(x: f64, y: f64, z: f64) {
        let sh = spherical_harmonic::<L>(x, y, z);
        let mut sum = sh[0] * sh[0];
        let mut cm = x;
        let mut sm = y;
        for k in 0..L {
            let pos = sh.mp[k];
            let neg = pos * sm / cm;
            sum += pos * pos + neg * neg;
            let prev_cm = cm;
            let prev_sm = sm;
            cm = prev_cm * x - prev_sm * y;
            sm = prev_cm * y + prev_sm * x;
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

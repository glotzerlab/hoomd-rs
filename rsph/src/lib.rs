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

    HarmonicOutput { m0: h_0, mp: out_pos }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
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

    /// Compare against sphrs for a given L at a point on the unit sphere.
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

    fn test_points() -> Vec<(f64, f64, f64)> {
        let inv3 = 1.0 / 3.0_f64.sqrt();
        let th = 0.6_f64;
        let ph = 0.3_f64;
        vec![
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (inv3, inv3, inv3),
            (th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()),
        ]
    }

    macro_rules! sphrs_test {
        ($($name:ident, $l:literal);* $(;)?) => {
            $(
                #[test]
                fn $name() {
                    for &(x, y, z) in &test_points() {
                        check_against_sphrs::<$l>(x, y, z);
                    }
                }
            )*
        };
    }

    sphrs_test!(
        sphrs_l0, 0;
        sphrs_l1, 1;
        sphrs_l2, 2;
        sphrs_l3, 3;
        sphrs_l4, 4;
        sphrs_l5, 5;
        sphrs_l6, 6;
        sphrs_l7, 7;
        sphrs_l8, 8;
        sphrs_l9, 9;
        sphrs_l10, 10;
    );

    /// Completeness check: sum_m |Y_l^m|^2 = (2L+1) / (4π).
    /// Reconstructs negative-m terms from positive-m output via sm/cm ratio:
    ///   out_pos[k] = h[k]·cm[k],  neg[k] = h[k]·sm[k] = out_pos[k]·sm/cm
    #[test]
    fn completeness_sweep() {
        let theta = 0.7_f64;
        let phi = 0.3_f64;
        let x = theta.sin() * phi.cos();
        let y = theta.sin() * phi.sin();
        let z = theta.cos();

        let mut max_abs_err = 0.0_f64;
        let mut max_rel_err = 0.0_f64;
        let mut max_err_l = 0_usize;

        macro_rules! check_l {
            ($l:literal) => {{
                let sh = spherical_harmonic::<$l>(x, y, z);
                let mut sum = sh.m0 * sh.m0;
                let mut cm = x;
                let mut sm = y;
                for k in 0..$l {
                    let neg = sh.mp[k] * sm / cm;
                    sum += sh.mp[k] * sh.mp[k] + neg * neg;
                    let prev_cm = cm;
                    let prev_sm = sm;
                    cm = prev_cm * x - prev_sm * y;
                    sm = prev_cm * y + prev_sm * x;
                }
                let expected = (2 * $l + 1) as f64 / (4.0 * PI);
                let abs_err = (sum - expected).abs();
                let rel_err = abs_err / expected;
                eprintln!(
                    "L={:3}  abs_err={:.3e}  rel_err={:.3e}",
                    $l, abs_err, rel_err
                );
                if abs_err > max_abs_err {
                    max_abs_err = abs_err;
                    max_rel_err = rel_err;
                    max_err_l = $l;
                }
            }};
        }

        check_l!(0);
        check_l!(1);
        check_l!(2);
        check_l!(3);
        check_l!(4);
        check_l!(5);
        check_l!(6);
        check_l!(7);
        check_l!(8);
        check_l!(9);
        check_l!(10);
        check_l!(11);
        check_l!(12);
        check_l!(13);
        check_l!(14);
        check_l!(15);
        check_l!(16);
        check_l!(17);
        check_l!(18);
        check_l!(19);
        check_l!(20);
        check_l!(21);
        check_l!(22);
        check_l!(23);
        check_l!(24);
        check_l!(25);
        check_l!(26);
        check_l!(27);
        check_l!(28);
        check_l!(29);
        check_l!(30);
        check_l!(31);
        check_l!(32);
        check_l!(33);
        check_l!(34);
        check_l!(35);
        check_l!(36);
        check_l!(37);
        check_l!(38);
        check_l!(39);
        check_l!(40);
        check_l!(41);
        check_l!(42);
        check_l!(43);
        check_l!(44);
        check_l!(45);
        check_l!(46);
        check_l!(47);
        check_l!(48);
        check_l!(49);
        check_l!(50);

        eprintln!(
            "\nWorst: L={}  abs_err={:.3e}  rel_err={:.3e}",
            max_err_l, max_abs_err, max_rel_err
        );
        assert!(
            max_abs_err < 1e-5,
            "completeness violated at L={}: abs_err={:.3e}",
            max_err_l,
            max_abs_err
        );
    }
}

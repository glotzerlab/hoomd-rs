//! ...

use std::f64::consts::PI;

/// Compute all real spherical harmonics Y_l^m(x, y, z) for m = -l..=l.
///
/// The point (x, y, z) must lie on the unit sphere.
/// Returns a vector of length `2*l + 1` where `result[l + m]` = Y_l^m.
#[inline]
pub fn spherical_harmonic<const L: usize>(x: f64, y: f64, z: f64) -> (f64, [f64; L]) {
    let rxy2 = x * x + y * y;

    // Normalized seed: prefactor(L, L) * (2L-1)!! = sqrt((2L+1) * r(L) / (2π))
    // where r(L) = (2L-1)!! / (2^L * L!) stays O(1) via r(l) = r(l-1) * (2l-1)/(2l)
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
    let mut h_0 = 0.0;
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

    (h_0, out_pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn l0() {
        let (y0, y_pos) = spherical_harmonic::<0>(0.0, 0.0, 1.0);
        assert!(approx_eq(y0, 1.0 / (2.0 * f64::sqrt(PI)), 1e-12));
        assert_eq!(y_pos.len(), 0);
    }

    #[test]
    fn l1_north_pole() {
        let (y0, y_pos) = spherical_harmonic::<1>(0.0, 0.0, 1.0);
        let c = f64::sqrt(3.0 / (4.0 * PI));
        assert!(approx_eq(y0, c, 1e-12));
        assert!(approx_eq(y_pos[0], 0.0, 1e-12));
    }

    #[test]
    fn l1_x_axis() {
        let (y0, y_pos) = spherical_harmonic::<1>(1.0, 0.0, 0.0);
        let c = f64::sqrt(3.0 / (4.0 * PI));
        assert!(approx_eq(y0, 0.0, 1e-12));
        assert!(approx_eq(y_pos[0], c, 1e-12));
    }

    #[test]
    fn l1_y_axis() {
        let (y0, y_pos) = spherical_harmonic::<1>(0.0, 1.0, 0.0);
        assert!(approx_eq(y0, 0.0, 1e-12));
        assert!(approx_eq(y_pos[0], 0.0, 1e-12));
    }

    #[test]
    fn l2_finite() {
        let inv3 = 1.0 / 3.0_f64.sqrt();
        let (y0, y_pos) = spherical_harmonic::<2>(inv3, inv3, inv3);
        assert_eq!(y_pos.len(), 2);
        assert!(y0.is_finite());
        for &v in &y_pos {
            assert!(v.is_finite());
        }
    }

    /// Compare against sphrs for a given L at a point on the unit sphere.
    fn check_against_sphrs<const L: usize>(x: f64, y: f64, z: f64) {
        use approxim::assert_abs_diff_eq;
        use sphrs::{Coordinates, RealSH, SHEval};

        let (m0, mp) = spherical_harmonic::<L>(x, y, z);
        let p = Coordinates::cartesian(x, y, z);

        let expected_m0: f64 = RealSH::Spherical.eval(L as i64, 0, &p);
        assert_abs_diff_eq!(m0, expected_m0, epsilon = 1e-8);

        for m in 1..=L {
            let expected: f64 = RealSH::Spherical.eval(L as i64, m as i64, &p);
            assert_abs_diff_eq!(mp[m - 1], expected, epsilon = 1e-8);
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
}

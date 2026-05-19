//! ...

use std::f64::consts::PI;

/// Compute all real spherical harmonics Y_l^m(x, y, z) for m = -l..=l.
///
/// The point (x, y, z) must lie on the unit sphere.
/// Returns a vector of length `2*l + 1` where `result[l + m]` = Y_l^m.
#[inline]
pub fn spherical_harmonic<const L: usize>(x: f64, y: f64, z: f64) -> (f64, [f64; L]) {
    let rxy2 = x * x + y * y;

    // Azimuthal factors c_m, s_m
    let mut cm = [0.0; L];
    let mut sm = [0.0; L];

    if L > 0 {
        cm[0] = x;
        sm[0] = y;
        for m in 1..L {
            cm[m] = cm[m - 1] * x - sm[m - 1] * y;
            sm[m] = cm[m - 1] * y + sm[m - 1] * x;
        }
    }

    // Cartesian associated Legendre polynomials Q_l^m
    let mut q_0 = 0.0;
    let mut q = [0.0; L];

    if L == 0 {
        q_0 = seed(0);
    } else {
        q[L - 1] = seed(L);
        if L == 1 {
            q_0 = -z * q[0];
        } else {
            q[L - 2] = -z * q[L - 1];
        }

        // Loop from L-1 down to 0
        for m in (0..L.saturating_sub(1)).rev() {
            let coeff = -1.0 / (((L + m + 1) as f64) * ((L - m) as f64));
            let twomz = 2.0 * (m + 1) as f64 * z;

            if m == 0 {
                q_0 = coeff * (twomz * q[0] + rxy2 * q[1]);
            } else {
                q[m - 1] = coeff * (twomz * q[m] + rxy2 * q[m + 1]);
            }
        }
    }

    let p0 = f64::sqrt((2 * L + 1) as f64 / (4.0 * PI));
    let out_0 = p0 * q_0;
    let mut out_pos = [0.0; L];

    if L > 0 {
        // prefactor(L, 1) = -p0 * sqrt(2 / (L*(L+1)))
        let mut p = -p0 * f64::sqrt(2.0 / ((L * (L + 1)) as f64));
        out_pos[0] = p * q[0] * cm[0];

        for m in 1..L {
            // prefactor(L, m+1) = -prefactor(L, m) * sqrt(1 / ((L-m)*(L+m+1)))
            p = -p * f64::sqrt(1.0 / (((L - m) * (L + m + 1)) as f64));
            out_pos[m] = p * q[m] * cm[m];
        }
    }

    // Returns: (m=0, positive m terms)
    (out_0, out_pos)
}

/// Seed: Q_l^l = (-1)^l · (2l-1)!!
#[inline]
fn seed(l: usize) -> f64 {
    let sign = if l % 2 == 0 { 1.0 } else { -1.0 };
    let mut df = 1.0;
    for k in (1..=(2 * l)).step_by(2) {
        df *= k as f64;
    }
    sign * df
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

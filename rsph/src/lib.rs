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
            let c = -1.0 / (((L + m + 1) as f64) * ((L - m) as f64));
            let twomz = 2.0 * (m + 1) as f64 * z;

            if m == 0 {
                q_0 = c * (twomz * q[0] + rxy2 * q[1]);
            } else {
                q[m - 1] = c * (twomz * q[m] + rxy2 * q[m + 1]);
            }
        }
    }

    // Assemble real spherical harmonics (positive m only)
    let out_0 = prefactor(L, 0) * q_0;
    let mut out_pos = [0.0; L];

    for m in 1..=L {
        let p = prefactor(L, m);
        // out_pos represents the original out[l + m]
        out_pos[m - 1] = p * q[m - 1] * cm[m - 1];
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

/// Normalization prefactor p_l^m
#[inline]
fn prefactor(l: usize, m: usize) -> f64 {
    if m == 0 {
        f64::sqrt((2 * l + 1) as f64 / (4.0 * PI))
    } else {
        let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
        let ratio = factorial_ratio(l, m);
        sign * f64::sqrt((2 * l + 1) as f64 / (2.0 * PI) * ratio)
    }
}

/// (l-m)! / (l+m)!
#[inline]
fn factorial_ratio(l: usize, m: usize) -> f64 {
    let mut prod = 1.0;
    for k in (l - m + 1)..=(l + m) {
        prod *= k as f64;
    }
    1.0 / prod
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
}

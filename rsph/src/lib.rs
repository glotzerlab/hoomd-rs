use std::f64::consts::PI;

/// Compute all real spherical harmonics Y_l^m(x, y, z) for m = -l..=l.
///
/// The point (x, y, z) must lie on the unit sphere.
/// Returns a vector of length `2*l + 1` where `result[l + m]` = Y_l^m.
#[inline]
pub fn spherical_harmonic(l: usize, x: f64, y: f64, z: f64) -> Vec<f64> {
    let rxy2 = x * x + y * y;

    // Azimuthal factors c_m, s_m
    let mut cm = vec![0.0; l + 1];
    let mut sm = vec![0.0; l + 1];
    cm[0] = 1.0;
    for m in 1..=l {
        cm[m] = cm[m - 1] * x - sm[m - 1] * y;
        sm[m] = cm[m - 1] * y + sm[m - 1] * x;
    }

    // Cartesian associated Legendre polynomials Q_l^m
    let mut q = vec![0.0; l + 1];
    q[l] = seed(l);
    if l > 0 {
        q[l - 1] = -z * q[l];
    }
    for m in (0..l.saturating_sub(1)).rev() {
        let c = -1.0 / (((l + m + 1) as f64) * ((l - m) as f64));
        let twomz = 2.0 * (m + 1) as f64 * z;
        q[m] = c * (twomz * q[m + 1] + rxy2 * q[m + 2]);
    }

    // Assemble real spherical harmonics
    let mut out = vec![0.0; 2 * l + 1];
    out[l] = prefactor(l, 0) * q[0];
    for m in 1..=l {
        let p = prefactor(l, m);
        out[l + m] = p * q[m] * cm[m];
        out[l - m] = p * q[m] * sm[m];
    }
    out
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
        let y = spherical_harmonic(0, 0.0, 0.0, 1.0);
        assert_eq!(y.len(), 1);
        assert!(approx_eq(y[0], 1.0 / (2.0 * f64::sqrt(PI)), 1e-12));
    }

    #[test]
    fn l1_north_pole() {
        let y = spherical_harmonic(1, 0.0, 0.0, 1.0);
        let c = f64::sqrt(3.0 / (4.0 * PI));
        assert!(approx_eq(y[0], 0.0, 1e-12));
        assert!(approx_eq(y[1], c, 1e-12));
        assert!(approx_eq(y[2], 0.0, 1e-12));
    }

    #[test]
    fn l1_x_axis() {
        let y = spherical_harmonic(1, 1.0, 0.0, 0.0);
        let c = f64::sqrt(3.0 / (4.0 * PI));
        assert!(approx_eq(y[0], 0.0, 1e-12));
        assert!(approx_eq(y[1], 0.0, 1e-12));
        assert!(approx_eq(y[2], c, 1e-12));
    }

    #[test]
    fn l1_y_axis() {
        let y = spherical_harmonic(1, 0.0, 1.0, 0.0);
        let c = f64::sqrt(3.0 / (4.0 * PI));
        assert!(approx_eq(y[0], c, 1e-12));
        assert!(approx_eq(y[1], 0.0, 1e-12));
        assert!(approx_eq(y[2], 0.0, 1e-12));
    }

    #[test]
    fn l2_finite() {
        let inv3 = 1.0 / 3.0_f64.sqrt();
        let sh = spherical_harmonic(2, inv3, inv3, inv3);
        assert_eq!(sh.len(), 5);
        for &v in &sh {
            assert!(v.is_finite());
        }
    }
}

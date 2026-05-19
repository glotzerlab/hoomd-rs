//! Tools for evaluating complex spherical harmonics in rust.
//!
//! This library uses a recurrence relation to evaluate spherical harmonics of a
//! particular azimuthal quantum number `l` and all positive magnetic quantum numbers
//! `m=0..=L`. The approach taken is much faster than more general recurrences, which
//! typically attempt to evaluate all values of `l` up to the target. When computing
//! Steinhardt order parameters or similar algorithms, this code is much faster than
//! alternatives, with competetive numerical stability even out to large values of `l`.
//!
//! # Example
//! ```
//! use rsph::SphericalHarmonic;
//! use num_complex::Complex64;
//! use hoomd_vector::{Cartesian, InnerProduct};
//! use approxim::assert_abs_diff_eq;
//! use std::f64::consts::PI;
//!
//! // Initialize the SphericalHarmonic container, which can be reused
//! // to compute Y_6^m at a large number of points.
//! let y_6 = SphericalHarmonic::<6>::new();
//!
//! // Values of m in 0..=L are returned as a HarmonicOutput<L> container, which behaves
//! // like a [f64; L+1] array.
//! let sh = y_6.eval([1.0 / f64::sqrt(3.0), 1.0 / f64::sqrt(3.0), 1.0 / f64::sqrt(3.0)]);
//! assert_eq!(sh.len(), 6+1);
//!
//! // Zonal harmonic (m=0) is always purely real
//! assert_eq!(sh[0].im, 0.0);
//!
//! // Y_6^0 = sqrt(13/(4pi)) * P_6(1/sqrt(3)) = sqrt(13/(4pi)) * 2/9
//! let expected_m0 = 2.0 * f64::sqrt(13.0 / (4.0 * PI)) / 9.0;
//! assert_abs_diff_eq!(sh[0].re, expected_m0, epsilon = 1e-15);
//!
//! /// Implement the Steinhardt order parameter q6.
//! fn q6(bonds: &[Cartesian<3>]) -> f64 {
//!     let mut accum = [Complex64::ZERO; 7];
//!     let y6 = SphericalHarmonic::<6>::new();
//!
//!     for &bond in bonds {
//!         let projected_onto_sphere = bond / bond.norm();
//!         let qlmi = y6.eval(projected_onto_sphere.coordinates);
//!         for m in 0..7 { accum[m] += qlmi[m]; }
//!     }
//!
//!     // We multiply the `m>0` components by two to account for `-m` contributions.
//!     let sum_sq = accum[0].norm_sqr()
//!         + 2.0 * accum[1..].iter().map(Complex64::norm_sqr).sum::<f64>();
//!     let n = bonds.len() as f64;
//!
//!     (4.0 * PI / 13.0 * sum_sq).sqrt() / n
//! }
//!
//! // FCC nearest neighbors: permutations of (±1, ±1, 0)
//! let fcc_bonds: Vec<Cartesian<3>> = [
//!     [-1.0, -1.0,  0.0], [-1.0,  1.0,  0.0], [1.0, -1.0,  0.0], [1.0,  1.0,  0.0],
//!     [-1.0,  0.0, -1.0], [-1.0,  0.0,  1.0], [1.0,  0.0, -1.0], [1.0,  0.0,  1.0],
//!     [ 0.0, -1.0, -1.0], [ 0.0, -1.0,  1.0], [0.0,  1.0, -1.0], [0.0,  1.0,  1.0],
//! ].map(Cartesian::<3>::from).to_vec();
//! assert_abs_diff_eq!(q6(&fcc_bonds), 0.57452416, epsilon = 1e-6);
//! ```

use num_complex::Complex64;
use std::{
    f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2},
    fmt,
    ops::Index,
};

/// Precomputed coefficients for evaluating complex spherical harmonics of degree L.
///
/// Once a [`SphericalHarmonic`] has been created with [`new`](Self::new),
/// [`eval`](Self::eval) can be called to rapidly evaluate the harmonic at a set of
/// points in three-dimensional space.
///
/// ```
/// use rsph::SphericalHarmonic;
///
/// let sh = SphericalHarmonic::<2>::new();
/// let out = sh.eval([0.0, 0.0, 1.0]);
///
/// // m=0 (zonal harmonic) is always purely real
/// assert_eq!(out[0].im, 0.0);
/// assert_eq!(out.len(), 3);
/// ```
#[derive(Copy, Clone, Debug)]
pub struct SphericalHarmonic<const L: usize> {
    /// Initial value for the recurrence.
    normalized_recurrence_seed: f64,
    /// Coefficient of the `z * h[m]` term in the Legendre recurrence.
    z_coeff: [f64; L],
    /// Coefficient of the `rxy2 * h[m+1]` term in the Legendre recurrence.
    rxy_coeff: [f64; L],
}

impl<const L: usize> SphericalHarmonic<L> {
    /// Precompute L-dependent coefficients.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        let normalized_recurrence_seed = {
            let mut r = 1.0;
            for k in 1..=L {
                r *= (2 * k - 1) as f64 / (2 * k) as f64;
            }
            f64::sqrt((2 * L + 1) as f64 * r / (2.0 * PI)) * FRAC_1_SQRT_2
        };

        let mut z_coeff = [0.0; L];
        let mut rxy_coeff = [0.0; L];

        let sqrt_2l = f64::sqrt(2.0 * L as f64);
        let mut carry = sqrt_2l;

        for m in (1..L).rev() {
            let denom = f64::sqrt(((L - m) * (L + m + 1)) as f64);
            z_coeff[m] = 2.0 * (m + 1) as f64 / denom;
            rxy_coeff[m] = carry / denom;
            carry = denom;
        }

        // m=0 step: √2 fused into coefficients
        if L > 0 {
            let denom_0 = f64::sqrt((2 * L * (L + 1)) as f64);
            z_coeff[0] = 2.0 * SQRT_2 / denom_0;
            rxy_coeff[0] = carry * SQRT_2 / denom_0;
        }

        Self {
            normalized_recurrence_seed,
            z_coeff,
            rxy_coeff,
        }
    }

    /// Evaluate `Y_L^m` for m = 0..=L at a point on the unit sphere.
    #[must_use]
    #[inline]
    pub fn eval(&self, point: [f64; 3]) -> HarmonicOutput<L> {
        let [x, y, z] = point;
        let rxy2 = x * x + y * y;

        let h_0;
        let mut h = [0.0; L];

        if L == 0 {
            h_0 = f64::sqrt(1.0 / (4.0 * PI));
        } else {
            h[L - 1] = self.normalized_recurrence_seed;
            let mut h_plus1 = 0.0;

            for m in (1..L).rev() {
                h[m - 1] = self.z_coeff[m] * z * h[m] - rxy2 * self.rxy_coeff[m] * h_plus1;
                h_plus1 = h[m];
            }

            h_0 = self.z_coeff[0] * z * h[0] - rxy2 * self.rxy_coeff[0] * h_plus1;
        }

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
}

impl<const L: usize> Default for SphericalHarmonic<L> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Complex spherical harmonics `Y_L^m` for a single degree L.
///
/// Index with `[m]` to access `Y_L^m` for m = 0..=L.
/// The m = 0 term is always purely real.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct HarmonicOutput<const L: usize> {
    /// `Y_L^0` (zonal harmonic, always real).
    m0: Complex64,
    /// `Y_L^m` for m = 1..=L, stored at index m − 1.
    mp: [Complex64; L],
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

impl<const L: usize> IntoIterator for HarmonicOutput<L> {
    type Item = Complex64;
    type IntoIter = std::iter::Chain<std::iter::Once<Complex64>, std::array::IntoIter<Complex64, L>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self.m0).chain(self.mp)
    }
}

impl<const L: usize> fmt::Display for HarmonicOutput<L> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for m in 0..=L {
            writeln!(f, "  {:+.12}{:+.12}i,  // m={m}", self[m].re, self[m].im)?;
        }
        write!(f, "]")
    }
}

impl<const L: usize> HarmonicOutput<L> {
    /// Get the harmonic degree L of the container.
    #[inline]
    #[must_use]
    pub const fn l(&self) -> usize {
        L
    }
    /// Iterate over all `L + 1` values, starting with `Y_L^0`.
    ///
    /// Negative-m values are recovered via [`neg_m`](Self::neg_m):
    ///
    /// ```
    /// use approxim::assert_abs_diff_eq;
    /// use num_complex::Complex64;
    /// use rsph::SphericalHarmonic;
    ///
    /// let out = SphericalHarmonic::<4>::new().eval([0.0, 0.0, 1.0]);
    ///
    /// // Build Y_4^m for m = -4..=4
    /// let full: Vec<Complex64> = (1..=4)
    ///     .rev()
    ///     .map(|m| out.neg_m(m))
    ///     .chain(out.iter())
    ///     .collect();
    /// assert_eq!(full.len(), 9);
    /// assert_abs_diff_eq!(full[0].conj(), full[2 * out.l()])
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = Complex64> + '_ {
        std::iter::once(self.m0).chain(self.mp.iter().copied())
    }
    /// Compute `Y_l^{-m} = (-1)^m · conj(Y_l^m)`.
    #[inline]
    #[must_use]
    pub fn neg_m(&self, m: usize) -> Complex64 {
        let val = self[m];
        if m.is_multiple_of(2) {
            val.conj()
        } else {
            -val.conj()
        }
    }
    /// The length of the container, equal to `L + 1`.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        L + 1
    }
    /// Check if the container is empty. This will always be false.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_abs_diff_eq;
    use rstest::rstest;
    use std::marker::PhantomData;

    type Degree<const L: usize> = PhantomData<[(); L]>;
    fn degree<const L: usize>() -> Degree<L> {
        Degree::default()
    }

    #[test]
    fn l0() {
        let sh = SphericalHarmonic::<0>::new();
        let out = sh.eval([0.0, 0.0, 1.0]);
        let expected = 1.0 / (2.0 * f64::sqrt(PI));
        assert_abs_diff_eq!(out[0].re, expected, epsilon = 1e-12);
        assert_abs_diff_eq!(out[0].im, 0.0, epsilon = 1e-12);
        assert_eq!(out.mp.len(), 0);
    }

    #[test]
    fn l1_north_pole() {
        let sh = SphericalHarmonic::<1>::new();
        let out = sh.eval([0.0, 0.0, 1.0]);
        let c = f64::sqrt(3.0 / (4.0 * PI));
        assert_abs_diff_eq!(out[0].re, c, epsilon = 1e-12);
        assert_abs_diff_eq!(out[0].im, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1].re, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1].im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn l1_x_axis() {
        let sh = SphericalHarmonic::<1>::new();
        let out = sh.eval([1.0, 0.0, 0.0]);
        let c = f64::sqrt(3.0 / (8.0 * PI));
        assert_abs_diff_eq!(out[0].re, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[0].im, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1].re, c, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1].im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn l1_y_axis() {
        let sh = SphericalHarmonic::<1>::new();
        let out = sh.eval([0.0, 1.0, 0.0]);
        let c = f64::sqrt(3.0 / (8.0 * PI));
        assert_abs_diff_eq!(out[0].re, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[0].im, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1].re, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1].im, c, epsilon = 1e-12);
    }

    #[test]
    fn l2_finite() {
        let inv3 = 3.0_f64.sqrt().recip();
        let sh = SphericalHarmonic::<2>::new();
        let out = sh.eval([inv3, inv3, inv3]);
        assert_eq!(out.mp.len(), 2);
        assert!(out.m0.re.is_finite());
        assert!(out.m0.im.is_finite());
        for v in &out.mp {
            assert!(v.re.is_finite());
            assert!(v.im.is_finite());
        }
    }

    #[test]
    fn iter_matches_index() {
        let sh = SphericalHarmonic::<4>::new();
        let out = sh.eval([0.6, 0.3, 0.4]);
        let values: Vec<_> = out.iter().collect();
        assert_eq!(values.len(), 5);
        for m in 0..=4 {
            assert_abs_diff_eq!(values[m].re, out[m].re, epsilon = 1e-15);
            assert_abs_diff_eq!(values[m].im, out[m].im, epsilon = 1e-15);
        }
    }

    #[test]
    fn display() {
        let sh = SphericalHarmonic::<4>::new();
        let out = sh.eval([0.6, 0.3, 0.4]);
        println!("{out}");
    }

    /// Validate against sphrs via Y_l^m = (S_l^{+m} + i·S_l^{-m}) / √2.
    fn check_against_sphrs<const L: usize>(point: [f64; 3]) {
        use sphrs::{Coordinates, RealSH, SHEval};
        let l = i64::try_from(L).expect("L would overflow i64");

        let sh = SphericalHarmonic::<L>::new();
        let out = sh.eval(point);
        let [x, y, z] = point;
        let coords = Coordinates::cartesian(x, y, z);

        let expected_m0: f64 = RealSH::Spherical.eval(l, 0, &coords);
        assert_abs_diff_eq!(out[0].re, expected_m0, epsilon = 1e-8);
        assert_abs_diff_eq!(out[0].im, 0.0, epsilon = 1e-8);

        for m in 1..=L {
            let m_i64 = i64::try_from(m).expect("m would overflow i64");
            let s_pos: f64 = RealSH::Spherical.eval(l, m_i64, &coords);
            let s_neg: f64 = RealSH::Spherical.eval(l, -m_i64, &coords);
            assert_abs_diff_eq!(out[m].re, s_pos * FRAC_1_SQRT_2, epsilon = 1e-8);
            assert_abs_diff_eq!(out[m].im, s_neg * FRAC_1_SQRT_2, epsilon = 1e-8);
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
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0_f64.sqrt().recip(); 3],
            [0.6_f64.sin() * 0.3_f64.cos(), 0.6_f64.sin() * 0.3_f64.sin(), 0.6_f64.cos()],
        )]
        point: [f64; 3],
    ) {
        check_against_sphrs::<L>(point);
    }

    /// Completeness: |Y_l^0|² + 2·Σ_{m=1}^l |Y_l^m|² = (2l+1) / (4π).
    fn check_completeness<const L: usize>(point: [f64; 3]) {
        let sh = SphericalHarmonic::<L>::new();
        let out = sh.eval(point);
        let mut sum = out[0].norm_sqr();
        for m in 1..=L {
            sum += 2.0 * out[m].norm_sqr();
        }
        let expected = (2 * L + 1) as f64 / (4.0 * PI);
        assert_abs_diff_eq!(sum, expected, epsilon = 1e-10);
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
        let point = [
            0.7_f64.sin() * 0.3_f64.cos(),
            0.7_f64.sin() * 0.3_f64.sin(),
            0.7_f64.cos(),
        ];
        check_completeness::<L>(point);
    }
}

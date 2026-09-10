// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Tools for evaluating complex spherical harmonics in rust.
//!
//! This library uses a recurrence relation to evaluate spherical harmonics of a
//! particular azimuthal quantum number `l` and all positive magnetic quantum numbers
//! `m=0..=L`. The approach taken is much faster than more general recurrences, which
//! typically attempt to evaluate all values of `l` up to the target. When computing
//! Steinhardt order parameters or similar algorithms, this code is much faster than
//! alternatives, with good numerical stability even out to large values of `l`.

use hoomd_vector::{Cartesian, Unit};
use num_complex::Complex;
use std::{
    array, f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2}, fmt, ops::{Add, AddAssign, Div, Index, Mul},
};

/// The spherical harmonic of degree `L`.
///
/// See Wolfram Mathworld for a detailed description of the [Spherical harmonics].
///
/// Construct a [`SphericalHarmonic`] with a given `L`, then call [`evaluate`] to compute
/// $` Y_L^m(\vec{r}) `$ for $` m \in [0,L] `$. Use the same [`SphericalHarmonic`] for many
/// calls to [`evaluate`] as [`new`] is computationally expensive.
///
/// [Spherical harmonics]: https://mathworld.wolfram.com/SphericalHarmonic.html
/// [`evaluate`]: Self::evaluate
/// [`new`]: Self::new
///
/// # Example
///
/// ```
/// use approxim::{assert_abs_diff_eq, assert_relative_eq};
/// use hoomd_order::math::SphericalHarmonic;
/// use hoomd_vector::Cartesian;
/// use num_complex::Complex;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let spherical_harmonic = SphericalHarmonic::<2>::new();
/// let output = spherical_harmonic.evaluate(&[0.0, 0.0, 1.0].try_into()?);
///
/// assert_eq!(output.len(), 3);
/// assert_relative_eq!(output[0], Complex::new(0.6307831305050400, 0.0));
/// assert_abs_diff_eq!(output[1], Complex::new(0.0, 0.0));
/// assert_abs_diff_eq!(output[2], Complex::new(0.0, 0.0));
/// # Ok(())
/// # }
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
    /// Construct the spherical harmonic of degree `L`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_order::math::SphericalHarmonic;
    ///
    /// let spherical_harmonic = SphericalHarmonic::<2>::new();
    /// ```
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

    /// Evaluate $` Y_L^m(\vec{r}) `$ for for $` m \in [0,L] `$ at a point on the unit sphere.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::{assert_abs_diff_eq, assert_relative_eq};
    /// use hoomd_order::math::SphericalHarmonic;
    /// use hoomd_vector::Cartesian;
    /// use num_complex::Complex;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let spherical_harmonic = SphericalHarmonic::<2>::new();
    /// let output = spherical_harmonic.evaluate(&[0.0, 0.0, 1.0].try_into()?);
    ///
    /// assert_eq!(output.len(), 3);
    /// assert_relative_eq!(output[0], Complex::new(0.6307831305050400, 0.0));
    /// assert_abs_diff_eq!(output[1], Complex::new(0.0, 0.0));
    /// assert_abs_diff_eq!(output[2], Complex::new(0.0, 0.0));
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn evaluate(&self, point: &Unit<Cartesian<3>>) -> SphericalHarmonicOutputs<L> {
        let [x, y, z] = point.get().coordinates;
        let rxy2 = x * x + y * y;

        let mut h = [0.0; L];

        let h_0 = if L == 0 {
            f64::sqrt(1.0 / (4.0 * PI))
        } else {
            h[L - 1] = self.normalized_recurrence_seed;
            let mut h_plus1 = 0.0;

            for m in (1..L).rev() {
                h[m - 1] = self.z_coeff[m] * z * h[m] - rxy2 * self.rxy_coeff[m] * h_plus1;
                h_plus1 = h[m];
            }

            self.z_coeff[0] * z * h[0] - rxy2 * self.rxy_coeff[0] * h_plus1
        };

        let mut result = [Complex::ZERO; L];

        if L > 0 {
            let mut cm = x;
            let mut sm = y;
            result[0] = Complex::new(h[0] * cm, h[0] * sm);

            for m in 1..L {
                let prev_cm = cm;
                let prev_sm = sm;
                cm = prev_cm * x - prev_sm * y;
                sm = prev_cm * y + prev_sm * x;
                result[m] = Complex::new(h[m] * cm, h[m] * sm);
            }
        }

        SphericalHarmonicOutputs {
            m0: Complex::new(h_0, 0.0),
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

/// Spherical harmonic values for $` m \in [0,L] `$.
///
/// Call [`SphericalHarmonic::evaluate`] to obtain the resulting value as
/// [`SphericalHarmonicOutputs`]. The output contains the values at
/// all non-negative values of *m*.
///
/// [`SphericalHarmonicOutputs`] add element-wise and can be multiplied and
/// divided by scalars.
///
/// # Examples
///
/// Directly index an output value:
///
/// ```
/// use hoomd_order::math::SphericalHarmonic;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let spherical_harmonic = SphericalHarmonic::<2>::new();
/// let output = spherical_harmonic.evaluate(&[0.0, 0.0, 1.0].try_into()?);
///
/// let total = output[0] + output[1] + output[2];
///
/// # Ok(())
/// # }
/// ```
///
/// Iterate over output values:
///
/// ```
/// use hoomd_order::math::SphericalHarmonic;
/// use hoomd_vector::Cartesian;
/// use num_complex::Complex;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let spherical_harmonic = SphericalHarmonic::<2>::new();
/// let output = spherical_harmonic.evaluate(&[0.0, 0.0, 1.0].try_into()?);
///
/// let total: Complex<f64> = output.iter().sum();
///
/// # Ok(())
/// # }
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SphericalHarmonicOutputs<const L: usize> {
    /// `Y_L^0` (zonal harmonic, always real).
    m0: Complex<f64>,
    /// `Y_L^m` for m = 1..=L, stored at index m − 1.
    mp: [Complex<f64>; L],
}

impl<const L: usize> Index<usize> for SphericalHarmonicOutputs<L> {
    type Output = Complex<f64>;

    #[inline]
    fn index(&self, index: usize) -> &Complex<f64> {
        match index {
            0 => &self.m0,
            n => &self.mp[n - 1],
        }
    }
}

impl<const L: usize> IntoIterator for SphericalHarmonicOutputs<L> {
    type Item = Complex<f64>;
    type IntoIter =
        std::iter::Chain<std::iter::Once<Complex<f64>>, std::array::IntoIter<Complex<f64>, L>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self.m0).chain(self.mp)
    }
}

impl<const L: usize> fmt::Display for SphericalHarmonicOutputs<L> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;
        for m in 0..=L {
            writeln!(f, "  {:+.12}{:+.12}i,  // m={m}", self[m].re, self[m].im)?;
        }
        write!(f, "]")
    }
}

impl<const L: usize> SphericalHarmonicOutputs<L> {
    /// Iterate over the values $` Y_L^0, Y_L^1, \ldots, Y_L^L `$.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_order::math::SphericalHarmonic;
    /// use hoomd_vector::Cartesian;
    /// use num_complex::Complex;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let spherical_harmonic = SphericalHarmonic::<2>::new();
    /// let output = spherical_harmonic.evaluate(&[0.0, 0.0, 1.0].try_into()?);
    ///
    /// let total: Complex<f64> = output.iter().sum();
    ///
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = Complex<f64>> + '_ {
        std::iter::once(self.m0).chain(self.mp.iter().copied())
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

impl<const L: usize> Default for SphericalHarmonicOutputs<L> {
    /// The default [`SphericalHarmonicOutputs`] contains all 0's
    ///
    /// # Example
    ///
    /// ```
    /// use num_complex::Complex;
    /// use hoomd_order::math::SphericalHarmonicOutputs;
    ///
    /// let default = SphericalHarmonicOutputs::<3>::default();
    /// assert_eq!(default[0], Complex::new(0.0, 0.0));
    /// assert_eq!(default[1], Complex::new(0.0, 0.0));
    /// assert_eq!(default[2], Complex::new(0.0, 0.0));
    /// assert_eq!(default[3], Complex::new(0.0, 0.0));
    /// ```
    #[inline(always)]
    fn default() -> Self {
        Self { m0: Complex::default(), mp: [Complex::default(); L] }
    }
}

impl<const L: usize> AddAssign for SphericalHarmonicOutputs<L> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.m0 += rhs.m0;
        for i in 0..L {
            self.mp[i] += rhs.mp[i];
        }
    }
}

impl<const L: usize> Add for SphericalHarmonicOutputs<L> {
    type Output = SphericalHarmonicOutputs<L>;

    #[inline(always)]
    fn add(self, rhs: SphericalHarmonicOutputs<L>) -> Self::Output {
        Self {
            m0: self.m0 + rhs.m0,
            mp: array::from_fn(|i| self.mp[i] + rhs.mp[i]),
        }
    }
}

impl<const L: usize> Div<f64> for SphericalHarmonicOutputs<L> {
    type Output = SphericalHarmonicOutputs<L>;

    #[inline(always)]
    fn div(self, rhs: f64) -> Self::Output {
        Self {
            m0: self.m0 / rhs,
            mp: array::from_fn(|i| self.mp[i] / rhs),
        }
    }
}

impl<const L: usize> Mul<f64> for SphericalHarmonicOutputs<L> {
    type Output = SphericalHarmonicOutputs<L>;

    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            m0: self.m0 * rhs,
            mp: array::from_fn(|i| self.mp[i] * rhs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_abs_diff_eq;
    use num_complex::Complex;
    use rstest::rstest;
    use std::marker::PhantomData;

    type Degree<const L: usize> = PhantomData<[(); L]>;
    fn degree<const L: usize>() -> Degree<L> {
        Degree::default()
    }

    #[test]
    fn l0() -> Result<(), hoomd_vector::Error> {
        let sh = SphericalHarmonic::<0>::new();
        let out = sh.evaluate(&[0.0, 0.0, 1.0].try_into()?);
        let expected = 1.0 / (2.0 * f64::sqrt(PI));
        assert_abs_diff_eq!(out[0], Complex::new(expected, 0.0f64), epsilon = 1e-12);
        assert_eq!(out.mp.len(), 0);
        Ok(())
    }

    #[test]
    fn l1_north_pole() -> Result<(), hoomd_vector::Error> {
        let sh = SphericalHarmonic::<1>::new();
        let out = sh.evaluate(&[0.0, 0.0, 1.0].try_into()?);
        let c = f64::sqrt(3.0 / (4.0 * PI));
        assert_abs_diff_eq!(out[0], Complex::new(c, 0.0), epsilon = 1e-12);
        assert_abs_diff_eq!(out[1], Complex::ZERO, epsilon = 1e-12);
        Ok(())
    }

    #[test]
    fn l1_x_axis() -> Result<(), hoomd_vector::Error> {
        let sh = SphericalHarmonic::<1>::new();
        let out = sh.evaluate(&[1.0, 0.0, 0.0].try_into()?);
        let c = f64::sqrt(3.0 / (8.0 * PI));
        assert_abs_diff_eq!(out[0], Complex::ZERO, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1], Complex::new(c, 0.0), epsilon = 1e-12);
        Ok(())
    }

    #[test]
    fn l1_y_axis() -> Result<(), hoomd_vector::Error> {
        let sh = SphericalHarmonic::<1>::new();
        let out = sh.evaluate(&[0.0, 1.0, 0.0].try_into()?);
        let c = f64::sqrt(3.0 / (8.0 * PI));
        assert_abs_diff_eq!(out[0], Complex::ZERO, epsilon = 1e-12);
        assert_abs_diff_eq!(out[1], Complex::new(0.0, c), epsilon = 1e-12);
        Ok(())
    }

    #[test]
    fn l2_finite() -> Result<(), hoomd_vector::Error> {
        let inv3 = 3.0_f64.sqrt().recip();
        let sh = SphericalHarmonic::<2>::new();
        let out = sh.evaluate(&[inv3, inv3, inv3].try_into()?);
        assert_eq!(out.mp.len(), 2);
        assert!(out.m0.re.is_finite());
        assert!(out.m0.im.is_finite());
        for v in &out.mp {
            assert!(v.re.is_finite());
            assert!(v.im.is_finite());
        }
        Ok(())
    }

    #[test]
    fn into_iter_matches_index() -> Result<(), hoomd_vector::Error> {
        let sh = SphericalHarmonic::<4>::new();
        let reference = sh.evaluate(&[0.6, 0.3, 0.4].try_into()?);
        let out = sh.evaluate(&[0.6, 0.3, 0.4].try_into()?);
        for (m, val) in out.into_iter().enumerate() {
            assert_abs_diff_eq!(val, reference[m], epsilon = 1e-15);
        }
        Ok(())
    }

    #[test]
    fn iter_matches_index() -> Result<(), hoomd_vector::Error> {
        let sh = SphericalHarmonic::<4>::new();
        let out = sh.evaluate(&[0.6, 0.3, 0.4].try_into()?);
        let values: Vec<_> = out.iter().collect();
        assert_eq!(values.len(), 5);
        for m in 0..=4 {
            assert_abs_diff_eq!(values[m], out[m], epsilon = 1e-15);
        }
        Ok(())
    }

    /// Validate against sphrs via `Y_l^m` = (`S_l^{+m`} + i·S_l^{-m}) / √2.
    fn check_against_sphrs<const L: usize>(point: [f64; 3]) -> Result<(), hoomd_vector::Error> {
        use sphrs::{Coordinates, RealSH, SHEval};
        let l = i64::try_from(L).expect("L should not overflow i64");

        let sh = SphericalHarmonic::<L>::new();
        let out = sh.evaluate(&point.try_into()?);
        let [x, y, z] = point;
        let coords = Coordinates::cartesian(x, y, z);

        let expected_m0: f64 = RealSH::Spherical.eval(l, 0, &coords);
        assert_abs_diff_eq!(out[0], Complex::new(expected_m0, 0.0), epsilon = 1e-8);

        for m in 1..=L {
            let m_i64 = i64::try_from(m).expect("m should not overflow i64");
            let s_pos: f64 = RealSH::Spherical.eval(l, m_i64, &coords);
            let s_neg: f64 = RealSH::Spherical.eval(l, -m_i64, &coords);
            assert_abs_diff_eq!(
                out[m],
                Complex::new(s_pos * FRAC_1_SQRT_2, s_neg * FRAC_1_SQRT_2),
                epsilon = 1e-8
            );
        }
        Ok(())
    }

    #[rstest]
    #[expect(
        clippy::used_underscore_binding,
        reason = "Required for const generic parameterization."
    )]
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
        check_against_sphrs::<L>(point).unwrap();
    }

    /// Completeness: |`Y_l^0|²` + 2·Σ_{m=1}^l |`Y_l^m|²` = (2l+1) / (4π).
    fn check_completeness<const L: usize>(point: [f64; 3]) -> Result<(), hoomd_vector::Error> {
        let sh = SphericalHarmonic::<L>::new();
        let out = sh.evaluate(&point.try_into()?);
        let mut sum = out[0].norm_sqr();
        for m in 1..=L {
            sum += 2.0 * out[m].norm_sqr();
        }
        let expected = (2 * L + 1) as f64 / (4.0 * PI);
        assert_abs_diff_eq!(sum, expected, epsilon = 1e-10);
        Ok(())
    }

    #[rstest]
    #[expect(
        clippy::used_underscore_binding,
        reason = "Required for const generic parameterization."
    )]
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
        check_completeness::<L>(point).unwrap();
    }

    #[test]
    fn add_outputs() {
        let a = SphericalHarmonicOutputs {
            m0: Complex::new(1.0, 0.0),
            mp: [Complex::new(2.0, 3.0), Complex::new(-4.0, -5.0), Complex::new(1.0, 0.5)],
        };
        let b = SphericalHarmonicOutputs {
            m0: Complex::new(2.0, 0.0),
            mp: [Complex::new(3.0, -4.0), Complex::new(6.0, -3.0), Complex::new(5.0, 8.0)],
        };

        let mut c = a;
        c += b;
        assert_eq!(c[0], Complex::new(3.0, 0.0));
        assert_eq!(c[1], Complex::new(5.0, -1.0));
        assert_eq!(c[2], Complex::new(2.0, -8.0));
        assert_eq!(c[3], Complex::new(6.0, 8.5));

        let c = a + b;
        assert_eq!(c[0], Complex::new(3.0, 0.0));
        assert_eq!(c[1], Complex::new(5.0, -1.0));
        assert_eq!(c[2], Complex::new(2.0, -8.0));
        assert_eq!(c[3], Complex::new(6.0, 8.5));
    }

    #[test]
    fn div_outputs() {
        let a = SphericalHarmonicOutputs {
            m0: Complex::new(1.0, 0.0),
            mp: [Complex::new(2.0, 4.0), Complex::new(-4.0, -6.0), Complex::new(1.0, 0.5)],
        };

        let b = a / 2.0;
        assert_eq!(b[0], Complex::new(0.5, 0.0));
        assert_eq!(b[1], Complex::new(1.0, 2.0));
        assert_eq!(b[2], Complex::new(-2.0, -3.0));
        assert_eq!(b[3], Complex::new(0.5, 0.25));
    }

    #[test]
    fn mul_outputs() {
        let a = SphericalHarmonicOutputs {
            m0: Complex::new(1.0, 0.0),
            mp: [Complex::new(2.0, 4.0), Complex::new(-4.0, -6.0), Complex::new(1.0, 0.5)],
        };

        let b = a * 2.0;
        assert_eq!(b[0], Complex::new(2.0, 0.0));
        assert_eq!(b[1], Complex::new(4.0, 8.0));
        assert_eq!(b[2], Complex::new(-8.0, -12.0));
        assert_eq!(b[3], Complex::new(2.0, 1.0));
    }
}

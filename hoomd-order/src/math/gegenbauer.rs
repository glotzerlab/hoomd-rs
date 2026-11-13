// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Evaluate Gegenbauer polynomials.
//!
//! The Gegenbauer (ultraspherical) polynomials $` C_L^{(\alpha)} `$ are the orthogonal
//! polynomials on $` [-1, 1] `$ with weight function $` (1 - x^2)^{\alpha - 1/2} `$.
//! This module implements the $` \alpha = 1 `$ case, which forms a basis set on the
//! hypersphere $` S^3 `$ ([`Gegenbauer`]).
//!
//! [`Gegenbauer`]: crate::math::Gegenbauer

/// Gegenbauer polynomial $` C_L^{(1)} `$.
///
/// $` C_L^{(1)} `$ is the Chebyshev polynomial of the second kind:
/// ```math
/// C_L^{(1)}(x) = U_L(x) = \frac{\sin\left((L + 1)\theta\right)}{\sin\theta},
/// \quad x = \cos\theta.
/// ```
///
/// On the 3-sphere, $` C_L^{(1)} `$ is the zonal (rotation-invariant) harmonic of
/// degree $` L `$. It is the basis set for the rotational autocorrelation function of
/// Karas et al., where the sum over all magnetic quantum numbers of the hyperspherical
/// harmonics of degree $` L `$ reduces to this polynomial of the 4-D dot product:
/// ```math
/// \sum_{m_1 m_2} Y_{L m_1 m_2}^{*}(\mathbf{q}_0) Y_{L m_1 m_2}(\mathbf{q}_1)
/// = \frac{C_L^{(1)}(\mathbf{q}_0 \cdot \mathbf{q}_1)}{L + 1}.
/// ```
///
/// The [`normalization`](Self::normalization) is $` C_L^{(1)}(1) = L + 1 `$
///
/// Construct a [`Gegenbauer`] with a given degree `L`, then call [`evaluate`] to
/// compute $` C_L^{(1)}(x) `$. The type is zero-sized and can be constructed freely.
///
/// [`evaluate`]: Self::evaluate
///
/// # Example
///
/// ```
/// use approxim::assert_relative_eq;
///
/// use hoomd_order::math::Gegenbauer;
///
/// let gegenbauer = Gegenbauer::<6>::new();
///
/// // C_6^{(1)}(1) = U_6(1) = 7
/// assert_relative_eq!(gegenbauer.evaluate(1.0), 7.0, epsilon = 1e-12);
/// ```
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Gegenbauer<const L: usize>;

impl<const L: usize> Gegenbauer<L> {
    /// Construct a Gegenbauer polynomial evaluator of degree `L`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_order::math::Gegenbauer;
    ///
    /// let gegenbauer = Gegenbauer::<4>::new();
    /// ```
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Evaluate $` C_L^{(1)}(x) `$.
    ///
    /// The valid domain is $` x \in [-1, 1] `$. The value is computed with the forward
    /// recurrence $` U_0 = 1 `$, $` U_1 = 2x `$, $` U_{L+1} = 2x U_L - U_{L-1} `$.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    ///
    /// use hoomd_order::math::Gegenbauer;
    ///
    /// let gegenbauer = Gegenbauer::<2>::new();
    ///
    /// assert_relative_eq!(gegenbauer.evaluate(0.5), 0.0, epsilon = 1e-12);
    /// ```
    #[must_use]
    #[inline]
    pub fn evaluate(&self, x: f64) -> f64 {
        if L == 0 {
            return 1.0;
        }

        let mut previous = 1.0;
        let mut current = 2.0 * x;
        for _ in 2..=L {
            let next = 2.0 * x * current - previous;
            previous = current;
            current = next;
        }
        current
    }

    /// The value of the polynomial at $` x = 1 `$.
    ///
    /// $` C_L^{(1)}(1) = L + 1 `$. Divide by this value to normalize
    /// [`evaluate`](Self::evaluate) to 1 at $` x = 1 `$.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_order::math::Gegenbauer;
    ///
    /// let gegenbauer = Gegenbauer::<6>::new();
    ///
    /// assert_eq!(gegenbauer.normalization(), 6.0 + 1.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn normalization(&self) -> f64 {
        (L + 1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::{assert_abs_diff_eq, assert_relative_eq};
    use rstest::rstest;

    /// Check `Gegenbauer::<L>` against the exact integer coefficients of `U_L`
    /// in ascending powers of `x`. Recover the coefficients with `SciPy`:
    /// ```text
    /// import numpy as np, scipy.special as sp
    /// L = 6
    /// x = np.linspace(-1, 1, L + 1)
    /// print(np.polynomial.polynomial.polyfit(x, sp.eval_chebyu(L, x), L).round())
    /// # [-1.  0. 24. -0. -80.  0. 64.]
    /// ```
    #[rstest]
    fn exact_polynomials<const L: usize>(
        #[values(
            (Gegenbauer::<0>::new(), vec![1.0]),
            (Gegenbauer::<1>::new(), vec![0.0, 2.0]),
            (Gegenbauer::<2>::new(), vec![-1.0, 0.0, 4.0]),
            (Gegenbauer::<3>::new(), vec![0.0, -4.0, 0.0, 8.0]),
            (Gegenbauer::<4>::new(), vec![1.0, 0.0, -12.0, 0.0, 16.0]),
            (Gegenbauer::<5>::new(), vec![0.0, 6.0, 0.0, -32.0, 0.0, 32.0]),
            (Gegenbauer::<6>::new(), vec![-1.0, 0.0, 24.0, 0.0, -80.0, 0.0, 64.0]),
        )]
        case: (Gegenbauer<L>, Vec<f64>),
    ) {
        let (gegenbauer, coefficients) = case;
        for x in [-1.0, -0.5, -0.123, 0.0, 0.31, 0.77, 1.0] {
            assert_relative_eq!(
                gegenbauer.evaluate(x),
                coefficients
                    .iter()
                    .rev()
                    .fold(0.0, |acc, &coefficient| acc * x + coefficient),
                epsilon = 1e-12
            );
        }
    }

    /// Check that `C_L^{(1)}(1) = L + 1` for both `normalization` and at `x=1`
    #[rstest]
    fn normalization_is_degree_plus_one<const L: usize>(
        #[values(
            Gegenbauer::<0>::new(),
            Gegenbauer::<1>::new(),
            Gegenbauer::<6>::new(),
            Gegenbauer::<20>::new(),
            Gegenbauer::<50>::new()
        )]
        gegenbauer: Gegenbauer<L>,
    ) {
        let l_plus_one = (L + 1) as f64;
        assert_eq!(gegenbauer.normalization(), l_plus_one);
        assert_eq!(gegenbauer.evaluate(1.0), l_plus_one);
    }

    /// Check the parity relation `U_L(-x) = (-1)^L U_L(x)`.
    #[rstest]
    fn parity<const L: usize>(
        #[values(
            Gegenbauer::<0>::new(),
            Gegenbauer::<1>::new(),
            Gegenbauer::<2>::new(),
            Gegenbauer::<6>::new(),
            Gegenbauer::<13>::new(),
            Gegenbauer::<20>::new()
        )]
        gegenbauer: Gegenbauer<L>,
    ) {
        let sign = if L.is_multiple_of(2) { 1.0 } else { -1.0 };
        for x in [0.123, 0.5, 0.9] {
            assert_relative_eq!(
                gegenbauer.evaluate(-x),
                sign * gegenbauer.evaluate(x),
                epsilon = 1e-12
            );
        }
    }

    /// Check the closed form (must be away from `theta = 0`)
    #[rstest]
    fn closed_form<const L: usize>(
        #[values(
            Gegenbauer::<0>::new(),
            Gegenbauer::<1>::new(),
            Gegenbauer::<2>::new(),
            Gegenbauer::<3>::new(),
            Gegenbauer::<4>::new(),
            Gegenbauer::<5>::new(),
            Gegenbauer::<6>::new(),
            Gegenbauer::<7>::new(),
            Gegenbauer::<8>::new(),
            Gegenbauer::<9>::new(),
            Gegenbauer::<10>::new(),
            Gegenbauer::<11>::new(),
            Gegenbauer::<12>::new()
        )]
        gegenbauer: Gegenbauer<L>,
    ) {
        for theta in [0.3_f64, 1.1, 2.4, 2.9] {
            let value = gegenbauer.evaluate(theta.cos());
            let expected = ((L + 1) as f64 * theta).sin() / theta.sin();
            assert_relative_eq!(value, expected, epsilon = 1e-10);
        }
    }

    #[rstest]
    fn high_degree_references<const L: usize>(
        #[values(
            (Gegenbauer::<10>::new(), 0.3, 1.025_249_177_600_000_1),
            (Gegenbauer::<10>::new(), 0.7, 0.875_458_457_599_999_2),
            (Gegenbauer::<10>::new(), -0.55, -1.188_348_829_9),
            (Gegenbauer::<12>::new(), 0.3, -0.715_612_811_264_000_1),
            (Gegenbauer::<12>::new(), 0.7, -1.110_156_898_304_000_7),
            (Gegenbauer::<12>::new(), -0.55, 0.334_633_765_620_999_2),
            (Gegenbauer::<20>::new(), 0.3, 1.041_317_266_858_977_6),
            (Gegenbauer::<20>::new(), 0.7, -1.174_811_634_460_515),
            (Gegenbauer::<20>::new(), -0.55, 1.130_127_921_508_465_6),
            (Gegenbauer::<50>::new(), 0.3, 1.033_414_175_826_541_6),
            (Gegenbauer::<50>::new(), 0.7, 0.380_732_504_431_420_9),
            (Gegenbauer::<50>::new(), -0.55, 0.172_482_362_541_056_2)
        )]
        case: (Gegenbauer<L>, f64, f64),
    ) {
        let (gegenbauer, x, expected) = case;
        assert_abs_diff_eq!(gegenbauer.evaluate(x), expected, epsilon = 1e-13);
    }
}

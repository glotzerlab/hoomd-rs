// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `RotationalAutocorrelation`

use hoomd_vector::{InnerProduct, Versor};

use crate::math::Gegenbauer;

/// Compute the rotational autocorrelation function $` F_L(t) `$.
///
/// [Karas et al.](https://doi.org/10.1103/PhysRevX.16.011030) characterize rotational
/// dynamics of anisotropic particles with an autocorrelation function of rotations.
/// Because rotations live on the group manifold SO(3), whose double cover has the group
/// manifold $` S^3 `$, $` F_L(t) `$ is an expansion in hyperspherical harmonics:
/// ```math
/// F_L(t) = \frac{1}{N} \sum_i \sum_{m_1 m_2}
/// Y_{L m_1 m_2}^{*}(0) Y_{L m_1 m_2}\left(\mathbf{q}_i(0)^{\dagger}
/// \mathbf{q}_i(t)\right).
/// ```
///
/// The sum over the magnetic quantum numbers $` m_1 `$ and $` m_2 `$ contracts to a
/// single Gegenbauer polynomial $` C_L^{(1)} `$ of the quaternion dot product:
/// ```math
/// F_L(t) = \frac{1}{N} \sum_i
/// \frac{C_L^{(1)}\left(\mathbf{q}_i(0) \cdot \mathbf{q}_i(t)\right)}{L + 1}.
/// ```
/// This normalization gives $` F_L(0) = 1 `$ and $` \lvert F_L \rvert \le 1 `$.
///
/// $` L `$ is the degree on $` S^3 `$, *not* the rank of the SO(3) irreducible
/// representation: $` L = 2n `$, so $` L = 6 `$ corresponds to rank 3. Odd values of
/// $` L `$ are rejected, as they break the symmetry of the double cover:
/// $` C_L^{(1)}(-x) = (-1)^L C_L^{(1)}(x) `$, so $` \mathbf{q} `$ and $` -\mathbf{q} `$
/// would yield different answers.
///
/// Construct a [`RotationalAutocorrelation`] with a given even degree `L`, then call
/// [`kernel`] for individual pairs of orientations or [`f_l`] to average over matched
/// pairs.
///
/// [`kernel`]: Self::kernel
/// [`f_l`]: Self::f_l
///
/// # Example
///
/// Compute $` F_6 `$ for a system of orientations matched by site tag across two frames:
/// ```
/// use approxim::assert_relative_eq;
///
/// use hoomd_order::RotationalAutocorrelation;
/// use hoomd_vector::Versor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let identity = Versor::default();
/// let rotated = Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, 0.5);
///
/// let orientations_0 = vec![identity, identity, rotated];
/// let orientations_t = vec![identity, rotated, identity];
///
/// let rot_autocorr = RotationalAutocorrelation::<6>::new();
/// let f_6 = rot_autocorr.f_l(orientations_0.into_iter().zip(orientations_t));
///
/// assert!(f_6.is_finite());
/// # Ok(())
/// # }
/// ```
#[derive(Copy, Clone, Debug, Default)]
pub struct RotationalAutocorrelation<const L: usize> {
    /// Evaluate the Gegenbauer polynomial.
    gegenbauer: Gegenbauer<L>,
}

impl<const L: usize> RotationalAutocorrelation<L> {
    /// Construct a new rotational autocorrelation evaluator with degree `L`.
    ///
    /// `L` must be even. Odd values of `L` do not compile.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_order::RotationalAutocorrelation;
    /// let rot_autocorr = RotationalAutocorrelation::<6>::new();
    /// ```
    ///
    /// This does not compile:
    ///
    /// ```compile_fail
    /// use hoomd_order::RotationalAutocorrelation;
    /// let rot_autocorr = RotationalAutocorrelation::<3>::new();
    /// ```
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        const { assert!(L.is_multiple_of(2), "L must be even") };
        Self {
            gegenbauer: Gegenbauer::new(),
        }
    }

    /// Compute the autocorrelation of one pair of orientations.
    ///
    /// ```math
    /// \frac{C_L^{(1)}\left(\mathbf{q}_0 \cdot \mathbf{q}_t\right)}{L + 1},
    /// ```
    /// where the argument of the polynomial is the 4-vector dot product of the
    /// two unit quaternions. The value is 1 for identical orientations.
    ///
    /// Average [`kernel`] over all particles with [`f_l`].
    ///
    /// [`kernel`]: Self::kernel
    /// [`f_l`]: Self::f_l
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    ///
    /// use hoomd_order::RotationalAutocorrelation;
    /// use hoomd_vector::Versor;
    ///
    /// let rot_autocorr = RotationalAutocorrelation::<6>::new();
    /// let identity = Versor::default();
    ///
    /// assert_relative_eq!(
    ///     rot_autocorr.kernel(&identity, &identity),
    ///     1.0,
    ///     epsilon = 1e-12
    /// );
    /// ```
    #[must_use]
    #[inline]
    pub fn kernel(&self, q_0: &Versor, q_t: &Versor) -> f64 {
        let (q_0, q_t) = (q_0.get(), q_t.get());
        let x = (q_0.scalar * q_t.scalar + q_0.vector.dot(&q_t.vector)).clamp(-1.0, 1.0);

        self.gegenbauer.evaluate(x) / self.gegenbauer.normalization()
    }

    /// Compute $` F_L(t) `$ averaged over matched pairs of orientations.
    ///
    /// ```math
    /// F_L(t) = \frac{1}{N} \sum_i
    /// \frac{C_L^{(1)}\left(\mathbf{q}_i(0) \cdot \mathbf{q}_i(t)\right)}{L + 1}.
    /// ```
    ///
    /// The `pairs` iterator produces the orientations of each site at the initial and
    /// current times. Matching sites across frames by tag is the caller's
    /// responsibility: (for example, by zipping two orientation maps keyed by tag).
    ///
    /// Returns `nan` when `pairs` is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    ///
    /// use hoomd_order::RotationalAutocorrelation;
    /// use hoomd_vector::Versor;
    ///
    /// let identity = Versor::default();
    ///
    /// // No rotation at all: F_L = 1.
    /// let f_6 = RotationalAutocorrelation::<6>::new()
    ///     .f_l(vec![(identity, identity), (identity, identity)]);
    ///
    /// assert_eq!(f_6, 1.0);
    /// ```
    ///
    /// Compute $` F_L(t) `$ of a fixed reference frame against each frame of a
    /// trajectory (e.g. orientations read from a GSD file, matched by particle
    /// index):
    /// ```
    /// use approxim::assert_relative_eq;
    ///
    /// use hoomd_order::RotationalAutocorrelation;
    /// use hoomd_vector::Versor;
    ///
    /// let reference = vec![Versor::default(); 3];
    /// let trajectory = vec![reference.clone(), vec![Versor::default(); 3]];
    ///
    /// let rot_autocorr = RotationalAutocorrelation::<6>::new();
    /// let f_l: Vec<_> = trajectory
    ///     .iter()
    ///     .map(|frame| rot_autocorr.f_l(reference.iter().copied().zip(frame.iter().copied())))
    ///     .collect();
    ///
    /// assert_relative_eq!(f_l[0], 1.0, epsilon = 1e-12);
    /// ```
    #[must_use]
    #[inline]
    pub fn f_l<I: IntoIterator<Item = (Versor, Versor)>>(&self, pairs: I) -> f64 {
        let (sum, count) = pairs
            .into_iter()
            .fold((0.0f64, 0_usize), |(s, n), (q_0, q_t)| {
                (s + self.kernel(&q_0, &q_t), n + 1)
            });
        sum / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::{assert_abs_diff_eq, assert_relative_eq};
    use hoomd_vector::{Cartesian, Rotation, Unit};
    use num_complex::Complex;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rstest::rstest;

    /// Sample the kernel of many Haar-random orientation pairs.
    fn kernel_samples<const L: usize>(
        rot_autocorr: RotationalAutocorrelation<L>,
        rng: &mut StdRng,
    ) -> Vec<f64> {
        (0..200_000)
            .map(|_| {
                let q_0: Versor = rng.random();
                let q_t: Versor = rng.random();
                rot_autocorr.kernel(&q_0, &q_t)
            })
            .collect()
    }

    /// Assert the kernel has mean 0 and standard deviation 1/(L + 1) over random pairs.
    fn assert_statistics<const L: usize>(
        rot_autocorr: RotationalAutocorrelation<L>,
        rng: &mut StdRng,
    ) {
        let values = kernel_samples(rot_autocorr, rng);
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let spread = (values.iter().map(|k| k * k).sum::<f64>() / n).sqrt();
        let expected = 1.0 / (L + 1) as f64;

        let mean_tolerance = 5.0 * expected / n.sqrt();
        assert!(
            mean.abs() < mean_tolerance,
            "mean = {mean}, tolerance = {mean_tolerance}"
        );
        assert_relative_eq!(spread, expected, epsilon = 0.1 * expected);
    }

    #[rstest]
    fn identical_orientations<const L: usize>(
        #[values(
            RotationalAutocorrelation::<0>::new(),
            RotationalAutocorrelation::<2>::new(),
            RotationalAutocorrelation::<6>::new(),
            RotationalAutocorrelation::<10>::new(),
            RotationalAutocorrelation::<20>::new()
        )]
        rot_autocorr: RotationalAutocorrelation<L>,
    ) {
        assert_relative_eq!(
            rot_autocorr.kernel(&Versor::default(), &Versor::default()),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn global_rotation_invariance() {
        let rot_autocorr = RotationalAutocorrelation::<6>::new();
        let global = Versor::from_axis_angle([1.0, 2.0, -1.0].try_into().unwrap(), 1.2345);

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..32 {
            let q_0: Versor = rng.random();
            let q_t: Versor = rng.random();

            assert_relative_eq!(
                rot_autocorr.kernel(&q_0, &q_t),
                rot_autocorr.kernel(&global.combine(&q_0), &global.combine(&q_t)),
                epsilon = 1e-12
            );
        }
    }

    /// The kernel vanishes when the rotation angle is φ = 2πk / (L+1) about any axis.
    #[rstest]
    fn zero_at_special_angles<const L: usize>(
        #[values(
            RotationalAutocorrelation::<2>::new(),
            RotationalAutocorrelation::<4>::new(),
            RotationalAutocorrelation::<6>::new()
        )]
        rot_autocorr: RotationalAutocorrelation<L>,
    ) {
        let axis: Unit<Cartesian<3>> = [1.0, 1.0, 1.0].try_into().unwrap();
        let identity = Versor::default();
        for k in 1..=L {
            let phi = 2.0 * std::f64::consts::PI * (k as f64) / ((L + 1) as f64);
            let rotated = Versor::from_axis_angle(axis, phi);
            assert!(
                rot_autocorr.kernel(&identity, &rotated).abs() < 1e-12,
                "k = {k}"
            );
        }
    }

    #[rstest]
    fn zero_mean_unit_spread<const L: usize>(
        #[values(
            RotationalAutocorrelation::<2>::new(),
            RotationalAutocorrelation::<6>::new()
        )]
        rot_autocorr: RotationalAutocorrelation<L>,
    ) {
        let mut rng = StdRng::seed_from_u64(7);
        assert_statistics(rot_autocorr, &mut rng);
    }

    /// Evaluate the hyperspherical harmonic $` Y_{L, L/2 - a, L/2 - b} `$ of
    /// Karas et al. (Eq. 4 of their SI) at a unit quaternion
    /// $` \mathbf{q} = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k} `$, in their
    /// parameterization $` \xi = x + i y `$, $` \zeta = z + i w `$.
    fn eval_with_greeks(l: u32, a: u32, b: u32, q: &Versor) -> Complex<f64> {
        let q = q.get();
        let xi = Complex::new(q.vector[0], q.vector[1]);
        let zeta = Complex::new(q.vector[2], q.scalar);

        let factorial = |n: u32| (1..=n).map(f64::from).product::<f64>();
        let prefactor = (factorial(a) * factorial(l - a) * factorial(b) * factorial(l - b)
            / f64::from(l + 1))
        .sqrt();

        let mut sum = Complex::new(0.0, 0.0);
        for k in 0..=l {
            let last = l + k;
            if last < a + b || k > a || k > b {
                continue;
            }
            let denominator =
                factorial(k) * factorial(last - a - b) * factorial(a - k) * factorial(b - k);
            sum += xi.conj().powu(k)
                * zeta.powu(b - k)
                * zeta.conj().powu(a - k)
                * (-xi).powu(last - a - b)
                / denominator;
        }

        prefactor * sum
    }

    #[rstest]
    fn matches_hyperspherical_harmonics<const L: usize>(
        #[values(
            RotationalAutocorrelation::<2>::new(),
            RotationalAutocorrelation::<4>::new(),
            RotationalAutocorrelation::<6>::new()
        )]
        rot_autocorr: RotationalAutocorrelation<L>,
    ) {
        let l = u32::try_from(L).expect("L should fit in u32");
        let mut rng = StdRng::seed_from_u64(11);
        for _ in 0..8 {
            let (q_0, q_t): (Versor, Versor) = rng.random();

            let harmonics: Complex<f64> = (0..=l)
                .flat_map(|a| (0..=l).map(move |b| (a, b)))
                .map(|(a, b)| {
                    eval_with_greeks(l, a, b, &q_0).conj() * eval_with_greeks(l, a, b, &q_t)
                })
                .sum();

            assert_abs_diff_eq!(
                harmonics.re,
                rot_autocorr.kernel(&q_0, &q_t),
                epsilon = 1e-12
            );
            assert_abs_diff_eq!(harmonics.im, 0.0, epsilon = 1e-12);
        }
    }
}

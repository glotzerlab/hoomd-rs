// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*!
Methods and implementations for an N-hypersphere, where N is the dimension.
*/
use crate::{IntersectsAt, Shape, SupportFn, Volume};
use hoomd_vector::{Rotate, Vector};
use std::f64::consts::PI;

/// The (single, double, ...)-factorial function
fn factorial(n: usize, ntuple: usize) -> usize {
    assert!(ntuple > 0);
    if n == 0 {
        1
    } else {
        (1..=n)
            .rev()
            .step_by(ntuple)
            .reduce(|acc, x| acc * x)
            .unwrap_or_default() // inaccessible: 1..=(n!=0) is never empty
    }
}

/// An n-hypersphere ===================================================================
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere<const N: usize> {
    /// Radius of the sphere
    pub r: f64,
}

impl<const N: usize> Default for Sphere<N> {
    #[inline]
    fn default() -> Self {
        Sphere { r: 1.0 }
    }
}

impl<const N: usize> From<f64> for Sphere<N> {
    #[inline]
    fn from(r: f64) -> Self {
        Sphere { r }
    }
}

// TRAITS

impl<const N: usize, V: Vector> SupportFn<V> for Sphere<N> {
    #[inline]
    fn support(&self, n: &V) -> V {
        *n / n.norm() * self.r
    }
}

impl<const N: usize> Shape<N> for Sphere<N> {
    #[inline]
    fn bounding_sphere(&self) -> Sphere<N> {
        *self
    }
}

impl<const N: usize> Volume for Sphere<N> {
    #[inline]
    #[allow(clippy::expect_used)] // If users need a 2^31-1 hypersphere, raise an issue.
    fn volume(&self) -> f64 {
        let dim_factor = (if N.rem_euclid(2) == 0 { N } else { N - 1 } / 2)
            .try_into()
            .expect("N > i32::MAX and would overflow!");
        if N.rem_euclid(2) == 0 {
            PI.powi(dim_factor) / (factorial(N / 2, 1) as f64)
        } else {
            2.0 * (2.0 * PI).powi(dim_factor) / (factorial(N, 2) as f64)
        } // TODO: replace with std::f64::gamma when its in main
    }
}

impl<const N: usize, V: Vector, R: Rotate<V>> IntersectsAt<Sphere<N>, V, R> for Sphere<N> {
    type OptionalRotation = Option<R>;
    #[inline]
    fn intersects_at(&self, other: &Sphere<N>, v_ij: &V, _o_ij: &Option<R>) -> bool {
        (v_ij).norm_squared() <= (other.r + self.r).powi(2)
    }
}

#[cfg(test)]
#[expect(
    clippy::used_underscore_binding,
    reason = "Used in test parameterization."
)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use hoomd_vector::Cartesian;
    use rstest::*;
    use std::marker::PhantomData;

    fn volume_map(n: usize) -> f64 {
        match n {
            0 => 1.0,
            1 => 2.0,
            2 => PI,
            3 => 4.0 / 3.0 * PI,
            4 => PI.powi(2) / 2.0,
            5 => 8.0 * PI.powi(2) / 15.0,
            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case(PhantomData::<Sphere<0>>)]
    #[case(PhantomData::<Sphere<1>>)]
    #[case(PhantomData::<Sphere<2>>)]
    #[case(PhantomData::<Sphere<3>>)]
    #[case(PhantomData::<Sphere<4>>)]
    #[case(PhantomData::<Sphere<5>>)]
    fn test_volume_and_radius<const N: usize>(#[case] _n: PhantomData<Sphere<N>>) {
        let s = Sphere::<N>::from(1.0);
        assert_eq!(s.r, 1.0);
        assert_eq!(s, Sphere::<N>::default());
        assert_relative_eq!(s.volume(), volume_map(N));
    }

    #[rstest]
    #[case(PhantomData::<Sphere<0>>)]
    #[case(PhantomData::<Sphere<1>>)]
    #[case(PhantomData::<Sphere<2>>)]
    #[case(PhantomData::<Sphere<3>>)]
    #[case(PhantomData::<Sphere<4>>)]
    #[case(PhantomData::<Sphere<5>>)]
    fn test_bounding_sphere<const N: usize>(#[case] _n: PhantomData<Sphere<N>>) {
        let s = Sphere::<N>::default();
        assert_eq!(s, s.bounding_sphere());
    }

    #[rstest]
    fn test_n_factorial(#[values(1, 2, 3, 4)] m: usize) {
        assert_eq!(factorial(m, m), m);
    }
    #[rstest]
    fn test_single_double_factorial(#[values(1, 5, 10, 18, 20)] n: usize) {
        assert_eq!(factorial(n, 1), factorial(n, 2) * factorial(n - 1, 2));
    }

    #[rstest]
    #[case(PhantomData::<Sphere<0>>)]
    #[case(PhantomData::<Sphere<1>>)]
    #[case(PhantomData::<Sphere<2>>)]
    #[case(PhantomData::<Sphere<3>>)]
    #[case(PhantomData::<Sphere<4>>)]
    #[case(PhantomData::<Sphere<5>>)]
    fn test_support_fn<const N: usize>(
        #[case] _n: PhantomData<Sphere<N>>,
        #[values(0.1, 1.0, 33.3)] r: f64,
    ) {
        let s = Sphere::<N>::from(r);
        let v = Cartesian::<N>::from([r.powi(2) / 1.8; N]);
        assert_eq!(v / v.norm() * r, s.support(&v));
    }
}

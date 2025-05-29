// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*!
Methods and implementations for an N-hypersphere, where N is the dimension.
*/
use crate::{
    BoundingShape, BoundingSphere, IntersectsAt, MinDistance, SupportMapping, Volume,
    xenocollide::collide3d,
};
use hoomd_vector::{Cartesian, Rotate, Vector};
use std::f64::consts::PI;

/// The (single, double, ...)-factorial function
pub(crate) fn factorial(n: usize, ntuple: usize) -> usize {
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

/// An n-hypersphere
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hypersphere<const N: usize> {
    /// Radius of the sphere
    pub r: f64,
}

/// A `Circle` in two dimensions.
pub type Circle = Hypersphere<2>;
/// A `Sphere` in three dimensions.
pub type Sphere = Hypersphere<3>;

impl<const N: usize> Default for Hypersphere<N> {
    #[inline]
    fn default() -> Self {
        Hypersphere { r: 1.0 }
    }
}

impl<const N: usize> Hypersphere<N> {
    /// Create a sphere from a float with a given radius.
    #[must_use]
    #[inline]
    pub fn from_radius(r: f64) -> Self {
        Hypersphere { r }
    }
}

// TRAITS

impl<const N: usize, V: Vector> SupportMapping<V> for Hypersphere<N> {
    #[inline]
    fn support_mapping(&self, n: &V) -> V {
        *n / n.norm() * self.r
    }
}

impl<const N: usize> Volume for Hypersphere<N> {
    #[inline]
    fn volume(&self) -> f64 {
        let dim_factor = (if N.rem_euclid(2) == 0 { N } else { N - 1 } / 2) as f64;
        let prefactor = if N.rem_euclid(2) == 0 {
            PI.powf(dim_factor) / (factorial(N / 2, 1) as f64)
        } else {
            2.0 * (2.0 * PI).powf(dim_factor) / (factorial(N, 2) as f64)
        };
        prefactor
            * self
                .r
                .powi(N.try_into().expect("Dimension would overflow i32!"))
        // TODO: replace with std::f64::gamma when its in main
    }
}

impl<const N: usize, V: Vector, R: Rotate<V>> IntersectsAt<Hypersphere<N>, V, R>
    for Hypersphere<N>
{
    #[inline]
    fn intersects_at(&self, other: &Hypersphere<N>, v_ij: &V, _o_ij: &R) -> bool {
        (v_ij).norm_squared() <= (other.r + self.r).powi(2)
    }
}

impl<const N: usize, V: Vector, R: Rotate<V>> BoundingShape<V, R> for Hypersphere<N> {
    type Shape = Hypersphere<N>;
    #[inline]
    fn bounding_shape(&self) -> Hypersphere<N> {
        *self
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

    fn volume_map(n: usize, r: f64) -> f64 {
        match n {
            0 => 1.0 * r.powi(i32::try_from(n).unwrap()),
            1 => 2.0 * r.powi(i32::try_from(n).unwrap()),
            2 => PI * r.powi(i32::try_from(n).unwrap()),
            3 => 4.0 / 3.0 * PI * r.powi(i32::try_from(n).unwrap()),
            4 => PI.powi(2) / 2.0 * r.powi(i32::try_from(n).unwrap()),
            5 => 8.0 * PI.powi(2) / 15.0 * r.powi(i32::try_from(n).unwrap()),
            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case(PhantomData::<Hypersphere<0>>)]
    #[case(PhantomData::<Hypersphere<1>>)]
    #[case(PhantomData::<Hypersphere<2>>)]
    #[case(PhantomData::<Hypersphere<3>>)]
    #[case(PhantomData::<Hypersphere<4>>)]
    #[case(PhantomData::<Hypersphere<5>>)]
    fn test_volume_and_radius<const N: usize>(
        #[case] _n: PhantomData<Hypersphere<N>>,
        #[values(0.01, 1.0, 33.3, 1e6)] r: f64,
    ) {
        let s = Hypersphere::<N> { r };

        if r == 1.0 {
            assert_eq!(s.r, 1.0);
            assert_eq!(s, Hypersphere::<N>::default());
        } else {
            assert_eq!(s.r, r);
        }

        assert_relative_eq!(s.volume(), volume_map(N, r));
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
    #[case(PhantomData::<Hypersphere<0>>)]
    #[case(PhantomData::<Hypersphere<1>>)]
    #[case(PhantomData::<Hypersphere<2>>)]
    #[case(PhantomData::<Hypersphere<3>>)]
    #[case(PhantomData::<Hypersphere<4>>)]
    #[case(PhantomData::<Hypersphere<5>>)]
    fn test_support_fn<const N: usize>(
        #[case] _n: PhantomData<Hypersphere<N>>,
        #[values(0.1, 1.0, 33.3)] r: f64,
    ) {
        let s = Hypersphere::<N> { r };
        let v = Cartesian::<N>::from([r.powi(2) / 1.8; N]);
        assert_eq!(v / v.norm() * r, s.support_mapping(&v));
    }
}

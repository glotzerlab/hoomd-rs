// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Hypersphere`]
use crate::{BoundingSphereRadius, IntersectsAt, IsPointInside, SupportMapping, Volume};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct, Rotate, Rotation, distribution::Ball};

use rand::{Rng, distr::Distribution};
use std::{f64::consts::PI, ops::Mul};

/// The (single, double, ...)-factorial function
pub fn factorial(n: usize, ntuple: usize) -> usize {
    assert!(ntuple > 0);
    if n == 0 {
        1
    } else {
        (1..=n)
            .rev()
            .step_by(ntuple)
            .reduce(usize::mul)
            .expect("1..=(n!=0) is never empty")
    }
}

/// Compute the volume prefactor for the volume of a rounded shape
pub(crate) fn sphere_volume_prefactor(n: usize) -> f64 {
    // FUTURE: replace with std::f64::gamma when its in main
    let dim_factor = (if n.rem_euclid(2) == 0 { n } else { n - 1 } / 2) as f64;
    if n.rem_euclid(2) == 0 {
        PI.powf(dim_factor) / (factorial(n / 2, 1) as f64)
    } else {
        2.0 * (2.0 * PI).powf(dim_factor) / (factorial(n, 2) as f64)
    }
}

/// All points within a given `radius` from the origin.
///
/// # Examples
///
/// Basic construction and methods:
/// ```
/// use hoomd_geometry::{SupportMapping, Volume, shape::Hypersphere};
/// use hoomd_vector::Cartesian;
/// use std::f64::consts::PI;
///
/// let unit_sphere = Hypersphere::<3>::default();
/// let volume = unit_sphere.volume();
///
/// assert_eq!(unit_sphere.radius.get(), 1.0);
/// assert_eq!(volume, 4.0 * PI / 3.0);
///
/// assert_eq!(
///     unit_sphere.support_mapping(&Cartesian::from([1.0; 3])),
///     [1.0 / f64::sqrt(3.0); 3].into()
/// )
/// ```
///
/// Test for intersections:
/// ```
/// use hoomd_geometry::{IntersectsAt, shape::Hypersphere};
/// use hoomd_vector::{Cartesian, Versor};
///
/// let unit_sphere = Hypersphere::<3>::default();
///
/// assert!(!unit_sphere.intersects_at(
///     &unit_sphere,
///     &Cartesian::from([2.1, 0.0, 0.0]),
///     &Versor::default()
/// ));
/// assert!(unit_sphere.intersects_at(
///     &unit_sphere,
///     &Cartesian::from([0.0, 1.9, 0.0]),
///     &Versor::default()
/// ));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Hypersphere<const N: usize> {
    /// Radius of the sphere
    pub radius: PositiveReal,
}

/// A circle in two dimensions.
///
/// # Examples
///
/// Basic construction and methods:
/// ```
/// use hoomd_geometry::{Volume, shape::Circle};
/// use hoomd_vector::Cartesian;
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let circle = Circle {
///     radius: 2.0.try_into()?,
/// };
/// let volume = circle.volume();
///
/// assert_eq!(volume, PI * 4.0);
/// # Ok(())
/// # }
/// ```
///
/// Test for intersections:
/// ```
/// use hoomd_geometry::{IntersectsAt, shape::Circle};
/// use hoomd_vector::{Angle, Cartesian};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let circle = Circle {
///     radius: 1.0.try_into()?,
/// };
///
/// assert_eq!(
///     circle.intersects_at(
///         &circle,
///         &Cartesian::from([2.1, 0.0]),
///         &Angle::default()
///     ),
///     false
/// );
/// assert_eq!(
///     circle.intersects_at(
///         &circle,
///         &Cartesian::from([0.0, 1.9]),
///         &Angle::default()
///     ),
///     true
/// );
/// # Ok(())
/// # }
/// ```
pub type Circle = Hypersphere<2>;

/// A sphere in three dimensions.
///
/// # Examples
///
/// Basic construction and methods:
/// ```
/// use hoomd_geometry::{SupportMapping, Volume, shape::Sphere};
/// use hoomd_vector::Cartesian;
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let unit_sphere = Sphere {
///     radius: 1.0.try_into()?,
/// };
/// let volume = unit_sphere.volume();
///
/// assert_eq!(unit_sphere.radius.get(), 1.0);
/// assert_eq!(volume, 4.0 * PI / 3.0);
///
/// assert_eq!(
///     unit_sphere.support_mapping(&Cartesian::from([1.0; 3])),
///     [1.0 / f64::sqrt(3.0); 3].into()
/// );
/// # Ok(())
/// # }
/// ```
///
/// Test for intersections:
/// ```
/// use hoomd_geometry::{IntersectsAt, shape::Sphere};
/// use hoomd_vector::{Cartesian, Versor};
///
/// let unit_sphere = Sphere::default();
///
/// assert!(!unit_sphere.intersects_at(
///     &unit_sphere,
///     &Cartesian::from([2.1, 0.0, 0.0]),
///     &Versor::default()
/// ));
/// assert!(unit_sphere.intersects_at(
///     &unit_sphere,
///     &Cartesian::from([0.0, 1.9, 0.0]),
///     &Versor::default()
/// ));
/// ```
pub type Sphere = Hypersphere<3>;

impl<const N: usize> Default for Hypersphere<N> {
    #[inline]
    fn default() -> Self {
        Hypersphere {
            radius: 1.0.try_into().expect("1.0 is a positive real"),
        }
    }
}

impl<const N: usize> Hypersphere<N> {
    /// Create a sphere with a given positive real radius.
    #[must_use]
    #[inline]
    pub fn with_radius(radius: PositiveReal) -> Self {
        Hypersphere { radius }
    }

    /// Test whether one sphere intersects with another.
    ///
    /// The vector `v_ij` points from the local origin of `self` to the local
    /// origin of `other`.
    #[inline]
    pub fn intersects<V>(&self, other: &Hypersphere<N>, v_ij: &V) -> bool
    where
        V: InnerProduct,
    {
        (v_ij).norm_squared() <= (other.radius.get() + self.radius.get()).powi(2)
    }
}

// TRAITS

impl<const N: usize, V: InnerProduct> SupportMapping<V> for Hypersphere<N> {
    #[inline]
    fn support_mapping(&self, n: &V) -> V {
        *n / n.norm() * self.radius.get()
    }
}

impl<const N: usize> Volume for Hypersphere<N> {
    #[inline]
    fn volume(&self) -> f64 {
        sphere_volume_prefactor(N)
            * self
                .radius
                .get()
                .powi(N.try_into().expect("Dimension should not overflow i32!"))
    }
}

impl<const N: usize, V, R> IntersectsAt<Hypersphere<N>, V, R> for Hypersphere<N>
where
    V: InnerProduct,
    R: Rotation + Rotate<V>,
{
    #[inline]
    fn intersects_at(&self, other: &Hypersphere<N>, v_ij: &V, _o_ij: &R) -> bool {
        (v_ij).norm_squared() <= (other.radius.get() + self.radius.get()).powi(2)
    }
}

impl<const N: usize> BoundingSphereRadius for Hypersphere<N> {
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        self.radius
    }
}

impl<const N: usize, V> IsPointInside<V> for Hypersphere<N>
where
    V: InnerProduct,
{
    /// Check if a vector is inside a hypersphere.
    ///
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::Sphere};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let sphere = Sphere {
    ///     radius: 3.0.try_into()?,
    /// };
    ///
    /// assert!(sphere.is_point_inside(&Cartesian::from([2.5, 0.0, 0.0])));
    /// assert!(!sphere.is_point_inside(&Cartesian::from([3.0, -3.0, 2.0])));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn is_point_inside(&self, point: &V) -> bool {
        point.dot(point) < self.radius.get().powi(2)
    }
}

impl<const N: usize> Distribution<Cartesian<N>> for Hypersphere<N> {
    /// Generate points uniformly distributed in the hypersphere.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::Sphere};
    /// use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let sphere = Sphere {
    ///     radius: 5.0.try_into()?,
    /// };
    /// let mut rng = StdRng::seed_from_u64(1);
    ///
    /// let point = sphere.sample(&mut rng);
    /// assert!(sphere.is_point_inside(&point));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<N> {
        let ball = Ball {
            radius: self.radius,
        };
        ball.sample(rng)
    }
}

#[cfg(test)]
#[expect(
    clippy::used_underscore_binding,
    reason = "Used in test parameterization."
)]
#[expect(
    clippy::unreadable_literal,
    reason = "exact test results need not be readable"
)]
mod tests {
    use super::*;
    use crate::Convex;
    use approxim::assert_relative_eq;
    use hoomd_vector::{Cartesian, Versor};
    use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
    use rstest::*;
    use std::marker::PhantomData;

    /// Number of random samples to test.
    const N: usize = 1024;

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
        #[values(0.01, 1.0, 33.3, 1e6)] radius: f64,
    ) {
        let s = Hypersphere::<N> {
            radius: radius.try_into().expect("test value is a positive real"),
        };

        if radius == 1.0 {
            assert_eq!(s.radius.get(), 1.0);
            assert_eq!(s, Hypersphere::<N>::default());
        } else {
            assert_eq!(s.radius.get(), radius);
        }

        assert_relative_eq!(s.volume(), volume_map(N, radius));
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
        #[values(0.1, 1.0, 33.3)] radius: f64,
    ) {
        let s = Hypersphere::<N> {
            radius: radius.try_into().expect("test value is a positive real"),
        };
        let v = Cartesian::<N>::from([radius.powi(2) / 1.8; N]);
        assert_eq!(v / v.norm() * radius, s.support_mapping(&v));
    }

    #[test]
    fn support_mapping() {
        let sphere = Sphere::with_radius(2.0.try_into().expect("test value is a positive real"));

        assert_relative_eq!(
            sphere.support_mapping(&Cartesian::from([0.0, 0.0, 1.0])),
            [0.0, 0.0, 2.0].into()
        );
        assert_relative_eq!(
            sphere.support_mapping(&Cartesian::from([0.0, 0.0, 01.0])),
            [0.0, 0.0, 2.0].into()
        );
        assert_relative_eq!(
            sphere.support_mapping(&Cartesian::from([0.0, 1.0, 0.0])),
            [0.0, 2.0, 0.0].into()
        );
        assert_relative_eq!(
            sphere.support_mapping(&Cartesian::from([0.0, -1.0, 0.0])),
            [0.0, -2.0, 0.0].into()
        );
        assert_relative_eq!(
            sphere.support_mapping(&Cartesian::from([1.0, 0.0, 0.0])),
            [2.0, 0.0, 0.0].into()
        );
        assert_relative_eq!(
            sphere.support_mapping(&Cartesian::from([-1.0, 0.0, 0.0])),
            [-2.0, 0.0, 0.0].into()
        );

        assert_relative_eq!(
            sphere.support_mapping(&Cartesian::from([1.0, 1.0, 1.0])),
            [1.1547005383792517, 1.1547005383792517, 1.1547005383792517].into()
        );
    }

    #[test]
    fn intersects_at() {
        let sphere0 = Sphere::with_radius(2.0.try_into().expect("test value is a positive real"));
        let sphere1 = Sphere::with_radius(4.0.try_into().expect("test value is a positive real"));
        let identity = Versor::default();

        assert!(sphere0.intersects_at(&sphere1, &[0.0, 0.0, 5.9].into(), &identity));
        assert!(sphere0.intersects_at(&sphere1, &[0.0, 5.9, 0.0].into(), &identity));
        assert!(sphere0.intersects_at(&sphere1, &[5.9, 0.0, 0.0].into(), &identity));
        assert!(sphere0.intersects_at(&sphere1, &[3.4, 3.4, 3.4].into(), &identity));

        assert!(!sphere0.intersects_at(&sphere1, &[0.0, 0.0, 6.1].into(), &identity));
        assert!(!sphere0.intersects_at(&sphere1, &[0.0, 6.1, 0.0].into(), &identity));
        assert!(!sphere0.intersects_at(&sphere1, &[6.1, 0.0, 0.0].into(), &identity));
        assert!(!sphere0.intersects_at(&sphere1, &[3.52, 3.52, 3.52].into(), &identity));

        let sphere0 = Convex(sphere0);
        let sphere1 = Convex(sphere1);

        assert!(sphere0.intersects_at(&sphere1, &[0.0, 0.0, 5.9].into(), &identity));
        assert!(sphere0.intersects_at(&sphere1, &[0.0, 5.9, 0.0].into(), &identity));
        assert!(sphere0.intersects_at(&sphere1, &[5.9, 0.0, 0.0].into(), &identity));
        assert!(sphere0.intersects_at(&sphere1, &[3.4, 3.4, 3.4].into(), &identity));

        assert!(!sphere0.intersects_at(&sphere1, &[0.0, 0.0, 6.1].into(), &identity));
        assert!(!sphere0.intersects_at(&sphere1, &[0.0, 6.1, 0.0].into(), &identity));
        assert!(!sphere0.intersects_at(&sphere1, &[6.1, 0.0, 0.0].into(), &identity));
        assert!(!sphere0.intersects_at(&sphere1, &[3.52, 3.52, 3.52].into(), &identity));
    }

    #[test]
    fn is_point_inside() {
        let circle = Circle::with_radius(2.0.try_into().expect("test value is a positive real"));

        assert!(circle.is_point_inside(&Cartesian::from([0.0, 0.0])));
        assert!(circle.is_point_inside(&Cartesian::from([0.0, 1.0])));
        assert!(circle.is_point_inside(&Cartesian::from([0.0, -1.0])));
        assert!(circle.is_point_inside(&Cartesian::from([1.0, 0.0])));
        assert!(circle.is_point_inside(&Cartesian::from([-1.0, 0.0])));
        assert!(circle.is_point_inside(&Cartesian::from([2.0f64.next_down(), 0.0])));
        assert!(circle.is_point_inside(&Cartesian::from([0.0, 2.0f64.next_down()])));

        assert!(!circle.is_point_inside(&Cartesian::from([2.0, 0.0])));
        assert!(!circle.is_point_inside(&Cartesian::from([0.0, 2.0])));
        assert!(!circle.is_point_inside(&Cartesian::from([1.5, 1.5])));
    }

    #[test]
    fn distribution() {
        let circle = Circle::with_radius(4.0.try_into().expect("test value is a positive real"));
        let mut rng = StdRng::seed_from_u64(4);

        let points: Vec<_> = (&circle).sample_iter(&mut rng).take(N).collect();
        assert!(&points.iter().all(|p| circle.is_point_inside(p)));
        assert!(&points.iter().any(|p| p.dot(p) > 3.9));
    }
}

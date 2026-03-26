// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Hypersphere`]

use rand::{Rng, distr::Distribution};
use serde::{Deserialize, Serialize};
use std::{array, f64::consts::PI, ops::Mul};

use crate::{
    BoundingSphereRadius, Error, IntersectsAt, IntersectsAtGlobal, IsPointInside, MapPoint, Scale,
    SupportMapping, Volume,
};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct, Rotation, distribution::Ball};

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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

impl<const N: usize, R> IntersectsAtGlobal<Hypersphere<N>, Cartesian<N>, R> for Hypersphere<N>
where
    R: Rotation
{
    #[inline]
    fn intersects_at_global(
        &self,
        other: &Hypersphere<N>,
        r_self: &Cartesian<N>,
        _o_self: &R,
        r_other: &Cartesian<N>,
        _o_other: &R,
    ) -> bool {
        let v_ij = *r_other - *r_self;
        let o_ij = R::identity();

        self.intersects_at(other, &v_ij, &o_ij)
    }
}

impl<const N: usize, V, R> IntersectsAt<Hypersphere<N>, V, R> for Hypersphere<N>
where
    V: InnerProduct,
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

impl<const N: usize> Scale for Hypersphere<N> {
    /// Construct a scaled hypersphere.
    ///
    /// The resulting hypersphere's radious $` r_\mathrm{new} `$ is
    /// the original's $` r `$ scaled by $` v `$:
    /// ```math
    /// r_\mathrm{new} = v r
    /// ```
    ///
    /// The centroid remains at the origin.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Sphere};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let sphere = Sphere {
    ///     radius: 5.0.try_into()?,
    /// };
    ///
    /// let scaled_sphere = sphere.scale_length(0.5.try_into()?);
    ///
    /// assert_eq!(scaled_sphere.radius.get(), 2.5);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        Self {
            radius: self.radius * v,
        }
    }

    /// Construct a scaled hypersphere.
    ///
    /// The resulting hypersphere's radius $` r_\mathrm{new} `$ is
    /// the original's $` r `$ scaled by $` v^\frac{1}{N} `$:
    /// ```math
    /// r_\mathrm{new} = v^\frac{1}{N} r
    /// ```
    ///
    /// The centroid remains at the origin.
    ///
    /// # Example
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Circle};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let sphere = Circle {
    ///     radius: 5.0.try_into()?,
    /// };
    ///
    /// let scaled_sphere = sphere.scale_volume(0.25.try_into()?);
    ///
    /// assert_eq!(scaled_sphere.radius.get(), 2.5);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let v = v.get().powf(1.0 / N as f64);
        self.scale_length(v.try_into().expect("v^{1/N} should be a positive real"))
    }
}

impl<const N: usize> MapPoint<Cartesian<N>> for Hypersphere<N> {
    /// Map a point from one hypersphere to another.
    ///
    /// Given a point P *inside `self`*, map it to the other shape
    /// by scaling.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PointOutsideShape`] when `point` is outside the shape
    /// `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{MapPoint, shape::Circle};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let closed_a = Circle {
    ///     radius: 10.0.try_into()?,
    /// };
    /// let closed_b = Circle {
    ///     radius: 20.0.try_into()?,
    /// };
    ///
    /// let mapped_point =
    ///     closed_a.map_point(Cartesian::from([-1.0, 1.0]), &closed_b);
    ///
    /// assert_eq!(mapped_point, Ok(Cartesian::from([-2.0, 2.0])));
    /// assert_eq!(
    ///     closed_a.map_point(Cartesian::from([-100.0, 1.0]), &closed_b),
    ///     Err(hoomd_geometry::Error::PointOutsideShape)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn map_point(&self, point: Cartesian<N>, other: &Self) -> Result<Cartesian<N>, Error> {
        if !self.is_point_inside(&point) {
            return Err(Error::PointOutsideShape);
        }

        // When multiplying by the scale factor, the floating point multiply
        // might round up in a way that places the mapped point outside the
        // other shape. Progressively make the scale smaller until the
        // check passes.
        let mut scale = other.radius / self.radius;
        loop {
            let point = Cartesian::from(array::from_fn(|i| scale.get() * point[i]));
            if other.is_point_inside(&point) {
                return Ok(point);
            }

            scale = scale
                .get()
                .next_down()
                .try_into()
                .expect("scale should remain a positive real");
        }
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
    use assert2::check;
    use hoomd_vector::{Cartesian, Versor};
    use rand::{
        SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };
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
    ) -> anyhow::Result<()> {
        let s = Hypersphere::<N> {
            radius: radius.try_into()?,
        };

        if radius == 1.0 {
            check!(s.radius.get() == 1.0);
            check!(s == Hypersphere::<N>::default());
        } else {
            check!(s.radius.get() == radius);
        }

        assert_relative_eq!(s.volume(), volume_map(N, radius));

        Ok(())
    }

    #[rstest]
    fn test_n_factorial(#[values(1, 2, 3, 4)] m: usize) {
        check!(factorial(m, m) == m);
    }
    #[rstest]
    fn test_single_double_factorial(#[values(1, 5, 10, 18, 20)] n: usize) {
        check!(factorial(n, 1) == factorial(n, 2) * factorial(n - 1, 2));
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
    ) -> anyhow::Result<()> {
        let s = Hypersphere::<N> {
            radius: radius.try_into()?,
        };
        let v = Cartesian::<N>::from([radius.powi(2) / 1.8; N]);
        check!(v / v.norm() * radius == s.support_mapping(&v));

        Ok(())
    }

    #[test]
    fn support_mapping() -> anyhow::Result<()> {
        let sphere = Sphere::with_radius(2.0.try_into()?);

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

        Ok(())
    }

    #[test]
    fn intersects_at() -> anyhow::Result<()> {
        let sphere0 = Sphere::with_radius(2.0.try_into()?);
        let sphere1 = Sphere::with_radius(4.0.try_into()?);
        let identity = Versor::default();

        check!(sphere0.intersects_at(&sphere1, &Cartesian::from([0.0, 0.0, 5.9]), &identity));
        check!(sphere0.intersects_at(&sphere1, &Cartesian::from([0.0, 5.9, 0.0]), &identity));
        check!(sphere0.intersects_at(&sphere1, &Cartesian::from([5.9, 0.0, 0.0]), &identity));
        check!(sphere0.intersects_at(&sphere1, &Cartesian::from([3.4, 3.4, 3.4]), &identity));

        check!(!sphere0.intersects_at(&sphere1, &Cartesian::from([0.0, 0.0, 6.1]), &identity));
        check!(!sphere0.intersects_at(&sphere1, &Cartesian::from([0.0, 6.1, 0.0]), &identity));
        check!(!sphere0.intersects_at(&sphere1, &Cartesian::from([6.1, 0.0, 0.0]), &identity));
        check!(!sphere0.intersects_at(&sphere1, &Cartesian::from([3.52, 3.52, 3.52]), &identity));

        let sphere0 = Convex(sphere0);
        let sphere1 = Convex(sphere1);

        check!(sphere0.intersects_at(&sphere1, &Cartesian::from([0.0, 0.0, 5.9]), &identity));
        check!(sphere0.intersects_at(&sphere1, &Cartesian::from([0.0, 5.9, 0.0]), &identity));
        check!(sphere0.intersects_at(&sphere1, &Cartesian::from([5.9, 0.0, 0.0]), &identity));
        check!(sphere0.intersects_at(&sphere1, &Cartesian::from([3.4, 3.4, 3.4]), &identity));

        check!(!sphere0.intersects_at(&sphere1, &Cartesian::from([0.0, 0.0, 6.1]), &identity));
        check!(!sphere0.intersects_at(&sphere1, &Cartesian::from([0.0, 6.1, 0.0]), &identity));
        check!(!sphere0.intersects_at(&sphere1, &Cartesian::from([6.1, 0.0, 0.0]), &identity));
        check!(!sphere0.intersects_at(&sphere1, &Cartesian::from([3.52, 3.52, 3.52]), &identity));

        Ok(())
    }

    #[test]
    fn is_point_inside() -> anyhow::Result<()> {
        let circle = Circle::with_radius(2.0.try_into()?);

        check!(circle.is_point_inside(&Cartesian::from([0.0, 0.0])));
        check!(circle.is_point_inside(&Cartesian::from([0.0, 1.0])));
        check!(circle.is_point_inside(&Cartesian::from([0.0, -1.0])));
        check!(circle.is_point_inside(&Cartesian::from([1.0, 0.0])));
        check!(circle.is_point_inside(&Cartesian::from([-1.0, 0.0])));
        check!(circle.is_point_inside(&Cartesian::from([2.0_f64.next_down(), 0.0])));
        check!(circle.is_point_inside(&Cartesian::from([0.0, 2.0_f64.next_down()])));

        check!(!circle.is_point_inside(&Cartesian::from([2.0, 0.0])));
        check!(!circle.is_point_inside(&Cartesian::from([0.0, 2.0])));
        check!(!circle.is_point_inside(&Cartesian::from([1.5, 1.5])));

        Ok(())
    }

    #[test]
    fn distribution() -> anyhow::Result<()> {
        let circle = Circle::with_radius(4.0.try_into()?);
        let mut rng = StdRng::seed_from_u64(4);

        let points: Vec<_> = (&circle).sample_iter(&mut rng).take(N).collect();
        check!(&points.iter().all(|p| circle.is_point_inside(p)));
        check!(&points.iter().any(|p| p.dot(p) > 3.9));

        Ok(())
    }

    #[test]
    fn test_scale_length() -> anyhow::Result<()> {
        let circle = Circle::with_radius(4.0.try_into()?);

        let scaled_circle = circle.scale_length(2.0.try_into()?);
        check!(scaled_circle.radius.get() == 8.0);

        let scaled_circle = circle.scale_length(0.5.try_into()?);
        check!(scaled_circle.radius.get() == 2.0);

        Ok(())
    }

    #[test]
    fn test_scale_volume() -> anyhow::Result<()> {
        let circle = Circle::with_radius(4.0.try_into()?);

        let scaled_circle = circle.scale_volume(4.0.try_into()?);
        check!(scaled_circle.radius.get() == 8.0);

        let scaled_circle = circle.scale_volume(0.25.try_into()?);
        check!(scaled_circle.radius.get() == 2.0);

        Ok(())
    }

    #[test]
    fn test_map_basic() -> anyhow::Result<()> {
        let circle_a = Circle::with_radius(4.0.try_into()?);
        let circle_b = Circle::with_radius(8.0.try_into()?);

        check!(
            circle_a.map_point(Cartesian::from([0.0, 0.0]), &circle_b)
                == Ok(Cartesian::from([0.0, 0.0]))
        );
        check!(
            circle_b.map_point(Cartesian::from([0.0, 0.0]), &circle_a)
                == Ok(Cartesian::from([0.0, 0.0]))
        );

        check!(
            circle_a.map_point(Cartesian::from([100.0, 0.0]), &circle_b)
                == Err(Error::PointOutsideShape)
        );
        check!(
            circle_b.map_point(Cartesian::from([0.0, -200.0]), &circle_a)
                == Err(Error::PointOutsideShape)
        );

        check!(
            circle_a.map_point(Cartesian::from([3.0, 0.0]), &circle_b)
                == Ok(Cartesian::from([6.0, 0.0]))
        );
        check!(
            circle_b.map_point(Cartesian::from([-6.0, 0.0]), &circle_a)
                == Ok(Cartesian::from([-3.0, 0.0]))
        );

        check!(
            circle_a.map_point(Cartesian::from([-1.0, 2.0]), &circle_b)
                == Ok(Cartesian::from([-2.0, 4.0]))
        );
        check!(
            circle_b.map_point(Cartesian::from([2.0, -4.0]), &circle_a)
                == Ok(Cartesian::from([1.0, -2.0]))
        );

        Ok(())
    }

    #[test]
    fn test_map_surface() -> anyhow::Result<()> {
        let mut rng = StdRng::seed_from_u64(3);
        let uniform_radius = Uniform::new(1.0, 1000.0)?;
        let uniform_angle = Uniform::new(0.0, 2.0 * PI)?;

        for _ in 0..16384 {
            let a = uniform_radius.sample(&mut rng);
            let b = uniform_radius.sample(&mut rng);
            let circle_a = Circle::with_radius(a.try_into()?);
            let circle_b = Circle::with_radius(b.try_into()?);

            // Test that points right on the boundary of one shape remain inside
            // the other shape. If not implemented correctly, map_point might
            // round  and place a point just outside the shape. This test fails
            // without corner case handling in `map_point`.

            let left = circle_a.map_point(Cartesian::from([(-a).next_up(), 0.0]), &circle_b)?;
            check!(
                circle_b.is_point_inside(&left),
                "{left:?} should be inside {circle_b:?}"
            );

            let right = circle_a.map_point(Cartesian::from([a.next_down(), 0.0]), &circle_b)?;
            check!(
                circle_b.is_point_inside(&right),
                "{right:?} should be inside {circle_b:?}"
            );

            let bottom = circle_a.map_point(Cartesian::from([0.0, (-a).next_up()]), &circle_b)?;
            check!(
                circle_b.is_point_inside(&bottom),
                "{bottom:?} should be inside {circle_b:?}"
            );

            let top = circle_a.map_point(Cartesian::from([0.0, a.next_down()]), &circle_b)?;
            check!(
                circle_b.is_point_inside(&top),
                "{top:?} should be inside {circle_b:?}"
            );

            for _ in 0..32 {
                let theta = uniform_angle.sample(&mut rng);
                let point = Cartesian::from([a * theta.cos(), b * theta.sin()]);

                // `point` may be rounded in such a way that it falls outside the shape.
                // Skip these cases. 32 iterations is sufficient to produce many interior
                // points.
                if !circle_a.is_point_inside(&point) {
                    continue;
                }

                let mapped_point = circle_a.map_point(point, &circle_b)?;
                check!(
                    circle_b.is_point_inside(&mapped_point),
                    "{mapped_point:?} should be inside {circle_b:?}"
                );
            }
        }

        Ok(())
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Hypersphere`] */
use crate::{BoundingSphereRadius, IntersectsAt, SupportMapping, Volume};
use hoomd_vector::{InnerProduct, Rotate};
use hoomd_utility::valid::PositiveReal;

use std::f64::consts::PI;
use std::ops::Mul;

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

/** All points within a given `radius` from the origin.

# Examples

Basic construction and methods:
```
use hoomd_geometry::{shape::Hypersphere, SupportMapping, Volume};
use hoomd_vector::Cartesian;
use std::f64::consts::PI;

let unit_sphere = Hypersphere::<3>::default();
let volume = unit_sphere.volume();

assert_eq!(unit_sphere.radius, 1.0);
assert_eq!(volume, 4.0 * PI / 3.0);

assert_eq!(
    unit_sphere.support_mapping(&Cartesian::from([1.0; 3])),
    [1.0 / f64::sqrt(3.0); 3].into()
)
```

Test for intersections:
```
use hoomd_geometry::{IntersectsAt, shape::Hypersphere};
use hoomd_vector::{Cartesian, Versor};

let unit_sphere = Hypersphere::<3>::default();

assert_eq!(
    unit_sphere.intersects_at(&unit_sphere, &Cartesian::from([2.1, 0.0, 0.0]), &Versor::default()),
    false
);
assert_eq!(
    unit_sphere.intersects_at(&unit_sphere, &Cartesian::from([0.0, 1.9, 0.0]), &Versor::default()),
    true
);
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hypersphere<const N: usize> {
    /// Radius of the sphere
    pub radius: PositiveReal,
}

/** A circle in two dimensions.

# Examples

Basic construction and methods:
```
use hoomd_geometry::{shape::Circle, Volume};
use hoomd_vector::Cartesian;
use std::f64::consts::PI;

let circle = Circle { radius: 2.0 };
let volume = circle.volume();

assert_eq!(volume, PI * 4.0);
```

Test for intersections:
```
use hoomd_geometry::{IntersectsAt, shape::Circle};
use hoomd_vector::{Cartesian, Angle};

let circle = Circle { radius: 1.0 };

assert_eq!(
    circle.intersects_at(&circle, &Cartesian::from([2.1, 0.0]), &Angle::default()),
    false
);
assert_eq!(
    circle.intersects_at(&circle, &Cartesian::from([0.0, 1.9]), &Angle::default()),
    true
);
```
*/
pub type Circle = Hypersphere<2>;

/** A sphere in three dimensions.

# Examples

Basic construction and methods:
```
use hoomd_geometry::{shape::Sphere, SupportMapping, Volume};
use hoomd_vector::Cartesian;
use std::f64::consts::PI;

let unit_sphere = Sphere { radius: 1.0 };
let volume = unit_sphere.volume();

assert_eq!(unit_sphere.radius, 1.0);
assert_eq!(volume, 4.0 * PI / 3.0);

assert_eq!(
    unit_sphere.support_mapping(&Cartesian::from([1.0; 3])),
    [1.0 / f64::sqrt(3.0); 3].into()
)
```

Test for intersections:
```
use hoomd_geometry::{IntersectsAt, shape::Sphere};
use hoomd_vector::{Cartesian, Versor};

let unit_sphere = Sphere::default();

assert_eq!(
    unit_sphere.intersects_at(&unit_sphere, &Cartesian::from([2.1, 0.0, 0.0]), &Versor::default()),
    false
);
assert_eq!(
    unit_sphere.intersects_at(&unit_sphere, &Cartesian::from([0.0, 1.9, 0.0]), &Versor::default()),
    true
);
```
*/
pub type Sphere = Hypersphere<3>;

impl<const N: usize> Default for Hypersphere<N> {
    #[inline]
    fn default() -> Self {
        Hypersphere { radius: 1.0.try_into().expect("1.0 is a positive real") }
    }
}

impl<const N: usize> Hypersphere<N> {
    /// Create a sphere with a given positive real radius.
    #[must_use]
    #[inline]
    pub fn with_radius(radius: PositiveReal) -> Self {
        Hypersphere { radius }
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
                .radius.get()
                .powi(N.try_into().expect("Dimension should not overflow i32!"))
    }
}

impl<const N: usize, V: InnerProduct, R: Rotate<V>> IntersectsAt<Hypersphere<N>, V, R>
    for Hypersphere<N>
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
    use approx::assert_relative_eq;
    use hoomd_vector::{Cartesian, Versor};
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
        #[values(0.01, 1.0, 33.3, 1e6)] radius: f64,
    ) {
        let s = Hypersphere::<N> { radius: radius.try_into().expect("test value is a positive real") };

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
        let s = Hypersphere::<N> { radius: radius.try_into().expect("test value is a positive real") };
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

        assert!(Convex(sphere0).intersects_at(
            &Convex(sphere1),
            &[0.0, 0.0, 5.9].into(),
            &identity
        ));
        assert!(Convex(sphere0).intersects_at(
            &Convex(sphere1),
            &[0.0, 5.9, 0.0].into(),
            &identity
        ));
        assert!(Convex(sphere0).intersects_at(
            &Convex(sphere1),
            &[5.9, 0.0, 0.0].into(),
            &identity
        ));
        assert!(Convex(sphere0).intersects_at(
            &Convex(sphere1),
            &[3.4, 3.4, 3.4].into(),
            &identity
        ));

        assert!(!Convex(sphere0).intersects_at(
            &Convex(sphere1),
            &[0.0, 0.0, 6.1].into(),
            &identity
        ));
        assert!(!Convex(sphere0).intersects_at(
            &Convex(sphere1),
            &[0.0, 6.1, 0.0].into(),
            &identity
        ));
        assert!(!Convex(sphere0).intersects_at(
            &Convex(sphere1),
            &[6.1, 0.0, 0.0].into(),
            &identity
        ));
        assert!(!Convex(sphere0).intersects_at(
            &Convex(sphere1),
            &[3.52, 3.52, 3.52].into(),
            &identity
        ));
    }
}

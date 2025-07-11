// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Capsule`] */

use crate::{BoundingSphereRadius, SupportMapping, Volume};

use hoomd_vector::{Cartesian, InnerProduct};

use super::sphere::sphere_volume_prefactor;

/** All points less than or equal to a distance `r` from a line segment of length `h`.

This line is oriented along the `[0 0 ... 1]` direction, and has extents `+h/2`,
`-h/2` along that axis.

# Examples

Construction and basic methods:
```
use hoomd_geometry::{BoundingSphereRadius, shape::Capsule, Volume};
use hoomd_vector::Cartesian;
use approx::assert_relative_eq;
use std::f64::consts::PI;

let capsule = Capsule::<2> { radius: 1.0, height: 8.0 };
let bounding_radius = capsule.bounding_sphere_radius();
let volume = capsule.volume();

assert_eq!(bounding_radius, 5.0);
assert_relative_eq!(volume, 16.0 + PI);
```

Intersection test:
```
use hoomd_geometry::{Convex, IntersectsAt, shape::Capsule};
use hoomd_vector::{Angle, Cartesian, Rotation};
use std::f64::consts::PI;

let capsule = Convex(Capsule::<2> { radius: 1.0, height: 8.0 });

assert_eq!(capsule.intersects_at(&capsule, &[1.75, 0.0].into(), &Angle::identity()), true);
assert_eq!(capsule.intersects_at(&capsule, &[4.0, 0.0].into(), &Angle::identity()), false);
assert_eq!(capsule.intersects_at(&capsule, &[4.0, 0.0].into(), &Angle::from(PI/2.0)), true);
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule<const N: usize> {
    /// Radius of of points that are considered enclosed in the shape.
    pub radius: f64,
    /// Length of the line segment.
    pub height: f64,
}

impl<const N: usize> SupportMapping<Cartesian<N>> for Capsule<N> {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        // Same support function as a ConvexPolyhedron with 2 vertices, plus the radius.
        let mut v_tip = [0.0; N];
        v_tip[N - 1] = self.height / 2.0;
        let v_tip = v_tip.into();

        let mut v_base = [0.0; N];
        v_base[N - 1] = -self.height / 2.0;
        let v_base = v_base.into();

        let (v_tip_dot_n, v_base_dot_n) = (n.dot(&v_tip), n.dot(&v_base));

        let rshift = *n * self.radius * n.norm();
        if v_tip_dot_n > v_base_dot_n {
            v_tip / n.norm() + rshift
        } else {
            v_base / n.norm() + rshift
        }
    }
}

impl<const N: usize> BoundingSphereRadius for Capsule<N> {
    #[inline]
    fn bounding_sphere_radius(&self) -> f64 {
        self.height / 2.0 + self.radius
    }
}

impl<const N: usize> Volume for Capsule<N> {
    #[inline]
    fn volume(&self) -> f64 {
        if N == 0 {
            return 0.0;
        }
        let r_n_minus_one = self.radius.powi(
            (N - 1)
                .try_into()
                .expect("dimension {N}-1 should fit in an i32"),
        );
        let cylinder_volume = sphere_volume_prefactor(N - 1) * r_n_minus_one * self.height;
        cylinder_volume + sphere_volume_prefactor(N) * (r_n_minus_one * self.radius)
    }
}

#[expect(clippy::used_underscore_binding, reason = "Required for const tests.")]
#[cfg(test)]
mod tests {

    use crate::shape::{Cylinder, Hypersphere};

    use super::*;
    use approx::assert_relative_eq;
    use rstest::*;
    use std::marker::PhantomData;

    #[rstest(
        _n => [
            PhantomData::<Capsule<1>>,
            PhantomData::<Capsule<2>>,
            PhantomData::<Capsule<3>>,
            PhantomData::<Capsule<4>>,
            PhantomData::<Capsule<5>>
        ],
        radius => [0.0, 1e-6, 1.0, 34.56],
    )]
    fn test_capsule_volume<const N: usize>(_n: PhantomData<Capsule<N>>, radius: f64) {
        assert_relative_eq!(
            Capsule::<N> {
                radius,
                height: 0.0
            }
            .volume(),
            Hypersphere::<N> { radius }.volume(),
            epsilon = 1e-6
        );
    }
    #[rstest(
        radius => [0.0, 1e-6, 1.0, 34.56],
        height => [0.0, 1e-6, 1.0, 34.56],
    )]
    fn test_elongated_capsule_volume(radius: f64, height: f64) {
        let cap = Capsule::<3> { radius, height };
        assert_relative_eq!(
            cap.volume(),
            Hypersphere::<3> { radius }.volume()
                + Cylinder {
                    radius,
                    height: cap.height
                }
                .volume()
        );
    }
}

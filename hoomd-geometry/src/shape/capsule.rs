// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Capsule`]

use super::sphere::sphere_volume_prefactor;
use crate::{BoundingSphereRadius, SupportMapping, Volume};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct};

/// All points less than or equal to a distance `r` from a line segment of length `h`.
///
/// This line is oriented along the `[0 0 ... 1]` direction, and has extents `+h/2`,
/// `-h/2` along that axis.
///
/// # Examples
///
/// Construction and basic methods:
/// ```
/// use approxim::assert_relative_eq;
/// use hoomd_geometry::{BoundingSphereRadius, Volume, shape::Capsule};
/// use hoomd_vector::Cartesian;
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let capsule = Capsule::<2> {
///     radius: 1.0.try_into()?,
///     height: 8.0.try_into()?,
/// };
/// let bounding_radius = capsule.bounding_sphere_radius();
/// let volume = capsule.volume();
///
/// assert_eq!(bounding_radius.get(), 5.0);
/// assert_relative_eq!(volume, 16.0 + PI);
/// # Ok(())
/// # }
/// ```
///
/// Intersection test:
/// ```
/// use hoomd_geometry::{Convex, IntersectsAt, shape::Capsule};
/// use hoomd_vector::{Angle, Cartesian, Rotation};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let capsule = Convex(Capsule::<2> {
///     radius: 1.0.try_into()?,
///     height: 8.0.try_into()?,
/// });
///
/// assert!(capsule.intersects_at(
///     &capsule,
///     &[1.75, 0.0].into(),
///     &Angle::identity()
/// ));
/// assert!(!capsule.intersects_at(
///     &capsule,
///     &[4.0, 2.0].into(),
///     &Angle::identity()
/// ),);
/// assert!(capsule.intersects_at(
///     &capsule,
///     &[4.0, -2.0].into(),
///     &Angle::from(PI / 2.0)
/// ));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Capsule<const N: usize> {
    /// Radius of of points that are considered enclosed in the shape.
    pub radius: PositiveReal,
    /// Length of the line segment.
    pub height: PositiveReal,
}

impl<const N: usize> SupportMapping<Cartesian<N>> for Capsule<N> {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        // Same support function as a ConvexPolyhedron with 2 vertices, plus the radius.
        let mut v_tip = [0.0; N];
        v_tip[N - 1] = self.height.get() / 2.0;
        let v_tip = v_tip.into();

        let mut v_base = [0.0; N];
        v_base[N - 1] = -self.height.get() / 2.0;
        let v_base = v_base.into();

        let (v_tip_dot_n, v_base_dot_n) = (n.dot(&v_tip), n.dot(&v_base));

        let rshift = *n / n.norm() * self.radius.get();
        if v_tip_dot_n > v_base_dot_n {
            v_tip + rshift
        } else {
            v_base + rshift
        }
    }
}

impl<const N: usize> BoundingSphereRadius for Capsule<N> {
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        (self.height.get() / 2.0 + self.radius.get())
            .try_into()
            .expect("this expression should evaluate to a positive real")
    }
}

impl<const N: usize> Volume for Capsule<N> {
    #[inline]
    fn volume(&self) -> f64 {
        if N == 0 {
            return 0.0;
        }
        let r_n_minus_one = self.radius.get().powi(
            (N - 1)
                .try_into()
                .expect("dimension {N}-1 should fit in an i32"),
        );
        let cylinder_volume = sphere_volume_prefactor(N - 1) * r_n_minus_one * self.height.get();
        cylinder_volume + sphere_volume_prefactor(N) * (r_n_minus_one * self.radius.get())
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        Convex, IntersectsAt,
        shape::{Circle, Cylinder, Hypersphere},
    };
    use hoomd_vector::{Angle, Versor};

    use super::*;
    use approxim::assert_relative_eq;
    use rstest::*;
    use std::f64::consts::PI;

    #[rstest(
        radius => [1e-6, 1.0, 34.56],
        height => [1e-6, 1.0, 34.56],
    )]
    fn test_elongated_capsule_volume(radius: f64, height: f64) {
        let capsule = Capsule::<3> {
            radius: radius.try_into().expect("test value is a positive real"),
            height: height.try_into().expect("test value is a positive real"),
        };
        assert_relative_eq!(
            capsule.volume(),
            Hypersphere::<3> {
                radius: radius.try_into().expect("test value is a positive real")
            }
            .volume()
                + Cylinder {
                    radius: radius.try_into().expect("test value is a positive real"),
                    height: capsule.height
                }
                .volume()
        );

        assert_relative_eq!(
            capsule.bounding_sphere_radius().get(),
            radius + height / 2.0
        );
    }

    #[test]
    fn intersect_xenocollide_2d() {
        let capsule_tall = Convex(Capsule::<2> {
            radius: 0.5.try_into().expect("test value is a positive real"),
            height: 6.0.try_into().expect("test value is a positive real"),
        });

        let circle = Convex(Circle::with_radius(
            0.5.try_into().expect("test value is a positive real"),
        ));

        let identity = Angle::default();
        let rotate = Angle::from(PI / 2.0);

        assert!(!capsule_tall.intersects_at(&circle, &[0.0, 4.1].into(), &identity));
        assert!(capsule_tall.intersects_at(&circle, &[0.0, 3.9].into(), &identity));
        assert!(!circle.intersects_at(&capsule_tall, &[0.0, 4.1].into(), &identity));
        assert!(circle.intersects_at(&capsule_tall, &[0.0, 3.9].into(), &identity));
        assert!(!circle.intersects_at(&capsule_tall, &[4.1, 0.0].into(), &rotate));
        assert!(circle.intersects_at(&capsule_tall, &[3.9, 0.0].into(), &rotate));

        assert!(capsule_tall.intersects_at(&capsule_tall, &[0.2, -0.4].into(), &rotate));
        assert!(capsule_tall.intersects_at(&capsule_tall, &[3.9, 2.0].into(), &rotate));
        assert!(!capsule_tall.intersects_at(&capsule_tall, &[4.1, -2.0].into(), &rotate));
    }

    #[test]
    fn intersect_xenocollide_3d() {
        let capsule_tall = Convex(Capsule::<3> {
            radius: 0.5.try_into().expect("test value is a positive real"),
            height: 6.0.try_into().expect("test value is a positive real"),
        });

        let sphere = Convex(Circle::with_radius(
            0.5.try_into().expect("test value is a positive real"),
        ));

        let identity = Versor::default();
        let rotate = Versor::from_axis_angle(
            [0.0, 1.0, 0.0]
                .try_into()
                .expect("hard-coded vector is non-zero"),
            PI / 2.0,
        );

        assert!(!capsule_tall.intersects_at(&sphere, &[0.0, 0.0, 4.1].into(), &identity));
        assert!(capsule_tall.intersects_at(&sphere, &[0.0, 0.0, 3.9].into(), &identity));
        assert!(!sphere.intersects_at(&capsule_tall, &[0.0, 0.0, 4.1].into(), &identity));
        assert!(sphere.intersects_at(&capsule_tall, &[0.0, 0.0, 3.9].into(), &identity));
        assert!(!sphere.intersects_at(&capsule_tall, &[4.1, 0.0, 0.0].into(), &rotate));
        assert!(sphere.intersects_at(&capsule_tall, &[3.9, 0.0, 0.0].into(), &rotate));

        assert!(capsule_tall.intersects_at(&capsule_tall, &[0.2, -0.4, 0.0].into(), &rotate));
        assert!(capsule_tall.intersects_at(&capsule_tall, &[3.9, 0.0, 2.0].into(), &rotate));
        assert!(!capsule_tall.intersects_at(&capsule_tall, &[4.1, 0.0, -2.0].into(), &rotate));
    }

    #[test]
    fn support_mapping_2d() {
        let capsule = Convex(Capsule::<3> {
            radius: 0.5.try_into().expect("test value is a positive real"),
            height: 6.0.try_into().expect("test value is a positive real"),
        });

        // top and bottom
        assert_relative_eq!(
            capsule.support_mapping(&[0.0, 0.0, 1.0].into()),
            [0.0, 0.0, 3.5].into()
        );
        assert_relative_eq!(
            capsule.support_mapping(&[0.0, 0.0, -1.0].into()),
            [0.0, 0.0, -3.5].into()
        );

        // the top ring
        assert_relative_eq!(
            capsule.support_mapping(&[1.0, 0.0, 1e-12].into()),
            [0.5, 0.0, 3.0].into(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            capsule.support_mapping(&[-1.0, 0.0, 1e-12].into()),
            [-0.5, 0.0, 3.0].into(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            capsule.support_mapping(&[0.0, 1.0, 1e-12].into()),
            [0.0, 0.5, 3.0].into(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            capsule.support_mapping(&[0.0, -1.0, 1e-12].into()),
            [0.0, -0.5, 3.0].into(),
            epsilon = 1e-6
        );

        // the bottom ring
        assert_relative_eq!(
            capsule.support_mapping(&[1.0, 0.0, -1e-12].into()),
            [0.5, 0.0, -3.0].into(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            capsule.support_mapping(&[-1.0, 0.0, -1e-12].into()),
            [-0.5, 0.0, -3.0].into(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            capsule.support_mapping(&[0.0, 1.0, -1e-12].into()),
            [0.0, 0.5, -3.0].into(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            capsule.support_mapping(&[0.0, -1.0, -1e-12].into()),
            [0.0, -0.5, -3.0].into(),
            epsilon = 1e-6
        );

        // on the caps is not so easy to test manually...
    }
}

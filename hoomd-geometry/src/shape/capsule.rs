// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Capsule`]

use super::sphere::sphere_volume_prefactor;
use crate::{BoundingSphereRadius, IntersectsAt, SupportMapping, Volume};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct, Rotate, Rotation};

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

/// .
#[inline]
fn axis_aligned_cartesian<const N: usize>(h: f64) -> Cartesian<N> {
    Cartesian::from(std::array::from_fn(|i| if i == (N - 1) { h } else { 0.0 }))
}

impl<const N: usize, R> IntersectsAt<Capsule<N>, Cartesian<N>, R> for Capsule<N>
where
    R: Rotate<Cartesian<N>> + Rotation,
    Cartesian<N>: From<[f64; N]>,
{
    #[inline]
    fn intersects_at(&self, other: &Capsule<N>, v_ij: &Cartesian<N>, o_ij: &R) -> bool {
        // Adapted from Real Time Collision Detection, D. Eberly, pp 148
        // Note we ignore fallbacks when the capsule length is small, as these
        // could be valid inputs. If we somehow underflow to NaN, a (possibly spurious)
        // overlap will be detected.

        let d1 = axis_aligned_cartesian::<N>(self.height.get());
        let p1 = d1 * -0.5;

        let d2 = o_ij.rotate(&axis_aligned_cartesian(other.height.get()));
        let p2 = *v_ij - d2 * 0.5;

        let distance_between_centers = p1 - p2;

        let d1_norm_sq = d1.dot(&d1);
        let d2_norm_sq = d2.dot(&d2);
        let f = d2.dot(&distance_between_centers);

        let c = d1.dot(&distance_between_centers);

        // The general nondegenerate case - very small capsules are valid.
        let d1_dot_d2 = d1.dot(&d2);
        let denom = d1_norm_sq * d2_norm_sq - d1_dot_d2 * d1_dot_d2;

        // If segments not parallel, compute closest point on L1 to L2 and
        // clamp to segment S1. Else pick arbitrary s (here 0)
        let s = if denom == 0.0 {
            0.0
        } else {
            ((d1_dot_d2 * f - c * d2_norm_sq) / denom).clamp(0.0, 1.0)
        };

        // Compute point on L2 closest to S1(s) using
        // t = Dot((P1 + D1*s) - P2,D2) / Dot(D2,D2) = (b*s + f) / e
        let tnom = d1_dot_d2 * s + f;
        let (t, s) = if tnom < 0.0 {
            (0.0, (-c / d1_norm_sq).clamp(0.0, 1.0))
        } else if tnom > d2_norm_sq {
            (1.0, ((d1_dot_d2 - c) / d1_norm_sq).clamp(0.0, 1.0))
        } else {
            (tnom / d2_norm_sq, s)
        };

        let (c1, c2) = (p1 + d1 * s, p2 + d2 * t);
        let dist_sq = (c1 - c2).norm_squared();

        let total_radius = self.radius.get() + other.radius.get();
        dist_sq <= total_radius.powi(2)
    }
}
#[cfg(test)]
mod tests {

    use crate::{
        Convex, IntersectsAt,
        shape::{Circle, Cylinder, Hypersphere},
    };
    use hoomd_vector::{Angle, Versor};
    use rand::{Rng, SeedableRng};

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

    #[rstest]
    #[case(true, 1.999_999, 0.0, 0.0)]
    #[case(true, 2.0, 0.0, 0.0)]
    #[case(false, 2.000_001, 0.0, 0.0)]
    fn test_intersect_capsule_capsule_2d(
        #[case] expected: bool,
        #[case] x: f64,
        #[case] y: f64,
        #[case] angle: f64,
    ) {
        let capsule1 = Capsule::<2> {
            radius: 1.0.try_into().unwrap(),
            height: 2.0.try_into().unwrap(),
        };
        let capsule2 = Capsule::<2> {
            radius: 1.0.try_into().unwrap(),
            height: 2.0.try_into().unwrap(),
        };

        let v_ij = [x, y].into();
        let o_ij = Angle::from(angle);
        assert_eq!(capsule1.intersects_at(&capsule2, &v_ij, &o_ij), expected);
        assert_eq!(
            capsule2.intersects_at(&capsule1, &(-v_ij), &o_ij.inverted()),
            expected
        );
    }

    #[rstest]
    #[case(true, 0.0, 2.0, 90.0)]
    #[case(true, 0.0, 3.0, 90.0)]
    #[case(false, 0.0, 3.000_001, 90.0)]
    fn test_intersect_capsule_capsule_2d_rotated(
        #[case] expected: bool,
        #[case] x: f64,
        #[case] y: f64,
        #[case] angle: f64,
    ) {
        let capsule1 = Capsule::<2> {
            radius: 1.0.try_into().unwrap(),
            height: 2.0.try_into().unwrap(),
        };
        let capsule2 = Capsule::<2> {
            radius: 1.0.try_into().unwrap(),
            height: 2.0.try_into().unwrap(),
        };

        let v_ij = [x, y].into();
        let o_ij = Angle::from(angle.to_radians());
        assert_eq!(capsule1.intersects_at(&capsule2, &v_ij, &o_ij), expected);
        assert_eq!(
            capsule2.intersects_at(&capsule1, &(-v_ij), &o_ij.inverted()),
            expected
        );
    }

    /// Nearly-0 height
    #[test]
    fn test_intersect_degenerate_capsules() {
        let sphere1 = Capsule::<3> {
            radius: 1.0.try_into().unwrap(),
            height: 1e-12.try_into().unwrap(),
        };
        let sphere2 = Capsule::<3> {
            radius: 1.0.try_into().unwrap(),
            height: 1e-12.try_into().unwrap(),
        };

        let o_ij = Versor::identity();
        // Intersecting
        let v_ij = [1.999_999, 0.0, 0.0].into();
        assert!(sphere1.intersects_at(&sphere2, &v_ij, &o_ij));
        // Touching
        let v_ij = [2.0, 0.0, 0.0].into();
        assert!(sphere1.intersects_at(&sphere2, &v_ij, &o_ij));
        // Not intersecting
        let v_ij = [2.000_001, 0.0, 0.0].into();
        assert!(!sphere1.intersects_at(&sphere2, &v_ij, &o_ij));

        let capsule = Capsule::<3> {
            radius: 1.0.try_into().unwrap(),
            height: 2.0.try_into().unwrap(),
        };
        // Intersecting
        let v_ij = [1.999_999, 0.0, 0.0].into();
        assert!(sphere1.intersects_at(&capsule, &v_ij, &o_ij));
        assert!(capsule.intersects_at(&sphere1, &v_ij, &o_ij));
        // Touching
        let v_ij = [2.0, 0.0, 0.0].into();
        assert!(sphere1.intersects_at(&capsule, &v_ij, &o_ij));
        assert!(capsule.intersects_at(&sphere1, &v_ij, &o_ij));
        // Not intersecting
        let v_ij = [2.000_001, 0.0, 0.0].into();
        assert!(!sphere1.intersects_at(&capsule, &v_ij, &o_ij));
        assert!(!capsule.intersects_at(&sphere1, &v_ij, &o_ij));
    }

    #[test]
    fn test_intersect_capsule_capsule_complex_3d_random() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);

        for _ in 0..10_000 {
            let r1 = rng.random_range(0.1..10.0);
            let h1 = rng.random_range(0.1..10.0);
            let r2 = rng.random_range(0.1..10.0);
            let h2 = rng.random_range(0.1..10.0);

            let capsule1 = Capsule::<3> {
                radius: r1.try_into()?,
                height: h1.try_into()?,
            };
            let capsule2 = Capsule::<3> {
                radius: r2.try_into()?,
                height: h2.try_into()?,
            };

            let v_ij = (rng.random::<Cartesian<3>>() * 10.0) - Cartesian::from([5.0; 3]);

            let o_ij = rng.random::<Versor>();

            let result_direct = capsule1.intersects_at(&capsule2, &v_ij, &o_ij);

            let result_xeno = Convex(capsule1).intersects_at(&Convex(capsule2), &v_ij, &o_ij);
            assert_eq!(
                result_direct, result_xeno,
                "Failed with r1={r1}, h1={h1}, r2={r2}, h2={h2}, v_ij={v_ij:?}, o_ij={o_ij:?}"
            );
        }
        Ok(())
    }
}

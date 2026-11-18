// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Capsule`]

use serde::{Deserialize, Serialize};

use super::sphere::sphere_volume_prefactor;
use crate::{
    BoundingSphereRadius, IntersectsAt, IntersectsAtGlobal, IsPointInside, Scale, SupportMapping, Volume,
};
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
/// let capsule = Capsule::<2> {
///     radius: 1.0.try_into()?,
///     height: 8.0.try_into()?,
/// };
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Capsule<const N: usize> {
    /// Radius of points that are considered enclosed in the shape.
    pub radius: PositiveReal,
    /// Length of the line segment.
    pub height: PositiveReal,
}

/// A sphere swept along a line segment in $`\mathbb{R}^3`$.
pub type Spherocylinder = Capsule<3>;

impl<const N: usize> SupportMapping<Cartesian<N>> for Capsule<N> {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        const { assert!(N > 0, "N must be greater than 0") };
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

impl<const N: usize> Scale for Capsule<N> {
    /// Construct a scaled [`Capsule`].
    ///
    /// The resulting capsule's radius and height are the original's
    /// scaled by *v*:
    /// ```math
    /// \begin{align*}
    /// r_\mathrm{new} &= v r \\
    /// h_\mathrm{new} &= v h \\
    /// \end{align*}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Capsule};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let capsule = Capsule::<3> {
    ///     radius: 2.0.try_into()?,
    ///     height: 3.0.try_into()?,
    /// };
    ///
    /// let scaled_capsule = capsule.scale_length(2.0.try_into()?);
    ///
    /// assert_eq!(scaled_capsule.radius.get(), 4.0);
    /// assert_eq!(scaled_capsule.height.get(), 6.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        Self {
            radius: self.radius * v,
            height: self.height * v,
        }
    }

    /// Construct a scaled [`Capsule`].
    ///
    /// The resulting capsule's radius and height are the original's
    /// scaled by $`v^\frac{1}{N}`$:
    /// ```math
    /// \begin{align*}
    /// r_\mathrm{new} &= v^\frac{1}{N} r \\
    /// h_\mathrm{new} &= v^\frac{1}{N} h \\
    /// \end{align*}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Capsule};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let capsule = Capsule::<3> {
    ///     radius: 2.0.try_into()?,
    ///     height: 3.0.try_into()?,
    /// };
    ///
    /// let scaled_capsule = capsule.scale_volume(8.0.try_into()?);
    ///
    /// assert_eq!(scaled_capsule.radius.get(), 4.0);
    /// assert_eq!(scaled_capsule.height.get(), 6.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let v = v.get().powf(1.0 / N as f64);
        self.scale_length(v.try_into().expect("v^{1/N} should be a positive real"))
    }
}

impl<const N: usize> IsPointInside<Cartesian<N>> for Capsule<N> {
    /// Check if a vector is inside a capsule.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::Capsule};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let capsule = Capsule {
    ///     radius: 1.0.try_into()?,
    ///     height: 4.0.try_into()?,
    /// };
    ///
    /// assert!(capsule.is_point_inside(&Cartesian::from([0.25, 0.0, 2.75])));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn is_point_inside(&self, point: &Cartesian<N>) -> bool {
        const { assert!(N > 0, "N must be greater than 0") };

        if point[N-1].abs() < self.height.get() / 2.0 {
            let dot: f64 = point.coordinates[0..N-1].iter().map(|x| x.powi(2)).sum();
            dot < self.radius.get().powi(2)
        } else {
            let delta_r_top = *point - axis_aligned_cartesian(self.height.get()/2.0);
            let delta_r_bottom = *point - axis_aligned_cartesian(-self.height.get()/2.0);

            delta_r_top.norm_squared() < self.radius.get().powi(2) || delta_r_bottom.norm_squared() < self.radius.get().powi(2)
        }
    }
}

/// Create a `Cartesian<N>` with zeros except in the final index.
#[inline]
fn axis_aligned_cartesian<const N: usize>(h: f64) -> Cartesian<N> {
    Cartesian::from(std::array::from_fn(|i| if i == (N - 1) { h } else { 0.0 }))
}

impl<const N: usize, R> IntersectsAtGlobal<Capsule<N>, Cartesian<N>, R> for Capsule<N>
where
    R: Rotation + Rotate<Cartesian<N>>,
{
    #[inline]
    fn intersects_at_global(
        &self,
        other: &Capsule<N>,
        r_self: &Cartesian<N>,
        o_self: &R,
        r_other: &Cartesian<N>,
        o_other: &R,
    ) -> bool {
        let (v_ij, o_ij) = hoomd_vector::pair_system_to_local(r_self, o_self, r_other, o_other);

        self.intersects_at(other, &v_ij, &o_ij)
    }
}

impl<const N: usize, R> IntersectsAt<Capsule<N>, Cartesian<N>, R> for Capsule<N>
where
    R: Rotate<Cartesian<N>>,
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

        let total_radius = self.radius + other.radius;
        dist_sq <= total_radius.get().powi(2)
    }
}
#[cfg(test)]
mod tests {

    use crate::{
        Convex, IntersectsAt,
        shape::{Circle, Cylinder, Hypersphere},
    };
    use hoomd_vector::{Angle, Versor};
    use rand::{RngExt, SeedableRng};

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

    #[test]
    fn test_scale_length() -> Result<(), Box<dyn std::error::Error>> {
        let capsule = Capsule::<3> {
            radius: 4.0.try_into()?,
            height: 6.0.try_into()?,
        };

        let scaled_capsule = capsule.scale_length(2.0.try_into()?);
        assert_relative_eq!(scaled_capsule.radius.get(), 8.0);
        assert_relative_eq!(scaled_capsule.height.get(), 12.0);
        assert_relative_eq!(scaled_capsule.bounding_sphere_radius().get(), 14.0);

        let scaled_capsule = capsule.scale_length(0.5.try_into()?);
        assert_relative_eq!(scaled_capsule.radius.get(), 2.0);
        assert_relative_eq!(scaled_capsule.height.get(), 3.0);
        assert_relative_eq!(scaled_capsule.bounding_sphere_radius().get(), 3.5);

        Ok(())
    }

    #[test]
    fn test_scale_volume() -> Result<(), Box<dyn std::error::Error>> {
        let capsule = Capsule::<3> {
            radius: 4.0.try_into()?,
            height: 6.0.try_into()?,
        };

        let scaled_capsule = capsule.scale_volume(8.0.try_into()?);
        assert_relative_eq!(scaled_capsule.radius.get(), 8.0);
        assert_relative_eq!(scaled_capsule.height.get(), 12.0);
        assert_relative_eq!(scaled_capsule.bounding_sphere_radius().get(), 14.0);

        let scaled_capsule = capsule.scale_volume(0.125.try_into()?);
        assert_relative_eq!(scaled_capsule.radius.get(), 2.0);
        assert_relative_eq!(scaled_capsule.height.get(), 3.0);
        assert_relative_eq!(scaled_capsule.bounding_sphere_radius().get(), 3.5);

        Ok(())
    }

    #[rstest]
    #[case::center([0.0, 0.0, 0.0], true)]
    #[case::left_inside([-0.875, 0.0, 1.5], true)]
    #[case::right_inside([0.875, 0.0, -1.5], true)]
    #[case::left_outside([-1.125, 0.0, 1.5], false)]
    #[case::top_outside([0.0, 1.125, 1.5], false)]
    #[case::up_inside([0.0, 0.0, 2.875], true)]
    #[case::down_inside([0.0, 0.0, -2.875], true)]
    #[case::up_outside([0.0, 0.0, 3.125], false)]
    #[case::down_outside([0.0, 0.0, -3.125], false)]
    #[rstest]
    fn test_is_point_inside(#[case] point: [f64; 3], #[case] expected: bool) -> anyhow::Result<()> {
    let capsule = Capsule {
        radius: 1.0.try_into()?,
        height: 4.0.try_into()?,
    };

    assert_eq!(capsule.is_point_inside(&point.into()), expected);

    Ok(())
    }
}

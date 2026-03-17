// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Convex`.

use serde::{Deserialize, Serialize};

use crate::{
    BoundingSphereRadius, IntersectsAt, IntersectsAtGlobal, SupportMapping,
    shape::{Circle, Sphere},
    xenocollide::{collide2d, collide3d},
};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, Metric, Rotate, Rotation, RotationMatrix};

/// A newtype that checks for intersections using [`xenocollide`](crate::xenocollide).
///
/// Use [`Convex`] to check for intersections between two convex shapes (possibly
/// with different types).
///
/// # Example
///
/// Test if a circle overlaps with a rounded rectangle:
/// ```
/// use hoomd_geometry::{
///     Convex, IntersectsAt,
///     shape::{Circle, Rectangle, Sphero},
/// };
/// use hoomd_vector::{Angle, Cartesian};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let circle = Convex(Circle {
///     radius: 0.5.try_into()?,
/// });
/// let rectangle = Rectangle {
///     edge_lengths: [3.0.try_into()?, 2.0.try_into()?],
/// };
/// let rounded_rectangle = Convex(Sphero {
///     shape: rectangle,
///     rounding_radius: 0.5.try_into()?,
/// });
///
/// assert!(rounded_rectangle.intersects_at(
///     &circle,
///     &[2.4, 0.0].into(),
///     &Angle::default()
/// ));
/// assert!(!rounded_rectangle.intersects_at(
///     &circle,
///     &[0.0, 2.4].into(),
///     &Angle::default()
/// ));
/// assert!(circle.intersects_at(
///     &rounded_rectangle,
///     &[0.0, 2.4].into(),
///     &Angle::from(PI / 2.0)
/// ));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Convex<S>(pub S);

impl<V, S> SupportMapping<V> for Convex<S>
where
    S: SupportMapping<V>,
{
    // Forward the call to the inner type.
    #[inline]
    fn support_mapping(&self, n: &V) -> V {
        self.0.support_mapping(n)
    }
}

impl<S> BoundingSphereRadius for Convex<S>
where
    S: BoundingSphereRadius,
{
    // Forward the call to the inner type.
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        self.0.bounding_sphere_radius()
    }
}

impl<A, B, R> IntersectsAt<Convex<A>, Cartesian<2>, R> for Convex<B>
where
    A: SupportMapping<Cartesian<2>> + BoundingSphereRadius,
    B: SupportMapping<Cartesian<2>> + BoundingSphereRadius,
    R: Rotate<Cartesian<2>> + Rotation,
    RotationMatrix<2>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &Convex<A>, v_ij: &Cartesian<2>, o_ij: &R) -> bool {
        let self_circle = Circle::with_radius(self.0.bounding_sphere_radius());
        let other_circle = Circle::with_radius(other.0.bounding_sphere_radius());

        self_circle.intersects_at(&other_circle, v_ij, o_ij) && collide2d(self, other, v_ij, o_ij)
    }
}

impl<A, B, R> IntersectsAtGlobal<Convex<A>, Cartesian<2>, R> for Convex<B>
where
    A: SupportMapping<Cartesian<2>> + BoundingSphereRadius,
    B: SupportMapping<Cartesian<2>> + BoundingSphereRadius,
    R: Rotate<Cartesian<2>> + Rotation,
    RotationMatrix<2>: From<R>,
{
    #[inline]
    fn intersects_at_global(
        &self,
        other: &Convex<A>,
        r_self: &Cartesian<2>,
        o_self: &R,
        r_other: &Cartesian<2>,
        o_other: &R,
    ) -> bool {
        let max_separation =
            self.0.bounding_sphere_radius().get() + other.0.bounding_sphere_radius().get();
        if r_self.distance_squared(r_other) >= max_separation.powi(2) {
            return false;
        }

        let (v_ij, o_ij) = hoomd_vector::pair_system_to_local(r_self, o_self, r_other, o_other);

        collide2d(self, other, &v_ij, &o_ij)
    }
}

impl<A, B, R> IntersectsAt<Convex<A>, Cartesian<3>, R> for Convex<B>
where
    A: SupportMapping<Cartesian<3>> + BoundingSphereRadius,
    B: SupportMapping<Cartesian<3>> + BoundingSphereRadius,
    R: Rotate<Cartesian<3>> + Rotation,
    RotationMatrix<3>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &Convex<A>, v_ij: &Cartesian<3>, o_ij: &R) -> bool {
        let self_sphere = Sphere::with_radius(self.0.bounding_sphere_radius());
        let other_sphere = Sphere::with_radius(other.0.bounding_sphere_radius());

        self_sphere.intersects_at(&other_sphere, v_ij, o_ij) && collide3d(self, other, v_ij, o_ij)
    }
}

impl<A, B, R> IntersectsAtGlobal<Convex<A>, Cartesian<3>, R> for Convex<B>
where
    A: SupportMapping<Cartesian<3>> + BoundingSphereRadius,
    B: SupportMapping<Cartesian<3>> + BoundingSphereRadius,
    R: Rotate<Cartesian<3>> + Rotation,
    RotationMatrix<3>: From<R>,
{
    #[inline]
    fn intersects_at_global(
        &self,
        other: &Convex<A>,
        r_self: &Cartesian<3>,
        o_self: &R,
        r_other: &Cartesian<3>,
        o_other: &R,
    ) -> bool {
        let r_cut = self.0.bounding_sphere_radius().get() + other.0.bounding_sphere_radius().get();
        if r_self.distance_squared(r_other) >= r_cut.powi(2) {
            return false;
        }

        let (v_ij, o_ij) = hoomd_vector::pair_system_to_local(r_self, o_self, r_other, o_other);

        collide3d(self, other, &v_ij, &o_ij)
    }
}

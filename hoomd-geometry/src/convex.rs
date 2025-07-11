// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `Convex`.
*/

use crate::{
    BoundingSphereRadius, IntersectsAt, SupportMapping,
    shape::{Circle, Sphere},
    xenocollide::{collide2d, collide3d},
};
use hoomd_vector::{Cartesian, Rotate, Rotation, RotationMatrix};

/// A newtype that checks for intersections using [`xenocollide`](crate::xenocollide).
#[derive(Clone, Debug, PartialEq)]
pub struct Convex<S>(pub S);

impl<V, S> SupportMapping<V> for Convex<S>
where
    S: SupportMapping<V>,
{
    /// Forward the call to the inner type.
    #[inline]
    fn support_mapping(&self, n: &V) -> V {
        self.0.support_mapping(n)
    }
}

impl<A, B, R> IntersectsAt<Convex<A>, Cartesian<2>, R> for Convex<B>
where
    A: SupportMapping<Cartesian<2>> + BoundingSphereRadius,
    B: SupportMapping<Cartesian<2>> + BoundingSphereRadius,
    R: Rotate<Cartesian<2>> + Rotation + Copy,
    RotationMatrix<2>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &Convex<A>, v_ij: &Cartesian<2>, o_ij: &R) -> bool {
        if !(Circle {
            radius: self.0.bounding_sphere_radius(),
        })
        .intersects_at(
            &Circle {
                radius: other.0.bounding_sphere_radius(),
            },
            v_ij,
            o_ij,
        ) {
            return false;
        }
        collide2d(self, other, v_ij, o_ij)
    }
}
impl<A, B, R> IntersectsAt<Convex<A>, Cartesian<3>, R> for Convex<B>
where
    A: SupportMapping<Cartesian<3>> + BoundingSphereRadius,
    B: SupportMapping<Cartesian<3>> + BoundingSphereRadius,
    R: Rotate<Cartesian<3>> + Rotation + PartialEq + Copy,
    RotationMatrix<3>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &Convex<A>, v_ij: &Cartesian<3>, o_ij: &R) -> bool {
        if !(Sphere {
            radius: self.0.bounding_sphere_radius(),
        })
        .intersects_at(
            &Sphere {
                radius: other.0.bounding_sphere_radius(),
            },
            v_ij,
            o_ij,
        ) {
            return false;
        }
        collide3d(self, other, v_ij, o_ij)
    }
}

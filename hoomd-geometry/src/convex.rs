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

/** A newtype that checks for intersections using [`xenocollide`](crate::xenocollide).

Use [`Convex`] to check for intersections between two convex shapes (possibly
with different types).

# Example

Test if a circle overlaps with a rounded rectangle:
```
use hoomd_geometry::{Convex, IntersectsAt, shape::{Circle, Rectangle, Sphero}};
use hoomd_vector::{Cartesian, Angle};
use std::f64::consts::PI;

let circle = Convex(Circle { radius:  0.5 });
let rectangle = Rectangle { edge_lengths: [3.0, 2.0].into() };
let rounded_rectangle = Convex(Sphero { shape: rectangle, rounding_radius: 0.5 });

assert!(rounded_rectangle.intersects_at(&circle, &[2.4, 0.0].into(), &Angle::default()));
assert!(!rounded_rectangle.intersects_at(&circle, &[0.0, 2.4].into(), &Angle::default()));
assert!(circle.intersects_at(&rounded_rectangle, &[0.0, 2.4].into(), &Angle::from(PI/2.0)));
```
*/
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
        if !(Circle::with_radius(self.0.bounding_sphere_radius()).intersects_at(
            &Circle::with_radius(other.0.bounding_sphere_radius()),
            v_ij,
            o_ij,
        )) {
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
        if !(Sphere::with_radius(self.0.bounding_sphere_radius()).intersects_at(
            &Sphere::with_radius(other.0.bounding_sphere_radius()),
            v_ij,
            o_ij,
        )) {
            return false;
        }
        collide3d(self, other, v_ij, o_ij)
    }
}

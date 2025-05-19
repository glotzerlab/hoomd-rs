// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*!
Traits for determining the intersection between various bodies.

Most `Shape`s should implement [`IntersectsAt`] to allow for the calculation of
intersections between two bodies without a built-in origin. This definition is compatible
with HPMC and allows for the method's definition without requiring internal state regarding
the position or orientation of each body.
For non-orientable shapes, or for bodies who have special intersection
tests for particular orientations, [`IntersectsAt`] can be written to accept an Option
rather than a pure Rotation.

```
use hoomd_geom::{Sphere, IntersectsAt};
use hoomd_vector::Versor;

let s0 = Sphere::<3>::from(1.0);
let s1 = Sphere::<3>::from(1.0);

// Spheres are not orientable, so we can provide a None rotation for clarity.
assert!(s0.intersects_at(&s1, &[1.0, 0.0, 0.0].into(), &None::<Versor>));
assert!(s0.intersects_at(&s1, &[1.0, 0.0, 0.0].into(), &Some(Versor::default())));

```
*/

use hoomd_vector::{Cartesian, Rotate, Rotation, RotationMatrix, Vector};

use crate::{SupportFn, xenocollide::collide3d};
// use crate::Shape; // TODO: do we want this as a trait bound on S?

/**
Define a position and orientation-independent intersection based solely on the geometry
of the shape.
*/
pub trait Intersects<S> {
    /// Determine whether a Shape intersects another shape (based on intrinsic location).
    fn intersects(&self, other: &S) -> bool;
}

/**
Define a position and orientation-dependent intersection between two bodies.
*/
pub trait IntersectsAt<S, V: Vector, R: Rotate<V>> {
    /// The associated Rotation type for a given intersection method.
    type OptionalRotation;
    /// Determine whether a Particle intersects another shape at some position and orientation.
    fn intersects_at(&self, other: &S, v_ij: &V, o_ij: &Self::OptionalRotation) -> bool;
}

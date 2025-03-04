// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*!
Traits for determining the intersection between various bodies.

Most `Shape`s should implement [`IntersectsAt`] to allow for the calculation of
intersections between two bodies without a built-in origin. This definition is compatible
with HPMC and allows for the method's definition without requiring internal state regarding
the position or orientation of each body. [`Intersects`] provides an alternative for
centered and oriented bodies, and can be automatically derived from [`Intersects`] in
some cases.

```
use hoomd_geom::{Sphere, IntersectsAt};
use hoomd_vector::Versor;

let s0 = Sphere::<3>::from(1.0);
let s1 = Sphere::<3>::from(1.0);

assert!(s0.intersects_at(&s1, &[1.0, 0.0, 0.0].into(), &Versor::default()))

```
*/

use hoomd_vector::{Rotate, Vector};
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
    ///Determine whether a Particle intersects another shape at some position and orientation.
    fn intersects_at(&self, other: &S, r_ij: &V, o_ij: &R) -> bool;
}


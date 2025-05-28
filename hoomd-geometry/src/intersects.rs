// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*!
Traits for determining the intersection between various bodies.

[`IntersectsAt`] allows for the calculation of intersections between two bodies without a built-in origin. This definition is compatible with HPMC and allows for the method's definition without requiring internal state regarding
the position or orientation of each body.
For non-orientable shapes, or for bodies who have special intersection
tests for particular orientations, and inherent method `intersects` can be implemented
as well.
```
use hoomd_geometry::{Cuboid, Sphere, IntersectsAt};
use hoomd_vector::Versor;

let c0 = Cuboid::<3>::from([1.0, 1.0, 1.0]);
let c1 = Cuboid::<3>::from([1.0, 1.0, 1.0]);

// Determine the intersection between two spheres.
assert!(c0.intersects_at(&c1, &[1.0, 0.0, 0.0].into(), &Versor::default()) == true);
assert!(c0.intersects_at(&c1, &[9.9, 0.0, 0.0].into(), &Versor::default()) == false);


// Determine the intersection between two *axis-aligned cuboids*. This yields the same
// results as the code above, but uses a faster intersection check!
assert!(c0.intersects_aligned(&c1, &[1.0, 0.0, 0.0].into()) == true);
assert!(c0.intersects_aligned(&c1, &[9.9, 0.0, 0.0].into()) == false);
```
*/

use hoomd_vector::{Rotate, Vector};

/**
Define a position and orientation-dependent intersection between two bodies.
*/
pub trait IntersectsAt<S, V: Vector, R: Rotate<V>> {
    /// The associated Rotation type for a given intersection method.
    /// Determine whether a Particle intersects another shape at some position and orientation.
    fn intersects_at(&self, other: &S, v_ij: &V, o_ij: &R) -> bool;
}

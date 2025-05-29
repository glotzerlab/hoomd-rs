// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Sphero`] */

use crate::{IntersectsAt, SupportMapping, xenocollide::{collide3d, collide2d}};
use hoomd_vector::{Angle, Cartesian, Vector, Versor};

/** Round a shape with a given radius.

[`Sphero`] modifies a given shape by sweeping it with a hypersphere of the given
radius. The resulting [`Sphero<S>`] type is a shape itself. If `S` implements
[`SupportMapping`], then [`Sphero<S>`] can be used in [`IntersectsAt`] tests with
other convex shapes. See the full list of implementations below to see what other
traits [`Sphero<S>`] implements for a given `S`.

# Example

Test if a circle overlaps with a rounded rectangle:
```
use hoomd_geometry::{Convex, IntersectsAt, shape::{Circle, Rectangle, Sphero}};
use hoomd_vector::{Cartesian, Angle};
use std::f64::consts::PI;

let circle = Convex(Circle { r: 0.5 });
let rectangle = Rectangle { edge_lengths: [3.0, 2.0].into() };
let rounded_rectangle = Convex(Sphero { shape: rectangle, rounding_radius: 0.5 });

assert!(rounded_rectangle.intersects_at(&circle, &[2.4, 0.0].into(), &Angle::default()));
assert!(!rounded_rectangle.intersects_at(&circle, &[0.0, 2.4].into(), &Angle::default()));
assert!(circle.intersects_at(&rounded_rectangle, &[0.0, 2.4].into(), &Angle::from(PI/2.0)));
*/
pub struct Sphero<S> {
    /// The shape round.
    pub shape: S,
    /// The radius of the rounding hypersphere.
    pub rounding_radius: f64,
}

impl<S, V> SupportMapping<V> for Sphero<S>
where
    S: SupportMapping<V>,
    V: Vector,
     {
    #[inline]
    fn support_mapping(&self, n: &V) -> V {
        self.shape.support_mapping(n) + *n / n.norm() * self.rounding_radius
    }
}


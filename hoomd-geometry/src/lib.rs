// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! General, performant computational geometry code.

`hoomd_geometry` implements common operations for widely-used geometric
primitives, with additional functionality to accommodate hard-particle Monte
Carlo simulations.

## Geometric Primitives

The [`Hypersphere`][shape::Hypersphere] demonstrates the design philosophy of
`hoomd_geometry`. The struct contains a single radius value, and immediately
provides access to a variety of methods. Hypersphers are well defined in
arbitrary dimension, and therefore the implementation is parameterized with a
const generic `N` representing the embedding dimension:
```
use hoomd_geometry::{ Volume, shape::Hypersphere };
use approx::assert_relative_eq;
use std::f64::consts::PI;

const N: usize = 3;
let s = Hypersphere::<N>::from_radius(1.0);
let volume = s.volume();
assert_relative_eq!(volume, (4.0/3.0 * PI));
```

## Traits
[`Volume`] provides a notion of the amount of space a primitive
occupies, and indicates the N-hypervolume of a given struct. For a
[`Rectangle`][shape::Rectangle], for example, [`Volume`] returns the area in the
plane, and for a [`Sphere`][shape::Sphere] returns the three-dimensional volume.

[`IntersectsAt`] determines if there is an intersection between two shapes,
where the second shape is placed in the coordinate system of the first.
This is the most efficient way to test for intersections in Monte Carlo
simulations as only the positions and orientations of the sites need to be
modified.

For non-orientable shapes, or for bodies who have special intersection
tests for particular orientations, and inherent method `intersects` can be
implemented as well:
```
use hoomd_geometry::{Convex, IntersectsAt, shape::Sphere};
use hoomd_vector::Versor;

let s0 = Sphere {radius: 1.0};
let s1 = Sphere {radius: 1.0};

let q_id = Versor::default();

assert_eq!(s0.intersects_at(&s1, &[1.9, 0.0, 0.0].into(), &q_id), true);
assert_eq!(s0.intersects_at(&s1, &[2.1, 0.0, 0.0].into(), &q_id), false);
```

Any pair of shapes (with possibly different types) that both implement the
[`SupportMapping`] trait can be tested for overlaps through the  [`Convex`]
newtype. [`IntersectsAt`] uses the [`xenocollide`] algorithm, provided for
2d and 3d shapes, to test for intersections between [`Convex`] shapes:
```
use hoomd_geometry::{Convex, IntersectsAt, shape::{Cuboid, Sphere}};
use hoomd_vector::Versor;
let s0 = Sphere {radius: 1.0};

let wrapped_cuboid = Convex(Cuboid::from([2.0, 2.0, 2.0]));

assert_eq!(
    Convex(s0).intersects_at(&wrapped_cuboid, &[1.9, 0.0, 0.0].into(), &Versor::default()),
    true
);
assert_eq!(
    Convex(s0).intersects_at(&wrapped_cuboid, &[2.1, 0.0, 0.0].into(), &Versor::default()),
    false
);
```
*/

use thiserror::Error;

mod convex;
pub use convex::Convex;

pub mod shape;
pub mod xenocollide;

/** The N-hypervolume of a geometry. In 2D, this is area and in 3D this is Volume.

# Example

```
use hoomd_geometry::{Volume, shape::Hypersphere};

const N: usize = 3;
let s = Hypersphere::<N>::from_radius(1.0);
let volume = s.volume();
```

*/
pub trait Volume {
    /// The N-hypervolume of a geometry.
    #[must_use]
    fn volume(&self) -> f64;
}

/** Compute the shortest distance between any points on two separate shapes.

By convention, the minimum distance is positive when the shapes are separated
and 0 when overlapping. 

TODO: Should this instead be defined as the minimum distance between a point
and a shape's surface? That is traditionally negative when the point is
inside.
*/
pub trait MinDistance<const N: usize, V, R, S> {
    /// Minimum distance between two shapes in `N` dimensions
    fn min_distance(&self, other: &S, v_ij: &V, o_ij: R) -> f64;
}

/** Find the point on a shape that is the furthest in a given direction.

# Example

```
use hoomd_geometry::{shape::Cuboid, SupportMapping};
use hoomd_vector::Cartesian;

let cuboid = Cuboid::from([3.0, 2.0]);

let upper_right = cuboid.support_mapping(&Cartesian::from([1.0, 1.0]));
let lower_right = cuboid.support_mapping(&Cartesian::from([1.0, -1.0]));

assert_eq!(upper_right, [1.5, 1.0].into());
assert_eq!(lower_right, [1.5, -1.0].into());
```
*/
pub trait SupportMapping<V> {
    /// Return the furthest extent of a shape in the direction of `n`.
    fn support_mapping(&self, n: &V) -> V;
}

/** Test whether two shapes share the same space.

# Examples

Some shapes implement [`IntersectsAt`] directly:
```
use hoomd_geometry::{Convex, IntersectsAt, shape::Sphere};
use hoomd_vector::Versor;

let s0 = Sphere {radius: 1.0};
let s1 = Sphere {radius: 1.0};

let q_id = Versor::default();

assert_eq!(s0.intersects_at(&s1, &[1.9, 0.0, 0.0].into(), &q_id), true);
assert_eq!(s0.intersects_at(&s1, &[2.1, 0.0, 0.0].into(), &q_id), false);
```

Others must be wrapped in [`Convex`]:
```
use hoomd_geometry::{Convex, IntersectsAt, shape::{Cuboid, Sphere}};
use hoomd_vector::Versor;
let s0 = Sphere {radius: 1.0};

let wrapped_cuboid = Convex(Cuboid::from([2.0, 2.0, 2.0]));

assert_eq!(
    Convex(s0).intersects_at(&wrapped_cuboid, &[1.9, 0.0, 0.0].into(), &Versor::default()),
    true
);
assert_eq!(
    Convex(s0).intersects_at(&wrapped_cuboid, &[2.1, 0.0, 0.0].into(), &Versor::default()),
    false
);
```
*/
pub trait IntersectsAt<S, V, R> {

    /** Test whether the set of points in one shape intersects with the set of another.

    Each shape (`self` and `other`) remain unmodified in their own local
    coordinate systems. The intersection test is performed in the local
    coordinate system of `self`. The vector `v_ij` points from the local
    origin of `self` to the local origin of `other`. Similarly, `o_ij` is the
    orientation of `other` in the local coordinate system of `self`.

    TODO: An example that shows computing `v_ij` and `o_ij` from two shapes
    in world coordinates.
    */ 
    fn intersects_at(&self, other: &S, v_ij: &V, o_ij: &R) -> bool;
}

/** Radius of an N-dimensional hypersphere that bounds a shape.

The hypersphere has the same local origin as the shape `self`.

Some [`IntersectsAt`] tests use the bounding sphere radius as a first pass
before calling a more expensive intersection test. To improve performance,
the bounding sphere should be as tightly fitting as possible.

# Example

```
use hoomd_geometry::{shape::Cuboid, BoundingSphereRadius};

let cuboid = Cuboid::from([6.0, 8.0]);
let bounding_radius = cuboid.bounding_sphere_radius();

assert_eq!(bounding_radius, 5.0);
```
*/
pub trait BoundingSphereRadius {
    /// Get the bounding radius.
    fn bounding_sphere_radius(&self) -> f64;
}

/// Enumerate possible sources of error in fallible shape methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// A set of vertices is not convex.
    #[error("Vertices do not define a convex body.")]
    NotConvex(),
}

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
use hoomd_geometry::{ IntersectsAt, Volume, shape::Hypersphere };
use approx::assert_relative_eq;
use std::f64::consts::PI;

const N: usize = 3;
let s = Hypersphere::<N>::from_radius(1.0);
assert_relative_eq!(s.volume(), (4.0/3.0 * PI));
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
use hoomd_geometry::{ IntersectsAt, shape::{Cuboid, Sphere} , Convex };
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
use hoomd_geometry::{ Convex, IntersectsAt, shape::{Cuboid, Sphere} };
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

pub mod shape;
pub mod xenocollide;

/// The N-hypervolume of a geometry. In 2D, this is area and in 3D this is Volume.
pub trait Volume {
    /// The N-hypervolume of a geometry.
    #[must_use]
    fn volume(&self) -> f64;
}

/**
Definitions of the minimum distance between two shapes. Will be zero if points are on
a boundary (within floating-point precision) and negative if the shapes are overlapping.
*/
pub trait MinDistance<const N: usize, V, R, S> {
    /// Minimum distance between two shapes in `N` dimensions
    fn min_distance(&self, other: &S, v_ij: &V, o_ij: R) -> f64;
}

/**
The support mapping of a geometry.

The function associated with this trait should take a direction vector and return the
point on a (convex) shape lying furthest in that direction.
*/
pub trait SupportMapping<V> {
    /// Return the furthest extent of a shape along a direction vector.
    fn support_mapping(&self, n: &V) -> V;
}

/**
Define a position and orientation-dependent intersection between two bodies.
*/
pub trait IntersectsAt<S, V, R> {
    /// The associated Rotation type for a given intersection method.
    /// Determine whether a Particle intersects another shape at some position and orientation.
    fn intersects_at(&self, other: &S, v_ij: &V, o_ij: &R) -> bool;
}
/**
Radius of an N-dimensional hypersphere that tightly bounds a shape.
 */
pub trait BoundingSphereRadius {
    /// A reasonably tight-fitting bounding [`Hypersphere`][crate::shape::Hypersphere] radius for a shape.
    fn bounding_sphere_radius(&self) -> f64;
}

/// A newtype wrapper that allows for intersection detection via Xenocollide.
mod convex;
pub use convex::Convex;

use thiserror::Error;
/// Enumerate possible sources of error in fallible utility methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// A set of vertices is not convex.
    #[error("Vertices do not define a convex body.")]
    NotConvex(),
}

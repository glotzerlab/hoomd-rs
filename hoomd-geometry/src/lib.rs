// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! General, performant computational geometry code.

`hoomd_geometry` implements common operations for widely-used geometric primitives, with
additional functionality to accommodate hard-particle Monte Carlo simulations.

## Geometric Primitives

The [`Hypersphere`] is an excellent example of the design philosophy of `hoomd_geometry`. The
struct is initialized from a single radius value, and immediately provides access to
a variety of methods. [`Hypersphere`]s are well defined in arbitrary dimension, and therefore
are parameterized with a const generic `N` representing the embedding dimension.

```
use hoomd_geometry::{ IntersectsAt, Volume, shape::Hypersphere };
use approx::assert_relative_eq;
use std::f64::consts::PI;

const N: usize = 3;
let s = Hypersphere::<N>::from_radius(1.0);
assert_relative_eq!(s.volume(), (4.0/3.0 * PI));
```

Common properties are implemented in the [`Shape`] trait, which provides the `Volume`
compute from the previous example. [`Shape`] also implements `bounding_sphere`, which
represents a tight-fitting (but not necessarily minimal) bounding sphere. For a sphere,
of course, this implementation is trivial.

In general, the [`Shape`] trait is designed to include commonly-used methods that are
relatively easy to implement for arbitrary shapes. More complicated properties are
included in additional methods, including [`IntersectsAt`], [`MinDistance`], and
[`SupportMapping`].

Traits for determining the intersection between various bodies.

[`IntersectsAt`] allows for the calculation of intersections between two bodies without a built-in origin. This definition is compatible with HPMC and allows for the method's definition without requiring internal state regarding
the position or orientation of each body.
For non-orientable shapes, or for bodies who have special intersection
tests for particular orientations, and inherent method `intersects` can be implemented
as well.
```
use hoomd_geometry::{ IntersectsAt, shape::{Cuboid, Sphere} , Convex };
use hoomd_vector::Versor;

let s0 = Sphere {radius: 1.0};
let s1 = Sphere {radius: 1.0};

let q_id = Versor::default();

// Determine the intersection between two spheres, using a fast overlap check
assert_eq!(s0.intersects_at(&s1, &[1.9, 0.0, 0.0].into(), &q_id), true);
assert_eq!(s0.intersects_at(&s1, &[2.1, 0.0, 0.0].into(), &q_id), false);

// For more complex bodies, the `Convex` wrapper allows for robust overlap checks using xenocollide
assert_eq!(Convex(s0).intersects_at(&Convex(s1), &[1.9, 0.0, 0.0].into(), &q_id), true);
assert_eq!(Convex(s0).intersects_at(&Convex(s1), &[2.1, 0.0, 0.0].into(), &q_id), false);

// The `Convex` wrapper also allows for overlap checks between heterogeneous particles
let cuboid = Cuboid::from([2.0, 2.0, 2.0]);

assert_eq!(
    Convex(s0).intersects_at(&Convex(cuboid), &[1.9, 0.0, 0.0].into(), &q_id),
    true
);
assert_eq!(
    Convex(s0).intersects_at(&Convex(cuboid), &[2.1, 0.0, 0.0].into(), &q_id),
    false
);
*/

pub mod shape;
pub mod xenocollide;

/// The N-hypervolume of a geometry. In 2D, this is area and in 3D this is Volume.
pub trait Volume {
    /// The N-hypervolume of a geometry
    #[must_use]
    fn volume(&self) -> f64;
}

/**
Definitions of the minimum distance between two `Shape`s. Will be zero if points are on
a boundary (within floating-point precision) and negative if the shapes are overlapping.
*/
pub trait MinDistance<const N: usize, V, R, S> {
    /// Minimum distance between two `Shape`s in `N` dimensions
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
/// TODO:
pub trait BoundingSphereRadius {
    /// A reasonably tight-fitting bounding Hypersphere radius for a shape.
    fn bounding_sphere_radius(&self) -> f64;
}

/// TODO
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

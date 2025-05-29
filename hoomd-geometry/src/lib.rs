// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

// TODO: shapes in module
// embedded trait?
// Bounded shape pair for intersections (prevents code repetition, but doesn't add anything new)
// Would need to implement a way to create bound anyway, so not that useful
// For polytope: private field for bounding shape is the right way to go
// TODO: [Jen] Generalize capsule on N
// TODO: [Jen] Define type alias for Ellipse in 2d, Rectangle, Circle, ConvexPolygon
// TODO: [Jen] SimplePolygon (same in memory, but interpreted differently)
// TODO: [Jen] Polyhedron (non-convex, not general on dimension, requires faces (ragged list?))
// TODO: use Sphero to xenocollide
// TODO: implement Volume for Sphero<Volume>
// TODO: meshes - talk to Joseph & Philipp
// TODO: "Functions with a clear receiver are methods" for ConvexPolytope
// TODO: implement MinDistance for spheres
// TODO: GSD_shape_spec trait
// TODO: impl Normals for Mesh: Mesh can have lots of cached information, with From
//       ConvexPolytope/Polytope
// TODO: IsInside trait (Dom)
// TODO: scale? instead of set_volume
// TODO: surface area, mean curvature

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

The [`Sphere`] is an excellent example of the design philosophy of `hoomd_geometry`. The
struct is initialized from a single radius value, and immediately provides access to
a variety of methods. [`Sphere`]s are well defined in arbitrary dimension, and therefore
are parameterized with a const generic `N` representing the embedding dimension.

```
use hoomd_geometry::{ IntersectsAt, Volume, shape::Sphere };
use approx::assert_relative_eq;
use std::f64::consts::PI;

const N: usize = 3;
let s = Sphere::<N>::from(1.0);
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
use hoomd_geometry::{ IntersectsAt, shape::Cuboid };
use hoomd_vector::Versor;

let c0 = Cuboid::<3>::from([1.0, 1.0, 1.0]);
let c1 = Cuboid::<3>::from([1.0, 1.0, 1.0]);

// Determine the intersection between two oriented cuboids.
assert!(c0.intersects_at(&c1, &[1.0, 0.0, 0.0].into(), &Versor::default()) == true);
assert!(c0.intersects_at(&c1, &[9.9, 0.0, 0.0].into(), &Versor::default()) == false);

// Determine the intersection between two *axis-aligned cuboids*. This yields the same
// results as the code above, but uses a faster intersection check!
assert!(c0.intersects_aligned(&c1, &[1.0, 0.0, 0.0].into()) == true);
assert!(c0.intersects_aligned(&c1, &[9.9, 0.0, 0.0].into()) == false);
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
The support function of a geometry.

TODO: SupportMapping should be called SupportMapping (fn typically returns dot product)
*/
pub trait SupportMapping<V> {
    /// Center of mass of the shape
    /// Distances from the origin to each supporting hyperplane.
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

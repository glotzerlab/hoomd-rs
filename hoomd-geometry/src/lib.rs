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
// TODO: move Sphero to `shape`
// TODO: remove Intersects
// TODO: remove Shape
// TODO: SupportFn -> SupportMapping
// TODO: move contents of poly to shapes
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
use hoomd_geometry::{Sphere, Volume, IntersectsAt};
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
[`SupportFn`].

*/
mod cuboid;
mod intersects;
pub mod modifiers;

mod common;
pub mod poly;
mod shape;
mod simplex3;
mod sphere;
pub mod xenocollide;

pub use {
    common::*,
    cuboid::Cuboid,
    intersects::IntersectsAt,
    modifiers::Sphero,
    shape::{MinDistance, SupportFn, Volume},
    simplex3::Simplex3,
    sphere::Sphere,
};

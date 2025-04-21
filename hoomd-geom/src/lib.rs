// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! General, performant computational geometry code.

`hoomd_geom` implements common operations for widely-used geometric primitives, with
additional functionality to accommodate hard-particle Monte Carlo simulations.

## Geometric Primitives

The [`Sphere`] is an excellent example of the design philosophy of `hoomd_geom`. The
struct is initialized from a single radius value, and immediately provides access to
a variety of methods. [`Sphere`]s are well defined in arbitrary dimension, and therefore
are parameterized with a const generic `N` representing the embedding dimension.

```
use hoomd_geom::{Sphere, Volume, IntersectsAt};
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

## Modifiers through Encapsulation

Note that the `Sphere` struct in the previous example is defined solely by a radius. To
maximize generality, no `Shape`s should explicitly store the center of mass. This allows
shared utility between HPMC and pure computational geometry applications without wasted
memory.

For cases where an explicit centroid is required, consider the [`Centered`] struct. This
-- and the [`Sphero`] struct -- provide extensions to a core shape definition through
encapsulation.

```
use hoomd_geom::{Cuboid, Centered, IntersectsAt};
use std::f64::consts::PI;

let centered_cuboid = Centered::from(
    (Cuboid::from([1.0, 2.0, 3.0]), [0.0, 0.0, 0.0])
);

assert_eq!(centered_cuboid.centroid, [0.0; 3].into());
```

*/
mod cuboid;
mod intersects;
pub mod modifiers;

mod common;
pub mod poly;
mod shape;
pub mod simplex3;
mod sphere;
pub mod xenocollide;

pub use {
    common::*,
    cuboid::Cuboid,
    intersects::{Intersects, IntersectsAt},
    modifiers::{Centered, Sphero},
    shape::{MinDistance, Shape, SupportFn, Volume},
    sphere::Sphere,
};

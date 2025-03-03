
// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Vector and quaternion math.

`hoomd_vector` implements vector math types and operations used in scientific
computations, specifically those used in the HOOMD molecular simulation software
suite. Its API is firmly rooted in mathematical principles. Users in
other fields may find `hoomd_vector` useful outside the context of `HOOMD`.

## Vectors

The [`Vector`] trait describes any type that is a member of a normed vector
space. Write code with a [`Vector`] trait bound when you can express the
computation with vector arithmetic and dot products. Your generic code can
then be invoked on vector types with any dimension or representation (e.g.
spherical coordinates).

```
use hoomd_vector::Vector;

fn some_function<V: Vector>(a: &V, b: &V) -> f64 {
    a.dot(b) / (a.norm_squared())
}
```
*/
mod sphere;
mod cuboid;
mod intersects;
mod shape;
pub mod modifiers;

pub use {
    sphere::Sphere,
    cuboid::Cuboid,
    intersects::{Intersects, IntersectsAt},
    shape::{Volume, SupportFn, Shape, MinDistance} 
};


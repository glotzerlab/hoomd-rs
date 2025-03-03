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
additional functionality to accomodate hard-particle Monte Carlo simulations.

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



## Struct Modifiers

## Traits

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


// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Cylinder`] */

use crate::Volume;

use super::Circle;

/** A circle with normal `[0 0 1]` swept by `h/2` in the `+z` and `-z` directions.

# Example
[`Cylinder`]s implement the [`Volume`] trait, which is equivalent to $π r^2 h$

```rust
use hoomd_geometry::{shape::Cylinder, Volume};

let cyl = Cylinder {radius: 2.0, height: 3.0};
assert_eq!(cyl.volume(), std::f64::consts::PI * (2.0 * 2.0) * 3.0);
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cylinder {
    /// Radius of the [`Cylinder`]
    pub radius: f64,
    /// Height of the [`Cylinder`]
    pub height: f64,
}

impl Volume for Cylinder {
    #[inline]
    fn volume(&self) -> f64 {
        Circle {
            radius: self.radius,
        }
        .volume()
            * self.height
    }
}

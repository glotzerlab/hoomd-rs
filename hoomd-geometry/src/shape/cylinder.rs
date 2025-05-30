// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Cylinder`] */

use hoomd_vector::{Cartesian, Rotate, Vector};

use crate::{BoundingShape, SupportMapping, Volume};

use super::{Capsule, Circle};

/** A [`Cylinder`] in three dimensions. This should be interpreted as a circle with
normal `[0 0 1]` swept by `h/2` in the `+z` and `-z` directions.*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cylinder {
    /// Radius of the [`Cylinder`]
    r: f64,
    /// Height of the [`Cylinder`]
    h: f64,
}

impl Volume for Cylinder {
    #[inline]
    fn volume(&self) -> f64 {
        Circle { r: self.r }.volume() * self.h
    }
}

// impl<R: Rotate<Cartesian<3>>> BoundingShape<Cartesian<3>, R> for Cylinder {
//     type Shape = Capsule<3>; // Requires IntersectsAt for Capsule<3>
//     fn bounding_shape(&self) -> Self::Shape {
//         Self::Shape {
//             r: self.r,
//             h: self.h,
//         }
//     }
// }

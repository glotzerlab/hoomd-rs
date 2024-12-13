// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement 2D and 3D rotations.
*/
mod angle;

use crate::{
    vector::{self, Cartesian},
    Cross, Rotate, Rotation, Vector,
};

pub use angle::Angle;

pub struct Quaternion;

impl Rotate<vector::Cartesian<3>> for Quaternion {
    #[inline]
    fn rotate(&self, vector: &vector::Cartesian<3>) -> vector::Cartesian<3> {
        todo!()
    }
}

impl Rotation for Quaternion {
    #[inline]
    fn identity() -> Self {
        todo!()
    }

    #[inline]
    fn inversed(self) -> Self {
        todo!()
    }

    #[inline]
    fn combine(&self, rotation: &Self) -> Self {
        todo!()
    }
}

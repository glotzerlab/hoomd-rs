// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement 2D and 3D rotations.
*/

use crate::{vector, Cross, Rotate, Rotation, Vector};

pub struct Quaternion;

impl Rotate<vector::Cartesian<3>> for Quaternion {
    #[inline]
    fn rotate(&self, vector: &vector::Cartesian<3>) -> vector::Cartesian<3> {
        vector::Cartesian::from([0.0, 0.0, 0.0])
    }
}

impl Rotation for Quaternion {
    #[inline]
    fn combine(&self, rotation: &Self) -> Self {
        Self::default();
    }
}

pub struct Angle;

impl Rotate<vector::Cartesian<2>> for Angle {
    #[inline]
    fn rotate(&self, vector: &vector::Cartesian<2>) -> vector::Cartesian<2> {
        vector::Cartesian::from([0.0, 0.0])
    }
}

impl Rotation for Angle {
    #[inline]
    fn combine(&self, rotation: &Self) -> Self {
        Self::default();
    }
}

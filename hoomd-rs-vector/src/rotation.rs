// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement 2D and 3D rotations.
*/

use crate::{vector::{self, Cartesian}, Cross, Rotate, Rotation, Vector};

pub struct Quaternion;

impl Rotate<vector::Cartesian<3>> for Quaternion {
    #[inline]
    fn rotate(&self, vector: &vector::Cartesian<3>) -> vector::Cartesian<3> {
        todo!()
    }
}

impl Rotation for Quaternion {
    #[inline]
    fn combine(&self, rotation: &Self) -> Self {
        todo!()
    }
}

pub struct Angle{
    pub angle: f64, // theta instead
}


impl Default for Angle {
    /** Create an angle of 0.0 radians. 

    ```
    # use hoomd_rs_vector::rotation;
    let a = rotation::Angle::default();
    assert_eq!(a, 0.0)
    ```
    */
    #[inline]
    #[must_use]
    fn default() -> Self {
        Angle::from(0.0)
    }
}


impl From<f64> for Angle {
    #[inline]
    fn from(angle: f64) -> Self {
        Self {angle}
    }
}

impl Rotate<vector::Cartesian<2>> for Angle {
    #[inline]
    fn rotate(&self, vector: &vector::Cartesian<2>) -> vector::Cartesian<2> {
        let sin = self.angle.sin();
        let cos = self.angle.cos();
        return Cartesian::from([vector.coordinates[0] * cos - vector.coordinates[1] * sin,
                                vector.coordinates[0] * sin + vector.coordinates[1] * cos]);
    }
}

impl Rotation for Angle {
    #[inline]
    fn combine(&self, rotation: &Self) -> Self {
        Self::from(self.angle+rotation.angle)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn test_rotate() {
        let angle = Angle::from(std::f64::consts::PI/2.0);
        let vec = Cartesian::from([1.0, 0.0]);
        let rotated_vec = angle.rotate(&vec);
        assert_eq!(rotated_vec, Cartesian::from([0.0, 1.0]));
    }
}

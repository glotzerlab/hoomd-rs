// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement 2D and 3D rotations.
*/

use crate::{vector::{self, Cartesian}, Cross, Rotate, Rotation, Vector};

/** A 3D rotation represented by a quaternion of rotation in 3D cartesian coordinates.

`Quaternion` is the 3D implementation for rotation of vectors.

## Examples:

```
use hoomd_rs_vector::rotation;
```
Create a quaternion from an axis and angle:
```
# use hoomd_rs_vector::rotation;
let a = rotation::Quaternion::from_axis_angle([0.0,0.0,1.0].into(),std::f64::consts::PI/2.0);
```
Using subsequent rotations can be done by combining angles instead of rotating the vector multiple times:
```
# use hoomd_rs_vector::rotation;
let a = rotation::Quaternion::from_axis_angle([0.0,0.0,1.0].into(),std::f64::consts::PI/2.0);
let b = rotation::Quaternion::from_axis_angle([0.0,0.0,1.0].into(),std::f64::consts::PI/4.0);
let c = a.combine(&b);
```
Access the angle directly when needed:
```
# use hoomd_rs_vector::rotation;
# let q = rotation::Quaternion::from_axis_angle([0.0,0.0,1.0].into(),std::f64::consts::PI/2.0);
let half_angle = rotation::Quaternion::from_axis_angle(q.v,q.s/2.0);
```
Rotate a 3D Cartesian vector:
```
# use hoomd_rs_vector::rotation;
# use hoomd_rs_vector::vector;
# let vec = vector::Cartesian::from([1.0, 1.0, 1.0]);
# let a = rotation::Quaternion::from_axis_angle([0.0,0.0,1.0].into(),std::f64::consts::PI/2.0);
let rotated_vec = a.rotate(&vec);
```
*/
pub struct Quaternion {
    /// Scalar component
    pub s: f64,

    /// Vector component
    pub v: Cartesian<3>
}

/// inherent methods
impl Quaternion {
    #[inline]
    pub fn from_axis_angle(axis: Cartesian<3>, angle: f64) -> Result<Self, crate::Error> {
        Ok(Quaternion {s: (angle/2.0).cos(), v: axis.normalized()? * (angle/2.0).sin()})
    }
} 


impl Rotate<Cartesian<3>> for Quaternion {
    #[inline]
    fn rotate(&self, vector: &Cartesian<3>) -> Cartesian<3> {
        let a = self;
        let b = vector;

        *b * (a.s * a.s - a.v.dot(&a.v)) + a.v.cross(&b) * (2.0 * a.s) + a.v * (2.0 * a.v.dot(&b))
    }
}

impl Rotation for Quaternion {
    #[inline]
    fn combine(&self, rotation: &Self) -> Self {
        let a = self;
        let b = rotation;

        Quaternion { s: (a.s * b.s - a.v.dot(&b.v)), v: (b.v * a.s + a.v * b.s + a.v.cross(&b.v)) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Angle{
    pub theta: f64,
}


impl Default for Angle {
    /** Create an angle of 0.0 radians. 

    ```
    # use hoomd_rs_rotation;
    let a = rotation::Angle::default();
    assert_eq!(a.theta, 0.0)
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
    fn from(theta: f64) -> Self {
        Self {theta}
    }
}

impl Rotate<Cartesian<2>> for Angle {
    #[inline]
    fn rotate(&self, vector: &Cartesian<2>) -> Cartesian<2> {
        let sin = self.theta.sin();
        let cos = self.theta.cos();
        Cartesian::from([vector.coordinates[0] * cos - vector.coordinates[1] * sin,
                                vector.coordinates[0] * sin + vector.coordinates[1] * cos])
    }
}

impl Rotation for Angle {
    #[inline]
    fn combine(&self, rotation: &Self) -> Self {
        Self::from(self.theta+rotation.theta)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rotate() {
        let angle = Angle::from(std::f64::consts::PI/2.0);
        let vec = Cartesian::from([1.0, 0.0]);
        let rotated_vec = angle.rotate(&vec);
        let vec2 = Cartesian::from([0.0, 1.0]);
        assert_relative_eq!(rotated_vec, vec2);
    }

    #[test]
    fn test_quaternion_rotate() {
        let quat = Quaternion::from_axis_angle([0.0, 0.0, 1.0].into(), 0.0).expect("A valid quaternion");
        let vec = Cartesian::from([1.0, 1.0, 1.0]);
        let rotated_quat = quat.rotate(&vec);
        assert_relative_eq!(rotated_quat, [1.0, 1.0, 1.0].into());
    }
}

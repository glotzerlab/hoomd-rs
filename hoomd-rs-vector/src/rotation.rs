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


/** A 2D rotation represented by an angle of rotation in 2D cartesian coordinates.

`Angle` is the 2D implementation for rotation of vectors.

## Examples:

```
use hoomd_rs_vector::rotation;
```
Create an angle from a float:
```
# use hoomd_rs_vector::rotation;
let a = rotation::Angle::from(std::f64::consts::PI/2.0);
```
Using subsequent rotations can be done by combining angles instead of rotating the vector multiple times:
```
# use hoomd_rs_vector::rotation;
# use hoomd_rs_vector::Rotation;
let a = rotation::Angle::from(std::f64::consts::PI/2.0);
let b = rotation::Angle::from(std::f64::consts::PI/4.0);
let c = a.combine(&b);
```
Access the angle directly when needed:
```
# use hoomd_rs_vector::rotation;
# let a = rotation::Angle::from(std::f64::consts::PI/2.0);
let half_angle = rotation::Angle::from((a.theta/2.0));
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Angle{
    /// Rotation in 2D Euclidean space using angle rotation.
    pub theta: f64,
}


impl Default for Angle {
    /** Create an angle of 0.0 radians. 

    ```
    # use hoomd_rs_vector::rotation;
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

impl Rotate<vector::Cartesian<2>> for Angle {
    #[inline]
    fn rotate(&self, vector: &vector::Cartesian<2>) -> vector::Cartesian<2> {
        let sin = self.theta.sin();
        let cos = self.theta.cos();
        Cartesian::from([
            vector.coordinates[0] * cos - vector.coordinates[1] * sin,
            vector.coordinates[0] * sin + vector.coordinates[1] * cos,
        ])
    }
}

impl Rotation for Angle {
    #[inline]
    fn combine(&self, rotation: &Self) -> Self {
        Self::from(self.theta + rotation.theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::*;
    use std::f64::consts::PI;

    #[rstest]
    #[case::pi_halves(PI/2.0, (1.0, -0.5), (0.5, 1.0))]
    #[case::negative_pi_thirds(-PI/3.0, (1.0, 0.0), (0.5, -f64::sqrt(3.0) / 2.0))]
    #[case::negative_pi(-PI, (3.1, -0.2), (-3.1, 0.2))]
    #[case::two_pi(PI*2.0, (3.1, -0.2), (3.1, -0.2))]
    #[case::zero(0.0, (3.1, -0.2), (3.1, -0.2))]
    #[case::negative_zero(-0.0, (3.1, -0.2), (3.1, -0.2))]
    fn test_rotate_2d(#[case] angle: f64, #[case] vec: (f64, f64), #[case] ans: (f64, f64)) {
        let angle = Angle::from(angle);
        let vec = Cartesian::from(vec);
        let ans = Cartesian::from(ans);

        assert_relative_eq!(angle.rotate(&vec), ans, epsilon = 4.0 * f64::EPSILON);
    }
}

// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Quaternion`]
*/

use crate::{
    vector::Cartesian,
    Cross, Error, Rotate, Rotation, Vector,
};

/** Represent a 3D rotation with a quaternion.

## Constructing [`Quaternion`]:

Create a quaternion from an axis and angle:
```
use hoomd_rs_vector::rotation;
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

impl Quaternion {
    /** Create a [`Quaternion`] from a given axis and a rotation angle about that axis.

    # Errors

    [`Error::InvalidMagnitude`] when `axis` is the 0 vector.
    */
    #[inline]
    pub fn from_axis_angle(axis: Cartesian<3>, angle: f64) -> Result<Self, Error> {
        Ok(Quaternion {s: (angle/2.0).cos(), v: axis.normalized()? * (angle/2.0).sin()})
    }
} 

impl Rotate<Cartesian<3>> for Quaternion {
    #[inline]
    fn rotate(&self, vector: &Cartesian<3>) -> Cartesian<3> {
        *vector * (self.s * self.s - self.v.dot(&self.v)) + self.v.cross(vector) * (2.0 * self.s) + self.v * (2.0 * self.v.dot(vector))
    }
}

impl Rotation for Quaternion {
    /** Combine two rotations.

    The resulting quaternion is the multiplication of the two.
    <!-- \mathbf{q}_{ab} = \mathbf{q}_a \mathbf{q}_b -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><msub><mi>𝐪</mi><mrow><mi>a</mi><mi>b</mi></mrow></msub><mo>=</mo><msub><mi>𝐪</mi><mi>a</mi></msub><msub><mi>𝐪</mi><mi>b</mi></msub></mrow></math>

    # Example

    ```
    use hoomd_rs_vector::rotation::Quaternion;
    let q_a = Quaternion::from_axis_angle([0.0, 1.0, 0.0].into(), 1.5);
    let q_b = Quaternion::from_axis_angle([1.0, 0.0, 0.0].into(), 0.125);
    let q_ab = q_a.combine(q_b);
    ```    
    */
    #[inline]
    fn combine(&self, other: &Self) -> Self {
        let a = self;
        let b = other;

        Quaternion { s: (a.s * b.s - a.v.dot(&b.v)), v: (b.v * a.s + a.v * b.s + a.v.cross(&b.v)) }
    }

    /** Create the identity [`Quaternion`]: [1, [0, 0, 0]]

    # Example

    ```
    use hoomd_rs_vector::rotation::Quaternion;
    let identity = Quaternion::identity();
    ```
    */
    #[inline]
    fn identity() -> Self {
        Quaternion { s: 1.0, v: [0.0, 0.0, 0.0].into() }
    }

    /** Create a [`Quaternion`] that is the conjugate of the given quaternion.
    
    <!-- \mathbf{q}^* -->
    <math display="block" class="tml-display" style="display:block math;"><msup><mi>𝐪</mi><mo>*</mo></msup></math>
    
    # Example

    ```
    use hoomd_rs_vector::rotation::Quaternion;
    let q = Quaternion::from_axis_angle([0.0, 1.0, 0.0].into(), 1.5);
    let q_star = q.inversed();
    ```
    */
    #[inline]
    fn inversed(self) -> Self {
        Quaternion { v: -self.v, ..self }
    }
}

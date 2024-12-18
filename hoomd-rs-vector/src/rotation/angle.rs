// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Angle`]
*/

use rand::distributions::{Distribution, Standard, Uniform};
use rand::Rng;
use std::f64::consts::PI;
use std::fmt;

use crate::{vector::Cartesian, Rotate, Rotation, Vector};

/** Represent a 2D rotation in the plane by an angle.

## Constructing [`Angle`]

Create an [`Angle`] with a given value:
```
use hoomd_rs_vector::rotation;
let a = rotation::Angle::from(std::f64::consts::PI/2.0);
assert_eq!(a.theta, std::f64::consts::PI/2.0);
```

Create a random [`Angle`]:
```
use hoomd_rs_vector::rotation;
use rand::{thread_rng, Rng};
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut rng = rand::thread_rng();
let v: rotation::Angle = rng.gen();
# Ok(())
# }
```

Combine two rotations together:
```
use hoomd_rs_vector::{rotation, Rotation};
let a = rotation::Angle::from(std::f64::consts::PI/2.0);
let b = rotation::Angle::from(-std::f64::consts::PI/4.0);
let c = a.combine(&b);
assert_eq!(c.theta, std::f64::consts::PI/4.0);
```

## Operations using [`Angle`]

Rotate a [`Cartesian<2>`] vector by an [`Angle`]:
```
use hoomd_rs_vector::{rotation, Rotate, Rotation, vector};
let v = vector::Cartesian::from([-1.0, 0.0]);
let a = rotation::Angle::from(std::f64::consts::PI/2.0);
let rotated = a.rotate(&v);
// rotated is approximately [0.0, -1.0]
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Angle {
    /// Rotation angle (radians).
    pub theta: f64,
}

/// A precomputed rotation about an [`Angle`]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Precomputed {
    /// Rows of the rotation matrix.
    rows: [Cartesian<2>; 2],
}

impl Angle {
    /** Normalize the rotation.

    [`Angle`] rotations are well-defined for any value of `theta`. However, there are numerical
    issues when combining small rotations with large ones due to floating point round-off error.
    Normalizing an [`Angle`] creates an equivalent rotation with `theta` in the rage from 0 to
    2 pi.

    # Example

    ```
    use hoomd_rs_vector::rotation;
    let a = rotation::Angle::from(20.0 * std::f64::consts::PI);
    let b= a.normalized();
    assert_eq!(b.theta, 0.0)
    ```
    */
    #[inline]
    #[must_use]
    pub fn normalized(self) -> Self {
        Self {
            theta: self.theta.rem_euclid(2.0 * PI),
        }
    }

    /** Precompute the rotation.

    When rotating many vectors by the same [`Angle`], precompute the rotation to improve
    performance.

    # Example
    ```
    use hoomd_rs_vector::{rotation, Rotate, Rotation, vector};
    let v = vector::Cartesian::from([-1.0, 0.0]);
    let a = rotation::Angle::from(std::f64::consts::PI/2.0);

    let precomputed = a.precomputed();
    let rotated = precomputed.rotate(&v);
    // rotated is approximately [0.0, -1.0]
    ```
    */
    #[inline]
    #[must_use]
    pub fn precomputed(&self) -> Precomputed {
        let sin_theta = self.theta.sin();
        let cos_theta = self.theta.cos();
        Precomputed {
            rows: [
                [cos_theta, -sin_theta].into(),
                [sin_theta, cos_theta].into(),
            ],
        }
    }
}

impl Default for Angle {
    /** Create a rotation by 0 radians.

    # Example
    ```
    use hoomd_rs_vector::rotation;
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
    /** Create a rotation by `theta` radians

    # Example
    ```
    use hoomd_rs_vector::rotation;
    let a = rotation::Angle::from(1.5);
    assert_eq!(a.theta, 1.5);
    ```
    */
    #[inline]
    fn from(theta: f64) -> Self {
        Self { theta }
    }
}

impl Rotate<Cartesian<2>> for Angle {
    #[inline]
    /** Rotate a [`Cartesian<2>`] in the plane by an [`Angle`]

    # Example
    ```
    use hoomd_rs_vector::{rotation, Rotate, Rotation, vector};
    let v = vector::Cartesian::from([-1.0, 0.0]);
    let a = rotation::Angle::from(std::f64::consts::PI/2.0);
    let rotated = a.rotate(&v);
    // rotated is approximately [0.0, -1.0]
    ```
    */
    fn rotate(&self, vector: &Cartesian<2>) -> Cartesian<2> {
        let sin_theta = self.theta.sin();
        let cos_theta = self.theta.cos();
        Cartesian::from([
            vector.coordinates[0] * cos_theta - vector.coordinates[1] * sin_theta,
            vector.coordinates[0] * sin_theta + vector.coordinates[1] * cos_theta,
        ])
    }
}

impl Rotate<Cartesian<2>> for Precomputed {
    #[inline]
    /** Rotate a [`Cartesian<2>`] in the plane by an [`Angle`]

    # Example
    ```
    use hoomd_rs_vector::{rotation, Rotate, Rotation, vector};
    let v = vector::Cartesian::from([-1.0, 0.0]);
    let a = rotation::Angle::from(std::f64::consts::PI/2.0);
    let precomputed = a.precomputed();
    let rotated = precomputed.rotate(&v);
    // rotated is approximately [0.0, -1.0]
    ```
    */
    fn rotate(&self, vector: &Cartesian<2>) -> Cartesian<2> {
        Cartesian::from([self.rows[0].dot(vector), self.rows[1].dot(vector)])
    }
}

impl Rotation for Angle {
    #[inline]
    /** Create an [`Angle`] that rotates by 0 radians.

    # Example
    ```
    use hoomd_rs_vector::{rotation, Rotation};
    let a = rotation::Angle::identity();
    assert_eq!(a.theta, 0.0);
    ```
    */
    fn identity() -> Self {
        Self::default()
    }

    #[inline]
    /** Create an [`Angle`] that rotates by the same amount in the opposite direction.

    # Example
    ```
    use hoomd_rs_vector::{rotation, Rotation};
    let a = rotation::Angle::from(std::f64::consts::PI/3.0);
    let b = a.inversed();
    assert_eq!(b.theta, -std::f64::consts::PI/3.0);
    ```
    */
    fn inversed(self) -> Self {
        Self::from(-self.theta)
    }

    #[inline]
    /** Create an [`Angle`] that rotates by the sum of the two angles.

    # Example
    ```
    use hoomd_rs_vector::{rotation, Rotation};
    let a = rotation::Angle::from(std::f64::consts::PI/2.0);
    let b = rotation::Angle::from(-std::f64::consts::PI/4.0);
    let c = a.combine(&b);
    assert_eq!(c.theta, std::f64::consts::PI/4.0);
    ```
    */
    fn combine(&self, other: &Self) -> Self {
        Self::from(self.theta + other.theta)
    }
}

impl fmt::Display for Angle {
    /// Format an Angle as `<{theta}>`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<{}>", self.theta)
    }
}

impl Distribution<Angle> for Standard {
    /** Sample a random angle from the uniform distribution.

    # Example

    ```
    use hoomd_rs_vector::rotation;
    use rand::{thread_rng, Rng};
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();
    let v: rotation::Angle = rng.gen();
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Angle {
        let uniform = Uniform::new(0.0, 2.0 * PI);
        Angle::from(uniform.sample(rng))
    }
}

#[cfg(test)]
mod approx {
    use approx::{AbsDiffEq, RelativeEq};

    impl AbsDiffEq for super::Angle {
        type Epsilon = <f64 as AbsDiffEq>::Epsilon;

        fn default_epsilon() -> Self::Epsilon {
            f64::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            f64::abs_diff_eq(&self.theta, &other.theta, epsilon)
        }
    }

    impl RelativeEq for super::Angle {
        fn default_max_relative() -> Self::Epsilon {
            f64::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: Self::Epsilon,
        ) -> bool {
            f64::relative_eq(&self.theta, &other.theta, epsilon, max_relative)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::approx::assert_relative_eq;
    use rstest::*;
    use std::f64::consts::PI;

    // Test named cases of the three input values (angle, vector input, and answer)
    #[rstest]
    #[case::pi_halves(PI/2.0, (1.0, -0.5), (0.5, 1.0))]
    #[case::negative_pi_thirds(-PI/3.0, (1.0, 0.0), (0.5, -f64::sqrt(3.0) / 2.0))]
    #[case::negative_pi(-PI, (3.1, -0.2), (-3.1, 0.2))]
    #[case::two_pi(PI*2.0, (3.1, -0.2), (3.1, -0.2))]
    #[case::zero(0.0, (3.1, -0.2), (3.1, -0.2))]
    #[case::negative_zero(-0.0, (3.1, -0.2), (3.1, -0.2))]
    fn rotate_2d(#[case] angle: f64, #[case] vec: (f64, f64), #[case] ans: (f64, f64)) {
        let angle = Angle::from(angle);
        let vec = Cartesian::from(vec);
        let ans = Cartesian::from(ans);

        assert_relative_eq!(angle.rotate(&vec), ans, epsilon = 4.0 * f64::EPSILON);
        assert_relative_eq!(
            angle.precomputed().rotate(&vec),
            ans,
            epsilon = 4.0 * f64::EPSILON
        );
    }

    // Test with Cartesian product of the input arrays
    #[rstest(
        ang1 => [0.0, PI/2.0, 1e-12 * PI, -3.0, 12345.6],
        ang2 => [-0.0, -PI/3.0, PI, 2.0 * PI]
    )]
    fn combine_2d(ang1: f64, ang2: f64) {
        let (angle1, angle2) = (Angle::from(ang1), Angle::from(ang2));
        assert_relative_eq!(angle1.combine(&angle2).theta, ang1 + ang2);
    }

    #[test]
    fn default() {
        let a = Angle::default();
        assert!(a.theta == 0.0);
    }

    #[test]
    fn identity() {
        let a = Angle::identity();
        assert!(a.theta == 0.0);
    }

    #[rstest(theta => [0.0, 1.0, 2.125, 14.875, -4.5])]
    fn inversed(theta: f64) {
        let angle1 = Angle::from(theta);
        let angle2 = angle1.inversed();
        assert!(angle2.theta == -theta);
        assert_relative_eq!(angle1.combine(&angle2), Angle::identity());
    }

    #[test]
    fn display() {
        let a = Angle::from(1.5);
        let s = format!("{a}");
        assert_eq!(s, "<1.5>");
    }

    #[test]
    fn normalized() {
        let two_pi = 2.0 * PI;

        assert_relative_eq!(Angle::from(0.125).normalized(), (0.125).into());
        assert_relative_eq!(Angle::from(2.0 * PI + 0.125).normalized(), (0.125).into());
        assert_relative_eq!(Angle::from(2.0 * 2.0 * PI + 0.5).normalized(), (0.5).into());
        assert_relative_eq!(Angle::from(3.0 * 2.0 * PI + 3.0).normalized(), (3.0).into());
        assert_relative_eq!(
            Angle::from(2.0 * PI - 0.125).normalized(),
            (2.0 * PI - 0.125).into()
        );

        assert_relative_eq!(Angle::from(two_pi).normalized(), (0.0).into());
        assert_relative_eq!(Angle::from(-0.125).normalized(), (2.0 * PI - 0.125).into());
        assert_relative_eq!(Angle::from(-3.0).normalized(), (2.0 * PI - 3.0).into());
        assert_relative_eq!(
            Angle::from(-2.0 * PI - 0.125).normalized(),
            (2.0 * PI - 0.125).into()
        );
        assert_relative_eq!(
            Angle::from(10.0 * -2.0 * PI - 0.125).normalized(),
            (2.0 * PI - 0.125).into()
        );
    }

    #[test]
    fn random() {
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            let a: Angle = rng.gen();
            assert!(a.theta >= 0.0 && a.theta < 2.0 * PI);
        }
    }
}

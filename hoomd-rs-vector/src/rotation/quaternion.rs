// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Quaternion`]
*/
use rand::distributions::{Distribution, Standard, Uniform};
use rand::Rng;
use std::fmt;

use crate::{vector::Cartesian, Cross, Error, Normalized, Rotate, Rotation, Vector};

/** Represent a 3D rotation with a quaternion.

[`Quaternion`] represents a 3D rotation with a **unit quaternion**. It stores the real part in
the scalar component `s` and the 3 complex values in the vector `v`. Rotation follows the standard
Hamilton convention.

## Constructing a [`Quaternion`]:

Create a [`Quaternion`] from an axis and angle:
```
use hoomd_rs_vector::rotation::Quaternion;
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let q = Quaternion::from_axis_angle([0.0, 1.0, 0.0].into(), PI/2.0);
# Ok(())
# }
```

Create a random [`Quaternion`]:
```
use hoomd_rs_vector::rotation::Quaternion;
use rand::{rngs::StdRng, Rng, SeedableRng};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut rng = StdRng::seed_from_u64(1);
let v: Quaternion = rng.gen();
# Ok(())
# }
```


Combine two rotations together:
```
use hoomd_rs_vector::{rotation::Quaternion, Rotation};
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let a = Quaternion::from_axis_angle([1.0, 0.0, 1.0].into(), PI/2.0);
let b = Quaternion::from_axis_angle([0.0, 0.0, 1.0].into(), PI/4.0);
let c = a.combine(&b);
# Ok(())
# }
```

## Operations using [`Quaternion`]

Rotate a [`Cartesian<3>`] vector by a [`Quaternion`]:
```
use hoomd_rs_vector::{rotation::Quaternion, Rotate, Rotation, vector::Cartesian};
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let a = Cartesian::from([-1.0, 0.0, 0.0]);
let q = Quaternion::from_axis_angle([0.0, 0.0, 1.0].into(), PI/2.0);
let b = q.rotate(&a);
// b is approximately [0.0, -1.0, 0.0]
# Ok(())
# }
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion {
    /// Scalar component
    pub s: f64,

    /// Vector component
    pub v: Cartesian<3>,
}

/// A precomputed rotation by a [`Quaternion`]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Precomputed {
    /// Rows of the rotation matrix.
    rows: [Cartesian<3>; 3],
}

impl Quaternion {
    /** Create a [`Quaternion`] from a given axis and a rotation angle about that axis.

    # Example

    ```
    use hoomd_rs_vector::rotation::Quaternion;
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Quaternion::from_axis_angle([0.0, 1.0, 0.0].into(), PI/2.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Normalized<Cartesian<3>>, angle: f64) -> Self {
        let Normalized(axis_vector) = axis;
        
        Quaternion {
            s: (angle / 2.0).cos(),
            v: axis_vector * (angle / 2.0).sin(),
        }
    }

    /** Normalize the quaternion.

    Nominally, all [`Quaternion`] instances remain normalized. Due to limited floating point
    precision, this assumption may not hold after repeated operations. Normalize quaternions
    when needed to correct this issue.

    # Example

    ```
    use hoomd_rs_vector::rotation::Quaternion;
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Quaternion::from_axis_angle([0.0, 1.0, 0.0].into(), PI/2.0);
    let b = a.normalized()?;
    # Ok(())
    # }
    ```

    # Errors

    [`Error::InvalidMagnitude`] when `self` is the 0 quaternion.
    */
    #[inline]
    pub fn normalized(self) -> Result<Self, Error> {
        let magnitude_squared = self.magnitude_squared();

        if magnitude_squared == 0.0 {
            Err(Error::InvalidMagnitude)
        } else {
            let f = 1.0 / magnitude_squared.sqrt();
            Ok(Self {
                s: self.s * f,
                v: self.v * f,
            })
        }
    }

    /** Precompute the rotation.

    When rotating many vectors by the same [`Quaternion`], precompute the rotation to improve
    performance.

    # Example
    ```
    use hoomd_rs_vector::{rotation::Quaternion, Rotate, Rotation, vector::Cartesian};
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Cartesian::from([-1.0, 0.0, 0.0]);
    let q = Quaternion::from_axis_angle([0.0, 0.0, 1.0].into(), PI/2.0);

    let precomputed = q.precomputed();
    let b = precomputed.rotate(&a);
    // b is approximately [0.0, -1.0, 0.0]
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn precomputed(&self) -> Precomputed {
        let a = self.s;
        let b = self.v[0];
        let c = self.v[1];
        let d = self.v[2];

        Precomputed {
            rows: [
                [
                    a * a + b * b - c * c - d * d,
                    2.0 * b * c - 2.0 * a * d,
                    2.0 * b * d + 2.0 * a * c,
                ]
                .into(),
                [
                    2.0 * b * c + 2.0 * a * d,
                    a * a - b * b + c * c - d * d,
                    2.0 * c * d - 2.0 * a * b,
                ]
                .into(),
                [
                    2.0 * b * d - 2.0 * a * c,
                    2.0 * c * d + 2.0 * a * b,
                    a * a - b * b - c * c + d * d,
                ]
                .into(),
            ],
        }
    }

    /** The magnitude of the quaternion, squared.
     */
    #[inline]
    #[must_use]
    fn magnitude_squared(&self) -> f64 {
        self.s * self.s + self.v.dot(&self.v)
    }
}

impl Default for Quaternion {
    /** Create an identity rotation.

    # Example
    ```
    use hoomd_rs_vector::rotation::Quaternion;

    let q = Quaternion::default();
    ```
    */
    #[inline]
    #[must_use]
    fn default() -> Self {
        Self {
            s: 1.0,
            v: [0.0, 0.0, 0.0].into(),
        }
    }
}

impl Rotate<Cartesian<3>> for Quaternion {
    /** Rotate a [`Cartesian<3>`] by a [`Quaternion`]

    <!-- \mathbf{q} \vec{a} \mathbf{q}^* -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mi>𝐪</mi><mover><mi>a</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><msup><mi>𝐪</mi><mo>*</mo></msup></mrow></math>

    # Example

    ```
    use hoomd_rs_vector::{rotation::Quaternion, Rotate, Rotation, vector::Cartesian};
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Cartesian::from([-1.0, 0.0, 0.0]);
    let q = Quaternion::from_axis_angle([0.0, 0.0, 1.0].into(), PI/2.0);
    let b = q.rotate(&a);
    // b is approximately [0.0, -1.0, 0.0]
    # Ok(())
    # }
    ```

    <div class="warning">

    The given [`Quaternion`] is assumed to be normalized. `rotate` produces undefined results
    when the quaternion is not normalized.

    </div>
    */
    #[inline]
    fn rotate(&self, vector: &Cartesian<3>) -> Cartesian<3> {
        *vector * (self.s * self.s - self.v.dot(&self.v))
            + self.v.cross(vector) * (2.0 * self.s)
            + self.v * (2.0 * self.v.dot(vector))
    }
}

impl Rotate<Cartesian<3>> for Precomputed {
    #[inline]
    /** Rotate a [`Cartesian<3>`] by a [`Quaternion`]

    # Example
    ```
    use hoomd_rs_vector::{rotation::Quaternion, Rotate, Rotation, vector::Cartesian};
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Cartesian::from([-1.0, 0.0, 0.0]);
    let q = Quaternion::from_axis_angle([0.0, 0.0, 1.0].into(), PI/2.0);

    let precomputed = q.precomputed();
    let b = precomputed.rotate(&a);
    // b is approximately [0.0, -1.0, 0.0]
    # Ok(())
    # }
    ```
    */
    fn rotate(&self, vector: &Cartesian<3>) -> Cartesian<3> {
        Cartesian::from([
            self.rows[0].dot(vector),
            self.rows[1].dot(vector),
            self.rows[2].dot(vector),
        ])
    }
}

impl Rotation for Quaternion {
    /** Combine two rotations.

    The resulting quaternion is the multiplication of the two.
    <!-- \mathbf{q}_{ab} = \mathbf{q}_a \mathbf{q}_b -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><msub><mi>𝐪</mi><mrow><mi>a</mi><mi>b</mi></mrow></msub><mo>=</mo><msub><mi>𝐪</mi><mi>a</mi></msub><msub><mi>𝐪</mi><mi>b</mi></msub></mrow></math>

    # Example

    ```
    use hoomd_rs_vector::{rotation::Quaternion, Rotation};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q_a = Quaternion::from_axis_angle([0.0, 1.0, 0.0].into(), 1.5);
    let q_b = Quaternion::from_axis_angle([1.0, 0.0, 0.0].into(), 0.125);
    let q_ab = q_a.combine(&q_b);
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn combine(&self, other: &Self) -> Self {
        let a = self;
        let b = other;

        Quaternion {
            s: (a.s * b.s - a.v.dot(&b.v)),
            v: (b.v * a.s + a.v * b.s + a.v.cross(&b.v)),
        }
    }

    /** Create the identity [`Quaternion`]: [1, [0, 0, 0]]

    # Example

    ```
    use hoomd_rs_vector::{rotation::Quaternion, Rotation};

    let identity = Quaternion::identity();
    ```
    */
    #[inline]
    fn identity() -> Self {
        Self::default()
    }

    /** Create a [`Quaternion`] that is the conjugate of the given quaternion.

    <!-- \mathbf{q}^* -->
    <math display="block" class="tml-display" style="display:block math;"><msup><mi>𝐪</mi><mo>*</mo></msup></math>

    # Example

    ```
    use hoomd_rs_vector::{rotation::Quaternion, Rotation};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Quaternion::from_axis_angle([0.0, 1.0, 0.0].into(), 1.5);
    let q_star = q.inversed();
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn inversed(self) -> Self {
        Quaternion { v: -self.v, ..self }
    }
}

impl fmt::Display for Quaternion {
    /// Format a [`Quaternion`] as `[{s}, [{v[0]}, {v[1]}, {v[2]}]`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.s, self.v)
    }
}

impl Distribution<Quaternion> for Standard {
    /** Sample a random [`Quaternion`] from the uniform distribution over all rotations.

    # Example

    ```
    use hoomd_rs_vector::rotation::Quaternion;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(1);
    let v: Quaternion = rng.gen();
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Quaternion {
        // Algorithm from: https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
        let uniform = Uniform::new(-1.0, 1.0);

        let (u, v) = loop {
            let u: f64 = uniform.sample(rng);
            let v: f64 = uniform.sample(rng);
            if u * u + v * v < 1.0 {
                break (u, v);
            }
        };

        let (x, y) = loop {
            let x: f64 = uniform.sample(rng);
            let y: f64 = uniform.sample(rng);
            if x * x + y * y < 1.0 {
                break (x, y);
            }
        };

        let scale = ((1.0 - (x * x + y * y)) / (u * u + v * v)).sqrt();
        Quaternion {
            s: x,
            v: [y, scale * u, scale * v].into(),
        }
    }
}

#[cfg(test)]
mod approx {
    use approx::{AbsDiffEq, RelativeEq};

    use crate::vector::Cartesian;

    impl AbsDiffEq for super::Quaternion {
        type Epsilon = <f64 as AbsDiffEq>::Epsilon;

        fn default_epsilon() -> Self::Epsilon {
            f64::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            f64::abs_diff_eq(&self.s, &other.s, epsilon)
                && Cartesian::abs_diff_eq(&self.v, &other.v, epsilon)
        }
    }

    impl RelativeEq for super::Quaternion {
        fn default_max_relative() -> Self::Epsilon {
            f64::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: Self::Epsilon,
        ) -> bool {
            f64::relative_eq(&self.s, &other.s, epsilon, max_relative)
                && Cartesian::relative_eq(&self.v, &other.v, epsilon, max_relative)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::approx::{assert_abs_diff_eq, assert_relative_eq};
    use rand::{rngs::StdRng, SeedableRng};
    use rstest::*;
    use std::f64::consts::PI;

    #[test]
    fn default() {
        let a = Quaternion::default();
        assert!(a.s == 1.0);
        assert!(a.v == [0.0, 0.0, 0.0].into());
    }

    #[test]
    fn identity() {
        let a = Quaternion::identity();
        assert!(a.s == 1.0);
        assert!(a.v == [0.0, 0.0, 0.0].into());
    }

    #[rstest(
        theta => [0.0, PI/2.0, 1e-12 * PI, -3.0, 12345.6],
        axis => [Cartesian::from([1.0, 0.0, 0.0]).to_normalized_unchecked(), Cartesian::from([1.0, -1.0, 1.0]).to_normalized_unchecked()],
    )]
    fn from_axis_angle(theta: f64, axis: Normalized<Cartesian<3>>) {
        let Normalized(axis_vector) = axis;
    
        let q = Quaternion::from_axis_angle(axis, theta);
        assert_relative_eq!(q.s, (theta / 2.0).cos());
        assert_relative_eq!(q.v, axis_vector * (theta / 2.0).sin());
    }

    #[rstest(
        theta_1 => [0.0, PI/2.0, -3.0],
        theta_2 => [-0.0, -PI/3.0, PI, 2.0 * PI]
    )]
    fn combine_same_axis(theta_1: f64, theta_2: f64) {
        let axis = Cartesian::from([1.0, 0.0, 0.0]).to_normalized_unchecked();
        let Normalized(axis_vector) = axis;
        
        let a = Quaternion::from_axis_angle(axis, theta_1);
        let b = Quaternion::from_axis_angle(axis, theta_2);
        let q = a.combine(&b);

        let theta = theta_1 + theta_2;
        assert_relative_eq!(q.s, (theta / 2.0).cos());
        assert_relative_eq!(q.v, axis_vector * (theta / 2.0).sin());
    }

    fn validate_rotations<R: Rotate<Cartesian<3>>>(z_pi_2: &R, y_pi_4: &R) {
        assert_relative_eq!(
            z_pi_2.rotate(&[0.0, 0.0, 1.0].into()),
            [0.0, 0.0, 1.0].into()
        );
        assert_relative_eq!(
            z_pi_2.rotate(&[1.0, 0.0, 4.25].into()),
            [0.0, 1.0, 4.25].into()
        );
        assert_relative_eq!(
            z_pi_2.rotate(&[0.0, 1.0, -8.75].into()),
            [-1.0, 0.0, -8.75].into()
        );

        let sqrt_2_2 = 2.0_f64.sqrt() / 2.0;
        assert_relative_eq!(
            y_pi_4.rotate(&[0.0, -10.0, 0.0].into()),
            [0.0, -10.0, 0.0].into()
        );
        assert_relative_eq!(
            y_pi_4.rotate(&[1.0, -15.0, 0.0].into()),
            [sqrt_2_2, -15.0, -sqrt_2_2].into()
        );
        assert_relative_eq!(
            y_pi_4.rotate(&[sqrt_2_2, -15.0, -sqrt_2_2].into()),
            [0.0, -15.0, -1.0].into()
        );
    }

    #[test]
    fn rotate() {
        let z_pi_2 =
            Quaternion::from_axis_angle(Cartesian::from([0.0, 0.0, 1.0]).to_normalized_unchecked(), PI / 2.0);
        let y_pi_4 =
            Quaternion::from_axis_angle(Cartesian::from([0.0, 1.0, 0.0]).to_normalized_unchecked(), PI / 4.0);

        validate_rotations(&z_pi_2, &y_pi_4);
    }

    #[test]
    fn precompute() {
        let z_pi_2 = Quaternion::from_axis_angle(Cartesian::from([0.0, 0.0, 1.0]).to_normalized_unchecked(), PI / 2.0)
            .precomputed();
        let y_pi_4 = Quaternion::from_axis_angle(Cartesian::from([0.0, 1.0, 0.0]).to_normalized_unchecked(), PI / 4.0)
            .precomputed();

        validate_rotations(&z_pi_2, &y_pi_4);
    }

    #[test]
    fn combine_different_axis() {
        let a =
            Quaternion::from_axis_angle(Cartesian::from([1.0, 0.0, 0.0]).to_normalized_unchecked(), PI / 4.0);
        let b =
            Quaternion::from_axis_angle(Cartesian::from([0.0, 0.0, 1.0]).to_normalized_unchecked(), PI / 2.0);

        let q = a.combine(&b);
        let v = q.rotate(&[1.0, 0.0, 0.0].into());
        assert_relative_eq!(v, [0.0, 2.0_f64.sqrt() / 2.0, 2.0_f64.sqrt() / 2.0].into());
    }

    #[rstest(theta => [0.0, 1.0, 2.125])]
    fn inversed(theta: f64) {
        let q1 =
            Quaternion::from_axis_angle(Cartesian::from([1.0, 0.5, -2.0]).to_normalized_unchecked(), theta);
        let q2 = q1.inversed();
        assert_relative_eq!(q1.combine(&q2), Quaternion::identity());
    }

    #[test]
    fn display() {
        let q = Quaternion {
            s: 0.5,
            v: [0.125, -0.875, 2.125].into(),
        };
        let s = format!("{q}");
        assert_eq!(s, "[0.5, [0.125, -0.875, 2.125]]");
    }

    #[test]
    fn normalized() {
        let q = Quaternion {
            s: 5.0,
            v: [3.0, -1.0, 1.0].into(),
        };
        assert_relative_eq!(
            q.normalized().expect("non-zero quaternion"),
            Quaternion {
                s: 5.0 / 6.0,
                v: [3.0 / 6.0, -1.0 / 6.0, 1.0 / 6.0].into()
            }
        );
    }

    #[test]
    fn random() {
        const CHECK_VECTORS: [Cartesian<3>; 3] = [
            Cartesian {
                coordinates: [1.0, 0.0, 0.0],
            },
            Cartesian {
                coordinates: [0.0, 1.0, 0.0],
            },
            Cartesian {
                coordinates: [1.0, 0.0, 1.0],
            },
        ];

        // Perform basic checks on random quaternions.
        // 1) Ensure that each randomly generated quaternion is normalized.
        // 2) Check that the result of rotating a reference vector by random quaternions does not
        // point in any special direction. The average dot product should be close to 0.
        let samples: u32 = 20_000;

        let reference = Cartesian::from([1.0, 0.0, 0.0]);
        let mut dot_sums = [0.0; CHECK_VECTORS.len()];

        let mut rng = StdRng::seed_from_u64(1);

        for _ in 0..samples {
            let q: Quaternion = rng.gen();
            assert_relative_eq!(q.magnitude_squared(), 1.0, max_relative = 1e-15);

            let v = q.rotate(&reference);
            for i in 0..CHECK_VECTORS.len() {
                dot_sums[i] += v.dot(&CHECK_VECTORS[i]);
            }
        }

        for dot_sum in dot_sums {
            assert_abs_diff_eq!(dot_sum / f64::from(samples), 0.0, epsilon = 0.01);
        }

        // TODO: Trevor has a better unit test, but it requires shape overlap tests.
    }
}

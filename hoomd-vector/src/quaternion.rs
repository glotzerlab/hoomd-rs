// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Quaternion`] and related types.
 */
use rand::Rng;
use rand::distr::{Distribution, StandardUniform, Uniform};
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{Cartesian, Cross, Error, Rotate, Rotation, RotationMatrix, Unit, Vector};

/** Extended complex number.

A quaternion has a real value and three complex values, represented by scalar and 3-vector
respectively:
<!-- \mathbf{q} = (s, \vec{v}) -->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>𝐪</mi><mo>=</mo><mo form="prefix" stretchy="false">(</mo><mi>s</mi><mo separator="true">,</mo><mover><mi>v</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo form="postfix" stretchy="false">)</mo></mrow></math>

Looking for the quaternion representation of 3D rotations? See [`Versor`].

## Constructing quaternions

Create a quaternion with an array of coordinates (`[scalar, vector_0, vector_1, vector_2]`).
```
use hoomd_vector::Quaternion;

let q = Quaternion::from([1.0, 2.0, 3.0, 4.0]);
assert_eq!(q.scalar, 1.0);
assert_eq!(q.vector, [2.0, 3.0, 4.0].into());
```

## Quaternion properties

Compute a quaternion's norm:
```
use hoomd_vector::Quaternion;

let q = Quaternion::from([3.0, 0.0, 4.0, 0.0]);
let norm = q.norm();
assert_eq!(norm, 5.0);
```

Form the conjugate:
```
use hoomd_vector::Quaternion;

let q = Quaternion::from([1.0, 2.0, 3.0, 4.0]);
let q_star = q.conjugate();
assert_eq!(q_star, [1.0, -2.0, -3.0, -4.0].into());
```

## Operating on quaternions

All operation examples use the following two quaternions:
```
use hoomd_vector::Quaternion;

let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
```

Addition:

```
# use hoomd_vector::Quaternion;
# let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
# let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
let c = a + b;
assert_eq!(c, [-1.0, 4.0, 10.0, -3.0].into());
```

```
# use hoomd_vector::Quaternion;
# let mut a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
# let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
a += b;
assert_eq!(a, [-1.0, 4.0, 10.0, -3.0].into());
```

Subtraction:

```
# use hoomd_vector::Quaternion;
# let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
# let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
let c = a - b;
assert_eq!(c, [3.0, -8.0, 2.0, -5.0].into());
```

```
# use hoomd_vector::Quaternion;
# let mut a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
# let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
a -= b;
assert_eq!(a, [3.0, -8.0, 2.0, -5.0].into());
```

Multiplication by a scalar:

```
# use hoomd_vector::Quaternion;
# let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
let c = a * 2.0;
assert_eq!(c, [2.0, -4.0, 12.0, -8.0].into());
```

```
# use hoomd_vector::Quaternion;
# let mut a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
# let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
a *= 2.0;
assert_eq!(a, [2.0, -4.0, 12.0, -8.0].into());
```

Division by a scalar:

```
# use hoomd_vector::Quaternion;
# let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
let c = a / 2.0;
assert_eq!(c, [0.5, -1.0, 3.0, -2.0].into());
```

```
# use hoomd_vector::Quaternion;
# let mut a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
# let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
a /= 2.0;
assert_eq!(a, [0.5, -1.0, 3.0, -2.0].into());
```

Quaternion multiplication:

```
# use hoomd_vector::Quaternion;
# let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
# let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
let c = a * b;
assert_eq!(c, [-10.0, 32.0, -30.0, -35.0].into());
```

```
# use hoomd_vector::Quaternion;
# let mut a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
# let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
a *= b;
assert_eq!(a, [-10.0, 32.0, -30.0, -35.0].into());
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion {
    /// Scalar component
    pub scalar: f64,

    /// Vector component
    pub vector: Cartesian<3>,
}

impl Quaternion {
    /** The norm of the quaternion, squared.
    <!-- |\mathbf{q}|^2 -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mi>|</mi><mi>𝐪</mi><msup><mi>|</mi><mn>2</mn></msup></mrow></math>

    # Example
    ```
    use hoomd_vector::Quaternion;

    let q = Quaternion::from([3.0, 0.0, 4.0, 0.0]);
    let norm_squared = q.norm_squared();
    assert_eq!(norm_squared, 25.0);
    ```
    */
    #[inline]
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        self.scalar * self.scalar + self.vector.dot(&self.vector)
    }

    /** The norm of the quaternion.
    <!-- |\mathbf{q}| -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mi>|</mi><mi>𝐪</mi><mi>|</mi></mrow></math>

    # Example
    ```
    use hoomd_vector::Quaternion;

    let q = Quaternion::from([3.0, 0.0, 4.0, 0.0]);
    let norm = q.norm();
    assert_eq!(norm, 5.0);
    ```
     */
    #[inline]
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /** Construct the conjugate of this quaternion.
    <!-- \mathbf{q}^* = (s, -\vec{v}) -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><msup><mi>𝐪</mi><mo>*</mo></msup><mo>=</mo><mo form="prefix" stretchy="false">(</mo><mi>s</mi><mo separator="true">,</mo><mo form="prefix" stretchy="false">−</mo><mover><mi>v</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo form="postfix" stretchy="false">)</mo></mrow></math>

    # Example
    ```
    use hoomd_vector::Quaternion;

    let q = Quaternion::from([1.0, 2.0, 3.0, 4.0]);
    let q_star = q.conjugate();
    assert_eq!(q_star, [1.0, -2.0, -3.0, -4.0].into());
    ```
    */
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self {
        Self {
            scalar: self.scalar,
            vector: -self.vector,
        }
    }

    /** Create a [`Versor`] by normalizing the given quaternion.

    <!-- \mathbf{v} = \frac{\mathbf{q}}{|\mathbf{q}|} -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mi>𝐯</mi><mo>=</mo><mfrac><mi>𝐪</mi><mrow><mi>|</mi><mi>𝐪</mi><mi>|</mi></mrow></mfrac></mrow></math>

    # Example

    ```
    use hoomd_vector::{Quaternion, Versor};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Quaternion::from([3.0, 0.0, 0.0, 4.0]);
    let v = q.to_versor()?;
    assert_eq!(*v.get(), [3.0/5.0, 0.0, 0.0, 4.0/5.0].into());
    # Ok(())
    # }
    ```

    # Errors

    [`Error::InvalidMagnitude`] when `self` is the 0 quaternion.
    */
    #[inline]
    pub fn to_versor(self) -> Result<Versor, Error> {
        let mag = self.norm();
        if mag == 0.0 {
            Err(Error::InvalidMagnitude)
        } else {
            Ok(Versor(self / mag))
        }
    }

    /** Create a [`Versor`] by normalizing the given quaternion.

    <!-- \mathbf{v} = \frac{\mathbf{q}}{|\mathbf{q}|} -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mi>𝐯</mi><mo>=</mo><mfrac><mi>𝐪</mi><mrow><mi>|</mi><mi>𝐪</mi><mi>|</mi></mrow></mfrac></mrow></math>

    # Example

    ```
    use hoomd_vector::{Quaternion, Versor};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Quaternion::from([3.0, 0.0, 0.0, 4.0]);
    let v = q.to_versor_unchecked();
    assert_eq!(*v.get(), [3.0/5.0, 0.0, 0.0, 4.0/5.0].into());
    # Ok(())
    # }
    ```

    # Panics

    Divide by 0 when `self` is the 0 quaternion.
    */
    #[inline]
    #[must_use]
    pub fn to_versor_unchecked(self) -> Versor {
        Versor(self / self.norm())
    }
}

impl From<[f64; 4]> for Quaternion {
    /** Construct a [`Quaternion`] from 4 values.

    The first value is the real part. The 2nd through 4th are the complex vector part:
    `[scalar, vector_0, vector_1, vector_2]`.

    # Example
    ```
    use hoomd_vector::Quaternion;

    let q = Quaternion::from([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(q.scalar, 1.0);
    assert_eq!(q.vector, [2.0, 3.0, 4.0].into());
    ```
    */
    #[inline]
    fn from(value: [f64; 4]) -> Self {
        Self {
            scalar: value[0],
            vector: [value[1], value[2], value[3]].into(),
        }
    }
}

impl fmt::Display for Quaternion {
    /// Format a [`Quaternion`] as `[{s}, [{v[0]}, {v[1]}, {v[2]}]]`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.scalar, self.vector)
    }
}

impl Add for Quaternion {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            scalar: self.scalar + rhs.scalar,
            vector: self.vector + rhs.vector,
        }
    }
}

impl AddAssign for Quaternion {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.scalar += rhs.scalar;
        self.vector += rhs.vector;
    }
}

impl Div<f64> for Quaternion {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self {
            scalar: self.scalar / rhs,
            vector: self.vector / rhs,
        }
    }
}

impl DivAssign<f64> for Quaternion {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.scalar /= rhs;
        self.vector /= rhs;
    }
}

impl Mul<f64> for Quaternion {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            scalar: self.scalar * rhs,
            vector: self.vector * rhs,
        }
    }
}

impl MulAssign<f64> for Quaternion {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.scalar *= rhs;
        self.vector *= rhs;
    }
}

impl Mul<Quaternion> for Quaternion {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Quaternion) -> Self {
        Self {
            scalar: (self.scalar * rhs.scalar - self.vector.dot(&rhs.vector)),
            vector: (rhs.vector * self.scalar
                + self.vector * rhs.scalar
                + self.vector.cross(&rhs.vector)),
        }
    }
}

impl MulAssign<Quaternion> for Quaternion {
    #[inline]
    fn mul_assign(&mut self, rhs: Quaternion) {
        let result = *self * rhs;
        self.scalar = result.scalar;
        self.vector = result.vector;
    }
}

impl Sub for Quaternion {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            scalar: self.scalar - rhs.scalar,
            vector: self.vector - rhs.vector,
        }
    }
}

impl SubAssign for Quaternion {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.scalar -= rhs.scalar;
        self.vector -= rhs.vector;
    }
}

/** A unit [`Quaternion`] that represents a 3D rotation.

[`Versor`] represents a 3D rotation with a **unit quaternion**. Rotation follows the Hamilton
convention.

## Constructing a [`Versor`]:

Create a [`Versor`] that rotates by an angle about an axis:
```
use hoomd_vector::Versor;
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let v = Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, PI/2.0);
assert_eq!(*v.get(), [(PI/4.0).cos(), 0.0, (PI/4.0).sin(), 0.0].into());
# Ok(())
# }
```

Create a [`Versor`] by normalizing a [`Quaternion`]:
```
use hoomd_vector::{Quaternion, Versor};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let q = Quaternion::from([3.0, 0.0, 0.0, 4.0]);
let v = q.to_versor()?;
assert_eq!(*v.get(), [3.0/5.0, 0.0, 0.0, 4.0/5.0].into());
# Ok(())
# }
```

Create a random [`Versor`]:
```
use hoomd_vector::Versor;
use rand::{rngs::StdRng, Rng, SeedableRng};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut rng = StdRng::seed_from_u64(1);
let v: Versor = rng.random();
# Ok(())
# }
```

## Operations using [`Versor`]

Rotate a [`Cartesian<3>`] by a [`Versor`]:
```
use hoomd_vector::{Versor, Rotate, Rotation, Cartesian};
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let a = Cartesian::from([-1.0, 0.0, 0.0]);
let v = Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, PI/2.0);
let b = v.rotate(&a);
// b is approximately [0.0, -1.0, 0.0]
# Ok(())
# }
```

Combine two rotations together:
```
use hoomd_vector::{Versor, Rotation};
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let a = Versor::from_axis_angle([1.0, 0.0, 1.0].try_into()?, PI/2.0);
let b = Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, PI/4.0);
let c = a.combine(&b);
# Ok(())
# }
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Versor(Quaternion);

impl Versor {
    /** Create a [`Versor`] that rotates by an angle (in radians)
    counterclockwise about an axis.

    # Example

    ```
    use hoomd_vector::Versor;
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, PI/2.0);
    assert_eq!(*v.get(), [(PI/4.0).cos(), 0.0, (PI/4.0).sin(), 0.0].into());
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Unit<Cartesian<3>>, angle: f64) -> Self {
        let Unit(axis_vector) = axis;

        Versor(Quaternion {
            scalar: (angle / 2.0).cos(),
            vector: axis_vector * (angle / 2.0).sin(),
        })
    }

    /** Normalize the versor.

    Nominally, all [`Versor`] instances retain a unit norm. Due to limited
    floating point precision, this assumption may not hold after repeated
    operations. Normalize versors when needed to correct this issue.

    # Example

    ```
    use hoomd_vector::Versor;
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, PI/2.0);
    let b = a.normalized();
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn normalized(self) -> Self {
        let Versor(q) = self;
        let f = 1.0 / q.norm();
        Self(Quaternion {
            scalar: q.scalar * f,
            vector: q.vector * f,
        })
    }

    /// Get the unit quaternion.
    #[inline]
    #[must_use]
    pub fn get(&self) -> &Quaternion {
        &self.0
    }
}

impl From<Versor> for RotationMatrix<3> {
    /** Construct a rotation matrix equivalent to this versor's rotation.

    When rotating many vectors by the same [`Versor`], improve performance
    by converting to a matrix first and applying that matrix to the vectors.

    # Example
    ```
    use hoomd_vector::{Versor, Rotate, RotationMatrix, Cartesian};
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Cartesian::from([-1.0, 0.0, 0.0]);
    let v = Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, PI/2.0);

    let matrix = RotationMatrix::from(v);
    let b = matrix.rotate(&a);
    // b is approximately [0.0, -1.0, 0.0]
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    fn from(versor: Versor) -> RotationMatrix<3> {
        let Versor(quaternion) = versor;
        let a = quaternion.scalar;
        let b = quaternion.vector[0];
        let c = quaternion.vector[1];
        let d = quaternion.vector[2];

        RotationMatrix {
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
}

impl Default for Versor {
    /** Create an identity rotation.

    # Example
    ```
    use hoomd_vector::Versor;

    let v = Versor::default();
    ```
    */
    #[inline]
    #[must_use]
    fn default() -> Self {
        Self(Quaternion {
            scalar: 1.0,
            vector: [0.0, 0.0, 0.0].into(),
        })
    }
}

impl Rotate<Cartesian<3>> for Versor {
    type Matrix = RotationMatrix<3>;

    /** Rotate a [`Cartesian<3>`] by a [`Versor`]

    <!-- \mathbf{q} \vec{a} \mathbf{q}^* -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mi>𝐪</mi><mover><mi>a</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><msup><mi>𝐪</mi><mo>*</mo></msup></mrow></math>

    # Example

    ```
    use hoomd_vector::{Versor, Rotate, Rotation, Cartesian};
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Cartesian::from([-1.0, 0.0, 0.0]);
    let v = Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, PI/2.0);
    let b = v.rotate(&a);
    // b is approximately [0.0, -1.0, 0.0]
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn rotate(&self, vector: &Cartesian<3>) -> Cartesian<3> {
        let Versor(quaternion) = self;

        *vector
            * (quaternion.scalar * quaternion.scalar - quaternion.vector.dot(&quaternion.vector))
            + quaternion.vector.cross(vector) * (2.0 * quaternion.scalar)
            + quaternion.vector * (2.0 * quaternion.vector.dot(vector))
    }
}

impl Rotation for Versor {
    /** Combine two rotations.

    The resulting versor is obtained by quaternion multiplication.
    <!-- \mathbf{q}_{ab} = \mathbf{q}_a \mathbf{q}_b -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><msub><mi>𝐪</mi><mrow><mi>a</mi><mi>b</mi></mrow></msub><mo>=</mo><msub><mi>𝐪</mi><mi>a</mi></msub><msub><mi>𝐪</mi><mi>b</mi></msub></mrow></math>

    # Example

    ```
    use hoomd_vector::{Versor, Rotation};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q_a = Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, 1.5);
    let q_b = Versor::from_axis_angle([1.0, 0.0, 0.0].try_into()?, 0.125);
    let q_ab = q_a.combine(&q_b);
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn combine(&self, other: &Self) -> Self {
        let Versor(a) = self;
        let Versor(b) = other;

        Versor(a.mul(*b))
    }

    /** Create the identity [`Versor`]: [1, [0, 0, 0]]

    # Example

    ```
    use hoomd_vector::{Versor, Rotation};

    let identity = Versor::identity();
    ```
    */
    #[inline]
    fn identity() -> Self {
        Self::default()
    }

    /** Create a [`Versor`] that performs the inverse rotation of the given versor.

    <!-- \mathbf{q}^* -->
    <math display="block" class="tml-display" style="display:block math;"><msup><mi>𝐪</mi><mo>*</mo></msup></math>

    # Example

    ```
    use hoomd_vector::{Versor, Rotation};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, 1.5);
    let v_star = v.inverted();
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn inverted(self) -> Self {
        let Versor(quaternion) = self;

        Versor(quaternion.conjugate())
    }
}

impl fmt::Display for Versor {
    /// Format a [`Versor`] as `[{s}, [{v[0]}, {v[1]}, {v[2]}]]`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Distribution<Versor> for StandardUniform {
    /** Sample a random [`Versor`] from the uniform distribution over all rotations.

    # Example

    ```
    use hoomd_vector::Versor;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(1);
    let v: Versor = rng.random();
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Versor {
        // Algorithm from: https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
        #[expect(
            clippy::expect_used,
            reason = "This constants chosen for this distribution are valid"
        )]
        let uniform = Uniform::new(-1.0, 1.0).expect("a valid distribution");

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
        Versor(Quaternion {
            scalar: x,
            vector: [y, scale * u, scale * v].into(),
        })
    }
}

#[cfg(test)]
mod approx {
    use super::{Quaternion, Versor};
    use approx::{AbsDiffEq, RelativeEq};

    use crate::Cartesian;

    impl AbsDiffEq for Quaternion {
        type Epsilon = <f64 as AbsDiffEq>::Epsilon;

        fn default_epsilon() -> Self::Epsilon {
            f64::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            f64::abs_diff_eq(&self.scalar, &other.scalar, epsilon)
                && Cartesian::abs_diff_eq(&self.vector, &other.vector, epsilon)
        }
    }

    impl AbsDiffEq for Versor {
        type Epsilon = <f64 as AbsDiffEq>::Epsilon;

        fn default_epsilon() -> Self::Epsilon {
            f64::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            super::Quaternion::abs_diff_eq(&self.0, &other.0, epsilon)
        }
    }

    impl RelativeEq for Quaternion {
        fn default_max_relative() -> Self::Epsilon {
            f64::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: Self::Epsilon,
        ) -> bool {
            f64::relative_eq(&self.scalar, &other.scalar, epsilon, max_relative)
                && Cartesian::relative_eq(&self.vector, &other.vector, epsilon, max_relative)
        }
    }

    impl RelativeEq for Versor {
        fn default_max_relative() -> Self::Epsilon {
            f64::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: Self::Epsilon,
        ) -> bool {
            Quaternion::relative_eq(&self.0, &other.0, epsilon, max_relative)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::approx::{assert_abs_diff_eq, assert_relative_eq};
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;
    use std::f64::consts::PI;

    mod quaternion {
        use super::*;

        #[test]
        fn from_array() {
            let q = Quaternion::from([2.0, -3.0, 4.0, 7.0]);
            assert!(q.scalar == 2.0);
            assert!(q.vector == [-3.0, 4.0, 7.0].into());
        }

        #[test]
        fn norm() {
            let q = Quaternion::from([1.0, 4.0, -3.0, -2.0]);
            assert_eq!(q.norm_squared(), 30.0);
            assert_eq!(q.norm(), 30.0_f64.sqrt());
        }

        #[test]
        fn conjugate() {
            let q1 = Quaternion::from([1.0, -2.0, 4.0, -0.5]);
            let q2 = q1.conjugate();
            assert_eq!(q2, [1.0, 2.0, -4.0, 0.5].into());
            assert_relative_eq!(q2 * q1, [q2.norm() * q1.norm(), 0.0, 0.0, 0.0].into());
        }

        #[test]
        fn to_versor() {
            let q = Quaternion::from([5.0, 3.0, -1.0, 1.0]);

            assert_relative_eq!(
                q.to_versor().expect("non-zero quaternion"),
                Versor(Quaternion {
                    scalar: 5.0 / 6.0,
                    vector: [3.0 / 6.0, -1.0 / 6.0, 1.0 / 6.0].into()
                })
            );

            assert_relative_eq!(
                q.to_versor_unchecked(),
                Versor(Quaternion {
                    scalar: 5.0 / 6.0,
                    vector: [3.0 / 6.0, -1.0 / 6.0, 1.0 / 6.0].into()
                })
            );

            let zero = Quaternion::from([0.0, 0.0, 0.0, 0.0]);
            assert!(matches!(zero.to_versor(), Err(Error::InvalidMagnitude)));
        }

        #[test]
        fn ops() {
            let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
            let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);

            // +, +=
            assert_eq!(a + b, [-1.0, 4.0, 10.0, -3.0].into());
            let mut c = a;
            c += b;
            assert_eq!(c, [-1.0, 4.0, 10.0, -3.0].into());

            // -, -=
            assert_eq!(a - b, [3.0, -8.0, 2.0, -5.0].into());
            let mut c = a;
            c -= b;
            assert_eq!(c, [3.0, -8.0, 2.0, -5.0].into());

            // Scalar * and /
            assert_eq!(a * 2.0, [2.0, -4.0, 12.0, -8.0].into());
            let mut c = a;
            c *= 2.0;
            assert_eq!(c, [2.0, -4.0, 12.0, -8.0].into());

            assert_eq!(a / 2.0, [0.5, -1.0, 3.0, -2.0].into());
            let mut c = a;
            c /= 2.0;
            assert_eq!(c, [0.5, -1.0, 3.0, -2.0].into());

            // Quaternion multiplication
            assert_eq!(a * b, [-10.0, 32.0, -30.0, -35.0].into());
            let mut c = a;
            c *= b;
            assert_eq!(c, [-10.0, 32.0, -30.0, -35.0].into());
        }

        #[test]
        fn display() {
            let q = Quaternion {
                scalar: 0.5,
                vector: [0.125, -0.875, 2.125].into(),
            };
            let s = format!("{q}");
            assert_eq!(s, "[0.5, [0.125, -0.875, 2.125]]");
        }
    }

    mod versor {
        use super::*;
        #[test]
        fn default() {
            let a = Versor::default();
            assert!(a.get() == &[1.0, 0.0, 0.0, 0.0].into());
        }

        #[test]
        fn identity() {
            let a = Versor::identity();
            assert!(a.get() == &[1.0, 0.0, 0.0, 0.0].into());
        }

        #[rstest(
        theta => [0.0, PI/2.0, 1e-12 * PI, -3.0, 12345.6],
        axis => [[1.0, 0.0, 0.0].try_into().expect("valid unit vector"), [1.0, -1.0, 1.0].try_into().expect("valid unit vector")],
    )]
        fn from_axis_angle(theta: f64, axis: Unit<Cartesian<3>>) {
            let Unit(axis_vector) = axis;

            let Versor(q) = Versor::from_axis_angle(axis, theta);
            assert_relative_eq!(q.scalar, (theta / 2.0).cos());
            assert_relative_eq!(q.vector, axis_vector * (theta / 2.0).sin());
        }

        #[rstest(
        theta_1 => [0.0, PI/2.0, -3.0],
        theta_2 => [-0.0, -PI/3.0, PI, 2.0 * PI]
    )]
        fn combine_same_axis(theta_1: f64, theta_2: f64) {
            let axis = [1.0, 0.0, 0.0].try_into().expect("valid unit vector");
            let Unit(axis_vector) = axis;

            let a = Versor::from_axis_angle(axis, theta_1);
            let b = Versor::from_axis_angle(axis, theta_2);
            let c = a.combine(&b);

            let theta = theta_1 + theta_2;
            let Versor(q) = c;
            assert_relative_eq!(q.scalar, (theta / 2.0).cos());
            assert_relative_eq!(q.vector, axis_vector * (theta / 2.0).sin());
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
            let z_pi_2 = Versor::from_axis_angle(
                [0.0, 0.0, 1.0].try_into().expect("valid unit vector"),
                PI / 2.0,
            );
            let y_pi_4 = Versor::from_axis_angle(
                [0.0, 1.0, 0.0].try_into().expect("valid unit vector"),
                PI / 4.0,
            );

            validate_rotations(&z_pi_2, &y_pi_4);
        }

        #[test]
        fn precompute() {
            let z_pi_2 = RotationMatrix::from(Versor::from_axis_angle(
                [0.0, 0.0, 1.0].try_into().expect("valid unit vector"),
                PI / 2.0,
            ));
            let y_pi_4 = RotationMatrix::from(Versor::from_axis_angle(
                [0.0, 1.0, 0.0].try_into().expect("valid unit vector"),
                PI / 4.0,
            ));

            validate_rotations(&z_pi_2, &y_pi_4);
        }

        #[test]
        fn combine_different_axis() {
            let a = Versor::from_axis_angle(
                [1.0, 0.0, 0.0].try_into().expect("valid unit vector"),
                PI / 4.0,
            );
            let b = Versor::from_axis_angle(
                [0.0, 0.0, 1.0].try_into().expect("valid unit vector"),
                PI / 2.0,
            );

            let q = a.combine(&b);
            let v = q.rotate(&[1.0, 0.0, 0.0].into());
            assert_relative_eq!(v, [0.0, 2.0_f64.sqrt() / 2.0, 2.0_f64.sqrt() / 2.0].into());
        }

        #[rstest(theta => [0.0, 1.0, 2.125])]
        fn inverted(theta: f64) {
            let q1 = Versor::from_axis_angle(
                [1.0, 0.5, -2.0].try_into().expect("valid unit vector"),
                theta,
            );
            let q2 = q1.inverted();
            assert_relative_eq!(q1.combine(&q2), Versor::identity());
        }

        #[test]
        fn display() {
            let v = Versor(Quaternion {
                scalar: 0.5,
                vector: [0.125, -0.875, 2.125].into(),
            });
            let s = format!("{v}");
            assert_eq!(s, "[0.5, [0.125, -0.875, 2.125]]");
        }

        #[test]
        fn normalized() {
            let v = Versor(Quaternion {
                scalar: 5.0,
                vector: [3.0, -1.0, 1.0].into(),
            });
            assert_relative_eq!(
                v.normalized(),
                Versor(Quaternion {
                    scalar: 5.0 / 6.0,
                    vector: [3.0 / 6.0, -1.0 / 6.0, 1.0 / 6.0].into()
                })
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

            // Perform basic checks on random versors.
            // 1) Ensure that each randomly generated versor is unit.
            // 2) Check that the result of rotating a reference vector by random versors does not
            // point in any special direction. The average dot product should be close to 0.
            let samples: u32 = 20_000;

            let reference = Cartesian::from([1.0, 0.0, 0.0]);
            let mut dot_sums = [0.0; CHECK_VECTORS.len()];

            let mut rng = StdRng::seed_from_u64(1);

            for _ in 0..samples {
                let q: Versor = rng.random();
                assert_relative_eq!(q.get().norm_squared(), 1.0, max_relative = 1e-15);

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
}

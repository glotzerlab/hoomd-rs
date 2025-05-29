// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Vector and quaternion math.

`hoomd_vector` implements vector math types and operations used in scientific
computations, specifically those used in the HOOMD molecular simulation software
suite. Its API is firmly rooted in mathematical principles. Users in
other fields may find `hoomd_vector` useful outside the context of `HOOMD`.

## Vectors

The [`Vector`] trait describes any type that is a member of a normed vector
space. Write code with a [`Vector`] trait bound when you can express the
computation with vector arithmetic and dot products. Your generic code can
then be invoked on vector types with any dimension or representation (e.g.
spherical coordinates).

```
use hoomd_vector::Vector;

fn some_function<V: Vector>(a: &V, b: &V) -> f64 {
    a.dot(b) / (a.norm_squared())
}
```

Require additional trait bounds to perform more specific operations, such as [`Cross`]:
```
use hoomd_vector::{Cross, Vector};

fn triple<V: Vector + Cross>(a: &V, b: &V, c: &V) -> f64 {
    a.dot(&b.cross(c))
}
```

Use the provided [`Cartesian`] type to concretely represent N-dimensional
vectors, or when your algorithm requires Cartesian coordinates:

```
use hoomd_vector::{Cartesian, Vector};

let a = Cartesian::from([1.0, 2.0]);
let b = Cartesian::from([-2.0, 1.0]);

let product = a.dot(&b);
assert_eq!(product, 0.0);

let x = a[0];
let y = a[1];
```

## Quaternions

Quaternions are generalized complex numbers and a convenient way to describe the motion
of rotating bodies. The [`Quaternion`] type describes a single quaternion and implements
the associated algebra.

```
use hoomd_vector::Quaternion;

let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);

let norm = a.norm();
assert_eq!(norm, 57.0_f64.sqrt());

let sum = a + b;
assert_eq!(sum, [-1.0, 4.0, 10.0, -3.0].into());

let product = a * b;
assert_eq!(product, [-10.0, 32.0, -30.0, -35.0].into());
```

A **unit quaternion** (called a [`Versor`] in mathematics) can represent a 3D rotation.

```
use hoomd_vector::{Quaternion, Versor};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let q = Quaternion::from([3.0, 0.0, 0.0, 4.0]);
let v = q.to_versor()?;
assert_eq!(*v.get(), [3.0/5.0, 0.0, 0.0, 4.0/5.0].into());
# Ok(())
# }
```

## Rotations

A [`Rotation`] describes a transformation from one orthonormal basis to
another. A type that implements [`Rotation`] has an
[`identity`](Rotation::identity). Instances of that type have an
[`inverse`](Rotation::inverted) and can be [`combined`](Rotation::combine)
with other rotations.

Through the [`Rotate<V>`] trait, a [`Rotation`] can rotate a vector.

As with [`Vector`], you can implement methods that operate on generic types:
```
use hoomd_vector::{Rotate, Vector};

fn rotate_and_translate<R: Rotate<V>, V: Vector>(r: &R, a: &V, b: &V) -> V {
    r.rotate(a) + *b
}
```

[`Angle`] implements rotations on [`Cartesian<2>`] vectors.
```
use hoomd_vector::{Angle, Rotate, Rotation, Cartesian};
use std::f64::consts::PI;

let v = Cartesian::from([-1.0, 0.0]);
let a = Angle::from(PI/2.0);
let rotated = a.rotate(&v);
// rotated is approximately [0.0, -1.0]
```

[`Versor`] implements rotations on [`Cartesian<3>`] vectors.
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

Convert to a [`RotationMatrix`] when you need to rotate many vectors by the same
rotation. [`RotationMatrix::rotate`] is typically several times faster than
[`Versor::rotate`].

# Random distributions

`hoomd_vector` interoperators with [`rand`] to generate random vectors and rotations.

The [`StandardUniform`](rand::distr::StandardUniform) distribution
samples rotations uniformly from the set of all rotations and vectors from the
`[-1,1]` hypercube.


```
use hoomd_vector::{Angle, Cartesian, Versor};
use rand::{rngs::StdRng, Rng, SeedableRng};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut rng = StdRng::seed_from_u64(1);
let angle: Angle = rng.random();
let vector: Cartesian::<3> = rng.random();
let versor: Versor = rng.random();
# Ok(())
# }
```

# Feature flags

These unstable features are intended for internal use. `hoomd-vector` may make
breaking changes to the code gated behind unstable features in any release.

* `approx`: Enable `assert_relative_eq` and `assert_abs_diff_eq` from the
  [`approx`](https://docs.rs/approx/latest/approx/) crate on [`Cartesian`],
  [`Quaternion`] and [`Versor`].
*/

mod angle;
mod cartesian;
pub mod distribution;
mod quaternion;

#[cfg(any(test, feature = "approx"))]
pub mod approx;

pub use {
    angle::Angle,
    cartesian::{Cartesian, RotationMatrix},
    quaternion::{Quaternion, Versor},
};

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use thiserror::Error;

/// Enumerate possible sources of error in fallible vector math operations.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Attempted converting a value to a vector with a dimension not equal to the value's length.
    #[error("source length does not match the target dimensions")]
    InvalidVectorLength,

    /// Attempted to normalize a vector with an invalid magnitude.
    #[error("cannot normalize the 0 vector")]
    InvalidVectorMagnitude,

    /// Attempted to normalize a quaternion with an invalid magnitude.
    #[error("cannot normalize the 0 quaternion")]
    InvalidQuaternionMagnitude,
}

/** Operate on elements of a normed vector space.

Specifically, [`Vector`] defines methods that can be performed on any vector in a normed vector
space (an inner product space by default).

## Vector Operations

The following examples demonstrate vector operations applied to the following
vectors:

```
use hoomd_vector::Cartesian;

# fn main() {
let mut a = Cartesian::from([1.0, 2.0]);
let mut b = Cartesian::from([4.0, 8.0]);
# }
```

Vector addition:

```
# use hoomd_vector::Cartesian;
# fn main() {
# let mut a = Cartesian::from([1.0, 2.0]);
# let mut b = Cartesian::from([4.0, 8.0]);
let c = a + b;
assert_eq!(c, [5.0, 10.0].into())
# }
```

```
# use hoomd_vector::Cartesian;
# fn main() {
# let mut a = Cartesian::from([1.0, 2.0]);
# let mut b = Cartesian::from([4.0, 8.0]);
a += b;
assert_eq!(a, [5.0, 10.0].into())
# }
```

Vector subtraction:

```
# use hoomd_vector::Cartesian;
# fn main() {
# let mut a = Cartesian::from([1.0, 2.0]);
# let mut b = Cartesian::from([4.0, 8.0]);
let c = b - a;
assert_eq!(c, [3.0, 6.0].into())
# }
```

```
# use hoomd_vector::Cartesian;
# fn main() {
# let mut a = Cartesian::from([1.0, 2.0]);
# let mut b = Cartesian::from([4.0, 8.0]);
b -= a;
assert_eq!(b, [3.0, 6.0].into())
# }
```

Multiplication of a vector by a scalar:

```
# use hoomd_vector::Cartesian;
# fn main() {
# let mut a = Cartesian::from([1.0, 2.0]);
# let mut b = Cartesian::from([4.0, 8.0]);
let c = a * 2.0;
assert_eq!(c, [2.0, 4.0].into())
# }
```

```
# use hoomd_vector::Cartesian;
# fn main() {
# let mut a = Cartesian::from([1.0, 2.0]);
# let mut b = Cartesian::from([4.0, 8.0]);
a *= 2.0;
assert_eq!(a, [2.0, 4.0].into())
# }
```

Division of a vector by a scalar:

```
# use hoomd_vector::Cartesian;
# fn main() {
# let mut a = Cartesian::from([1.0, 2.0]);
# let mut b = Cartesian::from([4.0, 8.0]);
let c = b / 2.0;
assert_eq!(c, [2.0, 4.0].into())
# }
```

```
# use hoomd_vector::Cartesian;
# fn main() {
# let mut a = Cartesian::from([1.0, 2.0]);
# let mut b = Cartesian::from([4.0, 8.0]);
b /= 2.0;
assert_eq!(b, [2.0, 4.0].into())
# }
```

Negation:

```
# use hoomd_vector::Cartesian;
# fn main() {
# let mut a = Cartesian::from([1.0, 2.0]);
# let mut b = Cartesian::from([4.0, 8.0]);
let mut c = -a;
assert_eq!(c, [-1.0, -2.0].into());
# }
```

Equality:

```
# use hoomd_vector::Cartesian;
# fn main() {
# let mut a = Cartesian::from([1.0, 2.0]);
# let mut b = Cartesian::from([4.0, 8.0]);
assert!(a != b)
# }
```
*/
pub trait Vector:
    Add<Self, Output = Self>
    + AddAssign
    + Copy
    + Div<f64, Output = Self>
    + DivAssign<f64>
    + PartialEq
    + Mul<f64, Output = Self>
    + MulAssign<f64>
    + Sub<Self, Output = Self>
    + SubAssign
    + Neg<Output = Self>
{
    /** Compute the squared norm of the vector.

    ```math
    \left| \vec{v} \right|^2
    ```

    # Example
    ```
    use hoomd_vector::{Cartesian, Vector};

    # fn main() {
    let v = Cartesian::from([2.0, 4.0]);
    let norm_squared = v.norm_squared();
    assert_eq!(norm_squared, 20.0);
    # }
    ```
    */
    #[must_use]
    #[inline]
    fn norm_squared(&self) -> f64 {
        self.dot(self)
    }

    /** Compute the norm of the vector.

    ```math
    \left| \vec{v} \right|
    ```

    <div class="warning">

    Computing the norm calls `sqrt`. Prefer
    [`norm_squared`](Vector::norm_squared) when possible.

    </div>

    # Example
    ```
    use hoomd_vector::{Cartesian, Vector};

    # fn main() {
    let v = Cartesian::from([3.0, 4.0]);
    let norm= v.norm();
    assert_eq!(norm, 5.0);
    # }
    ```
    */
    #[must_use]
    #[inline]
    fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /** Compute the vector dot product between two vectors.

    ```math
    c = \vec{a} \cdot \vec{b}
    ```

    # Example
    ```
    use hoomd_vector::{Cartesian, Vector};

    # fn main() {
    let a = Cartesian::from([1.0, 2.0]);
    let b = Cartesian::from([3.0, 4.0]);
    let c = a.dot(&b);
    assert_eq!(c, 11.0);
    # }
    ```
    */
    #[must_use]
    fn dot(&self, other: &Self) -> f64;

    /** Project one vector onto another.

    ```math
    \frac{\vec{a} \cdot \vec{b}}{\left| \vec{b} \right|^2} \vec{b}
    ```
    where `self` is $`\vec{a}`$.

    # Example
    ```
    use hoomd_vector::{Cartesian, Vector};

    let a = Cartesian::from([1.0, 2.0]);
    let b = Cartesian::from([4.0, 0.0]);
    let c = a.project(&b);
    assert_eq!(c, [1.0, 0.0].into());
    ```
    */
    #[inline]
    #[must_use]
    fn project(&self, b: &Self) -> Self {
        *b * self.dot(b) / b.norm_squared()
    }

    /** Create a vector of unit length pointing in the same direction as the given vector.

    Returns a tuple containing unit vector along with the original vector's norm:
    ```math
    \frac{\vec{v}}{|\vec{v}|}
    ```

    # Example

    ```
    use hoomd_vector::{Cartesian, Unit, Vector};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Cartesian::from([3.0, 4.0]);
    let (unit, norm) = a.to_unit()?;
    assert_eq!(*unit.get(), [3.0/5.0, 4.0/5.0].into());
    assert_eq!(norm, 5.0);
    # Ok(())
    # }
    ```

    # Errors

    [`Error::InvalidVectorMagnitude`] when `self` is the 0 vector.
    */
    #[inline]
    fn to_unit(self) -> Result<(Unit<Self>, f64), Error> {
        let norm = self.norm();
        if norm == 0.0 {
            Err(Error::InvalidVectorMagnitude)
        } else {
            Ok((Unit(self / norm), norm))
        }
    }

    /** Create a vector of unit length pointing in the same direction as the given vector.

    Returns a tuple containing unit vector along with the original vector's norm:
    ```math
    \frac{\vec{v}}{|\vec{v}|}
    ```

    # Example

    ```
    use hoomd_vector::{Cartesian, Unit, Vector};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Cartesian::from([3.0, 4.0]);
    let (unit, norm) = a.to_unit_unchecked();
    assert_eq!(*unit.get(), [3.0/5.0, 4.0/5.0].into());
    assert_eq!(norm, 5.0);
    # Ok(())
    # }
    ```

    # Panics

    Divide by 0 when `self` is the 0 vector.
    */
    #[inline]
    fn to_unit_unchecked(self) -> (Unit<Self>, f64) {
        let norm = self.norm();
        (Unit(self / norm), norm)
    }
}

/// A [`Vector`] with magnitude 1.0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unit<V>(V);

impl<V: Vector> Unit<V> {
    /// Get the unit vector.
    #[inline]
    pub fn get(&self) -> &V {
        &self.0
    }
}

/** A vector space where the cross product is defined.
 */
pub trait Cross {
    /** Perform the cross product.
    Compute the cross product (right-handed) of two vectors:

    ```math
    \vec{c} = \vec{a} × \vec{b}
    ```

    # Example
    ```
    use hoomd_vector::{Cartesian, Cross, Vector};

    # fn main() {
    let a = Cartesian::from([1.0, 0.0, 0.0]);
    let b = Cartesian::from([0.0, 1.0, 0.0]);
    let c = a.cross(&b);
    assert_eq!(c, [0.0, 0.0, 1.0].into());
    # }
    ```
    */
    #[must_use]
    fn cross(&self, other: &Self) -> Self;
}

/** Applies the rotation operation to vectors.

The [`Rotate`] trait describes a type that can rotate a given vector. The rotated vector has the
same magnitude, but possibly a different direction.

Types that implement [`Rotate`] may or _may not_ implement [`Rotation`].
*/
pub trait Rotate<V: Vector> {
    /// Type of the related rotation matrix
    type Matrix: Rotate<V>;

    /** Rotate a vector.

    ```math
    \vec{b} = R(\vec{a})
    ```

    # Example
    ```
    use hoomd_vector::{Angle, Rotate, Rotation, Cartesian};

    let v = Cartesian::from([-1.0, 0.0]);
    let a = Angle::from(std::f64::consts::PI/2.0);
    let rotated = a.rotate(&v);
    // rotated is approximately [0.0, -1.0]
    ```
     */
    #[must_use]
    fn rotate(&self, vector: &V) -> V;
}

/** Describes the transformation from one orthonormal basis to another.

A [`Rotation`] represents a single rotation operation. Rotations change the direction of a vector
while keeping its magnitude constant. To maintain generality, this documentation shows rotations
mathematically as _functions_:
```math
\vec{b} = R(\vec{a})
```

All types that implement [`Rotation`] _should_ implement [`Rotate`] for at least one vector type.
*/
pub trait Rotation {
    /** The identity rotation.
    ```math
    \vec{a} = I(\vec{a})
    ```
    */
    #[must_use]
    fn identity() -> Self;

    /** Inverse the rotation.
    ```math
    \vec{a} = R^{-1}(R(\vec{a}))
    ```

    # Example
    ```
    # use hoomd_vector::{Rotation};
    # fn inverse<R: Rotation>(r: R) {
    let r_inverse = r.inverted();
    # }
    ```
    */
    #[must_use]
    fn inverted(self) -> Self;

    /** Combine two rotations.

    The resulting rotation `R_ab` will rotate by **first** `R_b` _followed by_ a
    rotation of `R_a`.

    ```math
    R_{ab}(\vec{v})= R_a(R_b(\vec{v}))
    ```

    # Example
    ```
    # use hoomd_vector::{Rotation};
    # fn inverse<R: Rotation>(R_a: &R, R_b: &R) {
    let R_ab = R_a.combine(R_b);
    # }
    ```
    */
    #[must_use]
    fn combine(&self, other: &Self) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compute_add_generic<T>(a: T, b: T) -> T
    where
        T: Vector,
    {
        a + b
    }

    #[test]
    fn add_generic() {
        let a = Cartesian::from([1.0, 2.0, 3.0]);
        let b = Cartesian::from([4.0, 5.0, 6.0]);
        let c = compute_add_generic(a, b);
        assert_eq!(c, [5.0, 7.0, 9.0].into());
    }
}

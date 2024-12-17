// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Vector and quaternion math.

## Overview

`hoomd_rs_vector` implements common vector math operations. The base traits [`Vector`] and
[`Rotation`] provide a generic interface. Generalize code on these base traits when possible so
so that it can be used with any type that implements the traits:

```
use hoomd_rs_vector::Vector;

fn some_function<V: Vector>(a: &V, b: &V) -> f64 {
    a.dot(b) / (a.magnitude() * b.magnitude())
}
```

Require additional trait bounds to perform more specific operations, such as [`Cross`]:
```
use hoomd_rs_vector::{Cross, Vector};

fn triple<V: Vector + Cross>(a: &V, b: &V, c: &V) -> f64 {
    a.dot(&b.cross(c))
}
```

Use trait bounds similarly when working with rotations:
```
use hoomd_rs_vector::{Rotate, Vector};

fn rotate_and_translate<R: Rotate<V>, V: Vector>(r: &R, a: &V, b: &V) -> V {
    r.rotate(a) + *b
}
```

Using these traits, you can implement custom vector types for use througought `hoomd-rs`.

## Canonical implementations

`hoomd_rs_vector` provides canonical implementations of [`Vector`] and [`Rotation`]:

* [`vector::Cartesian`] - N-dimensional cartesian vector.
* [`rotation::Angle`] - Rotation in the xy plane (interoperates with [`vector::Cartesian<2>`].
* [`rotation::Quaternion`] - 3D rotation (interoperates with [`vector::Cartesian<3>`].
*/

pub mod rotation;
pub mod vector;

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use thiserror::Error;

/// The error type provided by all fallible vector math operations.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Attempted converting a value to a vector with a dimension not equal to the value's length.
    #[error("Source does not match the target vector length.")]
    InvalidVectorLength,
}

/** A generic vector.

Specifically, [`Vector`] defines methods that can be performed on any vector in a normed vector
space (an inner product space by default).

## Vector Operations

The following examples demonstrate vector operations applied to the following
vectors:

```
use hoomd_rs_vector::vector;
# fn main() {
let mut a = vector::Cartesian::from([1.0, 2.0]);
let mut b = vector::Cartesian::from([4.0, 8.0]);
# }
```

Vector addition:

```
# use hoomd_rs_vector::vector;
# fn main() {
# let mut a = vector::Cartesian::from([1.0, 2.0]);
# let mut b = vector::Cartesian::from([4.0, 8.0]);
let c = a + b;
assert_eq!(c, [5.0, 10.0].into())
# }
```

```
# use hoomd_rs_vector::vector;
# fn main() {
# let mut a = vector::Cartesian::from([1.0, 2.0]);
# let mut b = vector::Cartesian::from([4.0, 8.0]);
a += b;
assert_eq!(a, [5.0, 10.0].into())
# }
```

Vector subtraction:

```
# use hoomd_rs_vector::vector;
# fn main() {
# let mut a = vector::Cartesian::from([1.0, 2.0]);
# let mut b = vector::Cartesian::from([4.0, 8.0]);
let c = b - a;
assert_eq!(c, [3.0, 6.0].into())
# }
```

```
# use hoomd_rs_vector::vector;
# fn main() {
# let mut a = vector::Cartesian::from([1.0, 2.0]);
# let mut b = vector::Cartesian::from([4.0, 8.0]);
b -= a;
assert_eq!(b, [3.0, 6.0].into())
# }
```

Multiplication of a vector by a scalar:

```
# use hoomd_rs_vector::vector;
# fn main() {
# let mut a = vector::Cartesian::from([1.0, 2.0]);
# let mut b = vector::Cartesian::from([4.0, 8.0]);
let c = a * 2.0;
assert_eq!(c, [2.0, 4.0].into())
# }
```

```
# use hoomd_rs_vector::vector;
# fn main() {
# let mut a = vector::Cartesian::from([1.0, 2.0]);
# let mut b = vector::Cartesian::from([4.0, 8.0]);
a *= 2.0;
assert_eq!(a, [2.0, 4.0].into())
# }
```

Division of a vector by a scalar:

```
# use hoomd_rs_vector::vector;
# fn main() {
# let mut a = vector::Cartesian::from([1.0, 2.0]);
# let mut b = vector::Cartesian::from([4.0, 8.0]);
let c = b / 2.0;
assert_eq!(c, [2.0, 4.0].into())
# }
```

```
# use hoomd_rs_vector::vector;
# fn main() {
# let mut a = vector::Cartesian::from([1.0, 2.0]);
# let mut b = vector::Cartesian::from([4.0, 8.0]);
b /= 2.0;
assert_eq!(b, [2.0, 4.0].into())
# }
```

Negation:

```
# use hoomd_rs_vector::vector;
# fn main() {
# let mut a = vector::Cartesian::from([1.0, 2.0]);
# let mut b = vector::Cartesian::from([4.0, 8.0]);
let mut c = -a;
assert_eq!(c, [-1.0, -2.0].into());
# }
```

Equality:

```
# use hoomd_rs_vector::vector;
# fn main() {
# let mut a = vector::Cartesian::from([1.0, 2.0]);
# let mut b = vector::Cartesian::from([4.0, 8.0]);
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
    + Neg
{
    /** Compute the squared magnitude of the vector.

    # Example:
    ```
    # use hoomd_rs_vector::{vector, Vector};
    # fn main() {
    let v = vector::Cartesian::from([2.0, 4.0]);
    let magnitude_squared = v.magnitude_squared();
    assert_eq!(magnitude_squared, 20.0);
    # }
    ```
    */
    #[must_use]
    #[inline]
    fn magnitude_squared(&self) -> f64 {
        self.dot(self)
    }

    /** Compute the magnitude of the vector.

    <div class="warning">

    Computing the magnitude calls `sqrt`. Prefer
    [`magnitude_squared`](Vector::magnitude_squared) unless you need the actual magnitude.

    </div>

    # Example:
    ```
    # use hoomd_rs_vector::{vector, Vector};
    # fn main() {
    let v = vector::Cartesian::from([3.0, 4.0]);
    let magnitude= v.magnitude();
    assert_eq!(magnitude, 5.0);
    # }
    ```
    */
    #[must_use]
    #[inline]
    fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    /** Compute the vector dot product between two vectors.

    # Example:
    ```
    # use hoomd_rs_vector::{vector, Vector};
    # fn main() {
    let a = vector::Cartesian::from([1.0, 2.0]);
    let b = vector::Cartesian::from([3.0, 4.0]);
    let product = a.dot(&b);
    assert_eq!(product, 11.0);
    # }
    ```
    */
    #[must_use]
    fn dot(&self, other: &Self) -> f64;
}

/** The vector cross product.
*/
pub trait Cross {
    /** Perform the cross product.
    Compute the cross product (right-handed) of two vectors:

    <!-- \vec{c} = \vec{a} \cross \vec{b} -->
    <math display="block" class="tml-display" style="display:block math;"><semantics><mrow><mover><mi>c</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo>=</mo><mover><mi>a</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mrow><mspace width="0.2222em"></mspace><mo lspace="0em" rspace="0em" style="font-weight:bold;">×</mo><mspace width="0.2222em"></mspace></mrow><mover><mi>b</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover></mrow><annotation encoding="application/x-tex">\vec{c} = \vec{a} \cross \vec{b}</annotation></semantics></math>

    ## Example
    ```
    # use hoomd_rs_vector::{vector, Cross, Vector};
    # fn main() {
    let a = vector::Cartesian::from([1.0, 0.0, 0.0]);
    let b = vector::Cartesian::from([0.0, 1.0, 0.0]);
    let c = a.cross(&b);
    assert_eq!(c, [0.0, 0.0, 1.0].into());
    # }
    ```
    */
    #[must_use]
    fn cross(&self, other: &Self) -> Self;
}

/** Rotate a vector.

The [`Rotate`] trait describes a type that can rotate a given vector. The rotated vector has the
same magnitude, but possibly a different direction.

Types that implement [`Rotate`] may or _may not_ implement [`Rotation`].
*/
pub trait Rotate<V: Vector> {
    /** Rotate a vector.

    <!-- \vec{b} = R(\vec{a}) -->
    <math display="block" class="tml-display" style="display:block math;"><semantics><mrow><mover><mi>b</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo>=</mo><mi>R</mi><mo form="prefix" stretchy="false">(</mo><mover><mi>a</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo form="postfix" stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">\vec{b} = R(\vec{a})</annotation></semantics></math>

    ## Example
    ```
    # use hoomd_rs_vector::{Rotate, Vector};
    # fn rotate<R: Rotate<V>, V: Vector>(r: &R, a: &V) {
    let b = r.rotate(a);
    # }
    ```
     */
    #[must_use]
    fn rotate(&self, vector: &V) -> V;
}

/** A rotation.

A [`Rotation`] represents a single rotation operation. Rotations change the direction of a vector
while keeping its magnitude constant. To maintain generality, this documentation shows rotations
mathematically as _functions_:
<!-- \vec{b} = R(\vec{a}) -->
<math display="block" class="tml-display" style="display:block math;"><semantics><mrow><mover><mi>b</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo>=</mo><mi>R</mi><mo form="prefix" stretchy="false">(</mo><mover><mi>a</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo form="postfix" stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">\vec{b} = R(\vec{a})</annotation></semantics></math>

All types that implement [`Rotation`] _should_ implement [`Rotate`] for at least one vector type.
*/
pub trait Rotation {
    /** The identity rotation.
    <!-- \vec{a} = I(\vec{a}) -->
    <math display="block" class="tml-display" style="display:block math;"><semantics><mrow><mover><mi>a</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo>=</mo><mi>I</mi><mo form="prefix" stretchy="false">(</mo><mover><mi>a</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo form="postfix" stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">\vec{a} = I(\vec{a})</annotation></semantics></math>
    */
    #[must_use]
    fn identity() -> Self;

    /** Inverse the rotation.
    <!-- \vec{a} = R^{-1}(R(\vec{a})) -->
    <math display="block" class="tml-display" style="display:block math;"><semantics><mrow><mover><mi>a</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo>=</mo><msup><mi>R</mi><mrow><mo lspace="0em" rspace="0em">−</mo><mn>1</mn></mrow></msup><mo form="prefix" stretchy="false">(</mo><mi>R</mi><mo form="prefix" stretchy="false">(</mo><mover><mi>a</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo form="postfix" stretchy="false">)</mo><mo form="postfix" stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">\vec{a} = R^{-1}(R(\vec{a}))</annotation></semantics></math>

    ## Example
    ```
    # use hoomd_rs_vector::{Rotation};
    # fn inverse<R: Rotation>(r: R) {
    let r_inverse = r.inversed();
    # }
    ```
    */
    #[must_use]
    fn inversed(self) -> Self;

    #[allow(clippy::doc_markdown)]
    /** Combine two rotations.

    The resulting rotation `R_ab` will rotate by `R_b` followed by a rotation of
    `R_a`.

    <!-- R_{ab}(\vec{v})= R_a(R_b(\vec{v})) -->
    <math display="block" class="tml-display" style="display:block math;"><semantics><mrow><msub><mi>R</mi><mrow><mi>a</mi><mi>b</mi></mrow></msub><mo form="prefix" stretchy="false">(</mo><mover><mi>v</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo form="postfix" stretchy="false">)</mo><mo>=</mo><msub><mi>R</mi><mi>a</mi></msub><mo form="prefix" stretchy="false">(</mo><msub><mi>R</mi><mi>b</mi></msub><mo form="prefix" stretchy="false">(</mo><mover><mi>v</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo form="postfix" stretchy="false">)</mo><mo form="postfix" stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">R_{ab}(\vec{v})= R_a(R_b(\vec{v}))</annotation></semantics></math>

    ## Example
    ```
    # use hoomd_rs_vector::{Rotation};
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
        let a = vector::Cartesian::from([1.0, 2.0, 3.0]);
        let b = vector::Cartesian::from([4.0, 5.0, 6.0]);
        let c = compute_add_generic(a, b);
        assert_eq!(c, [5.0, 7.0, 9.0].into());
    }
}

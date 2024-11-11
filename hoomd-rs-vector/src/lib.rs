// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![warn(clippy::cargo)]
#![warn(clippy::pedantic)]
// allow some pedantic rules
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::float_cmp)]
// restrictions
#![warn(
    clippy::allow_attributes,
    clippy::allow_attributes_without_reason,
    clippy::exhaustive_enums,
    clippy::impl_trait_in_params,
    clippy::missing_inline_in_public_items,
    clippy::partial_pub_fields,
    clippy::print_stderr,
    clippy::print_stdout,
    clippy::mod_module_files,
    clippy::redundant_type_annotations,
    clippy::renamed_function_params,
    clippy::same_name_method,
    clippy::todo
)]
// nursery
#![warn(
    clippy::fallible_impl_from,
    clippy::needless_collect,
    clippy::needless_pass_by_ref_mut
)]
#![deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![warn(missing_docs)]
#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Vector and quaternion math.

Generic vector and quaternion operations exposed through _traits_.

[`vector::Cartesian`] is the canonical vector representation. You can use it directly when
a specific representation is needed.

```
use hoomd_rs_vector::vector;
```

## Operations

The following examples demonstrate vector operations applied to the following
vectors:

```
# use hoomd_rs_vector::vector;
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

# Functions on generic vectors
Use the [`Vector`] trait to implement generic functions that do not depend on the dimension
or the representation of the vector:

```
use hoomd_rs_vector::Vector;

fn some_function<T: Vector>(a: &T, b: &T) -> f64 {
    a.dot(b) / (a.magnitude() * b.magnitude())
}
```

Require additional trait bounds to perform more specific operations, such as [`Cross`]:

```
use hoomd_rs_vector::{Cross, Vector};

fn triple<T: Vector + Cross>(a: &T, b: &T, c: &T) -> f64 {
    a.dot(&b.cross(c))
}
```

Or to require a specific number of dimensions with [`Dimension`]:

```
use hoomd_rs_vector::{Dimension, Vector};

fn some_3d_function<T: Vector<Dimension = Dimension<3>>>(a: &T) {
    // ...
}
```
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

    /// Attempted normalizing a vector with an invalid magnitude (e.g., zero).
    #[error("Invalid magnitude for normalization.")]
    InvalidMagnitude,
}

/** Placeholder type for use in trait bounds on [`Vector`] dimension.

    This placeholder will be removed when Rust allows the use of [associated constants in trait
    bounds](https://github.com/rust-lang/rust/issues/92827).
*/
pub struct Dimension<const N: usize>;

/// A generic vector.
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
    /// The dimension of the vector space.
    type Dimension;

    // Ideally, we would use:
    // const DIMENSION: usize;
    // but trait bounds cannot be applied to associated constants as of rust 1.81.0.
    // https://github.com/rust-lang/rust/issues/92827
    // Instead, introduce an empty type for use with trait bounds on an associated type.

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
    fn dot(&self, rhs: &Self) -> f64;

    #[must_use]
    fn normalized(&self) -> Result<Self, crate::Error> {
        let mag = self.magnitude();
        if mag == 0.0 {
            Err(Error::InvalidMagnitude)
        } else {
            Ok(*self / mag)
        }
    }
}

/** The vector cross product.

Compute the cross product (right-handed) of two vectors.

```
# use hoomd_rs_vector::{vector, Cross, Vector};
# fn main() {
let a = vector::Cartesian::from([1.0, 0.0, 0.0]);
let b = vector::Cartesian::from([0.0, 1.0, 0.0]);
let product = a.cross(&b);
assert_eq!(product, [0.0, 0.0, 1.0].into());
# }
```
*/
pub trait Cross {
    /// Perform the cross product.
    #[must_use]
    fn cross(&self, rhs: &Self) -> Self;
}

/** Rotate a vector or rotation.
*/
pub trait Rotate<V: Vector> {
    /** Rotate a vector.
     */
    #[must_use]
    fn rotate(&self, vector: &V) -> V;
}

/** A rotation.
*/
pub trait Rotation {
    /** Combine two rotations.

    The resulting rotation `c = a.combine(b)` will rotate by `b` followed by a rotation of
    `a`.
    */
    #[must_use]
    fn combine(&self, rotation: &Self) -> Self;
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

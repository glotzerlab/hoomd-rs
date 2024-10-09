// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd_rs, released under the BSD 3-Clause License.

#![warn(clippy::cargo)]
#![warn(clippy::pedantic)]

// allow some pedantic rules
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::float_cmp)]

// restrictions
#![warn(clippy::allow_attributes,
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
        clippy::todo)]

// nursery
#![warn(clippy::fallible_impl_from,
        clippy::needless_collect,
        clippy::needless_pass_by_ref_mut)]

#![deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Vector and quaternion math.
//!
//! Generic vector and quaternion operations exposed through _traits_.
//!

mod cartesian;

pub use cartesian::CartesianVector;

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use thiserror::Error;

#[non_exhaustive]
#[derive(Error, Debug)]
pub enum Error {
    #[error("Source does not match the target vector length.")]
    InvalidVectorLength,
}

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
{
    /// The dimension of the vector space.
    const DIMENSION: usize;

    /// Length of the vector, squared.
    #[must_use]
    #[inline]
    fn length_squared(&self) -> f64 {
        self.dot(self)
    }

    /// Length of the vector.
    #[must_use]
    #[inline]
    fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Vector dot product
    #[must_use]
    fn dot(&self, rhs: &Self) -> f64;
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
        let a = CartesianVector::from([1.0, 2.0, 3.0]);
        let b = CartesianVector::from([4.0, 5.0, 6.0]);
        let c = compute_add_generic(a, b);
        assert_eq!(c, [5.0, 7.0, 9.0].into());
    }
}

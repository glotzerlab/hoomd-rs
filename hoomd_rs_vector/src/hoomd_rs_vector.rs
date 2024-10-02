// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd_rs, released under the BSD 3-Clause License.

//! Vector and quaternion math.
//!
//! Generic vector and quaternion operations exposed through _traits_.
//!

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

mod cartesian;
mod vec3;

pub use cartesian::CartesianVector;
pub use vec3::CartesianVector3;

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
    fn length_squared(&self) -> f64 {
        self.dot(self)
    }

    /// Length of the vector.
    #[must_use]
    fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Vector dot product
    #[must_use]
    fn dot(&self, rhs: &Self) -> f64;
}

// This is an alternative to supertraits. It allows trait bounds to opt-in to requiring
// vector operations. I'm not sure that we need that flexibility, the supertrait solution is
// simpler - JAA.
// pub trait  : Add + Copy + PartialEqVectorOps<Rhs = Self, Output = Self>:
//     Add<Rhs, Output = Output>
//     + Copy
//     + PartialEq
//     // + Sub<Rhs, Output = Output>
//     // + Mul<Rhs, Output = Output>
//     // + Div<Rhs, Output = Output>
// {
// }

// impl<T, Rhs, Output> VectorOps<Rhs, Output> for T where
//     T: Add<Rhs, Output = Output>
//     + Copy
//     + PartialEq
//         // + Sub<Rhs, Output = Output>
//         // + Mul<Rhs, Output = Output>
//         // + Div<Rhs, Output = Output>
// {
// }

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

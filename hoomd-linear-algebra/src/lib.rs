// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Traits and subroutines for common linear algebra operations.
//!
//! This crate places an emphasis on generality and simplicity, with optimization efforts
//! targeted at small matrices. Some complex routines (SVD, matrix inversion, etc.) will
//! only be implemented for certain shapes, and generally consist of specialized algorithms
//! optimal for those inputs.
//!
//! This crate should not be considered a replacement for a dedicated linear algebra library
//! like [faer-rs](https://github.com/sarah-quinones/faer-rs.git) or
//! [nalgebra](https://github.com/dimforge/nalgebra). Instead, it can be used as a simple
//! and lightweight dependency for matrix methods common to molecular simulation and
//! analysis.

use std::ops::{Add, Index, Mul, Neg};

/// Structs implementing a large subset of Matrix traits.
pub mod matrix;
use matrix::DiagonalMatrix;

/// A lightweight representation of a diagonal matrix.
mod diagonal;

/// Define whether a matrix $`A`$ has an inverse $`A^{-1}`$ such that $`AA^{-1} = A^{-1}A = I`$
pub trait Invertible
where
    Self: Sized,
{
    /// Compute the inverse of a matrix. Will be `None` if the matrix is not invertible.
    #[must_use]
    fn inverse(&self) -> Option<Self>;
}

/// Define the general matrix multiplication (GEMM) subroutine.
///
/// # Example
/// ```
/// use hoomd_linear_algebra::{
///     Full, GeneralMatrix, MatMul, SquareMatrix,
///     matrix::{DiagonalMatrix, Matrix, Matrix22},
/// };
///
/// let mat = Matrix22::full(5.0);
/// assert_eq!(mat.matmul(&Matrix22::identity()), mat);
///
/// let diag = Matrix::with_diagonal([3.0, 2.0]);
///
/// assert_eq!(
///     mat.matmul(&diag),
///     mat.matmul(&DiagonalMatrix { elements: [3.0, 2.0] })
/// );
/// ```
pub trait MatMul<RHS>
where
    RHS: GeneralMatrix,
{
    /// The type of the output matrix.
    ///
    /// This type is likely to be Self for dynamically sized [`GeneralMatrix`] types, but
    /// will necessarily be different for statically allocated rectangular matrices.
    type Output;

    /// Multiply a matrix by a general matrix RHS.
    #[must_use]
    fn matmul(&self, rhs: &RHS) -> Self::Output;
}

/// General implementation for size and container-agnostic matrices.
///
/// This trait is designed to function with row-major ordering, but this is not strictly
/// required for correct functionality.
pub trait GeneralMatrix:
    Sized
    + Mul<f64, Output = Self>
    + Add<Self, Output = Self>
    + Index<(usize, usize), Output = f64>
    + Neg
{
    /// Fill a matrix with zeros.
    #[must_use]
    fn zeros() -> Self;

}

/// Initialize matrices with a constant value.
pub trait Full {
    /// Construct a matrix the same value in every element.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{Full, matrix::Matrix22};
    /// let m = Matrix22::full(5.0);
    /// assert_eq!(m.rows, [[5.0, 5.0], [5.0, 5.0]]);
    /// ```
    #[must_use]
    fn full(value: f64) -> Self;
}

/// Define properties and implementations that are well-defined for all square matrices.
pub trait SquareMatrix: GeneralMatrix
where
    Self: Sized,
{
    /// Construct an N x N identity matrix, with ones on the diagonal and zeros elsewhere.
    #[must_use]
    fn identity() -> Self;
}

/// Solve the quadratic form `A.transpose().matmul(x).matmul(A)`.
pub trait QuadraticForm<const N: usize>: SquareMatrix {
    /// Evaluate the quadratic form for a matrix `A` and a vector `x`.
    #[must_use]
    fn compute_quadratic_form(&self, variables: &DiagonalMatrix<N>) -> f64;
}

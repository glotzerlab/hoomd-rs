// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Traits and subroutines for common linear algebra operations.
//!
//! This crate places an emphasis on generality and simplicity, with optimization efforts
//! targeted at small matrices. Some complex routines (SVD, matrix inversion, etc.) will
//! only be implemented for certain shapes, and generally consist of specialized algorithms
//! optimal for those inputs.
//!
//! This crate should not be considered a replacement for a dedicated linear
//! algebra library like [faer-rs] or [nalgebra]. Instead, it can be used as a
//! simple and lightweight dependency for matrix methods common to molecular
//! simulation and analysis.
//!
//! [faer-rs]: https://github.com/sarah-quinones/faer-rs.git
//! [nalgebra]: https://github.com/dimforge/nalgebra

use std::ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};

/// Structs implementing a large subset of Matrix traits.
pub mod matrix;

/// A lightweight representation of a diagonal matrix.
mod diagonal;

/// Compute the inverse of a matrix.
///
/// A matrix $`A`$ has an inverse $`A^{-1}`$ such that $`AA^{-1} = A^{-1}A = I`$.
pub trait Invertible
where
    Self: Sized
{
    /// Compute the inverse of a matrix.
    ///
    /// Returns `None` when the matrix is not invertible.
    ///
    /// # Example
    /// 
    /// ```
    /// use hoomd_linear_algebra::{Invertible, SquareMatrix, matrix::Matrix};
    /// let m = Matrix::identity() * 5.0;
    /// let m_inv = m.inverse();
    ///
    /// assert_eq!(m_inv, Some(Matrix::with_diagonal([1.0 / 5.0; 3])));
    /// ```
    #[must_use]
    fn inverse(&self) -> Option<Self>;
}

/// Matrix multiplication.
pub trait MatMul<Rhs> {
    /// The type of the output matrix.
    type Output;

    /// Multiply two matrices.
    ///
    /// # Example
    /// ```
    /// use hoomd_linear_algebra::{
    ///     Full, GeneralMatrix, MatMul, SquareMatrix,
    ///     matrix::{DiagonalMatrix, Matrix, Matrix22},
    /// };
    ///
    /// let a = Matrix22::full(5.0);
    /// assert_eq!(a.matmul(&Matrix22::identity()), a);
    ///
    /// let b = Matrix::with_diagonal([3.0, 2.0]);
    ///
    /// assert_eq!(
    ///     a.matmul(&b).rows,
    ///     [[15.0, 10.0], [15.0, 10.0]]
    /// );
    /// ```
    #[must_use]
    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

/// Common operations for all matrices.
///
/// Matrices can be added:
/// ```
/// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
/// let a = Matrix { rows: [[1.0, -3.0], [-2.0, 4.0]] };
/// let b = Matrix { rows: [[4.0, -8.0], [6.0, 7.0]] };
///
/// let c = a + b;
/// ```
///
/// Subtracted:
/// ```
/// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
/// let a = Matrix { rows: [[1.0, -3.0], [-2.0, 4.0]] };
/// let b = Matrix { rows: [[4.0, -8.0], [6.0, 7.0]] };
///
/// let c = a - b;
/// ```
///
/// Multiplied by a scalar:
/// ```
/// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
/// let a = -2.0;
/// let b = Matrix { rows: [[4.0, -8.0], [6.0, 7.0]] };
///
/// let c = a * b;
/// ```
///
/// Negated:
/// ```
/// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
/// let a = Matrix { rows: [[1.0, -3.0], [-2.0, 4.0]] };
///
/// let b = -a;
/// ```
///
/// and indexed (in row,column ordering):
/// ```
/// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
/// let a = Matrix { rows: [[1.0, -3.0], [-2.0, 4.0]] };
///
/// let element = a[(1,0)];
/// ```
pub trait GeneralMatrix:
    Add<Self, Output = Self>
    + AddAssign<Self>
    + Index<(usize, usize), Output = f64>
    + Mul<f64, Output = Self>
    + MulAssign<f64>
    + Neg<Output = Self>
    + Sized
    + Sub<Self, Output = Self>
    + SubAssign<Self>
{
    /// Fill a matrix with zeros.
    ///
    /// # Example
    /// 
    /// ```
    /// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix};
    ///
    /// let a = Matrix::zeros();
    ///
    /// assert_eq!(a.rows, [[0.0, 0.0], [0.0, 0.0]]);
    /// ```
    #[must_use]
    fn zeros() -> Self;

}

/// Initialize matrices with identical elements.
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

/// Matrices that have the same number of rows and columns.
pub trait SquareMatrix: GeneralMatrix
{
    /// Construct an N x N identity matrix.
    ///
    /// # Example
    /// ```
    /// use hoomd_linear_algebra::{SquareMatrix, matrix::Matrix22};
    /// let m = Matrix22::identity();
    /// assert_eq!(m.rows, [[1.0, 0.0], [0.0, 1.0]]);
    /// ```
    #[must_use]
    fn identity() -> Self;
}

/// Solve the quadratic form.
///
/// ```math
/// x^T A x
/// ```
pub trait QuadraticForm<const N: usize>: SquareMatrix {
    /// Evaluate the quadratic form.
    ///
    /// The matrix `A` is given by `self` and the vector `x` in the argument.
    #[must_use]
    fn compute_quadratic_form(&self, x: &[f64; N]) -> f64;
}

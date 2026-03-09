// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Linear algebra optimized for small matrices.
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
//!
//! ## Matrices
//!
//! Construct a small rectangular [`Matrix`] on the stack:
//! ```
//! use hoomd_linear_algebra::matrix::Matrix;
//!
//! let m = Matrix {
//!     rows: [[1.0, 2.0, 3.0], [-6.0, 3.0, 2.0]],
//! };
//! ```
//!
//! [`Matrix`]: matrix::Matrix
//!
//! Diagonal matrices ([`DiagonalMatrix`]) store only the diagonal elements:
//! ```
//! use hoomd_linear_algebra::matrix::DiagonalMatrix;
//!
//! let m = DiagonalMatrix {
//!     elements: [-2.0, 4.0, -5.0],
//! };
//! ```
//!
//! [`DiagonalMatrix`]: matrix::DiagonalMatrix
//!
//! [`Matrix22`], [`Matrix33`], and [`Matrix44`] are type aliases for commonly used
//! matrix sizes. Construct a 2x2 matrix with every element set to 4:
//! ```
//! use hoomd_linear_algebra::{Full, matrix::Matrix22};
//!
//! let m = Matrix22::full(4.0);
//! ```
//!
//! [`Matrix22`]: matrix::Matrix22
//! [`Matrix33`]: matrix::Matrix33
//! [`Matrix44`]: matrix::Matrix44
//!
//! Construct a 3x3 identity matrix $` \mathbf{I} `$:
//! ```
//! use hoomd_linear_algebra::{SquareMatrix, matrix::Matrix44};
//!
//! let m = Matrix44::identity();
//! ```
//!
//! Index matrix entries by `(row, column)`:
//! ```
//! use hoomd_linear_algebra::matrix::Matrix;
//!
//! let m = Matrix {
//!     rows: [[1.0, 2.0, 3.0], [-6.0, 3.0, 2.0]],
//! };
//!
//! let element = m[(1, 2)];
//! ```
//!
//! ## Matrix Operations
//!
//! Matrices can be added:
//! ```
//! use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
//! let a = Matrix {
//!     rows: [[1.0, -3.0], [-2.0, 4.0]],
//! };
//! let b = Matrix {
//!     rows: [[4.0, -8.0], [6.0, 7.0]],
//! };
//!
//! let mut c = a + b;
//! c += a;
//! ```
//!
//! subtracted:
//! ```
//! use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
//! let a = Matrix {
//!     rows: [[1.0, -3.0], [-2.0, 4.0]],
//! };
//! let b = Matrix {
//!     rows: [[4.0, -8.0], [6.0, 7.0]],
//! };
//!
//! let mut c = a - b;
//! c += a;
//! ```
//!
//! multiplied by a scalar:
//! ```
//! use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
//! let a = Matrix {
//!     rows: [[4.0, -8.0], [6.0, 7.0]],
//! };
//! let b = -2.0;
//!
//! let mut c = a * b;
//! c *= b;
//! ```
//!
//! negated:
//! ```
//! use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
//! let a = Matrix {
//!     rows: [[1.0, -3.0], [-2.0, 4.0]],
//! };
//!
//! let b = -a;
//! ```
//!
//! and indexed (in row,column ordering):
//! ```
//! use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
//! let a = Matrix {
//!     rows: [[1.0, -3.0], [-2.0, 4.0]],
//! };
//!
//! let element = a[(1, 0)];
//! ```
//!
//! You can also perform matrix-matrix multiplication:
//! ```
//! use hoomd_linear_algebra::{MatMul, matrix::Matrix};
//!
//! let a = Matrix {
//!     rows: [[1.0, 2.0], [3.0, 4.0]],
//! };
//! let b = Matrix {
//!     rows: [[4.0, 3.0], [2.0, 1.0]],
//! };
//!
//! let c = a.matmul(&b);
//! ```
//!
//! and invert matrices:
//! ```
//! use hoomd_linear_algebra::{Invertible, matrix::Matrix};
//!
//! let a = Matrix {
//!     rows: [[1.0, 2.0], [3.0, 4.0]],
//! };
//!
//! let b = a.inverse();
//! ```
//!
//! ## Numerical Algorithms
//!
//! `hoomd-linear-algebra` implements a number of numerical algorithms on
//! matrices:
//!
//! * [`Determinant`](matrix::Matrix::determinant)
//! * [`Singular value decomposition (2x2)`](matrix::Matrix22::svd)
//! * [`Singular value decomposition (3x3)`](matrix::Matrix33::svd)
//! * [`Quadratic form`](QuadraticForm)
//!
//! # Complete documentation
//!
//! `hoomd-linear-algebra` is is a part of *hoomd-rs*. Read the [complete documentation]
//! for more information.
//!
//! [complete documentation]: https://glotzerlab-hoomd-rs.readthedocs-hosted.com

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
    Self: Sized,
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
    /// assert_eq!(a.matmul(&b).rows, [[15.0, 10.0], [15.0, 10.0]]);
    /// ```
    #[must_use]
    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}

/// Common operations for all matrices.
///
/// Matrices can be added:
/// ```
/// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
/// let a = Matrix {
///     rows: [[1.0, -3.0], [-2.0, 4.0]],
/// };
/// let b = Matrix {
///     rows: [[4.0, -8.0], [6.0, 7.0]],
/// };
///
/// let mut c = a + b;
/// c += a;
/// ```
///
/// subtracted:
/// ```
/// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
/// let a = Matrix {
///     rows: [[1.0, -3.0], [-2.0, 4.0]],
/// };
/// let b = Matrix {
///     rows: [[4.0, -8.0], [6.0, 7.0]],
/// };
///
/// let mut c = a - b;
/// c += a;
/// ```
///
/// multiplied by a scalar:
/// ```
/// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
/// let a = Matrix {
///     rows: [[4.0, -8.0], [6.0, 7.0]],
/// };
/// let b = -2.0;
///
/// let mut c = a * b;
/// c *= b;
/// ```
///
/// negated:
/// ```
/// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
/// let a = Matrix {
///     rows: [[1.0, -3.0], [-2.0, 4.0]],
/// };
///
/// let b = -a;
/// ```
///
/// and indexed (in row,column ordering):
/// ```
/// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix};
/// let a = Matrix {
///     rows: [[1.0, -3.0], [-2.0, 4.0]],
/// };
///
/// let element = a[(1, 0)];
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

    /// Get the shape of a matrix (``n_rows,n_columns``).
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix};
    ///
    /// let a = Matrix::<5, 7>::zeros();
    /// assert_eq!(a.shape(), (5, 7));
    /// ```
    #[must_use]
    fn shape(&self) -> (usize, usize);
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
pub trait SquareMatrix: GeneralMatrix {
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
/// \mathbf{x}^{\intercal} \mathbf{A} \mathbf{x}
/// ```
pub trait QuadraticForm<const N: usize>: SquareMatrix {
    /// Evaluate the quadratic form.
    ///
    /// The matrix `A` is given by `self` and the vector `x` in the argument.
    #[must_use]
    fn compute_quadratic_form(&self, x: &[f64; N]) -> f64;
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! asdf

TODO: Expand documentation.
*/

use std::ops::{Add, Index, Mul};

/** Define whether a matrix $ A $ has an inverse $ A^-1 $ such that $ AA^-1 = A^-1A = I $
*/
pub trait Invertible {
    /// Compute the inverse of a matrix.
    #[must_use]
    fn inverse(&self) -> Self;
}

/** Define operations for matrix multiplication.
*/
pub trait MatMul {
    /// The type of the righthand side of the multiplication.
    type RHS;
    /// The type of the output matrix. May or may not be Self.
    type Output;
    /// Multiply a matrix by a general RHS
    #[must_use]
    fn matmul(&self, rhs: &Self::RHS) -> Self::Output;
    /// Multiply a matrix by a diagonal RHS.
    #[must_use]
    fn matmul_diagonal(&self, rhs: &Self::RHS) -> Self::Output;
}

/** General implementation for size and container-agnostic matrixes.

This trait is designed to function with row-major ordering, but this is not strictly
required for correct functionality.
*/
pub trait GeneralMatrix: Sized + Mul<f64, Output = Self> + Add<Self, Output = Self> {
    /// TODO
    #[must_use]
    fn zeros() -> Self;

    /// Iterate over the rows of a matrix.
    #[must_use]
    fn iter_rows(&self) -> impl Iterator<Item = impl IntoIterator<Item = &f64>>;

    /// Return a matrix where every element is equal to val
    #[must_use]
    fn full(val: f64) -> Self;

    // TODO: Index<(usize, usize)>
}

/// Marker trait to indicate a sequence of values can be read as a diagonal matrix.
pub trait Diagonal: Index<usize, Output = f64> {}

/** Define properties and implementations that are well-defined for all square matrixes.
*/
pub trait SquareMatrix: GeneralMatrix
where
    Self: Sized,
{
    /// Return an N x N identity matrix, with ones on the diagonal and zeros elsewhere.
    #[must_use]
    fn eye() -> Self;

    /** Solve the quadratic form $ A^T @ x @ A $ for a matrix .*/
    #[must_use]
    fn compute_quadratic_form(&self, vars: &impl Diagonal) -> f64;
}

/// Compute the signed hypervolume of the hyperparallelepiped defined by a matrix.
pub trait Determinant: SquareMatrix {
    /** Compute the determinant of a matrix.*/
    #[must_use]
    fn det(&self) -> f64;
}

/**
*/
pub mod matrix;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;
    use approx::assert_relative_eq;
    use faer::Mat;
    use rstest::rstest;

    fn fill_faer<const N: usize, const M: usize>(m: [[f64; M]; N]) -> Mat<f64> {
        let mut faer_matrix = Mat::<f64>::zeros(N, N);
        for (i, row) in m.iter().enumerate() {
            for (j, el) in row.iter().enumerate() {
                *faer_matrix.get_mut(i, j) = *el;
            }
        }
        faer_matrix
    }
    #[rstest(
        rows,
        case([[-9.0]]),
        case([[1.0, -2.0], [3.0, 4.0]]),
        case([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]),
        case([[2.0, 0.0, 1.0], [3.0, 0.0, 0.0], [5.0, 1.0, 1.0]]),
        case(Matrix::<4, 4>::eye().rows),
        case(Matrix::<5, 5>::full(3.6).diag().rows),
        case(Matrix::<8, 8>::eye().rows),
    )]
    fn test_determinant_parametrized<const N: usize>(rows: [[f64; N]; N]) {
        let matrix = Matrix { rows };
        let faer_matrix = fill_faer(rows);

        let custom_det = matrix.det();
        let faer_det = faer_matrix.determinant();

        assert_relative_eq!(custom_det, faer_det, max_relative = 1e-14);
    }
}

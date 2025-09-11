// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! asdf

TODO: Expand documentation.
*/

use std::ops::{Add, Index, Mul};

use hoomd_vector::Vector;

/** Define whether a matrix $ A $ has an inverse $ A^-1 $ such that $ AA^-1 = A^-1A = I $
*/
pub trait Invertible {
    /// Compute the inverse of a matrix.
    #[must_use]
    fn inverse(&self) -> Self;
}

/** Define operations for matrix multiplication.
*/
pub trait MatMul<RHS>
where
    RHS: GeneralMatrix,
{
    /// The type of the output matrix. May or may not be Self.
    type Output;

    /// Multiply a matrix by a general matrix RHS (gemm).
    #[must_use]
    fn matmul(&self, rhs: &RHS) -> Self::Output;
}

// /// Multiply a matrix by a vector. (gemm)
// #[must_use]
// fn matmul_vec(&self, rhs: &impl Vector) -> Self::OutputMatrix;

/** General implementation for size and container-agnostic matrixes.

This trait is designed to function with row-major ordering, but this is not strictly
required for correct functionality.
*/
pub trait GeneralMatrix:
    Sized + Mul<f64, Output = Self> + Add<Self, Output = Self> + Index<(usize, usize), Output = f64>
{
    /// Fill a matrix with zeros.
    #[must_use]
    fn zeros() -> Self;

    /// Return a matrix where every element is equal to val.
    #[must_use]
    fn full(val: f64) -> Self;
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
    use std::marker::PhantomData;

    type RectSize<const M: usize, const N: usize> = PhantomData<([f64; M], [f64; N])>;

    fn fill_faer<const N: usize, const M: usize>(m: [[f64; M]; N]) -> Mat<f64> {
        let mut faer_matrix = Mat::<f64>::zeros(N, N);
        for (i, row) in m.iter().enumerate() {
            for (j, el) in row.iter().enumerate() {
                *faer_matrix.get_mut(i, j) = *el;
            }
        }
        faer_matrix
    }
    fn assert_matrixes_relative_eq<const N: usize, const M: usize>(
        m0: Matrix<N, M>,
        m1: faer::Mat<f64>,
    ) {
        for i in 0..N {
            for j in 0..M {
                assert_relative_eq!(m0[(i, j)], m1[(i, j)], max_relative = 1e-14);
            }
        }
    }
    #[rstest(
        rows,
        case([[-9.0]]),
        case([[1.0, -2.0], [3.0, 4.0]]),
        case([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]),
        case([[2.0, 0.0, 1.0], [3.0, 0.0, 0.0], [5.0, 1.0, 1.0]]),
        case(Matrix::<4, 4>::eye().rows),
        case(Matrix::<5, 5>::full(3.6).diag().as_dense().rows),
        case(Matrix::<8, 8>::eye().rows),
    )]
    fn test_determinant<const N: usize>(rows: [[f64; N]; N]) {
        let matrix = Matrix { rows };
        let faer_matrix = fill_faer(rows);

        let custom_det = matrix.det();
        let faer_det = faer_matrix.determinant();

        assert_relative_eq!(custom_det, faer_det, max_relative = 1e-14);
    }
    #[rstest(
        a_rows, b_rows,
        case([[-9.0]], [[-9.0]]),
        case(
            [[1.0, -2.0], [3.0, 4.0]], [[0.0, 1.0], [1.0, 0.0]]
        ),
        case(
            [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]],
            [[-2.0, 1.0, 0.0], [3.0, 0.0, 1.0], [1.0, 4.0, -1.0]]
        ),
        case(
            [[2.0, 0.0, 1.0], [3.0, 0.0, 0.0], [5.0, 1.0, 1.0]],
            [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [4.0, 0.0, 0.0]]
        ),
        case(Matrix::<4, 4>::eye().rows, Matrix::<4, 4>::full(2.0).rows),
        case(Matrix::<5, 5>::full(3.6).diag().as_dense().rows, Matrix::<5, 5>::eye().rows),
        case(Matrix::<8, 8>::eye().rows, Matrix::<8, 8>::full(1.5).rows),
    )]
    fn test_matrix_multiply_square<const N: usize>(a_rows: [[f64; N]; N], b_rows: [[f64; N]; N]) {
        let a = Matrix { rows: a_rows };
        let b = Matrix { rows: b_rows };

        let faer_a = fill_faer(a_rows);
        let faer_b = fill_faer(b_rows);

        let custom_prod = a.matmul(&b);
        let faer_prod = faer_a * faer_b;
        assert_matrixes_relative_eq(custom_prod, faer_prod);
    }


    #[rstest]
    #[case(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], 
    )]
    #[case(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], 
        [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], 
    )]
    #[case(
        [[1.0, 2.0]], 
        [[3.0], [4.0]], 
    )]
    #[case(
        [[2.0, 0.0, 1.0]], 
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], 
    )]
    fn test_rectangular_matrix_multiply<
        const M: usize, const K: usize, const N: usize
    >(
        #[case] a_rows: [[f64; K]; M],
        #[case] b_rows: [[f64; N]; K],
    ) {
        let a = Matrix { rows: a_rows };
        let b = Matrix { rows: b_rows };

        let faer_a = fill_faer(a_rows);
        let faer_b = fill_faer(b_rows);

        let custom_prod = a.matmul(&b);
        let faer_prod = faer_a * faer_b;
        assert_matrixes_relative_eq(custom_prod, faer_prod);
    }


}

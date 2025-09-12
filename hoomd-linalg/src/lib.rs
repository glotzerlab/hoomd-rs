// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Traits and subroutines for common linear algebra operations.

This crate places an emphasis on generality and simplicity, with optimization efforts
targeted at small matrixes. Some complex routines (SVD, matrix inversion, etc.) will
only be implemented for certain shapes, and generally consist of specialized algorithms
optimal for those inputs.
*/

use std::ops::{Add, Index, Mul, Neg};

/** Define whether a matrix $ A $ has an inverse $ A^-1 $ such that $ AA^-1 = A^-1A = I $
*/
pub trait Invertible {
    /// Compute the inverse of a matrix.
    #[must_use]
    fn inverse(&self) -> Self;
}

/** Define the general matrix multiplication (GEMM) subroutine.

    # Example
    ```
    use hoomd_linalg::{matrix::Matrix22, MatMul, SquareMatrix, GeneralMatrix};

    let mat = Matrix22::full(5.0);
    assert_eq!(mat.matmul(&Matrix22::eye()), mat);


    let diag = Matrix22::from_diag(&[3.0, 2.0]);

    assert_eq!(
      mat.matmul(&diag),
      mat.matmul_diagonal(&[3.0, 2.0])
    );
    ```
*/
pub trait MatMul<RHS>
where
    RHS: GeneralMatrix,
{
    /** The type of the output matrix.

    This type is likely to be Self for dynamically sized [`GeneralMatrix`] types, but
    will necessarily be different for statically allocated rectangular matrixes.
    */
    type Output;

    /** Multiply a matrix by a general matrix RHS.
     */
    #[must_use]
    fn matmul(&self, rhs: &RHS) -> Self::Output;
}

/** General implementation for size and container-agnostic matrixes.

This trait is designed to function with row-major ordering, but this is not strictly
required for correct functionality.
*/
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

/** Compute the signed hypervolume of the hyperparallelepiped defined by a matrix.

    # Example
    ```
    use hoomd_linalg::{matrix::Matrix22, Determinant, SquareMatrix};

    let eye = Matrix22::eye();
    assert_eq!(eye.det(), 1.0);

    let scaled = eye * 2.0;
    assert_eq!(scaled.det(), 2.0 * 2.0);
    ```
*/
pub trait Determinant: SquareMatrix {
    /// Compute the determinant of a matrix.
    #[must_use]
    fn det(&self) -> f64;
}

/** Structs implementing a large subset of Matrix traits.
*/
pub mod matrix;

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use super::*;
    use crate::matrix::{Matrix, Matrix22};
    use approx::{assert_relative_eq, assert_ulps_eq, ulps_eq};
    use faer::Mat;
    use rstest::rstest;

    const EPS: f64 = 1e-13;

    fn fill_faer<const N: usize, const M: usize>(m: [[f64; M]; N]) -> Mat<f64> {
        let mut faer_matrix = Mat::<f64>::zeros(N, M);
        for (i, row) in m.iter().enumerate() {
            for (j, el) in row.iter().enumerate() {
                *faer_matrix.get_mut(i, j) = *el;
            }
        }
        faer_matrix
    }
    fn assert_matrixes_ulps_eq<
        const N: usize,
        const M: usize,
        T0: Index<(usize, usize), Output = f64> + Debug,
        T1: Index<(usize, usize), Output = f64> + Debug,
    >(
        m0: &T0,
        m1: &T1,
    ) {
        for i in 0..N {
            for j in 0..M {
                if !ulps_eq!(m0[(i, j)], m1[(i, j)], epsilon = EPS) {
                    assert_ulps_eq!(m0[(i, j)], m1[(i, j)], epsilon = EPS);
                }
            }
        }
    }
    fn assert_diags_ulps_eq<const N: usize, T: Diagonal>(
        m0: &T,
        m1: &impl Index<usize, Output = f64>,
    ) {
        for i in 0..N {
            assert_ulps_eq!(m0[i], m1[i], epsilon = EPS);
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
        assert_matrixes_ulps_eq::<N, N, _, _>(&custom_prod, &faer_prod);
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
    fn test_rectangular_matrix_multiply<const M: usize, const K: usize, const N: usize>(
        #[case] a_rows: [[f64; M]; N],
        #[case] b_rows: [[f64; K]; M],
    ) {
        let a = Matrix { rows: a_rows };
        let b = Matrix { rows: b_rows };

        let faer_a = fill_faer(a_rows);
        let faer_b = fill_faer(b_rows);

        let custom_prod = a.matmul(&b);
        let faer_prod = faer_a * faer_b;
        assert_matrixes_ulps_eq::<N, K, _, _>(&custom_prod, &faer_prod);
    }

    #[rstest(
        rows,
        case::identity(Matrix22::eye().rows),
        case::mixed_sign([[1.0, -2.0], [3.0, 4.0]]),
        case::det_zero([[12.0, 2.0], [4.0, 0.0]]),
        case::large_range([[1000.0, 0.0], [0.0, 1e-4]]),
        case::jordan_block([[1.0, 1.0], [0.0, 1.0]]),
        case::full_ones(Matrix22::full(1.0).rows),
        case::shear([[1.0, 2.0], [0.0, 1.0]]),
        case::nilpotent([[0.0, 1.0], [0.0, 0.0]]),
        case::scaling([[2.0, 0.0], [0.0, 3.0]]),
        /* None of these examples work using the fast algorithm.*/
        // case::reflect([[0.0, -1.0], [1.0, 0.0]]),
        // case::negative_identity((Matrix22::eye()*-1.0).rows),
        // case::anti_diagonal([[0.0, 1.0], [1.0, 0.0]]),
        // case::singular([[1.0, 2.0], [2.0, 4.0]]),
    )]
    fn test_svd_2x2_faer(rows: [[f64; 2]; 2]) {
        let matrix = Matrix22 { rows };
        let (u, s, vt) = matrix.svd();

        // Verify we can rebuild A from UΣVt
        assert_matrixes_ulps_eq::<2, 2, _, _>(&u.matmul(&s).matmul(&vt), &matrix);

        // Test against faer
        let faer = fill_faer(rows);
        let faersvd = faer.svd().unwrap();
        let (mut faeru, faers, mut faerv) =
            (faersvd.U().to_owned(), faersvd.S(), faersvd.V().to_owned());

        if faeru.determinant().signum() != u.det().signum() {
            faeru[(0, 1)] *= -1.0;
            faeru[(1, 1)] *= -1.0;
        }
        if faerv.determinant().signum() != vt.det().signum() {
            faerv[(0, 1)] *= -1.0;
            faerv[(1, 1)] *= -1.0;
        }

        assert_matrixes_ulps_eq::<2, 2, _, _>(&u, &faeru);
        assert_diags_ulps_eq::<2, _>(&s, &faers);
        // Note that faer returns V, not Vt
        assert_matrixes_ulps_eq::<2, 2, _, _>(&vt, &faerv.transpose());
    }

    #[rstest(
        rows,
        case::identity(Matrix22::eye().rows),
        case::mixed_sign([[1.0, -2.0], [3.0, 4.0]]),
        case::det_zero([[12.0, 2.0], [4.0, 0.0]]),
        case::large_range([[1000.0, 0.0], [0.0, 1e-4]]),
        case::jordan_block([[1.0, 1.0], [0.0, 1.0]]),
        case::full_ones(Matrix22::full(1.0).rows),
        case::shear([[1.0, 2.0], [0.0, 1.0]]),
        case::nilpotent([[0.0, 1.0], [0.0, 0.0]]),
        case::scaling([[2.0, 0.0], [0.0, 3.0]]),
        case::reflect([[0.0, -1.0], [1.0, 0.0]]), // Numerical stability
        case::negative_identity((Matrix22::eye()*-1.0).rows),
        case::anti_diagonal([[0.0, 1.0], [1.0, 0.0]]),
        case::singular([[1.0, 2.0], [2.0, 4.0]]),
    )]
    fn test_svd_2x2_nalgebra(rows: [[f64; 2]; 2]) {
        let matrix = Matrix22 { rows };
        let (u, s, vt) = matrix.svd();

        // Verify we can rebuild A from UΣVt
        assert_matrixes_ulps_eq::<2, 2, _, _>(&u.matmul(&s).matmul(&vt), &matrix);

        // Test against nalgebra
        let na = nalgebra::Matrix2::from(rows).transpose();
        let nasvd = na.svd(true, true);
        let (nau, nas, navt) = (nasvd.u.unwrap(), nasvd.singular_values, nasvd.v_t.unwrap());

        assert_matrixes_ulps_eq::<2, 2, _, _>(&u, &nau);
        assert_diags_ulps_eq::<2, _>(&s, &nas);
        assert_matrixes_ulps_eq::<2, 2, _, _>(&vt, &navt);
    }
}

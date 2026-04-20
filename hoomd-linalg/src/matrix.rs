// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::{
    fmt,
    ops::{Add, Index, IndexMut, Mul, Neg, Sub},
};

use crate::{Diagonal, GeneralMatrix, Invertible, MatMul, QuadraticForm, SquareMatrix};
// use hoomd_vector::{Cartesian, RotationMatrix};

/// A matrix with N rows and M columns, allocated on the stack.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<const N: usize, const M: usize> {
    /// The elements of the matrix
    pub rows: [[f64; M]; N],
}
/// A square, diagonal matrix with N rows and N columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiagonalMatrix<const N: usize> {
    /// The elements of the diagonal of the matrix
    pub rows: [f64; N],
}
/// Index the on-diagonal components of a diagonal matrix
/// # Examples
/// ```
/// use hoomd_linalg::{SquareMatrix, matrix::DiagonalMatrix};
/// let mat = DiagonalMatrix {
///     rows: [1.0, 2.0, 3.0],
/// };
/// assert_eq!(mat[0], 1.0);
/// assert_eq!(mat[1], 2.0);
/// assert_eq!(mat[2], 3.0);
/// ```
impl<const N: usize> Index<usize> for DiagonalMatrix<N> {
    type Output = f64;
    #[inline]
    fn index(&self, index: usize) -> &f64 {
        &self.rows[index]
    }
}

/// A 2x2 matrix, allocated on the stack.
pub type Matrix22 = Matrix<2, 2>;
/// A 3x3 matrix, allocated on the stack.
pub type Matrix33 = Matrix<3, 3>;
/// A 4x4 matrix, allocated on the stack.
pub type Matrix44 = Matrix<4, 4>;

/// Index the dense view of a diagonal matrix. Off-diagonal elements will be `0.0`.
/// # Examples
/// ```
/// use hoomd_linalg::{SquareMatrix, matrix::DiagonalMatrix};
/// let mat = DiagonalMatrix {
///     rows: [1.0, 2.0, 3.0],
/// };
/// assert_eq!(mat[(0, 0)], 1.0);
/// assert_eq!(mat[(1, 1)], 2.0);
/// assert_eq!(mat[(0, 2)], 0.0);
/// ```
impl<const N: usize> Index<(usize, usize)> for DiagonalMatrix<N> {
    type Output = f64;
    #[inline]
    fn index(&self, index: (usize, usize)) -> &f64 {
        let (i, j) = index;
        if i == j { &self.rows[i] } else { &0.0 }
    }
}
/// Index the rows and columns of a [`Matrix`]
///
/// Indices for [`Matrix`] types are zero-indexed and reflect the indexing pattern of
/// the underlying data. This results in the pattern `(row, column)`, which mirrors the
/// behavior of Numpy and similar array languages.
///
/// # Examples
/// ```
/// use hoomd_linalg::matrix::Matrix;
/// let rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let mat = Matrix { rows };
/// assert_eq!(mat[(0, 1)], rows[0][1]);
/// assert_eq!(mat[(2, 1)], 6.0);
/// assert_eq!(mat[(1, 1)], 4.0);
/// // Out-of-bounds: would panic!
/// // mat[(3, 0)];
/// ```
impl<const N: usize, const M: usize> Index<(usize, usize)> for Matrix<N, M> {
    type Output = f64;
    #[inline]
    fn index(&self, index: (usize, usize)) -> &f64 {
        let (i, j) = index;
        &self.rows[i][j]
    }
}
impl<const N: usize, const M: usize> IndexMut<(usize, usize)> for Matrix<N, M> {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f64 {
        let (i, j) = index;
        &mut self.rows[i][j]
    }
}

impl<const N: usize, const M: usize> GeneralMatrix for Matrix<N, M> {
    #[inline]
    fn zeros() -> Self {
        Self {
            rows: std::array::from_fn(|_| std::array::from_fn(|_| 0.0)),
        }
    }
    #[inline]
    fn full(val: f64) -> Self {
        Self {
            rows: std::array::from_fn(|_| std::array::from_fn(|_| val)),
        }
    }
}

impl<const N: usize> GeneralMatrix for DiagonalMatrix<N> {
    #[inline]
    fn zeros() -> Self {
        Self {
            rows: std::array::from_fn(|_| 0.0),
        }
    }
    #[inline]
    fn full(val: f64) -> Self {
        Self {
            rows: std::array::from_fn(|_| val),
        }
    }
}

impl<const N: usize> SquareMatrix for Matrix<N, N> {
    #[inline]
    fn identity() -> Self {
        Self {
            rows: std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 })),
        }
    }
}

impl<const N: usize> DiagonalMatrix<N> {
    /// Return a dense view of the diagonal matrix, with zeros on the off-diagonals.
    #[must_use]
    #[inline]
    pub fn as_dense(&self) -> Matrix<N, N> {
        Matrix::<N, N>::from_diag(&self.rows)
    }
}

impl<const N: usize, const M: usize, const K: usize> MatMul<Matrix<M, K>> for Matrix<N, M> {
    type Output = Matrix<N, K>;
    #[inline]
    fn matmul(&self, rhs: &Matrix<M, K>) -> Self::Output {
        let mut result = Self::Output::zeros();
        for n in 0..N {
            for k in 0..K {
                for m in 0..M {
                    result.rows[n][k] += self.rows[n][m] * rhs.rows[m][k];
                }
            }
        }

        result
    }
}
// impl<const N: usize> MatMul<Cartesian<M, K>> for Matrix<N, M> {
//     type Output = Matrix<N, K>;
//     #[inline]
//     fn matmul(&self, rhs: &Matrix<M, K>) -> Self::Output {
//         let mut result = Self::Output::zeros();
//         for n in 0..N {
//             for k in 0..K {
//                 for m in 0..M {
//                     result.rows[n][k] += self.rows[n][m] * rhs.rows[m][k];
//                 }
//             }
//         }

//         result
//     }
// }

impl<const N: usize, const M: usize> MatMul<DiagonalMatrix<M>> for Matrix<N, M> {
    type Output = Matrix<M, M>;
    /// Scale each column of a [`Matrix`] by the corresponding element in a [`DiagonalMatrix`].
    ///
    /// # Example
    /// ```
    /// use hoomd_linalg::{
    ///     GeneralMatrix, MatMul,
    ///     matrix::{DiagonalMatrix, Matrix22},
    /// };
    /// let diag = DiagonalMatrix { rows: [3.0, 4.0] };
    /// let mat = Matrix22::full(1.0).matmul(&diag);
    /// assert_eq!(mat[(0, 1)], 4.0);
    /// assert_eq!(mat[(1, 0)], 3.0);
    /// ```
    #[inline]
    fn matmul(&self, rhs: &DiagonalMatrix<M>) -> Self::Output {
        let mut result = Self::Output::zeros();
        for (i, row) in result.rows.iter_mut().enumerate().take(M) {
            for j in 0..M {
                row[j] = self.rows[i][j] * rhs[j];
            }
        }
        result
    }
}

impl<const N: usize, const M: usize> Matrix<N, M> {
    /// Interchange the rows and columns of matrix `A` such that `A.transpose()[(j, i)] = A[(i, j)]`
    #[inline]
    #[must_use]
    pub fn transpose(&self) -> Matrix<M, N> {
        Matrix {
            rows: std::array::from_fn(|j| std::array::from_fn(|i| self[(i, j)])),
        }
    }
}
impl<const N: usize> Matrix<N, N> {
    /// Compute the signed hypervolume of the hyperparallelepiped defined by a matrix.
    ///
    /// This implementation uses the Laplace expansion, which is optimal for small
    /// matrices but will be extremely slow for large matrixes due to its O(N!)
    /// complexity.
    ///
    /// # Examples
    ///
    /// ```
    /// use hoomd_linalg::{SquareMatrix, matrix::Matrix22};
    ///
    /// let identity = Matrix22::identity();
    /// assert_eq!(identity.det(), 1.0);
    ///
    /// let scaled = identity * 2.0;
    /// assert_eq!(scaled.det(), 2.0 * 2.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn det(&self) -> f64 {
        // Because math with const generics is not allowed in rust, we compute the indices
        // of each submatrix and recur on those noncontiguous segments of the input.
        #[inline]
        fn det_recursive_noslice<const N: usize>(
            matrix: &Matrix<N, N>,
            row: usize,
            col_indices: &[usize; N],
            minor_size: usize,
        ) -> f64 {
            if minor_size == 2 {
                let j0 = col_indices[0];
                let j1 = col_indices[1];
                return matrix.rows[row][j0] * matrix.rows[row + 1][j1]
                    - matrix.rows[row][j1] * matrix.rows[row + 1][j0];
            }

            (0..minor_size).fold(0.0, |acc, idx| {
                let minor_size = minor_size - 1;
                let mut minor_cols = [0; N];
                for j in 0..minor_size {
                    // Store the indices for the next recursion, skipping col idx
                    minor_cols[j] = col_indices[j + usize::from(j >= idx)];
                }

                let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
                acc + sign
                    * matrix.rows[row][col_indices[idx]]
                    * det_recursive_noslice(matrix, row + 1, &minor_cols, minor_size)
            })
        }
        // This would be handled by the iteration, but this simplifies the code
        match N {
            0 => return 0.0,
            1 => return self.rows[0][0],
            2 => return self.rows[0][0] * self.rows[1][1] - self.rows[1][0] * self.rows[0][1],
            _ => (),
        }

        let col_indices = std::array::from_fn(|i| i);
        det_recursive_noslice(self, 0, &col_indices, N)
    }
    /// Extract the diagonal elements from a square matrix.
    ///
    /// This method returns a `DiagonalMatrix<N>` containing the diagonal elements
    /// of the input matrix, where the element at position `(i, i)` is taken from
    /// the input matrix. All off-diagonal elements are ignored.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linalg::matrix::Matrix33;
    /// let mat = Matrix33 {
    ///     rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    /// };
    /// let diag = mat.diag();
    /// assert_eq!(diag.rows, [1.0, 5.0, 9.0]);
    /// ```
    #[must_use]
    #[inline]
    pub fn diag(&self) -> DiagonalMatrix<N> {
        DiagonalMatrix {
            rows: std::array::from_fn(|i| self.rows[i][i]),
        }
    }

    /// Compute a full `NxN` matrix from N diagonal elements, setting all others to 0.
    /// # Examples
    /// ```
    /// use hoomd_linalg::matrix::Matrix33;
    /// let mat = Matrix33::from_diag(&[1.0, 5.0, 9.0]);
    /// assert_eq!(mat.diag().rows, [1.0, 5.0, 9.0]);
    /// assert_eq!(mat[(1, 2)], 0.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn from_diag<T: Diagonal>(other: &T) -> Self {
        Matrix {
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { other[i] } else { 0.0 })
            }),
        }
    }
    // #[must_use]
    // #[inline]
    // /// Solve the quadratic form $` A.transpose().matmul(x).matmul(A) `$ for a matrix A and vector x.
    // pub fn compute_quadratic_form<T: Diagonal>(&self, vars: &T) -> f64 {
    //     let mut result = 0.0;

    //     for i in 0..N {
    //         for j in 0..N {
    //             result += vars[i] * self.rows[i][j] * vars[j];
    //         }
    //     }
    //     result
    // }
}
impl<const N: usize> QuadraticForm for Matrix<N, N> {
    #[inline]
    fn compute_quadratic_form<T: Diagonal>(&self, vars: &T) -> f64 {
        let mut result = 0.0;

        for i in 0..N {
            for j in 0..N {
                result += vars[i] * self[(i, j)] * vars[j];
            }
        }
        result
    }
}

/// Compute the elementwise scalar multiplication of a [`Matrix`]
impl<const N: usize, const M: usize> Mul<f64> for Matrix<N, M> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            rows: self.rows.map(|r| r.map(|x| x * rhs)),
        }
    }
}
/// Compute the elementwise negation of a [`Matrix`]
impl<const N: usize, const M: usize> Neg for Matrix<N, M> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            rows: self.rows.map(|r| r.map(|x| -x)),
        }
    }
}

/// Compute the elementwise scalar multiplication of a [`DiagonalMatrix`]
impl<const N: usize> Mul<f64> for DiagonalMatrix<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            rows: self.rows.map(|r| r * rhs),
        }
    }
}
/// Compute the elementwise negation of a [`DiagonalMatrix`]
impl<const N: usize> Neg for DiagonalMatrix<N> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            rows: self.rows.map(|r| -r),
        }
    }
}

impl<const N: usize, const M: usize> Add<Self> for Matrix<N, M> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| self.rows[i][j] + rhs.rows[i][j])
            }),
        }
    }
}
impl<const N: usize, const M: usize> Sub<Self> for Matrix<N, M> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| self.rows[i][j] - rhs.rows[i][j])
            }),
        }
    }
}
impl<const N: usize> Add<Self> for DiagonalMatrix<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            rows: std::array::from_fn(|i| self[i] + rhs[i]),
        }
    }
}
impl<const N: usize> Sub<Self> for DiagonalMatrix<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            rows: std::array::from_fn(|i| self[i] - rhs[i]),
        }
    }
}

impl Invertible for Matrix<2, 2> {
    #[inline]
    fn inverse(&self) -> Self {
        let inv_det = self.det().recip();
        Self {
            rows: [
                [inv_det * self.rows[1][1], inv_det * -self.rows[0][1]],
                [inv_det * -self.rows[1][0], inv_det * self.rows[0][0]],
            ],
        }
    }
}

// impl<const N: usize, const M: usize> fmt::Display for Matrix<N, M> {
//     #[inline]
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(
//             f,
//             "[{}]",
//             self.rows
//                 .map(|row| Cartesian::<M>::from(row).to_string())
//                 .into_iter()
//                 .collect::<Vec<String>>()
//                 .join("\n ")
//         )
//     }
// }

impl Matrix<2, 2> {
    /// Decompose a [`Matrix22`] into a rotation `U`, a scaling`Σ`, and a second rotation`Vt` such that `A=UΣVt`.
    ///
    /// This implementation is based on the math in 10.1109/38.486688, and ensures good
    /// (but not optimal) numerical stability. For certain pathological inputs,
    /// preconditioning the inputs could provide a benefit.
    ///
    /// We define all singular values to be positive.
    #[must_use]
    #[inline]
    pub fn svd(&self) -> (Self, DiagonalMatrix<2>, Self) {
        let a_plus_d = f64::midpoint(self[(0, 0)], self[(1, 1)]);

        let a_minus_d = (self[(0, 0)] - self[(1, 1)]) / 2.0;
        let b_plus_c = f64::midpoint(self[(0, 1)], self[(1, 0)]); // TODO sign
        let b_minus_c = (self[(1, 0)] - self[(0, 1)]) / 2.0;
        let (q, r) = (
            (a_plus_d.powi(2) + b_minus_c.powi(2)).sqrt(),
            (a_minus_d.powi(2) + b_plus_c.powi(2)).sqrt(),
        );

        let sy = q - r;
        let sign_sy = sy.signum();

        let (a1, a2) = (
            f64::atan2(b_plus_c, a_minus_d),
            f64::atan2(b_minus_c, a_plus_d),
        );

        let gamma = f64::midpoint(a1, a2);
        let beta = (a2 - a1) / 2.0;

        let (sr, cr) = beta.sin_cos();
        let (sl, cl) = gamma.sin_cos();

        let u = Matrix22 {
            rows: [[cl, -sl], [sl, cl]],
        };
        let vt = Matrix22 {
            rows: [[cr, -sr], [sr * sign_sy, cr * sign_sy]],
        };

        let singular_values = DiagonalMatrix::<2> {
            rows: [q + r, sy.abs()],
        };

        (u, singular_values, vt)
    }
}

impl Copy for Matrix<2, 2> {}
impl Copy for Matrix<3, 3> {}
impl Copy for Matrix<4, 4> {}
impl<const N: usize> Diagonal for DiagonalMatrix<N> {}
impl<const N: usize> Diagonal for [f64; N] {}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use super::*;
    use crate::matrix::{Matrix, Matrix22};
    use approxim::{assert_relative_eq, assert_ulps_eq, ulps_eq};
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
        case(Matrix::<4, 4>::identity().rows),
        case(Matrix::<5, 5>::full(3.6).diag().as_dense().rows),
        case(Matrix::<8, 8>::identity().rows),
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
        case(Matrix::<4, 4>::identity().rows, Matrix::<4, 4>::full(2.0).rows),
        case(Matrix::<5, 5>::full(3.6).diag().as_dense().rows, Matrix::<5, 5>::identity().rows),
        case(Matrix::<8, 8>::identity().rows, Matrix::<8, 8>::full(1.5).rows),
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
        case::identity(Matrix22::identity().rows),
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
        // case::negative_identity((Matrix22::identity()*-1.0).rows),
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
        case::identity(Matrix22::identity().rows),
        case::mixed_sign([[1.0, -2.0], [3.0, 4.0]]),
        case::det_zero([[12.0, 2.0], [4.0, 0.0]]),
        case::large_range([[1000.0, 0.0], [0.0, 1e-4]]),
        case::jordan_block([[1.0, 1.0], [0.0, 1.0]]),
        case::full_ones(Matrix22::full(1.0).rows),
        case::shear([[1.0, 2.0], [0.0, 1.0]]),
        case::nilpotent([[0.0, 1.0], [0.0, 0.0]]),
        case::scaling([[2.0, 0.0], [0.0, 3.0]]),
        case::reflect([[0.0, -1.0], [1.0, 0.0]]), // Numerical stability
        case::negative_identity((Matrix22::identity()*-1.0).rows),
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

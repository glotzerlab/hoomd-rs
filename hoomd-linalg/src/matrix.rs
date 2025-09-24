// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

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
    #[allow(dead_code, reason = "No use case yet.")]
    #[inline]
    #[must_use]
    fn transpose(&self) -> Matrix<M, N> {
        Matrix {
            rows: std::array::from_fn(|j| std::array::from_fn(|i| self[(i, j)])),
        }
    }

    /// Apply a function to an array elementwise, returning a new array with the same shape.
    ///
    /// # Example
    /// ```
    /// use hoomd_linalg::{GeneralMatrix, matrix::Matrix33};
    /// let m = Matrix33::full(3.0);
    /// assert_eq!(m.map_elementwise(|x| x + 2.0), m + Matrix33::full(2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn map_elementwise<F>(self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        Self {
            rows: self.rows.map(|v| v.map(&f)),
        }
    }

    /// Returns an iterator over every element in the [`Matrix`]
    /// The iterator yields all items from start to end.
    ///
    /// # Example
    /// ```
    /// use hoomd_linalg::{SquareMatrix, matrix::Matrix22};
    /// let x = Matrix22 {
    ///     rows: [[1.0, 2.0], [3.0, 4.0]],
    /// };
    /// let mut iterator = x.iter_flat();
    /// assert_eq!(iterator.next(), Some(1.0));
    /// assert_eq!(iterator.next(), Some(2.0));
    /// assert_eq!(iterator.next(), Some(3.0));
    /// assert_eq!(iterator.next(), Some(4.0));
    /// assert_eq!(iterator.next(), None);
    /// ```
    #[inline]
    pub fn iter_flat(&self) -> impl Iterator<Item = f64> + '_ {
        self.rows.iter().flat_map(|row| row.iter().copied())
    }
    /// Folds every element into an accumulator by applying an operation, returning the final result.
    ///
    /// [`fold_elementwise`] takes two arguments: an initial value, and a closure with two arguments: an ‘accumulator’, and an element. The closure returns the value that the accumulator should have for the next iteration.
    ///
    /// The initial value is the value the accumulator will have on the first call.
    /// After applying this closure to every element of the flattened iterator, [`fold_elementwise`] returns the accumulator.
    ///
    /// # Example
    /// ```
    /// use hoomd_linalg::{GeneralMatrix, matrix::Matrix22};
    /// let m = Matrix22::full(3.0);
    /// // Sum the elements of a matrix
    /// assert_eq!(m.fold_elementwise(0.0, |acc, x| acc + x), 3.0 * 4.0);
    /// ```
    #[inline]
    pub fn fold_elementwise<B, F>(self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, f64) -> B,
    {
        let mut accum = init;
        for x in self.iter_flat() {
            accum = f(accum, x);
        }
        accum
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
        self.map_elementwise(|x| x * rhs)
    }
}
/// Compute the elementwise negation of a [`Matrix`]
impl<const N: usize, const M: usize> Neg for Matrix<N, M> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        self.map_elementwise(f64::neg)
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
//                 .map(|row| format!("[{}]", row.map(f64::to_string)))
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
    /// preconditioning the matrix could provide a benefit in numerical stability.
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

impl Matrix<3, 3> {
    /// Compute the decomposition of a [`Matrix33`] into a quaternion rotation and a an associated eigenvalue.
    ///
    /// This method is an implementation of the Quaternion Characteristic Polynomial
    /// (QCP) algorithm proposed by [Douglas Theobald et. al.](). It allows for for
    /// extremely rapid alignment of coordinates, and is commonly used for molecular
    /// superposition and point-set registration.
    ///
    /// # Theory
    /// The singular value decomposition of a matrix $`M`$ is defined as
    /// $`M=U Σ V^\top`$, where U and V are rotoreflection matrices and Σ is a diagonal
    /// matrix consisting of the singular values of $`M`$. If $`M`$ is an inner product
    /// between two sets of coordinates $`M=A^\topB`$, $`U V^\top`$ is the rotation
    /// that optimally aligns the two sets of points $`A`$ and $`B`$, and $`\tr(Σ)`$ is
    /// the root mean-squared deviation between those sets of points.
    ///
    /// This decomposition of $`M`$ into 3D rotation matrices can be reinterpreted as
    /// a decomposition of a symmetric 4 x 4 matrix $`K`$. Under this construction, the
    /// largest eigenvalue of $`K`$ is $`\tr(Σ)`$ (the point-set RMSD) and the largest
    /// eigenvector of $`K`$ is a quaternion equivalent to the rotation $`U V^\top`$.
    /// As one only needs to compute a single eigenvector and eigenvalue, this approach
    /// is much faster than computing a full 3x3 singular value decomposition.
    #[must_use]
    #[inline]
    pub fn quaternion_decomposition(&self) -> (f64, [f64; 4]) {
        // let [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = self.rows;

        // let m_sq = self.map_elementwise(|x| x * x);
        // let syz_szy_m_syy_szz_2 = 2.0 * (syz * szy - syy * szz);
        // let syysq_p_szzsq_m_sxxsq_syzsq_p_szy_sq = syy_sq + szz_sq - sxx_sq + syz_sq + szy_sq;

        // let sum_m_squared =

        (0.0, [0.0; 4])
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
    fn fill_faer_column<const N: usize>(c: [f64; N]) -> Mat<f64> {
        let mut faer_matrix = Mat::<f64>::zeros(N, 1);
        for (i, el) in c.iter().enumerate() {
            *faer_matrix.get_mut(i, 0) = *el;
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

    #[test]
    fn test_matrix_multiply_diagonal_2x2() {
        let a_rows = [[1.0, 2.0], [3.0, 4.0]];
        let b_diag = [5.0, 6.0];
        let a = Matrix::<2, 2> { rows: a_rows };
        let b = DiagonalMatrix::<2> { rows: b_diag };

        let faer_a = fill_faer(a_rows);
        let faer_b_dense = fill_faer(Matrix::<2, 2>::from_diag(&b).rows);

        let custom_prod = a.matmul(&b);
        let faer_prod = faer_a * faer_b_dense;

        assert_matrixes_ulps_eq::<2, 2, _, _>(&custom_prod, &faer_prod);
    }

    #[test]
    fn test_matrix_multiply_diagonal_3x2() {
        let a_rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let b_diag = [2.0, 3.0];
        let a = Matrix::<3, 2> { rows: a_rows };
        let b = DiagonalMatrix::<2> { rows: b_diag };

        let faer_a = fill_faer(a_rows);
        let faer_b_dense = fill_faer(Matrix::<2, 2>::from_diag(&b).rows);

        let custom_prod = a.matmul(&b);
        let faer_prod = faer_a * faer_b_dense;

        assert_matrixes_ulps_eq::<2, 2, _, _>(&custom_prod, &faer_prod);
    }

    #[test]
    fn test_transpose_2x2() {
        let rows = [[1.0, -2.0], [3.0, 4.0]];
        let matrix = Matrix::<2, 2> { rows };
        let faer_matrix = fill_faer(rows);
        let custom_transpose = matrix.transpose();
        let faer_transpose = faer_matrix.transpose();
        assert_matrixes_ulps_eq::<2, 2, _, _>(&custom_transpose, &faer_transpose);
    }

    #[test]
    fn test_transpose_2x3() {
        let rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let matrix = Matrix::<2, 3> { rows };
        let faer_matrix = fill_faer(rows);
        let custom_transpose = matrix.transpose();
        let faer_transpose = faer_matrix.transpose();
        assert_matrixes_ulps_eq::<3, 2, _, _>(&custom_transpose, &faer_transpose);
    }

    #[test]
    fn test_transpose_3x2() {
        let rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let matrix = Matrix::<3, 2> { rows };
        let faer_matrix = fill_faer(rows);
        let custom_transpose = matrix.transpose();
        let faer_transpose = faer_matrix.transpose();
        assert_matrixes_ulps_eq::<2, 3, _, _>(&custom_transpose, &faer_transpose);
    }

    #[test]
    fn test_transpose_1x1() {
        let rows = [[-9.0]];
        let matrix = Matrix::<1, 1> { rows };
        assert_matrixes_ulps_eq::<1, 1, _, _>(&matrix.transpose(), &matrix);
    }

    #[test]
    fn test_matrix_add_2x2() {
        let a_rows = [[1.0, 2.0], [3.0, 4.0]];
        let b_rows = [[5.0, 6.0], [7.0, 8.0]];
        let a = Matrix::<2, 2> { rows: a_rows };
        let b = Matrix::<2, 2> { rows: b_rows };
        let faer_a = fill_faer(a_rows);
        let faer_b = fill_faer(b_rows);

        let custom_sum = a + b;
        let faer_sum = faer_a + faer_b;

        assert_matrixes_ulps_eq::<2, 2, _, _>(&custom_sum, &faer_sum);
    }

    #[test]
    fn test_matrix_add_2x3() {
        let a_rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b_rows = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
        let a = Matrix::<2, 3> { rows: a_rows };
        let b = Matrix::<2, 3> { rows: b_rows };
        let faer_a = fill_faer(a_rows);
        let faer_b = fill_faer(b_rows);

        let custom_sum = a + b;
        let faer_sum = faer_a + faer_b;

        assert_matrixes_ulps_eq::<2, 3, _, _>(&custom_sum, &faer_sum);
    }

    #[test]
    fn test_matrix_sub_2x2() {
        let a_rows = [[1.0, 2.0], [3.0, 4.0]];
        let b_rows = [[5.0, 6.0], [7.0, 8.0]];
        let a = Matrix::<2, 2> { rows: a_rows };
        let b = Matrix::<2, 2> { rows: b_rows };
        let faer_a = fill_faer(a_rows);
        let faer_b = fill_faer(b_rows);

        let custom_diff = a - b;
        let faer_diff = faer_a - faer_b;

        assert_matrixes_ulps_eq::<2, 2, _, _>(&custom_diff, &faer_diff);
    }

    #[test]
    fn test_matrix_sub_2x3() {
        let a_rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b_rows = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
        let a = Matrix::<2, 3> { rows: a_rows };
        let b = Matrix::<2, 3> { rows: b_rows };
        let faer_a = fill_faer(a_rows);
        let faer_b = fill_faer(b_rows);

        let custom_diff = a - b;
        let faer_diff = faer_a - faer_b;

        assert_matrixes_ulps_eq::<2, 3, _, _>(&custom_diff, &faer_diff);
    }

    #[rstest(
        rows,
        case([[1.0, -2.0], [3.0, 4.0]]),
        case([[0.0, 0.0], [0.0, 0.0]])
    )]
    fn test_matrix_neg_2x2(rows: [[f64; 2]; 2]) {
        let matrix = Matrix::<2, 2> { rows };
        let faer_matrix = fill_faer(rows);

        let custom_neg = -matrix;
        let faer_neg = -faer_matrix;

        assert_matrixes_ulps_eq::<2, 2, _, _>(&custom_neg, &faer_neg);
    }

    #[rstest]
    #[case([[1.0, 2.0], [3.0, 4.0]], 5.0)]
    #[case([[1.0, 2.0], [3.0, 4.0]], -1.0)]
    #[case([[1.0, 2.0], [3.0, 4.0]], 0.0)]
    fn test_matrix_scalar_mul_2x2(#[case] rows: [[f64; 2]; 2], #[case] scalar: f64) {
        let matrix = Matrix::<2, 2> { rows };
        let faer_matrix = fill_faer(rows);

        let custom_prod = matrix * scalar;
        let faer_prod = faer_matrix * scalar;

        assert_matrixes_ulps_eq::<2, 2, _, _>(&custom_prod, &faer_prod);
    }

    #[test]
    fn test_diagonal_matrix_add_n2() {
        let a_diag = [1.0, 2.0];
        let b_diag = [3.0, 4.0];
        let a = DiagonalMatrix::<2> { rows: a_diag };
        let b = DiagonalMatrix::<2> { rows: b_diag };
        let expected: Vec<f64> = a_diag
            .iter()
            .zip(b_diag.iter())
            .map(|(x, y)| x + y)
            .collect();
        let custom_sum = a + b;
        assert_diags_ulps_eq::<2, _>(&custom_sum, &expected);
    }

    #[test]
    fn test_diagonal_matrix_add_n3() {
        let a_diag = [1.0, 2.0, 3.0];
        let b_diag = [4.0, 5.0, 6.0];
        let a = DiagonalMatrix::<3> { rows: a_diag };
        let b = DiagonalMatrix::<3> { rows: b_diag };
        let expected: Vec<f64> = a_diag
            .iter()
            .zip(b_diag.iter())
            .map(|(x, y)| x + y)
            .collect();
        let custom_sum = a + b;
        assert_diags_ulps_eq::<3, _>(&custom_sum, &expected);
    }

    #[test]
    fn test_diagonal_matrix_sub_n2() {
        let a_diag = [1.0, 2.0];
        let b_diag = [3.0, 4.0];
        let a = DiagonalMatrix::<2> { rows: a_diag };
        let b = DiagonalMatrix::<2> { rows: b_diag };
        let expected: Vec<f64> = a_diag
            .iter()
            .zip(b_diag.iter())
            .map(|(x, y)| x - y)
            .collect();
        let custom_sub = a - b;
        assert_diags_ulps_eq::<2, _>(&custom_sub, &expected);
    }

    #[test]
    fn test_diagonal_matrix_sub_n3() {
        let a_diag = [1.0, 2.0, 3.0];
        let b_diag = [4.0, 5.0, 6.0];
        let a = DiagonalMatrix::<3> { rows: a_diag };
        let b = DiagonalMatrix::<3> { rows: b_diag };
        let expected: Vec<f64> = a_diag
            .iter()
            .zip(b_diag.iter())
            .map(|(x, y)| x - y)
            .collect();
        let custom_sub = a - b;
        assert_diags_ulps_eq::<3, _>(&custom_sub, &expected);
    }

    #[test]
    fn test_diagonal_matrix_neg_n2() {
        let diag = [1.0, -2.0];
        let matrix = DiagonalMatrix::<2> { rows: diag };
        let expected: Vec<f64> = diag.iter().map(|x| -x).collect();
        let custom_neg = -matrix;
        assert_diags_ulps_eq::<2, _>(&custom_neg, &expected);
    }

    #[test]
    fn test_diagonal_matrix_neg_n3() {
        let diag = [1.0, -2.0, 0.0];
        let matrix = DiagonalMatrix::<3> { rows: diag };
        let expected: Vec<f64> = diag.iter().map(|x| -x).collect();
        let custom_neg = -matrix;
        assert_diags_ulps_eq::<3, _>(&custom_neg, &expected);
    }

    #[rstest]
    #[case([1.0, 2.0], 5.0)]
    #[case([1.0, 2.0], -1.0)]
    #[case([1.0, 2.0], 0.0)]
    fn test_diagonal_matrix_scalar_mul_n2(#[case] diag: [f64; 2], #[case] scalar: f64) {
        let matrix = DiagonalMatrix::<2> { rows: diag };
        let expected: Vec<f64> = diag.iter().map(|x| x * scalar).collect();
        let custom_mul = matrix * scalar;
        assert_diags_ulps_eq::<2, _>(&custom_mul, &expected);
    }

    #[test]
    fn test_indexing() {
        // Matrix
        let mat = Matrix::<2, 3> {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        };
        assert_eq!(mat[(0, 2)], 3.0);
        assert_eq!(mat[(1, 1)], 5.0);

        // DiagonalMatrix
        let diag_mat = DiagonalMatrix::<3> {
            rows: [1.0, 2.0, 3.0],
        };
        assert_eq!(diag_mat[1], 2.0); // 1D indexing
        assert_eq!(diag_mat[(2, 2)], 3.0); // 2D on-diagonal
        assert_eq!(diag_mat[(0, 1)], 0.0); // 2D off-diagonal
    }

    #[test]
    fn test_mut_indexing() {
        let mut mat = Matrix::<2, 2>::zeros();
        mat[(0, 1)] = 99.0;
        mat[(1, 0)] = -5.5;
        assert_eq!(mat[(0, 1)], 99.0);
        assert_eq!(mat[(1, 0)], -5.5);
        assert_eq!(mat[(1, 1)], 0.0);
    }

    #[test]
    fn test_general_matrix_methods() {
        // Matrix
        let zeros = Matrix::<2, 3>::zeros();
        let full = Matrix::<2, 3>::full(7.5);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(zeros[(i, j)], 0.0);
                assert_eq!(full[(i, j)], 7.5);
            }
        }

        // DiagonalMatrix
        let diag_zeros = DiagonalMatrix::<4>::zeros();
        let diag_full = DiagonalMatrix::<4>::full(-3.0);
        for i in 0..4 {
            assert_eq!(diag_zeros[i], 0.0);
            assert_eq!(diag_full[i], -3.0);
        }
    }

    #[test]
    fn test_square_matrix_methods() {
        let identity = Matrix::<3, 3>::identity();
        let expected = Matrix::<3, 3> {
            rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        assert_matrixes_ulps_eq::<3, 3, _, _>(&identity, &expected);

        let diag_mat = DiagonalMatrix::<3> {
            rows: [10.0, 20.0, 30.0],
        };
        let dense = diag_mat.as_dense();
        let expected_dense = Matrix::<3, 3>::from_diag(&[10.0, 20.0, 30.0]);
        assert_matrixes_ulps_eq::<3, 3, _, _>(&dense, &expected_dense);
    }

    #[test]
    fn test_diag_conversions() {
        let mat = Matrix::<3, 3> {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        };
        let diag = mat.diag();
        let expected_diag = DiagonalMatrix {
            rows: [1.0, 5.0, 9.0],
        };
        assert_diags_ulps_eq::<3, _>(&diag, &expected_diag.rows);

        let from_diag = Matrix::<3, 3>::from_diag(&diag.rows);
        let expected_from_diag = Matrix {
            rows: [[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 9.0]],
        };
        assert_matrixes_ulps_eq::<3, 3, _, _>(&from_diag, &expected_from_diag);
    }

    #[rstest(
        rows, vars,
        case(
            [[1.0, 2.0], [3.0, 4.0]],
            [0.5, 1.5]
        ),
        case(
            [[2.0, 0.0, 1.0], [3.0, 0.0, 0.0], [5.0, 1.0, 1.0]],
            [1.0, 2.0, 3.0]
        ),
        case(
            [[-33.0, 2.0, 0.0, 1.0], [3.0, -45.0, 0.0, 0.0], [5.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            [1.0, 2.0, 3.0, 4.0]
        ),
    )]
    fn test_quadratic_form<const N: usize>(rows: [[f64; N]; N], vars: [f64; N]) {
        let matrix = Matrix { rows };
        let result = matrix.compute_quadratic_form(&vars);
        assert_relative_eq!(
            result,
            (fill_faer_column(vars).transpose() * fill_faer(rows) * fill_faer_column(vars))[(0, 0)],
            max_relative = 1e-14
        );
    }

    #[rstest(
        rows,
        case([[1.0, -2.0], [3.0, 4.0]]),
        case([[10.0, 0.0], [0.0, 0.1]]),
        case([[1.0, 1.0], [0.0, 1.0]]),
    )]
    fn test_inverse_2x2(rows: [[f64; 2]; 2]) {
        let matrix = Matrix22 { rows };
        let inv_matrix = matrix.inverse();
        let product = matrix.matmul(&inv_matrix);
        let identity = Matrix22::identity();

        assert_matrixes_ulps_eq::<2, 2, _, _>(&product, &identity);
    }
}

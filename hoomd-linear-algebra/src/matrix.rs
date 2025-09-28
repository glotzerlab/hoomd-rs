// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/// ``std::ops`` implementations for [`Matrix`]
mod ops;

pub use crate::diagonal::DiagonalMatrix;

use std::fmt;

/// A lightweight representation of a diagonal matrix.
use crate::{Diagonal, GeneralMatrix, Invertible, MatMul, QuadraticForm, SquareMatrix};

/// A matrix with N rows and M columns, allocated on the stack.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<const N: usize, const M: usize> {
    /// The elements of the matrix
    pub rows: [[f64; M]; N],
}
/// A 2x2 matrix, allocated on the stack.
pub type Matrix22 = Matrix<2, 2>;
/// A 3x3 matrix, allocated on the stack.
pub type Matrix33 = Matrix<3, 3>;
/// A 4x4 matrix, allocated on the stack.
pub type Matrix44 = Matrix<4, 4>;

impl<const N: usize, const M: usize> GeneralMatrix for Matrix<N, M> {
    /// Fill a matrix with zeros.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix22};
    /// let m = Matrix22::zeros();
    /// assert_eq!(m.rows, [[0.0, 0.0], [0.0, 0.0]]);
    /// ```
    #[inline]
    fn zeros() -> Self {
        Self {
            rows: std::array::from_fn(|_| std::array::from_fn(|_| 0.0)),
        }
    }
    /// Fill a matrix with some scalar value.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix22};
    /// let m = Matrix22::full(5.0);
    /// assert_eq!(m.rows, [[5.0, 5.0], [5.0, 5.0]]);
    /// ```
    #[inline]
    fn full(val: f64) -> Self {
        Self {
            rows: std::array::from_fn(|_| std::array::from_fn(|_| val)),
        }
    }
}

impl<const N: usize> SquareMatrix for Matrix<N, N> {
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{SquareMatrix, matrix::Matrix22};
    /// let m = Matrix22::identity();
    /// assert_eq!(m.rows, [[1.0, 0.0], [0.0, 1.0]]);
    /// ```
    #[inline]
    fn identity() -> Self {
        Self {
            rows: std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 })),
        }
    }
}

impl<const N: usize, const M: usize, const K: usize> MatMul<Matrix<M, K>> for Matrix<N, M> {
    type Output = Matrix<N, K>;
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{
    ///     MatMul,
    ///     matrix::{Matrix, Matrix22},
    /// };
    /// let a = Matrix22 {
    ///     rows: [[1.0, 2.0], [3.0, 4.0]],
    /// };
    /// let b = Matrix22 {
    ///     rows: [[5.0, 6.0], [7.0, 8.0]],
    /// };
    /// let c = a.matmul(&b);
    /// assert_eq!(c.rows, [[19.0, 22.0], [43.0, 50.0]]);
    /// ```
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

impl<const N: usize, const M: usize> MatMul<DiagonalMatrix<M>> for Matrix<N, M> {
    type Output = Matrix<N, M>;
    /// Multiply a matrix by a diagonal matrix RHS.
    ///
    /// This is equivalent to scaling each column of a [`Matrix`] by the corresponding
    /// element in a [`DiagonalMatrix`].
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{
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
        for (i, row) in result.rows.iter_mut().enumerate().take(N) {
            for j in 0..M {
                row[j] = self.rows[i][j] * rhs[j];
            }
        }
        result
    }
}

impl<const N: usize, const M: usize> Matrix<N, M> {
    /// Interchange the rows and columns of matrix `A` such that `A.transpose()[(j, i)] = A[(i, j)]`
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::matrix::Matrix;
    ///
    /// let m: Matrix<2, 3> = Matrix {
    ///     rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    /// };
    /// let m_t = m.transpose();
    /// assert_eq!(m_t.rows, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    /// ```
    #[allow(dead_code, reason = "No use case yet.")]
    #[inline]
    #[must_use]
    pub fn transpose(&self) -> Matrix<M, N> {
        Matrix {
            rows: std::array::from_fn(|j| std::array::from_fn(|i| self[(i, j)])),
        }
    }

    /// Apply a function to an [`Matrix`] by rows, returning a new matrix with the same shape.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix22};
    /// let m = Matrix22::full(3.0);
    /// assert_eq!(
    ///     m.map_rows(|v| [v[0] + 2.0, v[1]]),
    ///     Matrix22 {
    ///         rows: [[5.0, 3.0], [5.0, 3.0]]
    ///     }
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn map_rows<F>(self, f: F) -> Self
    where
        F: FnMut([f64; M]) -> [f64; M],
    {
        Self {
            rows: self.rows.map(f),
        }
    }

    /// Apply a function to an [`Matrix`] by columns, returning a new matrix with the same shape.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix22};
    /// let m = Matrix22::full(3.0);
    /// assert_eq!(
    ///     m.map_cols(|v| [v[0] + 2.0, v[1]]),
    ///     Matrix22 {
    ///         rows: [[5.0, 5.0], [3.0, 3.0]]
    ///     }
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn map_cols<F>(self, f: F) -> Self
    where
        F: FnMut([f64; N]) -> [f64; N],
    {
        self.clone().transpose().map_rows(f).transpose()
    }
    /// Apply a function to a [`Matrix`] elementwise, returning a new matrix with the same shape.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix33};
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
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{SquareMatrix, matrix::Matrix22};
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

    /// Returns an iterator over mutable references to every element in the [`Matrix`]
    /// The iterator yields all items from start to end.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{SquareMatrix, matrix::Matrix22};
    /// let mut x = Matrix22 {
    ///     rows: [[1.0, 2.0], [3.0, 4.0]],
    /// };
    /// let x_copy = x.clone();
    /// let mut iterator = x.iter_flat_mut();
    /// iterator.for_each(|x| *x *= 2.0);
    /// assert_eq!(x, x_copy * 2.0);
    /// ```
    #[inline]
    pub fn iter_flat_mut(&mut self) -> impl Iterator<Item = &mut f64> + '_ {
        self.rows.iter_mut().flat_map(|row| row.iter_mut())
    }

    /// Returns an iterator over every element in the [`Matrix`]
    /// The iterator yields all items from start to end.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{SquareMatrix, matrix::Matrix22};
    /// let x = Matrix22 {
    ///     rows: [[1.0, 2.0], [3.0, 4.0]],
    /// };
    /// let mut iterator = x.iter();
    /// assert_eq!(iterator.next(), Some([1.0, 2.0]));
    /// assert_eq!(iterator.next(), Some([3.0, 4.0]));
    /// assert_eq!(iterator.next(), None);
    /// ```
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = [f64; M]> + '_ {
        self.rows.iter().copied()
    }
    /// Folds every element into an accumulator by applying an operation, returning the final result.
    ///
    /// `fold_elementwise` takes two arguments: an initial value, and a closure with two arguments: an ‘accumulator’, and an element. The closure returns the value that the accumulator should have for the next iteration.
    ///
    /// The initial value is the value the accumulator will have on the first call.
    /// After applying this closure to every element of the flattened iterator, `fold_elementwise` returns the accumulator.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix22};
    /// let m = Matrix22::full(3.0);
    /// // Sum the elements of a matrix
    /// assert_eq!(m.fold_elementwise(0.0, |acc, x| acc + x), 3.0 * 4.0);
    /// ```
    #[inline]
    pub fn fold_elementwise<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, f64) -> B,
    {
        let mut accum = init;
        for x in self.iter_flat() {
            accum = f(accum, x);
        }
        accum
    }

    /// Folds every row into an accumulator by applying an operation, returning the final result.
    ///
    /// `fold` takes two arguments: an initial value, and a closure with two arguments: an ‘accumulator’, and an element. The closure returns the value that the accumulator should have for the next iteration.
    ///
    /// The initial value is the value the accumulator will have on the first call.
    /// After applying this closure to every element of the flattened iterator, `fold` returns the accumulator.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix22};
    /// let m = Matrix22 {
    ///     rows: [[1.0, 2.0], [3.0, 4.0]],
    /// };
    /// // Average the columns of a matrix
    /// let n_rows = m.n_rows() as f64;
    /// assert_eq!(
    ///     m.fold([0.0; 2], |acc, x| [acc[0] + x[0], acc[1] + x[1]])
    ///         .map(|x| x / n_rows),
    ///     [(1.0 + 3.0) / 2.0, (2.0 + 4.0) / 2.0]
    /// );
    /// ```
    #[inline]
    pub fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, [f64; M]) -> B,
    {
        let mut accum = init;
        for x in self.iter() {
            accum = f(accum, x);
        }
        accum
    }

    /// Get the number of rows in the [`Matrix`].
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::matrix::Matrix;
    ///
    /// let m: Matrix<2, 3> = Matrix {
    ///     rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    /// };
    /// assert_eq!(m.n_rows(), 2);
    /// ```
    #[must_use]
    #[inline]
    pub const fn n_rows(&self) -> usize {
        N
    }
    /// Get the number of columns in the [`Matrix`].
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::matrix::Matrix;
    ///
    /// let m: Matrix<2, 3> = Matrix {
    ///     rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    /// };
    /// assert_eq!(m.n_cols(), 3);
    /// ```
    #[must_use]
    #[inline]
    pub const fn n_cols(&self) -> usize {
        M
    }

    /// Get the shape of the [`Matrix`] `(n_rows, n_cols)`.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::matrix::Matrix;
    ///
    /// let m: Matrix<2, 3> = Matrix {
    ///     rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    /// };
    /// assert_eq!(m.shape(), (2, 3));
    /// ```
    #[must_use]
    #[inline]
    pub const fn shape(&self) -> (usize, usize) {
        (self.n_rows(), self.n_cols())
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
    /// use hoomd_linear_algebra::{SquareMatrix, matrix::Matrix22};
    ///
    /// let identity = Matrix22::identity();
    /// assert_eq!(identity.determinant(), 1.0);
    ///
    /// let scaled = identity * 2.0;
    /// assert_eq!(scaled.determinant(), 2.0 * 2.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn determinant(&self) -> f64 {
        // Compute the determinant of a 2x2 minor.
        #[inline]
        fn det2(a: f64, b: f64, c: f64, d: f64) -> f64 {
            a * d - b * c
        }
        // Because math with const generics is not allowed in rust, we compute the indices
        // of each submatrix and recur on those noncontiguous segments of the input.
        #[inline]
        fn det_recursive_noslice<const N: usize>(
            matrix: &Matrix<N, N>,
            row: usize,
            col_indices: [usize; N],
            minor_size: usize,
        ) -> f64 {
            // If we recurr any lower than 4x4 minors, performance drops dramatically
            if minor_size == 4 {
                let r = matrix.rows;
                let c = col_indices;

                // Map recursive indices to direct matrix indices
                let (i0, i1, i2, i3) = (row, row + 1, row + 2, row + 3);
                let [j0, j1, j2, j3] = c[..4] else {
                    unreachable!() // N >= 4 if we reach this point
                };

                let m0 = det2(r[i2][j2], r[i2][j3], r[i3][j2], r[i3][j3]);
                let m1 = det2(r[i2][j1], r[i2][j3], r[i3][j1], r[i3][j3]);
                let m2 = det2(r[i2][j1], r[i2][j2], r[i3][j1], r[i3][j2]);
                let m3 = det2(r[i2][j0], r[i2][j3], r[i3][j0], r[i3][j3]);
                let m4 = det2(r[i2][j0], r[i2][j2], r[i3][j0], r[i3][j2]);
                let m5 = det2(r[i2][j0], r[i2][j1], r[i3][j0], r[i3][j1]);

                return r[i0][j0] * (r[i1][j1] * m0 - r[i1][j2] * m1 + r[i1][j3] * m2)
                    - r[i0][j1] * (r[i1][j0] * m0 - r[i1][j2] * m3 + r[i1][j3] * m4)
                    + r[i0][j2] * (r[i1][j0] * m1 - r[i1][j1] * m3 + r[i1][j3] * m5)
                    - r[i0][j3] * (r[i1][j0] * m2 - r[i1][j1] * m4 + r[i1][j2] * m5);
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
                    * det_recursive_noslice(matrix, row + 1, minor_cols, minor_size)
            })
        }
        // Early exit for small matrices to ensure we get the optimal code.
        match N {
            0 => return 0.0,
            1 => return self[(0, 0)],
            2 => return det2(self[(0, 0)], self[(1, 0)], self[(0, 1)], self[(1, 1)]),
            3 => {
                return self[(0, 0)] * det2(self[(1, 1)], self[(1, 2)], self[(2, 1)], self[(2, 2)])
                    - self[(0, 1)] * det2(self[(1, 0)], self[(1, 2)], self[(2, 0)], self[(2, 2)])
                    + self[(0, 2)] * det2(self[(1, 0)], self[(1, 1)], self[(2, 0)], self[(2, 1)]);
            }
            _ => (),
        }

        let col_indices = std::array::from_fn(|i| i);
        det_recursive_noslice(self, 0, col_indices, N)
    }
    /// Compute the sum of diagonal elements of a square matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use hoomd_linear_algebra::{SquareMatrix, matrix::Matrix22};
    ///
    /// let identity = Matrix22::identity();
    /// assert_eq!(identity.trace(), 2.0);
    ///
    /// let scaled = identity * 3.0;
    /// assert_eq!(scaled.trace(), 3.0 + 3.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn trace(&self) -> f64 {
        std::array::from_fn::<_, N, _>(|i| self[(i, i)])
            .iter()
            .sum()
    }

    /// Compute a matrix to an integer power, equivalent to $`\prod_{i=1}^n A`$.
    ///
    /// # Examples
    ///
    /// ```
    /// use hoomd_linear_algebra::{GeneralMatrix, MatMul, matrix::Matrix22};
    ///
    /// let matrix = Matrix22::full(2.0);
    ///
    /// // powi(2) is equivalent to x.matmul(&x)
    /// assert_eq!(matrix.powi(2), matrix.matmul(&matrix));
    ///
    /// // Standard power rules are respected.
    /// assert_eq!(matrix.powi(2).powi(2), matrix.powi(4));
    /// ```
    #[must_use]
    #[inline]
    pub fn powi(&self, n: i32) -> Self {
        (0..n).fold(Self::identity(), |acc, _| acc.matmul(self))
    }

    /// Extract the diagonal elements from a square matrix.
    ///
    /// This method returns a `DiagonalMatrix<N>` containing the diagonal elements
    /// of the input matrix, where the element at position `(i, i)` is taken from
    /// the input matrix. All off-diagonal elements are ignored.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::matrix::Matrix33;
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
    /// use hoomd_linear_algebra::matrix::Matrix33;
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

impl Invertible for Matrix<2, 2> {
    /// Compute the inverse of a matrix. Will be `None` if the matrix is not invertible.
    ///
    /// This implementation uses a closed form solution for the matrix inverse.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{Invertible, matrix::Matrix22};
    /// let m = Matrix22 {
    ///     rows: [[1.0, 2.0], [3.0, 4.0]],
    /// };
    /// let m_inv = m.inverse().unwrap();
    /// assert_eq!(m_inv.rows, [[-2.0, 1.0], [1.5, -0.5]]);
    /// ```
    #[inline]
    fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == 0.0 {
            None
        } else {
            let inv_det = det.recip();
            Some(Self {
                rows: [
                    [inv_det * self.rows[1][1], inv_det * -self.rows[0][1]],
                    [inv_det * -self.rows[1][0], inv_det * self.rows[0][0]],
                ],
            })
        }
    }
}

impl Invertible for Matrix<3, 3> {
    /// Compute the inverse of a matrix. Will be `None` if the matrix is not invertible.
    ///
    /// This implementation uses a closed form solution for the matrix inverse based on
    /// the cross product of rows.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{Invertible, SquareMatrix, matrix::Matrix33};
    /// let m = Matrix33::identity() * 5.0;
    /// let m_inv = m.inverse().unwrap();
    /// // The inverse of a diagonal matrix is the reciprocal of each element.
    /// assert_eq!(m_inv, Matrix33::from_diag(&[1.0 / 5.0; 3]));
    /// ```
    #[inline]
    fn inverse(&self) -> Option<Self> {
        #[inline]
        fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
            [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ]
        }
        let [x0, x1, x2] = self.rows;
        let det = self.determinant();
        if det == 0.0 {
            return None;
        }
        let rows = [cross(x1, x2), cross(x2, x0), cross(x0, x1)];
        Some(det.recip() * Self { rows }.transpose())
    }
}
impl Invertible for Matrix<4, 4> {
    /// Compute the inverse of a matrix. Will be `None` if the matrix is not invertible.
    ///
    /// This implementation uses a closed form solution for the matrix inverse based on
    /// the Cayley–Hamilton method.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{
    ///     Invertible, MatMul, SquareMatrix, matrix::Matrix44,
    /// };
    /// let m = Matrix44::identity();
    /// let m_inv = m.inverse().unwrap();
    /// assert_eq!(m_inv.rows, m.rows);
    /// ```
    #[inline]
    fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == 0.0 {
            return None;
        }
        // Compute components of Cayley–Hamilton factorization
        let tr_a = self.trace();
        let a_sq = self.powi(2);
        let tr_a_sq = a_sq.trace();
        let a_cb = a_sq.matmul(self);
        let tr_a_cb = a_cb.trace();
        let left =
            (1.0 / 6.0) * (tr_a.powi(3) - 3.0 * tr_a * tr_a_sq + 2.0 * tr_a_cb) * Self::identity();
        let center = (1.0 / 2.0) * *self * (tr_a.powi(2) - tr_a_sq);
        Some(det.recip() * (left - center + a_sq * tr_a - a_cb))
    }
}

impl<const N: usize, const M: usize> fmt::Display for Matrix<N, M> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&format!(
            "[{}]",
            self.iter()
                .map(|row| {
                    format!(
                        "[{}]",
                        row.iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
                .collect::<Vec<_>>()
                .join(",\n ")
        ))
    }
}
impl Matrix<2, 2> {
    /// Decompose a [`Matrix22`] into a rotation, a scaling, and a second rotation.
    ///
    /// ```math
    /// A = U Σ V^\top;
    /// ```
    /// This implementation is based on the math in doi:10.1109/38.486688, and
    /// ensures good (but not optimal) numerical stability. For certain
    /// pathological inputs, preconditioning the matrix could provide a benefit
    /// in numerical stability.
    ///
    /// `svd` sets all singular values to be positive.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{
    ///     MatMul,
    ///     matrix::{DiagonalMatrix, Matrix22},
    /// };
    /// let m = Matrix22 {
    ///     rows: [[1.0, 2.0], [3.0, 4.0]],
    /// };
    /// let (u, s, vt) = m.svd();
    /// let m_recon = u.matmul(&s.as_dense()).matmul(&vt);
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         assert!((m.rows[i][j] - m_recon.rows[i][j]).abs() < 1e-9);
    ///     }
    /// }
    /// ```
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

/// Macro to generate impls for a given row size `N` and multiple column sizes `M`.
macro_rules! impl_copy_for_m {
    ($N:literal, $($M:literal),+) => { $(#[doc(hidden)]impl Copy for Matrix<$N, $M> {})+ };
}
/// Implement Copy for matrices of an input size `N`, `M`
macro_rules! impl_copy_for_n_m {
    ($($N:literal),+) => { $(impl_copy_for_m!($N, 1, 2, 3, 4);)+ };
}

impl_copy_for_n_m!(1, 2, 3, 4);

impl<const N: usize> Diagonal for [f64; N] {}

#[cfg(test)]
mod tests {
    use std::{
        fmt::Debug,
        ops::{Index, Mul},
    };

    use super::*;
    use crate::matrix::{Matrix, Matrix22, Matrix33, Matrix44};
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
        m1: &impl std::ops::Index<usize, Output = f64>,
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
        case([[2.0, 0.0, 1.0], [3.0, 9.0, 9.0], [5.0, 1.0, 1.0]]),
        case(Matrix::<4, 4>::identity().rows),
        case([
            [-10.0, 4.0, 3.0, 4.0],
            [300.0, 5.0, 6.0, 7.0],
            [3.0, 6.0, 8.0, 9.0],
            [4.0, 7.0, 9.0, 10.0]
        ]),
        case(Matrix::<5, 5>::full(3.6).diag().as_dense().rows),
        case(Matrix::<8, 8>::identity().rows),
    )]
    fn test_determinant<const N: usize>(rows: [[f64; N]; N]) {
        let matrix = Matrix { rows };
        let faer_matrix = fill_faer(rows);

        let custom_det = matrix.determinant();
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
        // The fast algorithm does not compute the correct result for these degenerate
        // cases. test_svd_2x2_nalgebra verifies we reproduce the result expected for
        // this algorithm.
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

        if faeru.determinant().signum() != u.determinant().signum() {
            faeru[(0, 1)] *= -1.0;
            faeru[(1, 1)] *= -1.0;
        }
        if faerv.determinant().signum() != vt.determinant().signum() {
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
    }

    #[test]
    fn test_square_matrix_methods() {
        let identity = Matrix::<3, 3>::identity();
        let expected = Matrix::<3, 3> {
            rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        assert_matrixes_ulps_eq::<3, 3, _, _>(&identity, &expected);
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
        assert_diags_ulps_eq::<3, _>(&diag, &expected_diag);

        let from_diag = Matrix::<3, 3>::from_diag(&diag);
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
        let inv_matrix = matrix.inverse().expect("invertible");
        let product = matrix.matmul(&inv_matrix);
        let identity = Matrix22::identity();

        assert_matrixes_ulps_eq::<2, 2, _, _>(&product, &identity);
    }
    #[rstest(
        rows,
        case(Matrix33::identity().rows),
        case([[1.0, -3.0, 4.5], [3.0, 4.0,5.0], [8.0, -9.3, 10.0]]),
        case([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [1.0, 0.0, 1.0]]),
        case([[5.0, -2.0, 3.0], [1.0, 0.0, 4.0], [-1.0, 2.0, 1.0]])
    )]
    fn test_inverse_3x3(rows: [[f64; 3]; 3]) {
        let matrix = Matrix33 { rows };
        let inv_matrix = matrix.inverse().expect("invertible");
        let product = matrix.matmul(&inv_matrix);
        let identity = Matrix33::identity();

        assert_matrixes_ulps_eq::<3, 3, _, _>(&product, &identity);
    }
    #[rstest(
        rows,
        case(Matrix44::identity().rows),
        case([[1.0, -4.0, 4.5,1.0], [4.0, 4.0,5.0,0.0], [8.0, -9.4, 10.0,9.0], [-1.0,-1.0,1.0,1.0]]),
    )]
    fn test_inverse_4x4(rows: [[f64; 4]; 4]) {
        let matrix = Matrix44 { rows };
        let inv_matrix = matrix.inverse().expect("invertible");
        let product = matrix.matmul(&inv_matrix);
        let identity = Matrix44::identity();

        assert_matrixes_ulps_eq::<4, 4, _, _>(&product, &identity);
    }
}

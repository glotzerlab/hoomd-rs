// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use super::Matrix;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

/// Index the rows and columns of a [`Matrix`]
///
/// Indices for [`Matrix`] types are zero-indexed and reflect the indexing pattern of
/// the underlying data. This results in the pattern `(row, column)`, which mirrors the
/// behavior of Numpy and similar array languages.
///
/// # Examples
/// ```
/// use hoomd_linear_algebra::matrix::Matrix;
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

/// Compute the elementwise scalar multiplication of a [`Matrix`]
///
/// # Examples
/// ```
/// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix22};
/// let matrix = Matrix22::full(2.0);
/// let scalar = 2.0;
/// assert_eq!(matrix * scalar, matrix + matrix);
/// ```
impl<const N: usize, const M: usize> Mul<f64> for Matrix<N, M> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        self.map_elementwise(|x| x * rhs)
    }
}
/// Multiply an `f64` scalar by a [`Matrix`] rhs.
///
/// # Examples
/// ```
/// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix22};
/// let matrix = Matrix22::full(2.0);
/// let scalar = 3.0;
/// assert_eq!(scalar * matrix, matrix * scalar);
/// ```
impl<const N: usize, const M: usize> Mul<Matrix<N, M>> for f64 {
    type Output = Matrix<N, M>;

    #[inline]
    fn mul(self, rhs: Self::Output) -> Self::Output {
        rhs.map_elementwise(|x| x * self)
    }
}

/// Compute the elementwise, in-place scalar multiplication of a [`Matrix`]
///
/// # Examples
/// ```
/// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix22};
/// let mut matrix = Matrix22::full(2.0);
/// let matrix_copy = matrix.clone();
/// matrix *= 3.0;
/// assert_eq!(matrix, matrix_copy * 3.0);
/// ```
impl<const N: usize, const M: usize> MulAssign<f64> for Matrix<N, M> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.iter_flat_mut().for_each(|x| *x *= rhs);
    }
}

/// Compute the elementwise negation of a [`Matrix`].
///
/// # Examples
/// ```
/// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix22};
/// let matrix = Matrix22::full(5.0);
/// assert_eq!(-matrix, Matrix22::zeros() - matrix);
/// ```
impl<const N: usize, const M: usize> Neg for Matrix<N, M> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        self.map_elementwise(f64::neg)
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
impl<const N: usize, const M: usize> AddAssign for Matrix<N, M> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.iter_flat_mut()
            .zip(rhs.iter_flat())
            .for_each(|(x, r)| *x += r);
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
impl<const N: usize, const M: usize> SubAssign for Matrix<N, M> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.iter_flat_mut()
            .zip(rhs.iter_flat())
            .for_each(|(x, r)| *x -= r);
    }
}
impl<const N: usize, const M: usize> Matrix<N, M> {
    /// Extract a single row from a matrix.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::matrix::Matrix;
    ///
    /// let m: Matrix<2, 3> = Matrix {
    ///     rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    /// };
    /// let row = m.get_row(1);
    /// assert_eq!(row.rows, [[4.0, 5.0, 6.0]]);
    /// ```
    #[inline]
    #[must_use]
    pub fn get_row(&self, row_index: usize) -> Matrix<1, M> {
        Matrix {
            rows: [self.rows[row_index]],
        }
    }

    /// Extract a single column from a matrix.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::matrix::Matrix;
    ///
    /// let m: Matrix<2, 3> = Matrix {
    ///     rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    /// };
    /// let col = m.get_col(1);
    /// assert_eq!(col.rows, [[2.0], [5.0]]);
    /// ```
    #[inline]
    #[must_use]
    pub fn get_col(&self, col_index: usize) -> Matrix<N, 1> {
        Matrix {
            rows: std::array::from_fn(|i| [self.rows[i][col_index]]),
        }
    }

    /// Extract a submatrix of size `R`x`C` starting at `(start_row, start_col)`.
    ///
    /// # Panics
    ///
    /// If the requested data is out-of-bounds.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::matrix::Matrix;
    ///
    /// let m: Matrix<3, 3> = Matrix {
    ///     rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    /// };
    /// let sub = m.get_submatrix::<2, 2>(1, 1);
    /// assert_eq!(sub.rows, [[5.0, 6.0], [8.0, 9.0]]);
    /// ```
    #[inline]
    #[must_use]
    pub fn get_submatrix<const R: usize, const C: usize>(
        &self,
        start_row: usize,
        start_col: usize,
    ) -> Matrix<R, C> {
        Matrix {
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| self.rows[start_row + i][start_col + j])
            }),
        }
    }

    /// Returns a slice of a part of a row.
    ///
    /// # Panics
    ///
    /// Panics if the slice is out of bounds.
    #[must_use]
    #[inline]
    pub fn get_row_slice(&self, row_index: usize, start_col: usize, num_cols: usize) -> &[f64] {
        assert!(start_col + num_cols <= M);
        &self.rows[row_index][start_col..start_col + num_cols]
    }

    /// Returns an iterator over the elements of a part of a column.
    ///
    /// # Panics
    ///
    /// Panics if the slice is out of bounds.
    #[inline]
    pub fn get_col_slice_iter(
        &'_ self,
        col_index: usize,
        start_row: usize,
        num_rows: usize,
    ) -> impl Iterator<Item = f64> + '_ {
        assert!(start_row + num_rows <= N);
        assert!(col_index < M);
        self.as_slice()[start_row * M + col_index..]
            .iter()
            .step_by(M)
            .take(num_rows)
            .copied()
    }

    /// Returns the matrix as a flat slice of `f64`.
    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        // This is safe because the layout of [[f64; M]; N] is contiguous.
        unsafe { std::slice::from_raw_parts(self.rows.as_ptr().cast::<f64>(), N * M) }
    }

    /// Returns an iterator over slices of each row in a submatrix view.
    ///
    /// The submatrix is defined by its top-left corner `(start_row, start_col)`
    /// and its dimensions `(num_rows, num_cols)`.
    ///
    /// # Panics
    ///
    /// Panics if the submatrix is out of bounds.
    #[inline]
    pub fn submatrix_slice_iter(
        &'_ self,
        start_row: usize,
        start_col: usize,
        num_rows: usize,
        num_cols: usize,
    ) -> impl Iterator<Item = &'_ [f64]> {
        assert!(start_row + num_rows <= N);
        assert!(start_col + num_cols <= M);
        self.rows[start_row..start_row + num_rows]
            .iter()
            .map(move |row| &row[start_col..start_col + num_cols])
    }
}

#[cfg(test)]
mod tests {
    use crate::{GeneralMatrix, matrix::Matrix};
    use rstest::rstest;

    #[test]
    fn test_matrix_add_2x2() {
        let a_rows = [[1.0, 2.0], [3.0, 4.0]];
        let b_rows = [[5.0, 6.0], [7.0, 8.0]];

        let a = Matrix::<2, 2> { rows: a_rows };
        let b = Matrix::<2, 2> { rows: b_rows };
        let c = Matrix::<2, 2> {
            rows: [[6.0, 8.0], [10.0, 12.0]],
        };

        assert_eq!(a + b, c);
    }

    #[test]
    fn test_matrix_add_2x3() {
        let a_rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b_rows = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
        let a = Matrix::<2, 3> { rows: a_rows };
        let b = Matrix::<2, 3> { rows: b_rows };
        let c = Matrix::<2, 3> {
            rows: [[8.0, 10.0, 12.0], [14.0, 16.0, 18.0]],
        };

        assert_eq!(a + b, c);
    }

    #[test]
    fn test_matrix_sub_2x2() {
        let a_rows = [[1.0, 2.0], [3.0, 4.0]];
        let b_rows = [[5.0, 6.0], [7.0, 8.0]];
        let a = Matrix::<2, 2> { rows: a_rows };
        let b = Matrix::<2, 2> { rows: b_rows };
        let c = Matrix::<2, 2> {
            rows: [[-4.0, -4.0], [-4.0, -4.0]],
        };
        assert_eq!(a - b, c);
    }

    #[test]
    fn test_matrix_sub_2x3() {
        let a_rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b_rows = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
        let a = Matrix::<2, 3> { rows: a_rows };
        let b = Matrix::<2, 3> { rows: b_rows };
        let c = Matrix::<2, 3> {
            rows: [[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]],
        };
        assert_eq!(a - b, c);
    }

    #[rstest(
        rows,
        case([[1.0, -2.0], [3.0, 4.0]]),
        case([[0.0, 0.0], [0.0, 0.0]])
    )]
    fn test_matrix_neg_2x2(rows: [[f64; 2]; 2]) {
        let matrix = Matrix::<2, 2> { rows };
        let expected = Matrix {
            rows: rows.map(|row| row.map(|x| -x)),
        };
        assert_eq!(-matrix, expected);
    }

    #[rstest]
    #[case([[1.0, 2.0], [3.0, 4.0]], 5.0)]
    #[case([[1.0, 2.0], [3.0, 4.0]], -1.0)]
    #[case([[1.0, 2.0], [3.0, 4.0]], 0.0)]
    fn test_matrix_scalar_mul_2x2(#[case] rows: [[f64; 2]; 2], #[case] scalar: f64) {
        let matrix = Matrix::<2, 2> { rows };
        let expected = Matrix {
            rows: rows.map(|row| row.map(|x| x * scalar)),
        };
        assert_eq!(matrix * scalar, expected);
    }

    #[test]
    fn test_indexing() {
        // Matrix
        let mat = Matrix::<2, 3> {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        };
        assert_eq!(mat[(0, 2)], 3.0);
        assert_eq!(mat[(1, 1)], 5.0);
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

    #[rstest]
    #[case(
        [[1.0, 2.0], [ 3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
    )]
    #[case(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [[2.0, 3.0], [ 4.0, 5.0], [6.0, 7.0]],
    )]
    #[case(
        [[1.0],[ 2.0]],
        [[3.0], [4.0]],
    )]
    #[case(
        [[1.0, 2.0], [3.0, 4.0], [1.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )]
    fn test_add_assign<const M: usize, const N: usize>(
        #[case] a_rows: [[f64; M]; N],
        #[case] b_rows: [[f64; M]; N],
    ) {
        let mut a = Matrix { rows: a_rows };
        let b = Matrix { rows: b_rows };
        let c = a.clone() + b.clone();

        a += b;
        assert_eq!(a, c);
    }
    #[rstest]
    #[case(
        [[1.0, 2.0], [ 3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
    )]
    #[case(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [[2.0, 3.0], [ 4.0, 5.0], [6.0, 7.0]],
    )]
    #[case(
        [[1.0],[ 2.0]],
        [[3.0], [4.0]],
    )]
    #[case(
        [[1.0, 2.0], [3.0, 4.0], [1.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )]
    fn test_sub_assign<const M: usize, const N: usize>(
        #[case] a_rows: [[f64; M]; N],
        #[case] b_rows: [[f64; M]; N],
    ) {
        let mut a = Matrix { rows: a_rows };
        let b = Matrix { rows: b_rows };
        let c = a.clone() - b.clone();

        a -= b;
        assert_eq!(a, c);
    }
    #[rstest]
    #[case(
        [[1.0, 2.0], [ 3.0, 4.0], [5.0, 6.0]], 0.0
    )]
    #[case(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], -91.0
    )]
    #[case(
        [[1.0],[ 2.0]], 33.3
    )]
    #[case(
        [[1.0, 2.0], [3.0, 4.0], [1.0, 1.0]], 84.0
    )]
    fn test_mul_assign<const M: usize, const N: usize>(
        #[case] a_rows: [[f64; M]; N],
        #[case] x: f64,
    ) {
        let mut a = Matrix { rows: a_rows };
        let c = a.clone() * x;

        a *= x;
        assert_eq!(a, c);
    }

    #[rstest]
    #[case(
        [[1.0, 2.0], [ 3.0, 4.0], [5.0, 6.0]], 0.0
    )]
    #[case(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], -91.0
    )]
    #[case(
        [[1.0],[ 2.0]], 33.3
    )]
    #[case(
        [[1.0, 2.0], [3.0, 4.0], [1.0, 1.0]], 84.0
    )]
    fn test_mul_left<const M: usize, const N: usize>(
        #[case] a_rows: [[f64; M]; N],
        #[case] x: f64,
    ) {
        let a = Matrix { rows: a_rows };
        assert_eq!(a.clone() * x, x * a);
    }

    #[test]
    fn test_get_row() {
        let m: Matrix<2, 3> = Matrix {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        };
        let row0 = m.get_row(0);
        assert_eq!(row0.rows, [[1.0, 2.0, 3.0]]);
        let row1 = m.get_row(1);
        assert_eq!(row1.rows, [[4.0, 5.0, 6.0]]);
    }

    #[test]
    fn test_get_col() {
        let m: Matrix<2, 3> = Matrix {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        };
        let col0 = m.get_col(0);
        assert_eq!(col0.rows, [[1.0], [4.0]]);
        let col2 = m.get_col(2);
        assert_eq!(col2.rows, [[3.0], [6.0]]);
    }

    #[test]
    fn test_get_submatrix() {
        let m: Matrix<3, 4> = Matrix {
            rows: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
        };
        let sub = m.get_submatrix::<2, 2>(1, 1);
        assert_eq!(sub.rows, [[6.0, 7.0], [10.0, 11.0]]);

        let sub2 = m.get_submatrix::<1, 3>(0, 1);
        assert_eq!(sub2.rows, [[2.0, 3.0, 4.0]]);
    }

    #[test]
    fn test_get_submatrix_square() {
        let m: Matrix<3, 3> = Matrix {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        };
        let sub = m.get_submatrix::<2, 2>(0, 0);
        assert_eq!(sub.rows, [[1.0, 2.0], [4.0, 5.0]]);
    }

    #[test]
    #[should_panic]
    fn test_get_row_panic() {
        let m: Matrix<2, 3> = Matrix {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        };
        m.get_row(2);
    }

    #[test]
    #[should_panic]
    fn test_get_col_panic() {
        let m: Matrix<2, 3> = Matrix {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        };
        m.get_col(3);
    }

    #[test]
    #[should_panic]
    fn test_get_submatrix_panic_row() {
        let m: Matrix<3, 4> = Matrix {
            rows: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
        };
        // this should panic because it tries to access row 3
        let _ = m.get_submatrix::<2, 2>(2, 0);
    }

    #[test]
    #[should_panic]
    fn test_get_submatrix_panic_col() {
        let m: Matrix<3, 4> = Matrix {
            rows: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
        };
        // this should panic because it tries to access col 4
        let _ = m.get_submatrix::<2, 2>(0, 3);
    }

    #[test]
    fn test_get_row_slice() {
        let m: Matrix<2, 3> = Matrix {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        };
        let row_slice = m.get_row_slice(1, 1, 2);
        assert_eq!(row_slice, &[5.0, 6.0]);
    }

    #[test]
    fn test_get_col_slice_iter() {
        let m: Matrix<3, 4> = Matrix {
            rows: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
        };
        let col_iter = m.get_col_slice_iter(1, 0, 2);
        let col_vec: Vec<f64> = col_iter.collect();
        assert_eq!(col_vec, vec![2.0, 6.0]);
    }

    #[test]
    fn test_as_slice() {
        let m: Matrix<2, 3> = Matrix {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        };
        let slice = m.as_slice();
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_submatrix_slice_iter() {
        let m: Matrix<3, 4> = Matrix {
            rows: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
        };
        let mut sub_iter = m.submatrix_slice_iter(1, 1, 2, 2);
        assert_eq!(sub_iter.next(), Some(&[6.0, 7.0] as &[f64]));
        assert_eq!(sub_iter.next(), Some(&[10.0, 11.0] as &[f64]));
        assert_eq!(sub_iter.next(), None);
    }

    #[test]
    #[should_panic]
    fn test_submatrix_slice_iter_panic() {
        let m: Matrix<3, 4> = Matrix {
            rows: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
        };
        // This should panic.
        let _ = m.submatrix_slice_iter(1, 1, 3, 3);
    }

    #[test]
    fn test_submatrix_slice_iter_full() {
        let m: Matrix<2, 2> = Matrix {
            rows: [[1.0, 2.0], [3.0, 4.0]],
        };
        let mut sub_iter = m.submatrix_slice_iter(0, 0, 2, 2);
        assert_eq!(sub_iter.next(), Some(&[1.0, 2.0] as &[f64]));
        assert_eq!(sub_iter.next(), Some(&[3.0, 4.0] as &[f64]));
        assert_eq!(sub_iter.next(), None);
    }

    #[test]
    fn test_submatrix_slice_iter_single_element() {
        let m: Matrix<2, 2> = Matrix {
            rows: [[1.0, 2.0], [3.0, 4.0]],
        };
        let mut sub_iter = m.submatrix_slice_iter(1, 1, 1, 1);
        assert_eq!(sub_iter.next(), Some(&[4.0] as &[f64]));
        assert_eq!(sub_iter.next(), None);
    }

    #[test]
    fn test_submatrix_slice_iter_empty() {
        let m: Matrix<2, 2> = Matrix {
            rows: [[1.0, 2.0], [3.0, 4.0]],
        };
        let mut sub_iter_zero_rows = m.submatrix_slice_iter(1, 1, 0, 1);
        assert_eq!(sub_iter_zero_rows.next(), None);

        let mut sub_iter_zero_cols = m.submatrix_slice_iter(1, 1, 1, 0);
        assert_eq!(sub_iter_zero_cols.next(), Some(&[] as &[f64]));
        assert_eq!(sub_iter_zero_cols.next(), None);
    }
}

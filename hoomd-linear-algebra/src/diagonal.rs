// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use crate::{GeneralMatrix, matrix::Matrix};

/// A square, diagonal matrix with N rows and N columns.
///
/// # Example
/// ```
/// use hoomd_linear_algebra::{GeneralMatrix, matrix::DiagonalMatrix};
/// let a = DiagonalMatrix {
///     elements: [-2.0, 3.0],
/// };
///
/// assert_eq!(a[(0,0)], -2.0);
/// assert_eq!(a[(0,1)], 0.0);
/// assert_eq!(a[(1,0)], 0.0);
/// assert_eq!(a[(1,1)], 3.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiagonalMatrix<const N: usize> {
    /// The members of the diagonal of the matrix
    pub elements: [f64; N],
}

impl<const N: usize> Index<usize> for DiagonalMatrix<N> {
    type Output = f64;

    /// Index the diagonal components.
    ///
    /// # Example
    /// ```
    /// use hoomd_linear_algebra::{SquareMatrix, matrix::DiagonalMatrix};
    /// let a = DiagonalMatrix {
    ///     elements: [1.0, 2.0, 3.0],
    /// };
    /// assert_eq!(a[0], 1.0);
    /// assert_eq!(a[1], 2.0);
    /// assert_eq!(a[2], 3.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &f64 {
        &self.elements[index]
    }
}
impl<const N: usize> IndexMut<usize> for DiagonalMatrix<N> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        &mut self.elements[index]
    }
}

impl<const N: usize> Index<(usize, usize)> for DiagonalMatrix<N> {
    type Output = f64;

    /// Index matrix elements by (row, column).
    ///
    /// # Example
    /// ```
    /// use hoomd_linear_algebra::matrix::DiagonalMatrix;
    /// let a = DiagonalMatrix {
    ///     elements: [1.0, 2.0, 3.0],
    /// };
    /// assert_eq!(a[(0, 0)], 1.0);
    /// assert_eq!(a[(1, 1)], 2.0);
    /// assert_eq!(a[(0, 2)], 0.0);
    /// assert_eq!(a[(2, 2)], 3.0);
    /// ```
    #[inline]
    fn index(&self, index: (usize, usize)) -> &f64 {
        let (i, j) = index;
        if i == j { &self.elements[i] } else { &0.0 }
    }
}

impl<const N: usize> GeneralMatrix for DiagonalMatrix<N> {
    #[inline]
    fn zeros() -> Self {
        Self {
            elements: std::array::from_fn(|_| 0.0),
        }
    }
}

impl<const N: usize> DiagonalMatrix<N> {
    /// Construct a dense matrix with the given diagonal.
    ///
    /// # Example
    /// ```
    /// use hoomd_linear_algebra::{
    ///     matrix::{DiagonalMatrix, Matrix22},
    /// };
    ///
    /// let a = DiagonalMatrix { elements: [-2.0, 3.0], };
    /// let b = a.to_dense();
    /// assert_eq!(b.rows, [[-2.0, 0.0], [0.0, 3.0]]);
    /// ```
    #[must_use]
    #[inline]
    pub fn to_dense(self) -> Matrix<N, N> {
        Matrix {
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { self.elements[i] } else { 0.0 })
            }),
        }
    }
}

impl<const N: usize> Mul<f64> for DiagonalMatrix<N> {
    type Output = Self;

    /// Multiply a diagonal matrix by a scalar.
    ///
    /// # Example
    /// ```
    /// use hoomd_linear_algebra::matrix::DiagonalMatrix;
    /// let a = DiagonalMatrix {
    ///     elements: [-3.0, 2.0, -8.0],
    /// };
    ///
    /// let b = a * 3.0;
    /// assert_eq!(b[0], -9.0);
    /// assert_eq!(b[1], 6.0);
    /// assert_eq!(b[2], -24.0);
    /// ```
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            elements: self.elements.map(|r| r * rhs),
        }
    }
}

impl<const N: usize> Neg for DiagonalMatrix<N> {
    type Output = Self;

    /// Negate a diagonal matrix.
    ///
    /// # Example
    /// ```
    /// use hoomd_linear_algebra::matrix::DiagonalMatrix;
    /// let a = DiagonalMatrix {
    ///     elements: [-3.0, 2.0, -8.0],
    /// };
    ///
    /// let b = -a;
    /// assert_eq!(b[0], 3.0);
    /// assert_eq!(b[1], -2.0);
    /// assert_eq!(b[2], 8.0);
    /// ```
    #[inline]
    fn neg(self) -> Self {
        Self {
            elements: self.elements.map(|r| -r),
        }
    }
}

impl<const N: usize> Add<Self> for DiagonalMatrix<N> {
    type Output = Self;

    /// Add two diagonal matrices.
    ///
    /// # Example
    /// ```
    /// use hoomd_linear_algebra::matrix::DiagonalMatrix;
    /// let a = DiagonalMatrix {
    ///     elements: [-3.0, 2.0, -8.0],
    /// };
    /// let b = DiagonalMatrix {
    ///     elements: [4.0, -4.0, 12.0],
    /// };
    ///
    /// let c = a + b;
    /// assert_eq!(c[0], 1.0);
    /// assert_eq!(c[1], -2.0);
    /// assert_eq!(c[2], 4.0);
    /// ```
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            elements: std::array::from_fn(|i| self[i] + rhs[i]),
        }
    }
}
impl<const N: usize> Sub<Self> for DiagonalMatrix<N> {
    type Output = Self;

    /// Subtract two diagonal matrices.
    ///
    /// # Example
    /// ```
    /// use hoomd_linear_algebra::matrix::DiagonalMatrix;
    /// let a = DiagonalMatrix {
    ///     elements: [-3.0, 2.0, -8.0],
    /// };
    /// let b = DiagonalMatrix {
    ///     elements: [4.0, -4.0, 12.0],
    /// };
    ///
    /// let c = a - b;
    /// assert_eq!(c[0], -7.0);
    /// assert_eq!(c[1], 6.0);
    /// assert_eq!(c[2], -20.0);
    /// ```
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            elements: std::array::from_fn(|i| self[i] - rhs[i]),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use rstest::rstest;
    use std::ops::Index;

    use super::*;
    use crate::GeneralMatrix;

    fn assert_diags_ulps_eq<const N: usize>(
        m0: &DiagonalMatrix<N>,
        m1: &impl Index<usize, Output = f64>,
    ) {
        for i in 0..N {
            assert_ulps_eq!(m0[i], m1[i], epsilon = 1e-13);
        }
    }

    #[test]
    fn test_diagonal_matrix_add_n2() {
        let a_diag = [1.0, 2.0];
        let b_diag = [3.0, 4.0];
        let a = DiagonalMatrix::<2> { elements: a_diag };
        let b = DiagonalMatrix::<2> { elements: b_diag };
        let expected: Vec<f64> = a_diag
            .iter()
            .zip(b_diag.iter())
            .map(|(x, y)| x + y)
            .collect();
        let custom_sum = a + b;
        assert_diags_ulps_eq(&custom_sum, &expected);
    }

    #[test]
    fn test_diagonal_matrix_add_n3() {
        let a_diag = [1.0, 2.0, 3.0];
        let b_diag = [4.0, 5.0, 6.0];
        let a = DiagonalMatrix::<3> { elements: a_diag };
        let b = DiagonalMatrix::<3> { elements: b_diag };
        let expected: Vec<f64> = a_diag
            .iter()
            .zip(b_diag.iter())
            .map(|(x, y)| x + y)
            .collect();
        let custom_sum = a + b;
        assert_diags_ulps_eq(&custom_sum, &expected);
    }

    #[test]
    fn test_diagonal_matrix_sub_n2() {
        let a_diag = [1.0, 2.0];
        let b_diag = [3.0, 4.0];
        let a = DiagonalMatrix::<2> { elements: a_diag };
        let b = DiagonalMatrix::<2> { elements: b_diag };
        let expected: Vec<f64> = a_diag
            .iter()
            .zip(b_diag.iter())
            .map(|(x, y)| x - y)
            .collect();
        let custom_sub = a - b;
        assert_diags_ulps_eq(&custom_sub, &expected);
    }

    #[test]
    fn test_diagonal_matrix_sub_n3() {
        let a_diag = [1.0, 2.0, 3.0];
        let b_diag = [4.0, 5.0, 6.0];
        let a = DiagonalMatrix::<3> { elements: a_diag };
        let b = DiagonalMatrix::<3> { elements: b_diag };
        let expected: Vec<f64> = a_diag
            .iter()
            .zip(b_diag.iter())
            .map(|(x, y)| x - y)
            .collect();
        let custom_sub = a - b;
        assert_diags_ulps_eq(&custom_sub, &expected);
    }

    #[test]
    fn test_diagonal_matrix_neg_n2() {
        let diag = [1.0, -2.0];
        let matrix = DiagonalMatrix::<2> { elements: diag };
        let expected: Vec<f64> = diag.iter().map(|x| -x).collect();
        let custom_neg = -matrix;
        assert_diags_ulps_eq(&custom_neg, &expected);
    }

    #[test]
    fn test_diagonal_matrix_neg_n3() {
        let diag = [1.0, -2.0, 0.0];
        let matrix = DiagonalMatrix::<3> { elements: diag };
        let expected: Vec<f64> = diag.iter().map(|x| -x).collect();
        let custom_neg = -matrix;
        assert_diags_ulps_eq(&custom_neg, &expected);
    }

    #[rstest]
    #[case([1.0, 2.0], 5.0)]
    #[case([1.0, 2.0], -1.0)]
    #[case([1.0, 2.0], 0.0)]
    fn test_diagonal_matrix_scalar_mul_n2(#[case] diag: [f64; 2], #[case] scalar: f64) {
        let matrix = DiagonalMatrix::<2> { elements: diag };
        let expected: Vec<f64> = diag.iter().map(|x| x * scalar).collect();
        let custom_mul = matrix * scalar;
        assert_diags_ulps_eq(&custom_mul, &expected);
    }

    #[test]
    fn test_indexing() {
        // DiagonalMatrix
        let diag_mat = DiagonalMatrix::<3> {
            elements: [1.0, 2.0, 3.0],
        };
        assert_eq!(diag_mat[1], 2.0); // 1D indexing
        assert_eq!(diag_mat[(2, 2)], 3.0); // 2D on-diagonal
        assert_eq!(diag_mat[(0, 1)], 0.0); // 2D off-diagonal
    }

    #[test]
    fn test_general_matrix_methods() {
        // DiagonalMatrix
        let diag_zeros = DiagonalMatrix::<4>::zeros();
        for i in 0..4 {
            assert_eq!(diag_zeros[i], 0.0);
        }
    }
}

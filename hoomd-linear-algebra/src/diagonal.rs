// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use crate::{Diagonal, GeneralMatrix, matrix::Matrix};

/// A square, diagonal matrix with N rows and N columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiagonalMatrix<const N: usize> {
    /// The members of the diagonal of the matrix
    pub elements: [f64; N],
}

/// Index the on-diagonal components of a diagonal matrix
/// # Examples
/// ```
/// use hoomd_linear_algebra::{SquareMatrix, matrix::DiagonalMatrix};
/// let mat = DiagonalMatrix {
///     elements: [1.0, 2.0, 3.0],
/// };
/// assert_eq!(mat[0], 1.0);
/// assert_eq!(mat[1], 2.0);
/// assert_eq!(mat[2], 3.0);
/// ```
impl<const N: usize> Index<usize> for DiagonalMatrix<N> {
    type Output = f64;
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

/// Index the dense view of a diagonal matrix. Off-diagonal elements will be `0.0`.
/// # Examples
/// ```
/// use hoomd_linear_algebra::{SquareMatrix, matrix::DiagonalMatrix};
/// let mat = DiagonalMatrix {
///     elements: [1.0, 2.0, 3.0],
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
    #[inline]
    fn full(val: f64) -> Self {
        Self {
            elements: std::array::from_fn(|_| val),
        }
    }
}

impl<const N: usize> DiagonalMatrix<N> {
    /// Return a dense view of the diagonal matrix, with zeros on the off-diagonals.
    #[must_use]
    #[inline]
    pub fn as_dense(&self) -> Matrix<N, N> {
        Matrix::<N, N>::from_diag(&self.elements)
    }
}

/// Compute the elementwise scalar multiplication of a [`DiagonalMatrix`]
impl<const N: usize> Mul<f64> for DiagonalMatrix<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            elements: self.elements.map(|r| r * rhs),
        }
    }
}
/// Compute the elementwise negation of a [`DiagonalMatrix`]
impl<const N: usize> Neg for DiagonalMatrix<N> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            elements: self.elements.map(|r| -r),
        }
    }
}

impl<const N: usize> Add<Self> for DiagonalMatrix<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            elements: std::array::from_fn(|i| self[i] + rhs[i]),
        }
    }
}
impl<const N: usize> Sub<Self> for DiagonalMatrix<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            elements: std::array::from_fn(|i| self[i] - rhs[i]),
        }
    }
}

impl<const N: usize> Diagonal for DiagonalMatrix<N> {}

#[cfg(test)]
mod tests {
    use approx::assert_ulps_eq;
    use rstest::rstest;
    use std::ops::Index;

    use super::*;
    use crate::{Diagonal, GeneralMatrix};

    fn assert_diags_ulps_eq<const N: usize, T: Diagonal>(
        m0: &T,
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
        assert_diags_ulps_eq::<2, _>(&custom_sum, &expected);
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
        assert_diags_ulps_eq::<3, _>(&custom_sum, &expected);
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
        assert_diags_ulps_eq::<2, _>(&custom_sub, &expected);
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
        assert_diags_ulps_eq::<3, _>(&custom_sub, &expected);
    }

    #[test]
    fn test_diagonal_matrix_neg_n2() {
        let diag = [1.0, -2.0];
        let matrix = DiagonalMatrix::<2> { elements: diag };
        let expected: Vec<f64> = diag.iter().map(|x| -x).collect();
        let custom_neg = -matrix;
        assert_diags_ulps_eq::<2, _>(&custom_neg, &expected);
    }

    #[test]
    fn test_diagonal_matrix_neg_n3() {
        let diag = [1.0, -2.0, 0.0];
        let matrix = DiagonalMatrix::<3> { elements: diag };
        let expected: Vec<f64> = diag.iter().map(|x| -x).collect();
        let custom_neg = -matrix;
        assert_diags_ulps_eq::<3, _>(&custom_neg, &expected);
    }

    #[rstest]
    #[case([1.0, 2.0], 5.0)]
    #[case([1.0, 2.0], -1.0)]
    #[case([1.0, 2.0], 0.0)]
    fn test_diagonal_matrix_scalar_mul_n2(#[case] diag: [f64; 2], #[case] scalar: f64) {
        let matrix = DiagonalMatrix::<2> { elements: diag };
        let expected: Vec<f64> = diag.iter().map(|x| x * scalar).collect();
        let custom_mul = matrix * scalar;
        assert_diags_ulps_eq::<2, _>(&custom_mul, &expected);
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
        let diag_full = DiagonalMatrix::<4>::full(-3.0);
        for i in 0..4 {
            assert_eq!(diag_zeros[i], 0.0);
            assert_eq!(diag_full[i], -3.0);
        }
    }
}

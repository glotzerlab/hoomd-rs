// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use super::Matrix;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

impl<const N: usize, const M: usize> Index<(usize, usize)> for Matrix<N, M> {
    type Output = f64;

    /// Access matrix elements..
    ///
    /// Elements are indexed by `(row, column)`.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::matrix::Matrix;
    /// 
    /// let rows = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let a = Matrix { rows };
    /// assert_eq!(a[(0, 1)], rows[0][1]);
    /// assert_eq!(a[(2, 1)], 6.0);
    /// assert_eq!(a[(1, 1)], 4.0);
    /// ```
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

impl<const N: usize, const M: usize> Mul<f64> for Matrix<N, M> {
    type Output = Self;

    #[inline]
    /// Matrix-scalar multiplication.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix22};
    /// 
    /// let matrix = Matrix22::full(2.0);
    /// let scalar = 2.0;
    /// assert_eq!(matrix * scalar, matrix + matrix);
    /// ```
    fn mul(self, rhs: f64) -> Self {
        self.map_elementwise(|x| x * rhs)
    }
}

impl<const N: usize, const M: usize> Mul<Matrix<N, M>> for f64 {
    type Output = Matrix<N, M>;

    /// Matrix-scalar multiplication .
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix22};
    /// 
    /// let matrix = Matrix22::full(2.0);
    /// let scalar = 3.0;
    /// assert_eq!(scalar * matrix, matrix * scalar);
    /// ```
    #[inline]
    fn mul(self, rhs: Self::Output) -> Self::Output {
        rhs.map_elementwise(|x| x * self)
    }
}

impl<const N: usize, const M: usize> MulAssign<f64> for Matrix<N, M> {
    #[inline]
    /// Matrix-scalar multiplication assignment.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix22};
    /// 
    /// let mut matrix = Matrix22::full(2.0);
    /// let matrix_copy = matrix.clone();
    /// matrix *= 3.0;
    /// assert_eq!(matrix, matrix_copy * 3.0);
    /// ```
    fn mul_assign(&mut self, rhs: f64) {
        self.iter_flat_mut().for_each(|x| *x *= rhs);
    }
}

impl<const N: usize, const M: usize> Neg for Matrix<N, M> {
    type Output = Self;

    /// Matrix negation.
    ///
    /// # Examples
    /// ```
    /// use hoomd_linear_algebra::{Full, GeneralMatrix, matrix::Matrix22};
    /// 
    /// let matrix = Matrix22::full(5.0);
    /// assert_eq!(-matrix, Matrix22::zeros() - matrix);
    /// ```
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
}

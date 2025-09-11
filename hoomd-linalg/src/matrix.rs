use std::ops::{Add, Index, Mul};

use crate::{Determinant, Diagonal, GeneralMatrix, Invertible, SquareMatrix};
use hoomd_vector::{Cartesian, RotationMatrix};

/// A matrix with N rows and M columns, allocated on the stack.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<const N: usize, const M: usize> {
    /// The elements of the matrix
    rows: [[f64; M]; N],
}
/// A square, diagonal matrix with N rows and N columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiagonalMatrix<const N: usize> {
    /// The elements of the diagonal of the matrix
    rows: [f64; N],
}

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
    #[inline]
    fn iter_rows(&self) -> impl Iterator<Item = impl IntoIterator<Item = &f64>> {
        self.rows.iter()
    }
}

impl<const N: usize> SquareMatrix for Matrix<N, N> {
    #[inline]
    fn eye() -> Self {
        Self {
            rows: std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 })),
        }
    }
    #[inline]
    fn compute_quadratic_form(&self, vars: &impl Diagonal) -> f64 {
        let mut result = 0.0;

        for i in 0..N {
            for j in 0..N {
                result += vars[i] * self.rows[i][j] * vars[j];
            }
        }
        result
    }
}

impl<const N: usize> Determinant for Matrix<N, N> {
    /**Compute the determinant of a matrix via a Laplace expansion.
    Note that, while this implementation is optimal for small matrixes, it has O(N!)
    time complexity and will be extremely slow for large matrixes.
    */
    #[inline]
    fn det(&self) -> f64 {
        /*
        Because math with const generics is not allowed in rust, we compute the indices
        of each submatrix and recur on those noncontiguous segments of the input.
        */
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
}

impl<const N: usize> From<RotationMatrix<N>> for Matrix<N, N> {
    #[inline]
    fn from(value: RotationMatrix<N>) -> Self {
        Self {
            rows: value.rows().map(|arr| arr.coordinates),
        }
    }
}

impl<const N: usize> Matrix<N, N> {
    /// Extract the diagonal elements from a matrix
    #[must_use]
    #[inline]
    pub fn diag(&self) -> DiagonalMatrix<N> {
        DiagonalMatrix {
            rows: std::array::from_fn(|i| self.rows[i][i]),
        }
    }
    /// Compute a full `NxN` matrix from N diagonal elements, setting all others to 0.
    #[must_use]
    #[inline]
    pub fn from_diag(other: &[f64; N]) -> Self {
        Matrix {
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { other[i] } else { 0.0 })
            }),
        }
    }

    /// Multiply a [`Matrix`] by a diagonal matrix on the right hand side
    #[must_use]
    #[inline]
    pub fn matmul_diagonal(&self, diag: &[f64; N]) -> Self {
        let mut rows = [[0f64; N]; N];
        for (i, row) in rows.iter_mut().enumerate().take(N) {
            for j in 0..N {
                row[j] = self.rows[i][j] * diag[j];
            }
        }
        Self { rows }
    }

    /// (Naive) Matrix multiplication of two square matrixes
    #[must_use]
    #[inline]
    pub fn matmul(&self, other: &Self) -> Self {
        let mut result = Self {
            rows: [[0.0; N]; N],
        };
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    result.rows[i][j] += self.rows[i][k] * other.rows[k][j];
                }
            }
        }

        result
    }
}

/**Compute the elementwise scalar multiplication of a [`Matrix`]*/
impl<const N: usize, const M: usize> Mul<f64> for Matrix<N, M> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            rows: self.rows.map(|r| r.map(|x| x * rhs)),
        }
    }
}
/**Compute the elementwise scalar multiplication of a [`DiagonalMatrix`]*/
impl<const N: usize> Mul<f64> for DiagonalMatrix<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            rows: self.rows.map(|r| r * rhs),
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

impl Copy for Matrix<2, 2> {}
impl Copy for Matrix<3, 3> {}
impl Copy for Matrix<4, 4> {}
impl<const N: usize> Diagonal for DiagonalMatrix<N> {}
impl<const N: usize> Diagonal for [f64; N] {}
impl<const N: usize> Diagonal for Cartesian<N> {}

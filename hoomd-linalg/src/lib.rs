// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! asdf

TODO: Expand documentation.
*/

use hoomd_vector::RotationMatrix;

use std::ops::{Add, Mul};

/** Define whether a matrix $ A $ has an inverse $ A^-1 $ such that $ AA^-1 = A^-1A = I $
*/
pub trait Invertible {
    /// Compute the inverse of a matrix.
    #[must_use]
    fn inverse(&self) -> Self;
}

// /** Compute the determinant of a matrix.
// */
// pub trait Determinant: Sized
// where
//     Self: SquareMatrix,
// {

// }
// a few options: Leibniz, SVD Decompose, laplace
// Key difficulty: specialization. it would be nice to have this for general N, but this
// is obviously nontrivial. BUT: for 2x2 and 3x3, Laplace is optimal

/** General implementation for size and container-agnostic matrixes.

This trait is designed to function with row-major ordering, but this is not strictly
required for correct functionality.
*/
pub trait GeneralMatrix: Mul<f64, Output = Self> {
    /// TODO
    #[must_use]
    fn zeros() -> Self;

    /// Iterate over the rows of a matrix.
    #[must_use]
    fn iter_rows(&self) -> impl Iterator<Item = impl IntoIterator<Item = &f64>>;
}

/// TODO
pub trait SquareMatrix: GeneralMatrix
where
    Self: Sized,
{
    /// TODO
    #[must_use]
    fn diag() -> Self;

    /** Compute the determinant of a matrix using the Laplace expansion.

    Note that, while this implementation is optimal for small matrixes, it has O(n!)
    time complexity and will be extremely slow for large matrixes.
    */
    #[must_use]
    fn det(&self) -> f64;
}

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

/// A 2x2 matrix, allocated on the stack.
pub type Matrix22 = Matrix<2, 2>;

impl<const N: usize, const M: usize> GeneralMatrix for Matrix<N, M> {
    // type MatrixType = Matrix<N, M>;
    #[inline]
    fn zeros() -> Self {
        Self {
            rows: std::array::from_fn(|_| std::array::from_fn(|_| 0.0)),
        }
    }
    #[inline]
    fn iter_rows(&self) -> impl Iterator<Item = impl IntoIterator<Item = &f64>> {
        self.rows.iter()
    }
}

impl<const N: usize> SquareMatrix for Matrix<N, N> {
    #[inline]
    fn diag() -> Self {
        Self {
            rows: std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 })),
        }
    }

    #[inline]
    fn det(&self) -> f64 {
        #[inline]
        fn det_recursive<const N: usize>(
            matrix: &Matrix<N, N>,
            row: usize,
            col_indices: &[usize; N],
            size: usize,
        ) -> f64 {
            if size == 2 {
                let j0 = col_indices[0];
                let j1 = col_indices[1];
                return matrix.rows[row][j0] * matrix.rows[row + 1][j1]
                    - matrix.rows[row][j1] * matrix.rows[row + 1][j0];
            }

            let mut det_sum = 0.0;

            for idx in 0..size {
                // Build minor column indices without dynamic allocation
                let mut minor_cols: [usize; N] = [0; N];
                let mut minor_size = 0;
                for j in 0..size {
                    if j != idx {
                        minor_cols[minor_size] = col_indices[j];
                        minor_size += 1;
                    }
                }

                let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
                det_sum += sign
                    * matrix.rows[row][col_indices[idx]]
                    * det_recursive(matrix, row + 1, &minor_cols, minor_size);
            }

            det_sum
        }
        // This would be handled by the iteration, but this simplifies the code
        match N {
            1 => return self.rows[0][0],
            2 => return self.rows[0][0] * self.rows[1][1] - self.rows[1][0] * self.rows[0][1],
            _ => (),
        }

        let mut col_indices: [usize; N] = [0; N];
        for i in 0..N {
            col_indices[i] = i;
        }

        det_recursive(self, 0, &col_indices, N)
    }
}
// impl<const N: usize> GeneralMatrix for DiagonalMatrix<N> {
//     // type MatrixType = DiagonalMatrix<N>;
//     #[inline]
//     fn zeros() -> Self {
//         Self {
//             rows: std::array::from_fn(|_| 0.0),
//         }
//     }
// }

impl<const N: usize> From<RotationMatrix<N>> for Matrix<N, N> {
    #[inline]
    fn from(value: RotationMatrix<N>) -> Self {
        Self {
            rows: value.rows().map(|arr| arr.coordinates),
        }
    }
}

impl<const N: usize> Matrix<N, N> {
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
    pub fn mul_diagonal(&self, diag: &[f64; N]) -> Self {
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
    /// Solve the quadratic form for a pair of matrices.
    #[must_use]
    #[inline]
    pub fn compute_quadratic_form(&self, other: &[f64; N]) -> f64 {
        let mut result = 0.0;

        for i in 0..N {
            for j in 0..N {
                result += other[i] * self.rows[i][j] * other[j];
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

impl Matrix<2, 2> {
    /// The determinant of a 2x2 square matrix.
    // #[must_use]
    // #[inline]
    // pub fn det(&self) -> f64 {
    //     self.rows[0][0] * self.rows[1][1] - self.rows[1][0] * self.rows[0][1]
    // }
    /// The inverse of a 2x2 square matrix.
    #[must_use]
    #[inline]
    pub fn inverse(&self) -> Self {
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

/*I think it makes the most sense to keep this general in terms of dimension. Rather than worry about specialization, we can just implement specific approaches for single size matrices like 3 x 3, rather than general methods. Obviously specialization would be nice, but people probably should defer to more robust libraries for complicated linear algebra problems anyway so it makes sense to focus on small specific cases. We can give a little bit more specialization by implementing types of various levels of restriction, for example, symmetric, diagonal and square matrixes.

If it turns out we really want a large matrix SVD or something, we can have a MatrixLike
wrapper class that implements that subroutine. This also allows us to have separate
dynamically allocated classes.
*/

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::{Mat, mat};
    use rstest::rstest;

    fn fill_faer<const N: usize, const M: usize>(m: [[f64; M]; N]) -> Mat<f64> {
        let mut faer_matrix = Mat::<f64>::zeros(N, N);
        for (i, row) in m.iter().enumerate() {
            for (j, el) in row.iter().enumerate() {
                *faer_matrix.get_mut(i, j) = *el;
            }
        }
        faer_matrix
    }
    #[rstest(
        rows,
        case([[1.0, 2.0], [3.0, 4.0]]),
        case([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]),
        case([[1.0, 0.0, 0.0], [0.0, 1.0, 4.0], [0.0, 1.0, 0.0]]),
        case([[2.0, 0.0, 1.0], [3.0, 0.0, 0.0], [5.0, 1.0, 1.0]])
    )]
    fn test_determinant_parametrized<const N: usize>(rows: [[f64; N]; N]) {
        let matrix = Matrix { rows };
        let faer_matrix = fill_faer(rows);

        let custom_det = matrix.det();
        let faer_det = faer_matrix.determinant();

        assert_relative_eq!(custom_det, faer_det, max_relative = 1e-14);
    }
}

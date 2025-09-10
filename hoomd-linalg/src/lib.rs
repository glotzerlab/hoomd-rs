/*! asdf

TODO: Expand documentation.
*/

use hoomd_vector::RotationMatrix;

use std::ops::{Add, Mul};

/// asdf
// pub trait MatMul<const N: usize, const M: usize, const K: usize> {
//     #[must_use]
//     #[inline]
//     fn matmul(left: &Matrix<N, M>, right: &Matrix<M, K>) -> Matrix<N, K> {
//         let mut result = Matrix::<N, K>::zeros();
//         for i in 0..N {
//             for j in 0..N {
//                 for k in 0..N {
//                     result.rows[i][j] += left.rows[i][k] * right.rows[k][j];
//                 }
//             }
//         }

//         result
//     }
// }

/// in total we should have 1 type and 2 traits?
pub trait GeneralMatrix {
    /// TODO
    type T;

    /// TODO
    #[must_use]
    fn zeros() -> Self::T;
}

/// An ``NxM`` matrix, allocated on the stack
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<const N: usize, const M: usize> {
    /// The elements of the matrix
    rows: [[f64; M]; N],
}

/// A 2x2 matrix, allocated on the stack.
pub type Matrix22 = Matrix<2, 2>;

impl<const N: usize, const M: usize> GeneralMatrix for Matrix<N, M> {
    type T = Matrix<N, M>;
    #[inline]
    fn zeros() -> Self::T {
        Self::T {
            rows: std::array::from_fn(|_| std::array::from_fn(|_| 0.0)),
        }
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

impl<const N: usize> Default for Matrix<N, N> {
    #[inline]
    fn default() -> Matrix<N, N> {
        Matrix {
            rows: std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 })),
        }
    }
}

impl<const N: usize> Matrix<N, N> {
    /// Compute a full `NxN` matrix from N diagonal elements, setting all others to 0.
    #[inline]
    pub fn from_diag(other: &[f64; N]) -> Self {
        Matrix {
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { other[i] } else { 0.0 })
            }),
        }
    }

    /// Multiply a [`StackMatrix`] by a diagonal matrix on the right hand side
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

/**Compute the elementwise multiplication of a [`StackMatrix`]*/
impl<const N: usize, const M: usize> Mul<f64> for Matrix<N, M> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            rows: self.rows.map(|r| r.map(|x| x * rhs)),
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
    #[inline]
    pub fn det(&self) -> f64 {
        self.rows[0][0] * self.rows[1][1] - self.rows[1][0] * self.rows[0][1]
    }
    /// The inverse of a 2x2 square matrix.
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

/*I think it makes the most sense to keep this general in terms of dimension. Rather than worry about specialization, we can just implement specific approaches for single size matrices like 3 x 3, rather than general methods. Obviously specialization would be nice, but people probably should defer to more robust libraries for complicated linear algebra problems anyway so it makes sense to focus on small specific cases. We can give a little bit more specialization by implementing types of various levels of restriction, for example, symmetric, diagonal and square matrixes.

If it turns out we really want a large matrix SVD or something, we can have a MatrixLike
wrapper class that implements that subroutine. This also allows us to have separate
dynamically allocated classes.
*/

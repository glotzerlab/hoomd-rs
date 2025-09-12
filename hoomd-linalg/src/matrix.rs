// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, Neg};

use crate::{Determinant, Diagonal, GeneralMatrix, Invertible, MatMul, SVD, SquareMatrix};
use hoomd_vector::{Angle, Cartesian, RotationMatrix};

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

impl<const N: usize> Index<(usize, usize)> for DiagonalMatrix<N> {
    type Output = f64;
    #[inline]
    fn index(&self, index: (usize, usize)) -> &f64 {
        let (i, _) = index;
        &self.rows[i]
    }
}
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

impl<const N: usize, const M: usize> MatMul<DiagonalMatrix<M>> for Matrix<N, M> {
    type Output = Matrix<M, M>;
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

impl<const N: usize> Matrix<N, N> {
    /** Extract the diagonal elements from a square matrix.

    This method returns a `DiagonalMatrix<N>` containing the diagonal elements
    of the input matrix, where the element at position `(i, i)` is taken from
    the input matrix. All off-diagonal elements are ignored.

    # Examples
    ```
    use hoomd_linalg::matrix::Matrix33;
    let mat = Matrix33 {
        rows: [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
    ]};
    let diag = mat.diag();
    assert_eq!(diag.rows, [1.0, 5.0, 9.0]);
    ```
    */
    #[must_use]
    #[inline]
    pub fn diag(&self) -> DiagonalMatrix<N> {
        DiagonalMatrix {
            rows: std::array::from_fn(|i| self.rows[i][i]),
        }
    }

    /** Compute a full `NxN` matrix from N diagonal elements, setting all others to 0.
    # Examples
    ```
    use hoomd_linalg::matrix::Matrix33;
    let mat = Matrix33::from_diag(&[1.0, 5.0, 9.0]);
    assert_eq!(mat.diag().rows, [1.0, 5.0, 9.0]);
    assert_eq!(mat[(1, 2)], 0.0);
    ```
    */
    #[must_use]
    #[inline]
    pub fn from_diag<T: Diagonal>(other: &T) -> Self {
        Matrix {
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { other[i] } else { 0.0 })
            }),
        }
    }

    /** Scale each column of a [`Matrix`] by the corresponding element in a [`Diagonal`].

    # Example
    ```
    use hoomd_linalg::matrix::{Matrix22, DiagonalMatrix};
    use hoomd_linalg::GeneralMatrix;
    let diag = DiagonalMatrix { rows: [3.0, 4.0] };
    let mat = Matrix22::full(1.0).matmul_diagonal(&diag);
    assert_eq!(mat[(0, 1)], 4.0);
    assert_eq!(mat[(1, 0)], 3.0);
    ```
    */
    #[must_use]
    #[inline]
    pub fn matmul_diagonal<T: Diagonal>(&self, diag: &T) -> Self {
        let mut rows = [[0f64; N]; N];
        for (i, row) in rows.iter_mut().enumerate().take(N) {
            for j in 0..N {
                row[j] = self.rows[i][j] * diag[j];
            }
        }
        Self { rows }
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
/**Compute the elementwise negation of a [`Matrix`]*/
impl<const N: usize, const M: usize> Neg for Matrix<N, M> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            rows: self.rows.map(|r| r.map(|x| -x)),
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
/**Compute the elementwise negation of a [`DiagonalMatrix`]*/
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
impl<const N: usize> Add<Self> for DiagonalMatrix<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            rows: std::array::from_fn(|i| self[i] + rhs[i]),
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

impl SVD for Matrix<2, 2> {
    type SingularValues = DiagonalMatrix<2>;
    /** Decompose a [`Matrix22`] into a rotation `U`, a scaling`Σ`, and a second rotation`V` such that `A=UΣV`.

    This implementation is based on the math in 10.1109/38.486688, and ensures good
    numerical stability.

    We define all singular values to be positive. If the determinant of the input matrix
    is positive, the determinants of both U and V are also positive. If the determinant
    of the input matrix is negative, we ensure the determinant of U is positive and V
    is negative.
    */
    #[inline]
    fn svd(&self) -> (Self, Self::SingularValues, Self) {
        let a_plus_d = f64::midpoint(self[(0, 0)], self[(1, 1)]);
        let a_minus_d = (self[(0, 0)] - self[(1, 1)]) / 2.0;
        let b_plus_c = f64::midpoint(self[(0, 1)], self[(1, 0)]);
        let b_minus_c = (self[(0, 1)] - self[(1, 0)]) / 2.0;

        let (q, r) = (
            (a_plus_d.powi(2) + b_minus_c.powi(2)).sqrt(),
            (a_minus_d.powi(2) + b_plus_c.powi(2)).sqrt(),
        );

        let mut q_minus_r = q - r;
        println!("q-r: {q_minus_r}");
        println!("a.det(): {}", self.det());

        let (a1, a2) = (
            f64::atan2(b_plus_c, a_minus_d),
            f64::atan2(b_minus_c, a_plus_d),
        );

        let gamma = f64::midpoint(a1, a2);
        let beta = (a2 - a1) / 2.0;

        let u = Matrix22::from(RotationMatrix::from(Angle::from(beta)));
        let v = Matrix22::from(RotationMatrix::from(Angle::from(gamma)));
        // println!("u.det(): {}", u.det());

        // println!("v.det(): {}", v.det());

        // if u.det() < 0.0 {
        //     u[(1, 0)] *= -1.0;
        //     u[(1, 1)] *= -1.0;
        // }
        // if q_minus_r < 0.0 {
        //     v[(0, 1)] *= -1.0;
        //     v[(1, 1)] *= -1.0;
        //     q_minus_r *= -1.0;
        // }
        #[expect(non_snake_case, reason = "convention")]
        let Σ = Self::SingularValues {
            rows: [q + r, q_minus_r.abs()], // TODO: positive sy? get det from this
        };
        println!("sigma: \n{Σ:?}");
        // TODO: should we swap the sign of v

        (u, Σ, v)
    }
}

impl<const N: usize, const M: usize> fmt::Display for Matrix<N, M> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.rows
                .map(|row| Cartesian::<M>::from(row).to_string())
                .into_iter()
                .collect::<Vec<String>>()
                .join("\n ")
        )
    }
}

impl Copy for Matrix<2, 2> {}
impl Copy for Matrix<3, 3> {}
impl Copy for Matrix<4, 4> {}
impl<const N: usize> Diagonal for DiagonalMatrix<N> {}
impl<const N: usize> Diagonal for [f64; N] {}
impl<const N: usize> Diagonal for Cartesian<N> {}

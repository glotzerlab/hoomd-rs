//! General-purpose linear algebra functions on slices.

use super::{GeneralMatrix, Matrix};
/// .
pub(super) fn qr_decomposition<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
) -> (Matrix<M, M>, Matrix<N, M>) {
    // Return default TODO;
    (Matrix::<M, M>::zeros(), Matrix::<N, M>::zeros())
}
/// Performs a general row vector-matrix product `y = x^T * A`, writing the result to `y`.
///
/// - `a`: The matrix `A`, represented as an iterator of row-slices.
/// - `x`: The row vector `x`, represented as a slice.
/// - `y`: The mutable output slice to write the result vector to.
pub fn gemv_row_slice<'a, I>(a: I, x: &[f64], y: &mut [f64])
where
    I: Iterator<Item = &'a [f64]> + Clone,
{
    let a_rows = a.clone().count();
    let a_cols = y.len();

    assert_eq!(
        a_rows,
        x.len(),
        "Matrix and row vector dimensions are incompatible"
    );

    for j in 0..a_cols {
        y[j] = a.clone().zip(x.iter()).map(|(row, x_i)| x_i * row[j]).sum();
    }
}

/// Performs a general matrix-column vector product `y = A * x`, writing the result to `y`.
///
/// - `a`: The matrix `A`, represented as an iterator of row-slices.
/// - `x`: The column vector `x`, represented as a slice.
/// - `y`: The mutable output slice to write the result vector to.
pub fn gemv_col_slice<'a, M, I>(a: M, x: I, y: &mut [f64])
where
    M: Iterator<Item = &'a [f64]> + Clone,
    I: Iterator<Item = f64> + Clone,
{
    let a_rows = a.clone().count();
    assert_eq!(a_rows, y.len(), "Output slice has incorrect length");

    for (i, row_slice) in a.enumerate() {
        if i == 0 {
            assert_eq!(
                row_slice.len(),
                x.len(),
                "Matrix and column vector dimensions are incompatible"
            );
        }
        y[i] = row_slice
            .iter()
            .zip(x.iter())
            .map(|(a_ij, x_j)| a_ij * x_j)
            .sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[test]
    fn test_gemv_col_slice() {
        let a = Matrix::<2, 3> {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        };
        let a_iter = a.submatrix_slice_iter(0..2, 0..3);
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 2];
        gemv_col_slice(a_iter, &x, &mut y);
        assert_eq!(y, vec![14.0, 32.0]);
    }

    #[test]
    fn test_gemv_row_slice() {
        let a = Matrix::<2, 3> {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        };
        let a_iter = a.submatrix_slice_iter(0..2, 0..3);
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 3];
        gemv_row_slice(a_iter, &x, &mut y);
        assert_eq!(y, vec![9.0, 12.0, 15.0]);
    }

    #[rstest]
    #[case(Matrix::<3, 3>::zeros().rows)]
    #[case(Matrix::<4, 4>::zeros().rows)]
    #[case(Matrix::<5, 5>::zeros().rows)]
    fn test_gemv_submatrix_col_vector<const N: usize>(#[case] mut rows: [[f64; N]; N]) {
        for i in 0..N {
            for j in 0..N {
                rows[i][j] = (i * N + j + 1) as f64;
            }
        }
        let a = Matrix::<N, N> { rows };

        // column vector x = A[1..N, 1]
        let x = a.get_col_slice_iter(1, 1..N);

        // submatrix B = A[1..N, 1..N]
        let b_iter = a.submatrix_slice_iter(1..N, 1..N);

        // y = B * x
        let mut y = vec![0.0; N - 1];
        gemv_col_slice(b_iter, x, &mut y);

        // Calculate expected result
        let mut expected_y = vec![0.0; N - 1];
        for i in 0..(N - 1) {
            let mut sum = 0.0;
            for j in 0..(N - 1) {
                let b_ij = a.rows[i + 1][j + 1];
                let x_j = a.rows[j + 1][1];
                sum += b_ij * x_j;
            }
            expected_y[i] = sum;
        }

        assert_eq!(y, expected_y);
    }
}

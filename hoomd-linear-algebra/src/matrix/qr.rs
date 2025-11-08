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
pub fn gemv_row_slice<'a, I>(a: &I, x: &[f64], y: &mut [f64])
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
pub fn gemv_col_slice<'a, M, I, O>(a: M, x: &I, y: O)
where
    M: ExactSizeIterator<Item = &'a [f64]> + Clone,
    I: ExactSizeIterator<Item = f64> + Clone,
    O: ExactSizeIterator<Item = &'a mut f64>,
{
    assert_eq!(a.len(), y.len(), "Output iterator has incorrect length");
    if let Some(first_row) = a.clone().next() {
        assert_eq!(
            first_row.len(),
            x.len(),
            "Matrix and column vector dimensions are incompatible"
        );
    }

    for (y_i, row_slice) in y.zip(a) {
        *y_i = row_slice
            .iter()
            .zip(x.clone())
            .map(|(a_ij, x_j)| a_ij * x_j)
            .sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    // #[test]
    // fn test_gemv_col_slice() {
    //     let a = Matrix::<2, 3> {
    //         rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    //     };
    //     let a_iter = a.submatrix_slice_iter(0..2, 0..3);
    //     let x = a.iter_flat().take(3);
    //     let mut y = vec![0.0; 2];
    //     gemv_col_slice(a_iter, &x, &mut y);
    //     assert_eq!(y, vec![14.0, 32.0]);
    // }

    #[test]
    fn test_gemv_row_slice() {
        let a = Matrix::<2, 3> {
            rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        };
        let a_iter = a.submatrix_slice_iter(0..2, 0..3);
        let x = &a[(0, 0..2)];
        let mut y = vec![0.0; 3];
        gemv_row_slice(&a_iter, x, &mut y);
        assert_eq!(y, vec![9.0, 12.0, 15.0]);
    }

    #[rstest]
    #[case(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        vec![73.0, 112.0]
    )]
    #[case(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0]
        ],
        vec![218.0, 338.0, 458.0]
    )]
    #[case(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0, 25.0]
        ],
        vec![518.0, 808.0, 1098.0, 1388.0]
    )]
    fn test_gemv_submatrix_col_vector<const N: usize>(
        #[case] rows: [[f64; N]; N],
        #[case] expected_y: Vec<f64>,
    ) {
        let mut a = Matrix::<N, N> { rows };

        // column vector x = A[1..N, 1]
        let x = a.get_col_slice_iter(1, 1..N);

        // submatrix B = A[1..N, 1..N]
        let b_iter = a.submatrix_slice_iter(1..N, 1..N);

        // y = B * x, where y is the first column of A (from row 1)
        let y_iter = a.get_col_slice_iter_mut(0, 1..N);
        gemv_col_slice(b_iter, &x, y_iter);

        // Check if the column was updated correctly
        let result_col: Vec<f64> = a.get_col_slice_iter(0, 1..N).collect();
        assert_eq!(result_col, expected_y);
    }
}

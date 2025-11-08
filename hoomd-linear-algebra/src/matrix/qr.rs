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
pub fn gemv_sub_col_into_col<const N: usize>(
    a: &mut Matrix<N, N>,
    b_rows: std::ops::Range<usize>,
    b_cols: std::ops::Range<usize>,
    x_rows: std::ops::Range<usize>,
    x_col: usize,
    y_rows: std::ops::Range<usize>,
    y_col: usize,
) {
    // Safety precondition: The output column `y` must not overlap with any
    // of the input columns from `B` or `x`. This is crucial because we are
    // reading from the inputs while writing to the output in the same loop.
    assert_ne!(
        y_col, x_col,
        "Output column cannot be the same as the input vector column."
    );
    assert!(
        !b_cols.contains(&y_col),
        "Output column cannot be within the input matrix columns."
    );

    // SAFETY: We use a raw pointer to the matrix data to bypass the borrow checker.
    // This is safe because we have asserted that the read and write regions
    // do not overlap, preventing data races. The pointer `matrix_ptr` is
    // derived from a valid mutable reference `a` and is only used within this
    // function, so its lifetime is valid.
    let matrix_ptr = a.rows.as_mut_ptr();

    for i in 0..y_rows.len() {
        let b_row_idx = b_rows.start + i;
        let y_row_idx = y_rows.start + i;

        let mut sum = 0.0;
        for j in 0..b_cols.len() {
            let b_col_idx = b_cols.start + j;
            let x_row_idx = x_rows.start + j;

            // SAFETY: We are reading from the matrix using raw pointers.
            // This is safe because:
            // 1. `matrix_ptr` is valid.
            // 2. The indices are within the bounds of the matrix (this is
            //    implicitly trusted by using ranges, but can be asserted for
            //    extra safety if needed).
            // 3. We've asserted that the read locations do not alias the
            //    write location for this loop iteration.
            unsafe {
                let b_ij = (*matrix_ptr.add(b_row_idx))[b_col_idx];
                let x_j = (*matrix_ptr.add(x_row_idx))[x_col];
                sum += b_ij * x_j;
            }
        }

        // SAFETY: We are writing to the matrix using a raw pointer.
        // This is safe because we've asserted that the output column `y_col`
        // does not overlap with any input columns.
        unsafe {
            (*matrix_ptr.add(y_row_idx))[y_col] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

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

        gemv_sub_col_into_col(
            &mut a,
            1..N, // b_rows
            1..N, // b_cols
            1..N, // x_rows
            1,    // x_col
            1..N, // y_rows
            0,    // y_col
        );

        let result_col: Vec<f64> = a.get_col_slice_iter(0, 1..N).collect();
        assert_eq!(result_col, expected_y);
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use super::Matrix;

/// Performs an in-place matrix-vector multiplication using sub-regions of a matrix.
///
/// This function computes `y = B * x`, where `B` is a submatrix of `a`, `x` is a
/// column vector from `a`, and `y` is a column of `a` that gets overwritten
/// with the result.
///
/// This safe version uses intermediate buffers to store input and output data,
/// satisfying the borrow checker at the cost of extra allocations.
///
/// # Panics
///
/// This function will panic if:
/// - Any of the provided ranges are out of bounds for the matrix `a`.
/// - The dimensions of `B`, `x`, and `y` are not compatible for multiplication.
#[inline]
pub fn gemv_submatrix_column_into_column<const N: usize>(
    a: &mut Matrix<N, N>,
    b_rows: std::ops::Range<usize>,
    b_cols: std::ops::Range<usize>,
    x_rows: std::ops::Range<usize>,
    x_col: usize,
    y_rows: std::ops::Range<usize>,
    y_col: usize,
) {
    assert!(
        b_rows.end <= N && b_cols.end <= N,
        "Input matrix B is out of bounds."
    );
    assert!(
        x_rows.end <= N && x_col < N,
        "Input vector x is out of bounds."
    );
    assert!(
        y_rows.end <= N && y_col < N,
        "Output vector y is out of bounds."
    );
    assert_eq!(
        b_cols.len(),
        x_rows.len(),
        "Incompatible dimensions between B and x."
    );
    assert_eq!(
        b_rows.len(),
        y_rows.len(),
        "Incompatible dimensions between B and y."
    );

    let x_data: Vec<f64> = a.get_col_slice_iter(x_col, x_rows).collect();

    let y_data: Vec<f64> = a
        .submatrix_slice_iter(b_rows, b_cols)
        .map(|b_row| {
            b_row
                .iter()
                .zip(x_data.iter())
                .map(|(b_ij, x_j)| b_ij * x_j)
                .sum()
        })
        .collect();

    let y_iter = a.get_col_slice_iter_mut(y_col, y_rows);
    for (y_i, result) in y_iter.zip(y_data) {
        *y_i = result;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GeneralMatrix;
    use rstest::*;

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

        gemv_submatrix_column_into_column_unsafe(
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

    #[test]
    #[should_panic(expected = "Input vector x is out of bounds.")]
    fn test_gemv_out_of_bounds_x() {
        let mut a = Matrix::<3, 3>::zeros();
        gemv_submatrix_column_into_column_unsafe(
            &mut a,
            0..2, // b_rows
            0..2, // b_cols
            0..2, // x_rows
            3,    // x_col (out of bounds)
            0..2, // y_rows
            2,    // y_col
        );
    }

    #[test]
    #[should_panic(expected = "Output vector y is out of bounds.")]
    fn test_gemv_out_of_bounds_y() {
        let mut a = Matrix::<3, 3>::zeros();
        gemv_submatrix_column_into_column_unsafe(
            &mut a,
            0..2, // b_rows
            0..2, // b_cols
            0..2, // x_rows
            1,    // x_col
            0..4, // y_rows (out of bounds)
            2,    // y_col
        );
    }

    #[test]
    #[should_panic(expected = "Incompatible dimensions between B and x.")]
    fn test_gemv_incompatible_dims_b_x() {
        let mut a = Matrix::<3, 3>::zeros();
        gemv_submatrix_column_into_column_unsafe(
            &mut a,
            0..2, // b_rows
            0..2, // b_cols (len 2)
            0..1, // x_rows (len 1)
            1,    // x_col
            0..2, // y_rows
            2,    // y_col
        );
    }

    #[test]
    #[should_panic(expected = "Incompatible dimensions between B and y.")]
    fn test_gemv_incompatible_dims_b_y() {
        let mut a = Matrix::<3, 3>::zeros();
        gemv_submatrix_column_into_column_unsafe(
            &mut a,
            0..2, // b_rows (len 2)
            0..2, // b_cols
            0..2, // x_rows
            1,    // x_col
            0..1, // y_rows (len 1)
            2,    // y_col
        );
    }
}

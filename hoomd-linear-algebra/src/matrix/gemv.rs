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

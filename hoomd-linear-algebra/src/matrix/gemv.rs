// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use super::{GeneralMatrix, Matrix};

// /// Performs a general row vector-matrix product `y = x^T * A`, writing the result to `y`.
// ///
// /// - `a`: The matrix `A`, represented as an iterator of row-slices.
// /// - `x`: The row vector `x`, represented as a slice.
// /// - `y`: The mutable output slice to write the result vector to.
// #[inline]
// pub(super) fn gemv_row_slice<'a, I>(a: &I, x: &[f64], y: &mut [f64])
// where
//     I: Iterator<Item = &'a mut [f64]> + Clone,
// {
//     let a_cols = y.len();

//     for j in 0..a_cols {
//         y[j] = a.clone().zip(x.iter()).map(|(row, x_i)| x_i * row[j]).sum();
//     }
// }

/// Performs an in-place matrix-vector multiplication using sub-regions of a matrix.
///
/// This function computes `y = B * x`, where `B` is a submatrix of `a`, `x` is a
/// column vector from `a`, and `y` is a column of `a` that gets overwritten
/// with the result.
///
/// # Panics
///
/// This function will panic if:
/// - Any of the provided ranges are out of bounds for the matrix `a`.
/// - The dimensions of `B`, `x`, and `y` are not compatible for multiplication.
/// - The output column `y` overlaps with any of the input columns from `B` or `x`.
///
#[inline]
pub(super) fn gemv_submatrix_column_into_column<const N: usize>(
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
    assert_ne!(
        y_col, x_col,
        "Output column cannot be the same as the input vector column."
    );
    assert!(
        !b_cols.contains(&y_col),
        "Output column cannot be within the input matrix columns."
    );

    for (i, b_row_idx) in b_rows.clone().enumerate() {
        let y_row_idx = y_rows.start + i;
        // let mut sum = 0.0;
        // for (j, b_col_idx) in b_cols.clone().enumerate() {
        //     let x_row_idx = x_rows.start + j;
        //     sum += a[(b_row_idx, b_col_idx)] * a[(x_row_idx, x_col)];
        // }
        let sum = b_cols
            .clone()
            .zip(x_rows.clone())
            .map(|(b_col_idx, x_row_idx)| a[(b_row_idx, b_col_idx)] * a[(x_row_idx, x_col)])
            .sum();
        a[(y_row_idx, y_col)] = sum;
    }
}

// pub(super) fn gemv_submatrix_column_into_column<const N: usize>(
//     a: &mut Matrix<N, N>,
//     b_rows: std::ops::Range<usize>,
//     b_cols: std::ops::Range<usize>,
//     x_rows: std::ops::Range<usize>,
//     x_col: usize,
//     y_rows: std::ops::Range<usize>,
//     y_col: usize,
// ) {
//     // Precondition checks for memory safety and correctness.
//     assert!(
//         b_rows.end <= N && b_cols.end <= N,
//         "Input matrix B is out of bounds."
//     );
//     assert!(
//         x_rows.end <= N && x_col < N,
//         "Input vector x is out of bounds."
//     );
//     assert!(
//         y_rows.end <= N && y_col < N,
//         "Output vector y is out of bounds."
//     );
//     assert_eq!(
//         b_cols.len(),
//         x_rows.len(),
//         "Incompatible dimensions between B and x."
//     );
//     assert_eq!(
//         b_rows.len(),
//         y_rows.len(),
//         "Incompatible dimensions between B and y."
//     );

//     // Safety precondition: The output column `y` must not overlap with any
//     // of the input columns from `B` or `x`.
//     assert_ne!(
//         y_col, x_col,
//         "Output column cannot be the same as the input vector column."
//     );
//     assert!(
//         !b_cols.contains(&y_col),
//         "Output column cannot be within the input matrix columns."
//     );

//     // SAFETY: We use a raw pointer to the matrix data so that we can read and write
//     // from the same matrix. This is safe because we have asserted that the read and
//     // write regions do not overlap. The pointer `matrix_ptr` is
//     // derived from a valid mutable reference `a` and is only used within this
//     // function, so its lifetime is valid.
//     let matrix_ptr = a.rows.as_mut_ptr();

//     for i in 0..y_rows.len() {
//         let b_row_idx = b_rows.start + i;
//         let y_row_idx = y_rows.start + i;

//         let mut sum = 0.0;
//         for j in 0..b_cols.len() {
//             let b_col_idx = b_cols.start + j;
//             let x_row_idx = x_rows.start + j;

//             // SAFETY: We are reading from the matrix using raw pointers.
//             // This is safe because:
//             // 1. `matrix_ptr` is valid.
//             // 2. The top-level assertions guarantee the indices are in-bounds.
//             // 3. We've asserted that the read locations do not alias the
//             //    write location for this loop iteration.
//             unsafe {
//                 debug_assert!(b_row_idx < N);
//                 debug_assert!(b_col_idx < N);
//                 debug_assert!(x_row_idx < N);
//                 let b_ij = (*matrix_ptr.add(b_row_idx))[b_col_idx];
//                 let x_j = (*matrix_ptr.add(x_row_idx))[x_col];
//                 sum += b_ij * x_j;
//             }
//         }

//         // SAFETY: We are writing to the matrix using a raw pointer.
//         // This is safe because we've asserted that the output column `y_col`
//         // does not overlap with any input columns.
//         unsafe {
//             debug_assert!(y_row_idx < N);
//             (*matrix_ptr.add(y_row_idx))[y_col] = sum;
//         }
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    // #[test]
    // fn test_gemv_row_slice() {
    //     let mut a = Matrix::<2, 3> {
    //         rows: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    //     };
    //     let a_iter = a.submatrix_slice_iter_mut(0..2, 0..3);
    //     let x = &a[(0, 0..2)];
    //     let mut y = vec![0.0; 3];
    //     gemv_row_slice(&a_iter, x, &mut a[(1, 0..3)]);
    //     assert_eq!(y, vec![9.0, 12.0, 15.0]);
    // }

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

        gemv_submatrix_column_into_column(
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
        gemv_submatrix_column_into_column(
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
        gemv_submatrix_column_into_column(
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
        gemv_submatrix_column_into_column(
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
        gemv_submatrix_column_into_column(
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

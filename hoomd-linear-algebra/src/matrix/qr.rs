//! General-purpose linear algebra functions on slices.

use super::{GeneralMatrix, Matrix};
pub(super) fn qr_decomposition<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
) -> (Matrix<M, M>, Matrix<N, M>) {
    // Return default TODO;
    (Matrix::<M, M>::zeros(), Matrix::<N, M>::zeros())
}
/// Performs a general row vector-matrix product `y = x^T * A`, writing the result to `y`.
///
/// - `a`: The matrix `A`, represented as a slice of row-slices.
/// - `x`: The row vector `x`, represented as a slice.
/// - `y`: The mutable output slice to write the result vector to.
pub fn gemv_row_slice(a: &[&[f64]], x: &[f64], y: &mut [f64]) {
    let a_rows = a.len();
    let a_cols = a.get(0).map_or(0, |row| row.len());

    assert_eq!(
        a_rows,
        x.len(),
        "Matrix and row vector dimensions are incompatible"
    );
    assert_eq!(a_cols, y.len(), "Output slice has incorrect length");

    for j in 0..a_cols {
        y[j] = (0..a_rows).map(|i| x[i] * a[i][j]).sum();
    }
}

/// Performs a general matrix-column vector product `y = A * x`, writing the result to `y`.
///
/// - `a`: The matrix `A`, represented as a slice of row-slices.
/// - `x`: The column vector `x`, represented as a slice.
/// - `y`: The mutable output slice to write the result vector to.
pub fn gemv_col_slice(a: &[&[f64]], x: &[f64], y: &mut [f64]) {
    let a_rows = a.len();
    let a_cols = a.get(0).map_or(0, |row| row.len());

    assert_eq!(
        a_cols,
        x.len(),
        "Matrix and column vector dimensions are incompatible"
    );
    assert_eq!(a_rows, y.len(), "Output slice has incorrect length");

    for i in 0..a_rows {
        y[i] = (0..a_cols).map(|j| a[i][j] * x[j]).sum();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemv_col_slice() {
        let a_slices: Vec<&[f64]> = vec![&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]];
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 2];
        gemv_col_slice(&a_slices, &x, &mut y);
        assert_eq!(y, vec![14.0, 32.0]);
    }

    #[test]
    fn test_gemv_row_slice() {
        let a_slices: Vec<&[f64]> = vec![&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]];
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 3];
        gemv_row_slice(&a_slices, &x, &mut y);
        assert_eq!(y, vec![9.0, 12.0, 15.0]);
    }
}

//! General-purpose linear algebra functions on slices.
use super::{GeneralMatrix, Matrix};

/// Performs a general row vector-matrix product `y = x^T * A`.
///
/// - `a`: The matrix `A`, represented as a slice of row-slices.
/// - `x`: The row vector `x`, represented as a slice.
///
/// Returns the result vector `y` as a `Vec<f64>`.
pub fn gemv_row(a: &[&[f64]], x: &[f64]) -> Vec<f64> {
    let a_rows = a.len();
    if a_rows == 0 {
        return Vec::new();
    }
    let a_cols = a.get(0).map_or(0, |row| row.len());
    if a_cols == 0 {
        return vec![0.0; 0];
    }

    assert_eq!(
        a_rows,
        x.len(),
        "Matrix and row vector dimensions are incompatible"
    );
    let mut y = vec![0.0; a_cols];
    for j in 0..a_cols {
        for i in 0..a_rows {
            y[j] += x[i] * a[i][j];
        }
    }
    y
}

/// Performs a general matrix-column vector product `y = A * x`.
///
/// - `a`: The matrix `A`, represented as a slice of row-slices.
/// - `x`: The column vector `x`, represented as a slice.
///
/// Returns the result vector `y` as a `Vec<f64>`.
pub fn gemv_col(a: &[&[f64]], x: &[f64]) -> Vec<f64> {
    let a_rows = a.len();
    if a_rows == 0 {
        return Vec::new();
    }
    let a_cols = a.get(0).map_or(0, |row| row.len());
    if a_cols == 0 {
        return vec![0.0; a_rows];
    }

    assert_eq!(
        a_cols,
        x.len(),
        "Matrix and column vector dimensions are incompatible"
    );
    let mut y = vec![0.0; a_rows];
    for i in 0..a_rows {
        for j in 0..a_cols {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemv_col() {
        let a_slices: Vec<&[f64]> = vec![&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]];
        let x = vec![1.0, 2.0, 3.0];
        // A (2x3) * x (3x1) -> y (2x1)
        let y = gemv_col(&a_slices, &x);
        // 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        // 4*1 + 5*2 + 6*3 = 4 + 10 + 18 = 32
        assert_eq!(y, vec![14.0, 32.0]);
    }

    #[test]
    fn test_gemv_row() {
        let a_slices: Vec<&[f64]> = vec![&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]];
        let x = vec![1.0, 2.0];
        // x (1x2) * A (2x3) -> y (1x3)
        let y = gemv_row(&a_slices, &x);
        // col 0: 1*1 + 2*4 = 1 + 8 = 9
        // col 1: 1*2 + 2*5 = 2 + 10 = 12
        // col 2: 1*3 + 2*6 = 3 + 12 = 15
        assert_eq!(y, vec![9.0, 12.0, 15.0]);
    }
}
pub(super) fn qr_decomposition<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
) -> (Matrix<M, M>, Matrix<N, M>) {
    // Return default TODO;
    (Matrix::<M, M>::zeros(), Matrix::<N, M>::zeros())
}

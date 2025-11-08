// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! General-purpose linear algebra functions on slices.

use super::gemv::gemv_submatrix_column_into_column;
use super::{GeneralMatrix, Matrix};

/// .
pub(super) fn qr_decomposition<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
) -> (Matrix<M, M>, Matrix<N, M>) {
    // Return default TODO;
    (Matrix::<M, M>::zeros(), Matrix::<N, M>::zeros())
}

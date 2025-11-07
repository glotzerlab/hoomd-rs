use super::{GeneralMatrix, Matrix};

/// Compute the QR decomposition of a [`Matrix`] $`a`$
pub(super) fn qr_decomposition<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
) -> (Matrix<M, M>, Matrix<N, M>) {
    // Return default TODO;
    (Matrix::<M, M>::zeros(), Matrix::<N, M>::zeros())
}

use crate::matrix::Matrix;

///
///
/// Implementation based on <https://www.sfu.ca/~jtmulhol/py4math/linalg/np-gramschmidt/>
#[must_use]
#[inline]
pub fn gram_schmidt<const N: usize, const M: usize>(a: &Matrix<N, M>) -> Matrix<N, M> {
    let mut a = a.clone();
    for j in 0..a.n_columns() {
        // For the vector in column k, find the perpendicular of the projection onto
        // the previous orthogonal vectors.
        for k in (0..j) {
            let j_dot_k = a
                .get_col_slice_iter(k, 0..N)
                .zip(a.get_col_slice_iter(j, 0..N))
                .fold(0.0, |acc, (l, r)| acc + (l * r));
            for i in 0..N {
                a[(i, j)] -= a[(k, i)] * j_dot_k;
            }
//         # If original vectors aren't lin indep then we can check for this:
//         #
//         if np.isclose(np.linalg.norm(A[:, j]), 0, rtol=1e-15, atol=1e-14, equal_nan=False):
//             A[:, j] = np.zeros(A.shape[0])
//         else:
//             A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
            let column_j_norm = a
                .get_col(j).iter_elements().fold(0.0, |acc, x| acc + x*x).sqrt();
            if column_j_norm.is_finite() {
                a.get_col_slice_iter_mut(j, 0..N).for_each(|x|*x /= column_j_norm);
            }
             else {
                 
             }
    }
    a
}

// def gram_schmidt(A):
//     for j in range(n):
//         # For the vector in column j, find the perpendicular
//         # of the projection onto the previous orthogonal vectors.
//         for k in range(j):
//             A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
//         # If original vectors aren't lin indep then we can check for this:
//         #
//         if np.isclose(np.linalg.norm(A[:, j]), 0, rtol=1e-15, atol=1e-14, equal_nan=False):
//             A[:, j] = np.zeros(A.shape[0])
//         else:
//             A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
//     return A

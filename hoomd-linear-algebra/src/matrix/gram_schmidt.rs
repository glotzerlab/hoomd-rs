use crate::matrix::Matrix;

///
///
/// Implementation based on https://www.sfu.ca/~jtmulhol/py4math/linalg/np-gramschmidt/
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
            let proj_j_onto_k = a.get_col_slice_iter(k, 0..N).map(|x| x * j_dot_k);
            a.get_col_slice_iter_mut(j, 0..N)
                .zip(proj_j_onto_k)
                .map(|(a_ji, proj_i)| *a_ji -= proj_i);
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

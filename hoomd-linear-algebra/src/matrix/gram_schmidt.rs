use crate::matrix::Matrix;

/// Tolerance for norms below which a vector is considered to be the zero vector in Gram-Schmidt.
pub const GRAM_SCHMIDT_EPSILON: f64 = 1e-12;

/// Construct an orthonormal basis from the vectors in a [`Matrix`] using the modified Gram-Schmidt procedure.
///
/// Implementation based on <https://www.sfu.ca/~jtmulhol/py4math/linalg/np-gramschmidt/>
#[must_use]
#[inline]
pub fn gram_schmidt<const N: usize, const M: usize>(a: &Matrix<N, M>) -> Matrix<N, M> {
    let mut a = a.clone();
    for j in 0..a.n_columns() {
        // For the vector in column k, find the perpendicular of the projection onto
        // the previous orthogonal vectors.
        for k in 0..j {
            let j_dot_k = a
                .get_col(k)
                .iter_elements()
                .zip(a.get_col(j).iter_elements())
                .map(|(l, r)| l * r)
                .sum::<f64>();
            // Apply the projection
            for i in 0..N {
                a[(i, j)] -= a[(i, k)] * j_dot_k;
            }
        } // end loop over k
        let column_j_norm = a
            .get_col(j)
            .iter_elements()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        // If the initial vectors are not linearly independent, zero out the column
        if column_j_norm > GRAM_SCHMIDT_EPSILON {
            a.get_col_slice_iter_mut(j, 0..N)
                .for_each(|x| *x /= column_j_norm);
        } else {
            a.get_col_slice_iter_mut(j, 0..N).for_each(|x| *x = 0.0);
        }
    } // end loop over j
    a
}

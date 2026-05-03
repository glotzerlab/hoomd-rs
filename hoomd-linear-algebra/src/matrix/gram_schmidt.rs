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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Matrix;
    use crate::matrix::test_utils::assert_matrixes_ulps_eq;

    #[test]
    fn test_gram_schmidt_3x3() {
        // Example 1 from https://www.sfu.ca/~jtmulhol/py4math/linalg/np-gramschmidt/
        let a = Matrix::<3, 3> {
            rows: [[1.0, -1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 1.0]],
        };
        let q = gram_schmidt(&a);

        let sqrt2 = 2.0f64.sqrt();
        let sqrt11 = 11.0f64.sqrt();
        let sqrt22 = 22.0f64.sqrt();

        let expected = Matrix::<3, 3> {
            rows: [
                [1.0 / sqrt2, -3.0 / sqrt22, 1.0 / sqrt11],
                [1.0 / sqrt2, 3.0 / sqrt22, -1.0 / sqrt11],
                [0.0, 2.0 / sqrt22, 3.0 / sqrt11],
            ],
        };

        assert_matrixes_ulps_eq::<3, 3, _, _>(&q, &expected);
    }

    #[test]
    fn test_gram_schmidt_linearly_dependent() {
        // Example 2 from https://www.sfu.ca/~jtmulhol/py4math/linalg/np-gramschmidt/
        // A = [[1, 1, 2, 1], [1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 2, 1]] (columns)
        let a = Matrix::<4, 4> {
            rows: [
                [1.0, 1.0, 2.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 2.0, 1.0],
            ],
        };
        let q = gram_schmidt(&a);

        let sqrt3 = 3.0f64.sqrt();
        let sqrt15 = 15.0f64.sqrt();
        let sqrt10 = 10.0f64.sqrt();

        let expected = Matrix::<4, 4> {
            rows: [
                [1.0 / sqrt3, 1.0 / sqrt15, 0.0, 1.0 / sqrt10],
                [1.0 / sqrt3, -2.0 / sqrt15, 0.0, -2.0 / sqrt10],
                [0.0, 3.0 / sqrt15, 0.0, -2.0 / sqrt10],
                [1.0 / sqrt3, 1.0 / sqrt15, 0.0, 1.0 / sqrt10],
            ],
        };

        assert_matrixes_ulps_eq::<4, 4, _, _>(&q, &expected);
    }

    #[test]
    fn test_gram_schmidt_subspace() {
        // Example 3 from https://www.sfu.ca/~jtmulhol/py4math/linalg/np-gramschmidt/
        let a = Matrix::<3, 2> {
            rows: [[1.0, -1.0], [1.0, 2.0], [0.0, 1.0]],
        };
        let q = gram_schmidt(&a);

        let sqrt2 = 2.0f64.sqrt();
        let sqrt22 = 22.0f64.sqrt();

        let expected = Matrix::<3, 2> {
            rows: [
                [1.0 / sqrt2, -3.0 / sqrt22],
                [1.0 / sqrt2, 3.0 / sqrt22],
                [0.0, 2.0 / sqrt22],
            ],
        };

        assert_matrixes_ulps_eq::<3, 2, _, _>(&q, &expected);
    }
}

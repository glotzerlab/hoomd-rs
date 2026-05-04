// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

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
    for j in 0..M {
        // For the vector in column k, find the perpendicular of the projection onto
        // the previous orthogonal vectors.
        for k in 0..j {
            let mut j_dot_k = 0.0;
            for i in 0..N {
                j_dot_k += a[(i, k)] * a[(i, j)];
            }
            // Apply the projection
            for i in 0..N {
                a[(i, j)] -= a[(i, k)] * j_dot_k;
            }
        } // end loop over k

        let mut column_j_norm_sq = 0.0;
        for i in 0..N {
            column_j_norm_sq += a[(i, j)] * a[(i, j)];
        }
        let column_j_norm = column_j_norm_sq.sqrt();

        // If the initial vectors are not linearly independent, zero out the column
        if column_j_norm > GRAM_SCHMIDT_EPSILON {
            for i in 0..N {
                a[(i, j)] /= column_j_norm;
            }
        } else {
            for i in 0..N {
                a[(i, j)] = 0.0;
            }
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

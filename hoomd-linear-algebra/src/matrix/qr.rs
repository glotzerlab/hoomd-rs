// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use super::Matrix;
use crate::{GeneralMatrix, SquareMatrix};
use std::cmp::min;

/// Compute the QR decomposition of a matrix using Householder reflections.
///
/// # QR Decomposition
///
/// Factors a matrix `A` ($`N \times M`$), $`N \ge M`$, into an orthogonal matrix $`Q`$ ($`N \times N`$) and an
/// upper triangular matrix $`R`$ ($`N \times M`$) such that $`A = Q R`$. The columns of
/// $`Q`$ form an orthonormal basis for the column space of $`A`$, and $`R`$ encodes
/// the change of basis from $`A`$'s columns into that orthonormal basis.
///
/// ## Algorithm
///
/// The implementation is inspired by the LAPACK routines `DGEQR2` and
/// `DLARFG`. It uses successive **Householder reflections** to reduce $`A`$ to
/// upper triangular form in place. A Householder reflection has the form
///
/// ```math
/// H = I - \tau v v^T, \qquad \tau = (\beta - \alpha) / \beta
/// ```
///
/// where $`v`$ is the reflection vector (normal to the reflection plane) with a
/// leading element normalized to 1, $`\alpha = `$ `qr[(i, i)]` is the pivot element,
/// and $`\beta = -\text{sgn}(\alpha) \|y\|`$ is the target value after reflection. The
/// sign convention ensures that $`\alpha`$ and $`\beta`$ have opposite signs,
/// avoiding catastrophic cancellation when computing $`\alpha - \beta`$.
///
/// At step $`i`$, the reflection $`H_i`$ acts on the submatrix `A[i..N, i..M]`,
/// zeroing out the subdiagonal entries of column $`i`$ while leaving rows and
/// columns $`0..i`$ unchanged. After $`\min(N, M)`$ steps:
///
/// ```math
/// H_{n-1} \cdots H_1 H_0 A = R
/// ```
///
/// so that $`Q = H_0 H_1 \cdots H_{n-1}`$ (since each $`H_i`$ is its own
/// inverse).
///
/// ## Packed Storage
///
/// Rather than forming $`Q`$ explicitly at each step, the reflection vectors are
/// stored in the subdiagonal entries of the working matrix as it is reduced.
/// After the decomposition, the output `qr` matrix contains:
///
/// - **Upper triangle** (including diagonal): the matrix $`R`$.
/// - **Subdiagonal entries of column $`i`$** (rows `i+1..N`): the stored part
///   of the reflection vector $`v_i`$, with the leading element (always 1)
///   implicit.
///
/// ```text
/// [ r  r  r ]
/// [ v0 r  r ]
/// [ v0 v1 r ]
/// [ v0 v1 v2]
/// ```
///
/// The corresponding `taus` vector holds the scalar $`\tau_i`$ for each step.
///
/// ## Applying Q and Q^T
///
/// Since forming $`Q`$ explicitly is often unnecessary and costly, this module
/// provides functions to multiply by $`Q`$ or $`Q^T`$ directly from the packed
/// representation. Each applies the sequence of reflections in the appropriate
/// order, exploiting the identity $`H_i^T = H_i`$:
///
/// | Function   | Computes      | `A` shape |
/// |------------|---------------|-----------|
/// | `q_times`  | $`Q A`$       | `N×K`     |
/// | `qt_times` | $`Q^T A`$     | `N×K`     |
/// | `times_q`  | $`A Q`$       | `K×N`     |
/// | `times_qt` | $`A Q^T`$     | `K×N`     |
///
/// If you need $`Q`$ as an explicit matrix (e.g. for inspection or testing),
/// use `get_q`, which applies the reflections to the $`N \times N`$ identity.
/// This is equivalent to calling `q_times` on the identity but is provided
/// for convenience.
///
/// ## Solving Linear Systems
///
/// `qr_solve` uses the packed decomposition to solve $`A x = b`$ for
/// overdetermined ($`N > M`$) systems in the least-squares sense. It applies
/// $`Q^T`$ to $`b`$ via `qt_times`, then solves the resulting upper triangular
/// system $`R x = Q^T b`$ by back substitution.
pub(super) fn qr_decomposition<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
) -> (Matrix<N, M>, [f64; M]) {
    let mut qr = *a;
    let mut taus = [0.0_f64; M];

    for i in 0..M {
        let mut tau = 0.0;
        let mut beta = 0.0;

        let x_norm_2 = qr
            .iter_column_slice(i, (i + 1)..N)
            .map(|x| x * x)
            .sum::<f64>();

        if x_norm_2 != 0.0 {
            let alpha = qr[(i, i)];
            beta = -alpha.signum() * (x_norm_2 + alpha * alpha).sqrt();
            // TODO: scale beta in extreme cases to avoid overflow
            tau = (beta - alpha) / beta;
            for j in (i + 1)..N {
                qr[(j, i)] /= alpha - beta;
            }
            // TODO: unscale beta
            qr[(i, i)] = 1.0; // Temporary; rescaled to beta below
        }

        if tau == 0.0 {
            // Either in the last column or the remainder of the column is zero.
            taus[i] = tau;
            continue;
        }

        // Collect the Householder vector v for this step (stored in column i, rows i..N).
        // Note: w_t is indexed from 0 but corresponds to columns (i+1)..M of qr.
        let v_col: Vec<f64> = qr.iter_column_slice(i, i..N).collect();

        // Compute w^T = C^T * v, where C = qr[i..N, (i+1)..M].
        let mut w_t = vec![0.0; M - i - 1];
        for (row_slice, &v_r) in qr.iter_submatrix(i..N, (i + 1)..M).zip(v_col.iter()) {
            for (j, &val) in row_slice.iter().enumerate() {
                w_t[j] += val * v_r;
            }
        }

        // Apply the rank-1 update: C -= tau * v * w^T.
        for (row_slice_mut, &v_r) in qr
            .iter_submatrix_mut(i..N, (i + 1)..M)
            .zip(v_col.iter())
        {
            for (j, cell) in row_slice_mut.iter_mut().enumerate() {
                *cell -= tau * v_r * w_t[j];
            }
        }

        qr[(i, i)] = beta;
        taus[i] = tau;
    }

    (qr, taus)
}

/// Apply a single Householder reflector from the left.
///
/// This updates `result[iter..N, 0..K]` with
/// `(I - tau * v * v^T) * result[iter..N, 0..K]`, where the Householder
/// vector `v` is encoded in `qr` at column `iter` with an implicit leading 1.
///
/// The helper is used to multiply by `Q` or `Q^T` without explicitly forming
/// the orthogonal matrix.
#[inline]
fn apply_householder_left<const N: usize, const M: usize, const K: usize>(
    result: &mut Matrix<N, K>,
    qr: &Matrix<N, M>,
    iter: usize,
    tau: f64,
) {
    let tail = (iter + 1)..N;

    // w^T = v^T * result[iter..N, 0..K]
    let mut w_t = vec![0.0; K];

    for (col, &val) in result.rows[iter].iter().enumerate() {
        w_t[col] += val; // leading element of v is 1
    }
    for (row_slice, v_r) in result
        .iter_submatrix(tail.clone(), 0..K)
        .zip(qr.iter_column_slice(iter, tail.clone()))
    {
        for (col, &val) in row_slice.iter().enumerate() {
            w_t[col] += val * v_r;
        }
    }

    // result[iter..N, 0..K] -= tau * v * w^T
    for (col, val) in result
        .iter_submatrix_mut(iter..iter + 1, 0..K)
        .next()
        .expect("submatrix must contain at least one row")
        .iter_mut()
        .enumerate()
    {
        *val -= tau * w_t[col]; // leading element of v is 1
    }
    for (row_slice_mut, v_r) in result
        .iter_submatrix_mut(tail.clone(), 0..K)
        .zip(qr.iter_column_slice(iter, tail.clone()))
    {
        for (col, val) in row_slice_mut.iter_mut().enumerate() {
            *val -= tau * v_r * w_t[col];
        }
    }
}

/// Apply a single Householder reflector from the right.
///
/// This updates `result[0..K, iter..N]` with
/// `result[0..K, iter..N] * (I - tau * v * v^T)`, where the Householder
/// vector `v` is encoded in `qr` at column `iter` with an implicit leading 1.
///
/// It is used to compute products with `Q` or `Q^T` when the orthogonal factor
/// appears on the right side of the multiplication.
#[inline]
fn apply_householder_right<const N: usize, const M: usize, const K: usize>(
    result: &mut Matrix<K, N>,
    qr: &Matrix<N, M>,
    iter: usize,
    tau: f64,
) {
    let tail = (iter + 1)..N;

    // w = result[0..K, iter..N] * v
    let mut w = vec![0.0; K];

    for (row, row_slice) in result
        .iter_submatrix(0..K, iter..iter + 1)
        .enumerate()
    {
        w[row] += row_slice[0]; // leading element of v is 1
    }
    for (row, row_slice) in result.iter_submatrix(0..K, tail.clone()).enumerate() {
        for (&val, v_r) in row_slice
            .iter()
            .zip(qr.iter_column_slice(iter, tail.clone()))
        {
            w[row] += val * v_r;
        }
    }

    // result[0..K, iter..N] -= tau * w * v^T
    for (row, row_slice_mut) in result
        .iter_submatrix_mut(0..K, iter..iter + 1)
        .enumerate()
    {
        row_slice_mut[0] -= tau * w[row]; // leading element of v is 1
    }
    for (row, row_slice_mut) in result
        .iter_submatrix_mut(0..K, tail.clone())
        .enumerate()
    {
        for (val, v_r) in row_slice_mut
            .iter_mut()
            .zip(qr.iter_column_slice(iter, tail.clone()))
        {
            *val -= tau * w[row] * v_r;
        }
    }
}

/// Extract the upper triangular factor `R` from a packed QR factorization.
///
/// The input `qr` matrix stores the upper triangular factor in its upper
/// triangle and Householder vectors in its strict lower triangle. This helper
/// zeros the strict lower triangle and preserves the upper triangular entries.
#[inline]
#[must_use]
pub fn get_r<const N: usize, const M: usize>(qr: &Matrix<N, M>) -> Matrix<N, M> {
    let mut r = *qr;
    for row in 1..N {
        for col in 0..min(row, M) {
            r[(row, col)] = 0.;
        }
    }
    r
}

/// Compute the inverse of the upper triangular `R` factor stored in a packed QR decomposition.
///
/// The input `qr` is assumed to contain the `R` factor in its upper triangle,
/// and the leading `M × M` block must be non-singular. This routine performs
/// back substitution on each column of the identity matrix to obtain `R^{-1}`.
#[inline]
#[must_use]
pub fn get_r_inv<const N: usize, const M: usize>(qr: &Matrix<N, M>) -> Matrix<M, M> {
    let mut inv_r = Matrix::<M, M>::zeros();

    for j in 0..M {
        for i in (0..M).rev() {
            let mut value = if i == j { 1.0 } else { 0.0 };
            for k in (i + 1)..M {
                value -= qr[(i, k)] * inv_r[(k, j)];
            }
            inv_r[(i, j)] = value / qr[(i, i)];
        }
    }

    inv_r
}

/// Construct the explicit orthogonal matrix `Q` from a packed QR factorization.
///
/// This applies the stored Householder reflectors in reverse order to the
/// identity matrix, producing an `N × N` orthogonal matrix.
#[must_use]
#[inline]
pub fn get_q<const N: usize, const M: usize>(qr: &Matrix<N, M>, taus: &[f64]) -> Matrix<N, N> {
    let mut q = Matrix::<N, N>::identity();
    for (iter, &tau) in taus.iter().enumerate().rev() {
        if tau != 0.0 {
            apply_householder_left(&mut q, qr, iter, tau);
        }
    }
    q
}

/// Compute the product `Q^T * A` using a packed QR factorization.
///
/// The input matrix `a` has shape `N × K`. The orthogonal factor `Q` is
/// represented implicitly in `qr` and `taus`, so this helper applies the stored
/// reflectors without forming `Q` explicitly.
#[inline]
#[must_use]
fn qt_times<const N: usize, const M: usize, const K: usize>(
    a: &Matrix<N, K>,
    qr: &Matrix<N, M>,
    taus: &[f64],
) -> Matrix<N, K> {
    let mut result = *a;
    for (iter, &tau) in taus.iter().enumerate() {
        if tau != 0.0 {
            apply_householder_left(&mut result, qr, iter, tau);
        }
    }
    result
}

/// Compute the product `Q * A` using a packed QR factorization.
///
/// The input matrix `a` has shape `N × K`. The orthogonal factor `Q` is encoded
/// implicitly in `qr` and `taus`, so this helper applies the stored reflectors
/// in reverse order without explicitly forming `Q`.
#[inline]
#[must_use]
pub fn q_times<const N: usize, const M: usize, const K: usize>(
    a: &Matrix<N, K>,
    qr: &Matrix<N, M>,
    taus: &[f64],
) -> Matrix<N, K> {
    let mut result = *a;
    for (iter, &tau) in taus.iter().enumerate().rev() {
        if tau != 0.0 {
            apply_householder_left(&mut result, qr, iter, tau);
        }
    }
    result
}

/// Compute the product `A * Q` using a packed QR factorization.
///
/// The input matrix `a` has shape `K × N`. The orthogonal factor `Q` is encoded
/// implicitly in `qr` and `taus`. This helper applies the stored reflectors on
/// the right in forward order, producing `A Q` without forming `Q`.
#[inline]
#[must_use]
pub fn times_q<const N: usize, const M: usize, const K: usize>(
    a: &Matrix<K, N>,
    qr: &Matrix<N, M>,
    taus: &[f64],
) -> Matrix<K, N> {
    let mut result = *a;
    for (iter, &tau) in taus.iter().enumerate() {
        // forward: H_{k-1} first, H_0 last
        if tau != 0.0 {
            apply_householder_right(&mut result, qr, iter, tau);
        }
    }
    result
}

/// Compute the product `A * Q^T` using a packed QR factorization.
///
/// The input matrix `a` has shape `K × N`. The orthogonal factor `Q` is encoded
/// implicitly in `qr` and `taus`. This helper applies the stored reflectors on
/// the right in reverse order, producing `A Q^T` without forming `Q` explicitly.
#[inline]
#[must_use]
pub fn times_qt<const N: usize, const M: usize, const K: usize>(
    a: &Matrix<K, N>,
    qr: &Matrix<N, M>,
    taus: &[f64],
) -> Matrix<K, N> {
    let mut result = *a;
    for (iter, &tau) in taus.iter().enumerate().rev() {
        // reverse: H_0 first, H_{k-1} last
        if tau != 0.0 {
            apply_householder_right(&mut result, qr, iter, tau);
        }
    }
    result
}

/// Solve `A x = b` in the least-squares sense using QR decomposition.
///
/// The function computes a packed QR decomposition of `A`, applies `Q^T` to the
/// right-hand side `b`, and then solves the upper triangular system `R x = Q^T b`
/// by back substitution. The output has shape `M × 1`.
#[inline]
#[must_use]
pub fn qr_solve<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
    b: &Matrix<N, 1>,
) -> Matrix<M, 1> {
    let (qr, taus) = super::qr_decomposition(a);
    let rank = N.min(M);

    // Compute Q^T * b.
    let qt_b = qt_times(b, &qr, &taus);

    // Solve R * x = Q^T b by back substitution.
    let mut x = Matrix::<M, 1>::zeros();
    for row_id in (0..rank).rev() {
        let mut sum = 0.0;
        for col_idx in (row_id + 1)..rank {
            sum += qr[(row_id, col_idx)] * x[(col_idx, 0)];
        }
        x[(row_id, 0)] = (qt_b[(row_id, 0)] - sum) / qr[(row_id, row_id)];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::Matrix;
    use crate::{
        MatMul, SquareMatrix,
        matrix::{
            qr::{get_q, get_r, q_times, qr_solve, qt_times, times_q, times_qt},
            test_utils::assert_matrices_ulps_eq,
        },
    };

    #[test]
    fn test_qr_square() {
        let (qr, _taus) = super::qr_decomposition(&Matrix::<3, 3> {
            rows: [[2., 9., 24.], [1., 10., 10.], [2., 10., 10.]],
        });

        let correct_answer = Matrix::<3, 3> {
            rows: [[-3., -16., -26.], [0.2, -5., 0.], [0.4, 0., -10.]],
        };
        assert_matrices_ulps_eq::<3, 3, _, _>(&correct_answer, &qr);
    }

    #[test]
    fn test_qr_tall() {
        let test_a = Matrix::<4, 3> {
            rows: [[-1., -1., 1.], [1., 3., 3.], [-1., -1., 5.], [1., 3., 7.]],
        };
        let (qr, taus) = super::qr_decomposition(&test_a);

        let correct_answer = Matrix::<4, 3> {
            rows: [
                [2., 4., 2.],
                [-1. / 3., -2., -8.],
                [1. / 3., 1. / 5., -4.],
                [-1. / 3., 2. / 5., 1. / 3.],
            ],
        };

        assert_matrices_ulps_eq::<4, 3, _, _>(&correct_answer, &qr);

        let test_q = get_q(&qr, &taus);
        let test_r = get_r(&qr);

        let identity = Matrix::<4, 4>::identity();
        assert_matrices_ulps_eq::<4, 3, _, _>(&test_a, &test_q.matmul(&test_r));
        assert_matrices_ulps_eq::<4, 3, _, _>(&test_a, &q_times(&test_r, &qr, &taus));
        assert_matrices_ulps_eq::<4, 4, _, _>(&identity, &times_q(&test_q.transpose(), &qr, &taus));
        assert_matrices_ulps_eq::<4, 4, _, _>(&identity, &times_qt(&test_q, &qr, &taus));
        assert_matrices_ulps_eq::<4, 4, _, _>(&identity, &qt_times(&test_q, &qr, &taus));
        assert_matrices_ulps_eq::<4, 4, _, _>(&test_q, &q_times(&identity, &qr, &taus));
        assert_matrices_ulps_eq::<4, 4, _, _>(&test_q, &times_q(&identity, &qr, &taus));
        assert_matrices_ulps_eq::<4, 4, _, _>(
            &test_q.transpose(),
            &qt_times(&identity, &qr, &taus),
        );
        assert_matrices_ulps_eq::<4, 4, _, _>(
            &test_q.transpose(),
            &times_qt(&identity, &qr, &taus),
        );

        let test_b = Matrix::<4, 1> {
            rows: [[0.], [16.], [12.], [28.]],
        };
        let x_actual = Matrix::<3, 1> {
            rows: [[1.0], [2.0], [3.0]],
        };
        let test_x = qr_solve(&test_a, &test_b);
        assert_matrices_ulps_eq::<3, 1, _, _>(&x_actual, &test_x);
    }

    #[test]
    fn test_get_q_and_r_reconstruct() {
        let a = Matrix::<4, 3> {
            rows: [[-1., -1., 1.], [1., 3., 3.], [-1., -1., 5.], [1., 3., 7.]],
        };
        let (qr, taus) = super::qr_decomposition(&a);
        let q = super::get_q(&qr, &taus);
        let r = super::get_r(&qr);
        assert_matrices_ulps_eq::<4, 3, _, _>(&a, &q.matmul(&r));
    }

    #[test]
    fn test_get_r_inv() {
        let (qr, _) = super::qr_decomposition(&Matrix::<3, 3> {
            rows: [[2., 9., 24.], [1., 10., 10.], [2., 10., 10.]],
        });
        let r = super::get_r(&qr);
        let r_inv = super::get_r_inv(&qr);
        let identity = Matrix::<3, 3>::identity();
        assert_matrices_ulps_eq::<3, 3, _, _>(&identity, &r.matmul(&r_inv));
        assert_matrices_ulps_eq::<3, 3, _, _>(&identity, &r_inv.matmul(&r));
    }

    #[test]
    fn test_times_q_and_times_qt_identity() {
        let a = Matrix::<4, 3> {
            rows: [[-1., -1., 1.], [1., 3., 3.], [-1., -1., 5.], [1., 3., 7.]],
        };
        let (qr, taus) = super::qr_decomposition(&a);
        let q = super::get_q(&qr, &taus);

        let id4 = Matrix::<4, 4>::identity();

        let t_q = super::times_q(&id4, &qr, &taus);
        assert_matrices_ulps_eq::<4, 4, _, _>(&q, &t_q);

        let t_qt = super::times_qt(&id4, &qr, &taus);
        assert_matrices_ulps_eq::<4, 4, _, _>(&q.transpose(), &t_qt);
    }

    #[test]
    fn test_q_times_and_qt_times_identity() {
        let a = Matrix::<4, 3> {
            rows: [[-1., -1., 1.], [1., 3., 3.], [-1., -1., 5.], [1., 3., 7.]],
        };
        let (qr, taus) = super::qr_decomposition(&a);
        let q = super::get_q(&qr, &taus);

        let id4 = Matrix::<4, 4>::identity();

        let q_left = super::q_times(&id4, &qr, &taus);
        assert_matrices_ulps_eq::<4, 4, _, _>(&q, &q_left);

        let qt_left = super::qt_times(&id4, &qr, &taus);
        assert_matrices_ulps_eq::<4, 4, _, _>(&q.transpose(), &qt_left);
    }
}

// #[test]
// fn test_qr_wide() {
//     let (qr, taus) = super::qr_decomposition(&Matrix::<3, 4> {
//         rows: [[-1., -1., 1., 0.], [1., 3., 3., 0.], [-1., -1., 5., 0.]],
//     });

//     for row in 0..3 {
//         for col in 0..4 {
//             print!("{:8.4} ", qr[(row, col)]);
//         }
//         println!();
//     }

//     let correct_answer = Matrix::<3, 4> {
//         rows: [
//             [2., 4., 2., 0.],
//             [-1. / 3., -2., -8., 0.],
//             [1. / 3., 1. / 5., -4., 0.],
//         ],
//     };
//     assert_matrices_ulps_eq::<3, 4, _, _>(&correct_answer, &qr);
// }

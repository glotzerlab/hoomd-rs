// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use super::Matrix;
use crate::{GeneralMatrix, SquareMatrix};
use std::cmp::min;

/// Compute the QR decomposition of a matrix using Householder reflections.
///
/// Returns the packed QR matrix (R in upper triangle, Householder vectors in lower triangle)
/// and the vector of tau values for each reflection.
pub(super) fn qr_decomposition<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
) -> (Matrix<N, M>, Vec<f64>) {
    let mut qr = a.clone();
    let mut taus = Vec::new();

    for i in 0..M {
        let mut tau = 0.0;
        let mut beta = 0.0;

        let x_norm_2 = qr
            .get_col_slice_iter(i, (i + 1)..N)
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
            taus.push(tau);
            continue;
        }

        // Collect the Householder vector v for this step (stored in column i, rows i..N).
        // Note: w_t is indexed from 0 but corresponds to columns (i+1)..M of qr.
        let v_col: Vec<f64> = qr.get_col_slice_iter(i, i..N).collect();

        // Compute w^T = C^T * v, where C = qr[i..N, (i+1)..M].
        let mut w_t = vec![0.0; M - i - 1];
        for (row_slice, &v_r) in qr.submatrix_slice_iter(i..N, (i + 1)..M).zip(v_col.iter()) {
            for (j, &val) in row_slice.iter().enumerate() {
                w_t[j] += val * v_r;
            }
        }

        // Apply the rank-1 update: C -= tau * v * w^T.
        for (row_slice_mut, &v_r) in qr
            .submatrix_slice_iter_mut(i..N, (i + 1)..M)
            .zip(v_col.iter())
        {
            for (j, cell) in row_slice_mut.iter_mut().enumerate() {
                *cell -= tau * v_r * w_t[j];
            }
        }

        qr[(i, i)] = beta;
        taus.push(tau);
    }

    (qr, taus)
}

/// Apply a single Householder reflector on the left:
/// result[iter..N, 0..K] = (I - tau * v * v^T) * result[iter..N, 0..K]
/// where v has a leading 1 at position `iter` and stored values below.
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

    for (col, &val) in result.get_row(iter).as_slice().iter().enumerate() {
        w_t[col] += val; // leading element of v is 1
    }
    for (row_slice, v_r) in result
        .submatrix_slice_iter(tail.clone(), 0..K)
        .zip(qr.get_col_slice_iter(iter, tail.clone()))
    {
        for (col, &val) in row_slice.iter().enumerate() {
            w_t[col] += val * v_r;
        }
    }

    // result[iter..N, 0..K] -= tau * v * w^T
    for (col, val) in result
        .submatrix_slice_iter_mut(iter..iter + 1, 0..K)
        .next()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *val -= tau * w_t[col]; // leading element of v is 1
    }
    for (row_slice_mut, v_r) in result
        .submatrix_slice_iter_mut(tail.clone(), 0..K)
        .zip(qr.get_col_slice_iter(iter, tail.clone()))
    {
        for (col, val) in row_slice_mut.iter_mut().enumerate() {
            *val -= tau * v_r * w_t[col];
        }
    }
}

/// Apply a single Householder reflector on the right:
/// result[0..K, iter..N] = result[0..K, iter..N] * (I - tau * v * v^T)
/// where v has a leading 1 at position `iter` and stored values below.
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
        .submatrix_slice_iter(0..K, iter..iter + 1)
        .enumerate()
    {
        w[row] += row_slice[0]; // leading element of v is 1
    }
    for (row, row_slice) in result.submatrix_slice_iter(0..K, tail.clone()).enumerate() {
        for (&val, v_r) in row_slice
            .iter()
            .zip(qr.get_col_slice_iter(iter, tail.clone()))
        {
            w[row] += val * v_r;
        }
    }

    // result[0..K, iter..N] -= tau * w * v^T
    for (row, row_slice_mut) in result
        .submatrix_slice_iter_mut(0..K, iter..iter + 1)
        .enumerate()
    {
        row_slice_mut[0] -= tau * w[row]; // leading element of v is 1
    }
    for (row, row_slice_mut) in result
        .submatrix_slice_iter_mut(0..K, tail.clone())
        .enumerate()
    {
        for (val, v_r) in row_slice_mut
            .iter_mut()
            .zip(qr.get_col_slice_iter(iter, tail.clone()))
        {
            *val -= tau * w[row] * v_r;
        }
    }
}

fn get_r<const N: usize, const M: usize>(qr: &Matrix<N, M>) -> Matrix<N, M> {
    let mut r = qr.clone();
    for row in 1..N {
        for col in 0..min(row, M) {
            r[(row, col)] = 0.;
        }
    }
    r
}

fn get_q<const N: usize, const M: usize>(qr: &Matrix<N, M>, taus: &[f64]) -> Matrix<N, N> {
    let mut q = Matrix::<N, N>::identity();
    for (iter, &tau) in taus.iter().enumerate().rev() {
        if tau != 0.0 {
            apply_householder_left(&mut q, qr, iter, tau);
        }
    }
    q
}

fn qt_times<const N: usize, const M: usize, const K: usize>(
    a: &Matrix<N, K>,
    qr: &Matrix<N, M>,
    taus: &[f64],
) -> Matrix<N, K> {
    let mut result = a.clone();
    for (iter, &tau) in taus.iter().enumerate() {
        if tau != 0.0 {
            apply_householder_left(&mut result, qr, iter, tau);
        }
    }
    result
}

fn q_times<const N: usize, const M: usize, const K: usize>(
    a: &Matrix<N, K>,
    qr: &Matrix<N, M>,
    taus: &[f64],
) -> Matrix<N, K> {
    let mut result = a.clone();
    for (iter, &tau) in taus.iter().enumerate().rev() {
        if tau != 0.0 {
            apply_householder_left(&mut result, qr, iter, tau);
        }
    }
    result
}

fn times_q<const N: usize, const M: usize, const K: usize>(
    a: &Matrix<K, N>,
    qr: &Matrix<N, M>,
    taus: &[f64],
) -> Matrix<K, N> {
    let mut result = a.clone();
    for (iter, &tau) in taus.iter().enumerate() {
        // forward: H_{k-1} first, H_0 last
        if tau != 0.0 {
            apply_householder_right(&mut result, qr, iter, tau);
        }
    }
    result
}

fn times_qt<const N: usize, const M: usize, const K: usize>(
    a: &Matrix<K, N>,
    qr: &Matrix<N, M>,
    taus: &[f64],
) -> Matrix<K, N> {
    let mut result = a.clone();
    for (iter, &tau) in taus.iter().enumerate().rev() {
        // reverse: H_0 first, H_{k-1} last
        if tau != 0.0 {
            apply_householder_right(&mut result, qr, iter, tau);
        }
    }
    result
}

#[inline]
pub fn qr_solve<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
    mut b: Matrix<N, 1>,
) -> Matrix<M, 1> {
    let (qr, taus) = super::qr_decomposition(a);
    let n = N.min(M);

    // Compute Q^T * b by applying H_0, H_1, ..., H_{n-1} in order.
    // Mirrors qt_times with K=1.
    for i in 0..n {
        if taus[i] == 0.0 {
            continue;
        }

        let tail = (i + 1)..N;

        // alpha = v^T * b[i..N], with leading element of v = 1
        let mut alpha = b[(i, 0)];
        for (u_j, b_j) in qr
            .get_col_slice_iter(i, tail.clone())
            .zip(tail.clone().map(|j| b[(j, 0)]))
        {
            alpha += u_j * b_j;
        }
        alpha *= taus[i];

        // b[i..N] -= alpha * v
        b[(i, 0)] -= alpha; // leading element of v = 1
        for (j, u_j) in qr.get_col_slice_iter(i, tail.clone()).enumerate() {
            b[(i + 1 + j, 0)] -= alpha * u_j;
        }
    }

    // Solve R * x = b[0..n] by back substitution.
    let mut x = Matrix::<M, 1>::zeros();
    for row_id in (0..n).rev() {
        let mut sum = 0.0;
        for k in (row_id + 1)..n {
            sum += qr[(row_id, k)] * x[(k, 0)];
        }
        x[(row_id, 0)] = (b[(row_id, 0)] - sum) / qr[(row_id, row_id)];
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
            test_utils::assert_matrixes_ulps_eq,
        },
    };

    #[test]
    fn test_qr_square() {
        let (qr, taus) = super::qr_decomposition(&Matrix::<3, 3> {
            rows: [[2., 9., 24.], [1., 10., 10.], [2., 10., 10.]],
        });

        let correct_answer = Matrix::<3, 3> {
            rows: [[-3., -16., -26.], [0.2, -5., 0.], [0.4, 0., -10.]],
        };
        assert_matrixes_ulps_eq::<3, 3, _, _>(&correct_answer, &qr);
    }

    #[test]
    fn test_qr_tall() {
        let a = Matrix::<4, 3> {
            rows: [[-1., -1., 1.], [1., 3., 3.], [-1., -1., 5.], [1., 3., 7.]],
        };
        let (qr, taus) = super::qr_decomposition(&a);

        let correct_answer = Matrix::<4, 3> {
            rows: [
                [2., 4., 2.],
                [-1. / 3., -2., -8.],
                [1. / 3., 1. / 5., -4.],
                [-1. / 3., 2. / 5., 1. / 3.],
            ],
        };

        assert_matrixes_ulps_eq::<4, 3, _, _>(&correct_answer, &qr);

        let q = get_q(&qr, &taus);
        let r = get_r(&qr);

        // println!("Q:");
        // for row in 0..4 {
        //     for col in 0..4 {
        //         print!("{:8.2} ", q[(row, col)]);
        //     }
        //     println!();
        // }
        // println!("R:");
        // for row in 0..4 {
        //     for col in 0..3 {
        //         print!("{:8.2} ", r[(row, col)]);
        //     }
        //     println!();
        // }

        let identity = Matrix::<4, 4>::identity();
        assert_matrixes_ulps_eq::<4, 3, _, _>(&a, &q.matmul(&r));
        assert_matrixes_ulps_eq::<4, 3, _, _>(&a, &q_times(&r, &qr, &taus));
        assert_matrixes_ulps_eq::<4, 4, _, _>(&identity, &times_q(&q.transpose(), &qr, &taus));
        assert_matrixes_ulps_eq::<4, 4, _, _>(&identity, &times_qt(&q, &qr, &taus));
        assert_matrixes_ulps_eq::<4, 4, _, _>(&identity, &qt_times(&q, &qr, &taus));
        assert_matrixes_ulps_eq::<4, 4, _, _>(&q, &q_times(&identity, &qr, &taus));
        assert_matrixes_ulps_eq::<4, 4, _, _>(&q, &times_q(&identity, &qr, &taus));
        assert_matrixes_ulps_eq::<4, 4, _, _>(&q.transpose(), &qt_times(&identity, &qr, &taus));
        assert_matrixes_ulps_eq::<4, 4, _, _>(&q.transpose(), &times_qt(&identity, &qr, &taus));

        let b = Matrix::<4, 1> {
            rows: [[0.], [16.], [12.], [28.]],
        };
        let x_actual = Matrix::<3, 1> {
            rows: [[1.0], [2.0], [3.0]],
        };
        let x = qr_solve(&a, b);
        assert_matrixes_ulps_eq::<3, 1, _, _>(&x_actual, &x);
    }

    #[test]
    fn test_get_q_and_r_reconstruct() {
        let a = Matrix::<4, 3> {
            rows: [[-1., -1., 1.], [1., 3., 3.], [-1., -1., 5.], [1., 3., 7.]],
        };
        let (qr, taus) = super::qr_decomposition(&a);
        let q = super::get_q(&qr, &taus);
        let r = super::get_r(&qr);
        assert_matrixes_ulps_eq::<4, 3, _, _>(&a, &q.matmul(&r));
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
        assert_matrixes_ulps_eq::<4, 4, _, _>(&q, &t_q);

        let t_qt = super::times_qt(&id4, &qr, &taus);
        assert_matrixes_ulps_eq::<4, 4, _, _>(&q.transpose(), &t_qt);
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
        assert_matrixes_ulps_eq::<4, 4, _, _>(&q, &q_left);

        let qt_left = super::qt_times(&id4, &qr, &taus);
        assert_matrixes_ulps_eq::<4, 4, _, _>(&q.transpose(), &qt_left);
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
//     assert_matrixes_ulps_eq::<3, 4, _, _>(&correct_answer, &qr);
// }

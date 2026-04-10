// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//TODO: Figure out to handle fat matrices... Should I just multiple extra columns by reflectors? But we don't have space in the matrix to store them?
//c.f. https://math.stackexchange.com/questions/3293263/why-should-qr-decomposition-not-be-possible-for-a-fat-matrix

//TODO: Decide whether to return the taus or not. They can be calculated from the matrix, though it is more efficient to store them separately.
// tau = 2 / ||u||^2 where u is the reflector vector with a leading 1.

use super::Matrix;
use crate::GeneralMatrix;
use std::cmp::min;

/// .
pub(super) fn qr_decomposition<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
) -> (Matrix<N, M>, Vec<f64>) {
    // Return default TODO;

    let mut qr = a.clone();
    let mut taus = Vec::new();
    for i in 0..M {
        let mut tau = 0.;
        let mut beta = 0.;

        let x_norm_2 = qr
            .get_col_slice_iter(i, (i + 1)..N)
            .map(|x| x * x)
            .sum::<f64>();
        if x_norm_2 != 0.0 {
            let alpha = qr[(i, i)];
            beta = -alpha.signum() * (x_norm_2 + alpha * alpha).sqrt();
            //TODO: scale beta in extreme cases
            tau = (beta - alpha) / beta;
            for j in i + 1..N {
                qr[(j, i)] /= alpha - beta;
            }
            //TODO: unscale beta
            qr[(i, i)] = 1.0; //Temporary. Rescale to beta later. Could be optimized out.
        }

        if tau == 0.0 {
            //either in last column or remainder of column is zero
            taus.push(tau);
            continue;
        } else {
            let v_col: Vec<f64> = qr.get_col_slice_iter(i, i..N).collect();

            // Compute w^T = (C^T) * v where C is the submatrix qr[i..N, i..M].
            let mut w_t = vec![0.0; M - i];
            for (row_slice, &v_r) in qr.submatrix_slice_iter(i..N, (i + 1)..M).zip(v_col.iter()) {
                for (j, &val) in row_slice.iter().enumerate() {
                    w_t[j] += val * v_r;
                }
            }

            // Now change the submatrix rows using the precomputed wT.
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
    }
    (qr, taus)
}

// fn get_Q<const N: usize, const M: usize>(qr: &Matrix<N, M>, taus: Vec<f64>) -> Matrix<N, N> {
//     let mut q = Matrix::<N, N>::identity();
//     for (iter, &tau) in taus.iter().enumerate().rev() {
//         if tau == 0.0 {
//             continue;
//         }
//         let reflector: Matrix<N, 1> = Matrix::<N, 1>::from_rows(&[qr.get_col_slice_iter(i, i..N).collect()]);
//         for row in iter..M {
//             for col in iter..M{
//                 Q[row, column] -= tau * reflector[row] * reflector[col];
//             }
//         }
//     }
//     q
// }

fn get_R<const N: usize, const M: usize>(qr: &Matrix<N, M>) -> Matrix<N, M> {
    let mut r = qr.clone();
    for row in 1..N {
        for col in 0..min(row, M) {
            r[(row, col)] = 0.;
        }
    }
    r
}

fn times_Q() {
    //Note: Typically you shouldn't need to form Q explicitly!
    //TODO
    unimplemented!()
}

fn times_QT() {
    //TODO
    unimplemented!()
}

fn Q_times() {
    //TODO
    unimplemented!()
}

fn QT_times() {
    //TODO
    unimplemented!()
}

pub fn qr_solve<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
    mut b: Matrix<N, 1>,
) -> Matrix<M, 1> {
    // Solve Ax = b using the QR decomposition.
    // Calculate Q^T b = H_n H_{n-1} ... H_{1} b = y
    // H_i b = (I - tau u u^T) b = b - tau (u^T b) u = b - tau * alpha *  u
    let (qr, taus) = super::qr_decomposition(&a);

    let n = N.min(M);
    for i in 0..n {
        let u_iter = qr.get_col_slice_iter(i + 1, i..M);
        let mut alpha = b[(i, 1)];
        for (j, u_j) in u_iter.enumerate() {
            alpha += u_j * b[(i + j + 1, 1)];
        }
        alpha *= taus[i];

        b[(i, 1)] -= alpha; // implicit 1 component
        let u_iter = qr.get_col_slice_iter(i, i..M);
        for (j, u_j) in u_iter.enumerate() {
            b[(i + j + 1, 1)] -= alpha * u_j;
        }
    }

    // Solve Rx = y by back substitution
    let mut x = Matrix::<M, 1>::zeros();
    for row_id in n..0 {
        let mut sum = 0.0;
        let row = qr[(row_id, (row_id + 1)..M)].into_iter();
        for (k, r_jk) in row.enumerate() {
            sum += r_jk * x[(k + 1 + row_id, 1)];
        }
        x[(row_id, 1)] = (b[(row_id, 1)] - sum) / qr[(row_id, row_id)];
    }
    x
}

#[cfg(test)]
mod tests {
    use std::convert::identity;

    use super::Matrix;
    use crate::MatMul;
    use crate::matrix::{qr::get_R, qr::qr_solve, test_utils::assert_matrixes_ulps_eq};

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

        // let q = get_Q(&qr, taus);
        // let r = get_R(&qr);

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

        // assert_matrixes_ulps_eq::<4, 4, _, _>(&a, &q.matmul(&r));
        //assert_matrixes_ulps_eq::<4, 4, _, _>(&a, &r.Q_times(&qr, &taus));
        //assert_matrixes_ulps_eq::<4, 4, _, _>(&identity(x), &q.transpose().times_Q(&qr, &taus));
        //assert_matrixes_ulps_eq::<4, 4, _, _>(&identity(x), &q.times_Q_T(&qr, &taus));
        //assert_matrixes_ulps_eq::<4, 4, _, _>(&identity(x), &q.Q_T_times(&qr, &taus));

        let b = Matrix::<4, 1> {
            rows: [[0.], [16.], [12.], [28.]],
        };
        let x_actual = Matrix::<3, 1> {
            rows: [[1.0], [2.0], [3.0]],
        };
        let x = qr_solve(&a, b);
        assert_matrixes_ulps_eq::<3, 1, _, _>(&x_actual, &x);
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

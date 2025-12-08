// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! General-purpose linear algebra functions on slices.
use super::Matrix;

/// .
pub(super) fn qr_decomposition<const N: usize, const M: usize>(a: &Matrix<N, M>) -> Matrix<N, M> {
    // Return default TODO;

    let mut qr = a.clone();
    for i in 0..N {
        let mut tau = 0.;
        let mut beta = 0.;
        if i <= (N - 1) {
            let x_norm_2 = qr
                .get_col_slice_iter(i, (i + 1)..N)
                .map(|x| x * x)
                .sum::<f64>();
            println!("x_norm_2 = {}", x_norm_2);
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
        }
        if tau == 0.0 {
            //either in last row or remainder of column is zero
            continue;
        } else {
            let v_col: Vec<f64> = qr.get_col_slice_iter(i, i..N).collect();

            // Compute w^T = (C^T) * v where C is the submatrix qr[i..N, i..M].
            // We can compute it using an immutable submatrix iterator then
            // perform the mutable updates once the immutable borrow is dropped.
            let mut w_t = vec![0.0; M - i];
            for (row_slice, &v_r) in qr.submatrix_slice_iter(i..N, (i + 1)..M).zip(v_col.iter()) {
                for (j, &val) in row_slice.iter().enumerate() {
                    w_t[j] += val * v_r;
                }
            }

            // Now mutate the submatrix rows using the precomputed wT.
            for (row_slice_mut, &v_r) in qr
                .submatrix_slice_iter_mut(i..N, (i + 1)..M)
                .zip(v_col.iter())
            {
                for (j, cell) in row_slice_mut.iter_mut().enumerate() {
                    *cell -= tau * v_r * w_t[j];
                }
            }
            qr[(i, i)] = beta;
        }
    }
    qr
}

#[cfg(test)]
mod tests {
    use super::Matrix;
    use approx::{assert_ulps_eq, ulps_eq};
    use std::{fmt::Debug, ops::Index};

    const EPS: f64 = 1e-13;

    #[test]
    fn test_qr_decomp() {
        let qr = super::qr_decomposition(&Matrix::<3, 3> {
            rows: [[2., 9., 24.], [1., 10., 10.], [2., 10., 10.]],
        });

        let correct_answer = Matrix::<3, 3> {
            rows: [[-3., -16., -26.], [0.2, 5., 0.], [0.4, 0., -10.]],
        };
        assert_matrixes_ulps_eq::<3, 3, _, _>(correct_answer, qr);
    }
}

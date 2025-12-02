// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! General-purpose linear algebra functions on slices.

use super::gemv::gemv_submatrix_column_into_column;
use super::{GeneralMatrix, Matrix};

/// .
pub(super) fn qr_decomposition<const N: usize, const M: usize>(a: &Matrix<N, M>) -> (Matrix<M, M>) {
    // Return default TODO;
    (Matrix::<M, M>::zeros(), Matrix::<N, M>::zeros());

    let mut A = a.clone();
    for i in 0..N {
        let tau = 0.;
        let beta = 1.;
        if i == (N - 1) {
        } else if let x_norm_2 = A
            .get_col_slice_iter(i, (i + 1)..N)
            .map(|x| x * x)
            .sum::<f64>()
        {
            if x_norm_2 != 0.0 {
                let alpha = A[(i, i)];
                beta = -alpha.signum() * (x_norm_2 + alpha * alpha).sqrt();
                //TODO: scale beta
                tau = (beta - alpha) / beta;
                for j in i + 1..N {
                    A[(i, j)] /= alpha - beta;
                }
                //TODO: unscale beta
                A[(i, i)] = 1.0;
            }
        }
        if tau == 0.0 {
            continue;
        } else {
            let mut C = A.submatrix_slice_iter(i..N, i..M);
            let v = A.get_col_slice_iter(i, i..M);

            //let (wT, other_rows) = C.split_first_mut().unwrap();
            let wT = C.first();
            for row in other_rows {
                for (j, (value, v_jp1)) in row.zip(v.clone().skip()).enumerate() {
                    wT[j] += value * v_jp1;
                }
            }

            for row in i..N {
                for col in i..M {
                    A[(row, col)] -= tau * v[row] * wT[col];
                }
            }
        }
        A[(i, i)] = beta;
    }
}

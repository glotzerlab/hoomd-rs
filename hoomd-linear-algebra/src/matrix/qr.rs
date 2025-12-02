// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! General-purpose linear algebra functions on slices.

use super::Matrix;

/// .
pub(super) fn qr_decomposition<const N: usize, const M: usize>(a: &Matrix<N, M>) -> Matrix<N, M> {
    // Return default TODO;

    let mut A = a.clone();
    for i in 0..N {
        let mut tau = 0.;
        let beta = 0.;
        if i == (N - 1) {
        } else if let x_norm_2 = A
            .get_col_slice_iter(i, (i + 1)..N)
            .map(|x| x * x)
            .sum::<f64>()
        {
            if x_norm_2 != 0.0 {
                let alpha = A[(i, i)];
                let beta = -alpha.signum() * (x_norm_2 + alpha * alpha).sqrt();
                //TODO: scale beta in extreme cases
                tau = (beta - alpha) / beta;
                for j in i + 1..N {
                    A[(i, j)] /= alpha - beta;
                }
                //TODO: unscale beta
                A[(i, i)] = 1.0; //Temporary. Rescale to beta later.
            }
        }
        if tau == 0.0 { //either in last row or remainder of column is zero
            continue;
        } else {
            let v_col: Vec<f64> = A.get_col_slice_iter(i, i..N).collect();

            // Compute w^T = (C^T) * v where C is the submatrix A[i..N, i..M].
            // We can compute it using an immutable submatrix iterator then
            // perform the mutable updates once the immutable borrow is dropped.
            let mut w_t = vec![0.0; M - i];
            for (row_slice, &v_r) in A.submatrix_slice_iter(i..N, i..M).zip(v_col.iter()) {
                for (j, &val) in row_slice.iter().enumerate() {
                    w_t[j] += val * v_r;
                }
            }

            // Now mutate the submatrix rows using the precomputed wT.
            for (row_slice_mut, &v_r) in A.submatrix_slice_iter_mut(i..N, i..M)
                .zip(v_col.iter())
            {
                for (j, cell) in row_slice_mut.iter_mut().enumerate() {
                    *cell -= tau * v_r * w_t[j];
                }
            }
        A[(i, i)] = beta;
        }
    }
    A
}

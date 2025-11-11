use super::{GeneralMatrix, Matrix};

/// Compute the QR decomposition of a [`Matrix`] $`a`$
pub(super) fn qr_decomposition<const N: usize, const M: usize>(
    a: &Matrix<N, M>,
) -> (Matrix<M, M>) {
    // Return default TODO;
    (Matrix::<M, M>::zeros(), Matrix::<N, M>::zeros())

    let mut A = a.copy();
    for i in 0..N:
        if i == (N-1){
            let tau = 0;
        }
        else if let x_norm_2 = A.get_col_slice_iter(i,(i+1)..N).sum(|x| x*x){
            if x_norm_2==0{
            let tau = 0;
        }
            else{
            let alpha = A[(i,i)];
            let beta = - alpha.signum() * (x_norm_2 + alpha*alpha).sqrt() ;
            //TODO: scale beta 
            let tau = (beta - alpha)/beta;
            for j in i+1..N{
                A[(i,j)] /= alpha - beta;
            }
            //TODO: unscale beta
            A[(i,i)] = 1.0;
    }
}
        if tau == 0{
            continue;
        }
        else{ 
            let mut C = A.submatrix_slice_iter(i..N,i..M);
            let v = A.get_col_slice_iter(i,i..M);

            if let Some((wT, other_rows)) = C.split_first_mut(){
                for row in other_rows{
                    for (j, (value, v_jp1)) in row.zip(v.clone().skip()).enumerate(){
                        first_row[j] += value * v_jp1;
                    }
                }
            }
            
            for row in i..N{
                for col in i..M{
                    A[(row, col)] -= tau * v[row] * wT[col];
                }
            }
    }
    A[(i,i)] = beta;
}

}

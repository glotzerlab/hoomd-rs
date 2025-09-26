// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! TODO

use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix};
use hoomd_vector::Cartesian;

/// Compute the matrix whose elements are the covariance between the `i`'th and `j`th elements of two vectors of data.
///
/// ```math
/// M_{ij} = A^\top B;
/// M_{ij} = \sum_{k=1}^N A_{ki} B_{kj};
/// ```
///
/// This is equivalent to a Gram matrix computed between two different vectors.
pub trait CrossCovariance<M>
where
    // TODO: should Other be not self?
    Self: ExactSizeIterator,
{
    /// Compute the cross-covariance between two sets of vectors.
    ///
    /// The result will be `None` if the two sets of points have differing numbers of
    /// points.
    fn cross_covariance(self, other: Self) -> Option<M>;
}

impl<const N: usize> CrossCovariance<Matrix<N, N>> for std::slice::Iter<'_, Cartesian<N>> {
    /// Compute the cross-covariance between two sets of vectors.
    ///
    /// The result will be `None` if the two sets of points have differing numbers of
    /// points.
    /// # Examples
    /// ```
    /// # fn main() {
    /// use hoomd_linear_algebra::matrix::Matrix22;
    /// use hoomd_order::CrossCovariance;
    /// let points_a = [[0.0, 0.0].into(), [1.0, 0.0].into(), [0.0, 1.0].into()];
    /// let points_b = [[0.0, 0.0].into(), [2.0, 0.0].into(), [0.0, 2.0].into()];
    /// let result = points_a
    ///     .iter()
    ///     .cross_covariance(points_b.iter())
    ///     .expect("Sets match.");
    ///
    /// // The enclosed area of b is 4 times greater than a
    /// assert_eq!(result.det(), 4.0);
    ///
    /// // a and b are aligned, so their cross-covariance is diagonal
    /// assert_eq!(result, Matrix22::from_diag(&[2.0, 2.0]));
    /// # }
    /// ```
    #[inline]
    fn cross_covariance(self, other: Self) -> Option<Matrix<N, N>> {
        // TODO: better error?
        if self.len() != other.len() {
            return None;
        }
        Some(
            self.zip(other)
                .fold(Matrix::<N, N>::zeros(), |mut acc, (l, r)| {
                    for i in 0..N {
                        for j in 0..N {
                            acc[(i, j)] += l[i] * r[j];
                        }
                    }
                    acc
                }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_linear_algebra::{MatMul, matrix::Matrix22};
    use rstest::rstest;

    #[rstest]
    #[
    case(
        [
            Cartesian::from([1.0, 2.0]), Cartesian::from([3.0, 4.0]), Cartesian::from([9.0, 8.0])],
        [
            Cartesian::from([5.0, 6.0]), Cartesian::from([7.0, 8.0]), Cartesian::from([0.0, 9.0])],
    )]
    #[
    case(
        [Cartesian::from([-45.0, 45.0]), Cartesian::from([3.0, 74.9])],
        [Cartesian::from([5.0, 6.7]), Cartesian::from([7.0, 83.0])],
    )
    ]
    fn test_cross_covariance_arr<const M: usize>(
        #[case] a: [Cartesian<2>; M],
        #[case] b: [Cartesian<2>; M],
    ) {
        let result = a.iter().cross_covariance(b.iter()).unwrap();
        let expected = Matrix::<M, 2> {
            rows: a.map(|v| v.coordinates),
        }
        .transpose()
        .matmul(&Matrix::<M, 2> {
            rows: b.map(|v| v.coordinates),
        });
        assert_eq!(result, expected);
    }
    #[rstest]
    #[
    case(
        vec![Cartesian::from([-99.0, 45.0]), Cartesian::from([3.0, 4.9])],
        vec![Cartesian::from([5.0, 6.7]), Cartesian::from([7.0, 8.0])],
    )
    ]
    fn test_cross_covariance_vec(#[case] a: Vec<Cartesian<2>>, #[case] b: Vec<Cartesian<2>>) {
        let result = a.iter().cross_covariance(b.iter()).unwrap();
        let expected = Matrix22 {
            rows: [a[0].coordinates, a[1].coordinates],
        }
        .transpose()
        .matmul(&Matrix22 {
            rows: [b[0].coordinates, b[1].coordinates],
        });
        assert_eq!(result, expected);
    }
    #[rstest]
    #[
        case(&[Cartesian::from([1.0, 2.0])], &[Cartesian::from([-3.0, 1.0])])
    ]
    fn test_cross_covariance_slice(#[case] a: &[Cartesian<2>], #[case] b: &[Cartesian<2>]) {
        let result = a.iter().cross_covariance(b.iter()).unwrap();
        let expected = Matrix::<1, 2> {
            rows: [a[0].coordinates],
        }
        .transpose()
        .matmul(&Matrix::<1, 2> {
            rows: [b[0].coordinates],
        });
        assert_eq!(result, expected);
    }
}

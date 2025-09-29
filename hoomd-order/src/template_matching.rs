// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//!

#![allow(dead_code, reason = "wip")]

use crate::CrossCovariance;
use hoomd_linear_algebra::{GeneralMatrix, MatMul, matrix::Matrix};
use hoomd_vector::{Cartesian, RotationMatrix};
use std::ops::Index;

/// TODO
#[derive(Clone, Debug, PartialEq)]
pub struct Template<'a, P> {
    /// The coordinates defining the geometry of the template.
    pub(crate) coordinates: &'a [P],

    /// The center of mass of the `coordinates`.
    pub(crate) center: P,
}

impl<'a, const N: usize, I> CrossCovariance<I, Matrix<N, N>> for Template<'_, Cartesian<N>>
where
    I: ExactSizeIterator + Index<usize, Output = Cartesian<N>>,
{
    /// Compute the cross-covariance between two sets of vectors.
    ///
    /// The result will be `None` if the two sets of points have differing numbers of
    /// points.
    #[inline]
    fn cross_covariance(self, other: I) -> Option<Matrix<N, N>> {
        // TODO: better error?
        if self.coordinates.len() != other.len() {
            return None;
        }
        Some(
            self.coordinates
                .iter()
                .zip(other)
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

impl Template<'_, Cartesian<3>> {
    /// Compute the rotation and translation that optimally align two point sets in $`\mathbb{R}^3`$
    ///
    ///
    // fn template_match<I: ExactSizeIterator<Item = Cartesian<3>>>(
    fn template_match<
        I: ExactSizeIterator<Item = Cartesian<3>> + Index<usize, Output = Cartesian<3>>,
    >(
        &self,
        other: I,
    ) -> (RotationMatrix<3>, Cartesian<3>, f64) {
        // TODO: pre-center self
        // self.center = self
        //     .coordinates
        //     .iter()
        //     .fold(Cartesian::default(), |acc, &x| acc + x);
        // / Cartesian::from([self.coordinates.len() as f64; 3]);
        // let n = a_coords.len() as f64;
        // let a_center = a_coords
        //     .iter()
        //     .fold(Cartesian::default(), |acc, &v| acc + v)
        //     / n;
        // let b_center = b_coords
        //     .iter()
        //     .fold(Cartesian::default(), |acc, &v| acc + v)
        //     / n;
        let m = self
            .clone()
            .cross_covariance(other)
            .expect("Point set sizes did not match!");

        let (u, _, vt) = m.svd();
        let r = u.matmul(&vt).transpose();

        (
            r.try_into().expect("Should be unitary by construction."),
            Cartesian::default(),
            0.0,
        )
    }
}

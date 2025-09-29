// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! TODO

#![allow(dead_code, reason = "wip")]

use crate::CrossCovariance;
use hoomd_linear_algebra::{GeneralMatrix, MatMul, matrix::Matrix};
use hoomd_vector::{Cartesian, Rotate, RotationMatrix};

/// TODO
#[derive(Clone, Debug, PartialEq)]
pub struct Template<P> {
    /// The coordinates defining the geometry of the template, centered at the origin.
    pub(crate) coordinates: Vec<P>,

    /// The center of mass of the `coordinates`.
    pub(crate) center: P,
}

impl<const N: usize> From<Vec<Cartesian<N>>> for Template<Cartesian<N>> {
    fn from(value: Vec<Cartesian<N>>) -> Self {
        let centroid =
            value.iter().fold(Cartesian::default(), |acc, &v| acc + v) / value.len() as f64;
        Self {
            coordinates: value.iter().map(|&v| v - centroid).collect::<Vec<_>>(),
            center: centroid,
        }
    }
}

impl<const N: usize, I> CrossCovariance<I, Matrix<N, N>> for Template<Cartesian<N>>
where
    I: ExactSizeIterator<Item = Cartesian<N>>,
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

/// Compute the root-mean squared deviation between two sets of points.
fn compute_rmsd<const N: usize, I>(test_set: I, reference_set: &[Cartesian<N>]) -> f64
where
    I: IntoIterator<Item = Cartesian<N>>,
{
    test_set
        .into_iter()
        .zip(reference_set.iter())
        .fold(0.0, |acc, (x, &y)| {
            acc + (x - y)
                .coordinates
                .iter()
                .fold(0.0, |sum, p| sum + p.powi(2))
        })
}

impl Template<Cartesian<3>> {
    /// Compute the rotation and translation that optimally align points in `test_set` to a [`Template`].
    ///
    /// # Examples
    /// ```
    /// ```
    fn template_match(&self, test_set: &[Cartesian<3>]) -> (RotationMatrix<3>, Cartesian<3>, f64) {
        let test_set_centroid = test_set
            .iter()
            .fold(Cartesian::default(), |acc, &v| acc + v)
            / self.coordinates.len() as f64;
        let test_set_centered = test_set.iter().map(|&v| v - test_set_centroid);

        let m = self
            .clone()
            .cross_covariance(test_set_centered)
            .expect("Point set sizes did not match!");

        let (u, _, vt) = m.svd();
        let r: RotationMatrix<3> = u
            .matmul(&vt)
            .try_into()
            .expect("Should be unitary by construction.");

        let t = r.rotate(&test_set_centroid);

        (
            r,
            t,
            compute_rmsd(test_set.iter().map(|&v| r.rotate(&v)), &self.coordinates),
        )
    }
}

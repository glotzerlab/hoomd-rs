//!

#![allow(dead_code, reason = "wip")]

use crate::CrossCovariance;
use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix};
use hoomd_vector::{Cartesian, Versor};

/// TODO
#[derive(Clone, Debug, PartialEq)]
pub struct Template<'a, P> {
    /// The coordinates defining the geometry of the template.
    pub(crate) coordinates: &'a [P],

    /// The center of mass of the `coordinates`.
    pub(crate) center: P,
}

impl<'a, const N: usize> CrossCovariance<Template<'a, Cartesian<N>>, Matrix<N, N>>
    for Template<'_, Cartesian<N>>
{
    /// Compute the cross-covariance between two sets of vectors.
    ///
    /// The result will be `None` if the two sets of points have differing numbers of
    /// points.
    #[inline]
    fn cross_covariance(self, other: Template<'a, Cartesian<N>>) -> Option<Matrix<N, N>> {
        // TODO: better error?
        if self.coordinates.len() != other.coordinates.len() {
            return None;
        }
        Some(self.coordinates.iter().zip(other.coordinates.iter()).fold(
            Matrix::<N, N>::zeros(),
            |mut acc, (l, r)| {
                for i in 0..N {
                    for j in 0..N {
                        acc[(i, j)] += l[i] * r[j];
                    }
                }
                acc
            },
        ))
    }
}

impl Template<'_, Cartesian<3>> {
    /// Compute the rotation and translation that optimally align two point sets in $`\mathbb{R}^3`$
    ///
    ///
    // fn template_match<I: ExactSizeIterator<Item = Cartesian<3>>>(
    fn template_match(&self, other: Self) -> (Versor, Cartesian<3>, f64) {
        let m = self
            .clone()
            .cross_covariance(other)
            .expect("Point set sizes did not match!");
        (Versor::default(), Cartesian::default(), 0.0)
    }
}

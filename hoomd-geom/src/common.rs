// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Common geometric primitives that implement only a small number of operations.*/

use crate::{IntersectsAt, SupportFn};
use hoomd_vector::{Cartesian, Vector};

/// A [`Cylinder`] in three dimensions.
#[derive(Clone, Copy, Debug)]
pub struct Cylinder {
    /// Radius of the [`Cylinder`]
    r: f64,
    /// Height of the [`Cylinder`]
    h: f64,
}

/// An N-Dimensional [`HyperEllipsoid`] defined by its semi-major axes.
#[derive(Clone, Copy, Debug)]
pub struct HyperEllipsoid<const N: usize> {
    /// The principle semi-axes of the [`HyperEllipsoid`] along each direction.
    axes: Cartesian<N>,
}
impl<const N: usize> IntoIterator for HyperEllipsoid<N> {
    type Item = f64;
    type IntoIter = <[f64; N] as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.axes.into_iter()
    }
}

impl<const N: usize> SupportFn<Cartesian<N>> for HyperEllipsoid<N> {
    #[inline]
    fn support(&self, n: &Cartesian<N>) -> Cartesian<N> {
        let mut denominator = self.into_iter().zip(*n).map(|(r, n)| r * n);
        let denominator: f64 = Cartesian::<N>::from(std::array::from_fn(|_| {
            denominator.next().unwrap_or_default()
        }))
        .norm();
        let mut iter = n
            .into_iter()
            .zip(self.axes)
            .map(|(r, n)| r.powi(2) * n / denominator);
        std::array::from_fn(|_| iter.next().unwrap_or_default()).into()
    }
}

impl HyperEllipsoid<3> {
    #[inline]
    #[must_use]
    /// Compute a matrix representation of the ellipsoid.
    pub fn compute_ellipsoid_matrix() -> Cartesian<10> {
        Cartesian::default()
    }
}

// impl IntersectsAt for HyperEllipsoid<3> {
//     fn intersects_at(&self, other: &S, r_ij: &V, o_ij: &R) -> bool {}
// }

impl<const N: usize> HyperEllipsoid<N> {} // TODO matrix form and IntersectsAt

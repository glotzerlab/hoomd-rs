// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Common geometric primitives that implement only a small number of operations.*/

use crate::SupportFn;
use hoomd_vector::Cartesian;

/// An N-Dimensional [`HyperEllipsoid`] defined by its semi-major axes.
pub struct HyperEllipsoid<const N: usize> {
    /// The principle semi-axes of the [`HyperEllipsoid`] along each direction.
    axes: Cartesian<N>,
}

impl<const N: usize> SupportFn<Cartesian<N>> for HyperEllipsoid<N> {
    #[inline]
    fn support(&self, n: &Cartesian<N>) -> Cartesian<N> {
        self.axes
            .into_iter()
            .zip(n.into_iter())
            .map(|(r, n)| r.powi(2) * n)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap() // TODO: divide by norm of r not squared
    }
}

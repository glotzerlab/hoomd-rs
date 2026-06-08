// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ClosedSpherical`

use arrayvec::ArrayVec;

use super::{Error, GenerateGhosts, MAX_GHOSTS, Wrap};
use crate::property::Position;
use hoomd_manifold::Spherical;

/// [`ClosedSpherical<N>`] implements a hypercubic box enclosing a unit-radius
/// $`(N-1)`$-sphere. Use `ClosedSpherical` alongside [`SphericalVecCell`] to
/// implement [`ParallelSweep`] for [`Spherical`] bodies. `ClosedSpherical` is
/// otherwise functionally identical to using [`Open`] for [`Spherical`]
/// simulations.
///
/// # Example
/// ```
/// use hoomd_microstate::boundary::ClosedSpherical;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let closed_spherical: ClosedSpherical<3> = ClosedSpherical {};
/// # Ok(())
/// # }
/// ```
///
/// Similar to [`Closed`], `ClosedSpherical` does not wrap bodies and sites,
/// nor does it generate ghost sites.
///
/// [`SphericalVecCell`]: hoomd_spatial::SphericalVecCell;
/// [`ParallelSweep`]: hoomd_mc::ParallelSweep;
/// [`Spherical`]: hoomd_manifold::Spherical;
pub struct ClosedSpherical<const N: usize> {}

impl<BS, const N: usize> Wrap<BS> for ClosedSpherical<N>
where
    BS: Position<Position = Spherical<N>>,
{
    #[inline]
    fn wrap(&self, properties: BS) -> Result<BS, Error> {
        Ok(properties)
    }
}

impl<S, const N: usize> GenerateGhosts<S> for ClosedSpherical<N>
where
    S: Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        std::f64::consts::PI
    }
    #[inline]
    fn generate_ghosts(&self, _site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        ArrayVec::new()
    }
}

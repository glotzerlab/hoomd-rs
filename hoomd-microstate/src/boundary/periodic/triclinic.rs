// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for cuboids in cartesian space.

use tinyvec::ArrayVec;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use hoomd_geometry::{
    IsPointInside,
    shape::{Hypercuboid, Hyperparallelepiped, Triclinic},
};
use hoomd_linear_algebra::{
    MatMul,
    matrix::{Matrix, Matrix33, qr},
};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, Cross, InnerProduct};
use tinyvec::ArrayVec;

impl MaximumAllowableInteractionRange for Triclinic {
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        todo!();
        let minimum_l = self
            .edge_lengths // TODO: Change this to L_i's
            .iter()
            .map(PositiveReal::get)
            .reduce(f64::min)
            .expect("cuboid should have dimension 1 or greater");
        minimum_l / 2.0
    }
}

impl<P, const N: usize> Wrap<P> for Periodic<Triclinic>
where
    P: Position<Position = Cartesian<N>>,
{
    #[inline]
    fn wrap(&self, properties: P) -> Result<P, Error> {
        todo!();
    }
}

impl<S> GenerateGhosts<S> for Periodic<Triclinic>
where
    S: Position<Position = Cartesian<2>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    #[inline]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        todo!();
    }
}

#[cfg(test)]
mod tests {}

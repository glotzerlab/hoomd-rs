// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for triclinic boxes in cartesian space.

use arrayvec::ArrayVec;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use hoomd_geometry::{IsPointInside, shape::Triclinic};

use hoomd_vector::Cartesian;

impl MaximumAllowableInteractionRange for Triclinic {
    todo!();
}

impl Periodic<Triclinic> {
    pub fn to_fractional(&self, pos: &Cartesian<2>) -> Cartesian<2> {
        todo!();
    }
    pub fn to_absolute(&self, frac: &Cartesian<3>) -> Cartesian<1> {
        todo!();
    }
}

impl<P> Wrap<P> for Periodic<Triclinic>
where
    P: Position<Position = Cartesian<2>>,
{
    todo!();
}

impl<S> GenerateGhosts<S> for Periodic<Triclinic>
where
    S: Position<Position = Cartesian<2>> + Copy + Default,
{
    todo!();
}

#[cfg(test)]
mod tests {
    todo!();
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::intersects::IntersectsAt;
use hoomd_vector::Cartesian;
use hoomd_vector::{Rotate};

/** An axis-aligned N-cuboid
*/
#[derive(Clone, Copy, Debug)]
pub struct Cuboid<const N: usize> {
    /// The lengths of each edge of the cuboid.
    pub edge_lengths: Cartesian<N>,
}

impl Cuboid<3> {
    /// Length of the `Cuboid` edge along the x axis
    pub fn a(&self) -> f64 {
        self.edge_lengths[0]
    }
    /// Length of the `Cuboid` edge along the y axis
    pub fn b(&self) -> f64 {
        self.edge_lengths[1]
    }
    /// Length of the `Cuboid` edge along the z axis
    pub fn c(&self) -> f64 {
        self.edge_lengths[2]
    }
}

impl<const N: usize> From<[f64; N]> for Cuboid<N> {
    fn from(edge_lengths: [f64; N]) -> Cuboid<N> { Cuboid { edge_lengths: edge_lengths.into() } }
}
impl<const N: usize> From<Cartesian<N>> for Cuboid<N> {
    fn from(edge_lengths: Cartesian<N>) -> Cuboid<N> { Cuboid { edge_lengths } }
}

impl<const N: usize> Cuboid<N> {
    /// Determine the maximal extents of the cuboid along each Cartesian axis.
    pub fn maximal_extents(&self) -> Cartesian<N> {
        self.edge_lengths / 2.0
    }
    /// Determine the minimal extents of the cuboid along each Cartesian axis.
    pub fn minimal_extents(&self) -> Cartesian<N> {
        -self.edge_lengths / 2.0
    }
}

impl<const N: usize, R: Rotate<Cartesian<N>>> IntersectsAt<Cuboid<N>, Cartesian<N>, R> for Cuboid<N> {
    // TODO: wip, these conditions are not the correct checks.
    fn intersects_at(&self, other: &Cuboid<N>, r_ij: &Cartesian<N>, o_ij: &R) -> bool {
        // TODO: how can we assert that o_ij does not rotate the vector?
        println!("{}", r_ij <= &Cartesian::<N>::from([0.0; N]));
        // let it = self.minimal_extents() <= (*r_ij + other.maximal_extents()).into_iter().zip( 
        //     self.maximal_extents() <= (*r_ij + other.minimal_extents())
        // );
        // println!("{:?}", it);
        true
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest(
        edges0 => [[2.0, 2.0, 2.0]],
        edges1 => [[1.0, 1.0, 1.0]],
        c1 => [[0.0; 3], [1.0; 3], [4.0; 3]],
    )]
    fn check_box_intersections(edges0: [f64; 3], edges1: [f64; 3], c1: [f64; 3]) {
        // let (s0, s1) = (Cuboid::<3>::from(r0), Cuboid::<3>::from((r1, c1)));
        // assert!(s0.intersects(s1) == (s1.c[0] <= (s0.r + s1.r)))
    }
}

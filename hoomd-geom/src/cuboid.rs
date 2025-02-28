// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::intersects::IntersectsAt;
use crate::sphere::Sphere;
use hoomd_vector::Cartesian;
use hoomd_vector::{Rotate, Vector};
use std::iter::zip;

/** An axis-aligned N-cuboid
*/
#[derive(Clone, Copy, Debug)]
pub struct Cuboid<const N: usize> {
    /// The lengths of each edge of the cuboid.
    pub edge_lengths: Cartesian<N>,
    /// The center of mass of the Cuboid.
    pub center: Cartesian<N>,
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

impl<const N: usize> Cuboid<N> {
    /// Determine the maximal extents of the cuboid along each Cartesian axis.
    pub fn maximal_extents(&self) -> Cartesian<N> {
        self.center + (self.edge_lengths / 2.0)
    }
    /// Determine the minimal extents of the cuboid along each Cartesian axis.
    pub fn minimal_extents(&self) -> Cartesian<N> {
        self.center - (self.edge_lengths / 2.0)
    }
}

// impl<const N: usize, V: Vector, R: Rotate<V>, S> IntersectsAt<S, V, R> for Cuboid<N> {
//     // TODO: wip, these conditions are not the correct checks.
//     type S = Self;
//     fn intersects_at(&self, other: &S, r_ij: &V, o_ij: &R) -> bool {
//         todo!()
//     }
// }
impl<const N: usize, V: Vector, R: Rotate<V>> IntersectsAt<Sphere<N>, V, R> for Cuboid<N> {
    // TODO: wip, these conditions are not the correct checks.
    fn intersects_at(&self, other: &Sphere<N>, r_ij: &V, o_ij: &R) -> bool {
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

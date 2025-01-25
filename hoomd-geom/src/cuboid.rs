// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::intersects::Intersects;
use hoomd_vector::Cartesian;
use std::iter::zip;

/** An axis-aligned N-cuboid


*/
#[derive(Clone, Copy, Debug)]
pub struct Cuboid<const N: usize> {
    pub edge_lengths: Cartesian<N>,
    pub center: Cartesian<N>,
}

impl Cuboid<3> {
    pub fn a(&self) -> f64 {
        self.edge_lengths[0]
    }
    pub fn b(&self) -> f64 {
        self.edge_lengths[1]
    }
    pub fn c(&self) -> f64 {
        self.edge_lengths[2]
    }
}

impl<const N: usize> Cuboid<N> {
    pub fn maximal_extents(&self) -> Cartesian<N> {
        self.center + (self.edge_lengths / 2.0)
    }
    pub fn minimal_extents(&self) -> Cartesian<N> {
        self.center - (self.edge_lengths / 2.0)
    }
}

impl<const N: usize> Intersects<Cuboid<N>> for Cuboid<N> {
    // TODO: wip, these conditions are not the correct checks.
    fn intersects(&self, other: Self) -> bool {
        let (self_max, other_min) = (self.maximal_extents(), other.minimal_extents());
        let (self_min, other_max) = (self.minimal_extents(), other.maximal_extents());
        let temp0 =
            zip(self_max.coordinates.iter(), other_min.coordinates.iter()).any(|(x, y)| x > y);
        let temp1 =
            zip(self_min.coordinates.iter(), other_max.coordinates.iter()).any(|(x, y)| x > y);

        // println!("temp: {:?}, {:?}", temp0, temp1);
        // println!("selfmax: {}", self_max);
        // println!("othermin: {}", other_min);
        temp0 && temp1
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

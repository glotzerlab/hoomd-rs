// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use hoomd_microstate::Microstate;
use hoomd_spatial::{AllPairs, PointsNearBall};
use hoomd_vector::Cartesian;

// /// CN.
// pub fn coordination_number<B, X, C>(m: &Microstate<B, Cartesian<2>, X, C>) -> u64 {
//     for body in m.bodies() {
//         let pos = body.sites;
// }

pub trait LocalOrder<X, Y, Z, N> {
    fn compute_pair(x: X) -> Y; // pair compute
    fn accumulate(&self, n: N) -> Z; // accumulate over nlist
}

pub struct CoordinationNumber {
    pub particle_cn: Vec<u32>,
}

impl LocalOrder<u32, u32, u32, AllPairs<Cartesian<2>>> for CoordinationNumber {
    fn compute_pair(_: u32) -> u32 {
        1
    }
    fn accumulate(&self, n: AllPairs<Cartesian<2>>) -> u32 {
        self.particle_cn.iter().sum()
    }
}

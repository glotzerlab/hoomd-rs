// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

// use crate::{BoundingSphereRadius, SupportMapping, Volume};
use hoomd_vector::Cartesian;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hyperparallelepiped<const N: usize> {
    /// The lengths of each edge of the cuboid.
    pub edge_vectors: [Cartesian<N>; N],
}

pub type Parallelogram = Hyperparallelepiped<2>;
pub type Parallelepiped = Hyperparallelepiped<3>;

impl<const N: usize> Default for Hyperparallelepiped<N> {
    fn default() -> Self {
        Self {
            edge_vectors: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { 1. } else { 0. }).into()
            }),
        }
    }
}

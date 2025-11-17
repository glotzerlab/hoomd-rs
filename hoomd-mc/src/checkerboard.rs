// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Define checkerboard

use rand::Rng;

use hoomd_utility::valid::PositiveReal;

mod hypercuboid;

pub use hypercuboid::HypercuboidCheckerboard;

pub trait Checkerboard<P> {
    fn point_to_space_index(&self, point: &P) -> Option<usize>;
    fn space_indices_by_color(&self) -> &[Vec<usize>];
    fn num_spaces(&self) -> usize;
} 

pub trait Cover<P> {
    type Checkerboard: Checkerboard<P> + Sync;
    
    fn cover<R: Rng + ?Sized>(&self, rng: &mut R, interaction_range: PositiveReal) -> Self::Checkerboard;

    fn cover_into<R: Rng + ?Sized>(&self, checkerboard: &mut Self::Checkerboard, rng: &mut R, interaction_range: PositiveReal);
}

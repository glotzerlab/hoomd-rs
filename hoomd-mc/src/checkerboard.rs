// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Define checkerboard

mod hypercuboid;

pub trait Checkerboard<P> {
    fn point_to_space_index(&self, point: &P) -> Option<usize>;
    fn space_indices_by_color(&self) -> &[Vec<usize>];
} 


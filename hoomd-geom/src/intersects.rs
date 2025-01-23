// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use hoomd_vector::vector::Cartesian;

pub trait Shape {}

impl<const N: usize> Intersects<Sphere> for Shape<N> {
    fn intersects(self, &rhs: Shape) where Shape: Intersects::<Sphere> -> bool {
        todo!()
    }
    // Xeonocollide only needs vertices - prioritize implementing this
}// all we need for MC
trait Convex {} // Not necessarily needed for mc, but could be useful down the line


trait Volume {}
trait Particle {} // In different crate! - should be copyable (array based?)
trait Shape {} // vec![Cartesian::from(etc!())]

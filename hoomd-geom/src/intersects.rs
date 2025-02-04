// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::Sphere;
use hoomd_vector::{Cartesian, Vector, Rotate};
// use crate::Shape; // TODO: do we want this as a trait bound on S?

/**
Define a position and orientation-independent intersection based solely on the geometry
of the shape.
*/
pub trait Intersects<S> {
    /// Determine whether a Shape intersects another shape (based on intrinsic location).
    fn intersects(&self, other: S) -> bool;
}

/**
Define a position and orientation-dependent intersection between two bodies.
*/
pub trait IntersectsAt<S, V: Vector, R: Rotate<V>> {
    ///Determine whether a Particle intersects another shape at some position and orientation.
    fn intersects_at(&self, other:S, r_ij: &V, o_ij: &R) -> bool;
}

/// Whether a shape is fully bounded by another
pub trait Contains<S> {
    fn contains(&self, other: S) -> bool;
}

impl<const N: usize> Intersects<Sphere<N, Cartesian<N>>> for Sphere<N, Cartesian<N>> {
    fn intersects(&self, other: Sphere<N, Cartesian<N>>) -> bool {
        (other.c - self.c).norm_squared() <= (other.r + self.r).powi(2)
    }
}

// TODO: Jen - Xenocollide
// TODO: Jen - Polyhedron
// TODO: further tests

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest(
        r0 => [0.0, 0.5, 1.234],
        r1 => [0.0, 2.0, 99.9],
        c1 => [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [99.9, 0.0, 0.0]],
    )]
    fn check_sphere_intersections(r0: f64, r1: f64, c1: [f64; 3]) {
        let (s0, s1) = (
            Sphere::<3, Cartesian<3>>::from(r0),
            Sphere::<3, Cartesian<3>>::from((r1, c1)),
        );
        assert!(s0.intersects(s1) == (s1.c[0] <= (s0.r + s1.r)))
    }
    #[rstest(
        r0 => [0.0, 0.5, 1.234],
        r1 => [0.0, 2.0, 99.9],
        c1 => [[0.0, 0.0], [0.0, 2.0], [0.0, 99.9]],
    )]
    fn check_disc_intersections(r0: f64, r1: f64, c1: [f64; 2]) {
        let (s0, s1) = (
            Sphere::<2, Cartesian<2>>::from(r0),
            Sphere::<2, Cartesian<2>>::from((r1, c1)),
        );
        assert!(s0.intersects(s1) == (s1.c[1] <= (s0.r + s1.r)))
    }
}

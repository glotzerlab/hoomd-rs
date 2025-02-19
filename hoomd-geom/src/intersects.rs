// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::Sphere;
use hoomd_vector::{Cartesian, Rotate, Vector};
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
    fn intersects_at(&self, other: S, r_ij: &V, o_ij: &R) -> bool;
}

impl<const N: usize, V: Vector, R: Rotate<V>> IntersectsAt<Sphere<N>, V, R> for Sphere<N> {
    fn intersects_at(&self, other: Sphere<N>, r_ij: &V, _o_ij: &R) -> bool {
        (r_ij).norm_squared() <= (other.r + self.r).powi(2)
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

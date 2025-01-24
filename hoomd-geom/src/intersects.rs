// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::Sphere;
use hoomd_vector::Vector;
// use crate::Shape; // TODO: do we want this as a trait bound on S?

pub trait Intersects<S> {
    fn intersects(&self, other: S) -> bool;
}

impl<const N: usize> Intersects<Sphere<N>> for Sphere<N> {
    fn intersects(&self, other: Sphere<N>) -> bool {
        (other.c - self.c).norm_squared() <= (other.r + self.r).powi(2)
    }
}

// TODO: refactor into proper filenames
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
        let (s0, s1) = (Sphere::<3>::from(r0), Sphere::<3>::from((r1, c1)));
        assert!(s0.intersects(s1) == (s1.c[0] <= (s0.r + s1.r)))
    }
}

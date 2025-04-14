// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*!Axis-aligned N-cuboids, particularly used for bounding volume hierarchies.*/
use crate::intersects::IntersectsAt;
use crate::{SupportFn, Volume};
use hoomd_vector::Cartesian;
use hoomd_vector::{Rotate, Rotation, Vector};
use itertools::multizip;
use std::cmp::PartialEq;

/** An axis-aligned N-cuboid
*/
#[derive(Clone, Copy, Debug)]
pub struct Cuboid<const N: usize> {
    /// The lengths of each edge of the cuboid.
    pub edge_lengths: Cartesian<N>,
}

impl Cuboid<3> {
    /// Length of the `Cuboid` edge along the x axis
    #[inline]
    #[must_use]
    pub fn a(&self) -> f64 {
        self.edge_lengths[0]
    }
    /// Length of the `Cuboid` edge along the y axis
    #[inline]
    #[must_use]
    pub fn b(&self) -> f64 {
        self.edge_lengths[1]
    }
    /// Length of the `Cuboid` edge along the z axis
    #[inline]
    #[must_use]
    pub fn c(&self) -> f64 {
        self.edge_lengths[2]
    }
}

impl<const N: usize> Volume for Cuboid<N> {
    #[inline]
    fn volume(&self) -> f64 {
        self.edge_lengths
            .into_iter()
            .reduce(|acc, x| acc * x)
            .unwrap_or(0.0)
    }
}

// TODO: requires test
impl<const N: usize> SupportFn<Cartesian<N>> for Cuboid<N> {
    #[inline]
    fn support(&self, n: &Cartesian<N>) -> Cartesian<N> {
        let mut result = Cartesian::<N>::default();
        result
            .coordinates
            .iter_mut()
            .zip((*n).into_iter().zip(self.edge_lengths))
            .for_each(|(x, (n_i, l_i))| *x = l_i / 2.0 * n_i.signum());
        result
    }
}

impl<const N: usize> From<[f64; N]> for Cuboid<N> {
    #[inline]
    fn from(edge_lengths: [f64; N]) -> Cuboid<N> {
        Cuboid {
            edge_lengths: edge_lengths.into(),
        }
    }
}
impl<const N: usize> From<Cartesian<N>> for Cuboid<N> {
    #[inline]
    fn from(edge_lengths: Cartesian<N>) -> Cuboid<N> {
        Cuboid { edge_lengths }
    }
}

impl<const N: usize> Cuboid<N> {
    #[inline]
    #[must_use]
    /// Determine the maximal extents of the cuboid along each Cartesian axis.
    pub fn maximal_extents(&self) -> Cartesian<N> {
        self.edge_lengths / 2.0
    }
    #[inline]
    #[must_use]
    /// Determine the minimal extents of the cuboid along each Cartesian axis.
    pub fn minimal_extents(&self) -> Cartesian<N> {
        -self.edge_lengths / 2.0
    }
}

impl<const N: usize, R: Rotate<Cartesian<N>> + Rotation + PartialEq>
    IntersectsAt<Cuboid<N>, Cartesian<N>, R> for Cuboid<N>
{
    // TODO: Should o_ij be an Option?
    /**
    Determine the intersection between two axis-aligned cuboids.
    MUST be passed an identity `Rotation` or the method will panic.
    */
    #[inline]
    fn intersects_at(&self, other: &Cuboid<N>, r_ij: &Cartesian<N>, o_ij: &R) -> bool {
        assert!(*o_ij == R::identity());
        let other_mins = other.minimal_extents() + *r_ij;
        let other_maxs = other.maximal_extents() + *r_ij;
        multizip((
            self.minimal_extents(),
            other_maxs,
            self.maximal_extents(),
            other_mins,
        ))
        .all(|(l_min, o_max, l_max, o_min)| (l_min <= o_max) && (l_max >= o_min))
    }
}

#[cfg(test)]
#[allow(clippy::used_underscore_binding)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use hoomd_vector::Versor;
    use rstest::*;
    use std::marker::PhantomData;

    #[rstest(
        edges0 => [[2.0, 2.0, 2.0]],
        edges1 => [[1.0, 1.0, 1.0]],
    )]
    fn test_box_intersections(edges0: [f64; 3], edges1: [f64; 3]) {
        let (s0, s1) = (Cuboid::<3>::from(edges0), Cuboid::<3>::from(edges1));
        // Should all be false (no intersection), which we invert to true
        assert!(!s0.intersects_at(&s1, &[10.0, 10.0, 10.0].into(), &Versor::identity()));
        // Boundaries are aligned
        assert!(s0.intersects_at(&s1, &[1.5, 1.5, 1.5].into(), &Versor::identity()));
        // Both at origin - will always intersect for any cuboids
        assert!(s0.intersects_at(&s1, &[0.0, 0.0, 0.0].into(), &Versor::identity()));
        // TODO: is there a more programmatic way to test this?
    }

    #[rstest(
        _n => [
            PhantomData::<Cuboid<0>>,
            PhantomData::<Cuboid<1>>,
            PhantomData::<Cuboid<2>>,
            PhantomData::<Cuboid<3>>,
            PhantomData::<Cuboid<4>>
        ],
        l => [1e-6, 1.0, 3.456, 99_999_999.9],
    )]
    fn test_box_extents<const N: usize>(_n: PhantomData<Cuboid<N>>, l: f64) {
        let c = Cuboid::from([l; N]);
        assert_eq!(c.maximal_extents(), [l / 2.0; N].into());
        assert_eq!(c.minimal_extents(), [-l / 2.0; N].into());
    }

    #[rstest(
        _n => [
            PhantomData::<Cuboid<0>>,
            PhantomData::<Cuboid<1>>,
            PhantomData::<Cuboid<2>>,
            PhantomData::<Cuboid<3>>,
            PhantomData::<Cuboid<4>>
        ],
        l => [1e-6, 1.0, 3.456, 99_999_999.9],
    )]
    fn test_box_volume<const N: usize>(_n: PhantomData<Cuboid<N>>, l: f64) {
        let c = Cuboid::from([l; N]);
        assert_relative_eq!(
            c.volume(),
            if N != 0 {
                l.powi(i32::try_from(N).unwrap())
            } else {
                0.0
            }
        );
    }

    #[rstest(
        l => [1e-6, 1.0, 3.456, 99_999_999.9],
    )]
    fn test_box_abc(l: f64) {
        let c = Cuboid::from([l; 3]);
        assert_eq!([c.a(), c.b(), c.c()], [l; 3]);
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*!N-cuboids, which may or may not be treated as axis aligned.*/
use crate::{
    BoundingSphereRadius, IntersectsAt, SupportMapping, Volume,
    xenocollide::{collide2d, collide3d},
};
use hoomd_vector::{Cartesian, Rotate, Rotation, RotationMatrix};
use itertools::multizip;

use super::Hypersphere;

/** An axis-aligned N-cuboid
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cuboid<const N: usize> {
    /// The lengths of each edge of the cuboid.
    pub edge_lengths: Cartesian<N>, // TODO: use array of PositiveReal
}

/**A rectangle defined by its edge lengths.*/
pub type Rectangle = Cuboid<2>;

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
    // TODO: inherent implementation for intersects_aligned
}

impl<const N: usize> Cuboid<N> {
    /// Compute the intersection between two *axis-aligned* cuboids.
    #[must_use]
    #[inline]
    pub fn intersects_aligned(&self, other: &Cuboid<N>, v_ij: &Cartesian<N>) -> bool {
        let b_mins = other.minimal_extents() + *v_ij;
        let b_maxs = other.maximal_extents() + *v_ij;
        multizip((
            self.minimal_extents(),
            b_maxs,
            self.maximal_extents(),
            b_mins,
        ))
        .all(|(a_min, b_max, a_max, b_min)| (a_min <= b_max) && (a_max >= b_min))
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

impl<const N: usize> BoundingSphereRadius for Cuboid<N> {
    #[inline]
    fn bounding_sphere_radius(&self) -> f64 {
        f64::sqrt(3.0) / 2.0 * self.edge_lengths.into_iter().fold(f64::NAN, f64::max)
    }
}

// TODO: requires test
impl<const N: usize> SupportMapping<Cartesian<N>> for Cuboid<N> {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        let mut iter = n
            .into_iter()
            .zip(self.edge_lengths)
            .map(|(n_i, l_i)| l_i / 2.0 * n_i.signum());
        std::array::from_fn(|_| iter.next().unwrap_or_default()).into()
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

#[cfg(test)]
#[expect(clippy::used_underscore_binding, reason = "Required for const tests.")]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::*;
    use std::marker::PhantomData;

    #[rstest(
        edges0 => [[2.0, 2.0, 2.0]],
        edges1 => [[1.0, 1.0, 1.0]],
    )]
    fn test_box_intersections_aligned(edges0: [f64; 3], edges1: [f64; 3]) {
        let (s0, s1) = (Cuboid::<3>::from(edges0), Cuboid::<3>::from(edges1));
        // Should all be false (no intersection), which we invert to true
        assert!(!s0.intersects_aligned(&s1, &[10.0, 10.0, 10.0].into()));
        // Boundaries are aligned
        assert!(s0.intersects_aligned(&s1, &[1.5, 1.5, 1.5].into()));
        // Both at origin - will always intersect for any cuboids
        assert!(s0.intersects_aligned(&s1, &[0.0, 0.0, 0.0].into()));
    }
    #[rstest(
        edges0 => [[2.0, 2.0]],
        edges1 => [[1.0, 1.0]],
    )]
    fn test_box_intersections_2d_aligned(edges0: [f64; 2], edges1: [f64; 2]) {
        let (c0, c1) = (Cuboid::<2>::from(edges0), Cuboid::<2>::from(edges1));
        // Should all be false (no intersection), which we invert to true
        assert!(!c0.intersects_aligned(&c1, &[10.0, 10.0].into()));
        // Boundaries are aligned
        assert!(c0.intersects_aligned(&c1, &[1.5, 1.5].into()));
        // Both at origin - will always intersect for any cuboids
        assert!(c0.intersects_aligned(&c1, &[0.0, 0.0].into()));
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

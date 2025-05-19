// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*!Axis-aligned N-cuboids, particularly used for bounding volume hierarchies.*/
use crate::intersects::IntersectsAt;
use crate::xenocollide::{collide2d, collide3d};
use crate::{Shape, Sphere, SupportFn, Volume};
use hoomd_vector::{Cartesian, RotationMatrix};
use hoomd_vector::{Rotate, Rotation};
use itertools::multizip;
use std::cmp::PartialEq;

/** An axis-aligned N-cuboid
*/
#[derive(Clone, Copy, Debug)]
pub struct Cuboid<const N: usize> {
    /// The lengths of each edge of the cuboid.
    pub edge_lengths: Cartesian<N>, // TODO: use PositiveReal
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

impl<const N: usize> Shape<N> for Cuboid<N> {
    #[inline]
    fn bounding_sphere(&self) -> Sphere<N> {
        Sphere::from(self.edge_lengths.into_iter().fold(f64::NAN, f64::max))
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

impl<R> IntersectsAt<Cuboid<2>, Cartesian<2>, R> for Cuboid<2>
where
    R: Rotate<Cartesian<2>> + Rotation + PartialEq + Copy,
    RotationMatrix<2>: From<R>,
{
    type OptionalRotation = Option<R>;

    #[inline]
    fn intersects_at(
        &self,
        other: &Cuboid<2>,
        v_ij: &Cartesian<2>,
        o_ij: &Self::OptionalRotation,
    ) -> bool {
        match o_ij {
            None => aabb_intersects(self, other, v_ij),
            Some(rotation) => collide2d(self, other, v_ij, rotation),
        }
    }
}

impl<R> IntersectsAt<Cuboid<3>, Cartesian<3>, R> for Cuboid<3>
where
    R: Rotate<Cartesian<3>> + Rotation + PartialEq + Copy,
    RotationMatrix<3>: From<R>,
{
    type OptionalRotation = Option<R>;

    #[inline]
    fn intersects_at(
        &self,
        other: &Cuboid<3>,
        v_ij: &Cartesian<3>,
        o_ij: &Self::OptionalRotation,
    ) -> bool {
        match o_ij {
            None => aabb_intersects(self, other, v_ij),
            Some(rotation) => collide3d(self, other, v_ij, rotation),
        }
    }
}

/// Determine whether two *axis-aligned* cuboids intersect.
#[inline]
fn aabb_intersects<const N: usize>(a: &Cuboid<N>, b: &Cuboid<N>, v_ij: &Cartesian<N>) -> bool {
    let b_mins = b.minimal_extents() + *v_ij;
    let b_maxs = b.maximal_extents() + *v_ij;
    multizip((a.minimal_extents(), b_maxs, a.maximal_extents(), b_mins))
        .all(|(a_min, b_max, a_max, b_min)| (a_min <= b_max) && (a_max >= b_min))
}

#[cfg(test)]
#[allow(clippy::used_underscore_binding)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use hoomd_vector::{Angle, Versor};
    use rstest::*;
    use std::marker::PhantomData;

    #[rstest(
        edges0 => [[2.0, 2.0, 2.0]],
        edges1 => [[1.0, 1.0, 1.0]],
    )]
    fn test_box_intersections_aligned(edges0: [f64; 3], edges1: [f64; 3]) {
        let (s0, s1) = (Cuboid::<3>::from(edges0), Cuboid::<3>::from(edges1));
        // Should all be false (no intersection), which we invert to true
        assert!(!s0.intersects_at(&s1, &[10.0, 10.0, 10.0].into(), &None::<Versor>));
        // Boundaries are aligned
        assert!(s0.intersects_at(&s1, &[1.5, 1.5, 1.5].into(), &None::<Versor>));
        // Both at origin - will always intersect for any cuboids
        assert!(s0.intersects_at(&s1, &[0.0, 0.0, 0.0].into(), &None::<Versor>));
        // TODO: is there a more programmatic way to test this?
    }
    #[rstest(
        edges0 => [[2.0, 2.0]],
        edges1 => [[1.0, 1.0]],
    )]
    fn test_box_intersections_2d_aligned(edges0: [f64; 2], edges1: [f64; 2]) {
        let (c0, c1) = (Cuboid::<2>::from(edges0), Cuboid::<2>::from(edges1));
        // Should all be false (no intersection), which we invert to true
        assert!(!c0.intersects_at(&c1, &[10.0, 10.0].into(), &None::<Angle>));
        // Boundaries are aligned
        assert!(c0.intersects_at(&c1, &[1.5, 1.5].into(), &None::<Angle>));
        // Both at origin - will always intersect for any cuboids
        assert!(c0.intersects_at(&c1, &[0.0, 0.0].into(), &None::<Angle>));
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
        _n => [
            PhantomData::<Cuboid<1>>,
            PhantomData::<Cuboid<2>>,
            PhantomData::<Cuboid<3>>,
            PhantomData::<Cuboid<4>>
        ],
        l => [0.0, 1e-6, 1.0, 3.456, 99_999_999.9],
    )]
    fn test_box_bounding_sphere<const N: usize>(_n: PhantomData<Cuboid<N>>, l: f64) {
        let c = Cuboid::from([l; N]);
        assert_relative_eq!(c.bounding_sphere().r, l);
    }

    #[rstest(
        l => [1e-6, 1.0, 3.456, 99_999_999.9],
    )]
    fn test_box_abc(l: f64) {
        let c = Cuboid::from([l; 3]);
        assert_eq!([c.a(), c.b(), c.c()], [l; 3]);
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::{
    BoundingShape, BoundingSphere, IntersectsAt, SupportMapping,
    xenocollide::{collide2d, collide3d},
};
use hoomd_vector::{Cartesian, Rotate, Rotation, RotationMatrix, Vector};

/// TODO
#[derive(Clone, Debug, PartialEq)]
pub struct Convex<S>(pub S);

impl<V: Vector, S> SupportMapping<V> for Convex<S>
where
    S: SupportMapping<V>,
{
    #[inline]
    fn support_mapping(&self, n: &V) -> V {
        self.0.support_mapping(n)
    }
}

impl<A, B, R, S> IntersectsAt<Convex<A>, Cartesian<2>, R> for Convex<B>
where
    A: SupportMapping<Cartesian<2>> + BoundingShape<Cartesian<2>, R, Shape = S>,
    B: SupportMapping<Cartesian<2>> + BoundingShape<Cartesian<2>, R, Shape = S>,
    S: IntersectsAt<S, Cartesian<2>, R>,
    R: Rotate<Cartesian<2>> + Rotation + PartialEq + Copy,
    RotationMatrix<2>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &Convex<A>, v_ij: &Cartesian<2>, o_ij: &R) -> bool {
        if !self
            .0
            .bounding_shape()
            .intersects_at(&other.0.bounding_shape(), v_ij, o_ij)
        {
            return false;
        }
        collide2d(self, other, v_ij, o_ij)
    }
}
impl<A, B, R, S> IntersectsAt<Convex<A>, Cartesian<3>, R> for Convex<B>
where
    A: SupportMapping<Cartesian<3>> + BoundingShape<Cartesian<3>, R, Shape = S>,
    B: SupportMapping<Cartesian<3>> + BoundingShape<Cartesian<3>, R, Shape = S>,
    S: IntersectsAt<S, Cartesian<3>, R>,
    R: Rotate<Cartesian<3>> + Rotation + PartialEq + Copy,
    RotationMatrix<3>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &Convex<A>, v_ij: &Cartesian<3>, o_ij: &R) -> bool {
        if !self
            .0
            .bounding_shape()
            .intersects_at(&other.0.bounding_shape(), v_ij, o_ij)
        {
            return false;
        }
        collide3d(self, other, v_ij, o_ij)
    }
}

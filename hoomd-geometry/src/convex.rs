// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::{
    IntersectsAt, SupportMapping,
    xenocollide::{collide2d, collide3d},
};
use hoomd_vector::{Cartesian, Rotate, Rotation, RotationMatrix, Vector};

/// TODO
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

impl<A, B, R> IntersectsAt<Convex<A>, Cartesian<3>, R> for Convex<B>
where
    A: SupportMapping<Cartesian<3>>,
    B: SupportMapping<Cartesian<3>>,
    R: Rotate<Cartesian<3>> + Rotation + PartialEq + Copy,
    RotationMatrix<3>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &Convex<A>, v_ij: &Cartesian<3>, o_ij: &R) -> bool {
        collide3d(self, other, v_ij, o_ij)
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement separating planes overlap check for `ConvexPolygon`.

use crate::{BoundingSphereRadius, IntersectsAt, IntersectsAtGlobal};
use super::ConvexPolygon;
use hoomd_vector::{Cartesian, InnerProduct, Metric, Rotate, Rotation, RotationMatrix};

impl<R> IntersectsAtGlobal<Self, Cartesian<2>, R> for ConvexPolygon
where
    R: Rotation + Rotate<Cartesian<2>>,
    RotationMatrix<2>: From<R>,
    R: Copy,
{
    #[inline]
    fn intersects_at_global(
        &self,
        other: &Self,
        r_self: &Cartesian<2>,
        o_self: &R,
        r_other: &Cartesian<2>,
        o_other: &R,
    ) -> bool {
        let max_separation =
            self.bounding_sphere_radius().get() + other.bounding_sphere_radius().get();
        if r_self.distance_squared(r_other) >= max_separation.powi(2) {
            return false;
        }

        let (v_ij, o_ij) = hoomd_vector::pair_system_to_local(r_self, o_self, r_other, o_other);

        self.intersects_at(other, &v_ij, &o_ij)
    }
}

impl<R> IntersectsAt<Self, Cartesian<2>, R> for ConvexPolygon
where
    RotationMatrix<2>: From<R>,
    R: Copy,
{
    /// TODO: Example
    #[inline]
    fn intersects_at(&self, other: &Self, v_ij: &Cartesian<2>, o_ij: &R) -> bool {

        let o_j = RotationMatrix::from(*o_ij);
        if b_edge_separates(self, other, v_ij, &o_j) {
            return false;
        }
        
        let o_j_inverted = o_j.inverted();
        let v_ji = o_j_inverted.rotate(&-*v_ij);
        if b_edge_separates(other, self, &v_ji, &o_j_inverted) {
            return false;
        }

        true
    }
}

/// Determine if any edge of `b` separates the points in `a` and `b`.
fn b_edge_separates(a: &ConvexPolygon, b: &ConvexPolygon,
                    v_ab: &Cartesian<2>,
                    o_b: &RotationMatrix<2>) -> bool
    {
    // SAFETY: Do not call this method if there are zero or one vertices (TODO)
    let mut previous = b.vertices.len() - 1;
    for current in 0..b.vertices.len()
        {
        let p = b.vertices[current];
        let edge = p - b.vertices[previous];

        // SAFETY: Vertices must be counter-clockwise ordered.
        let n = -edge.perpendicular();

        let p_in_frame_a = o_b.rotate(&p) + *v_ab;
        let n_in_frame_a = o_b.rotate(&n);

        // is this a separating plane?
        if is_separating(a, &p_in_frame_a, &n_in_frame_a)
            {
            return true;
            }

        // save previous vertex for next iteration
        previous = current;
        }

    false
    }

/// Determine if all of a's vertices are outside the given plane.
fn is_separating(a: &ConvexPolygon, p: &Cartesian<2>, n: &Cartesian<2>) -> bool {

    // check if n dot (v[i]-p) < 0 for every vertex in the polygon
    // distribute: (n dot v[i] - n dot p) < 0
    let n_dot_p = n.dot(p);

    for v in &a.vertices {
        if n.dot(v) - n_dot_p <= 0.0 {
            return false;
        } 
    }
    
    true
}

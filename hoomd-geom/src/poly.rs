// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::{IntersectsAt, Shape, SupportFn, Volume, xenocollide};
use hoomd_vector::{Cartesian, Rotate, Rotation, Vector};

/**
A convex, faceter polyhedron
*/
pub struct ConvexPolytope<const N: usize> {
    /// The vertices of the shape.
    vertices: Vec<Cartesian<N>>,
    // rounding_radius: f64,
    // minimal_centered_bounding_sphere_radius: f64,
}

/**
Calculate the intersection between two convex polygons in cartesian coordinates.
*/
impl<R: Rotate<Cartesian<2>>> IntersectsAt<Self, Cartesian<2>, R> for ConvexPolytope<2>
where
    R: Rotate<Cartesian<2>> + Copy + Rotation,
{
    fn intersects_at(&self, other: &Self, r_ij: &Cartesian<2>, o_ij: &R) -> bool {
        xenocollide::collide2d(self, other, r_ij, o_ij)
    }
}

#[allow(clippy::expect_used)]
impl<const N: usize> SupportFn<Cartesian<N>> for ConvexPolytope<N> {
    fn support(&self, n: &Cartesian<N>) -> Cartesian<N> {
        *self
            .vertices
            .iter()
            .max_by(|a, b| {
                a.dot(n)
                    .partial_cmp(&b.dot(n))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("Support function not valid with 0 vertices!")
    }
}

/**
Calculate the intersection between two convex polyhedra in cartesian coordinates.
*/
impl<R: Rotate<Cartesian<3>>> IntersectsAt<Self, Cartesian<3>, R> for ConvexPolytope<3> {
    ///
    fn intersects_at(&self, other: &Self, r_ij: &Cartesian<3>, o_ij: &R) -> bool {
        todo!() // TODO: Xenocollide 3d
    }
}

impl ConvexPolytope<2> {
    /// Xeonocolloide 2d
    fn mpr<R: Rotate<Cartesian<2>>>(&self, other: &Self, r_ij: &Cartesian<2>, o_ij: &R) -> bool {
        // 1a: Determine whether the origin lies in B⊖A, given only the support mapping
        // 1b: Obtain a point that lies deep in B⊖A:
        let p = r_ij; // self.centroid()-other.centroid() in extrinsic coords

        // 1c: Construct a normal pointing from p to the origin: this is just p̂?
        // Find support point in this direction

        false
    }
}

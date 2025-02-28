use crate::{IntersectsAt, Shape, Volume};
use hoomd_vector::{Cartesian, Rotate, Vector};

/**
A convex, faceter polyhedron
*/
pub struct ConvexPolytope<const N: usize> {
    /// The vertices of the shape.
    vertices: Vec<Cartesian<N>>,
    // rounding_radius: f64,
    // minimal_centered_bounding_sphere_radius: f64,
}
pub struct Sphero<S> {
    pub shape: S,
    pub rounding_radius: f64,
}

// impl ConvexPolytope<N> {
//   pub fn support_fn(&self)
// }

/**
Calculate the intersection between two convex polygons in cartesian coordinates.
*/
impl<R: Rotate<Cartesian<2>>> IntersectsAt<Self, Cartesian<2>, R> for ConvexPolytope<2> {
    ///
    fn intersects_at(&self, other: &Self, r_ij: &Cartesian<2>, o_ij: &R) -> bool {
        todo!() // TODO: Xenocollide 2d
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

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::{
    IntersectsAt, SupportFn,
    xenocollide::{self, collide3d},
};
use hoomd_vector::{Cartesian, Rotate, Rotation, RotationMatrix, Vector};

/**
A convex, faceter polyhedron
*/
pub struct ConvexPolytope<const N: usize> {
    /// The vertices of the shape.
    pub vertices: Vec<Cartesian<N>>,
}

/**
Calculate the intersection between two convex polygons in cartesian coordinates.
*/
impl<R: Rotate<Cartesian<2>>> IntersectsAt<Self, Cartesian<2>, R> for ConvexPolytope<2>
where
    R: Copy + Rotation,
    RotationMatrix<2>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &Self, r_ij: &Cartesian<2>, o_ij: &R) -> bool {
        xenocollide::collide2d(self, other, r_ij, o_ij)
    }
}

impl From<usize> for ConvexPolytope<2> {
    /** Create a regular N-gon with N vertices and circumradius one.

    # Example
    ```
    use hoomd_geom::poly::ConvexPolytope;

    let equilateral_triangle = ConvexPolytope::from(3);
    ```
    */
    #[inline]
    fn from(n: usize) -> ConvexPolytope<2> {
        ConvexPolytope {
            vertices: (0..n)
                .map(|x| {
                    let theta = std::f64::consts::PI * (x as f64) / (n as f64);
                    Cartesian::from([f64::cos(theta), f64::cos(theta)])
                })
                .collect::<Vec<_>>(),
        }
    }
}

impl<const N: usize> From<Vec<Cartesian<N>>> for ConvexPolytope<N> {
    /** Create a regular N-gon with N vertices and circumradius one.

    # Example
    ```
    use hoomd_geom::poly::ConvexPolytope;

    let equilateral_triangle = ConvexPolytope::from(3);
    ```
    */
    #[inline]
    fn from(vertices: Vec<Cartesian<N>>) -> ConvexPolytope<N> {
        ConvexPolytope { vertices }
    }
}

impl<const N: usize> FromIterator<Cartesian<N>> for ConvexPolytope<N> {
    /// Create a `ConvexPolytope` from an iterator of vertices.
    #[inline]
    fn from_iter<I: IntoIterator<Item = Cartesian<N>>>(iter: I) -> ConvexPolytope<N> {
        ConvexPolytope {
            vertices: iter.into_iter().collect::<Vec<_>>(),
        }
    }
}

#[allow(clippy::expect_used)]
impl<const N: usize> SupportFn<Cartesian<N>> for ConvexPolytope<N> {
    #[inline]
    fn support(&self, n: &Cartesian<N>) -> Cartesian<N> {
        let n = *n / n.norm(); // TODO: does this need to be normalized?
        *self
            .vertices
            .iter()
            .max_by(|a, b| {
                a.dot(&n)
                    .partial_cmp(&b.dot(&n))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("Support function not valid with 0 vertices!")
        // TODO: could test hand-coded SIMD here
    }
}

/**
Calculate the intersection between two convex polyhedra in cartesian coordinates.
*/
impl<R: Rotate<Cartesian<3>> + Rotation + Copy> IntersectsAt<Self, Cartesian<3>, R>
    for ConvexPolytope<3>
{
    ///
    #[inline]
    fn intersects_at(&self, other: &Self, v_ij: &Cartesian<3>, o_ij: &R) -> bool {
        // collide3d(self, &other, v_ij, o_ij)
        todo!()
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! N-Dimensional generalization of a convex polyhedron.*/
use crate::{
    Error, IntersectsAt, SupportMapping,
    xenocollide::{self, collide3d},
};
use hoomd_vector::{Cartesian, Rotate, Rotation, RotationMatrix, Vector};

/**
A convex, faceted polyhedron.
*/
pub struct ConvexPolytope<const N: usize> {
    /// The vertices of the shape.
    vertices: Vec<Cartesian<N>>,
    /// The radius of a bounding sphere of the geometry.
    bounding_radius: f64,
}

/**A two-dimensional faceted convex body.

```rust

use hoomd_geometry::shape::ConvexPolygon;
let poly = ConvexPolygon::from(6); // A regular hexagon
```
*/
pub type ConvexPolygon = ConvexPolytope<2>;
/**A three-dimensional faceted convex body.

```rust
use hoomd_geometry::shape::{ConvexPolyhedron, Simplex3};
# fn main() -> Result<(), hoomd_geometry::Error> {
// Create a regular tetrahedron from its vertices
let poly = ConvexPolyhedron::try_from(
    vec![
        [1.0, 1.0, 1.0].into(),
        [1.0, -1.0, -1.0].into(),
        [-1.0, 1.0, -1.0].into(),
        [-1.0, -1.0, 1.0].into(),
    ]
)?;

assert_eq!(poly.vertices(), Simplex3::default().vertices());
# Ok(())
# }
```
*/
pub type ConvexPolyhedron = ConvexPolytope<3>;

impl<const N: usize> ConvexPolytope<N> {
    /// The vertices of the shape.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> Vec<Cartesian<N>> {
        self.vertices.clone()
    }
}

/**
Calculate the intersection between two convex polygons in cartesian coordinates.
*/
impl<S: SupportMapping<Cartesian<2>>, R: Rotate<Cartesian<2>>> IntersectsAt<S, Cartesian<2>, R>
    for ConvexPolytope<2>
where
    R: Copy + Rotation,
    RotationMatrix<2>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &S, v_ij: &Cartesian<2>, o_ij: &R) -> bool {
        xenocollide::collide2d(self, other, v_ij, o_ij)
    }
}

impl From<usize> for ConvexPolytope<2> {
    /** Create a regular N-gon with N vertices and circumradius one.

    # Example
    ```
    use hoomd_geometry::shape::ConvexPolytope;

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
            bounding_radius: 1.0,
        }
    }
}

// TODO: should be TryFrom! some input vertices may not be convex
impl<const N: usize> TryFrom<Vec<Cartesian<N>>> for ConvexPolytope<N> {
    type Error = Error;
    /** Create a regular N-gon with N vertices and circumradius one.

    # Example
    ```
    use hoomd_geometry::shape::ConvexPolytope;

    let equilateral_triangle = ConvexPolytope::from(3);
    ```
    # Errors

    * `[Error::NotFinite]` when a vertex is not finite.
    * `[Error::NotPositive]` when all vertices are at the origin.
    */
    #[inline]
    fn try_from(vertices: Vec<Cartesian<N>>) -> Result<ConvexPolytope<N>, Error> {
        // TODO: compute convex hull and assert convex!
        let bounding_radius = vertices
            .iter()
            .map(Cartesian::norm_squared)
            .fold(f64::NAN, |max, x| f64::max(max, x))
            .sqrt();
        if true {
            Ok(ConvexPolytope {
                vertices,
                bounding_radius,
            }) // TODO: currently no verification that vertices are convex
        } else {
            Err(Error::NotConvex())
        }
    }
}

impl<const N: usize> FromIterator<Cartesian<N>> for ConvexPolytope<N> {
    /// Create a `ConvexPolytope` from an iterator of vertices.
    #[inline]
    fn from_iter<I: IntoIterator<Item = Cartesian<N>>>(iter: I) -> ConvexPolytope<N> {
        ConvexPolytope {
            vertices: iter.into_iter().collect::<Vec<_>>(),
            bounding_radius: 1.0, // TODO: use real value!
        }
    }
}

#[expect(
    clippy::unwrap_used,
    reason = "Unwrap case is handled in a match statement."
)]
impl<const N: usize> SupportMapping<Cartesian<N>> for ConvexPolytope<N> {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        match N {
            0 => Cartesian::<N>::default(),
            1 => self.vertices[0],
            _ => *self
                .vertices
                .iter()
                .max_by(|a, b| {
                    a.dot(n)
                        .partial_cmp(&b.dot(n))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap(),
        }
    }
}

/**
Calculate the intersection between two convex polyhedra in cartesian coordinates.
*/
impl<S: SupportMapping<Cartesian<3>>, R: Rotate<Cartesian<3>> + Rotation + Copy>
    IntersectsAt<S, Cartesian<3>, R> for ConvexPolytope<3>
where
    RotationMatrix<3>: From<R>,
{
    /// Determine whether a convex polyhedron intersects another shape at some position and orientation.
    #[inline]
    fn intersects_at(&self, other: &S, v_ij: &Cartesian<3>, o_ij: &R) -> bool {
        collide3d(self, other, v_ij, o_ij)
    }
}

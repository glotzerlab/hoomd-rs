// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! N-Dimensional generalization of a convex polyhedron.*/
use crate::{
    BoundingSphereRadius, Error, SupportMapping,
};
use hoomd_vector::{Cartesian, Vector};


/**
A convex, faceted polyhedron.
*/
pub struct ConvexPolytope<const N: usize> {
    /// The vertices of the shape.
    vertices: Vec<Cartesian<N>>,
    /// The radius of a bounding sphere of the geometry.
    pub(crate) bounding_radius: f64,
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

impl From<usize> for ConvexPolytope<2> {
    /** Create a regular *n*-gon with *n* vertices and circumradius one.

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

impl<const N: usize> TryFrom<Vec<Cartesian<N>>> for ConvexPolytope<N> {
    type Error = Error;
    /** Create an `N`-polytope from a `Vector` of `Cartesian<N>`.

    # Example
    ```
    use hoomd_geometry::shape::ConvexPolytope;

    # fn main() -> Result<(), hoomd_geometry::Error> {
    let equilateral_triangle = ConvexPolytope::try_from(
        vec![
            [1.0, 0.0].into(),
            [0.5, f64::sqrt(3.0)/2.0].into(),
            [-0.5, f64::sqrt(3.0)/2.0].into(),
       ]
    )?;
    # Ok(())
    # }
    ```
    # Errors

    * `[Error::NotConvex]` when the set of input vertices is not convex.
    */
    #[inline]
    fn try_from(vertices: Vec<Cartesian<N>>) -> Result<ConvexPolytope<N>, Error> {
        // TODO: compute convex hull and assert convex!
        let bounding_radius = vertices
            .iter()
            .map(Cartesian::norm_squared)
            .fold(f64::NAN, f64::max)
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

impl<const N: usize> BoundingSphereRadius for ConvexPolytope<N> {
    #[inline]
    fn bounding_sphere_radius(&self) -> f64 {
        self.bounding_radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;
    #[fixture]
    fn simplex3() -> ConvexPolyhedron {
        ConvexPolyhedron::try_from(vec![
            [1.0, 1.0, 1.0].into(),
            [1.0, -1.0, -1.0].into(),
            [-1.0, 1.0, -1.0].into(),
            [-1.0, -1.0, 1.0].into(),
        ])
        .unwrap()
    }

    #[fixture]
    fn equilateral_triangle() -> ConvexPolytope<2> {
        ConvexPolytope::try_from(vec![
            [1.0, 0.0].into(),
            [0.5, f64::sqrt(3.0) / 2.0].into(),
            [-0.5, f64::sqrt(3.0) / 2.0].into(),
        ])
        .unwrap()
    }

    #[rstest]
    fn test_bounding_radius_computed(
        simplex3: ConvexPolyhedron,
        equilateral_triangle: ConvexPolygon,
    ) {
        assert_eq!(simplex3.bounding_radius, f64::sqrt(3.0));
        assert_eq!(equilateral_triangle.bounding_radius, f64::sqrt(1.0));
    }

    #[rstest]
    fn test_bounding_radius_regular_polygons(#[values(1, 3, 8, 64)] n: usize) {
        assert_eq!(ConvexPolygon::from(n).bounding_radius, 1.0);
        assert_eq!(ConvexPolytope::from(n).bounding_radius, 1.0);
    }
}

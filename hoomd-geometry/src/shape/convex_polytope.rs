// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! N-Dimensional generalization of a convex polyhedron.*/

use crate::{BoundingSphereRadius, Error, SupportMapping};
use hoomd_vector::{Cartesian, InnerProduct};

/** A faceted solid defined by the convex hull of its vertices.

# Examples

Construction and basic methods:
```
use hoomd_geometry::{BoundingSphereRadius, shape::{ConvexPolyhedron}};
use approx::assert_relative_eq;

# fn main() -> Result<(), hoomd_geometry::Error> {
let tetrahedron = ConvexPolyhedron::with_vertices(
    vec![
        [1.0, 1.0, 1.0].into(),
        [1.0, -1.0, -1.0].into(),
        [-1.0, 1.0, -1.0].into(),
        [-1.0, -1.0, 1.0].into(),
    ]
)?;

let bounding_radius = tetrahedron.bounding_sphere_radius();

assert_relative_eq!(bounding_radius, 3.0_f64.sqrt());
# Ok(())
# }
```

Intersection tests:
```
use hoomd_geometry::{Convex, IntersectsAt, shape::ConvexPolygon};
use hoomd_vector::{Cartesian, Angle};
use std::f64::consts::PI;

# fn main() -> Result<(), hoomd_geometry::Error> {
let rectangle = ConvexPolygon::with_vertices(
    [[-2.0, -1.0].into(),
     [2.0, -1.0].into(),
     [2.0, 1.0].into(),
     [-2.0, 1.0].into()])?;
let rectangle = Convex(rectangle);

assert_eq!(rectangle.intersects_at(&rectangle, &[0.0, 2.1].into(), &Angle::default()), false);
assert_eq!(rectangle.intersects_at(&rectangle, &[0.0, 2.1].into(), &Angle::from(PI/2.0)), true);
# Ok(())
# }
```
*/
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexPolytope<const N: usize> {
    /// The vertices of the shape.
    vertices: Vec<Cartesian<N>>,
    /// The radius of a bounding sphere of the geometry.
    pub(crate) bounding_radius: f64,
}

/**A faceted convex body in two dimensions.

```rust

use hoomd_geometry::shape::ConvexPolygon;

# fn main() -> Result<(), hoomd_geometry::Error> {
let hexagon = ConvexPolygon::regular(6);
let square = ConvexPolygon::with_vertices(
    [[-1.0, -1.0].into(),
     [1.0, -1.0].into(),
     [1.0, 1.0].into(),
     [-1.0, 1.0].into()])?;
# Ok(())
# }
```
*/
pub type ConvexPolygon = ConvexPolytope<2>;

/**A faceted convex body in three dimensions.

```
use hoomd_geometry::shape::{ConvexPolyhedron, Simplex3};
# fn main() -> Result<(), hoomd_geometry::Error> {
// Create a regular tetrahedron from its vertices
let poly = ConvexPolyhedron::with_vertices(
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

impl ConvexPolytope<2> {
    /** Create a regular *n*-gon with *n* vertices and circumradius one.

    # Example
    ```
    use hoomd_geometry::shape::ConvexPolytope;

    let equilateral_triangle = ConvexPolytope::regular(3);
    ```
    */
    #[inline]
    #[must_use]
    pub fn regular(n: usize) -> ConvexPolytope<2> {
        ConvexPolytope {
            vertices: (0..n)
                .map(|x| {
                    let theta = std::f64::consts::PI * (x as f64) / (n as f64);
                    Cartesian::from([f64::cos(theta), f64::sin(theta)])
                })
                .collect::<Vec<_>>(),
            bounding_radius: 1.0,
        }
    }
}

impl<const N: usize> ConvexPolytope<N> {
    /** Create an `N`-polytope with the given vertices.

    # Example
    ```
    use hoomd_geometry::shape::ConvexPolytope;

    # fn main() -> Result<(), hoomd_geometry::Error> {
    let equilateral_triangle = ConvexPolytope::with_vertices(
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

    * `[Error::DegeneratePolytope]` when no vertices are provided.
    */
    #[inline]
    pub fn with_vertices<I>(vertices: I) -> Result<ConvexPolytope<N>, Error>
    where
        I: IntoIterator<Item = Cartesian<N>>,
    {
        let vertices = vertices.into_iter().collect::<Vec<_>>();

        if vertices.is_empty() {
            return Err(Error::DegeneratePolytope);
        }

        let bounding_radius = vertices
            .iter()
            .map(Cartesian::norm_squared)
            .fold(0.0, f64::max)
            .sqrt();

        Ok(ConvexPolytope {
            vertices,
            bounding_radius,
        })
    }

    /// The vertices of the shape.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[Cartesian<N>] {
        &self.vertices
    }
}

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
                .expect("the 0 match statement should handle empty vectors"),
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
    use approx::assert_relative_eq;
    use rstest::*;

    #[fixture]
    fn simplex3() -> ConvexPolyhedron {
        ConvexPolyhedron::with_vertices(vec![
            [1.0, 1.0, 1.0].into(),
            [1.0, -1.0, -1.0].into(),
            [-1.0, 1.0, -1.0].into(),
            [-1.0, -1.0, 1.0].into(),
        ])
        .unwrap()
    }

    #[fixture]
    fn equilateral_triangle() -> ConvexPolytope<2> {
        ConvexPolytope::with_vertices(vec![
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
        assert_eq!(ConvexPolygon::regular(n).bounding_radius, 1.0);
        assert_eq!(ConvexPolytope::regular(n).bounding_radius, 1.0);
    }

    #[test]
    fn degenerate_polytope() {
        let result = ConvexPolytope::<3>::with_vertices([]);
        assert_eq!(result, Err(Error::DegeneratePolytope));
    }

    #[test]
    fn support_mapping_2d() {
        let cuboid = ConvexPolygon::with_vertices([
            [-1.0, -2.0].into(),
            [1.0, -2.0].into(),
            [1.0, 2.0].into(),
            [-1.0, 2.0].into(),
        ])
        .expect("hard-coded vertices form a polygon");

        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, 0.1])),
            [1.0, 2.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, -0.1])),
            [1.0, -2.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-0.1, 1.0])),
            [-1.0, 2.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-0.1, -1.0])),
            [-1.0, -2.0].into()
        );
    }

    #[test]
    fn support_mapping_3d() {
        let cuboid = ConvexPolyhedron::with_vertices([
            [-1.0, -2.0, 3.0].into(),
            [1.0, -2.0, 3.0].into(),
            [1.0, 2.0, 3.0].into(),
            [-1.0, 2.0, 3.0].into(),
            [-1.0, -2.0, -3.0].into(),
            [1.0, -2.0, -3.0].into(),
            [1.0, 2.0, -3.0].into(),
            [-1.0, 2.0, -3.0].into(),
        ])
        .expect("hard-coded vertices form a polygon");

        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, 0.1, 0.1])),
            [1.0, 2.0, 3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, 0.1, -0.1])),
            [1.0, 2.0, -3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, -0.1, 0.1])),
            [1.0, -2.0, 3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, -0.1, -0.1])),
            [1.0, -2.0, -3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-1.0, 0.1, 0.1])),
            [-1.0, 2.0, 3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-1.0, 0.1, -0.1])),
            [-1.0, 2.0, -3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-1.0, -0.1, 0.1])),
            [-1.0, -2.0, 3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-1.0, -0.1, -0.1])),
            [-1.0, -2.0, -3.0].into()
        );
    }

    // TODO: Test intersects_at
}

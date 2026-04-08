// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! N-Dimensional generalization of a convex polyhedron.

use serde::{Deserialize, Serialize};

use crate::{BoundingSphereRadius, Error, SupportMapping, hull_2d_grahamscan};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct};

/// A faceted solid defined by the convex hull of its vertices.
///
/// # Examples
///
/// Construction and basic methods:
/// ```
/// use approxim::assert_relative_eq;
/// use hoomd_geometry::{BoundingSphereRadius, shape::ConvexPolyhedron};
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let tetrahedron = ConvexPolyhedron::with_vertices(vec![
///     [1.0, 1.0, 1.0].into(),
///     [1.0, -1.0, -1.0].into(),
///     [-1.0, 1.0, -1.0].into(),
///     [-1.0, -1.0, 1.0].into(),
/// ])?;
///
/// let bounding_radius = tetrahedron.bounding_sphere_radius();
///
/// assert_relative_eq!(bounding_radius.get(), 3.0_f64.sqrt());
/// # Ok(())
/// # }
/// ```
///
/// Intersection tests:
/// ```
/// use hoomd_geometry::{Convex, IntersectsAt, shape::ConvexPolygon};
/// use hoomd_vector::{Angle, Cartesian};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let rectangle = ConvexPolygon::with_vertices([
///     [-2.0, -1.0].into(),
///     [2.0, -1.0].into(),
///     [2.0, 1.0].into(),
///     [-2.0, 1.0].into(),
/// ])?;
/// let rectangle = Convex(rectangle);
///
/// assert!(!rectangle.intersects_at(
///     &rectangle,
///     &[0.0, 2.1].into(),
///     &Angle::default()
/// ));
/// assert!(rectangle.intersects_at(
///     &rectangle,
///     &[0.0, 2.1].into(),
///     &Angle::from(PI / 2.0)
/// ));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConvexPolytope<const N: usize> {
    /// The vertices of the shape.
    vertices: Vec<Cartesian<N>>,
    /// The radius of a bounding sphere of the geometry.
    pub(crate) bounding_radius: PositiveReal,
}

/// A faceted convex body in two dimensions.
///
/// ```rust
/// use hoomd_geometry::shape::ConvexPolygon;
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let hexagon = ConvexPolygon::regular(6);
/// let square = ConvexPolygon::with_vertices([
///     [-1.0, -1.0].into(),
///     [1.0, -1.0].into(),
///     [1.0, 1.0].into(),
///     [-1.0, 1.0].into(),
/// ])?;
/// # Ok(())
/// # }
/// ```
pub type ConvexPolygon = ConvexPolytope<2>;

/// A faceted convex body in three dimensions.
///
/// # Example
///
/// ```
/// use hoomd_geometry::shape::{ConvexPolyhedron, Simplex3};
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let poly = ConvexPolyhedron::with_vertices(vec![
///     [1.0, 1.0, 1.0].into(),
///     [1.0, -1.0, -1.0].into(),
///     [-1.0, 1.0, -1.0].into(),
///     [-1.0, -1.0, 1.0].into(),
/// ])?;
///
/// assert_eq!(poly.vertices(), Simplex3::default().vertices());
/// # Ok(())
/// # }
/// ```
pub type ConvexPolyhedron = ConvexPolytope<3>;

impl ConvexPolytope<2> {
    /// Create a 2D convex polygon with the given vertices.
    ///
    /// The vertices are reduced to their convex hull using a Graham scan.
    ///
    /// # Errors
    ///
    /// Returns `[Error::DegeneratePolytope]` when the hull has fewer than 3 vertices.
    #[inline]
    pub fn with_vertices_hull<I>(vertices: I) -> Result<ConvexPolytope<2>, Error>
    where
        I: IntoIterator<Item = Cartesian<2>>,
    {
        let mut vertices: Vec<Cartesian<2>> = vertices.into_iter().collect();

        if vertices.is_empty() {
            return Err(Error::DegeneratePolytope);
        }
        if vertices.len() < 3 {
            return Err(Error::DegeneratePolytope);
        }

        hull_2d_grahamscan(&mut vertices)?;

        Ok(ConvexPolytope {
            bounding_radius: Self::bounding_radius(&vertices),
            vertices,
        })
    }

    /// Create a regular *n*-gon with *n* vertices and circumradius 0.5.
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::shape::ConvexPolytope;
    ///
    /// let equilateral_triangle = ConvexPolytope::regular(3);
    /// ```
    #[inline]
    #[must_use]
    #[expect(
        clippy::missing_panics_doc,
        reason = "panic will never occur on a hard-coded constant"
    )]
    pub fn regular(n: usize) -> ConvexPolytope<2> {
        ConvexPolytope {
            vertices: (0..n)
                .map(|x| {
                    let theta = 2.0 * std::f64::consts::PI * (x as f64) / (n as f64);
                    Cartesian::from([0.5 * f64::cos(theta), 0.5 * f64::sin(theta)])
                })
                .collect::<Vec<_>>(),
            bounding_radius: 0.5
                .try_into()
                .expect("hard-coded constant should be positive"),
        }
    }
}

impl<const N: usize> ConvexPolytope<N> {
    /// Create an `N`-polytope with the given vertices.
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::shape::ConvexPolytope;
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let equilateral_triangle = ConvexPolytope::with_vertices(vec![
    ///     [1.0, 0.0].into(),
    ///     [0.5, f64::sqrt(3.0) / 2.0].into(),
    ///     [-0.5, f64::sqrt(3.0) / 2.0].into(),
    /// ])?;
    /// # Ok(())
    /// # }
    /// ```
    /// # Errors
    ///
    /// `[Error::DegeneratePolytope]` when no vertices are provided.
    #[inline]
    pub fn with_vertices<I>(vertices: I) -> Result<ConvexPolytope<N>, Error>
    where
        I: IntoIterator<Item = Cartesian<N>>,
    {
        let vertices = vertices.into_iter().collect::<Vec<_>>();

        if vertices.len() < (N + 1) {
            return Err(Error::DegeneratePolytope);
        }

        Ok(ConvexPolytope {
            bounding_radius: Self::bounding_radius(&vertices),
            vertices,
        })
    }

    /// The vertices of the shape.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[Cartesian<N>] {
        &self.vertices
    }

    /// Compute the bounding radius.
    fn bounding_radius(vertices: &[Cartesian<N>]) -> PositiveReal {
        vertices
            .iter()
            .map(Cartesian::norm_squared)
            .fold(0.0, f64::max)
            .sqrt()
            .try_into()
            .expect("convex polytope should have a positive bounding radius")
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
    fn bounding_sphere_radius(&self) -> PositiveReal {
        self.bounding_radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Convex, IntersectsAt};
    use hoomd_vector::{Angle, Cartesian, Rotate, Rotation, Versor};

    use approxim::assert_relative_eq;
    use rstest::*;
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

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
        assert_eq!(simplex3.bounding_radius.get(), f64::sqrt(3.0));
        assert_eq!(equilateral_triangle.bounding_radius.get(), f64::sqrt(1.0));
    }

    #[rstest]
    fn test_bounding_radius_regular_polygons(#[values(1, 3, 8, 64)] n: usize) {
        assert_eq!(ConvexPolygon::regular(n).bounding_radius.get(), 0.5);
        assert_eq!(ConvexPolytope::regular(n).bounding_radius.get(), 0.5);
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

    // ConvexPolygon tests from hoomd-blue's test_convex_polygon.cc

    #[fixture]
    fn square() -> Convex<ConvexPolygon> {
        Convex(
            ConvexPolygon::with_vertices([
                [-0.5, -0.5].into(),
                [0.5, -0.5].into(),
                [0.5, 0.5].into(),
                [-0.5, 0.5].into(),
            ])
            .expect("hard-coded vertices form a valid polygon"),
        )
    }

    #[fixture]
    fn triangle() -> Convex<ConvexPolygon> {
        Convex(
            ConvexPolygon::with_vertices([
                [-0.5, -0.5].into(),
                [0.5, -0.5].into(),
                [0.5, 0.5].into(),
            ])
            .expect("hard-coded vertices form a valid polygon"),
        )
    }

    #[rstest]
    fn square_no_rot(square: Convex<ConvexPolygon>) {
        let a = Angle::identity();
        assert!(!square.intersects_at(&square, &[10.0, 0.0].into(), &a));
        assert!(!square.intersects_at(&square, &[-10.0, 0.0].into(), &a));

        assert!(!square.intersects_at(&square, &[1.1, 0.0].into(), &a));
        assert!(!square.intersects_at(&square, &[-1.1, 0.0].into(), &a));
        assert!(!square.intersects_at(&square, &[0.0, 1.1].into(), &a));
        assert!(!square.intersects_at(&square, &[0.0, -1.1].into(), &a));

        assert!(square.intersects_at(&square, &[0.9, 0.2].into(), &a));
        assert!(square.intersects_at(&square, &[-0.9, 0.2].into(), &a));
        assert!(square.intersects_at(&square, &[-0.2, 0.9].into(), &a));
        assert!(square.intersects_at(&square, &[-0.2, -0.9].into(), &a));

        assert!(square.intersects_at(&square, &[1.0, 0.2].into(), &a));
    }

    #[rstest]
    fn square_rot(square: Convex<ConvexPolygon>) {
        let a = Angle::from(PI / 4.0);

        assert!(!square.intersects_at(&square, &[10.0, 0.0].into(), &a));
        assert!(!square.intersects_at(&square, &[-10.0, 0.0].into(), &a));

        assert!(!square.intersects_at(&square, &[1.3, 0.0].into(), &a));
        assert!(!square.intersects_at(&square, &[-1.3, 0.0].into(), &a));
        assert!(!square.intersects_at(&square, &[0.0, 1.3].into(), &a));
        assert!(!square.intersects_at(&square, &[0.0, -1.3].into(), &a));

        assert!(!square.intersects_at(&square, &[1.3, 0.2].into(), &a));
        assert!(!square.intersects_at(&square, &[-1.3, 0.2].into(), &a));
        assert!(!square.intersects_at(&square, &[-0.2, 1.3].into(), &a));
        assert!(!square.intersects_at(&square, &[-0.2, -1.3].into(), &a));

        assert!(square.intersects_at(&square, &[1.2, 0.2].into(), &a));
        assert!(square.intersects_at(&square, &[-1.2, 0.2].into(), &a));
        assert!(square.intersects_at(&square, &[-0.2, 1.2].into(), &a));
        assert!(square.intersects_at(&square, &[-0.2, -1.2].into(), &a));
    }

    fn test_overlap<A, B, R, const N: usize>(
        r_ab: Cartesian<N>,
        a: &A,
        b: &B,
        o_a: R,
        o_b: &R,
    ) -> bool
    where
        R: Rotation + Rotate<Cartesian<N>>,
        A: IntersectsAt<B, Cartesian<N>, R>,
    {
        let r_a_inverted = o_a.inverted();
        let v_ij = r_a_inverted.rotate(&r_ab);
        let o_ij = o_b.combine(&r_a_inverted);
        a.intersects_at(b, &v_ij, &o_ij)
    }

    fn assert_symmetric_overlap<A, B, R, const N: usize>(
        r_ab: Cartesian<N>,
        a: &A,
        b: &B,
        r_a: R,
        r_b: R,
        expected: bool,
    ) where
        R: Rotation + Rotate<Cartesian<N>>,
        A: IntersectsAt<B, Cartesian<N>, R>,
        B: IntersectsAt<A, Cartesian<N>, R>,
    {
        assert_eq!(test_overlap(r_ab, a, b, r_a, &r_b), expected);
        assert_eq!(test_overlap(-r_ab, b, a, r_b, &r_a), expected);
    }

    #[rstest]
    fn square_triangle(square: Convex<ConvexPolygon>, triangle: Convex<ConvexPolygon>) {
        let r_square = Angle::from(-PI / 4.0);
        let r_triangle = Angle::from(PI);

        assert_symmetric_overlap(
            [10.0, 0.0].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            false,
        );

        assert_symmetric_overlap(
            [1.3, 0.0].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            false,
        );

        assert_symmetric_overlap(
            [-1.3, 0.0].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            false,
        );

        assert_symmetric_overlap(
            [0.0, 1.3].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            false,
        );

        assert_symmetric_overlap(
            [0.0, -1.3].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            false,
        );

        assert_symmetric_overlap(
            [1.2, 0.2].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            true,
        );

        assert_symmetric_overlap(
            [-0.7, -0.2].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            true,
        );

        assert_symmetric_overlap(
            [0.4, 1.1].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            true,
        );

        assert_symmetric_overlap(
            [-0.2, -1.2].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            true,
        );
    }

    #[fixture]
    fn octahedron() -> Convex<ConvexPolyhedron> {
        Convex(
            ConvexPolyhedron::with_vertices([
                [-0.5, -0.5, 0.0].into(),
                [0.5, -0.5, 0.0].into(),
                [0.5, 0.5, 0.0].into(),
                [-0.5, 0.5, 0.0].into(),
                [0.0, 0.0, FRAC_1_SQRT_2].into(),
                [0.0, 0.0, -FRAC_1_SQRT_2].into(),
            ])
            .expect("hard-coded vertices form a valid polyhedron"),
        )
    }

    #[fixture]
    fn cube() -> Convex<ConvexPolyhedron> {
        Convex(
            ConvexPolyhedron::with_vertices([
                [-0.5, -0.5, -0.5].into(),
                [0.5, -0.5, -0.5].into(),
                [0.5, 0.5, -0.5].into(),
                [-0.5, 0.5, -0.5].into(),
                [-0.5, -0.5, 0.5].into(),
                [0.5, -0.5, 0.5].into(),
                [0.5, 0.5, 0.5].into(),
                [-0.5, 0.5, 0.5].into(),
            ])
            .expect("hard-coded vertices form a valid polyhedron"),
        )
    }

    #[rstest]
    fn overlap_octahedron_no_rot(octahedron: Convex<ConvexPolyhedron>) {
        let q = Versor::identity();

        assert_symmetric_overlap([0.0, 0.0, 0.0].into(), &octahedron, &octahedron, q, q, true);

        assert_symmetric_overlap(
            [10.0, 0.0, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap(
            [1.1, 0.0, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap(
            [0.0, 1.1, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap(
            [1.1, 0.2, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap(
            [-1.1, 0.2, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap(
            [-0.2, 1.1, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap(
            [-0.2, -1.1, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap([0.9, 0.2, 0.0].into(), &octahedron, &octahedron, q, q, true);

        assert_symmetric_overlap(
            [-0.9, 0.2, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            true,
        );

        assert_symmetric_overlap(
            [-0.2, 0.9, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            true,
        );

        assert_symmetric_overlap(
            [-0.2, -0.9, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            true,
        );

        assert_symmetric_overlap([1.0, 0.2, 0.0].into(), &octahedron, &octahedron, q, q, true);
    }

    #[rstest]
    fn overlap_cube_no_rot(cube: Convex<ConvexPolyhedron>) {
        let q = Versor::identity();

        assert_symmetric_overlap([0.0, 0.0, 0.0].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([10.0, 0.0, 0.0].into(), &cube, &cube, q, q, false);

        assert_symmetric_overlap([1.1, 0.0, 0.0].into(), &cube, &cube, q, q, false);
        assert_symmetric_overlap([0.0, 1.1, 0.0].into(), &cube, &cube, q, q, false);
        assert_symmetric_overlap([0.0, 0.0, 1.1].into(), &cube, &cube, q, q, false);
        assert_symmetric_overlap([1.1, 0.2, 0.0].into(), &cube, &cube, q, q, false);

        assert_symmetric_overlap([-1.1, 0.2, 0.0].into(), &cube, &cube, q, q, false);
        assert_symmetric_overlap([-0.2, 1.1, 0.0].into(), &cube, &cube, q, q, false);
        assert_symmetric_overlap([-0.2, -1.1, 0.0].into(), &cube, &cube, q, q, false);

        assert_symmetric_overlap([0.9, 0.2, 0.0].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([-0.9, 0.2, 0.0].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([-0.2, 0.9, 0.0].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([-0.2, -0.9, 0.0].into(), &cube, &cube, q, q, true);

        assert_symmetric_overlap([0.2, 0.0, 0.0].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([0.2, 0.00001, 0.00001].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([0.1, 0.2, 0.1].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([1.0, 0.2, 0.0].into(), &cube, &cube, q, q, true);
    }

    #[rstest]
    fn overlap_cube_rot1(cube: Convex<ConvexPolyhedron>) {
        let q_a = Versor::identity();
        let q_b = Versor::from_axis_angle(
            [0.0, 0.0, 1.0]
                .try_into()
                .expect("hard-coded vector is non-zero"),
            PI / 4.0,
        );

        assert_symmetric_overlap([10.0, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, false);

        assert_symmetric_overlap([1.3, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-1.3, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([0.0, 1.3, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([0.0, -1.3, 0.0].into(), &cube, &cube, q_a, q_b, false);

        assert_symmetric_overlap([1.3, 0.2, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-1.3, 0.2, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-0.2, 1.3, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-0.2, -1.3, 0.0].into(), &cube, &cube, q_a, q_b, false);

        assert_symmetric_overlap([1.2, 0.2, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([-1.2, 0.2, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([-0.2, 1.2, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([-0.2, -1.2, 0.0].into(), &cube, &cube, q_a, q_b, true);
    }

    #[rstest]
    fn overlap_cube_rot3(cube: Convex<ConvexPolyhedron>) {
        let q_a = Versor::identity();
        let q1 = Versor::from_axis_angle(
            [1.0, 0.0, 0.0]
                .try_into()
                .expect("hard-coded vector is non-zero"),
            PI / 4.0,
        );
        let q2 = Versor::from_axis_angle(
            [0.0, 0.0, 1.0]
                .try_into()
                .expect("hard-coded vector is non-zero"),
            PI / 4.0,
        );
        let q_b = q2.combine(&q1);

        assert_symmetric_overlap([10.0, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, false);

        assert_symmetric_overlap([1.4, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-1.4, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([0.0, 1.4, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([0.0, -1.4, 0.0].into(), &cube, &cube, q_a, q_b, false);

        assert_symmetric_overlap([1.4, 0.2, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-1.4, 0.2, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-0.2, 1.4, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-0.2, -1.4, 0.0].into(), &cube, &cube, q_a, q_b, false);

        assert_symmetric_overlap([0.0, 1.2, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([0.0, 1.2, 0.1].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([0.1, 1.2, 0.1].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([1.2, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([1.2, 0.1, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([1.2, 0.1, 0.1].into(), &cube, &cube, q_a, q_b, true);

        assert_symmetric_overlap([-0.9, 0.9, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([-0.9, 0.899, 0.001].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([0.9, -0.9, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([-0.9, 0.9, 0.1].into(), &cube, &cube, q_a, q_b, true);
    }
}

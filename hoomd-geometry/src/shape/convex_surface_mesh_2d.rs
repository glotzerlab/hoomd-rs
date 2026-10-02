// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Convex polygon represented by vertices and edges.

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    BoundingSphereRadius, Error, IntersectsAt, IntersectsAtGlobal, IsPointInside, Scale,
    SupportMapping, Volume, hull::ConvexHull, shape::ConvexPolytope,
};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct, Metric, Rotate, Rotation, RotationMatrix};

/// The vertices and edges that make up a convex polygon.
///
/// [`ConvexPolytope::<2>`] and [`ConvexSurfaceMesh2d`] can both represent
/// 2d convex polygons. The first is defined *implicitly* as the convex hull
/// of a set of points. It stores the given point set without any modification,
/// and can therefore be constructed quickly. The *implicit* convex hull is
/// formed by [`SupportMapping`] during intersection tests of
/// `Convex(ConvexPolygon)` with other `Convex(_)` types.
///
/// In contrast, [`ConvexSurfaceMesh2d`] *explicitly* computes the convex hull
/// on construction with [`ConvexHull`]. After construction, the [`vertices`] of
/// the shape include only the points on the convex hull in a counter-clockwise
/// order. Using this representation, [`ConvexSurfaceMesh2d`] is able to provide
/// implementations of [`Volume`], [`IsPointInside`], and [`IntersectsAt`].
/// The native `ConvexSurfaceMesh2d`--`ConvexSurfaceMesh2d` intersection test
/// is much faster than the generic `Convex(ConvexPolygon)` intersection test.
///
/// [`vertices`]: Self::vertices
///
/// # Examples
///
/// Construction:
/// ```
/// use hoomd_geometry::shape::ConvexSurfaceMesh2d;
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let triangle = ConvexSurfaceMesh2d::from_point_set([
///     [1.0, -1.0].into(),
///     [-1.0, -1.0].into(),
///     [0.0, 1.0].into(),
/// ])?;
/// # Ok(())
/// # }
/// ```
///
/// Intersection tests:
/// ```
/// use hoomd_geometry::{Convex, IntersectsAt, shape::ConvexSurfaceMesh2d};
/// use hoomd_vector::{Angle, Cartesian};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let rectangle = ConvexSurfaceMesh2d::from_point_set([
///     [-2.0, -1.0].into(),
///     [2.0, -1.0].into(),
///     [2.0, 1.0].into(),
///     [-2.0, 1.0].into(),
/// ])?;
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
pub struct ConvexSurfaceMesh2d {
    /// The vertices of the polygon in counter-clockwise order.
    vertices: Vec<Cartesian<2>>,
    /// The radius of a bounding sphere of the geometry.
    bounding_radius: PositiveReal,
}

impl ConvexSurfaceMesh2d {
    /// Create an convex surface mesh from the convex hull of the given set of points.
    ///
    /// The resulting shape contains a subset of the given points, including
    /// only the non-degenerate points on the convex hull. The result's vertices
    /// are arranged in a counter-clockwise order.
    ///
    /// # Errors
    ///
    /// * [`Error::DegeneratePolytope`] when there are fewer than 3 non-collinear points.
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::shape::ConvexSurfaceMesh2d;
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let equilateral_triangle = ConvexSurfaceMesh2d::from_point_set([
    ///     [1.0, 0.0].into(),
    ///     [0.5, f64::sqrt(3.0) / 2.0].into(),
    ///     [-0.5, f64::sqrt(3.0) / 2.0].into(),
    /// ])?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn from_point_set<I>(points: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = Cartesian<2>>,
    {
        // The edges of the hull are implied by the counter-clockwise order of the
        // vertices, so we just discard the facets.
        let (vertices, _edges) = Cartesian::<2>::convex_hull(points)?;

        Ok(Self {
            bounding_radius: ConvexPolytope::<2>::bounding_radius(&vertices),
            vertices,
        })
    }

    /// The vertices of the convex polygon in counter-clockwise order.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[Cartesian<2>] {
        &self.vertices
    }
}

impl SupportMapping<Cartesian<2>> for ConvexSurfaceMesh2d {
    /// Find the point on a shape that is the furthest in a given direction.
    ///
    /// [`ConvexSurfaceMesh2d`] implements [`SupportMapping`] to enable
    /// intersection tests between mixed convex types.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{
    ///     Convex, IntersectsAt,
    ///     shape::{Circle, ConvexSurfaceMesh2d, Sphero},
    /// };
    /// use hoomd_vector::{Angle, Cartesian};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let circle = Convex(Circle {
    ///     radius: 0.5.try_into()?,
    /// });
    /// let rectangle = ConvexSurfaceMesh2d::from_point_set([
    ///     [-1.5, -1.0].into(),
    ///     [1.5, -1.0].into(),
    ///     [-1.5, 1.0].into(),
    ///     [1.5, 1.0].into(),
    /// ])?;
    /// let rounded_rectangle = Convex(Sphero {
    ///     shape: rectangle,
    ///     rounding_radius: 0.5.try_into()?,
    /// });
    ///
    /// assert!(rounded_rectangle.intersects_at(
    ///     &circle,
    ///     &[2.4, 0.0].into(),
    ///     &Angle::default()
    /// ));
    /// assert!(!rounded_rectangle.intersects_at(
    ///     &circle,
    ///     &[0.0, 2.4].into(),
    ///     &Angle::default()
    /// ));
    /// assert!(circle.intersects_at(
    ///     &rounded_rectangle,
    ///     &[0.0, 2.4].into(),
    ///     &Angle::from(PI / 2.0)
    /// ));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn support_mapping(&self, n: &Cartesian<2>) -> Cartesian<2> {
        *self
            .vertices
            .iter()
            .max_by(|a, b| {
                a.dot(n)
                    .partial_cmp(&b.dot(n))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("there should be at least 3 vertices")
    }
}

impl BoundingSphereRadius for ConvexSurfaceMesh2d {
    /// Radius of a circle that bounds the shape.
    ///
    /// The circle has the same local origin as the shape `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{BoundingSphereRadius, shape::ConvexSurfaceMesh2d};
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let triangle = ConvexSurfaceMesh2d::from_point_set([
    ///     [1.0, -1.0].into(),
    ///     [-1.0, -1.0].into(),
    ///     [0.0, 1.0].into(),
    /// ])?;
    ///
    /// let bounding_radius = triangle.bounding_sphere_radius();
    ///
    /// assert_relative_eq!(bounding_radius.get(), 2.0_f64.sqrt());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        self.bounding_radius
    }
}

impl Volume for ConvexSurfaceMesh2d {
    /// Compute the area of the convex polygon.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{Volume, shape::ConvexSurfaceMesh2d};
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let triangle = ConvexSurfaceMesh2d::from_point_set([
    ///     [-1.0, -1.0].into(),
    ///     [1.0, -1.0].into(),
    ///     [1.0, 1.0].into(),
    /// ])?;
    ///
    /// let area = triangle.volume();
    ///
    /// assert_relative_eq!(area, 2.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn volume(&self) -> f64 {
        // Compute the polygon area with the shoelace formula:
        // https://mathworld.wolfram.com/PolygonArea.html
        0.5 * self
            .vertices
            .iter()
            .circular_tuple_windows()
            .fold(0.0, |total, (a, b)| total + a[0] * b[1] - b[0] * a[1])
    }
}

impl Scale for ConvexSurfaceMesh2d {
    /// Construct a scaled convex surface mesh.
    ///
    /// The resulting surface meshs' vertices $` \vec{p}_\mathrm{new} `$ are
    /// the original's $` \vec{p} `$ scaled by $` v `$:
    /// ```math
    /// \vec{p}_\mathrm{new} = v \vec}{p}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{
    ///     BoundingSphereRadius, Scale, Volume, shape::ConvexSurfaceMesh2d,
    /// };
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rectangle = ConvexSurfaceMesh2d::from_point_set([
    ///     [-2.0, -1.0].into(),
    ///     [2.0, -1.0].into(),
    ///     [2.0, 1.0].into(),
    ///     [-2.0, 1.0].into(),
    /// ])?;
    ///
    /// let scaled_rectangle = rectangle.scale_length(0.5.try_into()?);
    ///
    /// assert_eq!(scaled_rectangle.vertices()[0], [-1.0, -0.5].into());
    /// assert_eq!(scaled_rectangle.volume(), 2.0);
    /// assert_relative_eq!(
    ///     scaled_rectangle.bounding_sphere_radius().get(),
    ///     1.25_f64.sqrt()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        Self {
            vertices: self.vertices.iter().map(|&r| r * v.get()).collect(),
            bounding_radius: self.bounding_radius * v,
        }
    }

    /// Construct a scaled convex surface mesh.
    ///
    /// The resulting surface meshs' vertices $` \vec{p}_\mathrm{new} `$ are
    /// the original's $` \vec{p} `$ scaled by $` v^{\frac{1}{2}} `$:
    /// ```math
    /// \vec{p}_\mathrm{new} = v \vec}{p}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{
    ///     BoundingSphereRadius, Scale, Volume, shape::ConvexSurfaceMesh2d,
    /// };
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rectangle = ConvexSurfaceMesh2d::from_point_set([
    ///     [-2.0, -1.0].into(),
    ///     [2.0, -1.0].into(),
    ///     [2.0, 1.0].into(),
    ///     [-2.0, 1.0].into(),
    /// ])?;
    ///
    /// let scaled_rectangle = rectangle.scale_volume(0.25.try_into()?);
    ///
    /// assert_eq!(scaled_rectangle.vertices()[0], [-1.0, -0.5].into());
    /// assert_eq!(scaled_rectangle.volume(), 2.0);
    /// assert_relative_eq!(
    ///     scaled_rectangle.bounding_sphere_radius().get(),
    ///     1.25_f64.sqrt()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let v = v.get().powf(1.0 / 2.0);
        self.scale_length(v.try_into().expect("v^{1/N} should be a positive real"))
    }
}

impl IsPointInside<Cartesian<2>> for ConvexSurfaceMesh2d {
    /// Check if a point is inside the convex polygon.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::ConvexSurfaceMesh2d};
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let triangle = ConvexSurfaceMesh2d::from_point_set([
    ///     [1.0, -1.0].into(),
    ///     [-1.0, -1.0].into(),
    ///     [0.0, 1.0].into(),
    /// ])?;
    ///
    /// assert!(triangle.is_point_inside(&[0.0, 0.0].into()));
    /// assert!(triangle.is_point_inside(&[-0.9, -0.9].into()));
    /// assert!(!triangle.is_point_inside(&[-1.5, 2.0].into()));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn is_point_inside(&self, point: &Cartesian<2>) -> bool {
        for (a, b) in self.vertices.iter().circular_tuple_windows() {
            let edge = *b - *a;
            let n = -edge.perpendicular();

            let v = *point - *a;
            if v.dot(&n) > 0.0 {
                return false;
            }
        }

        true
    }
}

impl<R> IntersectsAtGlobal<Self, Cartesian<2>, R> for ConvexSurfaceMesh2d
where
    R: Rotation + Rotate<Cartesian<2>>,
    RotationMatrix<2>: From<R>,
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

impl<R> IntersectsAt<Self, Cartesian<2>, R> for ConvexSurfaceMesh2d
where
    RotationMatrix<2>: From<R>,
    R: Copy,
{
    /// Test convex polygon intersections using the separating planes method.
    ///
    /// When the number of vertices is small, separating planes is significantly
    /// faster than the Xenocollide algorithm implemented for the `Convex<ConvexPolygon>`
    /// type, even though separating planes is $` O(n^2) `$.
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::{IntersectsAt, shape::ConvexSurfaceMesh2d};
    /// use hoomd_vector::{Angle, Cartesian};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let rectangle = ConvexSurfaceMesh2d::from_point_set([
    ///     [-2.0, -1.0].into(),
    ///     [2.0, -1.0].into(),
    ///     [2.0, 1.0].into(),
    ///     [-2.0, 1.0].into(),
    /// ])?;
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
    #[inline]
    fn intersects_at(&self, other: &Self, v_ij: &Cartesian<2>, o_ij: &R) -> bool {
        assert!(
            self.vertices.len() > 2 && other.vertices.len() > 2,
            "A convex polygon must have at least 3 vertices."
        );

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
#[inline]
fn b_edge_separates(
    a: &ConvexSurfaceMesh2d,
    b: &ConvexSurfaceMesh2d,
    v_ab: &Cartesian<2>,
    o_b: &RotationMatrix<2>,
) -> bool {
    let mut previous = b.vertices.len() - 1;
    for current in 0..b.vertices.len() {
        let p = b.vertices[current];
        let edge = p - b.vertices[previous];

        let n = -edge.perpendicular();

        let p_in_frame_a = o_b.rotate(&p) + *v_ab;
        let n_in_frame_a = o_b.rotate(&n);

        if is_separating(a, &p_in_frame_a, &n_in_frame_a) {
            return true;
        }

        previous = current;
    }

    false
}

/// Determine if all of a's vertices are outside the given plane.
#[inline]
fn is_separating(a: &ConvexSurfaceMesh2d, p: &Cartesian<2>, n: &Cartesian<2>) -> bool {
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

impl<const MAX_VERTICES: usize> TryFrom<ConvexPolytope<2, MAX_VERTICES>> for ConvexSurfaceMesh2d {
    type Error = Error;

    /// Construct the convex hull of a [`ConvexPolytope<2>`].
    ///
    /// # Errors
    ///
    /// * [`Error::DegeneratePolytope`] when there are fewer than 3 non-collinear points.
    ///
    /// # Example
    ///
    /// Invalid conversion
    /// ```
    /// use hoomd_geometry::shape::{ConvexPolygon, ConvexSurfaceMesh2d};
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let equilateral_triangle = ConvexPolygon::regular(3);
    /// let mesh = ConvexSurfaceMesh2d::try_from(equilateral_triangle)?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn try_from(v: ConvexPolytope<2, MAX_VERTICES>) -> Result<ConvexSurfaceMesh2d, Error> {
        Self::from_point_set(v.vertices().iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use approxim::assert_relative_eq;
    use hoomd_vector::Angle;
    use rstest::*;

    #[test]
    fn support_mapping_2d() {
        let cuboid = ConvexSurfaceMesh2d::from_point_set([
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

    // ConvexPolygon tests from hoomd-blue's test_convex_polygon.cc

    #[fixture]
    fn square() -> ConvexSurfaceMesh2d {
        ConvexSurfaceMesh2d::from_point_set([
            [-0.5, -0.5].into(),
            [0.5, -0.5].into(),
            [0.5, 0.5].into(),
            [-0.5, 0.5].into(),
        ])
        .expect("hard-coded vertices form a valid polygon")
    }

    #[fixture]
    fn triangle() -> ConvexSurfaceMesh2d {
        ConvexSurfaceMesh2d::from_point_set([
            [-0.5, -0.5].into(),
            [0.5, -0.5].into(),
            [0.5, 0.5].into(),
        ])
        .expect("hard-coded vertices form a valid polygon")
    }

    #[rstest]
    fn square_no_rot(square: ConvexSurfaceMesh2d) {
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
    fn square_rot(square: ConvexSurfaceMesh2d) {
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
    fn square_triangle(square: ConvexSurfaceMesh2d, triangle: ConvexSurfaceMesh2d) {
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
}

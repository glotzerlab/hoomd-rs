// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Convex polygon represented by vertices and edges.

use std::cmp::Ordering;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    BoundingSphereRadius, Error, IntersectsAt, IntersectsAtGlobal, IsPointInside, SupportMapping,
    Volume, shape::ConvexPolytope,
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
/// on construction. After construction, the [`vertices`] of the shape include
/// only the points on the convex hull in a counter-clockwise order.
/// Using this representation, [`ConvexSurfaceMesh2d`] is able to provide
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

/// Find the lowest, leftmost point from a slice of Cartesian vectors.
#[inline]
fn find_lowest_leftmost(vertices: &[Cartesian<2>]) -> Option<usize> {
    vertices.iter().position_min_by(|a, b| {
        // Compare y-coordinates, then x.
        a[1].total_cmp(&b[1]).then(a[0].total_cmp(&b[0]))
    })
}

/// Get the key for a lexicographic order of points with respect to an anchor.
#[inline]
fn get_graham_key(p: Cartesian<2>, anchor: Cartesian<2>) -> (f64, f64) {
    let diff = p - anchor;
    (f64::atan2(diff[1], diff[0]), diff.dot(&diff))
}

/// Determines whether a point `test` is to the left, right, or collinear with `edge`.
///
/// # Warning
///
/// This predicate is not robust: points very close to the line may be misclassified
/// due to floating-point precision limits. For all practical inputs, this will not
/// result in issues.
///
/// # Note
///
/// This formulation (often referred to as the shoelace formula) guarantees
/// **cyclic invariance**, or the property that the orientation sign is identical
/// for any ordering of the three points `(e0, e1, t)`, `(e1, t, e0)`, and
/// `(t, e0, e1)`. As a result, the result is antisymmetric about the edge `e`
/// such that `p == -p'` for any `p'` reflected over `e`.
///
/// These properties do *not* prevent misclassification of points near the line—they
/// only ensure consistent behavior for related inputs.
///
/// **Source:** [Robust Arithmetic in Computational Geometry](https://observablehq.com/@mourner/non-robust-arithmetic-as-art)
#[inline]
fn predicate_orient2d((p, q): (Cartesian<2>, Cartesian<2>), test: Cartesian<2>) -> i64 {
    let orientation = (p[0] * q[1] - p[1] * q[0])
        + (q[0] * test[1] - q[1] * test[0])
        + (test[0] * p[1] - test[1] * p[0]);

    match orientation.total_cmp(&0.0) {
        Ordering::Greater => 1,
        Ordering::Less => -1,
        Ordering::Equal => 0,
    }
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
        let vertices = Self::construct_convex_hull(points.into_iter().collect())?;

        Ok(Self {
            bounding_radius: ConvexPolytope::<2>::bounding_radius(&vertices),
            vertices,
        })
    }

    /// Compute the convex hull of points in two dimensions.
    ///
    /// The resulting vector contains a subset of the given points, including
    /// only the non-degenerate points on the convex hull. The output vertices
    /// are arranged in a counter-clockwise order.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if the input vertices do not form a convex body with 3 or more points.
    ///
    /// [`Error`]: enum@Error
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::shape::ConvexSurfaceMesh2d;
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let points = vec![
    ///     [1.0, 0.0].into(),
    ///     [0.5, f64::sqrt(3.0) / 2.0].into(),
    ///     [-0.5, f64::sqrt(3.0) / 2.0].into(),
    /// ];
    ///
    /// let hull_vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn construct_convex_hull(
        mut points: Vec<Cartesian<2>>,
    ) -> Result<Vec<Cartesian<2>>, Error> {
        // No need to try and triangulate if the hull is degenerate
        if points.len() < 3 {
            return Err(Error::DegeneratePolytope);
        }

        let anchor_idx = find_lowest_leftmost(&points).ok_or(Error::DegeneratePolytope)?;

        // Move the anchor to the front of the list of vertices, as it is always in the hull
        points.swap(0, anchor_idx);
        let anchor = points[0];

        // Sort the remainder of the slice in-place
        points[1..].sort_unstable_by(|&a, &b| {
            let (a0, a1) = get_graham_key(a, anchor);
            let (b0, b1) = get_graham_key(b, anchor);
            a0.total_cmp(&b0).then(a1.total_cmp(&b1))
        });

        // Now vertices[..2] is an edge on the hull. Initialize counters for the hull length
        // and number of vertices on the hull
        let mut n_vertices_on_hull = 2;
        let mut next_candidate = 2;

        // Repeat until all interior points are gone
        while next_candidate < points.len() {
            let c = points[next_candidate];
            while n_vertices_on_hull >= 2 {
                let p = points[n_vertices_on_hull - 2];
                let n = points[n_vertices_on_hull - 1];

                if predicate_orient2d((p, n), c) <= 0 {
                    // Point n is inside the hull, remove it by shrinking the hull
                    n_vertices_on_hull -= 1;
                } else {
                    break;
                }
            }
            // Swap the vertex c onto the end of the hull, extending it by one
            points.swap(next_candidate, n_vertices_on_hull);
            n_vertices_on_hull += 1;
            next_candidate += 1;
        }

        points.truncate(n_vertices_on_hull);
        if points.len() >= 3 {
            Ok(points)
        } else {
            Err(Error::DegeneratePolytope)
        }
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
    use assert2::check;
    use hoomd_vector::Angle;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rstest::*;

    #[rstest]
    #[case::single_point(vec![[1.0, 2.0]], 0)]
    #[case::two_points_first_lower(vec![[1.0, 1.0], [2.0, 3.0]], 0)]
    #[case::two_points_second_lower(vec![[1.0, 3.0], [2.0, 1.0]], 1)]
    #[case::same_y_leftmost_wins(vec![[3.0, 1.0], [1.0, 1.0], [2.0, 1.0]], 1)]
    #[case::same_y_negative(vec![[0.0, -5.0], [-3.0, -5.0], [2.0, -5.0]], 1)]
    #[case::multiple_points(vec![[3.0, 5.0], [1.0, 1.0], [4.0, 2.0], [2.0, 3.0]], 1)]
    #[case::negative_coords(vec![[0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]], 1)]
    #[case::same_y_all_negative_x(vec![[5.0, 0.0], [-10.0, 0.0], [-5.0, 0.0]], 1)]
    #[case::lowest_is_only_point(vec![[100.0, -100.0]], 0)]
    #[case::diagonal_tiebreak(vec![[5.0, 5.0], [4.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]], 4)]
    fn test_find_lowest_leftmost(#[case] vertices: Vec<[f64; 2]>, #[case] expected_idx: usize) {
        let vertices: Vec<Cartesian<2>> = vertices.into_iter().map(Cartesian::from).collect();
        let idx = find_lowest_leftmost(&vertices).expect("returned None for non-empty input");
        assert_eq!(idx, expected_idx);
    }

    #[rstest]
    fn test_find_lowest_leftmost_empty() {
        let vertices: Vec<Cartesian<2>> = vec![];
        assert_eq!(find_lowest_leftmost(&vertices), None);
    }

    #[rstest]
    fn test_single_point_various_coords(
        #[values([0.0, 0.0], [-1.0, -1.0], [100.0, -50.0], [f64::MIN_POSITIVE, f64::MIN_POSITIVE])]
        coords: [f64; 2],
    ) {
        let vertices = vec![Cartesian::from(coords)];
        let idx = find_lowest_leftmost(&vertices).expect("returned None for non-empty input");
        assert_eq!(idx, 0);
    }

    #[rstest]
    fn test_square_corners() {
        let points: Vec<Cartesian<2>> = vec![[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        assert_eq!(vertices.len(), 4);

        let hull = [
            [0.0, 0.0].into(),
            [1.0, 0.0].into(),
            [1.0, 1.0].into(),
            [0.0, 1.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_square_corners_big() {
        let points: Vec<Cartesian<2>> = vec![
            [101.0, 101.0],
            [101.0, 100.0],
            [100.0, 101.0],
            [100.0, 100.0],
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        assert_eq!(vertices.len(), 4);

        let hull = [
            [100.0, 100.0].into(),
            [101.0, 100.0].into(),
            [101.0, 101.0].into(),
            [100.0, 101.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_square_with_edge_points() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0], // bottom edge
            [1.0, 0.5],
            [1.0, 1.0], // right edge
            [0.5, 1.0],
            [0.0, 1.0], // top edge
            [0.0, 0.5], // left edge
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        // Hull should have 4 corners (interior edge points excluded)
        assert_eq!(vertices.len(), 4);
        let hull = [
            [0.0, 0.0].into(),
            [1.0, 0.0].into(),
            [1.0, 1.0].into(),
            [0.0, 1.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_square_dense_boundary() {
        let mut pts: Vec<[f64; 2]> = Vec::new();
        // Bottom edge
        for i in 0..20 {
            pts.push([f64::from(i) / 19.0, 0.0]);
        }
        // Right edge
        for i in 0..20 {
            pts.push([1.0, f64::from(i) / 19.0]);
        }
        // Top edge
        for i in 0..20 {
            pts.push([f64::from(i) / 19.0, 1.0]);
        }
        // Left edge
        for i in 0..20 {
            pts.push([0.0, f64::from(i) / 19.0]);
        }
        let points: Vec<Cartesian<2>> = pts.into_iter().map(Cartesian::from).collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        assert_eq!(vertices.len(), 4);

        let hull = [
            [0.0, 0.0].into(),
            [1.0, 0.0].into(),
            [1.0, 1.0].into(),
            [0.0, 1.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_circle_uniform() {
        let n = 20;
        let points: Vec<Cartesian<2>> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                Cartesian::from([angle.cos(), angle.sin()])
            })
            .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        // All points on circle should be in hull
        assert_eq!(vertices.len(), n);
    }

    #[rstest]
    fn test_circle_with_interior_points() {
        let n_boundary = 12;
        let mut points: Vec<Cartesian<2>> = (0..n_boundary)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n_boundary as f64;
                Cartesian::from([angle.cos(), angle.sin()])
            })
            .collect();
        // Add interior points
        points.extend([[0.0, 0.0], [0.3, 0.3], [-0.2, 0.1], [0.1, -0.4]].map(Cartesian::from));
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        // Only boundary points should be in hull
        assert_eq!(vertices.len(), n_boundary);
    }

    #[rstest]
    fn test_circle_partial_arc() {
        let points: Vec<Cartesian<2>> = (0..10)
            .map(|i| {
                let angle = -std::f64::consts::FRAC_PI_4
                    + (std::f64::consts::FRAC_PI_2 * f64::from(i) / 9.0);
                Cartesian::from([angle.cos(), angle.sin()])
            })
            .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        // All points on partial arc should be in hull
        assert_eq!(vertices.len(), 10);
    }

    #[rstest]
    fn test_random_unit_square(#[values(0, 42, 64, 100_000)] seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let original: Vec<Cartesian<2>> = (0..50)
            .map(|_| Cartesian::from([rng.random::<f64>(), rng.random::<f64>()]))
            .collect();
        let points = original.clone();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        // Hull should have at least 3 points
        assert!(vertices.len() >= 3);
        // All hull points should be in original set
        for h in &vertices {
            assert!(original.iter().any(|v| (*v - *h).norm() < 1e-10));
        }
    }

    #[rstest]
    fn test_random_gaussian() {
        let mut rng = StdRng::seed_from_u64(42);
        let points: Vec<Cartesian<2>> = (0..100)
            .map(|_| {
                Cartesian::from([
                    rng.random::<f64>() * 2.0 - 1.0,
                    rng.random::<f64>() * 2.0 - 1.0,
                ])
            })
            .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        assert!(vertices.len() >= 3);
    }

    #[rstest]
    fn test_random_deterministic_output() {
        for _ in 0..3 {
            let mut rng = StdRng::seed_from_u64(123);
            let points1: Vec<Cartesian<2>> = (0..30)
                .map(|_| Cartesian::from([rng.random::<f64>(), rng.random::<f64>()]))
                .collect();
            let vertices1 = ConvexSurfaceMesh2d::construct_convex_hull(points1)
                .expect("hard-coded points should lie on a convex hull");

            let mut rng = StdRng::seed_from_u64(123);
            let points2: Vec<Cartesian<2>> = (0..30)
                .map(|_| Cartesian::from([rng.random::<f64>(), rng.random::<f64>()]))
                .collect();
            let vertices2 = ConvexSurfaceMesh2d::construct_convex_hull(points2)
                .expect("hard-coded points should lie on a convex hull");

            assert_eq!(vertices1.len(), vertices2.len());
        }
    }

    #[rstest]
    fn test_duplicate_lowest_points() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0], // Three duplicates of lowest
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        assert!(vertices.len() >= 3);

        let hull = [
            [0.0, 0.0].into(),
            [1.0, 0.0].into(),
            [1.0, 1.0].into(),
            [0.0, 1.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_many_duplicates_few_unique() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        // Should handle duplicates gracefully
        assert!(vertices.len() >= 3);

        let hull = [[0.0, 0.0].into(), [2.0, 0.0].into(), [1.0, 2.0].into()];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_collinear_bottom_edge() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],  // All at y=0
            [0.75, 1.0], // Apex
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        // Should pick leftmost of bottom points and include apex
        assert!(vertices.len() >= 3);

        let hull = [[0.0, 0.0].into(), [1.5, 0.0].into(), [0.75, 1.0].into()];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_leftmost_selected() {
        let points: Vec<Cartesian<2>> = vec![
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.0], // All y=0, but [0,0] is leftmost
            [0.5, 1.0],
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        // [0, 0] should be the anchor point
        let hull = [[0.0, 0.0].into(), [1.0, 0.0].into(), [0.5, 1.0].into()];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_many_bottom_points() {
        let points: Vec<Cartesian<2>> = (0..10)
            .map(|i| [f64::from(i) * 2.0 / 9.0, 0.0])
            .chain([[1.0, 1.0]])
            .map(Cartesian::from)
            .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        // Leftmost [0,0] and rightmost [2,0] should be in hull with apex
        assert!(vertices.len() >= 3);
    }

    #[rstest]
    fn test_collinear_from_anchor() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0], // Anchor (lowest leftmost)
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0], // Collinear at 45 degrees
            [1.0, 0.0],
            [0.0, 1.0], // Other corners
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        // Should only keep furthest point in each direction
        assert!(vertices.len() >= 3);

        let hull = [
            [0.0, 0.0].into(),
            [1.0, 0.0].into(),
            [3.0, 3.0].into(),
            [0.0, 1.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_radial_collinear_multiple_directions() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0], // Anchor
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0], // Along x-axis
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0], // Along y-axis
            [1.0, 1.0],
            [2.0, 2.0], // Diagonal
            [-1.0, 0.0],
            [-2.0, 0.0], // Negative x-axis (but won't be picked due to anchor)
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        // Should keep only outermost points
        assert!(vertices.len() >= 3);
    }

    #[rstest]
    fn test_star_pattern() {
        let mut points: Vec<Cartesian<2>> = vec![Cartesian::from([0.0, 0.0])]; // Anchor
        // Create points along 8 rays
        for i in 0..8 {
            let angle = 2.0 * std::f64::consts::PI * f64::from(i) / 8.0;
            for r in [0.5, 1.0, 1.5] {
                points.push(Cartesian::from([r * angle.cos(), r * angle.sin()]));
            }
        }
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        // Outer points should form the hull
        assert!(vertices.len() >= 8); // At least 8 outer points
    }

    #[rstest]
    fn test_minimum_triangle() {
        let points: Vec<Cartesian<2>> = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        assert_eq!(vertices.len(), 3);

        let hull = [[0.0, 0.0].into(), [1.0, 0.0].into(), [0.5, 1.0].into()];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_negative_coordinates() {
        let points: Vec<Cartesian<2>> = vec![[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        assert_eq!(vertices.len(), 4);

        let hull = [
            [-1.0, -1.0].into(),
            [1.0, -1.0].into(),
            [1.0, 1.0].into(),
            [-1.0, 1.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_large_coordinates() {
        let points: Vec<Cartesian<2>> = vec![[0.0, 0.0], [1e6, 0.0], [1e6, 1e6], [0.0, 1e6]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let vertices = ConvexSurfaceMesh2d::construct_convex_hull(points)
            .expect("hard-coded points should lie on a convex hull");
        assert_eq!(vertices.len(), 4);

        let hull = [
            [0.0, 0.0].into(),
            [1e6, 0.0].into(),
            [1e6, 1e6].into(),
            [0.0, 1e6].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_degenerate() {
        let points: Vec<Cartesian<2>> = vec![[0.0, 0.0], [0.5, 0.5], [0.25, 0.25], [1.0, 1.0]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let result = ConvexSurfaceMesh2d::construct_convex_hull(points);
        check!(result == Err(Error::DegeneratePolytope));
    }

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

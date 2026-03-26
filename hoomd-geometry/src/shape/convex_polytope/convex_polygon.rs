// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement separating planes overlap check for `ConvexPolygon`.

use super::ConvexPolygon;
use crate::{BoundingSphereRadius, IntersectsAt, IntersectsAtGlobal};
use hoomd_vector::{Cartesian, InnerProduct, Metric, Rotate, Rotation, RotationMatrix};

impl<R> IntersectsAtGlobal<Self, Cartesian<2>, R> for ConvexPolygon
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

impl<R> IntersectsAt<Self, Cartesian<2>, R> for ConvexPolygon
where
    RotationMatrix<2>: From<R>,
    R: Copy,
{
    /// Test convex polygon intersections with separating planes.
    ///
    /// When the number of vertices is small, separating planes is significantly
    /// faster than the Xenocollide algorithm implemented for the `Convex<ConvexPolygon>`
    /// type.
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::{IntersectsAt, shape::ConvexPolygon};
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
    a: &ConvexPolygon,
    b: &ConvexPolygon,
    v_ab: &Cartesian<2>,
    o_b: &RotationMatrix<2>,
) -> bool {
    let mut previous = b.vertices.len() - 1;
    for current in 0..b.vertices.len() {
        let p = b.vertices[current];
        let edge = p - b.vertices[previous];

        // SAFETY: Vertices must be counter-clockwise ordered (TODO).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntersectsAt;
    use hoomd_vector::{Angle, Cartesian, Rotate, Rotation};

    use rstest::*;

    use std::f64::consts::PI;

    // ConvexPolygon tests from hoomd-blue's test_convex_polygon.cc

    #[fixture]
    fn square() -> ConvexPolygon {
        ConvexPolygon::with_vertices([
            [-0.5, -0.5].into(),
            [0.5, -0.5].into(),
            [0.5, 0.5].into(),
            [-0.5, 0.5].into(),
        ])
        .expect("hard-coded vertices form a valid polygon")
    }

    #[fixture]
    fn triangle() -> ConvexPolygon {
        ConvexPolygon::with_vertices([[-0.5, -0.5].into(), [0.5, -0.5].into(), [0.5, 0.5].into()])
            .expect("hard-coded vertices form a valid polygon")
    }

    #[rstest]
    fn square_no_rot(square: ConvexPolygon) {
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
    fn square_rot(square: ConvexPolygon) {
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
    fn square_triangle(square: ConvexPolygon, triangle: ConvexPolygon) {
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

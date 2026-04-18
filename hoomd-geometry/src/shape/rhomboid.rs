// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, Metric, Rotate, Rotation, RotationMatrix};
use serde::{Deserialize, Serialize};

use crate::{
    BoundingSphereRadius, IntersectsAt, IntersectsAtGlobal, IsPointInside, Scale, SupportMapping,
    Volume,
};

/// An axis-aligned parallelogram defined by a 2 x 2 upper triangular matrix.
///
/// This shape is a general case of rhombus where pairs of sides are not equal.
/// We enforce the convention that the center of the shape is at the origin.
#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub struct Rhomboid {
    /// The extents [``L_x``, ``L_y``] of each edge along the Cartesian axes ``x`` and ``y``.
    extents: [PositiveReal; 2],
    /// The shear applied to the shape in the x direction relative to ``L_y``
    xy: f64,
}

impl From<(PositiveReal, PositiveReal, f64)> for Rhomboid {
    #[inline]
    fn from(value: (PositiveReal, PositiveReal, f64)) -> Self {
        Rhomboid {
            extents: [value.0, value.1],
            xy: value.2,
        }
    }
}

impl Rhomboid {
    #[inline(always)]
    pub fn lx(&self) -> PositiveReal {
        self.extents[0]
    }
    #[inline(always)]
    pub fn ly(&self) -> PositiveReal {
        self.extents[1]
    }
    #[inline(always)]
    pub fn xy(&self) -> f64 {
        self.xy
    }

    /// A @ [x, y] = [lx·x + ly·xy·y, ly·y]
    #[inline]
    fn matmul(&self, v: [f64; 2]) -> [f64; 2] {
        [
            self.lx().get() * v[0] + self.ly().get() * self.xy() * v[1],
            self.ly().get() * v[1],
        ]
    }

    /// A^T @ [x, y] = [lx·x, ly·xy·x + ly·y]
    #[inline]
    fn matmul_t(&self, v: [f64; 2]) -> [f64; 2] {
        [
            self.lx().get() * v[0],
            self.ly().get() * self.xy() * v[0] + self.ly().get() * v[1],
        ]
    }

    /// Compute the vertices of the Rhomboid assuming it is centered at the origin.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> [Cartesian<2>; 4] {
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(|c| self.matmul(c).into())
    }
    /// Represent a 2D triclinic box in the GSD box format.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomb = Rhomboid::from((1.0.try_into()?, 2.0.try_into()?, 1.5));
    ///
    /// let gsd_box = rhomb.to_gsd_box();
    /// assert_eq!(gsd_box, [1.0, 2.0, 0.0, 1.5, 0.0, 0.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use] // TODO: check
    pub fn to_gsd_box(&self) -> [f64; 6] {
        [
            self.extents[0].get(),
            self.extents[1].get(),
            0.0,
            self.xy,
            0.0,
            0.0,
        ]
    }
}

impl Volume for Rhomboid {
    #[inline]
    fn volume(&self) -> f64 {
        // When A is triangular, det(A) = det(diag(A))
        self.lx().get() * self.ly().get()
    }
}

impl Scale for Rhomboid {
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        Rhomboid {
            extents: [self.extents[0] * v, self.extents[1] * v],
            xy: self.xy,
        }
    }

    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let v = v
            .get()
            .sqrt()
            .try_into()
            .expect("sqrt of positive real is positive");
        self.scale_length(v)
    }
}

impl IsPointInside<Cartesian<2>> for Rhomboid {
    #[inline]
    fn is_point_inside(&self, point: &Cartesian<2>) -> bool {
        let [x, y] = point.coordinates;
        if y.abs() >= self.ly().get() / 2.0 {
            return false;
        }
        if (x - self.xy() * y).abs() >= self.lx().get() / 2.0 {
            return false;
        }
        true
    }
}

impl SupportMapping<Cartesian<2>> for Rhomboid {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<2>) -> Cartesian<2> {
        let d = self.matmul_t([n[0], n[1]]);
        let s = [d[0].signum() * 0.5, d[1].signum() * 0.5];
        self.matmul(s).into()
    }
}

impl BoundingSphereRadius for Rhomboid {
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        // || maximal_extent || / 2.0 = { lx + ly * |xy|, ly } || / 2.0
        (0.5 * f64::sqrt(
            (self.lx().get() + self.ly().get() * self.xy().abs()).powi(2) + self.ly().get().powi(2),
        ))
        .try_into()
        .expect("Norm is always positive.")
    }
}

impl<R> IntersectsAt<Rhomboid, Cartesian<2>, R> for Rhomboid
where
    R: Rotate<Cartesian<2>> + Rotation + Copy,
    RotationMatrix<2>: From<R>,
{
    /// Test rhomboid intersections using the separating axis theorem.
    ///
    /// Rhomboids have two unique edge directions each, giving four potential
    /// separating axes.
    ///
    /// The four axes are the normals to each rhomboid's two edges:
    /// - Axis 1: P1's horizontal edge normal `[0, 1]`
    /// - Axis 3: P2's horizontal edge normal (rotated)
    /// - Axis 2: P1's skewed edge normal `[-Ly1, Ly1·xy1]`
    /// - Axis 4: P2's skewed edge normal (rotated)
    #[inline]
    fn intersects_at(&self, other: &Rhomboid, v_ij: &Cartesian<2>, o_ij: &R) -> bool {
        let o_j = RotationMatrix::from(*o_ij);
        let [c, s] = [o_j.rows()[0][0], o_j.rows()[1][0]];

        let (lx1, ly1, xy1) = (self.lx().get(), self.ly().get(), self.xy());
        let (lx2, ly2, xy2) = (other.lx().get(), other.ly().get(), other.xy());
        let [tx, ty] = [v_ij[0], v_ij[1]];

        // Shared subexpressions for overlap checks.
        let s_lx2_abs = (s * lx2).abs();
        let term_a_abs = (xy2 * s + c).abs(); // used in axes 1 & 4

        // Axis 1: P1's horizontal edge normal [0, 1].
        // All comparisons are scaled by 2 to avoid halving the projection radii.
        if 2.0 * ty.abs() > ly1 + s_lx2_abs + ly2 * term_a_abs {
            return false;
        }
        let term_b_abs = (c - xy1 * s).abs(); // used in axes 2 & 3

        // Axis 3: P2's horizontal edge normal.
        let d3_term = ty * c - tx * s;
        let s_lx1_abs = (s * lx1).abs();
        if 2.0 * d3_term.abs() > s_lx1_abs + ly1 * term_b_abs + ly2 {
            return false;
        }
        let term_c_abs = (c * (xy1 - xy2) + s * (xy1 * xy2 + 1.0)).abs(); // axes 2 & 4

        // Axis 2: P1's skewed edge normal.
        if 2.0 * (xy1 * ty - tx).abs() > lx1 + lx2 * term_b_abs + ly2 * term_c_abs {
            return false;
        }

        // Axis 4: P2's skewed edge normal (rotated).
        let cross_term = tx * c + ty * s;
        if 2.0 * (xy2 * d3_term - cross_term).abs() > lx1 * term_a_abs + ly1 * term_c_abs + lx2 {
            return false;
        }

        true
    }
}

impl<R> IntersectsAtGlobal<Rhomboid, Cartesian<2>, R> for Rhomboid
where
    R: Rotate<Cartesian<2>> + Rotation + Copy,
    RotationMatrix<2>: From<R>,
{
    #[inline]
    fn intersects_at_global(
        &self,
        other: &Rhomboid,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{ConvexPolygon, ConvexSurfaceMesh2d, Hypercuboid};
    use crate::{Convex, IntersectsAt, IsPointInside};
    use approxim::assert_relative_eq;
    use hoomd_vector::Angle;
    use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};
    use rstest_reuse::{self, apply, template};
    use std::f64::consts::PI;

    /// Common rhomboid shapes used across multiple tests.
    #[template]
    #[rstest]
    #[case::unit_square(1.0, 1.0, 0.0)]
    #[case::square(2.0, 2.0, 0.0)]
    #[case::rectangle(3.0, 1.0, 0.0)]
    #[case::skinny(0.005, 5.0, 0.0)]
    #[case::mild_shear(2.0, 2.0, 0.5)]
    #[case::sheared(3.0, 2.0, 1.5)]
    #[case::sheared_2(1.0, 3.0, 1.0)]
    #[case::strong_shear(1.0, 5.0, 25.0)]
    #[case::negative_shear(2.0, 2.0, -5.5)]
    #[case::unit_shear(1.0, 1.0, 1.0)]
    fn rhomboid_shapes(#[case] lx: f64, #[case] ly: f64, #[case] xy: f64) {}

    fn random_rhomboid(rng: &mut StdRng) -> Rhomboid {
        let lx: f64 = rng.random_range(0.1..10.0);
        let ly: f64 = rng.random_range(0.1..10.0);
        let xy: f64 = rng.random_range(-2.0..2.0);
        (lx.try_into().unwrap(), ly.try_into().unwrap(), xy).into()
    }

    /// Check that the support value (dot product with direction) matches the polygon.
    /// This avoids tie-breaking differences when multiple vertices maximize the dot product.
    fn check_support_value(lx: f64, ly: f64, xy: f64, n: [f64; 2]) {
        let rhomboid: Rhomboid = (
            lx.try_into().expect("lx > 0"),
            ly.try_into().expect("ly > 0"),
            xy,
        )
            .into();

        let polygon = ConvexPolygon::with_vertices(rhomboid.vertices().to_vec())
            .expect("rhomboid vertices form a polygon");

        let r = rhomboid.support_mapping(&n.into());
        let p = polygon.support_mapping(&n.into());

        let support_rhomboid = r[0] * n[0] + r[1] * n[1];
        let support_polygon = p[0] * n[0] + p[1] * n[1];

        assert_relative_eq!(support_rhomboid, support_polygon, epsilon = 1e-12);
    }

    #[apply(rhomboid_shapes)]
    fn support_mapping_fixed_directions(#[case] lx: f64, #[case] ly: f64, #[case] xy: f64) {
        // Cardinal directions.
        check_support_value(lx, ly, xy, [1.0, 0.0]);
        check_support_value(lx, ly, xy, [0.0, 1.0]);
        check_support_value(lx, ly, xy, [-1.0, 0.0]);
        check_support_value(lx, ly, xy, [0.0, -1.0]);
        // Diagonal directions.
        check_support_value(lx, ly, xy, [1.0, 1.0]);
        check_support_value(lx, ly, xy, [-1.0, 1.0]);
        check_support_value(lx, ly, xy, [-1.0, -1.0]);
        check_support_value(lx, ly, xy, [1.0, -1.0]);
    }

    #[apply(rhomboid_shapes)]
    fn support_mapping_random_directions(#[case] lx: f64, #[case] ly: f64, #[case] xy: f64) {
        let mut rng = StdRng::seed_from_u64(42);
        let rhomboid: Rhomboid = (lx.try_into().unwrap(), ly.try_into().unwrap(), xy).into();
        let polygon =
            ConvexPolygon::with_vertices(rhomboid.vertices().to_vec()).expect("valid polygon");

        for _ in 0..1000 {
            let n: Cartesian<2> = rng.random();
            assert_relative_eq!(
                rhomboid.support_mapping(&n),
                polygon.support_mapping(&n),
                epsilon = 1e-12,
            );
        }
    }

    #[apply(rhomboid_shapes)]
    fn bounding_sphere_radius_matches_polygon(#[case] lx: f64, #[case] ly: f64, #[case] xy: f64) {
        let rhomboid: Rhomboid = (lx.try_into().unwrap(), ly.try_into().unwrap(), xy).into();
        let polygon = ConvexPolygon::with_vertices(rhomboid.vertices().to_vec())
            .expect("rhomboid vertices form a polygon");

        assert_relative_eq!(
            rhomboid.bounding_sphere_radius().get(),
            polygon.bounding_sphere_radius().get(),
            epsilon = 1e-12,
        );
    }

    /// Compare the Rhomboid SAT against the ConvexSurfaceMesh2d separating-planes
    /// implementation (an independently tested ground truth).
    fn check_sat_against_mesh(
        lx1: f64,
        ly1: f64,
        xy1: f64,
        lx2: f64,
        ly2: f64,
        xy2: f64,
        tx: f64,
        ty: f64,
        theta: f64,
    ) {
        let a: Rhomboid = (lx1.try_into().unwrap(), ly1.try_into().unwrap(), xy1).into();
        let b: Rhomboid = (lx2.try_into().unwrap(), ly2.try_into().unwrap(), xy2).into();

        let v_ij = Cartesian::from([tx, ty]);
        let o_ij = Angle::from(theta);

        let sat = a.intersects_at(&b, &v_ij, &o_ij);

        let mesh_a = ConvexSurfaceMesh2d::from_point_set(a.vertices().iter().copied()).unwrap();
        let mesh_b = ConvexSurfaceMesh2d::from_point_set(b.vertices().iter().copied()).unwrap();
        let mesh = mesh_a.intersects_at(&mesh_b, &v_ij, &o_ij);

        assert_eq!(
            sat, mesh,
            "SAT={sat}, mesh={mesh}\n\
             a=({lx1}, {ly1}, {xy1})\n\
             b=({lx2}, {ly2}, {xy2})\n\
             t=({tx}, {ty})\n\
             theta={theta}"
        );
    }

    /// Shape pairs for intersection tests with displacement and rotation.
    #[template]
    #[rstest]
    #[case::coincident(0.0, 0.0, 0.0)]
    #[case::half_overlap(0.5, 0.5, 0.0)]
    #[case::near_touch_x(1.999_999, 0.0, 0.0)]
    #[case::past_touch_x(2.000_001, 0.0, 0.0)]
    #[case::near_touch_y(0.0, 1.999_999, 0.0)]
    #[case::past_touch_y(0.0, 2.000_001, 0.0)]
    #[case::diagonal_45(1.0, 1.0, PI / 4.0)]
    #[case::rotated_60(1.3, 0.7, PI / 3.0)]
    #[case::rotated_90(0.0, 1.5, PI / 2.0)]
    fn square_displacements(#[case] tx: f64, #[case] ty: f64, #[case] theta: f64) {}

    #[apply(square_displacements)]
    fn intersects_at_identical_squares(#[case] tx: f64, #[case] ty: f64, #[case] theta: f64) {
        check_sat_against_mesh(2.0, 2.0, 0.0, 2.0, 2.0, 0.0, tx, ty, theta);
    }

    /// Displacements for mirror-sheared pairs (xy2 = -xy1).
    #[template]
    #[rstest]
    #[case::coincident(0.0, 0.0, 0.0)]
    #[case::shifted_x(1.0, 0.0, 0.0)]
    #[case::shifted_x2(2.0, 0.0, 0.0)]
    #[case::shifted_y(0.0, 1.0, 0.0)]
    #[case::diagonal(1.0, 1.0, 0.0)]
    #[case::rotated_45(0.5, 0.5, PI / 4.0)]
    #[case::rotated_60(1.0, 0.0, PI / 3.0)]
    fn mirror_shear_displacements(#[case] tx: f64, #[case] ty: f64, #[case] theta: f64) {}

    #[apply(mirror_shear_displacements)]
    fn intersects_at_mirror_sheared(#[case] tx: f64, #[case] ty: f64, #[case] theta: f64) {
        check_sat_against_mesh(2.0, 2.0, 1.0, 2.0, 2.0, -1.0, tx, ty, theta);
    }

    #[test]
    fn intersects_at_mixed_shapes() {
        // Different shapes, various displacements and rotations.
        check_sat_against_mesh(1.0, 3.0, 1.5, 2.0, 1.0, -0.5, 1.0, 0.5, 0.0);
        check_sat_against_mesh(1.0, 5.0, 2.0, 1.0, 5.0, 2.0, 1.0, 0.0, 0.0);
        check_sat_against_mesh(1.0, 3.0, 1.5, 2.0, 1.0, -0.5, 0.5, 0.5, PI / 6.0);
        check_sat_against_mesh(1.0, 5.0, 2.0, 1.0, 5.0, -2.0, 0.5, 0.0, PI / 2.0);
        check_sat_against_mesh(3.0, 1.0, -1.0, 1.0, 2.0, 0.5, 1.5, 0.3, PI / 5.0);
        check_sat_against_mesh(2.0, 1.0, 0.8, 1.5, 2.5, -0.3, 0.0, 1.0, PI / 8.0);
    }

    #[test]
    fn scale_preserves_aspect_ratio_and_volume() {
        let rhomboid: Rhomboid = (3.0.try_into().unwrap(), 2.0.try_into().unwrap(), 1.5).into();
        let original_volume = rhomboid.volume();
        let original_lx_over_ly = rhomboid.lx().get() / rhomboid.ly().get();

        let scaled = rhomboid.scale_length(2.0.try_into().unwrap());
        assert_relative_eq!(scaled.volume(), 4.0 * original_volume);
        assert_relative_eq!(scaled.lx().get() / scaled.ly().get(), original_lx_over_ly);
        assert_eq!(scaled.xy(), rhomboid.xy());

        let scaled = rhomboid.scale_volume(9.0.try_into().unwrap());
        assert_relative_eq!(scaled.volume(), 9.0 * original_volume);
        assert_relative_eq!(scaled.lx().get() / scaled.ly().get(), original_lx_over_ly);
        assert_eq!(scaled.xy(), rhomboid.xy());
    }

    /// Unsheared rhomboid shapes (xy=0) for rectangle comparison tests.
    #[template]
    #[rstest]
    #[case::unit_square(1.0, 1.0)]
    #[case::square(2.0, 2.0)]
    #[case::rectangle(3.0, 1.0)]
    #[case::skinny(0.005, 5.0)]
    fn unsheared_shapes(#[case] lx: f64, #[case] ly: f64) {}

    #[apply(unsheared_shapes)]
    fn is_point_inside_matches_rectangle(#[case] lx: f64, #[case] ly: f64) {
        let mut rng = StdRng::seed_from_u64(789);

        let rhomboid: Rhomboid = (lx.try_into().unwrap(), ly.try_into().unwrap(), 0.0).into();
        let rect = Hypercuboid {
            edge_lengths: [lx.try_into().unwrap(), ly.try_into().unwrap()],
        };

        for _ in 0..10_000 {
            let point: Cartesian<2> =
                rng.random::<Cartesian<2>>() * 20.0 - Cartesian::from([10.0; 2]);

            assert_eq!(
                rhomboid.is_point_inside(&point),
                rect.is_point_inside(&point),
                "Mismatch at ({}, {}) for lx={lx}, ly={ly}",
                point[0],
                point[1],
            );
        }
    }

    #[apply(rhomboid_shapes)]
    fn is_point_inside_area_fraction(#[case] lx: f64, #[case] ly: f64, #[case] xy: f64) {
        let mut rng = StdRng::seed_from_u64(1011);

        let rhomboid: Rhomboid = (lx.try_into().unwrap(), ly.try_into().unwrap(), xy).into();
        let area = rhomboid.volume();

        // Bounding box for sampling.
        let bx = lx + ly * xy.abs();
        let by = ly;
        let bbox_area = bx * by;

        let n_samples = 100_000_usize;
        let mut inside_count = 0_usize;

        for _ in 0..n_samples {
            let x = rng.random_range(-bx / 2.0..bx / 2.0);
            let y = rng.random_range(-by / 2.0..by / 2.0);
            if rhomboid.is_point_inside(&[x, y].into()) {
                inside_count += 1;
            }
        }

        let estimated_area = (inside_count as f64 / n_samples as f64) * bbox_area;
        assert_relative_eq!(estimated_area, area, max_relative = 0.02);
    }

    #[test]
    fn intersects_at_random() {
        let mut rng = StdRng::seed_from_u64(456);

        for i in 0..10_000 {
            let a = random_rhomboid(&mut rng);
            let b = random_rhomboid(&mut rng);

            let v_ij: Cartesian<2> =
                rng.random::<Cartesian<2>>() * 20.0 - Cartesian::from([10.0; 2]);
            let o_ij = Angle::from(rng.random_range(-std::f64::consts::PI..std::f64::consts::PI));

            let sat = a.intersects_at(&b, &v_ij, &o_ij);

            let mesh_a = ConvexSurfaceMesh2d::from_point_set(a.vertices().iter().copied()).unwrap();
            let mesh_b = ConvexSurfaceMesh2d::from_point_set(b.vertices().iter().copied()).unwrap();
            let mesh = mesh_a.intersects_at(&mesh_b, &v_ij, &o_ij);

            assert_eq!(
                sat,
                mesh,
                "Mismatch at iteration {i}\n\
                 a=({}, {}, {})\n\
                 b=({}, {}, {})\n\
                 t=({}, {})\n\
                 theta={}",
                a.lx().get(),
                a.ly().get(),
                a.xy(),
                b.lx().get(),
                b.ly().get(),
                b.xy(),
                v_ij[0],
                v_ij[1],
                o_ij.theta,
            );
        }
    }
}

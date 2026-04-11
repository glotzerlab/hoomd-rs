// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, Rotate, Rotation};

use crate::{BoundingSphereRadius, IntersectsAt, SupportMapping, Volume};

/// An axis-aligned parallelogram defined by a 2 x 2 upper triangular matrix.
///
/// This shape is a general case of rhombus where pairs of sides are not equal.
/// We enforce the convention that the center of the shape is at the origin.
#[derive(Debug, PartialEq, Copy, Clone)]
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

    /// Compute the vertices of the Rhomboid assuming it is centered at the origin.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> [Cartesian<2>; 4] {
        let half_lx = self.lx().get() * 0.5;
        let half_ly = self.ly().get() * 0.5;
        let half_ly_xy = half_ly * self.xy();

        [
            [-half_lx - half_ly_xy, -half_ly].into(),
            [half_lx - half_ly_xy, -half_ly].into(),
            [half_lx + half_ly_xy, half_ly].into(),
            [-half_lx + half_ly_xy, half_ly].into(),
        ]
    }
    /// Represent a 2D triclinic box in the GSD box format.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomb = Rhomboid::from((1.0.try_into(), 2.0.try_into(), 1.5))?;
    ///
    /// let gsd_box = triclinic.to_gsd_box();
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

impl SupportMapping<Cartesian<2>> for Rhomboid {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<2>) -> Cartesian<2> {
        // {self}^T @ n = [lx * nx, ly * xy * nx + ly * ny]
        let d0 = self.lx().get() * n[0];
        let d1 = self.ly().get() * self.xy() * n[0] + self.ly().get() * n[1];

        // {self} @ [sign(d0)/2, sign(d1)/2]
        let s0 = d0.signum() * 0.5;
        let s1 = d1.signum() * 0.5;

        [
            self.lx().get() * s0 + self.ly().get() * self.xy() * s1,
            self.ly().get() * s1,
        ]
        .into()
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
    R: Rotate<Cartesian<2>> + Rotation,
{
    #[inline]
    fn intersects_at(&self, other: &Rhomboid, v_ij: &Cartesian<2>, o_ij: &R) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::ConvexPolygon;
    use approxim::assert_relative_eq;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rstest::rstest;

    fn random_rhomboid(rng: &mut StdRng) -> Rhomboid {
        let lx: f64 = rng.random_range(0.1..10.0);
        let ly: f64 = rng.random_range(0.1..10.0);
        let xy: f64 = rng.random_range(-2.0..2.0);
        (lx.try_into().unwrap(), ly.try_into().unwrap(), xy).into()
    }

    /// Construct a ConvexPolygon from the Rhomboid's vertices and verify that
    /// both shapes return the same support mapping for a given direction.
    fn check_support_mapping(lx: f64, ly: f64, xy: f64, n: [f64; 2]) {
        let rhomboid: Rhomboid = (
            lx.try_into().expect("lx > 0"),
            ly.try_into().expect("ly > 0"),
            xy,
        )
            .into();

        let polygon = ConvexPolygon::with_vertices(rhomboid.vertices().to_vec())
            .expect("rhomboid vertices form a polygon");
        assert_relative_eq!(
            rhomboid.support_mapping(&n.into()),
            polygon.support_mapping(&n.into()),
            epsilon = 1e-12,
        );
    }

    #[rstest]
    #[case::right([1.0, 1e-6])]
    #[case::up([1e-6, 1.0])]
    #[case::left([-1.0, 1e-6])]
    #[case::down([1e-6, -1.0])]
    fn support_mapping_unsheared_square(#[case] n: [f64; 2]) {
        check_support_mapping(2.0, 2.0, 0.0, n);
    }

    #[rstest]
    #[case::ne([1.0, 1.0])]
    #[case::nw([-1.0, 1.0])]
    #[case::sw([-1.0, -1.0])]
    #[case::se([1.0, -1.0])]
    fn support_mapping_sheared(#[case] n: [f64; 2]) {
        check_support_mapping(3.0, 2.0, 1.5, n);
    }

    #[test]
    fn support_mapping_random_cases() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..10_000 {
            let rhomboid = random_rhomboid(&mut rng);
            let polygon = ConvexPolygon::with_vertices(rhomboid.vertices().to_vec())
                .expect("rhomboid vertices form a polygon");

            for _ in 0..10 {
                let n: Cartesian<2> = rng.random();
                assert_relative_eq!(
                    rhomboid.support_mapping(&n),
                    polygon.support_mapping(&n),
                    epsilon = 1e-12,
                );
            }
        }
    }

    #[test]
    fn bounding_sphere_radius_random() {
        let mut rng = StdRng::seed_from_u64(123);

        for _ in 0..10_000 {
            let rhomboid = random_rhomboid(&mut rng);
            let polygon = ConvexPolygon::with_vertices(rhomboid.vertices().to_vec())
                .expect("rhomboid vertices form a polygon");

            assert_relative_eq!(
                rhomboid.bounding_sphere_radius().get(),
                polygon.bounding_sphere_radius().get(),
                epsilon = 1e-12,
            );
        }
    }
}

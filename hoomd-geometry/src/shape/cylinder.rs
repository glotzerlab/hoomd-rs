// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Cylinder`]

use super::Circle;
use crate::Volume;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, Rotate};

/// A circle with normal `[0 0 1]` swept by `h/2` in the `+z` and `-z` directions.
///
/// # Example
///
/// [`Cylinder`] implements the [`Volume`] trait, which is equivalent to
/// $` \pi r^2 h `$.
/// ```
/// use hoomd_geometry::{Volume, shape::Cylinder};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cyl = Cylinder {
///     radius: 2.0.try_into()?,
///     height: 3.0.try_into()?,
/// };
/// assert_eq!(cyl.volume(), PI * (2.0 * 2.0) * 3.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Cylinder {
    /// Radius of the [`Cylinder`]
    pub radius: PositiveReal,
    /// Height of the [`Cylinder`]
    pub height: PositiveReal,
}

impl Volume for Cylinder {
    #[inline]
    fn volume(&self) -> f64 {
        Circle {
            radius: self.radius,
        }
        .volume()
            * self.height.get()
    }
}

impl Cylinder {
    /// Determine whether two infinitely long cylinders intersect
    fn intersects_at_infinite<R>(&self, other: &Self, v_ij: &Cartesian<3>, o_ij: &R) -> bool
    where
        R: Rotate<Cartesian<3>>,
    {
        const EPSILON: f64 = 1e-9;
        let [cx, cy, _cz] = v_ij.coordinates;
        let [sx, sy, _sz] = o_ij.rotate(&Cartesian::from([0., 0., 1.])).coordinates;

        // We only need the x and y components of the direction vector s
        // to find the magnitude of the cross product (v1 x v2) and check for parallelism.
        // v1 = (0, 0, 1)
        // v2 = s = (sx, sy, sz)
        // v1 x v2 = (-sy, sx, 0)
        // |v1 x v2| = sqrt(sx^2 + sy^2)
        let n_magnitude = (sx.powi(2) + sy.powi(2)).sqrt();

        let distance = if n_magnitude < EPSILON {
            // --- PARALLEL CASE ---
            // The axes are parallel (s is parallel to the z-axis).
            // The shortest distance `d` is just the distance from the point c
            // to the z-axis, which is the distance in the xy-plane.
            (cx.powi(2) + cy.powi(2)).sqrt()
        } else {
            // --- NON-PARALLEL CASE ---
            // We use the full formula: d = |(p2 - p1) . (v1 x v2)| / |v1 x v2|
            // p2 - p1 = c = (cx, cy, cz)
            // v1 x v2 = (-sy, sx, 0)
            // (p2 - p1) . (v1 x v2) = cx*(-sy) + cy*sx + cz*0 = cy*sx - cx*sy
            let dot_product = cy * sx - cx * sy;

            // d = |dot_product| / n_magnitude
            dot_product.abs() / n_magnitude
        };

        // A collision occurs if the shortest distance between the axes is <= r1+r2
        distance <= self.radius.get() + other.radius.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_vector::{Cartesian, InnerProduct, Versor};
    use rstest::rstest;
    use std::f64::consts::PI;

    #[rstest]
    #[case::parallel_intersecting(1.0, 1.0, [1.5, 0.0, 0.0], Versor::default(), true)]
    #[case::parallel_touching(1.0, 1.000, [2.0, 0.0, 0.0], Versor::default(), true)]
    #[case::parallel_barely_not_touching(1.0, 0.999_999, [2.0, 0.0, 0.0], Versor::default(), false)]
    #[case::parallel_not_intersecting(1.0, 1.0, [2.000_001, 0.0, 0.0], Versor::default(), false)]
    #[case::parallel_different_radii_touching(1.0, 0.5, [1.5, 0.0, 0.0], Versor::default(), true)]
    #[case::parallel_different_radii_not_intersecting(1.0, 0.5, [1.500_001, 0.0, 0.0], Versor::default(), false)]
    #[case::perpendicular_intersecting_at_origin(1.0, 1.0, [0.0, 0.0, 0.0], Versor::from_axis_angle(Cartesian::from([1., 0., 0.]).to_unit_unchecked().0, PI / 2.0), true)]
    #[case::perpendicular_skew_intersecting(1.0, 1.0, [1.5, 0.0, 0.0], Versor::from_axis_angle(Cartesian::from([1., 0., 0.]).to_unit_unchecked().0, PI / 2.0), true)]
    #[case::perpendicular_skew_touching(1.0, 1.0, [0.0, 2.0, 0.0], Versor::from_axis_angle(Cartesian::from([0., 1., 0.]).to_unit_unchecked().0, PI / 2.0), true)]
    #[case::perpendicular_skew_barely_not_touching(1.0, 0.999_999, [0.0, 2.0, 0.0], Versor::from_axis_angle(Cartesian::from([0., 1., 0.]).to_unit_unchecked().0, PI / 2.0), false)]
    #[case::perpendicular_skew_not_intersecting(1.0, 1.0, [2.000_001, 0.0, 5.0], Versor::from_axis_angle(Cartesian::from([1., 0., 0.]).to_unit_unchecked().0, PI / 2.0), false)]
    #[case::skew_intersecting(1.0, 1.0, [1.0, 1.0, 0.0], Versor::from_axis_angle(Cartesian::from([0., 1., 0.]).to_unit_unchecked().0, PI / 4.0), true)]
    #[case::skew_touching(1.0, 1.0, [0.0, 2.0, 0.0], Versor::from_axis_angle(Cartesian::from([0., 1., 0.]).to_unit_unchecked().0, PI / 4.0), true)]
    #[case::skew_not_intersecting(1.0, 1.0, [0.0, 2.000_001, 0.0], Versor::from_axis_angle(Cartesian::from([0., 1., 0.]).to_unit_unchecked().0, PI / 5.0), false)]
    fn test_intersects_at_infinite(
        #[case] r1: f64,
        #[case] r2: f64,
        #[case] v_ij: impl Into<Cartesian<3>>,
        #[case] o_ij: Versor,
        #[case] expected: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let c1 = Cylinder {
            radius: r1.try_into()?,
            height: 1.0.try_into()?,
        };
        let c2 = Cylinder {
            radius: r2.try_into()?,
            height: 1.0.try_into()?,
        };
        let v_ij_cartesian = v_ij.into();

        assert_eq!(
            c1.intersects_at_infinite(&c2, &v_ij_cartesian, &o_ij),
            expected,
        );

        Ok(())
    }
}

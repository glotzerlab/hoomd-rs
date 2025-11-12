// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Cylinder`]

use super::Circle;
use crate::Volume;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, Versor};

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
/// Precision below which cylinders are considered overlapping
const _CYLINDER_OVERLAP_PRECISION_SQUARED: f64 = 1e-14;

impl Cylinder {
    /// Determine whether two cylinders would intersect if they both had infinite length.
    ///
    /// This implementation is based on [Collision detection of cylindrical rigid bodies for motion planning](https://doi.org/10.1109/ROBOT.2006.1641925), and provides a
    /// useful alternative to bounding sphere-based checks when the underlying bodies
    /// are highly elongated.
    ///
    /// # Example
    ///
    /// This example is based on the scenario of two highly anisotropic shapes,
    /// whose bounding spheres have a large amount of volume relative to a tighter
    /// fitting bounding shape. Note that the effective volume checked for the infinite
    /// bounding cylinder is actually the intersection of that cylinder and the ball
    /// queried in the neighbor list, so it is guaranteed to be finite.
    ///
    /// ```
    /// # use hoomd_geometry::shape::Cylinder;
    /// # use hoomd_vector::{Cartesian, Versor};
    /// // Two parallel cylinders that wrap a dipyramid of height:width ratio 5:1.
    /// let c1 = Cylinder {
    ///     radius: 1.0.try_into().unwrap(),
    ///     height: 10.0.try_into().unwrap(),
    /// };
    /// let c2 = Cylinder {
    ///     radius: 1.0.try_into().unwrap(),
    ///     height: 10.0.try_into().unwrap(),
    /// };
    /// let o_ij = Versor::default();
    ///
    /// // Case 1: Bounding spheres would intersect, but the bounding cylinders do not.
    /// let v_ij_no_intersect = Cartesian::from([2.1, 0.0, 0.0]);
    /// assert!(!c1.intersects_at_infinite(&c2, &v_ij_no_intersect, &o_ij));
    ///
    /// // Case 2: Infinite cylinders intersect, meaning we need to further check the
    /// // underlying shapes for collision.
    /// let v_ij_intersect = Cartesian::from([1.9, 0.0, 0.0]);
    /// assert!(c1.intersects_at_infinite(&c2, &v_ij_intersect, &o_ij));
    /// ```
    #[must_use]
    #[inline]
    pub fn intersects_at_infinite(&self, other: &Self, v_ij: &Cartesian<3>, o_ij: &Versor) -> bool {
        let [vx, vy, _vz] = v_ij.coordinates;

        // Calculate sx and sy directly from the Versor components
        // for rotating (0, 0, 1) by quaternion q = (w, x, y, z):
        let q = o_ij.get();
        let sx = 2.0 * (q.vector[0] * q.vector[2] + q.scalar * q.vector[1]);
        let sy = 2.0 * (q.vector[1] * q.vector[2] - q.scalar * q.vector[0]);

        let n_magnitude_sq = sx.powi(2) + sy.powi(2);

        let distance_sq = if n_magnitude_sq < _CYLINDER_OVERLAP_PRECISION_SQUARED {
            // The axes of the cylinders are parallel
            // The shortest distance `d` is just the distance from the point c
            // to the z-axis, which is the distance in the xy-plane.
            vx.powi(2) + vy.powi(2)
        } else {
            // Non-Parallel
            // We use the full formula: d = |(p2 - p1) . (v1 x v2)| / |v1 x v2|
            // p2 - p1 = c = (cx, cy, cz)
            // v1 x v2 = (-sy, sx, 0)
            // (p2 - p1) . (v1 x v2) = cx*(-sy) + cy*sx + cz*0 = cy*sx - cx*sy
            let dot_product = vy * sx - vx * sy;
            dot_product.powi(2) / n_magnitude_sq
        };

        // A collision occurs if the shortest distance between the axes is <= r1+r2
        distance_sq <= (self.radius.get() + other.radius.get()).powi(2)
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
    #[case::skew_intersecting(1.0, 1.0, [1.0, 1.0, 0.0], Versor::from_axis_angle(Cartesian::from([0., 1., 0.]).to_unit_unchecked().0, PI / 5.0), true)]
    #[case::skew_touching(1.0, 1.0, [0.0, 2.0, 0.0], Versor::from_axis_angle(Cartesian::from([0., 1., 0.]).to_unit_unchecked().0, PI / 5.0), true)]
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

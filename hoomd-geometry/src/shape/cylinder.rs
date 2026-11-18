// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Cylinder`]

use serde::{Deserialize, Serialize};

use super::Circle;
use crate::{Error, IsPointInside, MapPoint, Scale, Volume};
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

impl Scale for Cylinder {
    /// Construct a scaled [`Cylinder`].
    ///
    /// The resulting cylinder's radius and height are the original's
    /// scaled by *v*:
    /// ```math
    /// \begin{align*}
    /// r_\mathrm{new} &= v r \\
    /// h_\mathrm{new} &= v h \\
    /// \end{align*}
    /// ```
    ///
    /// The centroid remains at the origin.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Cylinder};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let cylinder = Cylinder {
    ///     radius: 2.0.try_into()?,
    ///     height: 3.0.try_into()?,
    /// };
    ///
    /// let scaled_cylinder = cylinder.scale_length(2.0.try_into()?);
    ///
    /// assert_eq!(scaled_cylinder.radius.get(), 4.0);
    /// assert_eq!(scaled_cylinder.height.get(), 6.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        Self {
            radius: self.radius * v,
            height: self.height * v,
        }
    }

    /// Construct a scaled [`Cylinder`].
    ///
    /// The resulting cylinder's radius and height are the original's scaled by
    /// $`v^\frac{1}{3}`$:
    /// ```math
    /// \begin{align*}
    /// r_\mathrm{new} &= v^\frac{1}{3} r \\
    /// h_\mathrm{new} &= v^\frac{1}{3} h \\
    /// \end{align*}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Cylinder};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let cylinder = Cylinder {
    ///     radius: 2.0.try_into()?,
    ///     height: 3.0.try_into()?,
    /// };
    ///
    /// let scaled_cylinder = cylinder.scale_volume(8.0.try_into()?);
    ///
    /// assert_eq!(scaled_cylinder.radius.get(), 4.0);
    /// assert_eq!(scaled_cylinder.height.get(), 6.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let v = v.get().cbrt();
        self.scale_length(v.try_into().expect("v^{1/3} should be a positive real"))
    }
}

impl IsPointInside<Cartesian<3>> for Cylinder {
    /// Check if a vector is inside a cylinder.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::Cylinder};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let cylinder = Cylinder {
    ///     radius: 1.0.try_into()?,
    ///     height: 4.0.try_into()?,
    /// };
    ///
    /// assert!(cylinder.is_point_inside(&Cartesian::from([0.5, 0.0, 1.75])));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn is_point_inside(&self, point: &Cartesian<3>) -> bool {
        let projection = Cartesian::from([point[0], point[1]]);
        let circle = Circle { radius: self.radius };

        circle.is_point_inside(&projection) && point[2].abs() < self.height.get() / 2.0
    }
}

impl MapPoint<Cartesian<3>> for Cylinder {
    /// Map a point from one cylinder to another.
    ///
    /// Given a point P *inside `self`*, map it to the other shape
    /// by non-uniform scaling.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PointOutsideShape`] when `point` is outside the shape
    /// `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{MapPoint, shape::Cylinder};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let cylinder_a = Cylinder {
    ///     radius: 10.0.try_into()?,
    ///     height: 40.0.try_into()?,
    /// };
    /// let cylinder_b = Cylinder {
    ///     radius: 20.0.try_into()?,
    ///     height: 10.0.try_into()?,
    /// };
    ///
    /// let mapped_point =
    ///     cylinder_a.map_point(Cartesian::from([-1.0, 0.5, 10.0]), &cylinder_b);
    ///
    /// assert_eq!(mapped_point, Ok(Cartesian::from([-2.0, 1.0, 2.5])));
    /// assert_eq!(
    ///     cylinder_a.map_point(Cartesian::from([-100.0, 1.0, 0.0]), &cylinder_b),
    ///     Err(hoomd_geometry::Error::PointOutsideShape)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn map_point(&self, point: Cartesian<3>, other: &Self) -> Result<Cartesian<3>, Error> {
        if !self.is_point_inside(&point) {
            return Err(Error::PointOutsideShape);
        }

        // When multiplying by the scale factor, the floating point multiply
        // might round up in a way that places the mapped point outside the
        // other shape. Progressively make the scale smaller until the
        // check passes.
        let mut radial_scale = other.radius / self.radius;
        let mut height_scale = other.height / self.height;
        loop {
            let point = [point[0] * radial_scale.get(),
                point[1] * radial_scale.get(),
                point[2] * height_scale.get()].into();

            if other.is_point_inside(&point) {
                return Ok(point);
            }

            radial_scale = radial_scale
                .get()
                .next_down()
                .try_into()
                .expect("scale should remain a positive real");
            height_scale = height_scale
                .get()
                .next_down()
                .try_into()
                .expect("scale should remain a positive real");
        }
    }
}

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

        let dot_product = vy * sx - vx * sy;
        let mut distance_sq = dot_product.powi(2) / n_magnitude_sq;
        if !distance_sq.is_finite() {
            distance_sq = vx.powi(2) + vy.powi(2);
        }

        // A collision occurs if the shortest distance between the axes is <= r1+r2
        distance_sq <= (self.radius + other.radius).get().powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
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

    #[test]
    fn test_scale_length() -> Result<(), Box<dyn std::error::Error>> {
        let cylinder = Cylinder {
            radius: 4.0.try_into()?,
            height: 6.0.try_into()?,
        };

        let scaled_cylinder = cylinder.scale_length(2.0.try_into()?);
        assert_relative_eq!(scaled_cylinder.radius.get(), 8.0);
        assert_relative_eq!(scaled_cylinder.height.get(), 12.0);

        let scaled_cylinder = cylinder.scale_length(0.5.try_into()?);
        assert_relative_eq!(scaled_cylinder.radius.get(), 2.0);
        assert_relative_eq!(scaled_cylinder.height.get(), 3.0);

        Ok(())
    }

    #[test]
    fn test_scale_volume() -> Result<(), Box<dyn std::error::Error>> {
        let cylinder = Cylinder {
            radius: 4.0.try_into()?,
            height: 6.0.try_into()?,
        };

        let scaled_cylinder = cylinder.scale_volume(8.0.try_into()?);
        assert_relative_eq!(scaled_cylinder.radius.get(), 8.0);
        assert_relative_eq!(scaled_cylinder.height.get(), 12.0);

        let scaled_cylinder = cylinder.scale_volume(0.125.try_into()?);
        assert_relative_eq!(scaled_cylinder.radius.get(), 2.0);
        assert_relative_eq!(scaled_cylinder.height.get(), 3.0);

        Ok(())
    }

    #[rstest]
    #[case::center([0.0, 0.0, 0.0], true)]
    #[case::left_inside([-0.875, 0.0, 1.5], true)]
    #[case::right_inside([0.875, 0.0, -1.5], true)]
    #[case::left_outside([-1.125, 0.0, 1.5], false)]
    #[case::top_outside([0.0, 1.125, 1.5], false)]
    #[case::up_outside([0.0, 0.0, 2.125], false)]
    #[case::down_outside([0.0, 0.0, -2.125], false)]
    #[rstest]
    fn test_is_point_inside(#[case] point: [f64; 3], #[case] expected: bool) -> anyhow::Result<()> {
    let cylinder = Cylinder {
        radius: 1.0.try_into()?,
        height: 4.0.try_into()?,
    };

    assert_eq!(cylinder.is_point_inside(&point.into()), expected);

    Ok(())
    }

    #[rstest]
    #[case::center([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])]
    #[case::left([-0.5, 0.25, 0.0], [-1.0, 0.5, 0.0])]
    #[case::top([0.0, 0.0, 1.0], [0.0, 0.0, 0.25])]
    #[case::all([0.25, 0.5, -1.0], [0.5, 1.0, -0.25])]
    #[rstest]
    fn test_map_point_inside(#[case] point: [f64; 3], #[case] expected: [f64; 3]) -> anyhow::Result<()> {
    let cylinder_a = Cylinder {
        radius: 1.0.try_into()?,
        height: 4.0.try_into()?,
    };
    let cylinder_b = Cylinder {
        radius: 2.0.try_into()?,
        height: 1.0.try_into()?,
    };

    assert_relative_eq!(cylinder_a.map_point(point.into(), &cylinder_b)?, &expected.into());
    assert_relative_eq!(cylinder_b.map_point(expected.into(), &cylinder_a)?, &point.into());
    assert_relative_eq!(cylinder_a.map_point(point.into(), &cylinder_a)?, &point.into());

    Ok(())
    }

    #[rstest]
    #[case::left([-1.5, 0.25, 0.0])]
    #[case::top([0.0, 0.0, 3.0])]
    fn test_map_point_outside(#[case] point: [f64; 3]) -> anyhow::Result<()> {
    let cylinder_a = Cylinder {
        radius: 1.0.try_into()?,
        height: 4.0.try_into()?,
    };
    let cylinder_b = Cylinder {
        radius: 2.0.try_into()?,
        height: 1.0.try_into()?,
    };

    assert_eq!(cylinder_a.map_point(point.into(), &cylinder_b), Err(Error::PointOutsideShape));

    Ok(())
    }
}

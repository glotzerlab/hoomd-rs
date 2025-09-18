// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Point

use super::{Orientation, Point, Position};
use crate::Transform;
use hoomd_vector::{Rotate, Rotation, Vector};

/// The position and orientation of an extended body.
///
/// Use [`OrientedPoint`] as a [`Body`](crate::Body) or [`Site`](crate::Site) property type.
///
/// # Example
///
/// ```
/// use hoomd_microstate::property::OrientedPoint;
/// use hoomd_vector::{Angle, Cartesian};
///
/// let point = OrientedPoint {
///     position: Cartesian::from([1.0, -3.0]),
///     orientation: Angle::from(1.2),
/// };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrientedPoint<V, R> {
    /// The location of the extended body in space.
    pub position: V,
    /// Rotate from the body's reference frame into another frame.
    pub orientation: R,
}

/// Treat [`Point`] sites as constituents of oriented rigid bodies.
impl<V, R> Transform<Point<V>> for OrientedPoint<V, R>
where
    V: Vector,
    R: Rotate<V>,
{
    /// Move [`Point`] properties from the local body frame to the system frame.
    ///
    /// ```math
    /// \vec{r} = \vec{r}_\mathrm{body} + R_\mathrm{body}(\vec{r}_\mathrm{site})
    /// ```
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use hoomd_microstate::{
    ///     Transform,
    ///     property::{OrientedPoint, Point},
    /// };
    /// use hoomd_vector::{Angle, Cartesian};
    /// use std::f64::consts::PI;
    ///
    /// let body_properties = OrientedPoint {
    ///     position: Cartesian::from([1.0, -2.0]),
    ///     orientation: Angle::from(PI / 2.0),
    /// };
    /// let site_properties = Point::new(Cartesian::from([-1.0, 0.0]));
    ///
    /// let system_site = body_properties.transform(&site_properties);
    /// assert_relative_eq!(system_site.position, [1.0, -3.0].into());
    /// ```
    #[inline]
    fn transform(&self, site_properties: &Point<V>) -> Point<V> {
        Point {
            position: self.position + self.orientation.rotate(&site_properties.position),
        }
    }
}

/// Treat [`OrientedPoint`] sites as constituents of oriented rigid bodies.
impl<V, R> Transform<OrientedPoint<V, R>> for OrientedPoint<V, R>
where
    V: Vector,
    R: Rotate<V> + Rotation,
{
    /// Move [`Point`] properties from the local body frame to the system frame.
    ///
    /// ```math
    /// \vec{r} = \vec{r}_\mathrm{body} + R_\mathrm{body}(\vec{r}_\mathrm{site})
    /// ```
    /// ```math
    /// R = R_\mathrm{body}(R_\mathrm{site})
    /// ```
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use hoomd_microstate::{Transform, property::OrientedPoint};
    /// use hoomd_vector::{Angle, Cartesian};
    /// use std::f64::consts::PI;
    ///
    /// let body_properties = OrientedPoint {
    ///     position: Cartesian::from([1.0, -2.0]),
    ///     orientation: Angle::from(PI / 2.0),
    /// };
    /// let site_properties = OrientedPoint {
    ///     position: Cartesian::from([-1.0, 0.0]),
    ///     orientation: Angle::from(PI / 4.0),
    /// };
    ///
    /// let system_site = body_properties.transform(&site_properties);
    /// assert_relative_eq!(system_site.position, [1.0, -3.0].into());
    /// assert_relative_eq!(system_site.orientation.theta, 3.0 * PI / 4.0);
    /// ```
    #[inline]
    fn transform(&self, site_properties: &OrientedPoint<V, R>) -> OrientedPoint<V, R> {
        OrientedPoint {
            position: self.position + self.orientation.rotate(&site_properties.position),
            orientation: self.orientation.combine(&site_properties.orientation),
        }
    }
}

impl<V, R> Position for OrientedPoint<V, R> {
    type Vector = V;

    #[inline]
    fn position(&self) -> &V {
        &self.position
    }

    #[inline]
    fn position_mut(&mut self) -> &mut V {
        &mut self.position
    }
}

impl<V, R> Orientation for OrientedPoint<V, R> {
    type Rotation = R;

    #[inline]
    fn orientation(&self) -> &R {
        &self.orientation
    }

    #[inline]
    fn orientation_mut(&mut self) -> &mut R {
        &mut self.orientation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::approx::assert_relative_eq;
    use std::f64::consts::PI;

    use hoomd_vector::{Cartesian, Versor};

    #[test]
    fn transform_point() {
        let body = OrientedPoint {
            position: Cartesian::from([3.0, -4.0, 5.0]),
            orientation: Versor::from_axis_angle(
                [0.0, 1.0, 0.0]
                    .try_into()
                    .expect("hard-coded vector should be non-zero"),
                -PI / 2.0,
            ),
        };

        let site = Point::new(Cartesian::from([-1.0, 2.0, -3.0]));
        let transformed_site = body.transform(&site);
        assert_relative_eq!(transformed_site.position, [6.0, -2.0, 4.0].into());
    }

    #[test]
    fn transform_oriented_point() {
        let body = OrientedPoint {
            position: Cartesian::from([3.0, -4.0, 5.0]),
            orientation: Versor::from_axis_angle(
                [0.0, 1.0, 0.0]
                    .try_into()
                    .expect("hard-coded vector should be non-zero"),
                -PI / 2.0,
            ),
        };

        let site = OrientedPoint {
            position: Cartesian::from([-1.0, 2.0, -3.0]),
            orientation: Versor::from_axis_angle(
                [1.0, 0.0, 0.0]
                    .try_into()
                    .expect("hard-coded vector should be non-zero"),
                PI / 2.0,
            ),
        };
        let transformed_site = body.transform(&site);
        assert_relative_eq!(transformed_site.position, [6.0, -2.0, 4.0].into());
        assert_relative_eq!(
            transformed_site.orientation.get(),
            &[0.5, 0.5, -0.5, 0.5].into()
        );
    }
}

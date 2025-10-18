// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Point

use super::{Orientation, Point, Position};
use crate::Transform;
use hoomd_vector::{Rotate, Cartesian, Angle, Versor};
use hoomd_manifold::{Hyperbolic, Minkowski};

/// The position and orientation of an extended body in hyperbolic space.
///
/// Use [`OrientedHyperbolicPoint`] as a [`Body`](crate::Body) or [`Site`](crate::Site) property type.
///
/// # Example
///
/// ```
/// use hoomd_manifold::{Hyperbolic, Minkowski};
/// use hoomd_microstate::property::OrientedHyperbolicPoint;
/// use hoomd_vector::Angle;
///
/// let point = OrientedHyperbolicPoint {
///     position: Hyperbolic::from_minkowski_coordinates(
///         Minkowski::from([0.0, 0.0, 1.0]),
///          1.0_f64
///     ),
///     orientation: Angle::from(2.39),
/// };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrientedHyperbolicPoint<const N:usize, R> {
    /// The location of the extended body in the system frame.
    pub position: Hyperbolic<N>,
    /// Orientation of the body in the body frame
    pub orientation: R,
}

/// Treat `Point<Hyperbolic<4>>` sites as constituents of oriented rigid bodies.
impl Transform<Point<Hyperbolic<4>>> for OrientedHyperbolicPoint<4, Versor> {
    /// Move `Point<Hyperbolic<4>>` properties from the local body frame to the system frame.
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_manifold::Hyperbolic;
    /// use hoomd_microstate::{
    ///     Transform,
    ///     property::{OrientedHyperbolicPoint, Point, Position},
    /// };
    /// use hoomd_vector::Versor;
    /// use std::f64::consts::PI;
    /// 
    /// let body_boost = 1.3;
    /// let body_orientation = Versor::from_axis_angle(
    ///         [0.0, 0.0, 1.0]
    ///             .try_into()
    ///             .expect("hard-coded vector should be non-zero"),
    ///         PI/2.0,
    /// );
    /// let site_boost = 0.4;
    /// let body = OrientedHyperbolicPoint {
    ///     position: Hyperbolic::<4>::from_polar_coordinates(body_boost, 0.0, 0.0, 1.0),
    ///     orientation: body_orientation
    /// };
    /// let site = Point::new(Hyperbolic::<4>::from_polar_coordinates(
    ///     site_boost,
    ///     PI / 4.0,
    ///     0.0,
    ///     1.0,
    /// ));
    /// let transformed_site = body.transform(&site);
    /// assert_relative_eq!(
    ///     *transformed_site.position().point(),
    ///     [
    ///         (body_boost.sinh()) * (site_boost.cosh()) - ((PI/4.0).sin()) * (body_boost.cosh()) * (site_boost.sinh()),
    ///         ((PI/4.0).cos()) * site_boost.sinh(),
    ///         0.0,
    ///         (body_boost.cosh()) * (site_boost.cosh()) - ((PI/4.0).sin()) * (body_boost.sinh()) * (site_boost.sinh())
    ///     ]
    ///     .into(),
    ///     epsilon = 1e-12
    /// );
    /// ```
    #[inline]
    fn transform(&self, site_properties: &Point<Hyperbolic<4>>) -> Point<Hyperbolic<4>> {
        let body_point = self.position.coordinates();
        let skirt = self.position.skirt();
        let body_theta = (body_point[2].powi(2) + body_point[1].powi(2))
            .sqrt()
            .atan2(body_point[0]);
        let body_phi = body_point[2].atan2(body_point[1]);
        let body_pos_boost = (body_point[3] / self.position.skirt()).acosh();
        let body_angle = self.orientation;
        let site_pos_cart: Cartesian<3> = Cartesian::from([
            site_properties.position().coordinates()[0],
            site_properties.position().coordinates()[1],
            site_properties.position().coordinates()[2]
            ]);
        let rotated_site_cart = body_angle.rotate(&site_pos_cart);
        let rotated_site_pos = Minkowski::from([
            rotated_site_cart[0],
            rotated_site_cart[1],
            rotated_site_cart[2],
            site_properties.position().coordinates()[3]
        ]);
        let transformed_point = Minkowski::from([
            rotated_site_pos[0] * (body_pos_boost.cosh()) * (body_theta.cos())
                - rotated_site_pos[1] * (body_theta.sin())
                + rotated_site_pos[3] * (body_pos_boost.sinh()) * (body_theta.cos()),
            rotated_site_pos[0] * (body_pos_boost.cosh()) * (body_theta.sin()) * (body_phi.cos())
                + rotated_site_pos[1] * (body_theta.cos()) * (body_phi.cos())
                - rotated_site_pos[2] * (body_phi.sin())
                + rotated_site_pos[3] * (body_pos_boost.sinh()) * (body_theta.sin()) * (body_phi.cos()),
            rotated_site_pos[0] * (body_pos_boost.cosh()) * (body_theta.sin()) * (body_phi.sin())
                + rotated_site_pos[1] * (body_theta.cos()) * (body_phi.sin())
                + rotated_site_pos[2] * (body_phi.cos())
                + rotated_site_pos[3] * (body_pos_boost.sinh()) * (body_theta.sin()) * (body_phi.sin()),
            rotated_site_pos[0] * (body_pos_boost.sinh()) + rotated_site_pos[3] * (body_pos_boost.cosh()),
        ]);
        let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(transformed_point, skirt);
        Point::new(new_hyperbolic)
    }
}
/// Treat `Point<Hyperbolic<3>>` sites as constituents of oriented rigid bodies.
impl Transform<Point<Hyperbolic<3>>> for OrientedHyperbolicPoint<3, Angle> {
    /// Move `Point<Hyperbolic<3>>` properties from the local body frame to the system frame.
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_manifold::Hyperbolic;
    /// use hoomd_microstate::{
    ///     Transform,
    ///     property::{OrientedHyperbolicPoint, Point},
    /// };
    /// use hoomd_vector::Angle;
    /// use std::f64::consts::PI;
    ///
    /// let body_boost = 1.1;
    /// let body_orientation = PI/2.0;
    /// let site_boost = 0.1;
    /// let body = OrientedHyperbolicPoint {
    ///     position: Hyperbolic::<3>::from_polar_coordinates(body_boost, 0.0, 1.0),
    ///     orientation: Angle::from(body_orientation)
    /// };
    /// let site = Point::new(Hyperbolic::<3>::from_polar_coordinates(site_boost, -PI/4.0, 1.0));
    /// let transformed_site = body.transform(&site);
    /// assert_relative_eq!(
    ///     *transformed_site.position.point(),
    ///     [
    ///         (body_boost.sinh()) * (site_boost.cosh()) + ((PI/4.0).cos()) * (body_boost.cosh()) * (site_boost.sinh()),
    ///         ((PI/4.0).sin()) * site_boost.sinh(),
    ///         (body_boost.cosh()) * (site_boost.cosh()) + ((PI/4.0).cos()) * (body_boost.sinh()) * (site_boost.sinh()),
    ///     ]
    ///     .into(),
    ///     epsilon = 1e-12
    /// );
    /// ```
    #[inline]
    fn transform(&self, site_properties: &Point<Hyperbolic<3>>) -> Point<Hyperbolic<3>> {
        let body_pos = self.position.coordinates();
        let skirt = self.position.skirt();
        let body_pos_theta = body_pos[1].atan2(body_pos[0]);
        let body_pos_boost = (body_pos[2] / self.position.skirt()).acosh();
        let body_angle = self.orientation.theta;
        let site_pos = site_properties.position.coordinates();
        let rotated_site_pos = Minkowski::from([
            site_pos[0]*(body_angle.cos()) - site_pos[1]*(body_angle.sin()),
            site_pos[0]*(body_angle.sin()) + site_pos[1]*(body_angle.cos()),
            site_pos[2]
        ]);
        let transformed_point = Minkowski::from([
            rotated_site_pos[0] * (body_pos_boost.cosh()) * (body_pos_theta.cos())
                - rotated_site_pos[1] * (body_pos_theta.sin())
                + rotated_site_pos[2] * (body_pos_boost.sinh()) * (body_pos_theta.cos()),
            rotated_site_pos[0] * (body_pos_boost.cosh()) * (body_pos_theta.sin())
                + rotated_site_pos[1] * (body_pos_theta.cos())
                + rotated_site_pos[2] * (body_pos_boost.sinh()) * (body_pos_theta.sin()),
            rotated_site_pos[0] * (body_pos_boost.sinh()) + rotated_site_pos[2] * (body_pos_boost.cosh()),
        ]);
        let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(transformed_point, skirt);
        Point::new(new_hyperbolic)
    }
}

impl<const N: usize, R> Position for OrientedHyperbolicPoint<N, R> {
    type Position = Hyperbolic<N>;

    #[inline]
    fn position(&self) -> &Hyperbolic<N> {
        &self.position
    }

    #[inline]
    fn position_mut(&mut self) -> &mut Hyperbolic<N> {
        &mut self.position
    }
}

impl<const N: usize, R> Orientation for OrientedHyperbolicPoint<N, R> {
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
    use approxim::assert_relative_eq;
    use std::f64::consts::PI;
    use hoomd_vector::{Angle, Versor};

    #[test]
    fn transform_oriented_h2_point() {
        let body_boost = 1.1;
        let body_orientation = PI/2.0;
        let site_boost = 0.1;
        let body = OrientedHyperbolicPoint {
            position: Hyperbolic::<3>::from_polar_coordinates(body_boost, 0.0, 1.0),
            orientation: Angle::from(body_orientation)
        };
        let site = Point::new(Hyperbolic::<3>::from_polar_coordinates(site_boost, -PI/4.0, 1.0));
        let transformed_site = body.transform(&site);
        assert_relative_eq!(
            *transformed_site.position().point(),
            [
                (body_boost.sinh()) * (site_boost.cosh()) + ((PI/4.0).cos()) * (body_boost.cosh()) * (site_boost.sinh()),
                ((PI/4.0).sin()) * site_boost.sinh(),
                (body_boost.cosh()) * (site_boost.cosh()) + ((PI/4.0).cos()) * (body_boost.sinh()) * (site_boost.sinh()),
            ]
            .into(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn transform_oriented_h3_point() {
        let body_boost = 1.3;
        let body_orientation = Versor::from_axis_angle(
                [0.0, 0.0, 1.0]
                    .try_into()
                    .expect("hard-coded vector should be non-zero"),
                PI/2.0,
        );
        let site_boost = 0.4;
        let body = OrientedHyperbolicPoint {
            position: Hyperbolic::<4>::from_polar_coordinates(body_boost, 0.0, 0.0, 1.0),
            orientation: body_orientation
        };
        let site = Point::new(Hyperbolic::<4>::from_polar_coordinates(
            site_boost,
            PI / 4.0,
            0.0,
            1.0,
        ));
        let transformed_site = body.transform(&site);
        assert_relative_eq!(
            *transformed_site.position().point(),
            [
                (body_boost.sinh()) * (site_boost.cosh()) - ((PI/4.0).sin()) * (body_boost.cosh()) * (site_boost.sinh()),
                ((PI/4.0).cos()) * site_boost.sinh(),
                0.0,
                (body_boost.cosh()) * (site_boost.cosh()) - ((PI/4.0).sin()) * (body_boost.sinh()) * (site_boost.sinh())
            ]
            .into(),
            epsilon = 1e-12
        );
    }
}
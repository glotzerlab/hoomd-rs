// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Point

use super::{Orientation, Point, Position};
use crate::Transform;
use hoomd_manifold::{Hyperbolic, Minkowski};
use hoomd_vector::{Angle, Cartesian, Rotate, Versor};
use robust::{Coord, orient2d};
use std::f64::consts::PI;

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
///         1.0_f64,
///     ),
///     orientation: Angle::from(2.39),
/// };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrientedHyperbolicPoint<const N: usize, R> {
    /// The location of the extended body in the system frame.
    pub position: Hyperbolic<N>,
    /// Orientation of the body in the body frame
    pub orientation: R,
}

impl OrientedHyperbolicPoint<3, Angle> {
    /// Compute the signed angle change when an oriented hyperbolic point is
    /// translated to another point.
    #[must_use]
    #[inline]
    pub fn parallel_transport_angle(start: &Hyperbolic<3>, destination: &Hyperbolic<3>) -> f64 {
        let p = start.to_poincare();
        let p_r = p[0].powi(2) + p[1].powi(2);
        let p_m = [p[0] / p_r, p[1] / p_r];
        let q = destination.to_poincare();
        let q_r = q[0].powi(2) + q[1].powi(2);
        let q_m = [q[0] / q_r, q[1] / q_r];
        let lambda = (q_m[0] * (q_m[0] - p_m[0]) + q_m[1] * (q_m[1] - p_m[1]))
            / (2.0 * (p_m[0] * q_m[1] - p_m[1] * q_m[0]));
        let center = [
            p_m[0] / 2.0 - p_m[1] * lambda,
            p_m[1] / 2.0 + p_m[0] * lambda,
        ];
        let theta_1 = ((p[1] - center[1]).atan2(p[0] - center[0])).rem_euclid(2.0 * PI);
        let theta_2 = ((q[1] - center[1]).atan2(q[0] - center[0])).rem_euclid(2.0 * PI);
        let abs_delta_theta = ((theta_2 - theta_1).abs()).rem_euclid(2.0 * PI);
        // get the sign of the angle change
        let p_a = Coord { x: p[0], y: p[1] };
        let p_b = Coord { x: q[0], y: q[1] };
        let p_c = Coord {
            x: center[0],
            y: center[1],
        };
        let sign = orient2d(p_a, p_b, p_c).signum();
        if sign.is_nan() {
            0.0
        } else {
            sign * abs_delta_theta
        }
    }
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
    ///     [0.0, 0.0, 1.0]
    ///         .try_into()
    ///         .expect("hard-coded vector should be non-zero"),
    ///     PI / 2.0,
    /// );
    /// let site_boost = 0.4;
    /// let body = OrientedHyperbolicPoint {
    ///     position: Hyperbolic::<4>::from_polar_coordinates(
    ///         body_boost, 0.0, 0.0, 1.0,
    ///     ),
    ///     orientation: body_orientation,
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
    ///         (body_boost.sinh()) * (site_boost.cosh())
    ///             - ((PI / 4.0).sin())
    ///                 * (body_boost.cosh())
    ///                 * (site_boost.sinh()),
    ///         ((PI / 4.0).cos()) * site_boost.sinh(),
    ///         0.0,
    ///         (body_boost.cosh()) * (site_boost.cosh())
    ///             - ((PI / 4.0).sin())
    ///                 * (body_boost.sinh())
    ///                 * (site_boost.sinh())
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
            site_properties.position().coordinates()[2],
        ]);
        let rotated_site_cart = body_angle.rotate(&site_pos_cart);
        let rotated_site_pos = Minkowski::from([
            rotated_site_cart[0],
            rotated_site_cart[1],
            rotated_site_cart[2],
            site_properties.position().coordinates()[3],
        ]);
        let transformed_point = Minkowski::from([
            rotated_site_pos[0] * (body_pos_boost.cosh()) * (body_theta.cos())
                - rotated_site_pos[1] * (body_theta.sin())
                + rotated_site_pos[3] * (body_pos_boost.sinh()) * (body_theta.cos()),
            rotated_site_pos[0] * (body_pos_boost.cosh()) * (body_theta.sin()) * (body_phi.cos())
                + rotated_site_pos[1] * (body_theta.cos()) * (body_phi.cos())
                - rotated_site_pos[2] * (body_phi.sin())
                + rotated_site_pos[3]
                    * (body_pos_boost.sinh())
                    * (body_theta.sin())
                    * (body_phi.cos()),
            rotated_site_pos[0] * (body_pos_boost.cosh()) * (body_theta.sin()) * (body_phi.sin())
                + rotated_site_pos[1] * (body_theta.cos()) * (body_phi.sin())
                + rotated_site_pos[2] * (body_phi.cos())
                + rotated_site_pos[3]
                    * (body_pos_boost.sinh())
                    * (body_theta.sin())
                    * (body_phi.sin()),
            rotated_site_pos[0] * (body_pos_boost.sinh())
                + rotated_site_pos[3] * (body_pos_boost.cosh()),
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
    /// let body_orientation = PI / 2.0;
    /// let site_boost = 0.1;
    /// let body = OrientedHyperbolicPoint {
    ///     position: Hyperbolic::<3>::from_polar_coordinates(body_boost, 0.0, 1.0),
    ///     orientation: Angle::from(body_orientation),
    /// };
    /// let site = Point::new(Hyperbolic::<3>::from_polar_coordinates(
    ///     site_boost,
    ///     -PI / 4.0,
    ///     1.0,
    /// ));
    /// let transformed_site = body.transform(&site);
    /// assert_relative_eq!(
    ///     *transformed_site.position.point(),
    ///     [
    ///         (body_boost.sinh()) * (site_boost.cosh())
    ///             + ((PI / 4.0).cos())
    ///                 * (body_boost.cosh())
    ///                 * (site_boost.sinh()),
    ///         ((PI / 4.0).sin()) * site_boost.sinh(),
    ///         (body_boost.cosh()) * (site_boost.cosh())
    ///             + ((PI / 4.0).cos())
    ///                 * (body_boost.sinh())
    ///                 * (site_boost.sinh()),
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
            site_pos[0] * (body_angle.cos()) - site_pos[1] * (body_angle.sin()),
            site_pos[0] * (body_angle.sin()) + site_pos[1] * (body_angle.cos()),
            site_pos[2],
        ]);
        let transformed_point = Minkowski::from([
            rotated_site_pos[0] * (body_pos_boost.cosh()) * (body_pos_theta.cos())
                - rotated_site_pos[1] * (body_pos_theta.sin())
                + rotated_site_pos[2] * (body_pos_boost.sinh()) * (body_pos_theta.cos()),
            rotated_site_pos[0] * (body_pos_boost.cosh()) * (body_pos_theta.sin())
                + rotated_site_pos[1] * (body_pos_theta.cos())
                + rotated_site_pos[2] * (body_pos_boost.sinh()) * (body_pos_theta.sin()),
            rotated_site_pos[0] * (body_pos_boost.sinh())
                + rotated_site_pos[2] * (body_pos_boost.cosh()),
        ]);
        let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(transformed_point, skirt);
        Point::new(new_hyperbolic)
    }
}

impl Transform<OrientedHyperbolicPoint<3, Angle>> for OrientedHyperbolicPoint<3, Angle> {
    /// TODO
    #[inline]
    fn transform(
        &self,
        site_properties: &OrientedHyperbolicPoint<3, Angle>,
    ) -> OrientedHyperbolicPoint<3, Angle> {
        let body_pos = self.position.coordinates();
        let skirt = self.position.skirt();
        let body_pos_theta = body_pos[1].atan2(body_pos[0]);
        let body_pos_boost = (body_pos[2] / self.position.skirt()).acosh();
        let body_angle = self.orientation.theta;
        let site_pos = site_properties.position.coordinates();
        let rotated_site_pos = Minkowski::from([
            site_pos[0] * (body_angle.cos()) - site_pos[1] * (body_angle.sin()),
            site_pos[0] * (body_angle.sin()) + site_pos[1] * (body_angle.cos()),
            site_pos[2],
        ]);
        let transformed_point = Minkowski::from([
            rotated_site_pos[0] * (body_pos_boost.cosh()) * (body_pos_theta.cos())
                - rotated_site_pos[1] * (body_pos_theta.sin())
                + rotated_site_pos[2] * (body_pos_boost.sinh()) * (body_pos_theta.cos()),
            rotated_site_pos[0] * (body_pos_boost.cosh()) * (body_pos_theta.sin())
                + rotated_site_pos[1] * (body_pos_theta.cos())
                + rotated_site_pos[2] * (body_pos_boost.sinh()) * (body_pos_theta.sin()),
            rotated_site_pos[0] * (body_pos_boost.sinh())
                + rotated_site_pos[2] * (body_pos_boost.cosh()),
        ]);
        let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(transformed_point, skirt);
        OrientedHyperbolicPoint {
            position: new_hyperbolic,
            orientation: Angle::from(body_angle + site_properties.orientation.theta),
        }
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
    use hoomd_vector::{Angle, Versor};
    use std::f64::consts::PI;

    #[test]
    fn transform_oriented_h2_point() {
        let body_boost = 1.1;
        let body_orientation = PI / 2.0;
        let site_boost = 0.1;
        let body = OrientedHyperbolicPoint {
            position: Hyperbolic::<3>::from_polar_coordinates(body_boost, 0.0, 1.0),
            orientation: Angle::from(body_orientation),
        };
        let site = Point::new(Hyperbolic::<3>::from_polar_coordinates(
            site_boost,
            -PI / 4.0,
            1.0,
        ));
        let transformed_site = body.transform(&site);
        assert_relative_eq!(
            *transformed_site.position().point(),
            [
                (body_boost.sinh()) * (site_boost.cosh())
                    + ((PI / 4.0).cos()) * (body_boost.cosh()) * (site_boost.sinh()),
                ((PI / 4.0).sin()) * site_boost.sinh(),
                (body_boost.cosh()) * (site_boost.cosh())
                    + ((PI / 4.0).cos()) * (body_boost.sinh()) * (site_boost.sinh()),
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
            PI / 2.0,
        );
        let site_boost = 0.4;
        let body = OrientedHyperbolicPoint {
            position: Hyperbolic::<4>::from_polar_coordinates(body_boost, 0.0, 0.0, 1.0),
            orientation: body_orientation,
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
                (body_boost.sinh()) * (site_boost.cosh())
                    - ((PI / 4.0).sin()) * (body_boost.cosh()) * (site_boost.sinh()),
                ((PI / 4.0).cos()) * site_boost.sinh(),
                0.0,
                (body_boost.cosh()) * (site_boost.cosh())
                    - ((PI / 4.0).sin()) * (body_boost.sinh()) * (site_boost.sinh())
            ]
            .into(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn parallel_transport() {
        let boost = 1.098_612_288_668_11;
        let pt_1 = Hyperbolic::<3>::from_polar_coordinates(boost, 0.0, 1.0);
        let pt_2 = Hyperbolic::<3>::from_polar_coordinates(boost, PI / 2.0, 1.0);
        let del_angle = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(&pt_1, &pt_2);
        let answer = -0.643_501_108_793_284;
        assert_relative_eq!(answer, del_angle, epsilon = 1e-8);
    }
}

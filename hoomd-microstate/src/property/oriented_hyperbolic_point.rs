// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `OrientedHyperbolicPoint`

use super::{Orientation, Point, Position};
use crate::Transform;
use hoomd_manifold::{Hyperbolic, HyperbolicDisk, Minkowski};
use hoomd_vector::Angle;
use rand::{
    Rng,
    distr::{Distribution, Uniform},
};
use robust::{Coord, orient2d};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// The position and orientation of an extended body in hyperbolic space.
///
/// Use [`OrientedHyperbolicPoint`] as a [`Body`](crate::Body) or
/// [`Site`](crate::Site) property type.
///
/// # Example
///
/// ```
/// use hoomd_manifold::{Hyperbolic, Minkowski};
/// use hoomd_microstate::property::OrientedHyperbolicPoint;
/// use hoomd_vector::Angle;
///
/// let point = OrientedHyperbolicPoint {
///     position: Hyperbolic::from_minkowski_coordinates(Minkowski::from([
///         0.0, 0.0, 1.0,
///     ])),
///     orientation: Angle::from(2.39),
/// };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OrientedHyperbolicPoint<const N: usize, R> {
    /// The location of the extended body in the system frame.
    pub position: Hyperbolic<N>,
    /// Orientation of the body in the global frame.
    pub orientation: R,
}

impl OrientedHyperbolicPoint<3, Angle> {
    /// Compute the intersection of the geodesic arc passing through a point and the
    /// x axis, given the polar coordinates of the point. Returns a tuple containing
    /// the x coordinates of the intersection and the signed arc-angle of the path.
    #[must_use]
    #[inline]
    pub fn intersection_point(theta: f64, boost: f64) -> (f64, f64) {
        let rad = (boost.sinh()) / (1.0 + (boost.cosh())); //radius in poincare coordinates
        let x_0 = (1.0 + rad.powi(2)
            - ((1.0 + rad.powi(2)).powi(2) - 4.0 * (rad.powi(2)) * ((theta.cos()).powi(2))).sqrt())
            / (2.0 * rad * (theta.cos()));
        let p = Hyperbolic::<3>::from_polar_coordinates(boost, theta);
        let intersect = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            2.0 * x_0 / (1.0 - x_0.powi(2)),
            0.0,
            (1.0 + x_0.powi(2)) / (1.0 - x_0.powi(2)),
        ]));
        let angle_change = Self::parallel_transport_angle(&p, &intersect);
        let end_boost = (intersect.coordinates()[2]).acosh();
        (end_boost, angle_change)
    }
    /// Compute the signed angle change when an oriented hyperbolic point is
    /// translated to another point.
    #[must_use]
    #[inline]
    pub fn parallel_transport_angle(start: &Hyperbolic<3>, destination: &Hyperbolic<3>) -> f64 {
        let (b1, theta_1) = (
            (start.coordinates()[2]).acosh(),
            (start.coordinates()[1]).atan2(start.coordinates()[0]),
        );
        let (b2, theta_2) = (
            (destination.coordinates()[2]).acosh(),
            (destination.coordinates()[1]).atan2(destination.coordinates()[0]),
        );
        // rotate into non-pathological frame
        let phi_1 = -(theta_2 - theta_1) / 2.0;
        let phi_2 = (theta_2 - theta_1) / 2.0;

        let p = [
            (b1.sinh()) * (phi_1.cos()) / (1.0 + (b1.cosh())),
            (b1.sinh()) * (phi_1.sin()) / (1.0 + (b1.cosh())),
        ];
        let q = [
            (b2.sinh()) * (phi_2.cos()) / (1.0 + (b2.cosh())),
            (b2.sinh()) * (phi_2.sin()) / (1.0 + (b2.cosh())),
        ];
        let center = [
            ((p[0].powi(2) + p[1].powi(2) + 1.0) * q[1]
                - (q[0].powi(2) + q[1].powi(2) + 1.0) * p[1])
                / (2.0 * (p[0] * q[1] - q[0] * p[1])),
            (-(p[0].powi(2) + p[1].powi(2) + 1.0) * q[0]
                + (q[0].powi(2) + q[1].powi(2) + 1.0) * p[0])
                / (2.0 * (p[0] * q[1] - q[0] * p[1])),
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
    /// Compute the change in orientation associated with a transformation from a given
    /// position. Isometries of hyperbolic space can be expressed as a boost conjugated
    /// by a rotation, i.e., $`R(\theta) T(\eta) R(-\theta)`$. Such a transformation can
    /// be expressed in the Poincaré disk representation as the Mobius transformation
    /// ```math
    /// g(z) = \begin{bmatrix} \cosh(\eta/2) & e^{i\theta} \sinh(\eta/2) \\ e^{-i\theta}\sinh(\eta/2) & \cosh(\eta/2) \end{bmatrix}
    /// ```
    /// This transformation induces a rotation of angle $`\Delta\phi`$ in the tangent
    /// bundle, where
    /// ```math
    /// \Delta \phi = \operatorname{Arg} [g'(z)] = -2 \operatorname{Arg}[e^{-i\theta}\sinh(\eta/2)z + \cosh(\eta/2)]
    /// ```
    #[inline]
    #[must_use]
    pub fn deck_transform(boost: f64, rotation: f64, position: &Hyperbolic<3>) -> f64 {
        let tau_over_two = boost / 2.0;
        let poincare = position.to_poincare();
        -2.0 * (((tau_over_two.sinh())
            * ((rotation.cos()) * poincare[1] - (rotation.sin()) * poincare[0]))
            .atan2(
                (tau_over_two.cosh())
                    + (tau_over_two.sinh())
                        * ((rotation.cos()) * poincare[0] + (rotation.sin()) * poincare[1]),
            ))
    }
    /// Generate a random `HyperbolicDisk` point with the corresponding boost rapidity
    /// and rotation angle. Function returns a tuple with the random `Hyperbolic<3>`
    /// point in the first entry, the associated rapidity and angle in the second and
    /// third entries, respectively.
    ///
    /// # Panics
    /// Panics when maximum boost is a non-positive number.
    #[inline]
    #[must_use]
    pub fn sample<R: Rng + ?Sized>(
        disk: &HyperbolicDisk,
        rng: &mut R,
    ) -> (Hyperbolic<3>, f64, f64) {
        let max_boost = disk.disk_radius.get();
        let point = disk.point;
        let eta = (point.coordinates()[2]).acosh();
        let phi = point.coordinates()[1].atan2(point.coordinates()[0]);
        let trial_boost = Uniform::new(0.0, 1.0).expect("r is positive and real");
        let trial_rotation =
            Uniform::new(-PI, PI).expect("hard-coded distribution should be valid");
        let theta = trial_rotation.sample(rng);
        let v1: f64 = trial_boost.sample(rng);
        let v = v1.sqrt() * max_boost;
        let (v_sinh, eta_sinh, eta_cosh) = (v.sinh(), eta.sinh(), eta.cosh());
        let (phi_sin, phi_cos) = (phi.sin(), phi.cos());
        let trial_coords = [v_sinh * theta.cos(), v_sinh * theta.sin(), v.cosh()];
        let transformed_point = Minkowski::from([
            trial_coords[0] * (eta_cosh * (phi_cos.powi(2)) + phi_sin.powi(2))
                + trial_coords[1] * phi_sin * phi_cos * (eta_cosh - 1.0)
                + trial_coords[2] * eta_sinh * phi_cos,
            trial_coords[0] * phi_sin * phi_cos * (eta_cosh - 1.0)
                + trial_coords[1] * (eta_cosh * (phi_sin.powi(2)) + phi_cos.powi(2))
                + trial_coords[2] * eta_sinh * phi_sin,
            trial_coords[0] * eta_sinh * phi_cos
                + trial_coords[1] * eta_sinh * phi_sin
                + trial_coords[2] * eta_cosh,
        ]);
        (
            Hyperbolic::from_minkowski_coordinates(transformed_point),
            v,
            theta,
        )
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
    ///     position: Hyperbolic::<3>::from_polar_coordinates(body_boost, 0.0),
    ///     orientation: Angle::from(body_orientation),
    /// };
    /// let site = Point::new(Hyperbolic::<3>::from_polar_coordinates(
    ///     site_boost,
    ///     -PI / 4.0,
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
        let body_pos_theta = (body_pos[1].atan2(body_pos[0])).rem_euclid(2.0 * PI);
        let body_pos_boost = body_pos[2].acosh();
        let body_angle_system = self.orientation.theta;
        let body_angle_body =
            Self::deck_transform(-body_pos_boost, body_pos_theta, self.position())
                + body_angle_system;
        let site_pos = site_properties.position.coordinates();
        let (body_pos_boost_cosh, body_pos_boost_sinh) =
            (body_pos_boost.cosh(), body_pos_boost.sinh());
        let (bpb_cos, bpb_sin, diff_cos, diff_sin) = (
            body_pos_theta.cos(),
            body_pos_theta.sin(),
            (body_angle_body - body_pos_theta).cos(),
            (body_angle_body - body_pos_theta).sin(),
        );
        let transformed_point = Minkowski::from([
            site_pos[0]
                * (body_pos_boost_cosh * bpb_cos * diff_cos - diff_sin * (body_pos_theta).sin())
                - site_pos[1] * (bpb_sin * diff_cos + body_pos_boost_cosh * bpb_cos * diff_sin)
                + site_pos[2] * body_pos_boost_sinh * bpb_cos,
            site_pos[0] * (body_pos_boost_cosh * bpb_sin * diff_cos + bpb_cos * diff_sin)
                + site_pos[1] * (bpb_cos * diff_cos - body_pos_boost_cosh * bpb_sin * diff_sin)
                + site_pos[2] * body_pos_boost_sinh * bpb_sin,
            site_pos[0] * body_pos_boost_sinh * diff_cos
                - site_pos[1] * body_pos_boost_sinh * diff_sin
                + site_pos[2] * body_pos_boost_cosh,
        ]);
        let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(transformed_point);
        Point::new(new_hyperbolic)
    }
}

impl Transform<OrientedHyperbolicPoint<3, Angle>> for OrientedHyperbolicPoint<3, Angle> {
    #[inline]
    fn transform(
        &self,
        site_properties: &OrientedHyperbolicPoint<3, Angle>,
    ) -> OrientedHyperbolicPoint<3, Angle> {
        let body_pos = self.position.coordinates();
        let body_pos_theta = body_pos[1].atan2(body_pos[0]);
        let body_pos_boost = body_pos[2].acosh();
        let body_angle_system = self.orientation.theta;
        let body_angle_body =
            Self::deck_transform(-body_pos_boost, body_pos_theta, self.position())
                + body_angle_system;
        let (body_pos_boost_cosh, body_pos_boost_sinh) =
            (body_pos_boost.cosh(), body_pos_boost.sinh());
        let (bpb_cos, bpb_sin, diff_cos, diff_sin) = (
            body_pos_theta.cos(),
            body_pos_theta.sin(),
            (body_angle_body - body_pos_theta).cos(),
            (body_angle_body - body_pos_theta).sin(),
        );
        let site_pos = site_properties.position.coordinates();
        let transformed_point = Minkowski::from([
            site_pos[0] * (body_pos_boost_cosh * bpb_cos * diff_cos - diff_sin * bpb_sin)
                - site_pos[1] * (bpb_sin * diff_cos + body_pos_boost_cosh * bpb_cos * diff_sin)
                + site_pos[2] * body_pos_boost_sinh * bpb_cos,
            site_pos[0] * (body_pos_boost_cosh * bpb_sin * diff_cos + bpb_cos * diff_sin)
                + site_pos[1] * (bpb_cos * diff_cos - body_pos_boost_cosh * bpb_sin * diff_sin)
                + site_pos[2] * body_pos_boost_sinh * bpb_sin,
            site_pos[0] * body_pos_boost_sinh * diff_cos
                - site_pos[1] * body_pos_boost_sinh * diff_sin
                + site_pos[2] * body_pos_boost_cosh,
        ]);
        let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(transformed_point);
        let site_angle_system =
            Self::deck_transform(body_pos_boost, body_pos_theta, &site_properties.position)
                + site_properties.orientation.theta
                + body_angle_system;
        OrientedHyperbolicPoint {
            position: new_hyperbolic,
            orientation: Angle::from(site_angle_system),
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
    use hoomd_geometry::shape::EightEight;
    use hoomd_vector::{Angle, Metric};
    use std::f64::consts::PI;

    #[test]
    fn transform_oriented_h2_point() {
        let body_boost = 1.1;
        let body_orientation = PI / 2.0;
        let site_boost = 0.1;
        let body = OrientedHyperbolicPoint {
            position: Hyperbolic::<3>::from_polar_coordinates(body_boost, 0.0),
            orientation: Angle::from(body_orientation),
        };
        let site = Point::new(Hyperbolic::<3>::from_polar_coordinates(
            site_boost,
            -PI / 4.0,
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
    fn parallel_transport_method() {
        let p = [-(2.0_f64).sqrt() / 2.0, (2.0_f64).sqrt() / 2.0];
        let q = [(2.0_f64).sqrt() / 2.0, (2.0_f64).sqrt() / 2.0];
        let center = [
            ((p[0].powi(2) + p[1].powi(2) + 1.0) * q[1]
                - (q[0].powi(2) + q[1].powi(2) + 1.0) * p[1])
                / (2.0 * (p[0] * q[1] - q[0] * p[1])),
            (-(p[0].powi(2) + p[1].powi(2) + 1.0) * q[0]
                + (q[0].powi(2) + q[1].powi(2) + 1.0) * p[0])
                / (2.0 * (p[0] * q[1] - q[0] * p[1])),
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
        let del_angle = sign * abs_delta_theta;
        let answer = PI / 2.0;

        assert_relative_eq!(answer, del_angle, epsilon = 1e-8);
    }

    #[test]
    fn parallel_transport() {
        let p1 = Hyperbolic::<3>::from_polar_coordinates(EightEight::EIGHTEIGHT, -PI / 4.0);
        let p2 = Hyperbolic::<3>::from_polar_coordinates(EightEight::EIGHTEIGHT, 0.0);
        let del_angle = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(&p1, &p2);
        assert_relative_eq!(-PI / 2.0, del_angle, epsilon = 1e-12);
    }

    #[test]
    fn intersection_points() {
        let (x_0, del_theta) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            PI / 8.0,
            EightEight::EIGHTEIGHT,
        );
        assert_relative_eq!(EightEight::CUSP_TO_EDGE, x_0, epsilon = 1e-12);
        assert_relative_eq!(PI / 4.0, del_theta, epsilon = 1e-12);
    }

    #[test]
    fn shortest_distance() {
        let offset_boost = 0.2;
        let v: f64 = 2.448_452_447_678_076;
        let point = Hyperbolic::<3>::from_polar_coordinates(v - offset_boost, 0.01);

        // get nearest boundary point
        let point_in_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * point.coordinates()[0] - (v.sinh()) * point.coordinates()[2],
            point.coordinates()[1],
            -(v.sinh()) * point.coordinates()[0] + (v.cosh()) * point.coordinates()[2],
        ]));
        let (angle_c, boost_c) = (
            point_in_center.coordinates()[1].atan2(point_in_center.coordinates()[0]),
            (point_in_center.coordinates()[2]).acosh(),
        );
        let (int_o, _o_o) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            angle_c - 7.0 * PI / 8.0,
            boost_c,
        );
        let intersection_o = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * (int_o.sinh()) * ((7.0 * PI / 8.0).cos()) + (v.sinh()) * (int_o.cosh()),
            (int_o.sinh()) * ((7.0 * PI / 8.0).sin()),
            (v.sinh()) * (int_o.sinh()) * ((7.0 * PI / 8.0).cos()) + (v.cosh()) * (int_o.cosh()),
        ]));
        let blip = 0.05;
        let intersection_plus = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((blip + int_o).sinh()) * ((7.0 * PI / 8.0).cos())
                + (v.sinh()) * ((blip + int_o).cosh()),
            ((blip + int_o).sinh()) * ((7.0 * PI / 8.0).sin()),
            (v.sinh()) * ((blip + int_o).sinh()) * ((7.0 * PI / 8.0).cos())
                + (v.cosh()) * ((blip + int_o).cosh()),
        ]));
        let intersection_minus = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((blip - int_o).sinh()) * ((7.0 * PI / 8.0).cos())
                + (v.sinh()) * ((blip - int_o).cosh()),
            ((blip - int_o).sinh()) * ((7.0 * PI / 8.0).sin()),
            (v.sinh()) * ((blip - int_o).sinh()) * ((7.0 * PI / 8.0).cos())
                + (v.cosh()) * ((blip - int_o).cosh()),
        ]));
        assert!(point.distance(&intersection_plus) > point.distance(&intersection_o));
        assert!(point.distance(&intersection_minus) > point.distance(&intersection_o));
    }
}

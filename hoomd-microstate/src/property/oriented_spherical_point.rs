// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `OrientedSphericalPoint`

use super::{Orientation, Point, Position};
use crate::Transform;
use hoomd_manifold::Spherical;
use hoomd_vector::{Angle, Cartesian};
use rand::{
    Rng,
    distr::{Distribution, Uniform},
};
use robust::{Coord, orient2d};
use std::f64::consts::PI;

/// The position and orientation of an extended body in spherical space. The
/// orientation is defined as the angle of the body in the stereographic
/// projection through the "north pole" of the sphere.
///
/// Use [`OrientedSphericalPoint`] as a [`Body`](crate::Body) or
/// [`Site`](crate::Site) property type.
///
/// # Example
///
/// ```
/// use hoomd_manifold::Spherical;
/// use hoomd_microstate::property::OrientedSphericalPoint;
/// use hoomd_vector::Angle;
///
/// let point = OrientedSphericalPoint {
///     position: Spherical::<3>::from_polar_coordinates(0.1, 0.2),
///     orientation: Angle::from(0.3),
/// };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrientedSphericalPoint<const N: usize, R> {
    /// The location of the extended body in the system frame.
    pub position: Spherical<N>,
    /// Orientation of the body in the global frame.
    pub orientation: R,
}

impl OrientedSphericalPoint<3, Angle> {
    /// Compute the change in orientation associated with a transformation from a given
    /// position. Isometries of the 2-sphere can be expressed as a zenith rotation
    /// conjugated by an azimuthal rotation. These $`SO(3)`$ operations can be mapped
    /// to $`SU(2)`$ operations, where they act on the stereographic projected points
    /// as the Möbius transformation
    /// ```math
    /// g(z) = \begin{bmatrix} \cos(\theta/2) & e^{i\phi} \sin(\theta/2) \\ -e^{-i\theta}\sin(\theta/2) & \cos(\theta/2) \end{bmatrix}
    /// ```
    /// This transformation induces a rotation of angle $`\Delta\psi`$ in the tangent
    /// bundle, where
    /// ```math
    /// \Delta \phi = \operatorname{Arg} [g'(z)] = -2 \operatorname{Arg}[\cos(\theta/2) - e^{-i\phi}\sin(\theta/2)z]
    /// ```
    #[inline]
    #[must_use]
    pub fn deck_transform(zenith: f64, azimuth: f64, position: &Spherical<3>) -> f64 {
        let theta_over_two = zenith/2.0;
        let stereo = position.stereographic_projection_from_north_pole();
        -2.0 * (((theta_over_two.sin())
            * ((azimuth.sin()) * stereo[0] - (azimuth.cos()) * stereo[1]))
            .atan2(
                (theta_over_two.cos())
                    - (theta_over_two.sin())
                        * ((azimuth.cos()) * stereo[0] + (azimuth.sin()) * stereo[1]),
            ))
    }
}

impl<const N: usize, R> Position for OrientedSphericalPoint<N, R> {
    type Position = Spherical<N>;

    #[inline]
    fn position(&self) -> &Spherical<N> {
        &self.position
    }

    #[inline]
    fn position_mut(&mut self) -> &mut Self::Position {
        &mut self.position
    }
}

impl<const N: usize, R> Orientation for OrientedSphericalPoint<N, R> {
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

impl Transform<Point<Spherical<3>>> for OrientedSphericalPoint<3, Angle> {
    /// Move `Point<Spherical<3>>` from the local  body from to the system frame.
    /// TODO: example
    #[inline]
    fn transform(&self, site_properties: &Point<Spherical<3>>) -> Point<Spherical<3>> {
        let body_pos = self.position.coordinates();
        let body_pos_phi = body_pos[1].atan2(body_pos[0]);
        let body_pos_theta = (body_pos[2]).acos();
        let body_angle = self.orientation.theta;
        let body_angle_body = Self::deck_transform(-body_pos_theta, body_pos_phi, self.position()) + body_angle;
        let site_pos = site_properties.position.coordinates();
        let (theta_cos, theta_sin) = (body_pos_theta.cos(), body_pos_theta.sin());
        let (phi_cos, phi_sin, diff_cos, diff_sin) = (
            body_pos_phi.cos(),
            body_pos_phi.sin(),
            (body_angle_body - body_pos_phi).cos(),
            (body_angle_body - body_pos_phi).sin(),
        );
        let transformed_point = Cartesian::from([
            site_pos[0] * (theta_cos * phi_cos * diff_cos - phi_sin * diff_sin)
            - site_pos[1] * (theta_cos * phi_cos * diff_sin + phi_sin * diff_cos)
            + site_pos[2] * theta_sin * phi_cos,
            site_pos[0] * (theta_cos * phi_sin * diff_cos + phi_cos * diff_sin)
            + site_pos[1] * (phi_cos * diff_cos - theta_cos * phi_sin * diff_sin)
            + site_pos[2] * theta_sin * phi_sin,
            -site_pos[0] * theta_sin * diff_cos
            +site_pos[1] * theta_sin * diff_sin
            +site_pos[2] * theta_cos,
        ]);
        let new_spherical = Spherical::from_cartesian_coordinates(transformed_point);
        Point::new(new_spherical)
    }
}

impl Transform<OrientedSphericalPoint<3, Angle>> for OrientedSphericalPoint<3, Angle> {
    #[inline]
    fn transform(&self, site_properties: &OrientedSphericalPoint<3, Angle>) -> OrientedSphericalPoint<3, Angle> {
        let body_pos = self.position.coordinates();
        let body_pos_phi = body_pos[1].atan2(body_pos[0]);
        let body_pos_theta = (body_pos[2]).acos();
        let body_angle = self.orientation.theta;
        let site_pos = site_properties.position.coordinates();
        let (theta_cos, theta_sin) = (body_pos_theta.cos(), body_pos_theta.sin());
        let (phi_cos, phi_sin, diff_cos, diff_sin) = (
            body_pos_phi.cos(),
            body_pos_phi.sin(),
            (body_angle - body_pos_phi).cos(),
            (body_angle - body_pos_phi).sin(),
        );
        let transformed_point = Cartesian::from([
            site_pos[0] * (theta_cos * phi_cos * diff_cos - phi_sin * diff_sin)
            - site_pos[1] * (theta_cos * phi_cos * diff_sin + phi_sin * diff_cos)
            + site_pos[2] * theta_sin * phi_cos,
            site_pos[0] * (theta_cos * phi_sin * diff_cos + phi_cos * diff_sin)
            + site_pos[1] * (phi_cos * diff_cos - theta_cos * phi_sin * diff_sin)
            + site_pos[2] * theta_sin * phi_sin,
            -site_pos[0] * theta_sin * diff_cos
            +site_pos[1] * theta_sin * diff_sin
            +site_pos[2] * theta_cos,
        ]);
        let new_spherical = Spherical::from_cartesian_coordinates(transformed_point);
        let site_angle_system = Self::deck_transform(body_pos_theta, body_pos_phi, &site_properties.position) + site_properties.orientation.theta + body_angle;
        OrientedSphericalPoint {
            position: new_spherical,
            orientation: Angle::from(site_angle_system)

        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn deck_transforms() {
        let p1 = Spherical::<3>::from_polar_coordinates(PI/2.0, 0.0);
        let val1 = OrientedSphericalPoint::<3,Angle>::deck_transform(-PI/4.0, 0.0, &p1);
        //translation along a meridian should not change angle
        assert_relative_eq!(val1, 0.0);

        let val2 = OrientedSphericalPoint::<3,Angle>::deck_transform(PI/3.0, 1.0, &p1);
        //translation along a parallel should change angle
        assert_relative_eq!(val2,-1.2295910599142);

        let p2 = Spherical::<3>::from_polar_coordinates(PI/4.0, 0.5);
        let body_pos = p2.coordinates();
        let body_pos_phi = body_pos[1].atan2(body_pos[0]);
        let body_pos_theta = (body_pos[2]).acos();
        let body_angle_body = OrientedSphericalPoint::deck_transform(-body_pos_theta, body_pos_phi, &p2);
        // transformation along geodesic "line of sight" induces no change
        assert_relative_eq!(body_angle_body, 0.0);
    }

    #[test]
    fn transform_s2_point() {
        let theta = PI / 5.0;
        let blip = PI / 10.0;
        let body_position = Spherical::<3>::from_polar_coordinates(theta, 0.0);
        let site_position = Spherical::<3>::from_polar_coordinates(blip, PI/4.0);
        let body = OrientedSphericalPoint{
            position: body_position,
            orientation: Angle::from(PI/4.0),
        };
        let site = OrientedSphericalPoint {
            position: site_position,
            orientation: Angle::from(0.0)
        };
        let transformed_site = body.transform(&site);
        assert_relative_eq!(
            *transformed_site.position().point(),
            [
                (theta.sin()) * (blip.cos()),
                blip.sin(),
                (theta.cos()) * (blip.cos())
            ]
            .into()
        );
        let del_psi = OrientedSphericalPoint::deck_transform(theta, 0.0, &site_position);
        assert_relative_eq!(transformed_site.orientation().theta, del_psi + PI/4.0);
    }
}

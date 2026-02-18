// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Point

use serde::{Deserialize, Serialize};

use super::Position;
use crate::Transform;
use hoomd_manifold::{Hyperbolic, Minkowski, Spherical};
use hoomd_vector::Cartesian;

/// A position in space and nothing more.
///
/// Use [`Point`] as a [`Body`](crate::Body) or [`Site`](crate::Site) property type.
///
/// # Example
///
/// ```
/// use hoomd_microstate::property::Point;
/// use hoomd_vector::Cartesian;
///
/// let point = Point::new(Cartesian::from([1.0, -2.0, 3.0]));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point<P> {
    /// The location of the point in space.
    pub position: P,
}

impl<P> Point<P> {
    /// Construct a new point at the given position.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::property::Point;
    /// use hoomd_vector::Cartesian;
    ///
    /// let point = Point::new(Cartesian::from([1.0, -2.0, 3.0]));
    /// ```
    #[inline]
    #[must_use]
    pub fn new(position: P) -> Self {
        Self { position }
    }
}

/// Move [`Point`] properties from the local body frame to the system frame.
impl<const N: usize> Transform<Point<Cartesian<N>>> for Point<Cartesian<N>> {
    /// Points transform by vector addition.
    ///
    /// ```math
    /// \vec{r} = \vec{r}_\mathrm{body} + \vec{r}_\mathrm{site}
    /// ```
    ///
    /// ```
    /// use hoomd_microstate::{Transform, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// let body_properties = Point::new(Cartesian::from([1.0, -2.0, 3.0]));
    /// let site_properties = Point::new(Cartesian::from([-3.0, 2.0, 1.0]));
    ///
    /// let system_site = body_properties.transform(&site_properties);
    /// assert_eq!(system_site.position, [-2.0, 0.0, 4.0].into());
    /// ```
    #[inline]
    fn transform(&self, site_properties: &Point<Cartesian<N>>) -> Point<Cartesian<N>> {
        Point {
            position: self.position + site_properties.position,
        }
    }
}

impl Transform<Point<Hyperbolic<3>>> for Point<Hyperbolic<3>> {
    /// Move `Point<Hyperbolic<3>>` properties from the local body frame to the
    /// system frame.
    ///
    /// All positions in hyperbolic space are associated with some $`SO(2,1)`$
    /// transformation which translates the origin to that position. The local
    /// body frame is the frame in which the body position is the origin. The
    /// position of the sites in the system frame is obtained by applying the
    /// transformation associated with the body's position to the sites in the
    /// local body frame.
    #[inline]
    fn transform(&self, site_properties: &Point<Hyperbolic<3>>) -> Point<Hyperbolic<3>> {
        let body_pos = self.position.coordinates();
        let skirt = self.position.skirt();
        let body_theta = body_pos[1].atan2(body_pos[0]);
        let body_boost = (body_pos[2] / self.position.skirt()).acosh();
        let site_pos = site_properties.position.coordinates();
        let transformed_point = Minkowski::from([
            site_pos[0] * (body_boost.cosh()) * (body_theta.cos())
                - site_pos[1] * (body_theta.sin())
                + site_pos[2] * (body_boost.sinh()) * (body_theta.cos()),
            site_pos[0] * (body_boost.cosh()) * (body_theta.sin())
                + site_pos[1] * (body_theta.cos())
                + site_pos[2] * (body_boost.sinh()) * (body_theta.sin()),
            site_pos[0] * (body_boost.sinh()) + site_pos[2] * (body_boost.cosh()),
        ]);
        let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(transformed_point, skirt);
        Point::new(new_hyperbolic)
    }
}

impl Transform<Point<Hyperbolic<4>>> for Point<Hyperbolic<4>> {
    /// Move `Point<Hyperbolic<4>>` properties from the local body frame to the
    /// system frame.
    ///
    /// All positions in hyperbolic space are associated with some $`SO(3,1)`$
    /// transformation which translates the origin to that position. The local
    /// body frame is the frame in which the body position is the origin. The
    /// position of the sites in the system frame is obtained by applying the
    /// transformation associated with the body's position to the sites in the
    /// local body frame.
    #[inline]
    fn transform(&self, site_properties: &Point<Hyperbolic<4>>) -> Point<Hyperbolic<4>> {
        let body_point = self.position.coordinates();
        let skirt = self.position.skirt();
        let body_theta = (body_point[2].powi(2) + body_point[1].powi(2))
            .sqrt()
            .atan2(body_point[0]);
        let body_phi = body_point[2].atan2(body_point[1]);
        let body_boost = (body_point[3] / self.position.skirt()).acosh();
        let site_pos = site_properties.position.coordinates();
        let transformed_point = Minkowski::from([
            site_pos[0] * (body_boost.cosh()) * (body_theta.cos())
                - site_pos[1] * (body_theta.sin())
                + site_pos[3] * (body_boost.sinh()) * (body_theta.cos()),
            site_pos[0] * (body_boost.cosh()) * (body_theta.sin()) * (body_phi.cos())
                + site_pos[1] * (body_theta.cos()) * (body_phi.cos())
                - site_pos[2] * (body_phi.sin())
                + site_pos[3] * (body_boost.sinh()) * (body_theta.sin()) * (body_phi.cos()),
            site_pos[0] * (body_boost.cosh()) * (body_theta.sin()) * (body_phi.sin())
                + site_pos[1] * (body_theta.cos()) * (body_phi.sin())
                + site_pos[2] * (body_phi.cos())
                + site_pos[3] * (body_boost.sinh()) * (body_theta.sin()) * (body_phi.sin()),
            site_pos[0] * (body_boost.sinh()) + site_pos[3] * (body_boost.cosh()),
        ]);
        let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(transformed_point, skirt);
        Point::new(new_hyperbolic)
    }
}

impl Transform<Point<Spherical<3>>> for Point<Spherical<3>> {
    #[inline]
    /// Move `Point<Sphere<3>>` properties from the local body frame to the
    /// system frame.
    ///
    /// All positions on the 2-sphere are associated with some $`SO(3)`$
    /// transformation which translates the origin to that position. The local
    /// body frame is the frame in which the body position is the origin. The
    /// position of the sites in the system frame is obtained by applying the
    /// transformation associated with the body's position to the sites in the
    /// local body frame.
    fn transform(&self, site_properties: &Point<Spherical<3>>) -> Point<Spherical<3>> {
        let radius = self.position.radius();
        let body_point = self.position.coordinates();
        let body_phi = body_point[1].atan2(body_point[0]);
        let body_theta = (body_point[2] / radius).acos();
        let trial_coords = site_properties.position.coordinates();
        let transformed_point = Cartesian::from([
            trial_coords[0] * (body_theta.cos()) * (body_phi.cos())
                - trial_coords[1] * (body_phi.sin())
                + trial_coords[2] * (body_theta.sin()) * (body_phi.cos()),
            trial_coords[0] * (body_theta.cos()) * (body_phi.sin())
                + trial_coords[1] * (body_phi.cos())
                + trial_coords[2] * (body_theta.sin()) * (body_phi.sin()),
            -trial_coords[0] * (body_theta.sin()) + trial_coords[2] * (body_theta.cos()),
        ]);
        let new_sphere = Spherical::from_cartesian_coordinates(transformed_point, radius);
        Point::new(new_sphere)
    }
}

impl Transform<Point<Spherical<4>>> for Point<Spherical<4>> {
    /// Move `Point<Sphere<4>>` properties from the local body frame to the
    /// system frame.
    ///
    /// All positions on the 3-sphere are associated with some $`SO(4)`$
    /// transformation which translates the origin to that position. The local
    /// body frame is the frame in which the body position is the origin. The
    /// position of the sites in the system frame is obtained by applying the
    /// transformation associated with the body's position to the sites in the
    /// local body frame.
    #[inline]
    fn transform(&self, site_properties: &Point<Spherical<4>>) -> Point<Spherical<4>> {
        let radius = self.position.radius();
        let body_point = self.position.coordinates();
        let body_phi_1 = (body_point[2].powi(2) + body_point[1].powi(2))
            .sqrt()
            .atan2(body_point[0]);
        let body_theta = (body_point[0].powi(2) + body_point[1].powi(2) + body_point[2].powi(2))
            .sqrt()
            .atan2(body_point[3]);
        let body_phi_2 = body_point[2].atan2(body_point[1]);
        let trial_coords = site_properties.position.coordinates();
        let transformed_point = Cartesian::from([
            trial_coords[0] * (body_theta.cos()) * (body_phi_1.cos())
                - trial_coords[1] * (body_phi_1.sin())
                + trial_coords[3] * (body_theta.sin()) * (body_phi_1.cos()),
            trial_coords[0] * (body_theta.cos()) * (body_phi_1.sin()) * (body_phi_2.cos())
                + trial_coords[1] * (body_phi_1.cos()) * (body_phi_2.cos())
                - trial_coords[2] * (body_phi_2.sin())
                + trial_coords[3] * (body_theta.sin()) * (body_phi_1.sin()) * (body_phi_2.cos()),
            trial_coords[0] * (body_theta.cos()) * (body_phi_1.sin()) * (body_phi_2.sin())
                + trial_coords[1] * (body_phi_1.cos()) * (body_phi_2.sin())
                + trial_coords[2] * (body_phi_2.cos())
                + trial_coords[3] * (body_theta.sin()) * (body_phi_1.sin()) * (body_phi_2.sin()),
            -trial_coords[0] * (body_theta.sin()) + trial_coords[3] * (body_theta.cos()),
        ]);
        let new_sphere = Spherical::from_cartesian_coordinates(transformed_point, radius);
        Point::new(new_sphere)
    }
}

impl<P> Position for Point<P> {
    type Position = P;

    #[inline]
    fn position(&self) -> &P {
        &self.position
    }

    #[inline]
    fn position_mut(&mut self) -> &mut P {
        &mut self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approxim::assert_relative_eq;
    use hoomd_manifold::{Hyperbolic, Spherical};
    use hoomd_vector::Cartesian;
    use std::f64::consts::PI;

    #[test]
    fn transform_point() {
        let body = Point::new(Cartesian::from([3.0, -4.0, 5.0]));
        let site = Point::new(Cartesian::from([-1.0, 2.0, -3.0]));
        let transformed_site = body.transform(&site);
        assert_eq!(transformed_site.position, [2.0, -2.0, 2.0].into());
    }

    #[test]
    fn transform_h2_point() {
        let boost: f64 = 1.3;
        let bump: f64 = 0.1;
        let body = Point::new(Hyperbolic::<3>::from_polar_coordinates(boost, 0.0, 1.0));
        let site = Point::new(Hyperbolic::<3>::from_polar_coordinates(bump, PI / 2.0, 1.0));
        let transformed_site = body.transform(&site);
        assert_relative_eq!(
            *transformed_site.position().point(),
            [
                (boost.sinh()) * (bump.cosh()),
                bump.sinh(),
                (boost.cosh()) * (bump.cosh())
            ]
            .into(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn transform_h3_point() {
        let boost: f64 = 1.4;
        let bump: f64 = 0.7;
        let body = Point::new(Hyperbolic::<4>::from_polar_coordinates(
            boost, 0.0, 0.0, 1.0,
        ));
        let site = Point::new(Hyperbolic::<4>::from_polar_coordinates(
            bump,
            PI / 2.0,
            0.0,
            1.0,
        ));
        let transformed_site = body.transform(&site);
        assert_relative_eq!(
            *transformed_site.position().point(),
            [
                (boost.sinh()) * (bump.cosh()),
                bump.sinh(),
                0.0,
                (boost.cosh()) * (bump.cosh())
            ]
            .into(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn transform_s2_point() {
        let theta = PI / 5.0;
        let blip = PI / 10.0;
        let body = Point::new(Spherical::<3>::from_polar_coordinates(1.0, theta, 0.0));
        let site = Point::new(Spherical::<3>::from_polar_coordinates(1.0, blip, PI / 2.0));
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
    }

    #[test]
    fn transform_s3_point() {
        let theta = PI / 5.0;
        let blip = PI / 10.0;
        let body = Point::new(Spherical::<4>::from_polar_coordinates(1.0, theta, 0.0, 0.0));
        let site = Point::new(Spherical::<4>::from_polar_coordinates(
            1.0,
            blip,
            PI / 2.0,
            0.0,
        ));
        let transformed_site = body.transform(&site);
        assert_relative_eq!(
            *transformed_site.position().point(),
            [
                (theta.sin()) * (blip.cos()),
                blip.sin(),
                0.0,
                (theta.cos()) * (blip.cos())
            ]
            .into()
        );
    }
}

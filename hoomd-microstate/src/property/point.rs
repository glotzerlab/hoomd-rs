// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Point

use super::Position;
use crate::Transform;
use hoomd_manifold::{Hyperboloid, Minkowski, Sphere};
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
#[derive(Clone, Copy, Debug, Default, PartialEq)]
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

impl Transform<Point<Hyperboloid<3>>> for Point<Hyperboloid<3>> {
    #[inline]
    fn transform(&self, site_properties: &Point<Hyperboloid<3>>) -> Point<Hyperboloid<3>> {
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
        let new_hyperboloid = Hyperboloid::from_minkowski_coordinates(transformed_point, skirt);
        Point::new(new_hyperboloid)
    }
}

impl Transform<Point<Hyperboloid<4>>> for Point<Hyperboloid<4>> {
    #[inline]
    fn transform(&self, site_properties: &Point<Hyperboloid<4>>) -> Point<Hyperboloid<4>> {
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
        let new_hyperboloid = Hyperboloid::from_minkowski_coordinates(transformed_point, skirt);
        Point::new(new_hyperboloid)
    }
}

impl Transform<Point<Sphere<3>>> for Point<Sphere<3>> {
    #[inline]
    fn transform(&self, site_properties: &Point<Sphere<3>>) -> Point<Sphere<3>> {
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
        let new_sphere = Sphere::from_cartesian_coordinates(transformed_point, radius);
        Point::new(new_sphere)
    }
}

impl Transform<Point<Sphere<4>>> for Point<Sphere<4>> {
    #[inline]
    fn transform(&self, site_properties: &Point<Sphere<4>>) -> Point<Sphere<4>> {
        let radius = self.position.radius();
        let body_point = self.position.coordinates();
        let body_phi_1 = (body_point[2].powi(2) + body_point[1].powi(2))
            .sqrt()
            .atan2(body_point[0]);
        let body_theta = (body_point[0].powi(2)
            + body_point[1].powi(2)
            + body_point[2].powi(2))
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
        let new_sphere = Sphere::from_cartesian_coordinates(transformed_point, radius);
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

impl<const N: usize> Position for Hyperboloid<N> {
    type Position = Hyperboloid<N>;
    #[inline]
    fn position(&self) -> &Hyperboloid<N> {
        self
    }
    #[inline]
    fn position_mut(&mut self) -> &mut Hyperboloid<N> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use hoomd_vector::Cartesian;
    use hoomd_manifold::{Hyperboloid, Sphere};
    use std::f64::consts::PI;
    use approx::assert_relative_eq;


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
        let body = Point::new(Hyperboloid::<3>::from_polar_coordinates(boost, 0.0, 1.0));
        let site = Point::new(Hyperboloid::<3>::from_polar_coordinates(bump, PI/2.0, 1.0));
        let transformed_site = body.transform(&site);
        assert_relative_eq!(transformed_site.position().coordinates()[0], (boost.sinh())*(bump.cosh()), epsilon=1e-12);
        assert_relative_eq!(transformed_site.position().coordinates()[1], bump.sinh(), epsilon=1e-12);
        assert_relative_eq!(transformed_site.position().coordinates()[2], (boost.cosh())*(bump.cosh()), epsilon=1e-12);
    }

    #[test]
    fn transform_h3_point() {
        let boost: f64 = 1.4;
        let bump: f64 = 0.7;
        let body = Point::new(Hyperboloid::<4>::from_polar_coordinates(boost, 0.0, 0.0,1.0));
        let site = Point::new(Hyperboloid::<4>::from_polar_coordinates(bump, PI/2.0, 0.0,1.0));
        let transformed_site = body.transform(&site);
        assert_relative_eq!(transformed_site.position().coordinates()[0], (boost.sinh())*(bump.cosh()), epsilon=1e-12);
        assert_relative_eq!(transformed_site.position().coordinates()[1], bump.sinh(), epsilon=1e-12);
        assert_relative_eq!(transformed_site.position().coordinates()[2], 0.0, epsilon=1e-12);
        assert_relative_eq!(transformed_site.position().coordinates()[3], (boost.cosh())*(bump.cosh()), epsilon=1e-12);
    }

    #[test]
    fn transform_s2_point() {
        let theta = PI/5.0;
        let blip = PI/10.0;
        let body = Point::new(Sphere::<3>::from_polar_coordinates(1.0, theta, 0.0));
        let site = Point::new(Sphere::<3>::from_polar_coordinates(1.0, blip, PI/2.0));
        let transformed_site = body.transform(&site);
        assert_relative_eq!(transformed_site.position().coordinates()[0],(theta.sin())*(blip.cos()));
        assert_relative_eq!(transformed_site.position().coordinates()[1], blip.sin());
        assert_relative_eq!(transformed_site.position().coordinates()[2],(theta.cos())*(blip.cos()));
    }

    #[test]
    fn transform_s3_point() {
        let theta = PI/5.0;
        let blip = PI/10.0;
        let body = Point::new(Sphere::<4>::from_polar_coordinates(1.0, theta, 0.0, 0.0));
        let site = Point::new(Sphere::<4>::from_polar_coordinates(1.0, blip, PI/2.0, 0.0));
        let transformed_site = body.transform(&site);
        assert_relative_eq!(transformed_site.position().coordinates()[0],(theta.sin())*(blip.cos()));
        assert_relative_eq!(transformed_site.position.coordinates()[1], blip.sin());
        assert_relative_eq!(transformed_site.position().coordinates()[2], 0.0);
        assert_relative_eq!(transformed_site.position().coordinates()[3],(theta.cos())*(blip.cos()));
    }
}

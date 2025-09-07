// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Point */

use super::Position;
use crate::Transform;
use approx::assert_relative_eq;
use hoomd_manifold::{Hyperboloid, Minkowski, Sphere};
use hoomd_vector::Cartesian;

/** A position in space and nothing more.

Use [`Point`] as a [`Body`](crate::Body) or [`Site`](crate::Site) property type.

# Example

```
use hoomd_vector::Cartesian;
use hoomd_microstate::property::Point;

let point = Point::new(Cartesian::from([1.0, -2.0, 3.0]));
```
*/
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point<M> {
    /// The location of the point in space.
    pub position: M,
}

impl<M> Point<M> {
    /** Construct a new point at the given position.

    # Example

    ```
    use hoomd_vector::Cartesian;
    use hoomd_microstate::property::Point;

    let point = Point::new(Cartesian::from([1.0, -2.0, 3.0]));
    ```
    */
    #[inline]
    #[must_use]
    pub fn new(position: M) -> Self {
        Self { position }
    }
}

/** Move [`Point`] properties from the local body frame to the system frame.
*/
impl<const N: usize> Transform<Point<Cartesian<N>>> for Point<Cartesian<N>> {
    /** Points transform by vector addition.

    ```math
    \vec{r} = \vec{r}_\mathrm{body} + \vec{r}_\mathrm{site}
    ```

    ```
    use hoomd_vector::Cartesian;
    use hoomd_microstate::{property::Point, Transform};

    let body_properties = Point::new(Cartesian::from([1.0, -2.0, 3.0]));
    let site_properties = Point::new(Cartesian::from([-3.0, 2.0, 1.0]));

    let system_site = body_properties.transform(&site_properties);
    assert_eq!(system_site.position, [-2.0, 0.0, 4.0].into());
    ```
    */
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
        let body_pos = self.position.point;
        let body_theta = body_pos.coordinates[1].atan2(body_pos.coordinates[0]);
        let body_boost = (body_pos.coordinates[2] / self.position.skirt).acosh();
        let site_pos = site_properties.position.point;
        let transformed_point = Minkowski::from([
            site_pos[0] * (body_boost.cosh()) * (body_theta.cos())
                - site_pos[1] * (body_theta.sin())
                + site_pos[2] * (body_boost.sinh()) * (body_theta.cos()),
            site_pos[0] * (body_boost.cosh()) * (body_theta.sin())
                + site_pos[1] * (body_theta.cos())
                + site_pos[2] * (body_boost.sinh()) * (body_theta.sin()),
            site_pos[0] * (body_boost.sinh()) + site_pos[2] * (body_boost.cosh()),
        ]);
        let new_hyperboloid = Hyperboloid::from(&transformed_point);
        #[cfg(debug_assertions)]
        {
            assert_relative_eq!(
                self.position.skirt,
                new_hyperboloid.skirt(),
                epsilon = 1e-12
            );
        }
        Point::new(new_hyperboloid)
    }
}

impl Transform<Point<Hyperboloid<4>>> for Point<Hyperboloid<4>> {
    #[inline]
    fn transform(&self, site_properties: &Point<Hyperboloid<4>>) -> Point<Hyperboloid<4>> {
        let body_point = self.position.point;
        let body_theta = (body_point.coordinates[2].powi(2) + body_point.coordinates[1].powi(2))
            .sqrt()
            .atan2(body_point.coordinates[0]);
        let body_phi = body_point.coordinates[2].atan2(body_point.coordinates[1]);
        let body_boost = (body_point.coordinates[2] / self.position.skirt).acosh();
        let site_pos = site_properties.position.point;
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
        let new_hyperboloid = Hyperboloid::from(&transformed_point);
        #[cfg(debug_assertions)]
        {
            assert_relative_eq!(
                self.position.skirt,
                new_hyperboloid.skirt(),
                epsilon = 1e-12
            );
        }
        Point::new(new_hyperboloid)
    }
}

impl Transform<Point<Sphere<3>>> for Point<Sphere<3>> {
    #[inline]
    fn transform(&self, site_properties: &Point<Sphere<3>>) -> Point<Sphere<3>> {
        let radius = self.position.radius;
        let body_point = self.position.point;
        let body_phi = body_point.coordinates[1].atan2(body_point.coordinates[0]);
        let body_theta = (body_point.coordinates[2] / radius).acos();
        let trial_coords = site_properties.position.point.coordinates;
        let transformed_point = Cartesian::from([
            trial_coords[0] * (body_theta.cos()) * (body_phi.cos())
                - trial_coords[1] * (body_phi.sin())
                + trial_coords[2] * (body_theta.sin()) * (body_phi.cos()),
            trial_coords[0] * (body_theta.cos()) * (body_phi.sin())
                + trial_coords[1] * (body_phi.cos())
                + trial_coords[2] * (body_theta.sin()) * (body_phi.sin()),
            -trial_coords[0] * (body_theta.sin()) + trial_coords[2] * (body_theta.cos()),
        ]);
        let new_sphere = Sphere::from(&transformed_point);
        #[cfg(debug_assertions)]
        {
            assert_relative_eq!(radius, new_sphere.radius, epsilon = 1e-12);
        }
        Point::new(new_sphere)
    }
}

impl Transform<Point<Sphere<4>>> for Point<Sphere<4>> {
    #[inline]
    fn transform(&self, site_properties: &Point<Sphere<4>>) -> Point<Sphere<4>> {
        let radius = self.position.radius;
        let body_point = self.position.point;
        let body_phi_1 = (body_point.coordinates[2].powi(2) + body_point.coordinates[1].powi(2))
            .sqrt()
            .atan2(body_point.coordinates[0]);
        let body_theta = (body_point.coordinates[0].powi(2)
            + body_point.coordinates[1].powi(2)
            + body_point.coordinates[2].powi(2))
        .sqrt()
        .atan2(body_point.coordinates[3]);
        let body_phi_2 = body_point.coordinates[2].atan2(body_point.coordinates[1]);
        let trial_coords = site_properties.position.point.coordinates;
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
        let new_sphere = Sphere::from(&transformed_point);
        #[cfg(debug_assertions)]
        {
            assert_relative_eq!(radius, new_sphere.radius, epsilon = 1e-12);
        }
        Point::new(new_sphere)
    }
}

impl<M> Position for Point<M> {
    type Metric = M;

    #[inline]
    fn position(&self) -> &M {
        &self.position
    }

    #[inline]
    fn position_mut(&mut self) -> &mut M {
        &mut self.position
    }
}

impl<const N: usize> Position for Hyperboloid<N> {
    type Metric = Hyperboloid<N>;
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

    #[test]
    fn transform_point() {
        let body = Point::new(Cartesian::from([3.0, -4.0, 5.0]));
        let site = Point::new(Cartesian::from([-1.0, 2.0, -3.0]));
        let transformed_site = body.transform(&site);
        assert_eq!(transformed_site.position, [2.0, -2.0, 2.0].into());
    }
}

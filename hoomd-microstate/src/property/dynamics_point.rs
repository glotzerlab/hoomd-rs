// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement DynamicsPoint

use super::{Position, Velocity, Acceleration, Mass};
use super::point::Point;
use super::oriented_point::OrientedPoint;
use crate::Transform;
use hoomd_vector::Vector;

/// The position, mass, velocity, and acceleration of an extended body, such as
/// is useful for Molecular Dynamics simulations.
/// 
/// Use [`DynamicsPoint`] as a [`Body`](crate::Body) or [`Site`](crate::Site) property type.
/// 
/// # Example
/// 
/// ```
/// use hoomd_microstate::property::DynamicsPoint;
/// use hoomd_vector::Cartesian;
/// 
/// let dynamics_point = DynamicsPoint {
///     position: Cartesian::from([1.0, -3.0]),
///     mass: 1.0,
///     velocity: Cartesian::from([0.0, 1.0]),
///     acceleration: Cartesian::from([1.0, 0.0])
/// };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DynamicsPoint<V> {
    /// The location of the extended body in space.
    pub position: V,

    /// The velocity of the extended body in space.
    pub velocity: V,

    /// The acceleration of the extended body in space.
    pub acceleration: V,

    /// The mass of the extended body.
    pub mass: f64,
}

/// Move [`DynamicsPoint`] properties from the local body frame to the system frame.
impl<V> Transform<Point<V>> for DynamicsPoint<V>
where
    V: Vector,
{
    /// [`DynamicsPoint`] transforms [`Point`] by vector addition.
    /// 
    /// ```math
    /// \vec{r} = \vec{r}_\mathrm{body} + \vec{r}_\mathrm{site}
    /// ```
    /// 
    /// ```
    /// use hoomd_vector::Cartesian;
    /// use hoomd_microstate::{property::DynamicsPoint, Transform};
    /// 
    /// let body_properties = DynamicsPoint {
    ///     position: Cartesian::from([1.0, -2.0, 3.0]),
    ///     mass: 1.0,
    ///     velocity: Cartesian::from([0.0, 1.0, 1.0]),
    ///     acceleration: Cartesian::from([1.0, 0.0, 1.0])
    /// };
    /// let site_properties = Point::new(Cartesian::from([-3.0, 2.0, 1.0]));
    /// 
    /// let system_site = body_properties.transform(&site_properties);
    /// assert_relative_eq!(system_site.position, [-2.0, 0.0, 4.0].into());
    /// ```
    #[inline]
    fn transform(&self, site_properties: &Point<V>) -> Point<V> {
        Point { position: self.position + site_properties.position }
    }
}

impl<V, R> Transform<OrientedPoint<V, R>> for DynamicsPoint<V>
where
    V: Vector,
    R: Copy
{
    /// [`DynamicsPoint`] transforms [`OrientedPoint`] by vector addition.
    /// 
    /// ```math
    /// \vec{r} = \vec{r}_\mathrm{body} + \vec{r}_\mathrm{site}
    /// ```
    /// 
    /// ```
    /// use hoomd_vector::Cartesian;
    /// use hoomd_microstate::{property::DynamicsPoint, Transform};
    /// 
    /// let body_properties = DynamicsPoint {
    ///     position: Cartesian::from([1.0, -2.0, 3.0]),
    ///     mass: 1.0,
    ///     velocity: Cartesian::from([0.0, 1.0, 1.0]),
    ///     acceleration: Cartesian::from([1.0, 0.0, 1.0])
    /// };
    /// let site_properties = OrientedPoint {
    ///     position: Cartesian::from([-3.0, 2.0, 1.0]),
    ///     orientation: Angle::from(PI/2.0),
    /// };
    ///
    /// let system_site = body_properties.transform(&site_properties);
    /// assert_relative_eq!(system_site.position, [-2.0, 0.0, 4.0].into());
    /// ```
    #[inline]
    fn transform(&self, site_properties: &OrientedPoint<V, R>) -> OrientedPoint<V, R> {
        OrientedPoint {
            position: self.position + site_properties.position,
            ..*site_properties
        }
    }
}

impl<V> Position for DynamicsPoint<V> {
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

impl<V> Velocity for DynamicsPoint<V> {
    type Vector = V;

    #[inline]
    fn velocity(&self) -> &V {
        &self.velocity
    }

    #[inline]
    fn velocity_mut(&mut self) -> &mut V {
        &mut self.velocity
    }
}

impl<V> Acceleration for DynamicsPoint<V> {
    type Vector = V;

    #[inline]
    fn acceleration(&self) -> &V {
        &self.acceleration
    }

    #[inline]
    fn acceleration_mut(&mut self) -> &mut V {
        &mut self.acceleration
    }
}

impl<V> Mass for DynamicsPoint<V> {

    #[inline]
    fn mass(&self) -> &f64 {
        &self.mass
    }
}

// TODO: tests.

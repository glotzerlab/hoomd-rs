// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement DynamicPoint

use super::oriented_point::OrientedPoint;
use super::point::Point;
use super::{Mass, Momentum, Position};
use crate::Transform;
use crate::property::NetForce;
use hoomd_vector::Vector;

/// The position, mass, momentum, and net force of an extended body, such as is
/// useful for Molecular Dynamics simulations.
///
/// Use [`DynamicPoint`] as a [`Body`](crate::Body) property type.
///
/// # Example
///
/// ```
/// use hoomd_microstate::property::DynamicPoint;
/// use hoomd_vector::Cartesian;
///
/// let dynamics_point = DynamicPoint {
///     position: Cartesian::from([1.0, -3.0]),
///     mass: 1.0,
///     momentum: Cartesian::from([0.0, 1.0]),
///     net_force: Cartesian::from([0.0, 0.0]),
/// };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DynamicPoint<V> {
    /// The location of the extended body in space.
    pub position: V,

    /// The mass of the extended body.
    pub mass: f64,

    /// The momentum of the extended body in space.
    pub momentum: V,

    /// The net force of the extended body in space.
    pub net_force: V,
}

/// Move [`DynamicPoint`] properties from the local body frame to the system frame.
impl<V> Transform<Point<V>> for DynamicPoint<V>
where
    V: Vector,
{
    /// [`DynamicPoint`] transforms [`Point`] by vector addition.
    ///
    /// ```math
    /// \vec{r} = \vec{r}_\mathrm{body} + \vec{r}_\mathrm{site}
    /// ```
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::Cartesian;
    /// use hoomd_microstate::{property::{DynamicPoint, Point}, Transform};
    ///
    /// let body_properties = DynamicPoint {
    ///     position: Cartesian::from([1.0, -2.0, 3.0]),
    ///     mass: 1.0,
    ///     momentum: Cartesian::<3>::default(),
    ///     net_force: Cartesian::<3>::default(),
    /// };
    /// let site_properties = Point::new(Cartesian::from([-3.0, 2.0, 1.0]));
    ///
    /// let system_site = body_properties.transform(&site_properties);
    /// assert_relative_eq!(system_site.position, [-2.0, 0.0, 4.0].into());
    /// ```
    #[inline]
    fn transform(&self, site_properties: &Point<V>) -> Point<V> {
        Point {
            position: self.position + site_properties.position,
        }
    }
}

impl<V, R> Transform<OrientedPoint<V, R>> for DynamicPoint<V>
where
    V: Vector,
    R: Copy,
{
    /// [`DynamicPoint`] transforms [`OrientedPoint`] by vector addition.
    ///
    /// ```math
    /// \vec{r} = \vec{r}_\mathrm{body} + \vec{r}_\mathrm{site}
    /// ```
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::{Cartesian, Versor};
    /// use hoomd_microstate::{property::{DynamicPoint, OrientedPoint}, Transform};
    ///
    /// let body_properties = DynamicPoint {
    ///     position: Cartesian::from([1.0, -2.0, 3.0]),
    ///     mass: 1.0,
    ///     momentum: Cartesian::<3>::default(),
    ///     net_force: Cartesian::<3>::default(),
    /// };
    /// let site_properties = OrientedPoint {
    ///     position: Cartesian::from([-3.0, 2.0, 1.0]),
    ///     orientation: Versor::default(),
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

impl<P> Position for DynamicPoint<P> {
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

impl<V> Momentum for DynamicPoint<V>
where
    V: std::ops::Mul<f64, Output = V> + std::ops::Div<f64, Output = V> + Copy,
{
    type Momentum = V;

    #[inline]
    fn momentum(&self) -> &V {
        &self.momentum
    }

    #[inline]
    fn momentum_mut(&mut self) -> &mut V {
        &mut self.momentum
    }

    #[inline]
    fn velocity(&self) -> Self::Momentum {
        self.momentum / self.mass()
    }

    #[inline]
    fn set_velocity(&mut self, velocity: Self::Momentum) {
        *self.momentum_mut() = velocity * self.mass();
    }
}

impl<V> Mass for DynamicPoint<V> {
    #[inline]
    fn mass(&self) -> f64 {
        self.mass
    }
}

impl<V> NetForce for DynamicPoint<V> {
    type NetForce = V;

    #[inline]
    fn net_force(&self) -> &Self::NetForce {
        &self.net_force
    }

    #[inline]
    fn net_force_mut(&mut self) -> &mut Self::NetForce {
        &mut self.net_force
    }
}

// TODO: tests.

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement OrientedDynamicPoint

use super::oriented_point::OrientedPoint;
use super::point::Point;
use super::{AngularMomentum, Mass, MomentOfInertia, Momentum, NetForce, NetTorque, Position};
use crate::Transform;
use crate::property::Orientation;
use hoomd_vector::{Rotate, Rotation, Vector, Wedge};

/// The position, orientation, mass, velocity, acceleration, moment of inertia,
/// and angular velocity of an extended body, such as  is useful for Molecular
/// Dynamics simulations.
///
/// Use [`OrientedDynamicPoint`] as a [`Body`](crate::Body) or [`Site`](crate::Site) property type.
///
/// # Example
///
/// ```
/// use hoomd_microstate::property::OrientedDynamicPoint;
/// use hoomd_vector::{Cartesian, Angle};
///
/// let oriented_dynamics_point = OrientedDynamicPoint {
///     position: Cartesian::from([1.0, -3.0]),
///     orientation: Angle::default(),
///     mass: 1.0,
///     momentum: Cartesian::<2>::default(),
///     net_force: Cartesian::<2>::default(),
///     moment_of_inertia: 1.0,
///     angular_momentum: 0.0,
///     net_torque: 0.0
/// };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrientedDynamicPoint<V: Wedge, R> {
    /// The location of the extended body in space.
    pub position: V,

    ///Rotate from the body's reference frame into another frame.
    pub orientation: R,

    /// The mass of the extended body.
    pub mass: f64,

    /// The momentum of the extended body in space.
    pub momentum: V,

    /// The net force of the extended body in space.
    pub net_force: V,

    /// The moment of inertia of the extended body.
    pub moment_of_inertia: V::Bivector, // TODO: this is strictly speaking wrong

    /// The angular velocity of the extended body.
    pub angular_momentum: V::Bivector, // TODO: convert to Quat in integrator

    /// The torque velocity of the extended body.
    pub net_torque: V::Bivector,
}

/// Treat [`Point`] sites as constituents of oriented rigid bodies.
impl<V, R> Transform<Point<V>> for OrientedDynamicPoint<V, R>
where
    V: Vector + Wedge,
    R: Rotate<V>,
{
    /// Move [`Point`] properties from the local body frame to the system frame.
    ///
    /// ```math
    /// \vec{r} = \vec{r}_\mathrm{body} + R_\mathrm{body}(\vec{r}_\mathrm{site})
    /// ```
    ///
    /// TODO: Add example.
    #[inline]
    fn transform(&self, site_properties: &Point<V>) -> Point<V> {
        Point {
            position: self.position + self.orientation.rotate(&site_properties.position),
        }
    }
}

/// Treat [`OrientedPoint`] sites as constituents of oriented rigid bodies.
impl<V, R> Transform<OrientedPoint<V, R>> for OrientedDynamicPoint<V, R>
where
    V: Vector + Wedge,
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
    /// TODO: add example.
    #[inline]
    fn transform(&self, site_properties: &OrientedPoint<V, R>) -> OrientedPoint<V, R> {
        OrientedPoint {
            position: self.position + self.orientation.rotate(&site_properties.position),
            orientation: self.orientation.combine(&site_properties.orientation),
        }
    }
}

impl<V, R> Position for OrientedDynamicPoint<V, R>
where
    V: Wedge,
{
    // TODO: bring the associated type name into alignment with convention used elsewhere
    type Position = V;

    #[inline]
    fn position(&self) -> &V {
        &self.position
    }

    #[inline]
    fn position_mut(&mut self) -> &mut V {
        &mut self.position
    }
}

impl<V, R> Orientation for OrientedDynamicPoint<V, R>
where
    V: Wedge,
{
    type Rotation = R;

    #[inline]
    fn orientation(&self) -> &Self::Rotation {
        &self.orientation
    }

    #[inline]
    fn orientation_mut(&mut self) -> &mut Self::Rotation {
        &mut self.orientation
    }
}

impl<V, R> Momentum for OrientedDynamicPoint<V, R>
where
    V: std::ops::Mul<f64, Output = V> + std::ops::Div<f64, Output = V> + Copy + Wedge,
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

impl<V, R> Mass for OrientedDynamicPoint<V, R>
where
    V: Wedge,
{
    #[inline]
    fn mass(&self) -> f64 {
        self.mass
    }
}

impl<V, R> NetForce for OrientedDynamicPoint<V, R>
where
    V: Wedge,
{
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

impl<V, R> MomentOfInertia for OrientedDynamicPoint<V, R>
where
    V: Wedge,
{
    type MomentOfInertia = V::Bivector;

    #[inline]
    fn moment_of_inertia(&self) -> &V::Bivector {
        &self.moment_of_inertia
    }

    #[inline]
    fn moment_of_inertia_mut(&mut self) -> &mut V::Bivector {
        &mut self.moment_of_inertia
    }
}

impl<V, R> AngularMomentum for OrientedDynamicPoint<V, R>
where
    V: Wedge,
{
    type AngularMomentum = V::Bivector;

    #[inline]
    fn angular_momentum(&self) -> &V::Bivector {
        &self.angular_momentum
    }

    #[inline]
    fn angular_momentum_mut(&mut self) -> &mut V::Bivector {
        &mut self.angular_momentum
    }
}

impl<V, R> NetTorque for OrientedDynamicPoint<V, R>
where
    V: Wedge,
{
    type NetTorque = V::Bivector;

    #[inline]
    fn net_torque(&self) -> &V::Bivector {
        &self.net_torque
    }

    #[inline]
    fn net_torque_mut(&mut self) -> &mut V::Bivector {
        &mut self.net_torque
    }
}

// impl OrientedDynamicPoint<Cartesian<3>, Quaternion>
// {
//     /// Transform the three-dimensional
//     /// angular momentum as a quaternion of body to
//     /// angular velocity in vector form.
//     pub fn angular_velocity(&self) -> Cartesian<3> {
//         // transform angmom to vector form (angmom_vec.scalar should be 0.0)
//         let angmom_vec = (self.orientation.conjugate() * self.angular_momentum) * 0.5;
//         Cartesian::from([
//             angmom_vec.vector[0] / self.moment_of_inertia[0],
//             angmom_vec.vector[1] / self.moment_of_inertia[1],
//             angmom_vec.vector[2] / self.moment_of_inertia[2]
//         ])
//     }
// }

// impl OrientedDynamicPoint<f64, f64>
// {
//     /// Transform the two-dimensional
//     /// angular momentum to angular velocity.
//     pub fn angular_velocity(&self) -> f64 {
//         self.angular_momentum / self.moment_of_inertia
//     }
// }

// TODO: tests.

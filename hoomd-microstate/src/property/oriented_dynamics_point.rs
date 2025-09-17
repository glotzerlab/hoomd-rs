// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement OrientedDynamicsPoint */

use super::{Position, Velocity, Acceleration, Mass, MomentOfInertia, AngularMomentum, Torque};
use super::point::Point;
use super::oriented_point::OrientedPoint;
use crate::Transform;
use hoomd_vector::{Rotate, Rotation, Vector};

/** The position, orientation, mass, velocity, acceleration, moment of inertia,
and angular velocity of an extended body, such as  is useful for Molecular
Dynamics simulations.

Use [`OrientedDynamicsPoint`] as a [`Body`](crate::Body) or [`Site`](crate::Site) property type.

# Example

```
use hoomd_microstate::property::OrientedDynamicsPoint;
use hoomd_vector::Cartesian;

let oriented_dynamics_point = OrientedDynamicsPoint {
    position: Cartesian::from([1.0, -3.0]),
    orientation: 
    mass: 1.0,
    velocity: Cartesian::from([0.0, 1.0]),
    acceleration: Cartesian::from([1.0, 0.0]),
    moment_of_inertia: Cartesian::from([1.0, 1.0]),
    angular_velocity: Cartesian::from([1.0, 1.0])
};
```
*/
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrientedDynamicsPoint<V, R> {
    /// The location of the extended body in space.
    pub position: V,

    ///Rotate from the body's reference frame into another frame.
    pub orientation: R,

    /// The velocity of the extended body in space.
    pub velocity: V,

    /// The acceleration of the extended body in space.
    pub acceleration: V,

    /// The mass of the extended body.
    pub mass: f64,

    /// The moment of inertia of the extended body.
    pub moment_of_inertia: V,

    /// The angular velocity of the extended body.
    pub angular_momentum: R,

    /// The torque velocity of the extended body. 
    pub torque: V
}

/** Treat [`Point`] sites as constituents of oriented rigid bodies.
*/
impl<V, R> Transform<Point<V>> for OrientedDynamicsPoint<V, R>
where
    V: Vector,
    R: Rotate<V>,
{
    /** Move [`Point`] properties from the local body frame to the system frame.

    ```math
    \vec{r} = \vec{r}_\mathrm{body} + R_\mathrm{body}(\vec{r}_\mathrm{site})
    ```

    TODO: Add example.
    */
    #[inline]
    fn transform(&self, site_properties: &Point<V>) -> Point<V> {
        Point {
            position: self.position + self.orientation.rotate(&site_properties.position),
        }
    }
}

/** Treat [`OrientedPoint`] sites as constituents of oriented rigid bodies.
*/
impl<V, R> Transform<OrientedPoint<V, R>> for OrientedDynamicsPoint<V, R>
where
    V: Vector,
    R: Rotate<V> + Rotation,
{
    /** Move [`Point`] properties from the local body frame to the system frame.

    ```math
    \vec{r} = \vec{r}_\mathrm{body} + R_\mathrm{body}(\vec{r}_\mathrm{site})
    ```
    ```math
    R = R_\mathrm{body}(R_\mathrm{site})
    ```

    TODO: add example.
    */
    #[inline]
    fn transform(&self, site_properties: &OrientedPoint<V, R>) -> OrientedPoint<V, R> {
        OrientedPoint {
            position: self.position + self.orientation.rotate(&site_properties.position),
            orientation: self.orientation.combine(&site_properties.orientation),
        }
    }
}

impl<V, R> Position for OrientedDynamicsPoint<V, R> {
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

impl<V, R> Velocity for OrientedDynamicsPoint<V, R> {
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

impl<V, R> Acceleration for OrientedDynamicsPoint<V, R> {
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

impl<V, R> Mass for OrientedDynamicsPoint<V, R> {

    #[inline]
    fn mass(&self) -> &f64 {
        &self.mass
    }
}

impl<V, R> MomentOfInertia for OrientedDynamicsPoint<V, R> {
    type Vector = V;

    #[inline]
    fn moment_of_inertia(&self) -> &V {
        &self.moment_of_inertia
    }

    #[inline]
    fn moment_of_inertia_mut(&mut self) -> &mut V {
        &mut self.moment_of_inertia
    }
}

impl<V, R> AngularMomentum for OrientedDynamicsPoint<V, R> {
    type Rotation = R;

    #[inline]
    fn angular_momentum(&self) -> &R {
        &self.angular_momentum
    }

    #[inline]
    fn angular_momentum_mut(&mut self) -> &mut R {
        &mut self.angular_momentum
    }
}

impl<V, R> Torque for OrientedDynamicsPoint<V, R> {
    type Vector = V;

    #[inline]
    fn torque(&self) -> &V {
        &self.torque
    }

    #[inline]
    fn torque_mut(&mut self) -> &mut V {
        &mut self.torque
    }
}


// TODO: tests.

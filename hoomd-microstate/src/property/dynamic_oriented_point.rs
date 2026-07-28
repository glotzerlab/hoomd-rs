// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `DynamicOrientedPoint`

use serde::{Deserialize, Serialize};

use super::{
    AngularMomentum, Mass, MomentOfInertia, Momentum, NetForce, NetTorque, Orientation,
    OrientedPoint, Point, Position, RotationalMotionTypes,
};
use crate::{Transform, property::{Drag, NetVirial, RotationalDrag}};
use hoomd_vector::{Angle, Cartesian, Outer, Rotate, Rotation, Vector, Versor, Wedge};

impl RotationalMotionTypes for Angle {
    type MomentOfInertia = f64;
    type AngularMomentum = f64;
    type RotationalDrag = f64;
}

impl RotationalMotionTypes for Versor {
    type MomentOfInertia = [f64; 3];
    type AngularMomentum = Cartesian<3>;
    type RotationalDrag = [f64; 3];
}

/// A position in space with the properties necessary for translational and
/// rotational motion in MD.
///
/// Use [`DynamicOrientedPoint`] as a [`Body`](crate::Body) property type.
///
/// A default [`DynamicOrientedPoint`] has a mass of 1.0 and position, momentum,
/// and net force of $` \vec{0} `$, and a zero-tensor for net virial.
/// Orientation defaults to the identity. `DynamicOrientedPoint<_, Angle>` has a
/// default moment of inertia of 1.0. `DynamicOrientedPoint<_, Versor>` has a
/// default moment of inertia of `[1.0, 1.0, 1.0]`.
///
/// # Example
///
/// ```
/// use hoomd_microstate::property::DynamicOrientedPoint;
/// use hoomd_vector::{Angle, Cartesian};
/// use std::f64::consts::PI;
///
/// let oriented_dynamic_point = DynamicOrientedPoint {
///     position: Cartesian::from([1.0, -3.0]),
///     orientation: Angle::from(PI / 4.0),
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicOrientedPoint<V, R>
where
    V: Wedge + Outer,
    R: RotationalMotionTypes,
{
    /// The location of the extended body in space $`[\mathrm{length}]`$.
    pub position: V,

    /// Rotate from the body's reference frame to the system frame.
    pub orientation: R,

    /// The mass of the extended body $` [\mathrm{mass}] `$.
    pub mass: f64,

    /// The translational momentum of the extended body $`[ \mathrm{energy}^{1/2} \cdot \mathrm{mass}^{1/2}]`$.
    pub momentum: V,

    /// The net force applied to the body in a [`Microstate`](crate::Microstate) $`[ \mathrm{energy}^{1/2} \cdot \mathrm{mass}^{1/2}]`$.
    pub net_force: V,

    /// The net virial applied to the body in a [`Microstate`](crate::Microstate) $`[\mathrm{energy}]`$.
    pub net_virial: V::Tensor,

    /// The moment of inertia of the extended body $` [\mathrm{mass} \cdot \mathrm{length}^2] `$.
    pub moment_of_inertia: R::MomentOfInertia,

    /// The angular momentum of the extended body $` [\mathrm{mass} \cdot \mathrm{length}^2] `$.
    pub angular_momentum: R::AngularMomentum,

    /// The net torque applied to the body by others in a [`Microstate`](crate::Microstate) $` [\mathrm{energy}] `$.
    pub net_torque: V::Bivector,

    /// The translational drag coefficient.
    #[serde(default)]
    pub drag: f64,
    
    /// The rotational drag coefficient(s).
    #[serde(default)]
    pub rotational_drag: R::RotationalDrag,
}

impl<V> Default for DynamicOrientedPoint<V, Angle>
where
    V: Default + Wedge + Outer,
    V::Tensor: Default,
    V::Bivector: Default,
{
    /// Construct a [`DynamicOrientedPoint`] with mass 1.0 and moment of inertia 1.0.
    /// Position, orientation, momentum, angular momentum, net force, net virial, and net torque are set to 0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix};
    /// use hoomd_microstate::property::DynamicOrientedPoint;
    /// use hoomd_vector::{Angle, Cartesian};
    ///
    /// let dynamic_point = DynamicOrientedPoint::<Cartesian<2>, Angle>::default();
    /// assert_eq!(dynamic_point.mass, 1.0);
    /// assert_eq!(dynamic_point.moment_of_inertia, 1.0);
    /// assert_eq!(dynamic_point.position, [0.0, 0.0].into());
    /// assert_eq!(dynamic_point.orientation, 0.0.into());
    /// assert_eq!(dynamic_point.momentum, [0.0, 0.0].into());
    /// assert_eq!(dynamic_point.angular_momentum, 0.0.into());
    /// assert_eq!(dynamic_point.net_force, [0.0, 0.0].into());
    /// assert_eq!(dynamic_point.net_virial, Matrix::zeros());
    /// assert_eq!(dynamic_point.net_torque, 0.0);
    /// assert_eq!(dynamic_point.drag, 1.0);
    /// assert_eq!(dynamic_point.rotational_drag, 1.0);
    /// ```
    #[inline]
    fn default() -> Self {
        Self {
            position: Default::default(),
            orientation: Angle::default(),
            mass: 1.0,
            moment_of_inertia: 1.0,
            momentum: Default::default(),
            angular_momentum: Default::default(),
            net_force: Default::default(),
            net_virial: V::Tensor::default(),
            net_torque: Default::default(),
            drag: 1.0,
            rotational_drag: 1.0,
        }
    }
}

impl<V> Default for DynamicOrientedPoint<V, Versor>
where
    V: Default + Wedge + Outer,
    V::Bivector: Default,
{
    /// Construct a [`DynamicOrientedPoint`] with mass 1.0 and moment of inertia 1.0 on all axes.
    /// Position, momentum, angular momentum, net force, and net torque are set to 0.
    /// Orientation is set to the identity versor.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::property::DynamicOrientedPoint;
    /// use hoomd_vector::{Cartesian, Versor};
    ///
    /// let dynamic_point = DynamicOrientedPoint::<Cartesian<3>, Versor>::default();
    /// assert_eq!(dynamic_point.mass, 1.0);
    /// assert_eq!(dynamic_point.moment_of_inertia, [1.0, 1.0, 1.0]);
    /// assert_eq!(dynamic_point.position, [0.0, 0.0, 0.0].into());
    /// assert_eq!(dynamic_point.orientation, Versor::default());
    /// assert_eq!(dynamic_point.momentum, [0.0, 0.0, 0.0].into());
    /// assert_eq!(dynamic_point.angular_momentum, [0.0, 0.0, 0.0].into());
    /// assert_eq!(dynamic_point.net_force, [0.0, 0.0, 0.0].into());
    /// assert_eq!(dynamic_point.net_torque, [0.0, 0.0, 0.0].into());
    /// assert_eq!(dynamic_point.drag, 1.0);
    /// assert_eq!(dynamic_point.rotational_drag, [1.0; 3]);
    /// ```
    #[inline]
    fn default() -> Self {
        Self {
            position: Default::default(),
            orientation: Versor::default(),
            mass: 1.0,
            moment_of_inertia: [1.0, 1.0, 1.0],
            momentum: Default::default(),
            angular_momentum: Cartesian::default(),
            net_force: Default::default(),
            net_virial: V::default().outer(&V::default()),
            net_torque: Default::default(),
            drag: 1.0,
            rotational_drag: [1.0; 3],
        }
    }
}

impl<V, R> Transform<Point<V>> for DynamicOrientedPoint<V, R>
where
    V: Vector + Wedge + Outer,
    R: Rotate<V> + RotationalMotionTypes,
{
    /// Move [`Point`] properties from the local body frame to the system frame.
    ///
    /// ```math
    /// \vec{r} = \vec{r}_\mathrm{body} + R_\mathrm{body}(\vec{r}_\mathrm{site})
    /// ```
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_microstate::{
    ///     Transform,
    ///     property::{DynamicOrientedPoint, Point},
    /// };
    /// use hoomd_vector::{Angle, Cartesian};
    /// use std::f64::consts::PI;
    ///
    /// let body_properties = DynamicOrientedPoint {
    ///     position: Cartesian::from([1.0, -2.0]),
    ///     orientation: Angle::from(PI / 2.0),
    ///     ..Default::default()
    /// };
    /// let site_properties = Point::new(Cartesian::from([-1.0, 0.0]));
    ///
    /// let system_site = body_properties.transform(&site_properties);
    /// assert_relative_eq!(system_site.position, [1.0, -3.0].into());
    /// ```
    #[inline]
    fn transform(&self, site_properties: &Point<V>) -> Point<V> {
        Point {
            position: self.position + self.orientation.rotate(&site_properties.position),
        }
    }
}

impl<V, R> Transform<OrientedPoint<V, R>> for DynamicOrientedPoint<V, R>
where
    V: Vector + Wedge + Outer,
    R: Rotate<V> + Rotation + RotationalMotionTypes,
{
    /// Move [`OrientedPoint`] site properties from the local body frame to the system frame.
    ///
    /// ```math
    /// \vec{r} = \vec{r}_\mathrm{body} + R_\mathrm{body}(\vec{r}_\mathrm{site})
    /// ```
    /// ```math
    /// R = R_\mathrm{body}(R_\mathrm{site})
    /// ```
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_microstate::{
    ///     Transform,
    ///     property::{DynamicOrientedPoint, OrientedPoint},
    /// };
    /// use hoomd_vector::{Angle, Cartesian};
    /// use std::f64::consts::PI;
    ///
    /// let body_properties = DynamicOrientedPoint {
    ///     position: Cartesian::from([1.0, -2.0]),
    ///     orientation: Angle::from(PI / 2.0),
    ///     ..Default::default()
    /// };
    /// let site_properties = OrientedPoint {
    ///     position: Cartesian::from([-1.0, 0.0]),
    ///     orientation: Angle::from(PI / 4.0),
    /// };
    ///
    /// let system_site = body_properties.transform(&site_properties);
    /// assert_relative_eq!(system_site.position, [1.0, -3.0].into());
    /// assert_relative_eq!(system_site.orientation.theta, 3.0 * PI / 4.0);
    /// ```
    #[inline]
    fn transform(&self, site_properties: &OrientedPoint<V, R>) -> OrientedPoint<V, R> {
        OrientedPoint {
            position: self.position + self.orientation.rotate(&site_properties.position),
            orientation: self.orientation.combine(&site_properties.orientation),
        }
    }
}

impl<V, R> Position for DynamicOrientedPoint<V, R>
where
    V: Wedge + Outer,
    R: RotationalMotionTypes,
{
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

impl<V, R> Orientation for DynamicOrientedPoint<V, R>
where
    V: Wedge + Outer,
    R: RotationalMotionTypes,
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

impl<V, R> Momentum for DynamicOrientedPoint<V, R>
where
    V: std::ops::Mul<f64, Output = V> + std::ops::Div<f64, Output = V> + Copy + Wedge + Outer,
    R: RotationalMotionTypes,
{
    type Momentum = V;

    #[inline]
    fn momentum(&self) -> &Self::Momentum {
        &self.momentum
    }

    #[inline]
    fn momentum_mut(&mut self) -> &mut Self::Momentum {
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

impl<V, R> Mass for DynamicOrientedPoint<V, R>
where
    V: Wedge + Outer,
    R: RotationalMotionTypes,
{
    #[inline]
    fn mass(&self) -> f64 {
        self.mass
    }
}

impl<V, R> NetForce for DynamicOrientedPoint<V, R>
where
    V: Wedge + Outer,
    R: RotationalMotionTypes,
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

impl<V, R> NetVirial for DynamicOrientedPoint<V, R>
where
    V: Wedge + Outer,
    R: RotationalMotionTypes,
{
    type NetVirial = V::Tensor;

    #[inline]
    fn net_virial(&self) -> &Self::NetVirial {
        &self.net_virial
    }

    #[inline]
    fn net_virial_mut(&mut self) -> &mut Self::NetVirial {
        &mut self.net_virial
    }
}

impl<V, R> MomentOfInertia for DynamicOrientedPoint<V, R>
where
    V: Wedge + Outer,
    R: RotationalMotionTypes,
{
    type MomentOfInertia = R::MomentOfInertia;

    #[inline]
    fn moment_of_inertia(&self) -> &R::MomentOfInertia {
        &self.moment_of_inertia
    }

    #[inline]
    fn moment_of_inertia_mut(&mut self) -> &mut R::MomentOfInertia {
        &mut self.moment_of_inertia
    }
}

impl<V, R> AngularMomentum for DynamicOrientedPoint<V, R>
where
    V: Wedge + Outer,
    R: RotationalMotionTypes,
{
    type AngularMomentum = R::AngularMomentum;

    #[inline]
    fn angular_momentum(&self) -> &R::AngularMomentum {
        &self.angular_momentum
    }

    #[inline]
    fn angular_momentum_mut(&mut self) -> &mut R::AngularMomentum {
        &mut self.angular_momentum
    }
}

impl<V, R> NetTorque for DynamicOrientedPoint<V, R>
where
    V: Wedge + Outer,
    R: RotationalMotionTypes,
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

impl<V, R> Drag for DynamicOrientedPoint<V, R>
where
    V: Wedge + Outer,
    R: RotationalMotionTypes,
{
    fn drag(&self) -> &f64 {
        &self.drag
    }

    fn drag_mut(&mut self) -> &mut f64 {
        &mut self.drag
    }
}

impl<V, R> RotationalDrag for DynamicOrientedPoint<V, R>
where
    V: Wedge + Outer,
    R: RotationalMotionTypes
{
    type RotationalDrag = <R as RotationalMotionTypes>::RotationalDrag;

    fn rotational_drag(&self) -> &Self::RotationalDrag {
        &self.rotational_drag
    }

    fn rotational_drag_mut(&mut self) -> &mut Self::RotationalDrag {
        &mut self.rotational_drag
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use hoomd_vector::Cartesian;

    #[test]
    fn position() {
        let mut dynamic_point = DynamicOrientedPoint::<Cartesian<2>, Angle>::default();

        *dynamic_point.position_mut() = [1.0, 2.0].into();
        assert_eq!(dynamic_point.position, [1.0, 2.0].into());
        assert_eq!(dynamic_point.position(), &[1.0, 2.0].into());
    }

    #[test]
    fn orientation() {
        let mut dynamic_point = DynamicOrientedPoint::<Cartesian<2>, Angle>::default();

        *dynamic_point.orientation_mut() = 1.0.into();
        assert_eq!(dynamic_point.orientation, 1.0.into());
        assert_eq!(dynamic_point.orientation(), &1.0.into());
    }

    #[test]
    fn mass() {
        let dynamic_point = DynamicOrientedPoint::<Cartesian<2>, Angle> {
            mass: 3.0,
            ..Default::default()
        };

        assert_eq!(dynamic_point.mass(), 3.0);
    }

    #[test]
    fn momentum() {
        let mut dynamic_point = DynamicOrientedPoint::<Cartesian<2>, Angle>::default();

        *dynamic_point.momentum_mut() = [1.0, 2.0].into();
        assert_eq!(dynamic_point.momentum, [1.0, 2.0].into());
        assert_eq!(dynamic_point.momentum(), &[1.0, 2.0].into());
    }

    #[test]
    fn net_force() {
        let mut dynamic_point = DynamicOrientedPoint::<Cartesian<2>, Angle>::default();

        *dynamic_point.net_force_mut() = [1.0, 2.0].into();
        assert_eq!(dynamic_point.net_force, [1.0, 2.0].into());
        assert_eq!(dynamic_point.net_force(), &[1.0, 2.0].into());
    }

    #[test]
    fn moment_of_inertia() {
        let mut dynamic_point = DynamicOrientedPoint::<Cartesian<2>, Angle>::default();

        *dynamic_point.moment_of_inertia_mut() = 2.0;
        assert_eq!(dynamic_point.moment_of_inertia, 2.0);
        assert_eq!(dynamic_point.moment_of_inertia(), &2.0);
    }

    #[test]
    fn angular_momentum() {
        let mut dynamic_point = DynamicOrientedPoint::<Cartesian<2>, Angle>::default();

        *dynamic_point.angular_momentum_mut() = 2.0;
        assert_eq!(dynamic_point.angular_momentum, 2.0);
        assert_eq!(dynamic_point.angular_momentum(), &2.0);
    }

    #[test]
    fn net_torque() {
        let mut dynamic_point = DynamicOrientedPoint::<Cartesian<2>, Angle>::default();

        *dynamic_point.net_torque_mut() = 2.0;
        assert_eq!(dynamic_point.net_torque, 2.0);
        assert_eq!(dynamic_point.net_torque(), &2.0);
    }

    #[test]
    fn drag() {
        let mut dynamic_point = DynamicOrientedPoint::<Cartesian<2>, Angle>::default();

        *dynamic_point.drag_mut() = 2.0;
        assert_eq!(dynamic_point.drag, 2.0);
        assert_eq!(dynamic_point.drag(), &2.0);
    }

    #[test]
    fn rotational_drag() {
        let mut dynamic_point = DynamicOrientedPoint::<Cartesian<2>, Angle>::default();

        *dynamic_point.rotational_drag_mut() = 2.0;
        assert_eq!(dynamic_point.rotational_drag, 2.0);
        assert_eq!(dynamic_point.rotational_drag(), &2.0);
    }
}

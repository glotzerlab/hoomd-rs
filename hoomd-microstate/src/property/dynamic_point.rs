// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `DynamicPoint`

use serde::{Deserialize, Serialize};

use super::{Mass, Momentum, NetForce, OrientedPoint, Point, Position};
use crate::{Transform, property::{Drag, NetVirial}};
use hoomd_vector::{Outer, Vector};

/// A position in space with the properties necessary for translational motion in MD.
///
/// Use [`DynamicPoint`] as a [`Body`](crate::Body) property type.
///
/// A default [`DynamicPoint`] has a mass of 1.0. Position, momentum, and net force
/// of $` \vec{0} `$, and a zero-tensor for net virial.
///
/// # Example
///
/// ```
/// use hoomd_microstate::property::DynamicPoint;
/// use hoomd_vector::Cartesian;
///
/// let dynamic_point = DynamicPoint {
///     position: Cartesian::from([1.0, -3.0]),
///     mass: 1.0,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicPoint<V>
where
    V: Outer,
{
    /// The location of the extended body in space $`[\mathrm{length}]`$.
    pub position: V,

    /// The mass of the extended body $` [\mathrm{mass}] `$.
    pub mass: f64,

    /// The translational momentum of the extended body $`[\mathrm{mass} \cdot \mathrm{length}] \cdot \mathrm{time}^{-1}]`$.
    pub momentum: V,

    /// The net force applied to the body in a [`Microstate`](crate::Microstate) $`[\mathrm{mass} \cdot \mathrm{length}] \cdot \mathrm{time}^{-2}]`$.
    pub net_force: V,

    /// The net virial applied to the body in a [`Microstate`](crate::Microstate) $`[\mathrm{energy}]`$.
    pub net_virial: V::Tensor,

    /// The translational drag coefficient.
    #[serde(default)]
    pub drag: f64,
}

impl<V> Default for DynamicPoint<V>
where
    V: Default + Outer,
    V::Tensor: Default,
{
    /// Construct a [`DynamicPoint`] with mass 1.0. Position, momentum, and net force are set
    /// to the 0 vector.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_linear_algebra::{GeneralMatrix, matrix::Matrix};
    /// use hoomd_microstate::property::DynamicPoint;
    /// use hoomd_vector::Cartesian;
    ///
    /// let dynamic_point = DynamicPoint::<Cartesian<3>>::default();
    /// assert_eq!(dynamic_point.mass, 1.0);
    /// assert_eq!(dynamic_point.position, [0.0, 0.0, 0.0].into());
    /// assert_eq!(dynamic_point.momentum, [0.0, 0.0, 0.0].into());
    /// assert_eq!(dynamic_point.net_force, [0.0, 0.0, 0.0].into());
    /// assert_eq!(dynamic_point.net_virial, Matrix::zeros());
    /// assert_eq!(dynamic_point.drag, 1.0);
    /// ```
    #[inline]
    fn default() -> Self {
        Self {
            position: Default::default(),
            mass: 1.0,
            momentum: Default::default(),
            net_force: Default::default(),
            net_virial: V::Tensor::default(),
            drag: 1.0,
        }
    }
}

impl<V: Vector + Outer> Transform<Point<V>> for DynamicPoint<V> {
    /// [`DynamicPoint`] transforms [`Point`] by vector addition.
    ///
    /// ```math
    /// \vec{r} = \vec{r}_\mathrm{body} + \vec{r}_\mathrm{site}
    /// ```
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_microstate::{
    ///     Transform,
    ///     property::{DynamicPoint, Point},
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// let body_properties = DynamicPoint {
    ///     position: Cartesian::from([1.0, -2.0, 3.0]),
    ///     ..Default::default()
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
    V: Vector + Outer,
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
    /// use hoomd_microstate::{
    ///     Transform,
    ///     property::{DynamicPoint, OrientedPoint},
    /// };
    /// use hoomd_vector::{Cartesian, Versor};
    ///
    /// let body_properties = DynamicPoint {
    ///     position: Cartesian::from([1.0, -2.0, 3.0]),
    ///     ..Default::default()
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

impl<P: Outer> Position for DynamicPoint<P> {
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
    V: std::ops::Mul<f64, Output = V> + std::ops::Div<f64, Output = V> + Copy + Outer,
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

impl<V: Outer> Mass for DynamicPoint<V> {
    #[inline]
    fn mass(&self) -> f64 {
        self.mass
    }
}

impl<V: Outer> NetForce for DynamicPoint<V> {
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

impl<V: Outer> NetVirial for DynamicPoint<V> {
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

impl<V: Outer> Drag for DynamicPoint<V> {
    fn drag(&self) -> &f64 {
        &self.drag
    }

    fn drag_mut(&mut self) -> &mut f64 {
        &mut self.drag
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use hoomd_vector::Cartesian;

    #[test]
    fn position() {
        let mut dynamic_point = DynamicPoint::<Cartesian<2>>::default();

        *dynamic_point.position_mut() = [1.0, 2.0].into();
        assert_eq!(dynamic_point.position, [1.0, 2.0].into());
        assert_eq!(dynamic_point.position(), &[1.0, 2.0].into());
    }

    #[test]
    fn mass() {
        let dynamic_point = DynamicPoint::<Cartesian<2>> {
            mass: 3.0,
            ..Default::default()
        };

        assert_eq!(dynamic_point.mass(), 3.0);
    }

    #[test]
    fn momentum() {
        let mut dynamic_point = DynamicPoint::<Cartesian<2>>::default();

        *dynamic_point.momentum_mut() = [1.0, 2.0].into();
        assert_eq!(dynamic_point.momentum, [1.0, 2.0].into());
        assert_eq!(dynamic_point.momentum(), &[1.0, 2.0].into());
    }

    #[test]
    fn net_force() {
        let mut dynamic_point = DynamicPoint::<Cartesian<2>>::default();

        *dynamic_point.net_force_mut() = [1.0, 2.0].into();
        assert_eq!(dynamic_point.net_force, [1.0, 2.0].into());
        assert_eq!(dynamic_point.net_force(), &[1.0, 2.0].into());
    }

    #[test]
    fn drag() {
        let mut dynamic_point = DynamicPoint::<Cartesian<2>>::default();

        *dynamic_point.drag_mut() = 2.0;
        assert_eq!(dynamic_point.drag, 2.0);
        assert_eq!(dynamic_point.drag(), &2.0);
    }
}

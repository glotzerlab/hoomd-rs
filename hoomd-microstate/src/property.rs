// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Traits that describe body and/or site properties as a selection types that implement them.
//!
//! See the [crate-level documentation](crate) for an overview of how body and site
//! properties interact with [`Microstate`](crate::Microstate) and model methods.
//!
//! # Provided types
//!
//! The structs provided in `property` may be used as [`Body`](crate::Body) and/or
//! [`Site`](crate::Site) properties.
//!
//! [`Point`] represents a position in space:
//! ```
//! use hoomd_microstate::property::Point;
//! use hoomd_vector::Cartesian;
//!
//! let point = Point::new(Cartesian::from([1.0, -3.0]));
//! ```
//!
//! [`OrientedPoint`] contains both the position and orientation of an extended body:
//! ```
//! use hoomd_microstate::property::OrientedPoint;
//! use hoomd_vector::{Angle, Cartesian};
//!
//! let point = OrientedPoint {
//!     position: Cartesian::from([1.0, -3.0]),
//!     orientation: Angle::from(1.2),
//! };
//! ```
//!
//! [`DynamicPoint`] is a point in space with mass and momentum:
//! ```
//! use hoomd_microstate::property::DynamicPoint;
//! use hoomd_vector::Cartesian;
//!
//! let dynamic_point = DynamicPoint {
//!     position: Cartesian::from([1.0, -3.0]),
//!     momentum: Cartesian::from([-1.0, 2.0]),
//!     mass: 0.5,
//!     ..Default::default()
//! };
//! ```
//!
//! [`DynamicOrientedPoint`] is an extended body with position, orientation, mass, momentum,
//! a moment of inertia, and angular momentum:
//! ```
//! use hoomd_microstate::property::DynamicOrientedPoint;
//! use hoomd_vector::{Angle, Cartesian};
//! use std::f64::consts::PI;
//!
//! let dynamic_point = DynamicOrientedPoint {
//!     position: Cartesian::from([1.0, -3.0]),
//!     orientation: Angle::from(PI / 4.0),
//!     momentum: Cartesian::from([-1.0, 2.0]),
//!     mass: 0.5,
//!     moment_of_inertia: 2.0,
//!     angular_momentum: 1.5,
//!     ..Default::default()
//! };
//! ```
//!
//! Use the `Point`, `OrientedPoint` or a custom type to represent interaction sites.
//! `Point` and `OrientedPoint` can also be used for body properties in Monte Carlo simulations.
//! Use the `Dynamic` variants for body properties in molecular dynamics simulations.
//!
//! # Custom property types
//!
//! When none of the provided types meets your needs, you can define a custom type.
//! You must implement [`Position`] for your type and may implement other
//! property traits as needed by your model.
//!
//! For example, this `Custom` type implements [`Position`], [`Orientation`],
//! and has a `custom` field. The full site properties type is available when
//! hoomd-rs computes interactions on sites, so you can use the custom fields
//! in your own custom interaction potentials.
//!
//! ```
//! use hoomd_microstate::property::{Orientation, Position};
//! use hoomd_vector::{Cartesian, Versor};
//!
//! #[derive(Position, Orientation)]
//! struct Custom {
//!     position: Cartesian<3>,
//!     orientation: Versor,
//!     custom: f64,
//! }
//! ```
//!
//! ## Transformations
//!
//! Implement `Transform` to take sites from the body frame to the system frame.
//! Typically, this involves transforming position and orientation while leaving
//! all other fields unchanged. The three most common implementations of `Transform`
//! follow. All these examples are in 3D. To convert to 2D, replace `Cartesian<3>`
//! with `Cartesian<2>` and `Versor` with `Angle`.
//!
//! Non-oriented bodies and sites (i.e. point particles or non-rotating rigid bodies):
//! ```
//! use hoomd_microstate::{
//!     Transform,
//!     property::{Point, Position},
//! };
//! use hoomd_vector::Cartesian;
//!
//! #[derive(Position)]
//! struct Custom {
//!     position: Cartesian<3>,
//!     custom: f64,
//! }
//!
//! impl Transform<Custom> for Point<Cartesian<3>> {
//!     fn transform(&self, site_properties: &Custom) -> Custom {
//!         Custom {
//!             position: self.position + site_properties.position,
//!             ..*site_properties
//!         }
//!     }
//! }
//! ```
//!
//! Oriented bodies and non-oriented sites (i.e. rotating rigid bodies with
//! isotropic site-site interactions):
//! ```
//! use hoomd_microstate::{
//!     Transform,
//!     property::{OrientedPoint, Position},
//! };
//! use hoomd_vector::{Cartesian, Rotate, Rotation, Versor};
//!
//! #[derive(Position)]
//! struct Custom {
//!     position: Cartesian<3>,
//!     custom: f64,
//! }
//!
//! impl Transform<Custom> for OrientedPoint<Cartesian<3>, Versor> {
//!     fn transform(&self, site_properties: &Custom) -> Custom {
//!         Custom {
//!             position: self.position
//!                 + self.orientation.rotate(&site_properties.position),
//!             ..*site_properties
//!         }
//!     }
//! }
//! ```
//!
//! Oriented bodies and oriented sites (i.e. rotating rigid bodies with
//! anisotropic site-site interactions):
//! ```
//! use hoomd_microstate::{
//!     Transform,
//!     property::{Orientation, OrientedPoint, Position},
//! };
//! use hoomd_vector::{Cartesian, Rotate, Rotation, Versor};
//!
//! #[derive(Position, Orientation)]
//! struct Custom {
//!     position: Cartesian<3>,
//!     orientation: Versor,
//!     custom: f64,
//! }
//!
//! impl Transform<Custom> for OrientedPoint<Cartesian<3>, Versor> {
//!     fn transform(&self, site_properties: &Custom) -> Custom {
//!         Custom {
//!             position: self.position
//!                 + self.orientation.rotate(&site_properties.position),
//!             orientation: self
//!                 .orientation
//!                 .combine(&site_properties.orientation),
//!             ..*site_properties
//!         }
//!     }
//! }
//! ```

mod point;
use std::ops::{Div, Mul};

use hoomd_vector::{Angle, Cartesian, Outer, Versor, Wedge};
pub use point::Point;

mod oriented_point;
pub use oriented_point::OrientedPoint;

mod dynamic_point;
pub use dynamic_point::DynamicPoint;

mod dynamic_oriented_point;
pub use dynamic_oriented_point::DynamicOrientedPoint;

mod oriented_hyperbolic_point;
pub use oriented_hyperbolic_point::OrientedHyperbolicPoint;

pub use hoomd_derive::{Orientation, Position};
use serde::{Deserialize, Serialize};

use crate::Transform;

/// Locate a site or body in space: $` \vec{r} `$
///
/// When applied to body properties, [`Position`] describes the location of the body
/// relative to the origin of the system coordinate system.
///
/// When applied to site properties, [`Position`] has a context-dependent definition.
/// * Elements in [`Microstate::sites`] have a position located in the system frame.
/// * Elements in [`Body::sites`] have a position located in the body frame.
///
/// [`Body::Sites`]: crate::Body::sites
/// [`Microstate::sites`]: crate::Microstate::sites
///
/// # Units
///
/// Position vectors have units of $`[\mathrm{length}]`$.
///
/// # Derive macro
///
/// Use the [`Position`](macro@Position) derive macro to automatically implement
/// the `Position` trait on a type. The type **must** have a field named `position`.
/// ```
/// use hoomd_microstate::property::Position;
/// use hoomd_vector::Cartesian;
///
/// #[derive(Position)]
/// struct Custom {
///     position: Cartesian<3>,
/// }
/// ```
pub trait Position {
    /// Every position is located in this vector space.
    type Position;

    /// The position of this body or site $`[\mathrm{length}]`$.
    fn position(&self) -> &Self::Position;

    /// The mutable position of this body or site $`[\mathrm{length}]`$.
    fn position_mut(&mut self) -> &mut Self::Position;
}

/// The translational motion of a body: $` \vec{p} `$
///
/// [`Momentum`] describes the translational motion of the body relative to the origin of the
/// system coordinate system.
///
/// `hoomd_md` does not compute or utilize the momentum of sites.
///
/// # Units
///
/// Momentum vectors have units of $`[ \mathrm{energy}^{1/2} \cdot \mathrm{mass}^{1/2}]`$.
pub trait Momentum {
    /// Type that can express momentum and velocity.
    type Momentum;

    /// The momentum of this body $`[ \mathrm{energy}^{1/2} \cdot \mathrm{mass}^{1/2}]`$.
    fn momentum(&self) -> &Self::Momentum;

    /// The mutable momentum of this body $`[ \mathrm{energy}^{1/2} \cdot \mathrm{mass}^{1/2}]`$.
    fn momentum_mut(&mut self) -> &mut Self::Momentum;

    /// The velocity of this body $`[ \mathrm{energy}^{1/2} \cdot \mathrm{mass}^{-1/2}]`$.
    fn velocity(&self) -> Self::Momentum;

    /// Change the velocity of this body.
    fn set_velocity(&mut self, velocity: Self::Momentum);
}

/// The total force acting on a site or body: $` \vec{F} `$
///
/// [`NetForce`] is set only for bodies that belong to a microstate. It is always in the
/// system frame.
///
/// `hoomd_md` does not store the net force acting on individual sites. Use methods in
/// `hoomd_interaction` to compute forces on sites when needed.
///
/// # Units
///
/// Net force vectors have units of $`[\mathrm{energy} \cdot \mathrm{length}^{-1}]`$.
pub trait NetForce {
    /// Force vector type.
    type NetForce;

    /// The net force on this body $`[\mathrm{energy} \cdot \mathrm{length}^{-1}]`$.
    fn net_force(&self) -> &Self::NetForce;

    /// The mutable net force on this body $`[\mathrm{energy} \cdot \mathrm{length}^{-1}]`$.
    fn net_force_mut(&mut self) -> &mut Self::NetForce;
}

/// The total virial acting on a site or body: $` \mathbf{W} `$
///
/// [`NetVirial`] is set only for bodies that belong to a microstate. It is always in the
/// system frame.
///
/// `hoomd_md` does not store the net virial acting on individual sites. Use methods in
/// `hoomd_interaction` to compute virials on sites when needed.
///
/// # Units
///
/// Net virial matrices have units of $`[\mathrm{energy}`$.
pub trait NetVirial {
    /// Virial vector type.
    type NetVirial;

    /// The net virial on this body $`[\mathrm{energy}]`$.
    fn net_virial(&self) -> &Self::NetVirial;

    /// The mutable net virial on this body $`[\mathrm{energy}]`$.
    fn net_virial_mut(&mut self) -> &mut Self::NetVirial;
}

/// The orientation of a site or body: $` \theta `$ or $` \mathbf{q} `$.
///
/// When applied to site properties, [`Orientation`] has a context-dependent definition.
/// * Elements in [`Microstate::sites`] describe the rotation from the site's local frame
///   to the system frame.
/// * Elements in [`Body::sites`] describe the rotation from the site's local frame to the
///   body frame.
///
/// When applied to body properties, [`Orientation`] describes the rotation from
/// the body frame to the system frame.
///
/// [`Body::Sites`]: crate::Body::sites
/// [`Microstate::sites`]: crate::Microstate::sites
///
/// # Units
///
/// The units of [`Orientation`] depend on the representation chosen for `Rotation`.
/// For example, [`hoomd_vector::Angle`] has units of radians while
/// [`hoomd_vector::Versor`] is unitless.
///
/// # Derive macro
///
/// Use the [`Orientation`](macro@Orientation) derive macro to automatically implement
/// the `Orientation` trait on a type. The type **must** have a field named `orientation`.
/// ```
/// use hoomd_microstate::property::Orientation;
/// use hoomd_vector::Versor;
///
/// #[derive(Orientation)]
/// struct Custom {
///     orientation: Versor,
/// }
/// ```
pub trait Orientation {
    /// Type that can express the orientation of a site or body.
    type Rotation;

    /// The orientation of this site or body.
    fn orientation(&self) -> &Self::Rotation;

    /// The orientation of this site or body (mutable).
    fn orientation_mut(&mut self) -> &mut Self::Rotation;
}

/// A body's resistance to change in translational motion: $` m `$
///
/// [`Mass`] connects a body's linear momentum to its linear velocity: $` \vec{p} = m \vec{v} `$.
///
/// `hoomd_md` does not compute or utilize the mass of sites.
///
/// # Units
///
/// The units of [`Mass`] are $` [\mathrm{mass}] `$.
pub trait Mass {
    /// The mass of this body $` [\mathrm{mass}] `$.
    fn mass(&self) -> f64;
}

/// A body's resistance to a change in rotational motion: $` I `$
///
/// [`MomentOfInertia`] connects a body's angular momentum to its angular velocity:
/// $` \vec{L} = I \vec{\omega} `$.
///
/// `hoomd_md` does not compute or utilize the moment of inertia of sites.
///
/// # Units
///
/// The units of [`MomentOfInertia`] are $` [\mathrm{mass} \cdot \mathrm{length}^2] `$.
pub trait MomentOfInertia {
    /// Type that expresses the moment of inertia.
    type MomentOfInertia;

    /// The moment of inertia of this body $` [\mathrm{mass} \cdot \mathrm{length}^2] `$.
    fn moment_of_inertia(&self) -> &Self::MomentOfInertia;

    /// The mutable moment of inertia of this body $` [\mathrm{mass} \cdot \mathrm{length}^2] `$.
    fn moment_of_inertia_mut(&mut self) -> &mut Self::MomentOfInertia;
}

/// The rotational motion of a body: $` \vec{L} `$
///
/// [`AngularMomentum`] describes the rotational motion of the body in the *body* frame.
///
/// `hoomd_md` does not compute or utilize the angular momentum of sites.
///
/// # Units
///
/// The units of [`AngularMomentum`] are $` [\mathrm{mass}^{1/2} \cdot \mathrm{length} \cdot \mathrm{energy}^{1/2}] `$.
pub trait AngularMomentum {
    /// Type that can express the angular momentum of a site or body.
    type AngularMomentum;

    /// The angular momentum of this site or body $` [\mathrm{mass}^{1/2} \cdot \mathrm{length} \cdot \mathrm{energy}^{1/2}] `$.
    fn angular_momentum(&self) -> &Self::AngularMomentum;

    /// The mutable angular momentum of this site or body $` [\mathrm{mass}^{1/2} \cdot \mathrm{length} \cdot \mathrm{energy}^{1/2}] `$.
    fn angular_momentum_mut(&mut self) -> &mut Self::AngularMomentum;
}

/// The total torque acting on a body: $` \vec{\tau} `$
///
/// [`NetTorque`] is set only for bodies that belong to a microstate. It is always in the
/// system frame.
///
/// `hoomd_md` does not store the net force acting on individual sites. Use methods in
/// `hoomd_interaction` to compute torques on sites when needed.
///
/// # Units
///
/// The units of [`NetTorque`] are $` [\mathrm{energy}] `$.
pub trait NetTorque {
    /// Type that can express the net torque on a site or body.
    type NetTorque;

    /// The net torque on this site or body $` [\mathrm{energy}] `$.
    fn net_torque(&self) -> &Self::NetTorque;

    /// The mutable net torque on this site or body $` [\mathrm{energy}] `$.
    fn net_torque_mut(&mut self) -> &mut Self::NetTorque;
}

/// Moment of inertia and angular momentum types.
///
/// [`RotationalMotionTypes`] sets which structs store the moment of inertia
/// and angular momentum for a given `Rotation` representation.
pub trait RotationalMotionTypes {
    /// Type that stores the moment of inertia in the natural coordinate frame of the body's local rotation.
    type MomentOfInertia;
    /// Type that stores the angular momentum.
    type AngularMomentum;
}

// TODO: inline?

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomBodyCartesian2<R, E> {
    pub required: R,
    pub extra: E,
}

impl<R, E> Transform<Point<Cartesian<2>>> for CustomBodyCartesian2<R, E>
where
    R: Transform<Point<Cartesian<2>>>
{
    #[inline]
    fn transform(&self, site_properties: &Point<Cartesian<2>>) -> Point<Cartesian<2>> {
        self.required.transform(site_properties)
    }
}

impl<R, E> Transform<OrientedPoint<Cartesian<2>, Angle>> for CustomBodyCartesian2<R, E>
where
    R: Transform<OrientedPoint<Cartesian<2>, Angle>>
{
    #[inline]
    fn transform(
        &self,
        site_properties: &OrientedPoint<Cartesian<2>, Angle>
    ) -> OrientedPoint<Cartesian<2>, Angle> {
        self.required.transform(site_properties)
    }
}

impl<R: Position<Position = Cartesian<2>>, E> Position for CustomBodyCartesian2<R, E> {
    type Position = Cartesian<2>;

    fn position(&self) -> &Self::Position {
        self.required.position()
    }

    fn position_mut(&mut self) -> &mut Self::Position {
        self.required.position_mut()
    }
}

// these are here so that we can use a custom body with constant volume integration


impl<R, E> Orientation for CustomBodyCartesian2<R, E>
where
    R: Orientation<Rotation = Angle>
{
    type Rotation = Angle;

    fn orientation(&self) -> &Self::Rotation {
        self.required.orientation()
    }

    fn orientation_mut(&mut self) -> &mut Self::Rotation {
        self.required.orientation_mut()
    }
}

impl<R, E> Momentum for CustomBodyCartesian2<R, E>
where
    R: Momentum<Momentum = Cartesian<2>> + Mass
{
    type Momentum = Cartesian<2>;

    fn momentum(&self) -> &Self::Momentum {
        self.required.momentum()
    }

    fn momentum_mut(&mut self) -> &mut Self::Momentum {
        self.required.momentum_mut()
    }

    fn velocity(&self) -> Self::Momentum {
        *self.required.momentum() / self.required.mass()
    }

    fn set_velocity(&mut self, velocity: Self::Momentum) {
        *self.required.momentum_mut() = velocity * self.required.mass();
    }
}

impl<R: Mass, E> Mass for CustomBodyCartesian2<R, E> {
    fn mass(&self) -> f64 {
        self.required.mass()
    }
}

impl<R, E> NetForce for CustomBodyCartesian2<R, E>
where
    R: NetForce<NetForce = Cartesian<2>>
{
    type NetForce = Cartesian<2>;

    fn net_force(&self) -> &Self::NetForce {
        self.required.net_force()
    }

    fn net_force_mut(&mut self) -> &mut Self::NetForce {
        self.required.net_force_mut()
    }
}

impl<R, E> NetVirial for CustomBodyCartesian2<R, E>
where
    R: NetVirial<NetVirial = <Cartesian<2> as Outer>::Tensor>
{
    type NetVirial = <Cartesian<2> as Outer>::Tensor;

    fn net_virial(&self) -> &Self::NetVirial {
        self.required.net_virial()
    }

    fn net_virial_mut(&mut self) -> &mut Self::NetVirial {
        self.required.net_virial_mut()
    }
}

impl<R, E> MomentOfInertia for CustomBodyCartesian2<R, E>
where
    R: MomentOfInertia<MomentOfInertia = <Angle as RotationalMotionTypes>::MomentOfInertia>
{
    type MomentOfInertia = <Angle as RotationalMotionTypes>::MomentOfInertia;

    fn moment_of_inertia(&self) -> &Self::MomentOfInertia {
        self.required.moment_of_inertia()
    }

    fn moment_of_inertia_mut(&mut self) -> &mut Self::MomentOfInertia {
        self.required.moment_of_inertia_mut()
    }
}

impl<R, E> AngularMomentum for CustomBodyCartesian2<R, E>
where
    R: AngularMomentum<AngularMomentum = <Angle as RotationalMotionTypes>::AngularMomentum>
{
    type AngularMomentum = <Angle as RotationalMotionTypes>::AngularMomentum;

    fn angular_momentum(&self) -> &Self::AngularMomentum {
        self.required.angular_momentum()
    }

    fn angular_momentum_mut(&mut self) -> &mut Self::AngularMomentum {
        self.required.angular_momentum_mut()
    }
}

impl<R, E> NetTorque for CustomBodyCartesian2<R, E>
where
    R: NetTorque<NetTorque = <Cartesian<2> as Wedge>::Bivector>
{
    type NetTorque = <Cartesian<2> as Wedge>::Bivector;

    fn net_torque(&self) -> &Self::NetTorque {
        self.required.net_torque()
    }

    fn net_torque_mut(&mut self) -> &mut Self::NetTorque {
        self.required.net_torque_mut()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomBodyCartesian3<R, E> {
    pub required: R,
    pub extra: E,
}

impl<R, E> Transform<Point<Cartesian<3>>> for CustomBodyCartesian3<R, E>
where
    R: Transform<Point<Cartesian<3>>>
{
    #[inline]
    fn transform(&self, site_properties: &Point<Cartesian<3>>) -> Point<Cartesian<3>> {
        self.required.transform(site_properties)
    }
}

impl<R, E> Transform<OrientedPoint<Cartesian<3>, Versor>> for CustomBodyCartesian3<R, E>
where
    R: Transform<OrientedPoint<Cartesian<3>, Versor>>
{
    #[inline]
    fn transform(
        &self,
        site_properties: &OrientedPoint<Cartesian<3>, Versor>
    ) -> OrientedPoint<Cartesian<3>, Versor> {
        self.required.transform(site_properties)
    }
}

impl<R, E> Position for CustomBodyCartesian3<R, E>
where
    R: Position<Position = Cartesian<3>>
{
    type Position = Cartesian<3>;

    fn position(&self) -> &Self::Position {
        self.required.position()
    }

    fn position_mut(&mut self) -> &mut Self::Position {
        self.required.position_mut()
    }
}

impl<R, E> Orientation for CustomBodyCartesian3<R, E>
where
    R: Orientation<Rotation = Versor>
{
    type Rotation = Versor;

    fn orientation(&self) -> &Self::Rotation {
        self.required.orientation()
    }

    fn orientation_mut(&mut self) -> &mut Self::Rotation {
        self.required.orientation_mut()
    }
}

impl<R, E> Momentum for CustomBodyCartesian3<R, E>
where
    R: Momentum<Momentum = Cartesian<3>> + Mass
{
    type Momentum = Cartesian<3>;

    fn momentum(&self) -> &Self::Momentum {
        self.required.momentum()
    }

    fn momentum_mut(&mut self) -> &mut Self::Momentum {
        self.required.momentum_mut()
    }

    fn velocity(&self) -> Self::Momentum {
        *self.required.momentum() / self.required.mass()
    }

    fn set_velocity(&mut self, velocity: Self::Momentum) {
        *self.required.momentum_mut() = velocity * self.required.mass();
    }
}

impl<R: Mass, E> Mass for CustomBodyCartesian3<R, E> {
    fn mass(&self) -> f64 {
        self.required.mass()
    }
}

impl<R, E> NetForce for CustomBodyCartesian3<R, E>
where
    R: NetForce<NetForce = Cartesian<3>>
{
    type NetForce = Cartesian<3>;

    fn net_force(&self) -> &Self::NetForce {
        self.required.net_force()
    }

    fn net_force_mut(&mut self) -> &mut Self::NetForce {
        self.required.net_force_mut()
    }
}

impl<R, E> NetVirial for CustomBodyCartesian3<R, E>
where
    R: NetVirial<NetVirial = <Cartesian<3> as Outer>::Tensor>
{
    type NetVirial = <Cartesian<3> as Outer>::Tensor;

    fn net_virial(&self) -> &Self::NetVirial {
        self.required.net_virial()
    }

    fn net_virial_mut(&mut self) -> &mut Self::NetVirial {
        self.required.net_virial_mut()
    }
}

impl<R, E> MomentOfInertia for CustomBodyCartesian3<R, E>
where
    R: MomentOfInertia<MomentOfInertia = <Versor as RotationalMotionTypes>::MomentOfInertia>
{
    type MomentOfInertia = <Versor as RotationalMotionTypes>::MomentOfInertia;

    fn moment_of_inertia(&self) -> &Self::MomentOfInertia {
        self.required.moment_of_inertia()
    }

    fn moment_of_inertia_mut(&mut self) -> &mut Self::MomentOfInertia {
        self.required.moment_of_inertia_mut()
    }
}

impl<R, E> AngularMomentum for CustomBodyCartesian3<R, E>
where
    R: AngularMomentum<AngularMomentum = <Versor as RotationalMotionTypes>::AngularMomentum>
{
    type AngularMomentum = <Versor as RotationalMotionTypes>::AngularMomentum;

    fn angular_momentum(&self) -> &Self::AngularMomentum {
        self.required.angular_momentum()
    }

    fn angular_momentum_mut(&mut self) -> &mut Self::AngularMomentum {
        self.required.angular_momentum_mut()
    }
}

impl<R, E> NetTorque for CustomBodyCartesian3<R, E>
where
    R: NetTorque<NetTorque = <Cartesian<3> as Wedge>::Bivector>
{
    type NetTorque = <Cartesian<3> as Wedge>::Bivector;

    fn net_torque(&self) -> &Self::NetTorque {
        self.required.net_torque()
    }

    fn net_torque_mut(&mut self) -> &mut Self::NetTorque {
        self.required.net_torque_mut()
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Traits that describe body and/or site properties a a selection types that implement them.
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
pub use point::Point;

mod oriented_point;
pub use oriented_point::OrientedPoint;

mod dynamics_point;
pub use dynamics_point::DynamicPoint;

mod oriented_dynamics_point;
pub use oriented_dynamics_point::OrientedDynamicPoint;

mod oriented_hyperbolic_point;
pub use oriented_hyperbolic_point::OrientedHyperbolicPoint;

pub use hoomd_derive::{Orientation, Position};

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

/// The translational motion of a site or body: $` \vec{p} `$
/// 
/// When applied to body properties, [`Momentum`] describes the linear motion of the body
/// relative to the origin of the system coordinate system.
/// 
/// When applied to site properties, [`Momentum`] has a context-dependent definition.
/// * Elements in [`Microstate::sites`] have a linear momentum defined in the system frame.
/// * Linear momentum is undefined for elements in [`Body::sites`]. Sites cannot have
///   a natural momentum in the body frame, they momentum of a site in the body frame is
///   a property of the linear and angular momentum of the body.
///
/// [`Body::Sites`]: crate::Body::sites
/// [`Microstate::sites`]: crate::Microstate::sites
/// 
/// # Units
/// 
/// Momentum vectors have units of $`[\mathrm{length} \cdot \mathrm{mass} \cdot \mathrm{time}^{-1}]`$.
pub trait Momentum {
    /// Every momentum is within this vector space.
    type Momentum;

    /// The momentum of this site or body $`[\mathrm{length} \cdot \mathrm{mass} \cdot \mathrm{time}^{-1}]`$.
    fn momentum(&self) -> &Self::Momentum;

    /// The mutable momentum of this site or body $`[\mathrm{length} \cdot \mathrm{mass} \cdot \mathrm{time}^{-1}]`$.
    fn momentum_mut(&mut self) -> &mut Self::Momentum;

    /// The velocity of this site or body $`[\mathrm{length} \cdot \mathrm{time}^{-1}]`$.
    fn velocity(&self) -> Self::Momentum;

    /// Change the velocity of this site or body.
    fn set_velocity(&mut self, velocity: Self::Momentum);
}

/// The total force acting on a site or body: $` \vec{F} `$
///
/// [`NetForce`] is set only for bodies that belong to a microstate. It is always in the
/// system frame.
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
/// # Units
/// 
/// The units of [`Mass`] are $` [\mathrm{mass}] `$.
pub trait Mass {
    /// The mass of this site or body $` [\mathrm{mass}] `$.
    fn mass(&self) -> f64;
}

/// A body's resistance to a change in rotational motion: $` I `$
/// 
/// [`MomentOfInertia`] is the quantity which determines a site or body's angular
/// inertia, and which together with angular velocity determines a site or body's
/// angular momentum.
/// 
/// TODO: make this be a square matrix instead of a vector
/// 
/// # Units
/// 
/// The units of `[MomentOfInertia`] are *\[mass * \length^2\]*.
pub trait MomentOfInertia {
    /// Every moment of inertia is within this vector space.
    type Vector;

    /// The moment of inertia of this site or body *\[mass * \length^2\]*.
    fn moment_of_inertia(&self) -> &Self::Vector;

    /// The mutable moment of inertia of this site or body *\[mass * \length^2\]*.
    fn moment_of_inertia_mut(&mut self) -> &mut Self::Vector;
}

///The rotational motion of a site or body: $` \vec{L} `$
/// 
/// [`AngularMomentum`] is the quantity which determines a site or body's angular
/// momentum
/// 
/// # Units
/// 
/// The units of `[AngularMomentum`] are *\[radian/time\ * mass * \length^2\]*.
pub trait AngularMomentum {
    /// Type that can express the angular momentum of a site or body.
    type AngularMomentum;
    
    /// The angular momentum of this site or body *\[radian/time\ * mass * \length^2\]*.
    fn angular_momentum(&self) -> &Self::AngularMomentum;

    /// The mutable angular momentum of this site or body *\[radian/time\ * mass * \length^2\]*.
    fn angular_momentum_mut(&mut self) -> &mut Self::AngularMomentum;
}

/// The total torque acting on a body: $` \vec{\omega} `$
/// 
/// [`NetTorque`] is the quantity which determines a site or body's net torque.
/// 
/// # Units
/// 
/// The units of [`NetTorque`] are *\[radian/time^2\ * mass * \length^2\]*.
pub trait NetTorque {
    /// Type that can express the net torque on a site or body.
    type NetTorque;
    
    /// The net torque on this site or body *\[radian/time^2\ * mass * \length^2\]*.
    fn net_torque(&self) -> &Self::NetTorque;

    /// The mutable net torque on this site or body *\[radian/time^2\ * mass * \length^2\]*.
    fn net_torque_mut(&mut self) -> &mut Self::NetTorque;
}

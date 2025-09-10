// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Traits that describe body and/or site properties a a selection types that implement them.

See the [crate-level documentation](crate) for an overview of how body and site
properties interact with [`Microstate`](crate::Microstate) and model methods.

# Provided types

The structs provided in `property` may be used as [`Body`](crate::Body) and/or
[`Site`](crate::Site) properties.

[`Point`] represents a position in space:
```
use hoomd_microstate::property::Point;
use hoomd_vector::Cartesian;

let point = Point::new(Cartesian::from([1.0, -3.0]));
```

[`OrientedPoint`] contains both the position and orientation of an extended body:
```
use hoomd_microstate::property::OrientedPoint;
use hoomd_vector::{Angle, Cartesian};

let point = OrientedPoint { position: Cartesian::from([1.0, -3.0]),
    orientation: Angle::from(1.2),
};
```

[`DynamicsPoint`] contains the position, mass, velocity, and acceleration of an
extended body, such as is useful for Molecular Dynamics simulations:
```
use hoomd_microstate::property::DynamicsPoint;
use hoomd_vector::Cartesian;

let dynamics_point = DynamicsPoint {
    position: Cartesian::from([1.0, -3.0]),
    mass: 1.0,
    velocity: Cartesian::from([0.0, 1.0]),
    acceleration: Cartesian::from([1.0, 0.0])
};
```

# Custom property types

When none of the provided types meets your needs, you can define a custom type.
You must implement [`Position`] for your type and may implement other
property traits as needed by your model.

For example, this `Custom` type implements [`Position`], [`Orientation`],
and has a `custom` field. The full site properties type is available when
hoomd-rs computes interactions on sites, so you can use the custom fields
in your own custom interaction potentials.

```
use hoomd_vector::{Cartesian, Versor};
use hoomd_microstate::property::{Orientation, Position};

struct Custom {
    position: Cartesian<3>,
    orientation: Versor,
    custom: f64,
}

impl Orientation for Custom {
    type Rotation = Versor;

    fn orientation(&self) -> &Versor {
        &self.orientation
    }

    fn orientation_mut(&mut self) -> &mut Versor {
        &mut self.orientation
    }
}

impl Position for Custom {
    type Vector = Cartesian<3>;

    fn position(&self) -> &Cartesian<3> {
        &self.position
    }

    fn position_mut(&mut self) -> &mut Cartesian<3> {
        &mut self.position
    }
}
```

## Transformations


Implement `Transform` to take sites from the body frame to the system frame.
Typically, this involves transforming position and orientation while leaving
all other fields unchanged:

```
use hoomd_vector::{Cartesian, Rotate, Rotation, Versor};
use hoomd_microstate::{Transform, property::{Orientation, OrientedPoint, Position}};

struct Custom {
    position: Cartesian<3>,
    orientation: Versor,
    custom: f64,
    }

impl Transform<Custom> for OrientedPoint<Cartesian<3>, Versor> {
    fn transform(&self, site_properties: &Custom) -> Custom {
        Custom {
            position: self.position + self.orientation.rotate(&site_properties.position),
            orientation: self.orientation.combine(&site_properties.orientation),
            .. *site_properties
        }
    }
}
```
*/

mod point;
pub use point::Point;

mod oriented_point;
pub use oriented_point::OrientedPoint;

mod dynamics_point;
pub use dynamics_point::DynamicsPoint;

mod oriented_dynamics_point;
pub use oriented_dynamics_point::OrientedDynamicsPoint;

/** Locate sites and bodies.

When applied to site properties, [`Position`] describes the location of the site
relative to the origin of the body. In other words, it is the position of the
site in the body reference frame.

When applied to body properties [`Position`] describes the location of the body
relative to the origin of the system coordinate system. In other words, it is
the position of the body's origin in the system reference frame.

# Units

Position vectors have units of *\[length\]*.
*/
pub trait Position {
    /// Every position is located in this vector space.
    type Vector;

    /// The position of this body or site *\[length\]*.
    fn position(&self) -> &Self::Vector;

    /// The mutable position of this body or site *\[length\]*.
    fn position_mut(&mut self) -> &mut Self::Vector;
}

/** Move sites and bodies.

When applied to site properties, [`Velocity`] describes the velocity of the site
relative to the origin of the body. In other words, it is the velocity of the
site in the body reference frame.

When applied to body properties [`Velocity`] describes the velocity of the body 
relative to the origin of the system coordinate system. In other words, it is
the velocity of the body's center of mass in the system reference frame.

# Units

Velocity vectors have units of *\[length/time\]*.
*/
pub trait Velocity {
    /// Every velocity is within this vector space.
    type Vector;

    /// The velocity of this body or site *\[length/time\]*.
    fn velocity(&self) -> &Self::Vector;

    /// The mutable velocity of this body or site *\[length/time\]*.
    fn velocity_mut(&mut self) -> &mut Self::Vector;
}

/** Accelerate sites and bodies.

When applied to site properties, [`Acceleration`] describes the acceleration of
the site relative to the origin of the body. In other words, it is the
acceleration of the site in the body reference frame.

When applied to body properties [`Acceleration`] describes the acceleration of
the body relative to the origin of the system coordinate system. In other words,
it is the acceleration of the body's center of mass in the system reference frame.

# Units

Acceleration vectors have units of *\[length^2/time\]*.
*/
pub trait Acceleration {
    /// Every acceleration is within this vector space.
    type Vector;

    /// The acceleration of this body or site *\[length^2/time\]*.
    fn acceleration(&self) -> &Self::Vector;

    /// The mutable acceleration of this body or site *\[length^2/time\]*.
    fn acceleration_mut(&mut self) -> &mut Self::Vector;
}

/** Rotate sites and bodies.

When applied to site properties, [`Orientation`] describes the rotation from the
site's local coordinates to the body frame.

When applied to body properties, [`Orientation`] describes the rotation from the
body frame to the system.

# Units

The units of [`Orientation`] depend on the representation chosen for `R`.
For example, [`hoomd_vector::Angle`] has units of radians while
[`hoomd_vector::Versor`] is unitless.

*/
pub trait Orientation {
    /// Type that can express the orientation of a body or site.
    type Rotation;

    /// The orientation of this body or site.
    fn orientation(&self) -> &Self::Rotation;

    /// The orientation of this body or site (mutable).
    fn orientation_mut(&mut self) -> &mut Self::Rotation;
}

/** Get the mass of sites and bodies.

[`Mass`] is the quantity which determines a site or body's linear inertia, and
which together with velocity determines a site or body's momentum.

# Units

The units of [`mass`] are *\[mass\]*.
*/
pub trait Mass {
    /// The mass of this body or site *\[mass\]*.
    fn mass(&self) -> &f64;

    // TODO: do we want to provide a mutable getter?
}

/** Get the moment of inertia of sites and bodies.

[`MomentOfInertia`] is the quantity which determines a site or body's angular
inertia, and which together with angular velocity determines a site or body's
angular momentum.

TODO: make this be a square matrix instead of a vector

# Units

The units of `[MomentOfInertia`] are *\[mass * \length^2\]*.
*/
pub trait MomentOfInertia {
    /// Every moment of inertia is within this vector space.
    type Vector;

    /// The moment of inertia of this body or site *\[mass * \length^2\]*.
    fn moment_of_inertia(&self) -> &Self::Vector;

    /// The mutable moment of inertia of this body or site *\[mass * \length^2\]*.
    fn moment_of_inertia_mut(&mut self) -> &mut Self::Vector;
}

/** Get the angular velocity of sites and bodies.

[`AngularVelocity`] is the quantity which determines a site or body's angular
velocity, and which together with moment of inertia determines a site or body's
angular momentum. [TODO: re-phrase to be more clear]

# Units

The units of `[AngularVelocity`] are *\[radian/time\]*.
*/
pub trait AngularVelocity {
    /// Every angular velocity is within this vector space.
    type Vector;
    
    /// The angular velocity of this body or site *\[radian/time\]*.
    fn angular_velocity(&self) -> &Self::Vector;

    /// The mutable angular velocity of this body or site *\[radian/time\]*.
    fn angular_velocity_mut(&mut self) -> &mut Self::Vector;
}
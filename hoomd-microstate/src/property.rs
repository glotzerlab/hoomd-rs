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

impl Orientation<Versor> for Custom {
    fn orientation(&self) -> &Versor {
        &self.orientation
    }

    fn orientation_mut(&mut self) -> &mut Versor {
        &mut self.orientation
    }
}

impl Position<Cartesian<3>> for Custom {
    fn position(&self) -> &Cartesian<3> {
        &self.position
    }

    fn position_mut(&mut self) -> &mut Cartesian<3> {
        &mut self.position
    }
}
```

## Transformations

TODO: Demonstrate transform for a custom type. Need the `OrientedPoint` type to
use as a body property first. Transformations may not be formulaic enough for a
macro to work in general.
*/

mod point;
pub use point::Point;

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
pub trait Position<V> {
    /// The position of this body or site *\[length\]*.
    fn position(&self) -> &V;

    /// The mutable position of this body or site *\[length\]*.
    fn position_mut(&mut self) -> &mut V;
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
pub trait Orientation<R> {
    /// The orientation of this body or site.
    fn orientation(&self) -> &R;

    /// The orientation of this body or site (mutable).
    fn orientation_mut(&mut self) -> &mut R;
}

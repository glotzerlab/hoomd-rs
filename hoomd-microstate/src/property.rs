// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Traits that describe body and/or site properties a a selection types that implement them.
 */

use hoomd_vector::Vector;
use crate::Transform;

/** Locate sites and bodies.

When applied to site properties, [`Position`] describes the location of the site
relative to the origin of the body. In other words, it is the position of the
site in the body reference frame.

When applied to body properties [`Position`] describes the location of the body
relative to the origin of the system coordinate system. In other words, it is
the position of the body's origin in the system reference frame.

# Units

Position vectors have units of `[length]`.

# Usage

[`Position`] is implemented for a number of built-in types, such as TODO
and others (see below). To implement [`Position`] for a custom property type,
follow this example:

```
# TODO
```
*/
pub trait Position<V> {
    /// The position of this body or site `[length]`.
    fn position(&self) -> &V;

    /// The mutable position of this body or site `[length]`.
    fn position_mut(&mut self) -> &mut V;
}

/** Rotate sites and bodies.

When applied to site properties, [`Orientation`] describes the rotation from the
site's local coordinates to the body frame.

When applied to body properties, [`Orientation`] describes the rotation from the
body frame to the system.

# Units

The units of [`Orientation`] depend on the representation chosen for `R`.
For example, [`hoomd_vector::Angle`] has units of radians while [`hoomd_vector::Quaternion`]
is unitless.

# Usage

[`Orientation`] is implemented for a number of built-in types, such as TODO
and others (see below). To implement [`Orientation`] for a custom property type,
follow this example:

```
# TODO
```
*/
pub trait Orientation<R> {
    /// The orientation of this body or site.
    fn orientation(&self) -> &R;

    /// The orientation of this body or site (mutable).
    fn orientation_mut(&mut self) -> &mut R;
}

/** A position in space and nothing more.

Use [`Point`] as a [`Body`](crate::Body) or [`Site`](crate::Site) property type.

# Example

TODO
*/
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point<V> {
    /// The location of the point in space.
    pub position: V,
}

impl<V> Point<V> {

    /** Construct a new point at the given position.

    # Example

    ```
    # TODO
    ```
    */
    #[inline]
    #[must_use]
    pub fn new(position: V) -> Self {
        Self { position }
    }
}   

/** Move [`Point`] properties from the local body frame to the system frame.
*/
impl<V> Transform<Point<V>> for Point<V> where
V: Vector
{
    /** Points transform by vector addition.

    TODO: Math
    TODO: Example
    */
    #[inline]
    fn transform(&self, site_properties: &Point<V>) -> Point<V> {
        Point { position: self.position + site_properties.position }
    }
}

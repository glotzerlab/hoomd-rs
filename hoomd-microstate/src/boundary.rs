// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Traits that describe boundary conditions and a selection of types that implement them.

See the [crate-level documentation](crate) for an overview of how [`Boundary`]
interacts with [`Microstate`](crate::Microstate) and model methods.
 */

use crate::{Error, property::Position};

/** Define the subset of the vector space where body and site positions exist.

A [`Boundary`] also describes how body and site properties transform from/to
periodic images. See the specific implementations of [`Boundary`] for examples.

When implementing a custom [`Boundary`], you need to implement
[`Boundary::is_inside`] at a minimum. The default implementations of the other
methods describe fully non-periodic boundary conditions. You can make your
boundary periodic by implementing the other methods accordingly.
*/
pub trait Boundary<V> {
    /// Test whether a given point is inside the boundary.
    fn is_inside(&self, point: &V) -> bool;

    /** Transform body and site properties into the boundary.

    `wrap` takes a body/site with a position that may be outside the boundary.
    It attempts to wrap that body/site back inside following the boundary's
    periodicity. `wrap` returns [`Ok(properties)`](Ok) when this process is
    successful.

    # Errors

    `wrap` returns [`Error::CannotWrapPosition`] when it is not possible to wrap
    the body/site into the boundary. For example, when the position is outside
    the radius of a cylinder that is only periodic along its axis.
    */
    #[inline]
    fn wrap<P>(&self, properties: P) -> Result<P, Error>
    where
        P: Position<V>,
    {
        if self.is_inside(properties.position()) {
            Ok(properties)
        } else {
            Err(Error::CannotWrapPosition)
        }
    }
}

/** Allow bodies and sites to exist anywhere in space.

Every point lies inside `Open` boundary conditions.
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Open;

impl<V> Boundary<V> for Open {
    #[inline]
    fn is_inside(&self, _point: &V) -> bool {
        true
    }
}

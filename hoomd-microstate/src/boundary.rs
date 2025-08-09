// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Traits that describe boundary conditions and a selection of types that implement them.

See the [crate-level documentation](crate) for an overview of how [`Boundary`]
interacts with [`Microstate`](crate::Microstate) and model methods.
 */

use crate::property::Position;

use thiserror::Error;

mod open;
mod square;

pub use open::Open;
pub use square::Square;

/// Enumerate possible sources of error in fallible boundary methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Failed to wrap body properties.
    #[error("body property cannot be wrapped")]
    CannotWrapBodyProperties,

    /// Failed to wrap site properties.
    #[error("site property cannot be wrapped")]
    CannotWrapSiteProperties,
}

/** Define the subset of the vector space where body and site positions exist.

A [`Boundary`] also describes how body and site properties transform from/to
periodic images. See the specific implementations of [`Boundary`] for examples.

When implementing a custom [`Boundary`], you need to implement
[`Boundary::is_inside`] at a minimum. The default implementations of the other
methods describe fully non-periodic boundary conditions. You can make your
boundary periodic by implementing the other methods accordingly.

The generic type names are:
* `M`: The [`Metric`](hoomd_vector::Metriic) space in which bodies and sites exist.
* `B`: The [`Body::properties`](crate::Body) type.
* `S`: The [`Site::properties`](crate::Site) type.
*/
pub trait Boundary<M, B, S> {
    /// Test whether a given point is inside the boundary.
    fn is_inside(&self, point: &M) -> bool;

    /** Transform body properties into the boundary.

    `wrap_body` takes a body with a position that may be outside the boundary.
    It attempts to wrap that body back inside following the boundary's
    periodicity. `wrap` returns [`Ok(properties)`](Ok) when this process is
    successful.

    # Errors

    `wrap` returns [`Error::CannotWrapBodyProperties`] when it is not possible
    to wrap the body into the boundary. For example, when the position is
    outside the radius of a cylinder that is only periodic along its axis.
    */
    #[inline]
    fn wrap_body(&self, body_properties: B) -> Result<B, Error>
    where
        B: Position<Metric = M>,
    {
        if self.is_inside(body_properties.position()) {
            Ok(body_properties)
        } else {
            Err(Error::CannotWrapBodyProperties)
        }
    }

    /** Transform site properties into the boundary.

    `wrap_site` takes a site with a position that may be outside the boundary.
    It attempts to wrap that site back inside following the boundary's
    periodicity. `wrap` returns [`Ok(properties)`](Ok) when this process is
    successful.

    # Errors

    `wrap` returns [`Error::CannotWrapSiteProperties`] when it is not possible
    to wrap the site into the boundary. For example, when the position is
    outside the radius of a cylinder that is only periodic along its axis.
    */
    #[inline]
    fn wrap_site(&self, site_properties: S) -> Result<S, Error>
    where
        S: Position<Metric = M>,
    {
        if self.is_inside(site_properties.position()) {
            Ok(site_properties)
        } else {
            Err(Error::CannotWrapSiteProperties)
        }
    }

    // NOTE: One might think to make wrap<> generic on a property type.
    // That is not possible because types that implement this trait *must*
    // use the same bounds -- preventing more generic boundary conditions.
}

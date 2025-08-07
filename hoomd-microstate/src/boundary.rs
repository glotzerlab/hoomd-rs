// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Traits that describe boundary conditions and a selection of types that implement them.

See the [crate-level documentation](crate) for an overview of how boundary
conditions interact with [`Microstate`](crate::Microstate) and model methods.
 */

use thiserror::Error;
use tinyvec::ArrayVec;

mod closed;
mod open;

pub use closed::Closed;
pub use open::Open;

/// Enumerate possible sources of error in fallible boundary methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Failed to wrap body or site properties.
    #[error("property cannot be wrapped")]
    CannotWrapProperties,
}

/// The maximum number of possible ghosts.
const MAX_GHOSTS: usize = 8;

// Ideally, MAX_GHOSTS would be associated with the boundary type, but that is
// not currently possible in Rust.

/** Define the subset of the vector space where body and site positions exist.

A [`Boundary`] also describes how body and site properties transform from/to
periodic images. See the specific implementations of [`Boundary`] for examples.

When implementing a custom [`Boundary`], you need to implement
[`Boundary::is_inside`] at a minimum. The default implementations of the other
methods describe fully non-periodic boundary conditions. You can make your
boundary periodic by implementing the other methods accordingly.

The generic type names are:
* `V`: The [`Vector`](hoomd_vector::Vector) space in which bodies and sites exist.
* `P`: The [`Body::properties`](crate::Body) or [`Site::properties`](crate::Site) type.
*/
pub trait Wrap<P> {
    /** Transform body/point properties into the boundary.

    `wrap_properties` takes a body or site properties with a position that
    may be outside the boundary. It attempts to wrap that position back inside
    following the boundary's periodicity. `wrap` returns [`Ok(properties)`](Ok)
    when this process is successful.

    # Errors

    `wrap` returns [`Error::CannotWrapProperties`] when it is not possible
    to wrap the body into the boundary. For example, when the position is
    outside the radius of a cylinder that is only periodic along its axis.
    */
    fn wrap(&self, properties: P) -> Result<P, Error>;
}

pub trait GenerateGhosts<S> {
    /** The largest interaction distance between sites.

    The maximum interaction range is the largest distance between two
    interacting sites. [`Microstate`](crate::Microstate) will place ghosts
    within this range outside periodic boundaries.
    */
    fn maximum_interaction_range(&self) -> f64;

    /** Place periodic images of sites within the interaction range.

    Given `site_properties` inside the boundary, `generate_ghosts` places
    periodic images of that site. It must place all ghosts needed to compute
    interactions with other sites in the given [`maximum_interaction_range`].

    [`maximum_interaction_range`]: Self::maximum_interaction_range
    */
    fn generate_ghosts(&self, _site_properties: &S) -> ArrayVec<[S; MAX_GHOSTS]>;
}

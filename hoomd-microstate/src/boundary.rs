// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Traits that describe boundary conditions and a selection of types that implement them.
//!
//! See the [crate-level documentation](crate) for an overview of how boundary
//! conditions interact with [`Microstate`](crate::Microstate) and model methods.
//!
//! hoomd-rs* provides the boundary types [`Open`], [`Closed`], and [`Periodic`].
//! * [`Open`] boundaries allow bodies and sites to exist anywhere in space.
//! * [`Closed`] boundaries limit bodies and sites to the inside of a shape and
//!   are not periodic in any direction.
//! * [`Periodic`] boundaries limit bodies and sites to the inside of a shape,
//!   wrap particles anywhere outside that shape back inside, and place ghosts
//!   following the periodic tiling of the shape.
//!
//! The documentation of [`Closed`] and [`Periodic`] describes the shapes that
//! they each implement. If the shape you want is not supported, you can write
//! a custom shape type and implement `IsInside` so that it will work with
//! [`Closed`]. To implement a custom periodic boundary, create your custom
//! type and implement both [`Wrap`] and [`GenerateGhosts`] for it.

use thiserror::Error;
use arrayvec::ArrayVec;

mod closed;
mod open;
mod periodic;

pub use closed::Closed;
pub use open::Open;
pub use periodic::Periodic;

/// Enumerate possible sources of error in fallible boundary methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Failed to wrap body or site properties.
    #[error("property cannot be wrapped")]
    CannotWrapProperties,
    /// The maximum interaction range is larger than the periodic boundary condition will allow.
    #[error("the requested interaction range ({0}) is larger than the boundary will allow ({1})")]
    InteractionRangeTooLarge(f64, f64),
}

/// The maximum number of possible ghosts.
pub(crate) const MAX_GHOSTS: usize = 8;

// Ideally, MAX_GHOSTS would be associated with the boundary type, but that is
// not currently possible in Rust.

/// Attempt to move any body/site properties back into the simulation boundary.
///
/// [`Wrap`] and [`GenerateGhosts`] together define the behavior of simulation
/// boundary conditions.
///
/// The **boundary** defines the subset of points where bodies and sites are
/// allowed. The [`wrap`] method takes a given body or site properties that
/// is anywhere in space and attempts to wrap it back into the boundary. This
/// process succeeds when the boundary is periodic and fails when it is not
/// (some boundaries may be periodic in some directions and not in others).
///
/// The generic type name is:
/// * `P`: The [`Body::properties`](crate::Body) or [`Site::properties`](crate::Site) type.
///
/// [`wrap`]: Self::wrap
pub trait Wrap<P> {
    /// Transform body/point properties into the boundary.
    ///
    /// `wrap` takes a body or site properties with a position that may be outside
    /// the boundary. It attempts to wrap that position back inside following the
    /// boundary's periodicity. `wrap` returns [`Ok(properties)`](Ok) when this
    /// process is successful.
    ///
    /// # Errors
    ///
    /// `wrap` returns [`Error::CannotWrapProperties`] when it is not possible
    /// to wrap the body into the boundary. For example, when the position is
    /// outside the radius of a cylinder that is only periodic along its axis.
    fn wrap(&self, properties: P) -> Result<P, Error>;
}

/// Place periodic images of sites within the interaction range.
///
/// [`Wrap`] and [`GenerateGhosts`] together define the behavior of simulation
/// boundary conditions.
///
/// The **boundary** defines the subset of points where bodies and sites are
/// allowed. [`generate_ghosts`] places 0 or more sites that are periodic images
/// of the given site **and** within the maximum interaction range of the boundary.
/// [`Closed`] boundary conditions place no ghosts. [`Periodic`] boundary conditions
/// may place 0, 1, 2, or more ghosts depending on the location of the site. For
/// example sites in the center of a cubic box will have 0 ghosts, those near the
/// center of a face will have 1, those near an edge will have 2, and those near a
/// vertex will have 3.
///
/// To avoid costly dynamic memory allocations, [`generate_ghosts`] returns an
/// array-backed storage with a hard-coded maximum size of `MAX_GHOSTS`.
///
/// [`generate_ghosts`]: Self::generate_ghosts
pub trait GenerateGhosts<S> {
    /// The largest interaction distance between sites.
    ///
    /// The maximum interaction range is the largest distance between two
    /// interacting sites. [`Microstate`](crate::Microstate) will place ghosts
    /// within this range outside periodic boundaries.
    fn maximum_interaction_range(&self) -> f64;

    /// Place periodic images of sites within the interaction range.
    ///
    /// Given `site_properties` inside the boundary, `generate_ghosts` places
    /// periodic images of that site. It must place all ghosts needed to compute
    /// interactions with other sites in the given [`maximum_interaction_range`].
    ///
    /// [`maximum_interaction_range`]: Self::maximum_interaction_range
    fn generate_ghosts(&self, _site_properties: &S) -> ArrayVec<S, MAX_GHOSTS>;
}

/// Compute the largest value of the maximum interaction range.
///
/// In *hoomd-rs*, sites interact only with the *minimum images* of other sites.
/// In periodic boundary conditions, there is a maximum allowable interaction
/// range beyond which sites would start interacting with multiple images. The
/// [`MaximumAllowableInteractionRange`] trait computes that distance for a given
/// shape.
///
/// [`Periodic`] uses [`MaximumAllowableInteractionRange`] to trigger an error when
/// the caller requires an interaction range larger than is possible.
pub trait MaximumAllowableInteractionRange {
    /// The largest value that the maximum interaction range can take.
    fn maximum_allowable_interaction_range(&self) -> f64;
}

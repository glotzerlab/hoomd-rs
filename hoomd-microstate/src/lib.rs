// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Store and manage the simulation state.

   TODO: Expand documentation.
*/

pub mod properties;
mod microstate;

pub use microstate::Microstate;

/** Interactions in `hoomd-rs` apply between sites.

A [`Site`] (often called an *atom* or a *particle* in other codes) has a `tag`
that uniquely identities it in the [`Microstate`] and is associated with a given
`body` (see [`Body`]). All interactions in `hoomd-rs` occur between sites
as a function of their `properties`. At a minimum, [`Microstate`] assumes that
`properties` implements [`Position`]. The `properties` type is generic so that
users can build custom types that store orientation, charge, mass, color, or
whatever other fields are needed to implement their model.

Add sites to the [`Microstate`] as members of bodies ([`Body`]).
*/
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Site<S> {
    /// Every site in a [`Microstate`] has a unique value in its `tag`.
    pub tag: u32,
    /// `body` stores the body tag of the [`Body`] associated with this site.
    pub body: u32,
    /// The properties of the site (for example, position, orientation).
    pub properties: S,
}

/** A collection of interaction sites with that can be placed in a [`Microstate`].

The [`Body`] `properties` have a generic type that includes all the body's
degrees of freedom and any other fields needed to implement the user's model.
Bodies interact indirectly via one or more `sites`. The `sites` vector stores
the properties of the body's sites in the body frame. The body `properties`
stores the body's degrees of freedom (such as position and orientation) in the
system frame. The [`Transform`] describes how a given body transforms its sites
from the body frame to the system frame.

In typical cases, such as those implemented in `hoomd-rs`, [`Body`] describes
a rigid collection of sites that transform together. However, creative
implementations of [`Transform`] could achieve other behaviors.
*/
#[derive(Clone, Debug, PartialEq)]
pub struct Body<B, S> {
    pub properties: B,
    pub sites: Vec<S>,
}

/** Take [`Site`] properties in the body frame into the system frame.
*/
pub trait Transform<S> {
    /** Transform site properties.

    Given `site_properties` in the body frame, `transform` returns the
    corresponding site properties in the system frame.
    */
    #[must_use]
    fn transform(&self, site_properties: &S) -> S;
}

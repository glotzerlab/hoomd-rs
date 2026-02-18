// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

//! Store and manage the simulation state.
//!
//! # Microstate
//!
//! [`Microstate`] facilitates simulations of particle systems. It is the central
//! data structure used by MC, MD, and supporting calculations implemented in
//! `hoomd-rs`. [`Microstate`] holds a set of bodies that exist in a space defined
//! by the boundary conditions. The degrees of freedom consist of the properties of
//! the bodies and the parameters of the boundary conditions.
//!
//! [`Microstate`] also stores many auxiliary data structures and implements
//! convenience methods to facilitate the efficient implementation of simulation
//! models. These include a set of all interaction sites in the system frame
//! of reference, ghost sites near the periodic boundaries, and spatial data
//! structures.
//!
//! ## Step, substep and seed
//!
//! [`Microstate`] tracks the current simulation step ([`Microstate::step`]),
//! substep ([`Microstate::substep`]), and seed ([`Microstate::seed`])
//! that allow models to generate uncorrelated random number streams via
//! [`Microstate::counter`].
//!
//! The **step** is the current step in the simulation as defined by the user.
//! For example, a single step could be one MD timestep or the application
//! of a set of MC moves. Users should call [`Microstate::increment_step`]
//! after each step in the model is completed. The **substep** is a running
//! counter tracking the number of operations called so far during the current
//! step. Model methods in *hoomd-rs* (such as the MC trial move `apply`) call
//! [`increment_substep`](Microstate::increment_substep) internally. Users should
//! call it at the end of any custom methods that implement new substeps.
//! A failure to call [`increment_substep`](Microstate::increment_substep)
//! will cause the reuse of the same random numbers from one substep to the next!**
//!
//! The **seed** allows users to select independent random number streams for
//! simulations that would otherwise be identical. It may only be set once on
//! creation by [`MicrostateBuilder::seed`].
//!
//! ## Bodies and sites
//!
//! [`Microstate`] differentiates between the degrees of freedom of the system and
//! points where interactions take place. Each [`Body`] in the microstate has one
//! or more interaction sites ([`Site`]). The bodies have degrees of freedom that
//! are evolved by the simulation model, which defines how bodies interact through
//! their sites. The properties of the sites *in the system frame* are a function
//! of the body's properties and the site properties *in the body frame*. In a
//! particle-only simulation model each body has one site at the origin in the body
//! frame. In a simulation of squares, each body might be made up of four sites on
//! the vertices. In both cases, the net force on the body is the sum of the forces
//! applied to all of its sites.
//!
//! In [`Microstate`], the body properties (the generic type `B` throughout) and the
//! site properties (the generic type `S`) do not need to be the same. For example,
//! a body might have mass, position and velocity while that body's sites have
//! position and type.
//!
//! The `property` module provides a number of property types. It also defines
//! traits that you can use to implement custom property types. At a minimum, *both
//! `B` and `S` MUST implement [`property::Position`]* so that [`Microstate`] can
//! place your body's and sites inside the boundary conditions and maintain spatial
//! data structures. Some model methods (such as shape overlap energies) will
//! require other traits (such as [`property::Orientation`]). The `property` module
//! documentation provides more details on using the types it provides and how to
//! define custom types.
//!
//! [`Microstate::add_body`] and [`Microstate::extend_bodies`] add new bodies (and
//! their sites) to the microstate. Similarly, [`Microstate::remove_body`] removes a
//! body (and all associated sites). [`Microstate::update_body_properties`] modifies
//! the properties of a given body and correspondingly the properties of all sites
//! associated with that body. All these methods have a cost proportional to the
//! number of sites in the body.
//!
//! [`Microstate::bodies`] provides direct (immutable) access to the bodies in
//! the microstate, including their properties and sites in the body frame. While
//! some algorithms may find this useful, many model algorithms instead operate
//! on site properties in the system frame. [`Microstate::sites`] provides direct
//! (immutable) access to a slice of all sites in the system frame for this
//! purpose. Any method that adds, removes, or changes a body immediately updates
//! [`Microstate::sites`] accordingly. [`Microstate::iter_body_sites`] iterates over
//! all the sites (in the system frame) associated with a given body.
//!
//! ## Tags
//!
//! The elements of [`Microstate::bodies`] and [`Microstate::sites`] are stored in
//! **no particular order** to allow efficient addition and removal of bodies and
//! for the possibility sorting to improve cache coherency. Callers are welcome
//! to iterate over these data structures when computing order-independent overall
//! properties. However, a caller should never maintain indices into these vectors.
//! Instead, the caller should store the appropriate **tag** when it needs to
//! persistently refer to a specific body or site.
//!
//! Elements in [`Microstate::bodies`] have the type [`Tagged<Body>`].
//! The [`item`](Tagged::item) field holds the body itself while the
//! [`tag`](Tagged::tag) field is a unique identifier that identifies this
//! specific body. The tag will remain the same even when [`Microstate::bodies`] is
//! reordered. Use [`Microstate::body_indices`] to find the current index of a body
//! with a given tag.
//!
//! Elements in [`Microstate::sites`] have the type [`Site<S>`]. As with bodies,
//! each site has a unique [`site_tag`](Site::site_tag) that remains the same even
//! when sites are reordered. Each site also has a [`body_tag`](Site::body_tag) that
//! identifies which body the site is part of. Use [`Microstate::site_indices`]
//! to find the current index of a site with a given site tag and
//! [`Microstate::iter_body_sites`] to find all the sites associated with a given
//! body *index*.
//!
//! ## Boundary conditions
//!
//! The positions of all bodies and all sites **must** be inside the microstate's
//! boundary at all times. Periodic boundaries can wrap positions outside to a
//! corresponding point on the inside. When a boundary is aperiodic (or partially
//! aperiodic), the wrapping process may fail. MC models reject trial moves that
//! cannot be wrapped. MD models fail with an error should bodies or sites move in a
//! way that cannot be wrapped.
//!
//! [`Microstate`] is generic on the type of boundary condition. The [`boundary`]
//! module implements standard types and explains how you can provide custom
//! implementations.
//!
//! ## Ghost sites
//!
//! Periodic boundary conditions place **ghost sites** within a given **maximum
//! interaction range** outside the boundary. These ghost sites are images of real
//! sites that are inside the boundary. Access all of the ghosts with the
//! [`ghosts`] method. [`iter_sites_near`] will find both primary and ghost sites
//! as it searches for sites near the requested point.
//!
//! When using [`Open`] or [`Closed`] boundary conditions, [`ghosts`] will always
//! be empty.
//!
//! [`ghosts`]: Microstate::ghosts
//! [`iter_sites_near`]: Microstate::iter_sites_near
//! [`Open`]: crate::boundary::Open
//! [`Closed`]: crate::boundary::Closed
//!
//! ## Spatial searches
//!
//! TODO: Implement spatial search, then document.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod boundary;
mod microstate;
pub mod property;

pub use microstate::{Microstate, MicrostateBuilder, SiteKey, Tagged};
use property::Point;

/// Interactions in `hoomd-rs` apply between sites.
///
/// A [`Site`] (often called an *atom* or a *particle* in other codes) has a
/// `tag` that uniquely identities it in the [`Microstate`] and is associated
/// with a given `body` (see [`Body`]). All interactions in `hoomd-rs` occur
/// on or between sites as a function of their `properties` which has the
/// generic type `S`. At a minimum, [`Microstate`] assumes that `S` implements
/// [`Position`](property::Position). `S` is generic so that users can build custom
/// types that store orientation, charge, mass, color, or whatever other fields are
/// needed to implement their model.
///
/// Add sites to the [`Microstate`] as members of bodies ([`Body`]).
///
/// # Example
///
/// Find the center of all interaction sites in a [`Microstate`]:
/// ```
/// use hoomd_microstate::{Body, Microstate, MicrostateBuilder};
/// use hoomd_vector::{Cartesian, Vector};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let microstate = Microstate::builder()
///     .bodies([
///         Body::point(Cartesian::from([1.0, 0.0])),
///         Body::point(Cartesian::from([-1.0, 2.0])),
///     ])
///     .try_build()?;
///
/// let average_site_position = microstate
///     .sites()
///     .iter()
///     .map(|site| site.properties.position)
///     .sum::<Cartesian<2>>()
///     / (microstate.sites().len() as f64);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Site<S> {
    /// Every site in a [`Microstate`] has a unique value in `site_tag`.
    pub site_tag: usize,
    /// The body tag of the [`Body`] associated with this site.
    pub body_tag: usize,
    /// The properties of the site.
    pub properties: S,
}

/// A collection of interaction sites that can be placed in a [`Microstate`].
///
/// The [`Body`] `properties` have a generic type that includes all the body's
/// degrees of freedom and any other fields needed to implement the user's
/// model. Bodies interact indirectly via one or more `sites`. The `sites` vector
/// stores the properties of the body's sites in the body frame. The body field
/// `properties` stores the body's degrees of freedom (such as position and
/// orientation) in the system frame. [`Transform`] describes how a given body
/// transforms its sites from the body frame to the system frame.
///
/// In typical cases, such as those implemented in `hoomd-rs`, [`Body`] describes
/// a rigid collection of sites that transform together. However, creative
/// implementations of [`Transform`] could achieve other behaviors.
///
/// Use the properties defined in [`property`] to construct bodies that meet
/// the needs of your model.
///
/// # Examples
///
/// Construct body with a single interaction site at one point:
/// ```
/// use hoomd_microstate::Body;
/// use hoomd_vector::Cartesian;
///
/// let body = Body::point(Cartesian::from([-3.0, 5.0]));
/// ```
///
/// Construct an oriented body:
/// ```
/// use hoomd_microstate::{Body, property::OrientedPoint};
/// use hoomd_vector::{Angle, Cartesian};
///
/// let body_properties = OrientedPoint {
///     position: Cartesian::from([1.0, -3.0]),
///     orientation: Angle::from(1.2),
/// };
/// let site_properties = OrientedPoint {
///     position: Cartesian::<2>::default(),
///     orientation: Angle::default(),
/// };
///
/// let body = Body {
///     properties: body_properties,
///     sites: vec![site_properties],
/// };
/// ```
//
/// Construct a rigid body with several point sites:
/// ```
/// use hoomd_microstate::{
///     Body,
///     property::{OrientedPoint, Point},
/// };
/// use hoomd_vector::{Angle, Cartesian};
///
/// let body_properties = OrientedPoint {
///     position: Cartesian::from([1.0, -3.0]),
///     orientation: Angle::from(1.2),
/// };
///
/// let body = Body {
///     properties: body_properties,
///     sites: vec![
///         Point::new(Cartesian::from([0.0, -1.0])),
///         Point::new(Cartesian::from([0.0, 0.0])),
///         Point::new(Cartesian::from([0.0, 1.0])),
///     ],
/// };
/// ```
///
/// # Custom body and site properties
///
/// The [`property`] module documentation shows you how to define custom body
/// and site property types.
#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Body<B, S = B> {
    /// The body's degrees of freedom.
    pub properties: B,
    /// Interaction sites in the body's frame of reference.
    pub sites: Vec<S>,
}

impl<B, S> Clone for Body<B, S>
where
    B: Clone,
    S: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            properties: self.properties.clone(),
            sites: self.sites.clone(),
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        // `Sweep` and other methods use clone_from to efficiently generate
        // trial moves while minimizing memory copies. #[derive(Clone)] does
        // not implement `clone_from`, so it must be done manually.
        self.properties.clone_from(&source.properties);
        self.sites.clone_from(&source.sites);
    }
}

impl<V> Body<Point<V>, Point<V>> {
    /// Construct a point particle.
    ///
    /// A point particle is a [`Body`] with a single interaction site at the body's
    /// origin. The body and site property types are identical and have only a
    /// `position` field. Use point particles for simulations of monodisperse hard
    /// spheres, identical particles with pairwise interactions, or any time you
    /// need a [`Microstate`] that consists only of point particles.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::Body;
    /// use hoomd_vector::Cartesian;
    ///
    /// let body = Body::point(Cartesian::from([-3.0, 5.0]));
    /// assert_eq!(body.properties.position, [-3.0, 5.0].into());
    /// assert_eq!(body.sites.len(), 1);
    /// assert_eq!(body.sites[0].position, [0.0, 0.0].into());
    /// ```
    #[inline]
    #[must_use]
    pub fn point(position: V) -> Self
    where
        V: Default,
    {
        Self {
            properties: Point::new(position),
            sites: vec![Point::default()],
        }
    }
}

// TODO: Consider adding a shortcut here for dynamics point.

/// Take [`Site`] properties in the body frame into the system frame.
///
/// See the [`property`] module-level documentation for an example
/// that implements [`Transform`] for a custom type.
pub trait Transform<S> {
    /// Transform site properties.
    ///
    /// Given `site_properties` in the body frame, `transform` returns the
    /// corresponding site properties in the system frame relative to the
    /// body properties in `&self`.
    #[must_use]
    fn transform(&self, site_properties: &S) -> S;
}

/// Enumerate possible sources of error in fallible microstate methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Failed to add a body to a [`Microstate`].
    #[error("failed to add body (tag={0})")]
    AddBody(usize, #[source] boundary::Error),

    /// Failed to update a body in a [`Microstate`].
    #[error("failed to update body (tag={0})")]
    UpdateBody(usize, #[source] boundary::Error),
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Store and manage the simulation state.

# Microstate

[`Microstate`] facilitates simulations of particle systems. It is the central
data structure used by MC, MD, and supporting calculations implemented in
`hoomd-rs`. [`Microstate`] holds a set of bodies that exist in a space defined
by the boundary conditions. The degrees of freedom consist of the properties of
the bodies and the parameters of the boundary conditions.

[`Microstate`] also stores many auxiliary data structures and implements
convenience methods to facilitate the efficient implementation of simulation
models. These include a set of all interaction sites in the system frame
of reference, ghost sites near the periodic boundaries, and spatial data
structures.

## Step, substep and seed

[`Microstate`] tracks the current simulation step ([`Microstate::step`]),
substep ([`Microstate::substep`]), and seed ([`Microstate::seed`])
that allow models to generate uncorrelated random number streams via
[`Microstate::counter`].

The **step** is the current step in the simulation as defined by the user.
For example, a single step could be one MD timestep or the application
of a set of MC moves. Users should call [`Microstate::increment_step`]
after each step in the model is completed. The **substep** is a running
counter tracking the number of operations called so far during the current
step. Model methods in *hoomd-rs* (such as the MC trial move `apply`) call
[`increment_substep`](Microstate::increment_substep) internally. Users should
call it at the end of any custom methods that implement new substeps.
**A failure to call [`increment_substep`](Microstate::increment_substep)
will cause the reuse of the same random numbers from one substep to the next!**

The **seed** allows users to select independent random number streams for
simulations that would otherwise be identical. It may only be set once on
creation by [`MicrostateBuilder::seed`].

## Bodies and sites

[`Microstate`] differentiates between the degrees of freedom of the system and
points where interactions take place. Each [`Body`] in the microstate has one
or more interaction sites ([`Site`]). The bodies have degrees of freedom that
are evolved by the simulation model, which defines how bodies interact through
their sites. The properties of the sites *in the system frame* are a function
of the body's properties and the site properties *in the body frame*. In a
particle-only simulation model each body has one site at the origin in the body
frame. In a simulation of squares, each body might be made up of four sites on
the vertices. In both cases, the net force on the body is the sum of the forces
applied to all of its sites.

In [`Microstate`], the body properties (the generic type `B` throughout) and the
site properties (the generic type `S`) do not need to be the same. For example,
a body might have mass, position and velocity while that body's sites have
position and type.

The `property` module contains a number of ready-to-use property types. It also
defines traits that you can use to implement custom property types. At a
minimum, *both `B` and `S` MUST implement [`property::Position`]* so that
[`Microstate`] can place your bodys and sites inside the boundary conditions
and maintain spatial data structures. Some model methods (such as shap overlap
energies) will require other traits (such as [`property::Orientation`]).
The `property` module documentation provides more details on using the
types it provides and how to define custom types.

TODO: Add/remove/update
TODO: Access sites, iter_body_sites

## On tags

## Boundary conditions

## Ghost sites

## Spatial searches

## Constructing [`Microstate`]
*/

pub mod boundary;
mod microstate;
pub mod property;

pub use microstate::{Microstate, MicrostateBuilder, Tagged};

use property::Point;

/** Interactions in `hoomd-rs` apply between sites.

A [`Site`] (often called an *atom* or a *particle* in other codes) has a `tag`
that uniquely identities it in the [`Microstate`] and is associated with a
given `body` (see [`Body`]). All interactions in `hoomd-rs` occur between sites
as a function of their `properties`. At a minimum, [`Microstate`] assumes that
`properties` implements [`Position`](property::Position). The `properties` type
is generic so that users can build custom types that store orientation, charge,
mass, color, or whatever other fields are needed to implement their model.

Add sites to the [`Microstate`] as members of bodies ([`Body`]).

# Example

Find the center of all interaction sites in a [`Microstate`]:
```
use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
use hoomd_vector::{Vector, Cartesian};

let microstate = MicrostateBuilder::new()
    .bodies([Body::point(Cartesian::from([1.0, 0.0])),
             Body::point(Cartesian::from([-1.0, 2.0]))])
    .build();

let average_site_position = microstate.sites()
    .iter()
    .map(|site| site.properties.position)
    .sum::<Cartesian<2>>() / (microstate.sites().len() as f64);
```
*/
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Site<S> {
    /// Every site in a [`Microstate`] has a unique value in `site_tag`.
    pub site_tag: usize,
    /// The body tag of the [`Body`] associated with this site.
    pub body_tag: usize,
    /// The properties of the site.
    pub properties: S,
}

/** A collection of interaction sites that can be placed in a [`Microstate`].

The [`Body`] `properties` have a generic type that includes all the body's
degrees of freedom and any other fields needed to implement the user's
model. Bodies interact indirectly via one or more `sites`. The `sites` vector
stores the properties of the body's sites in the body frame. The body field
`properties` stores the body's degrees of freedom (such as position and
orientation) in the system frame. [`Transform`] describes how a given body
transforms its sites from the body frame to the system frame.

In typical cases, such as those implemented in `hoomd-rs`, [`Body`] describes
a rigid collection of sites that transform together. However, creative
implementations of [`Transform`] could achieve other behaviors.

Use the properties defined in [`property`] to construct bodies that meet
the needs of your model.

# Example

Construct body with a single interaction site at one point:
```
use hoomd_microstate::Body;
use hoomd_vector::Cartesian;

let body = Body::point(Cartesian::from([-3.0, 5.0]));
```

TODO: Construct a body with an oriented point.

# Custom body and site properties

The `property` module documentation shows you how to define custom body
and site property types.
*/
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Body<B, S> {
    /// The body's degrees of freedom.
    pub properties: B,
    /// Interaction sites in the body's frame of reference.
    pub sites: Vec<S>,
}

impl<V> Body<Point<V>, Point<V>> {
    /** Construct a point particle.

    A point particle is a [`Body`] with a single interaction site at the body's
    origin. The body and site property types are identical and have only a
    `position` field. Use point particles for simulations of monodisperse hard
    spheres, identical particles with pairwise interactions, or any time you
    need a [`Microstate`] that consists only of point particles.

    # Example

    ```
    use hoomd_microstate::Body;
    use hoomd_vector::Cartesian;

    let body = Body::point(Cartesian::from([-3.0, 5.0]));
    assert_eq!(body.properties.position, [-3.0, 5.0].into());
    assert_eq!(body.sites.len(), 1);
    assert_eq!(body.sites[0].position, [0.0, 0.0].into());
    ```
    */
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

/** Take [`Site`] properties in the body frame into the system frame.
*/
pub trait Transform<S> {
    /** Transform site properties.

    Given `site_properties` in the body frame, `transform` returns the
    corresponding site properties in the system frame relative to the
    body properties in `&self`.
    */
    #[must_use]
    fn transform(&self, site_properties: &S) -> S;
}

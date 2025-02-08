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

pub mod particle;

/** Properties common to all particles.

Every particle in a [`Microstate`] has a position vector that locates the
particle in space.

Every [`Particle`] type must implement [`Copy`] to ensure that it can be
efficiently copied.
*/
pub trait Particle<V>: Copy {
    /// The position of this particle `[length]`.
    fn position(&self) -> &V;

    /// The position of this particle (mutable).
    fn position_mut(&mut self) -> &mut V;
}

/** Particles that have an orientation.

A particle's `orientation` is a rotation that transforms vectors from the
local coordinate frame of the [`Particle`] to the global frame of the
[`Microstate`].
*/
pub trait Orientable<V, R> : Particle<V>
{
    /// The orientation of this particle.
    fn orientation(&self) -> &R;

    /// The orientation of this particle (mutable).
    fn orientation_mut(&mut self) -> &mut R;
}


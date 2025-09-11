// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Interact with simulations at a high level.

`hoomd-simulation` defines the [`Simulation`] trait. Implement it for a type
that holds the simulation parameters, types that implement the simulation
model, and the microstate (or microstates) that it operates on.

See the tutorials in the documentation for numerous examples.
*/

pub mod macrostate;

/** Store parameters, the model and the microstate(s) it acts on.

A [`Simulation`] type stores the microstate, all model actors, and any
macrostate parameters in fields. A given [`Simulation`] can be advanced
forward one step at a time via the `advance` method.
*/
pub trait Simulation {
    /** Advance the simulation forward one step.

    # Errors

    When an error occurs, return an `Err` with any type that implements
    [`Error`]. The caller will catch the error and
    display it when exiting.

    [`Error`]: std::error::Error
    */
    fn advance(&mut self) -> anyhow::Result<()>;

    /// Get the simulation step.
    fn step(&self) -> u64;
}

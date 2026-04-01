// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]

//! Interact with simulations.
//!
//! ## Simulation
//!
//! `hoomd-simulation` defines the [`Simulation`] trait. Implement it for a type
//! that holds the simulation parameters, types that implement the simulation
//! model, and the microstate (or microstates) that it operates on.
//!
//! See the tutorials in the documentation for numerous examples.
//!
//! ## Macrostate
//!
//! The [`macrostate`] module provides types that set the parameters of the
//! simulation. For example, set the temperature of the simulation with:
//! ```
//! use hoomd_simulation::macrostate::Isothermal;
//!
//! let macrostate = Isothermal { temperature: 1.2 };
//! ```
//!
//! # Complete documentation
//!
//! `hoomd-simulation` is is a part of *hoomd-rs*. Read the [complete documentation]
//! for more information.
//!
//! [complete documentation]: https://hoomd-rs.readthedocs.io

pub mod macrostate;

/// Store parameters, the model and the microstate(s) it acts on.
///
/// A [`Simulation`] type stores the microstate, all model actors, and any
/// macrostate parameters in fields. A given [`Simulation`] can be advanced
/// forward one step at a time via the `advance` method.
pub trait Simulation {
    /// Advance the simulation forward one step.
    ///
    /// # Errors
    ///
    /// When an error occurs, return an `Err` with any type that implements
    /// [`Error`]. The caller will catch the error and
    /// display it when exiting.
    ///
    /// [`Error`]: std::error::Error
    fn advance(&mut self) -> anyhow::Result<()>;

    /// Get the simulation step.
    fn step(&self) -> u64;
}

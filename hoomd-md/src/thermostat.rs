// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Methods for sampling the canonical distribution
//! of kinetic energy.
//!
//! 
use hoomd_microstate::Microstate;

mod no_thermostat;
mod mttk;
mod bussi;
mod nhc;

pub use no_thermostat::NoThermostat;
pub use mttk::MTTKThermostat;
pub use bussi::BussiThermostat;
pub use nhc::NHCThermostat;

/// Adjust the temperature of a system for sampling
/// the canonical distribution
/// of kinetic energy.
///
/// Implement [`Thermostat`] or use one of the
/// provided method in [`thermostat`](crate::thermostat)
/// in MD simulations.
pub trait Thermostat<B, S, X, C, M> {
    /// Integrate the thermostat dof foward, and return
    /// Note that translation and rotation are assumed to have identical math
    /// behind their scaling factors.
    fn integrate_step_one<P>(
        &mut self,
        microstate: &Microstate<B, S, X, C>,
        macrostate: &M,
        dt: &f64,
        compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, X, C>) -> (f64, f64);

    /// The scaling factor for velocities in the second half-step.
    /// Note that translation and rotation are assumed to have identical math
    /// behind their scaling factors.
    fn integrate_step_two<P>(
        &mut self,
        microstate: &Microstate<B, S, X, C>,
        macrostate: &M,
        dt: &f64,
        compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, X, C>) -> (f64, f64);
}

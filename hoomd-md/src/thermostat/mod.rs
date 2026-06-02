// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Methods for sampling the canonical distribution
//! of kinetic energy.
use rand::Rng;

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
pub trait Thermostat<M> {
    /// Integrate the thermostat dof foward, and return
    /// Note that translation and rotation are assumed to have identical math
    /// behind their scaling factors.
    fn integrate_step_one<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64;

    /// The scaling factor for velocities in the second half-step.
    /// Note that translation and rotation are assumed to have identical math
    /// behind their scaling factors.
    fn integrate_step_two<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64;
}

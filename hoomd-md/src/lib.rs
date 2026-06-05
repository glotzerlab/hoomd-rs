// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]

//! Apply the molecular dynamics simulation method to systems of bodies.
//! TODO: User guide

use rand::Rng;

pub mod thermostat;
pub mod method;

mod compute;
pub use compute::TranslationalKineticEnergy;
pub use compute::RotationalKineticEnergy;

mod modify;
pub use modify::ThermalizeAngularMomentum;
pub use modify::ThermalizeMomentum;
pub use modify::ZeroCenterMomentum;
pub use modify::ZeroCenterAngularMomentum;

mod update_net_force;
pub use update_net_force::UpdateNetForce;
pub use update_net_force::UpdateNetForceAndTorque;

/// Scale momenta to hold the system at constant temperature.
///
/// Use any of the thermostats in the [`thermostat`] module along with the
/// integration method of your choice.
///
/// The [`ConstantVolume`] integration method rescales every momentum in the
/// system following the given [`Thermostat`] to sample trajectories from the
/// canonical ensemble.
///
/// [`ConstantVolume`]: crate::method::ConstantVolume
pub trait Thermostat<M> {
    /// Integrate the thermostat one half step forward in time.
    ///
    /// Returns the momentum scaling factor to use during the first half step.
    fn integrate_step_one<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64;

    /// Integrate the thermostat one half step forward in time.
    ///
    /// Returns the momentum scaling factor to use during the second half step.
    fn integrate_step_two<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64;
}

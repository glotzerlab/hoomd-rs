// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]

//! Simulate molecular dynamics in systems of particles.
//! TODO: User guide

use rand::Rng;

pub mod thermostat;
pub mod methods;

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

pub trait Thermostat<M> {
    fn integrate_step_one<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64;

    fn integrate_step_two<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64;
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]

//! Apply molecular dynamics simulation methods to systems of bodies.
//! TODO: User guide

use rand::Rng;

use hoomd_microstate::{Body, Tagged};
use hoomd_microstate::Microstate;

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
pub use update_net_force::UpdateNetForceAndVirial;
pub use update_net_force::UpdateNetForceVirialAndTorque;

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
    fn integrate_half_step_one<R: Rng + ?Sized>(
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
    fn integrate_half_step_two<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64;
}

/// Integrate translational degrees of freedom.
///
/// [`TranslationalMotion`] integrates the [`Position`] and [`Momentum`] degrees of
/// freedom for selected bodies. 
/// 
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
/// * `S`: The [`Site::properties`](hoomd_microstate::Site) type.
/// * `X`: The spatial data structure type.
/// * `C`: The [`boundary`](hoomd_microstate::boundary) condition type.
/// * `M`: The [`macrostate`](hoomd_simulation::macrostate) type.
///
/// [`Position`]: hoomd_microstate::property::Position
/// [`Momentum`]: hoomd_microstate::property::Momentum
pub trait TranslationalMotion<B, S, X, C, M> {
    /// Integrate all body positions forward a full step and the momenta forward a half step.
    #[inline]
    fn integrate_translation_half_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
    ) {
        self.integrate_translation_half_step_one_with_filter(microstate, macrostate, |_| true);
    }

    /// Integrate selected body positions forward a full step and the momenta forward a half step.
    fn integrate_translation_half_step_one_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    );

    /// Integrate all body momenta forward a half step.
    #[inline]
    fn integrate_translation_half_step_two(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
    ) {
        self.integrate_translation_half_step_two_with_filter(microstate, macrostate, |_| true);
    }

    /// Integrate selected body momenta forward a half step.
    fn integrate_translation_half_step_two_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    );

    /// Integrate selected body translational degrees of freedom forward one step.
    #[inline]
    fn integrate_translation_with_filter<E, F>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        interaction_model: &E,
        should_integrate_body: F,
    ) where
    F: Fn(&Tagged<Body<B, S>>) -> bool,
    Microstate<B, S, X, C>: UpdateNetForceAndVirial<E> {
        self.integrate_translation_half_step_one_with_filter(microstate, macrostate, &should_integrate_body);
        microstate.update_net_force_and_virial(interaction_model);
        self.integrate_translation_half_step_two_with_filter(microstate, macrostate, &should_integrate_body);
    }

    /// Integrate all body translational degrees of freedom forward one step.
    #[inline]
    fn integrate_translation<E>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        interaction_model: &E,
    ) where
    Microstate<B, S, X, C>: UpdateNetForceAndVirial<E> {
        self.integrate_translation_half_step_one_with_filter(microstate, macrostate, |_| true);
        microstate.update_net_force_and_virial(interaction_model);
        self.integrate_translation_half_step_two_with_filter(microstate, macrostate, |_| true);
    }
}

/// Integrate rotational degrees of freedom.
///
/// [`RotationalMotion`] integrates the [`Orientation`] and [`AngularMomentum`] degrees of
/// freedom for selected bodies. 
/// 
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
/// * `S`: The [`Site::properties`](hoomd_microstate::Site) type.
/// * `X`: The spatial data structure type.
/// * `C`: The [`boundary`](hoomd_microstate::boundary) condition type.
/// * `M`: The [`macrostate`](hoomd_simulation::macrostate) type.
///
/// [`Orientation`]: hoomd_microstate::property::Orientation
/// [`AngularMomentum`]: hoomd_microstate::property::AngularMomentum
pub trait RotationalMotion<B, S, X, C, M> {
    /// Integrate all body orientations forward a full step and their angular momenta forward a half step.
    #[inline]
    fn integrate_rotation_half_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
    ) {
        self.integrate_rotation_half_step_one_with_filter(microstate, macrostate, |_| true);
    }
    
    /// Integrate selected body orientations forward a full step and their angular momenta forward a half step.
    fn integrate_rotation_half_step_one_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    );

    /// Integrate all body angular momenta forward a half step.
    #[inline]
    fn integrate_rotation_half_step_two(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
    ) {
        self.integrate_rotation_half_step_two_with_filter(microstate, macrostate, |_| true);
    }

    /// Integrate selected body angular momenta forward a half step.
    fn integrate_rotation_half_step_two_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    );

    /// Integrate selected body translational and rotational degrees of freedom forward one step.
    #[inline]
    fn integrate_translation_and_rotation_with_filter<E, F>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        interaction_model: &E,
        should_integrate_body: F,
    ) where
    F: Fn(&Tagged<Body<B, S>>) -> bool,
    Microstate<B, S, X, C>: UpdateNetForceVirialAndTorque<E>,
    Self: TranslationalMotion<B, S, X, C, M> 
    {
        self.integrate_translation_half_step_one_with_filter(microstate, macrostate, &should_integrate_body);
        self.integrate_rotation_half_step_one_with_filter(microstate, macrostate, &should_integrate_body);
        microstate.update_net_force_virial_and_torque(interaction_model);
        self.integrate_translation_half_step_two_with_filter(microstate, macrostate, &should_integrate_body);
        self.integrate_rotation_half_step_two_with_filter(microstate, macrostate, &should_integrate_body);
    }

    /// Integrate all body translational and rotational degrees of freedom forward one step.
    #[inline]
    fn integrate_translation_and_rotation<E>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        interaction_model: &E,
    ) where
    Microstate<B, S, X, C>: UpdateNetForceVirialAndTorque<E>,
    Self: TranslationalMotion<B, S, X, C, M> 
    {
        self.integrate_translation_half_step_one_with_filter(microstate, macrostate, |_| true);
        self.integrate_rotation_half_step_one_with_filter(microstate, macrostate, |_| true);
        microstate.update_net_force_virial_and_torque(interaction_model);
        self.integrate_translation_half_step_two_with_filter(microstate, macrostate, |_| true);
        self.integrate_rotation_half_step_two_with_filter(microstate, macrostate, |_| true);
    }
}

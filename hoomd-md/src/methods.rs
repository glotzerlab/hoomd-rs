// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! MD integration methods.

use hoomd_microstate::{Microstate};
mod constant_volume;
pub use constant_volume::ConstantVolume;

/// Integrate over translational degrees of freedom.
///
/// Conceptually, integration changes a system's [`Microstate`] according to
/// equations of motion that are determined by the system's degrees of freedom.
///
/// In translational integration, the equations of motion allow a body's
/// [`Position`], [`Momentum`] to evolve over time.
/// [`Thermostat`] that maintains system temperature according to the setpoint
/// stored in a [`macrostate`].
/// 
/// The generic type names are:
/// * `N`: The dimension of the system.
/// * `B`: The [`Body::properties`] type.
/// * `S`: The [`Site::properties`] type.
/// * `C`: The [`boundary`] condition type.
/// * `T`: The [`Thermostat`] type.
/// * `M`: The [`macrostate`] type.
pub trait TranslationalMotion<B, S, X, C, T, M> {
    /// Perform the first integration half-step, mutating the microstate and possibly the thermostat.
    fn integrate_translation_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        thermostat: &mut T,
        macrostate: &M,
    );

    /// Perform the second integration half-step, mutating the microstate and possibly the thermostat.
    fn integrate_translation_step_two(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        thermostat: &mut T,
        macrostate: &M,
    );
}

/// Integrate over rotational degrees of freedom.
///
/// Conceptually, integration changes a system's [`Microstate`] according to
/// equations of motion that are determined by the system's degrees of freedom.
///
/// In rotational integration, the equations of motion allow a body's
/// [`Orientation`] and [`AngularMomentum`] to evolve over time. 
/// [`Thermostat`] that maintains system temperature according to the setpoint
/// stored in a [`macrostate`].
/// 
/// The generic type names are:
/// * `N`: The dimension of the system.
/// * `B`: The [`Body::properties`] type.
/// * `S`: The [`Site::properties`] type.
/// * `C`: The [`boundary`] condition type.
/// * `T`: The [`Thermostat`] type.
/// * `M`: The [`macrostate`] type.
pub trait RotationalMotion<const N: usize, B, S, X, C, T, M> {
    /// Perform the first integration half-step, mutating the microstate and possibly the thermostat.
    fn integrate_rotation_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        thermostat: &mut T,
        macrostate: &M,
    );

    /// Perform the second integration half-step, mutating the microstate and possibly the thermostat.
    fn integrate_rotation_step_two(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        thermostat: &mut T,
        macrostate: &M,
    );
}

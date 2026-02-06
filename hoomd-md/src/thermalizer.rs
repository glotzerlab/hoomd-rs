// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Methods for thermalizing or modifying the momenta.
//!
use hoomd_microstate::{
    Microstate
};


mod translational_dof;
mod rotational_dof;
mod remove_com_momentum;
mod remove_com_angular_momentum;

pub use remove_com_momentum::ComMomentumRemover;
pub use remove_com_angular_momentum::ComAngularMomentumRemover;


/// Thermalize the translational motion of [`Microstate`].
///
/// Implement [`TranslationalThermalizer`] on a custom type
/// or use one of the provide method in
/// [`thermalizer`](crate::thermalizer) in MD simulations.
pub trait TranslationalThermalizer<const N: usize, B, S, C> {
    /// Thermalize the rotational motion.
    fn thermalize_translation(&self, microstate: &mut Microstate<B, S, C>);
}

/// Thermalize the rotational motion of [`Microstate`].
///
/// Implement [`RotationalThermalizer`] on a custom type
/// or use one of the provide method in
/// [`thermalizer`](crate::thermalizer) in MD simulations.
pub trait RotationalThermalizer<const N: usize, B, S, C> {
    /// Thermalize the rotational motion.
    fn thermalize_rotation(&self, microstate: &mut Microstate<B, S, C>);
}

/// Modify the translational momenta of [`Microstate`].
///
/// Implement [`TranslationalMomentumModifier`] on a custom type
/// or use one of the provide method in
/// [`thermalizer`](crate::thermalizer) in MD simulations.
pub trait TranslationalMomentumModifier<const N: usize, B, S, C> {
    /// Modify the translational momenta.
    fn modify(&self, microstate: &mut Microstate<B, S, C>);
}

/// Thermalize system's momenta
/// according to Maxwell-Boltzmann distribtion.
pub struct Thermalizer {
    /// The desired temperature
    pub kT: f64,
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Simulate molecular dynamics in systems of particles.

TODO: Expand documentation.
 */

pub mod thermostat;
use thermostat::{Thermostat, NoThermostat};
use hoomd_microstate::Microstate;

/** Integrate over translational degrees of freedom. 
 */
pub trait TranslationalMotion<B, S, C> {
    /// Integrate over translational degrees of freedom. 
    fn integrate_translation(&self, microstate: &mut Microstate<B, S, C>);
}

/** Integrate over rotational degrees of freedom. 
 */
pub trait RotationalMotion<B, S, C> {
    /// Integrate over rotational degrees of freedom.
    fn integrate_rotation(&self, microstate: &mut Microstate<B, S, C>);
}

/// TODO: add documentation
pub struct ConstantVolume<T: Thermostat> {
    /// The size of a timestep.
    pub dt: f64,

    /// The temperature in units of kT.
    pub kt: f64,

    /// The thermostat.
    pub thermostat: T,
}

/// TODO: add documentation
pub struct ConstantPressure;


impl<B, S, C, T: Thermostat> TranslationalMotion<B, S, C> for ConstantVolume<T>
where 
    T: Thermostat
{
    #[inline]
    fn integrate_translation(&self, microstate: &mut Microstate<B, S, C>) {
        // For loop over a range instead of bodies().iter() since the latter holds an immutable borrow.

        for body_index in 0..microstate.bodies().len() {

        }

        microstate.increment_substep();
    }
}

impl<B, S, C, T> RotationalMotion<B, S, C> for ConstantVolume<T>
where 
    T: Thermostat
{
    #[inline]
    fn integrate_rotation(&self, microstate: &mut Microstate<B, S, C>) {
        // TODO
    }
}
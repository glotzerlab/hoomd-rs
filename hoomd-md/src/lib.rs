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
pub trait TranslationalMotion {
    fn integrate_translation() {}
}

/** Integrate over rotational degrees of freedom. 
 */
pub trait RotationalMotion {}


pub struct ConstantVolume<T: Thermostat> {
    dt: f64,
    kT: f64,
    thermostat: T,
}

pub struct ConstantPressure;


impl<T: Thermostat> TranslationalMotion for ConstantVolume<T> {}

impl<T: Thermostat> RotationalMotion for ConstantVolume<T> {}
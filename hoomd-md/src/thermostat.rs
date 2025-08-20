// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Control system temperature.

TODO: Expand documentation.
 */

/** Adjust the temperature of a system.
TODO: Add Thermostat documentation.
*/
 pub trait Thermostat {
    /// TODO: documentation.
    fn temperature_factor(&self) -> f64;
}

/** TODO: documentation */
pub struct NoThermostat;

impl Thermostat for NoThermostat {
    #[inline]
    fn temperature_factor(&self) -> f64 {
        1.0
    }
}
// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Control system temperature.

TODO: Expand documentation.
 */

pub trait Thermostat {
    fn temperature_factor() -> f64;
}

pub struct NoThermostat;

impl Thermostat for NoThermostat {
    fn temperature_factor() -> f64 {
        1.0
    }
}
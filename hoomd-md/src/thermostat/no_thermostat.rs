// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `NoThermostat`

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::Thermostat;

/// Apply no momentum scaling.
///
/// Use [`NoThermostat`] with [`ConstantVolume`] to model the microcanonical (NVE) ensemble.
///
/// [`ConstantVolume`]: crate::method::ConstantVolume
///
/// # Example
///
/// ```
/// use hoomd_md::thermostat::NoThermostat;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let thermostat = NoThermostat;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoThermostat;

impl<M> Thermostat<M> for NoThermostat {
    #[inline]
    fn integrate_half_step_one<R: Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        _macrostate: &M,
        _delta_t: f64,
        _kinetic_energy: f64,
        _degrees_of_freedom: usize,
    ) -> f64 {
        1.0
    }

    #[inline]
    fn integrate_half_step_two<R: Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        _macrostate: &M,
        _delta_t: f64,
        _kinetic_energy: f64,
        _degrees_of_freedom: usize,
    ) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::check;
    use hoomd_microstate::{Body, Microstate};
    use hoomd_vector::Cartesian;

    struct Nve;

    #[test]
    fn test_no_thermostat() -> anyhow::Result<()> {
        // Instantiation
        let mut thermostat = NoThermostat;

        // Thermostat Implementation
        let mut microstate = Microstate::new();
        microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;
        let macrostate = Nve;
        let delta_t = 1.0;
        let mut rng = microstate.counter().make_rng();

        check!(1.0 == thermostat.integrate_half_step_one(&mut rng, &macrostate, delta_t, 1.0, 3));
        check!(1.0 == thermostat.integrate_half_step_two(&mut rng, &macrostate, delta_t, 1.0, 3));

        Ok(())
    }
}

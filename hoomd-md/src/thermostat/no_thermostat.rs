// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
use rand::Rng;
use crate::Thermostat;

/// [`NoThermostat`] implement the dummy method
/// that performs no adjustment on the temperature
/// for [`TranslationalMotion`](crate::methods::TranslationalMotion) 
/// and [`RotationalMotion`](crate::methods::RotationalMotion) 
/// as they require an input of a [`Thermostat`] during
/// integration.
pub struct NoThermostat;

impl<M> Thermostat<M> for NoThermostat {
    /// Dummy method that performs no temperature
    /// adjustment.
    #[inline]
    fn integrate_step_one<R: Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        _macrostate: &M,
        _delta_t: f64,
        _kinetic_energy: f64,
        _degrees_of_freedom: usize,
    ) -> f64
    {
        1.0
    }
    
    /// Dummy method that performs no temperature
    /// adjustment.
    #[inline]
    fn integrate_step_two<R: Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        _macrostate: &M,
        _delta_t: f64,
        _kinetic_energy: f64,
        _degrees_of_freedom: usize,
    ) -> f64
    {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use assert2::check;
    use super::*;
    use hoomd_microstate::{Body, Microstate};
    use hoomd_vector::Cartesian;

    struct NVE;

    #[test]
    fn test_no_thermostat() -> anyhow::Result<()> {
        // Instantiation
        let mut thermostat = NoThermostat;

        // Thermostat Implementation
        let mut microstate = Microstate::new();
        microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;
        let macrostate = NVE;
        let delta_t = 1.0;
        let mut rng = microstate.counter().make_rng();

        check!(1.0 == thermostat.integrate_step_one(&mut rng, &macrostate, delta_t, 1.0, 3));
        check!(1.0 == thermostat.integrate_step_two(&mut rng, &macrostate, delta_t, 1.0, 3));

        Ok(())
    }
}

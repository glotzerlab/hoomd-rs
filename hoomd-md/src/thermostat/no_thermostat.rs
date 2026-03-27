// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
use hoomd_microstate::Microstate;
use crate::thermostat::Thermostat;

/// [`NoThermostat`] implement the dummy method
/// that performs no adjustment on the temperature
/// for [`TranslationalMotion`](crate::methods::TranslationalMotion) 
/// and [`RotationalMotion`](crate::methods::RotationalMotion) 
/// as they require an input of a [`Thermostat`] during
/// integration.
pub struct NoThermostat;

impl<B, S, X, C, M> Thermostat<B, S, X, C, M> for NoThermostat {
    /// Dummy method that performs no temperature
    /// adjustment.
    #[inline]
    fn integrate_step_one<P>(
        &mut self,
        microstate: &Microstate<B, S, X, C>,
        _macrostate: &M,
        _dt: &f64,
        mut compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, X, C>) -> (f64, f64),
    {
        let (_, _) = compute_properties(&microstate);
        1.0
    }
    
    /// Dummy method that performs no temperature
    /// adjustment.
    #[inline]
    fn integrate_step_two<P>(
        &mut self,
        microstate: &Microstate<B, S, X, C>,
        _macrostate: &M,
        _dt: &f64,
        mut compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, X, C>) -> (f64, f64),
    {
        let (_, _) = compute_properties(&microstate);
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_microstate::Body;
    use hoomd_vector::Cartesian;

    struct NVE;

    fn compute_properties<B, S, X, C>(_m: &Microstate<B, S, X, C>) -> (f64, f64) {
        (1.0, 1.0)
    }

    #[test]
    fn test_no_thermostat() -> anyhow::Result<()> {
        // Instantiation
        let mut thermostat = NoThermostat;

        // Thermostat Implementation
        let mut microstate = Microstate::new();
        microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;
        let macrostate = NVE;
        let dt = 1.0;

        assert_eq!(1.0, thermostat.integrate_step_one(&microstate, &macrostate, &dt, compute_properties));
        assert_eq!(1.0, thermostat.integrate_step_two(&microstate, &macrostate, &dt, compute_properties));

        Ok(())
    }
}
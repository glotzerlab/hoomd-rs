// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
use hoomd_microstate::Microstate;
use crate::thermostat::Thermostat;

/// [`NoThermostat`] implement the dummy method
/// that performs no adjustment on the temperature
/// for [`TranslationalMotion`] and [`RotationalMotion`]
/// as they require an input of a [`Thermostat`] during
/// integration.
pub struct NoThermostat;

impl<B, S, C, M> Thermostat<B, S, C, M> for NoThermostat {
    #[inline]
    fn integrate_step_one<P>(
        &mut self,
        microstate: &Microstate<B, S, C>,
        _macrostate: &M,
        _dt: &f64,
        mut compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, C>) -> (f64, f64),
    {
        let (_, _) = compute_properties(&microstate);
        1.0
    }

    #[inline]
    fn integrate_step_two<P>(
        &mut self,
        microstate: &Microstate<B, S, C>,
        _macrostate: &M,
        _dt: &f64,
        mut compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, C>) -> (f64, f64),
    {
        let (_, _) = compute_properties(&microstate);
        1.0
    }
}
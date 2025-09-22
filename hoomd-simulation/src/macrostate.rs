// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Store global system parameters for use in thermostats, integrators, etc.

/// Store the kinetic temperature of the system.
pub trait Temperature {
    /// The kinetic temperature of the system.
    fn temperature(&self) -> &f64;

    /// The mutable kinetic temperature of the system.
    fn temperature_mut(&mut self) -> &mut f64;
}

/// Store the pressure of the system.
pub trait Pressure {
    /// The pressure of the system.
    fn pressure(&self) -> &f64;

    /// The mutable pressure of the system.
    fn pressure_mut(&mut self) -> &mut f64;
}

/// Macrostate for an iso-energy ensemble.
/// 
/// This is a convenience type which implements [`Temperature`] without storing
/// a temperature setpoint. Use it for simulating NVE ensembles, for example
/// with the [`ConstantVolume`] integrator.
/// 
/// This type should only be instantiated with `::default()`
pub struct Isoenergy {
    /// The faux temperature for the system
    _temperature: f64,
}
impl Default for Isoenergy {
    #[inline]
    fn default() -> Self {
        Isoenergy {
            _temperature: 1.0
        }
    }
}
impl Temperature for Isoenergy {
    #[inline]
    fn temperature(&self) -> &f64 {
        &self._temperature
    }

    #[inline]
    fn temperature_mut(&mut self) -> &mut f64 {
        &mut self._temperature
    }
}

/// Macrostate for an isothermal ensemble.
pub struct Isothermal {
    /// Kinetic temperature of the system.
    pub temperature: f64
}
impl Temperature for Isothermal {
    #[inline]
    fn temperature(&self) -> &f64 {
        &self.temperature
    }

    #[inline]
    fn temperature_mut(&mut self) -> &mut f64 {
        &mut self.temperature
    }
}

/// Macrostate for an isobaric ensemble.
pub struct Isobaric {
    /// Pressure of the system.
    pub pressure: f64
}

/// Macrostate for an isothermal and isobaric ensemble.
pub struct IsothermalIsobaric {
    /// Kinetic temperature of the system.
    pub temperature: f64,
    /// Pressure of the system.
    pub pressure: f64
}
impl Temperature for IsothermalIsobaric {
    #[inline]
    fn temperature(&self) -> &f64 {
        &self.temperature
    }

    #[inline]
    fn temperature_mut(&mut self) -> &mut f64 {
        &mut self.temperature
    }
}
impl Pressure for IsothermalIsobaric {
    #[inline]
    fn pressure(&self) -> &f64 {
        &self.pressure
    }

    #[inline]
    fn pressure_mut(&mut self) -> &mut f64 {
        &mut self.pressure
    }
}

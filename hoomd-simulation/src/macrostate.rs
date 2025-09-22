// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Store global system parameters for use in thermostats, integrators, etc.*/

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

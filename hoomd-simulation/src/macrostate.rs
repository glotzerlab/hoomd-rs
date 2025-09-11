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


/** Mark an ensemble as having constant volume.

Must be manually added for macrostates that do not implement [`Pressure`].
*/
pub trait Isochoric {}

/** Mark an ensemble as having constant entropy.

Must be manually added for macrostates that do not implement [`Temperature`].
*/
pub trait Isoentropic {}

/** Mark an ensemble as having constant temperature. 

This trait is automatically implemented for every macrostate that implements
[`Temperature`].
*/
pub trait Isothermal {}
impl<T> Isothermal for T where T: Temperature {}

/** Mark an ensemble as having constant pressure.

This trait is automatically implemented for every macrostate that implements
[`Pressure`].
*/
pub trait Isobaric {}
impl<T> Isobaric for T where T: Pressure {}


/// Macrostate for an isothermal ensemble.
pub struct IsothermalMacrostate {
    /// Kinetic temperature of the system.
    pub kT: f64
}
impl Temperature for IsothermalMacrostate {
    #[inline]
    fn temperature(&self) -> &f64 {
        &self.kT
    }

    #[inline]
    fn temperature_mut(&mut self) -> &mut f64 {
        &mut self.kT
    }
}
impl Isochoric for IsothermalMacrostate {}

/// Macrostate for an isobaric ensemble.
pub struct IsobaricMacrostate {
    /// Pressure of the system.
    pub pressure: f64
}
impl Pressure for IsobaricMacrostate {
    #[inline]
    fn pressure(&self) -> &f64 {
        &self.pressure
    }

    #[inline]
    fn pressure_mut(&mut self) -> &mut f64 {
        &mut self.pressure
    }
}
impl Isoentropic for IsobaricMacrostate {}

/// Macrostate for an isothermal and isobaric ensemble.
pub struct IsothermalIsobaricMacrostate {
    /// Kinetic temperature of the system.
    pub kT: f64,
    /// Pressure of the system.
    pub pressure: f64
}
impl Temperature for IsothermalIsobaricMacrostate {
    #[inline]
    fn temperature(&self) -> &f64 {
        &self.kT
    }

    #[inline]
    fn temperature_mut(&mut self) -> &mut f64 {
        &mut self.kT
    }
}
impl Pressure for IsothermalIsobaricMacrostate {
    #[inline]
    fn pressure(&self) -> &f64 {
        &self.pressure
    }

    #[inline]
    fn pressure_mut(&mut self) -> &mut f64 {
        &mut self.pressure
    }
}

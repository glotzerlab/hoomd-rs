// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Store global system parameters for use in thermostats, integrators, etc.*/

/// Store the kinetic temperature of the system.
trait Temperature {
    /// The kinetic temperature of the system.
    fn temperature(&self) -> &f64;

    /// The mutable kinetic temperature of the system.
    fn temperature_mut(&mut self) -> &mut f64;
}

/// Store the pressure of the system.
trait Pressure {
    /// The pressure of the system.
    fn pressure(&self) -> &f64;

    /// The mutable pressure of the system.
    fn pressure_mut(&mut self) -> &mut f64;
}


/** Mark an ensemble as having constant volume.

Must be manually added for macrostates that do not implement [`Pressure`].
*/
trait Isochoric {}

/** Mark an ensemble as having constant temperature. 

This trait is automatically implemented for every macrostate that implements
[`Temperature`].
*/
trait Isothermal {}
impl<T> Isothermal for T where T: Temperature {}

/** Mark an ensemble as having constant pressure.

This trait is automatically implemented for every macrostate that implements
[`Pressure`].
*/
trait Isobaric {}
impl<T> Isobaric for T where T: Pressure {}


/// Macrostate for an isothermal ensemble.
struct IsothermalMacrostate {
    kT: f64
}
impl Temperature for IsothermalMacrostate {
    fn temperature(&self) -> &f64 {
        &self.kT
    }

    fn temperature_mut(&mut self) -> &mut f64 {
        &mut self.kT
    }
}
impl Isochoric for IsothermalMacrostate {}

/// Macrostate for an isobaric ensemble.
struct IsobaricMacrostate {
    pressure: f64
}
impl Pressure for IsobaricMacrostate {
    fn pressure(&self) -> &f64 {
        &self.pressure
    }

    fn pressure_mut(&mut self) -> &mut f64 {
        &mut self.pressure
    }
}

/// Macrostate for an isothermal and isobaric ensemble.
struct IsothermalIsobaricMacrostate {
    kT: f64,
    pressure: f64
}
impl Temperature for IsothermalIsobaricMacrostate {
    fn temperature(&self) -> &f64 {
        &self.kT
    }

    fn temperature_mut(&mut self) -> &mut f64 {
        &mut self.kT
    }
}
impl Pressure for IsothermalIsobaricMacrostate {
    fn pressure(&self) -> &f64 {
        &self.pressure
    }

    fn pressure_mut(&mut self) -> &mut f64 {
        &mut self.pressure
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Traits that describe system macrostates and types that implement them.
//!
//! ## Macrostate
//!
//! Use a built-in macrostate type, such as [`Isothermal`], [`IsothermalIsobaric`]
//! or [`Isobaric`] store the temperature and/or pressure set points for use
//! with thermostats, barostats, and/or Monte Carlo trial moves:
//!
//! ```
//! use hoomd_simulation::macrostate::Isothermal;
//!
//! let macrostate = Isothermal { temperature: 1.2 };
//! ```
//!
//! The *actual* ensemble that your simulation samples is set by the methods
//! that you apply to it. These macrostate types exist merely to pass
//! parameters to methods that need them. For example, you could choose an
//! [`IsothermalIsobaric`] macrostate and use it only with local Monte Carlo
//! moves. Without boundary moves, the simulation will be in the NVT ensemble.
//!
//! ## Traits
//!
//! When you need additional macrostate parameters, use a custom type.
//! Implement [`Temperature`] and/or [`Pressure`] for your type as needed.

/// Set the thermodynamic temperature of a system.
///
/// Macrostates with the [`Temperature`] trait set the temperature
/// of the simulation. In *hoomd-rs*, temperature is given in units of
/// $` [\mathrm{energy}] `$: $` \mathrm{temperature} = kT `$.
///
/// # Example
/// ```
/// use hoomd_simulation::macrostate::Isothermal;
///
/// let macrostate = Isothermal { temperature: 1.2 };
/// ```
pub trait Temperature {
    /// The system's temperature $` ([\mathrm{energy}]) `$.
    fn temperature(&self) -> &f64;

    /// The system's temperature $` ([\mathrm{energy}]) `$.
    fn temperature_mut(&mut self) -> &mut f64;
}

/// Set the thermodynamic pressure of a system.
///
/// Macrostates with the [`Pressure`] trait set the pressure
/// of the simulation. In *hoomd-rs*, pressure is given in units of
/// $` [\mathrm{energy}] \cdot [\mathrm{length}]^{-D} `$ where $` D `$
/// is the dimensionality of the system.
///
/// # Example
/// ```
/// use hoomd_simulation::macrostate::Isobaric;
///
/// let macrostate = Isobaric { pressure: 0.4 };
/// ```
pub trait Pressure {
    /// The system's pressure $` ([\mathrm{energy}] \cdot [\mathrm{length}]^{-D}) `$.
    fn pressure(&self) -> &f64;

    /// The system's pressure $` ([\mathrm{energy}] \cdot [\mathrm{length}]^{-D}) `$.
    fn pressure_mut(&mut self) -> &mut f64;
}

/// Constant temperature macrostate.
///
/// Use [`Isothermal`] to set the system temperature using a thermostat or Monte
/// Carlo trial moves. Temperature is given in units of $` [\mathrm{energy}] `$:
/// $` \mathrm{temperature} = kT `$.
///
/// # Example
/// ```
/// use hoomd_simulation::macrostate::Isothermal;
///
/// let macrostate = Isothermal { temperature: 1.2 };
/// ```
pub struct Isothermal {
    /// The system's temperature $` ([\mathrm{energy}]) `$.
    pub temperature: f64,
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

/// Constant pressure macrostate.
///
/// Use [`Isobaric`] to set the system pressure using a thermostat or Monte
/// Carlo trial moves. Pressure is given in units of $` [\mathrm{energy}] \cdot
/// [\mathrm{length}]^{-D} `$ where $` D `$ is the dimensionality of the system.
///
/// # Example
/// ```
/// use hoomd_simulation::macrostate::Isobaric;
///
/// let macrostate = Isobaric { pressure: 0.4 };
/// ```
pub struct Isobaric {
    /// The system's pressure $` ([\mathrm{energy}] \cdot [\mathrm{length}]^{-D}) `$.
    pub pressure: f64,
}

/// Constant temperature, constant pressure macrostate.
///
/// Use [`IsothermalIsobaric`] to set both the system temperature and pressure
/// using a thermostat and barostat or Monte
/// Carlo trial moves.
///
/// * Temperature is given in units of $` [\mathrm{energy}] `$:
///   $` \mathrm{temperature} = kT `$.
/// * Pressure is given in units of $` [\mathrm{energy}] \cdot
///   [\mathrm{length}]^{-D} `$ where $` D `$ is the dimensionality of the system.
///
/// # Example
/// ```
/// use hoomd_simulation::macrostate::Isothermal;
///
/// let macrostate = Isothermal { temperature: 1.2 };
/// ```
pub struct IsothermalIsobaric {
    /// Kinetic temperature of the system.
    pub temperature: f64,
    /// Pressure of the system.
    pub pressure: f64,
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

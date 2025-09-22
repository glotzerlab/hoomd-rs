// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Control system temperature.
//! 
//! TODO: Expand documentation.

use hoomd_microstate::{
    Microstate, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{Acceleration, Mass, Position, Velocity},
};
use hoomd_simulation::macrostate::{Isothermal, Temperature};
use hoomd_vector::{Cartesian, Vector};
use rand_distr::{Distribution, Gamma, Normal};

/// Adjust the temperature of a system.
/// 
/// TODO: ensure that docs indicate two-step integration is baked into the thermostat trait
/// TODO: Add example.
pub trait Thermostat<B, S, C, M> {
    /// The scaling factor for velocities in the first half-step.
    /// Note that translation and rotation are assumed to have identical math
    /// behind their scaling factors.
    fn rescaling_factor_step_one<P>(
        &mut self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
        compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, C>) -> (f64, f64);

    /// The scaling factor for velocities in the second half-step.
    /// Note that translation and rotation are assumed to have identical math
    /// behind their scaling factors.
    fn rescaling_factor_step_two(
        &self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
    ) -> f64;

    /// TODO: add docs
    fn advance<P>(
        &mut self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
        compute_properties: P,
    ) where
        P: FnMut(&Microstate<B, S, C>) -> (f64, f64);
}

/// Constant temperature.
/// TODO: Add example.
pub struct NoThermostat;

impl<B, S, C, M> Thermostat<B, S, C, M> for NoThermostat
{
    #[inline]
    fn rescaling_factor_step_one<P>(
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
    fn rescaling_factor_step_two(
        &self,
        _microstate: &Microstate<B, S, C>,
        _macrostate: &M,
        _dt: &f64,
    ) -> f64 {
        1.0
    }

    #[inline]
    fn advance<P>(
        &mut self,
        _microstate: &Microstate<B, S, C>,
        _macrostate: &M,
        _dt: &f64,
        _compute_properties: P,
    ) where
        P: FnMut(&Microstate<B, S, C>) -> (f64, f64),
    {
    }
}

/// Bussi thermostat.
/// TODO: Add documentation.
/// TODO: Add example.
pub struct BussiThermostat {
    /// Thermostat time constant (`[time]`).
    tau: f64,
    /// Cumulative energy drift due to the thermostat. Useful for checking energy conservation.
    cumu_energy_drift: f64,
}
impl BussiThermostat {
    /// Constrcut MTTKThermostat.
    pub fn new(tau: f64) -> Self {
        assert!(tau >= 0.0, "MTTKThermostat requires tau >= 0");
        Self {
            tau: tau,
            cumu_energy_drift: 0.0,
        }
    }
    /// Calculate the energy drift due to the thermostat.
    pub fn energy_drift(&self, kinetic_energy_old: &f64, rescaling_factor: &f64) -> f64 {
        kinetic_energy_old * (1.0 - rescaling_factor.powi(2))
    }
}
/// TODO: add documentation
impl<B, S, C, M> Thermostat<B, S, C, M> for BussiThermostat
where
    M: Temperature,
{
    /// Calculate velocity rescaling factor following the Appendix in https://doi.org/10.1063/1.2408420.
    /// Bussi requires the rng, instataneous kinetic_energy, timestep and degrees-of-freedom,
    /// change the trait function definition accordingly?
    #[inline]
    fn rescaling_factor_step_one<P>(
        &mut self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
        mut compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, C>) -> (f64, f64),
    {
        let kT_setpoint = macrostate.temperature();

        let (ke, dof) = compute_properties(&microstate);

        // panic if momenta was not initialized
        assert!(
            !(ke == 0.0 && dof != 0.0),
            "Bussi thermostat requires non-zero initial momenta."
        );

        // get rng from microstate to ensure reproducibility
        let mut rng = microstate.counter().make_rng();

        // trivial case when no particles are present
        if dof == 0.0 {
            return 1.0;
        }

        // special case when tau is set to 0.
        let mut time_decay_factor = 0.0;
        // normal case time decay factor.
        if self.tau != 0.0 {
            time_decay_factor = (-dt / self.tau).exp();
        }
        // sample random number form standard normal distribution for the first dof.
        let random_normal_one: f64 = Normal::new(0.0, 1.0).unwrap().sample(&mut rng);
        // special case when dof is 1.
        let mut random_gamma: f64 = 0.0;
        // sample random numnber from gamma distribution for the rest of dof
        if dof > 0.0 {
            random_gamma = 2.0 * Gamma::new((dof - 1.0) / 2.0, 1.0).unwrap().sample(&mut rng);
        }
        // assemble everything
        let v = kT_setpoint / 2.0 / ke;
        let term1 = v * (1.0 - time_decay_factor) * (random_gamma + random_normal_one.powi(2));
        let term2 =
            2.0 * random_normal_one * (v * (1.0 - time_decay_factor) * time_decay_factor).sqrt();
        let alpha = (time_decay_factor + term1 + term2).sqrt();

        self.cumu_energy_drift += self.energy_drift(&ke, &alpha);
        alpha
    }

    #[inline]
    fn rescaling_factor_step_two(
        &self,
        _microstate: &Microstate<B, S, C>,
        _macrostate: &M,
        _dt: &f64,
    ) -> f64 {
        1.0
    }

    #[inline]
    fn advance<P>(
        &mut self,
        _microstate: &Microstate<B, S, C>,
        _macrostate: &M,
        _dt: &f64,
        _compute_properties: P,
    ) where
        P: FnMut(&Microstate<B, S, C>) -> (f64, f64),
    {
    }
}

/// MTTK thermostat.
/// TODO: Add documentation.
/// TODO: Add example.
pub struct MTTKThermostat {
    /// Thermostat time constant (`[time]`).
    tau: f64,
    /// Thermostat velocity.
    xi: f64,
    /// Thermostat position. Refer to the log(s) in Nose-Hoover's EOS.
    eta: f64,
    /// Energy the thermostat contributes to the Hamiltonian. Useful for checking energy conservation.
    energy: f64,
}

impl MTTKThermostat {
    /// Constrcut MTTKThermostat.
    pub fn new(tau: f64) -> Self {
        assert!(tau > 0.0, "MTTKThermostat requires tau > 0");
        Self {
            tau: tau,
            xi: 0.0,
            eta: 0.0,
            energy: 0.0,
        }
    }
    /// Choose random initial values for the thermostat momentum.
    pub fn thermalize<B, S, C, M>(
        &mut self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dof: &f64,
    ) where
        M: Temperature,
    {
        let kT_setpoint = macrostate.temperature();
        let mut rng = microstate.counter().make_rng();
        let sigma = 1.0 / *dof / self.tau.powi(2);

        self.xi = Normal::new(0.0, sigma.sqrt()).unwrap().sample(&mut rng);
        self.energy = self.thermostat_energy(kT_setpoint, dof)
    }
    /// Calculate thermostat energy.
    pub fn thermostat_energy(&self, kT_setpoint: &f64, dof: &f64) -> f64 {
        dof * kT_setpoint * (self.eta + 0.5 * (self.xi * self.tau).powi(2))
    }
}

impl<B, S, C, M> Thermostat<B, S, C, M> for MTTKThermostat
where
    M: Temperature,
{
    #[inline]
    fn rescaling_factor_step_one<P>(
        &mut self,
        _microstate: &Microstate<B, S, C>,
        _macrostate: &M,
        dt: &f64,
        _compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, C>) -> (f64, f64),
    {
        (-0.5 * self.xi * dt).exp()
    }

    #[inline]
    fn rescaling_factor_step_two(
        &self,
        _microstate: &Microstate<B, S, C>,
        _macrostate: &M,
        dt: &f64,
    ) -> f64 {
        (-0.5 * self.xi * dt).exp()
    }

    #[inline]
    fn advance<P>(
        &mut self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
        mut compute_properties: P,
    ) where
        P: FnMut(&Microstate<B, S, C>) -> (f64, f64),
    {
        let kT_setpoint = macrostate.temperature();

        let (ke, dof) = compute_properties(&microstate);

        let kT_instantaneous = 2.0 / dof * ke;

        // Thermostat acceleration
        let G = (kT_instantaneous / kT_setpoint - 1.0) / self.tau.powi(2);

        let xi_dt_half = self.xi + 0.5 * G * dt;
        self.eta += xi_dt_half * dt;
        self.xi = xi_dt_half + 0.5 * G * dt;
        self.energy = self.thermostat_energy(kT_setpoint, &dof);
    }
}

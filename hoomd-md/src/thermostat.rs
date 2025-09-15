// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Control system temperature.

TODO: Expand documentation.
 */

use hoomd_microstate::{
    Microstate, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{Acceleration, Mass, Position, Velocity},
};
use hoomd_simulation::macrostate::{Isochoric, Isoentropic, Isothermal, Temperature};
use hoomd_vector::{Cartesian, Vector};
use rand::Rng;
use rand_distr::{Distribution, Gamma, Normal};

/** Adjust the temperature of a system.

TODO: ensure that docs indicate two-step integration is baked into the thermostat trait
TODO: Add example.
*/
pub trait Thermostat<B, S, C, M> {
    /// The scaling factor for velocities in the first half-step.
    /// Note that translation and rotation are assumed to have identical math
    /// behind their scaling factors.
    fn rescaling_factor_step_one<P>(
        &self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
        compute_properties: P,
    ) -> f64
    where
        P: Fn(&Microstate<B, S, C>) -> (f64, f64);

    /// The scaling factor for velocities in the second half-step.
    /// Note that translation and rotation are assumed to have identical math
    /// behind their scaling factors.
    fn rescaling_factor_step_two(
        &self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
    ) -> f64;

    fn advance<P>(&mut self, dt: &f64, compute_properties: P)
    where
        P: Fn(&Microstate<B, S, C>) -> (f64, f64);
}

/** Constant temperature.
TODO: Add example.
*/
pub struct NoThermostat;

impl<B, S, C, M> Thermostat<B, S, C, M> for NoThermostat
where
    M: Isoentropic,
{
    #[inline]
    fn rescaling_factor_step_one<P>(
        &self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
        compute_properties: P,
    ) -> f64
    where
        P: Fn(&Microstate<B, S, C>) -> (f64, f64),
    {
        1.0
    }

    #[inline]
    fn rescaling_factor_step_two(
        &self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
    ) -> f64 {
        1.0
    }

    #[inline]
    fn advance<P>(&mut self, dt: &f64, compute_properties: P)
    where
        P: Fn(&Microstate<B, S, C>) -> (f64, f64),
    {
    }
}

/** Bussi thermostat.
TODO: Add documentation.
TODO: Add example.
*/
pub struct BussiThermostat {
    /// Thermostat time constant (`[time]`).
    pub tau: f64,
}

/// TODO: add documentation
impl<B, S, C, M> Thermostat<B, S, C, M> for BussiThermostat
where
    M: Isothermal + Temperature,
{
    /** Calculate velocity rescaling factor following the Appendix in https://doi.org/10.1063/1.2408420.
        Bussi requires the rng, instataneous kinetic_energy, timestep and degrees-of-freedom,
        change the trait function definition accordingly?
    */
    #[inline]
    fn rescaling_factor_step_one<P>(
        &self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
        compute_properties: P,
    ) -> f64
    where
        P: Fn(&Microstate<B, S, C>) -> (f64, f64),
    {
        let kT = macrostate.temperature();

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
        let v = kT / 2.0 / ke;
        let term1 = v * (1.0 - time_decay_factor) * (random_gamma + random_normal_one.powi(2));
        let term2 =
            2.0 * random_normal_one * (v * (1.0 - time_decay_factor) * time_decay_factor).sqrt();
        (time_decay_factor + term1 + term2).sqrt()
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
    fn advance<P>(&mut self, dt: &f64, _compute_properties: P)
    where
        P: Fn(&Microstate<B, S, C>) -> (f64, f64),
    {
    }
}

/** MTTK thermostat.
TODO: Add documentation.
TODO: Add example.
*/
pub struct MTTKThermostat {
    /// Thermostat time constant (`[time]`).
    tau: f64,

    xi: f64, // TODO: add thermalize method?

    eta: f64,
}

impl MTTKThermostat {
    pub fn new(tau: f64) -> Self {
        assert!(tau > 0.0, "MTTKThermostat requires tau >= 0");
        Self {
            tau: tau,
            xi: 0.0,
            eta: 0.0,
        }
    }

    pub fn thermalize<B, S, C>(&mut self, microstate: &Microstate<B, S, C>, dof: &i64) {
        let mut rng = microstate.counter().make_rng();
        let sigma = 1.0 / (*dof as f64) / self.tau.powi(2);

        self.xi = Normal::new(0.0, sigma).unwrap().sample(&mut rng);
    }
}

impl<B, S, C, M> Thermostat<B, S, C, M> for MTTKThermostat
where
    M: Isothermal + Temperature,
{
    #[inline]
    fn rescaling_factor_step_one<P>(
        &self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
        _compute_properties: P,
    ) -> f64
    where
        P: Fn(&Microstate<B, S, C>) -> (f64, f64),
    {
        // TODO
        1.0
    }

    #[inline]
    fn rescaling_factor_step_two(
        &self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
    ) -> f64 {
        // TODO
        1.0
    }

    #[inline]
    fn advance<P>(&mut self, dt: &f64, compute_properties: P)
    where
        P: Fn(&Microstate<B, S, C>) -> (f64, f64),
    {
        // TODO
    }
}

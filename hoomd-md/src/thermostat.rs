// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Control system temperature.

TODO: Expand documentation.
 */

use rand::Rng;
use rand_distr::{Distribution, Gamma, Normal};
use hoomd_vector::{Vector, Cartesian};
use hoomd_microstate::{boundary::{GenerateGhosts, Wrap}, property::{Acceleration, Mass, Position, Velocity}, Microstate, Transform};
use hoomd_simulation::macrostate::{Isoentropic, Isothermal};

/** Adjust the temperature of a system.

TODO: ensure that docs indicate two-step integration is baked into the thermostat trait
TODO: Add example.
*/
 pub trait Thermostat<M, B, S, C> {
    /// The scaling factor for velocities in the first half-step.
    /// Note that translation and rotation are assumed to have identical math
    /// behind their scaling factors.
    fn rescaling_factor_step_one(
        &self,
        macrostate: &M,
        microstate: &Microstate<B, S, C>,
        dt: f64,
        degrees_of_freedom: u32,
        kinetic_energy: f64,
    ) -> f64;

    /// The scaling factor for velocities in the second half-step.
    /// Note that translation and rotation are assumed to have identical math
    /// behind their scaling factors.
    fn rescaling_factor_step_two(
        &self,
        macrostate: &M,
        microstate: &Microstate<B, S, C>,
        dt: f64,
        degrees_of_freedom: u32,
        kinetic_energy: f64,
    ) -> f64;

    fn advance(&mut self, dt: f64);
}

/** Constant temperature.
TODO: Add example.
*/
pub struct NoThermostat;

impl<M, B, S, C> Thermostat<M, B, S, C> for NoThermostat
where
    M: Isoentropic
{
    #[inline]
    fn rescaling_factor_step_one(
        &self,
        macrostate: &M,
        microstate: &Microstate<B, S, C>,
        dt: f64,
        degrees_of_freedom: u32,
        kinetic_energy: f64,
    ) -> f64 {
        1.0
    }

    #[inline]
    fn rescaling_factor_step_two(
        &self,
        macrostate: &M,
        microstate: &Microstate<B, S, C>,
        dt: f64,
        degrees_of_freedom: u32,
        kinetic_energy: f64,
    ) -> f64 {
        1.0
    }

    #[inline]
    fn advance(&mut self, dt: f64) {}
}

/** Bussi thermostat.
TODO: Add documentation.
TODO: Add example.
*/
pub struct  BussiThermostat {
    /// Thermostat time constant (`[time]`).
    pub tau: f64,
}

/// TODO: add documentation
impl<M, B, S, C> Thermostat<M, B, S, C> for BussiThermostat
where
    M: Isothermal
{
    /** Calculate velocity rescaling factor following the Appendix in https://doi.org/10.1063/1.2408420.
        Bussi requires the rng, instataneous kinetic_energy, timestep and degrees-of-freedom,
        change the trait function definition accordingly?
    */
    #[inline]
    fn rescaling_factor_step_one(
        &self,
        macrostate: &M,
        microstate: &Microstate<B, S, C>,
        kinetic_energy: &f64,
        dt: &f64,
        dof: &i64
    ) -> f64 {
        let kT = macrostate.temperature();

        // panic if momenta was not initialized
        assert!(!(*kinetic_energy == 0.0 && *dof != 0), "Bussi thermostat requires non-zero initial momenta.");

        // get rng from microstate to ensure reproducibility
        let mut rng = microstate.counter().make_rng();

        // trivial case when no particles are present
        if *dof == 0 {
            return 1.0
        }

        // special case when tau is set to 0.
        let mut time_decay_factor = 0.0;
        // normal case time decay factor.
        if self.tau != 0.0 {
            time_decay_factor = (-dt/self.tau).exp();
        }
        // sample random number form standard normal distribution for the first dof.
        let random_normal_one: f64 = Normal::new(0.0, 1.0).unwrap().sample(&mut rng);
        // special case when dof is 1.
        let mut random_gamma: f64 = 0.0;
        // sample random numnber from gamma distribution for the rest of dof
        if *dof > 1 {
            random_gamma = 2.0 * Gamma::new((*dof as f64 - 1.0) / 2.0, 1.0).unwrap().sample(&mut rng);
        }
        // assemble everything
        let v = self.kT / 2.0 / kinetic_energy;
        let term1 = v * (1.0 - time_decay_factor) * (random_gamma + random_normal_one.powi(2));
        let term2 = 2.0 * random_normal_one * (v * (1.0 - time_decay_factor) * time_decay_factor).sqrt();
        (time_decay_factor + term1 + term2).sqrt()
    }

    #[inline]
    fn rescaling_factor_step_two(
        &self,
        macrostate: &M,
        microstate: &Microstate<B, S, C>,
        kinetic_energy: &f64,
        dt: &f64,
        dof: &i64
    ) -> f64 {
        1.0
    }

    #[inline]
    fn advance(&mut self, dt: f64) {}
}


/** MTTK thermostat.
TODO: Add documentation.
TODO: Add example.
*/
pub struct  MTTKThermostat {
    /// Thermostat time constant (`[time]`).
    pub tau: f64,

    pub xi: f64, // TODO: add thermalize method?

    pub eta: f64,
}

impl<M, B, S, C> Thermostat<M, B, S, C> for MTTKThermostat {
    #[inline]
    fn rescaling_factor_step_one(
        &self,
        macrostate: &M,
        microstate: &Microstate<B, S, C>,
        kinetic_energy: &f64,
        dt: &f64,
        dof: &i64
    ) -> f64 {
        // TODO
    }

    #[inline]
    fn rescaling_factor_step_two(
        &self,
        macrostate: &M,
        microstate: &Microstate<B, S, C>,
        kinetic_energy: &f64,
        dt: &f64,
        dof: &i64
    ) -> f64 {
        // TODO
    }

    #[inline]
    fn advance(&mut self, dt: f64) {
        // TODO
    }
}
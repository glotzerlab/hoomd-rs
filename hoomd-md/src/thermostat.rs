// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use rand_distr::{Distribution, Gamma, Normal};
use hoomd_vector::{Vector, Cartesian};
use hoomd_microstate::{boundary::{GenerateGhosts, Wrap}, property::{Acceleration, Mass, Position, Velocity}, Microstate, Transform};
/*! Control system temperature.

TODO: Expand documentation.
 */

/** Adjust the temperature of a system.
TODO: Add example.
*/
 pub trait Thermostat {
    /// The scaling factor for velocities.
    fn temperature_factor(&self) -> f64;
}

/** Constant temperature.
TODO: Add example.
*/
pub struct NoThermostat;

impl Thermostat for NoThermostat {
    #[inline]
    fn temperature_factor(&self) -> f64 {
        1.0
    }
}

/** Bussi thermostat.
TODO: Add documentation.
TODO: Add example.
*/
pub struct  BussiThermostat {
    /// Temperature set point for the thermostat (`[energy]`).
    pub kt: f64,

    /// Thermostat time constant (`[time]`).
    pub tau: f64,
}

/// TODO: add documentation
impl<V, B, S, C> Thermostat for BussiThermostat
where
    V: Default + Vector,
    B: Position<Vector = V> + Velocity<Vector = V> + Acceleration<Vector = V> + Mass + Transform<S> + Clone,
    S: Position<Vector = V> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    /** Calculate velocity rescaling factor following the Appendix in https://doi.org/10.1063/1.2408420.
        Bussi requires the rng, instataneous kinetic_energy, timestep and degrees-of-freedom,
        change the trait function definition accordingly?
    */
    #[inline]
    fn temperature_factor(
        &self,
        microstate: &mut Microstate<B, S, C>,
        kinetic_energy: &f64,
        dt: &f64,
        dof: &i64) -> f64 {
        // panic if momenta was not initialized
        assert!(!(*kinetic_energy == 0.0 && *dof != 0), "Bussi thermostat requires non-zero initial momenta.");

        // get rng from microstate to ensure reproducibility
        let mut rng = microstate.counter().make_rng();

        // trivial case when no particles present.
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
        let v = self.kt / 2.0 / kinetic_energy;
        let term1 = v * (1.0 - time_decay_factor) * (random_gamma + random_normal_one.powi(2));
        let term2 = 2.0 * random_normal_one * (v * (1.0 - time_decay_factor) * time_decay_factor).sqrt();
        (time_decay_factor + term1 + term2).sqrt()
    }

}

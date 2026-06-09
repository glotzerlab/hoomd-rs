// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
use rand::Rng;
use rand_distr::{Distribution, Gamma, Normal};
use serde::{Deserialize, Serialize};

use crate::Thermostat;
use hoomd_simulation::macrostate::Temperature;

// TODO: Apply common derive macros for all types in hoomd_md

/// Stochastic momentum rescaling.
///
/// The time constant $` \tau `$ sets how long the kinetic energy of the system
/// takes to decay to equilibrium.
///
/// When $`\tau`$ is 0, the rescaling factor $` \alpha `$ is:
/// ```math
///  \alpha = \sqrt{\frac{g kT}{K}}
/// ```
/// where $`K`$ is the instantaneous kinetic energy of the corresponding
/// translational or rotational degrees of freedom, $`N`$ is the number of
/// degrees of freedom, and $`g`$ is a random value sampled from the gamma
/// distribution $`\Gamma(N, 1)`$ with the probability density function:
/// ```math
///    f_N(g) = \frac{1}{\Gamma{(N)}} g^{N-1} e^{-g}
/// ```
///
/// When $`\tau`$ is non-zero, $` \alpha `$ is given by:
/// ```math
/// \alpha = \sqrt{e^{-\delta t / \tau}
///      + (1 - e^{-\delta t / \tau}) \frac{(2 g + n^2) kT}{2 K}
///      + 2 n \sqrt{e^{-\delta t / \tau} (1-e^{-\delta t / \tau})
///         \frac{kT}{2 K}}}
/// ```
/// where $`\delta t`$ is the integration time step size and $`n`$ is a random
/// value sampled from the standard normal distribution $`\mathcal{N}(0, 1)`$.
///
/// # Reference
///
/// * [Bussi et al. 2007](https://doi.org/10.1063/1.2408420)
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bussi {
    /// Thermostat time constant $`[ \mathrm{time} ]`$.
    tau: f64,
    
    /// Cumulative energy absorbed by the thermostat $`[ \mathrm{energy} ]`$
    energy: f64,
}

impl Bussi {
    /// Construct a new `Bussi` thermostat with the given time constant.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::thermostat::Bussi;
    ///    
    /// let bussi = Bussi::new(0.1);
    /// ```
    #[inline]
    pub fn new(tau: f64) -> Self {
        Self {
            tau,
            energy: 0.0,
        }
    }
    /// Calculate the energy drift on one time step.
    #[inline]
    fn energy_drift(kinetic_energy_old: f64, rescaling_factor: f64) -> f64 {
        kinetic_energy_old * (1.0 - rescaling_factor.powi(2))
    }

    /// The total energy of the thermostat.
    /// # Example
    ///
    /// ```
    /// use hoomd_md::thermostat::Bussi;
    ///    
    /// let bussi = Bussi::new(0.1);
    /// let energy = bussi.energy();
    /// ```
    #[inline]
    pub fn energy(&self) -> f64 {
        self.energy
    }
}

impl<M> Thermostat<M> for Bussi
where
    M: Temperature,
{
    #[inline]
    fn integrate_step_one<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64
    {
        // Calculate velocity rescaling factor following
        // the Appendix in https://doi.org/10.1063/1.2408420.
        if degrees_of_freedom == 0 {
            return 1.0;
        }

        assert!(kinetic_energy != 0.0,
            "The Bussi thermostat requires non-zero kinetic energy."
        );

        let time_decay_factor = if self.tau == 0.0 {
            0.0
        } else {
            (-delta_t / self.tau).exp()
        };

        let random_normal_one: f64 = Normal::new(0.0, 1.0)
            .expect("normal distribution should be valid")
            .sample(rng);

        let random_gamma = 
            if degrees_of_freedom > 0 {
                2.0 * Gamma::new((degrees_of_freedom as f64 - 1.0) / 2.0, 1.0)
                    .expect("gamma distribution should be valid")
                    .sample(rng)
            } else {
                0.0
            };

        let v = macrostate.temperature() / 2.0 / kinetic_energy;
        let term1 = v * (1.0 - time_decay_factor) * (random_gamma + random_normal_one.powi(2));
        let term2 =
            2.0 * random_normal_one * (v * (1.0 - time_decay_factor) * time_decay_factor).sqrt();
        let rescaling_factor = (time_decay_factor + term1 + term2).sqrt();

        self.energy += Self::energy_drift(kinetic_energy, rescaling_factor);
        rescaling_factor
    }

    #[inline]
    fn integrate_step_two<R: Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        _macrostate: &M,
        _delta_t: f64,
        _kinetic_energy: f64,
        _degrees_of_freedom: usize,
    ) -> f64
    {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::check;
    use rand::{SeedableRng, rngs::StdRng};

    use hoomd_simulation::macrostate::Isothermal;

    #[test]
    fn test_init() {
        let tau = 1.0;
        let bussi = Bussi::new(tau);

        check!(bussi.tau == tau);
        check!(bussi.energy() == 0.0);
    }

    #[test]
    fn test_scale_down() {
        let tau = 0.0;
        let mut bussi = Bussi::new(tau);

        let mut rng = StdRng::seed_from_u64(0);
        let macrostate = Isothermal { temperature: 0.4 };
        let alpha = bussi.integrate_step_one(&mut rng, &macrostate, 0.01, 1000.0, 100);

        check!(alpha < 1.0);
        check!(bussi.energy() > 0.0);
    }

    #[test]
    fn test_scale_up() {
        let tau = 0.0;
        let mut bussi = Bussi::new(tau);

        let mut rng = StdRng::seed_from_u64(0);
        let macrostate = Isothermal { temperature: 0.4 };
        let alpha = bussi.integrate_step_one(&mut rng, &macrostate, 0.01, 1000.0, 10_000);

        check!(alpha > 1.0);
        check!(bussi.energy() < 0.0);
    }
}

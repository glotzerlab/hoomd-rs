// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
use crate::Thermostat;
use hoomd_simulation::macrostate::Temperature;
use hoomd_utility::valid::PositiveReal;
use rand::Rng;
use rand_distr::{Distribution, Gamma, Normal};

/// [`Bussi`] adjust the temperature with a
/// canonical sampling thermostat that uses stochastic
/// velocity rescaling with Hamiltonian dynamics
/// given time constant $`\tau`$.
///
/// When $`\tau`$ is 0, the stochastic evolution of
/// system is instantly thermalized and the
/// rescaling factor $`\alpha`$:
/// ```math
///  \alpha = \sqrt{\frac{g k_BT}{K}}
/// ```
/// where $`K`$ is the instantaneous kinetic energy
/// of the corresponding translational or rotational
/// degrees of freedom, $`N`$ is the number of degrees
/// of freedom, and $`g`$ is a random value sampled from
/// the gamma distribution $`\mathrm{Gamma}(N, 1)`$ with the
/// probability density function:
/// ```math
///    f_N(g) = \frac{1}{\Gamma{(N)}} g^{N-1} e^{-g}
/// ```
///
/// When $`\tau`$ is non-zero, the kinetic energies decay to
/// equilibrium with the given characteristic time
/// constant $`\tau`$ and the $`\alpha`$ is given by:
/// ```math
///    \alpha = \sqrt{e^{-\delta t / \tau}
///         + (1 - e^{-\delta t / \tau}) \frac{(2 g + n^2) kT}{2 K}
///         + 2 n \sqrt{e^{-\delta t / \tau} (1-e^{-\delta t / \tau})
///            \frac{k_BT}{2 K}}}
/// ```
/// where $`\delta t`$ is the step size of [`TranslationalMotion`]
/// or [`RotationalMotion`] and $`n`$ is a random value
/// sampled from the standard normal distribution
/// $`\mathcal{N}(0, 1)`$.
///
/// # Reference
/// [Bussi et al. 2007]
///
/// [Bussi et al. 2007]: <https://doi.org/10.1063/1.2408420>
///
/// # Examples
///
/// ```
/// use hoomd_md::{thermostat::Bussi};
///    
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let dt = 0.001;
/// let tau = 100.0*dt;
/// let thermostat = Bussi::new(tau.try_into()?);
/// # Ok(())
/// # }
/// ```
pub struct Bussi {
    /// Thermostat time constant (`[time]`).
    tau: PositiveReal,
    /// Cumulative energy drift due to the thermostat. Useful for checking energy conservation.
    cumu_energy_drift: f64,
}

impl Bussi {
    /// Constrcut Bussi.
    pub fn new(tau: PositiveReal) -> Self {
        Self {
            tau: tau,
            cumu_energy_drift: 0.0,
        }
    }
    /// Calculate the energy drift due to the thermostat.
    pub fn energy_drift(&self, kinetic_energy_old: &f64, rescaling_factor: &f64) -> f64 {
        kinetic_energy_old * (1.0 - rescaling_factor.powi(2))
    }
    /// Get the energy of thermostat.
    pub fn get_energy(&self) -> &f64 {
        &self.cumu_energy_drift
    }
}

impl<M> Thermostat<M> for Bussi
where
    M: Temperature,
{
    /// Calculate velocity rescaling factor following
    /// the Appendix in <https://doi.org/10.1063/1.2408420>.
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
        let temperature_set_point = macrostate.temperature();

        // panic if momenta was not initialized
        assert!(
            !(kinetic_energy == 0.0 && degrees_of_freedom != 0),
            "Bussi thermostat requires non-zero initial momenta."
        );

        // trivial case when no particles are present
        if degrees_of_freedom == 0 {
            return 1.0;
        }

        // special case when tau is set to 0.
        let mut time_decay_factor = 0.0;

        // normal case time decay factor.
        if self.tau.get() != 0.0 {
            time_decay_factor = (-delta_t / self.tau.get()).exp();
        }

        // sample random number form standard normal distribution for the first dof.
        let random_normal_one: f64 = Normal::new(0.0, 1.0).unwrap().sample(rng);

        // special case when dof is 1.
        let mut random_gamma: f64 = 0.0;

        // sample random numnber from gamma distribution for the rest of dof
        if degrees_of_freedom > 0 {
            random_gamma = 2.0 * Gamma::new((degrees_of_freedom as f64 - 1.0) / 2.0, 1.0).unwrap().sample(rng);
        }

        // assemble everything
        let v = temperature_set_point / 2.0 / kinetic_energy;
        let term1 = v * (1.0 - time_decay_factor) * (random_gamma + random_normal_one.powi(2));
        let term2 =
            2.0 * random_normal_one * (v * (1.0 - time_decay_factor) * time_decay_factor).sqrt();
        let rescaling_factor = (time_decay_factor + term1 + term2).sqrt();

        // accumulate energy drift
        self.cumu_energy_drift += self.energy_drift(&kinetic_energy, &rescaling_factor);
        rescaling_factor
    }

    /// A dummy method that
    /// performs no temperature adjustment as the Bussi thermostat
    /// requires only one step to finish the temeperature adjustment.
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
    use rstest::*;

    #[rstest]
    fn test_init() -> anyhow::Result<()> {
        // Blanket Implementation
        let tau = 1.0;
        let bussi = Bussi::new(tau.try_into()?);

        assert_eq!(tau, bussi.tau.get());
        assert_eq!(0.0, *bussi.get_energy());

        // Instantiation
        let custom_bussi = Bussi {
            tau: tau.try_into()?,
            cumu_energy_drift: 1.0,
        };

        assert_eq!(tau, custom_bussi.tau.get());
        assert_eq!(1.0, custom_bussi.cumu_energy_drift);

        Ok(())
    }

    #[rstest]
    #[should_panic(expected = "tau should be positive: NotPositive(-1.0)")]
    fn test_invalid_tau() {
        let tau = -1.0;
        let _ = Bussi::new(tau.try_into().expect("tau should be positive"));
    }
}

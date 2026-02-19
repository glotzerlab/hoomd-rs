// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
use hoomd_microstate::Microstate;
use hoomd_simulation::macrostate::Temperature;
use hoomd_utility::valid::PositiveReal;
use crate::thermostat::Thermostat;
use rand_distr::{Distribution, Gamma, Normal};

/// [`BussiThermostat`] adjust the temperature with a
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
/// use hoomd_md::{thermostat::BussiThermostat};
///    
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let dt = 0.001;
/// let tau = 100.0*dt;
/// let thermostat = BussiThermostat::new(tau.try_into()?);
/// # Ok(())
/// # }
/// ```
pub struct BussiThermostat {
    /// Thermostat time constant (`[time]`).
    tau: PositiveReal,
    /// Cumulative energy drift due to the thermostat. Useful for checking energy conservation.
    cumu_energy_drift: f64,
}
impl BussiThermostat {
    /// Constrcut BussiThermostat.
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
    /// Get the energy of thermalstat.
    pub fn get_energy(&self) -> &f64 {
        &self.cumu_energy_drift
    }
}

impl<B, S, X, C, M> Thermostat<B, S, X, C, M> for BussiThermostat
where
    B: Clone,
    M: Temperature,
{
    /// Calculate velocity rescaling factor following
    /// the Appendix in <https://doi.org/10.1063/1.2408420>.
    #[inline]
    fn integrate_step_one<P>(
        &mut self,
        microstate: &Microstate<B, S, X, C>,
        macrostate: &M,
        dt: &f64,
        mut compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, X, C>) -> (f64, f64),
    {
        #![allow(non_snake_case)]
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
        if self.tau.get() != 0.0 {
            time_decay_factor = (-dt / self.tau.get()).exp();
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
        let rescaling_factor = (time_decay_factor + term1 + term2).sqrt();

        // accumulate energy drift
        self.cumu_energy_drift += self.energy_drift(&ke, &rescaling_factor);
        rescaling_factor
    }

    /// A dummy method that
    /// performs no temperature adjustment as the Bussi thermostat
    /// requires only one step to finish the temeperature adjustment.
    #[inline]
    fn integrate_step_two<P>(
        &mut self,
        microstate: &Microstate<B, S, X, C>,
        _macrostate: &M,
        _dt: &f64,
        mut compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, X, C>) -> (f64, f64),
    {
        let (_, _) = compute_properties(&microstate);
        1.0
    }
}
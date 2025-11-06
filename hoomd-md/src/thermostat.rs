// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Control system temperature.
//!
//! TODO: Expand documentation.

use hoomd_microstate::{
    Microstate, Transform,
    boundary::{GenerateGhosts, Wrap},
};
use hoomd_simulation::macrostate::{Isothermal, Temperature};
use hoomd_vector::{Cartesian, Vector};
use rand_distr::{Distribution, Gamma, Normal};

/// Adjust the temperature of a system for sampling
/// the kinetic energy in the form of
/// canonical distribution.
///
/// Implement [`Thermostat`] or use one of the
/// provided method in [`thermostat`](crate::thermostat)
/// in MD simulations.
pub trait Thermostat<B, S, C, M> {
    /// Integrate the thermostat dof foward, and return
    /// Note that translation and rotation are assumed to have identical math
    /// behind their scaling factors.
    fn integrate_step_one<P>(
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
    fn integrate_step_two<P>(
        &mut self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
        compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, C>) -> (f64, f64);
}

/// [`NoThermostat`] implement the dummy method
/// that performs no adjustment on the temperature
/// for [`TranslationalMotion`] and [`RotationalMotion`]
/// as they require an input of a [`Thermostat`] during
/// integration.
pub struct NoThermostat;

impl<B, S, C, M> Thermostat<B, S, C, M> for NoThermostat {
    #[inline]
    fn integrate_step_one<P>(
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
    fn integrate_step_two<P>(
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
}

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
/// let dt = 0.001;
/// let tau = 100.0*dt;
/// let thermostat = BussiThermostat::new(tau);
/// ```
pub struct BussiThermostat {
    /// Thermostat time constant (`[time]`).
    tau: f64,
    /// Cumulative energy drift due to the thermostat. Useful for checking energy conservation.
    cumu_energy_drift: f64,
}
impl BussiThermostat {
    /// Constrcut BussiThermostat.
    pub fn new(tau: f64) -> Self {
        assert!(tau >= 0.0, "BussiThermostat requires tau >= 0");
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
/// Calculate velocity rescaling factor following
/// the Appendix in <https://doi.org/10.1063/1.2408420>
/// in `integrate_step_one`.
///
/// `integrate_step_two` is a dummy method that
/// performs no temperature adjustment as the Bussi thermostat
/// requires only one step to finish temeperature adjustment.
impl<B, S, C, M> Thermostat<B, S, C, M> for BussiThermostat
where
    B: Clone,
    M: Temperature,
{
    #[inline]
    fn integrate_step_one<P>(
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
        let rescaling_factor = (time_decay_factor + term1 + term2).sqrt();

        // accumulate energy drift
        self.cumu_energy_drift += self.energy_drift(&ke, &rescaling_factor);
        rescaling_factor
    }

    #[inline]
    fn integrate_step_two<P>(
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
}

/// [`MTTKThermostat`] implement the Nose-Hoover thermostat
/// that adjsut temperture using non-Hamiltonian dynamics
/// given a time constant $`\tau`$.
///
/// It perform time integration on the
/// extra degrees-of-freedom in the non-Hamiltonian
/// equations of motion,
/// which are designed to sample the canonical (nvt).
///
/// [`MTTKThermostat`] store the extra degrees-of-freedom
/// as the one-dimensional thermostat position $`\eta`$ and
/// thermostat velocity $`\xi`$, resulting in the extended
/// Hamiltonian $`H`$
/// ```math
///    H = \frac{K}{\exp{(2\eta)}} + U
///         + N k_BT_\mathrm{setpoint} \eta
///         + N k_BT_\mathrm{setpoint} \frac{1}{2} (\xi\tau)^2
/// ```
/// Where $`K`$ is the kinetic energy of the system, $`U`$ is the
/// potential energy of the system, $`N`$ is the degrees-of-freedom,
/// $`k_BT_\mathrm{setpoint}`$ is the temperature setpoint.
///
/// Following the Trotter decomposition of Liouvillian,
/// [`MTTKThermostat`] integrate the $`\eta`$ and
/// $`\xi`$ forward by half time step $`\frac{\delta t}{2}`$
/// by the following procedure:
///
/// ```math
/// \begin{align}
///
/// &G_\mathrm{old} = \frac{1}{\tau^2} \left( \frac{k_B T_\mathrm{old}}{k_BT_\mathrm{setpoint}} - 1 \right) \\
/// &\xi \left[ t+\frac{\delta t} {4} \right] = \xi[t] + G_\mathrm{old}\frac{\delta t}{4} \\
/// &\alpha = \exp{\left[ -\xi\left[t+\frac{\delta t} {4} \right] \frac{dt}{2} \right]}  \quad\; \text{calculate rescaling factor} \\
/// &k_B T_\mathrm{new} = k_B T_\mathrm{old} \times \alpha^2 \quad\quad\quad\; \text{adjust temperature} \\
/// &\eta \left[ t+\frac{\delta t} {2} \right] = \eta[t] + \xi \left[ t+\frac{\delta t} {4} \right] \frac{\delta t}{2} \\
/// &G_\mathrm{new} = \frac{1}{\tau^2} \left( \frac{k_B T_\mathrm{new} }{k_BT_\mathrm{setpoint}} - 1 \right) \\
/// &\xi \left[ t+\frac{\delta t} {2} \right] = \xi \left[ t+\frac{\delta t} {4} \right] + G_\mathrm{new} \frac{\delta t}{4} \\
///         
/// \end{align}
/// ```
///
/// # Reference
/// [Tuckerman et al. 2006]
///
/// [Tuckerman et al. 2006]: <https://doi.org/10.1088/0305-4470/39/19/S18>
///
/// # Examples
///
/// ```
/// use hoomd_md::{thermostat::MTTKThermostat};
///
/// let dt = 0.001;
/// let tau = 100.0*dt;
/// let thermostat = MTTKThermostat::new(tau);
/// ```
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

    /// Get the energy of thermalstat.
    pub fn get_energy(&self) -> &f64 {
        &self.energy
    }

    /// Get the thermostat position.
    pub fn get_position(&self) -> &f64 {
        &self.eta
    }

    /// Get the thermostat velocity.
    pub fn get_velocity(&self) -> &f64 {
        &self.xi
    }
}

/// Integrate extra degrees-of-freedom and
/// return the velocity rescaling factor, following
/// Tuckerman's work <https://doi.org/10.1088/0305-4470/39/19/S18>
/// in `integrate_step_one`.
///
/// `integrate_step_two` call `intergrate_step_one`,
/// internally
impl<B, S, C, M> Thermostat<B, S, C, M> for MTTKThermostat
where
    B: Clone,
    M: Temperature,
{
    #[inline]
    fn integrate_step_one<P>(
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

        // Calculate current temperature
        let (ke, dof) = compute_properties(&microstate);
        let kT_instantaneous = 2.0 / dof * ke;

        // Thermostat acceleration
        let G = (kT_instantaneous / kT_setpoint - 1.0) / self.tau.powi(2);

        // Update thermostat velocity
        let xi_quater = self.xi + 0.25 * G * dt;

        // Calculate rescaling factor at 0.25*dt
        let rescaling_factor = (-0.5 * xi_quater * dt).exp();

        // Expected update
        let kT_instantaneous_new = kT_instantaneous * (rescaling_factor).powi(2);

        // Update thermostat position
        self.eta += 0.5 * xi_quater * dt;

        // New thermostat acceleration
        let G_new = (kT_instantaneous_new / kT_setpoint - 1.0) / self.tau.powi(2);

        // Update thermostat velocity
        self.xi = xi_quater + 0.25 * G_new * dt;

        // Update thermostat energy
        self.energy = self.thermostat_energy(kT_setpoint, &dof);
        rescaling_factor
    }

    #[inline]
    fn integrate_step_two<P>(
        &mut self,
        microstate: &Microstate<B, S, C>,
        macrostate: &M,
        dt: &f64,
        mut compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, C>) -> (f64, f64),
    {
        self.integrate_step_one(microstate, macrostate, &dt, &mut compute_properties)
    }
}

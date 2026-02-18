// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![allow(non_snake_case)]

use hoomd_microstate::Microstate;
use hoomd_simulation::macrostate::Temperature;
use hoomd_utility::valid::PositiveReal;
use crate::thermostat::Thermostat;
use rand_distr::{Distribution, Normal};

/// [`MTTKThermostat`] implement the Nos$`\text{\'e}`$-Hoover thermostat
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
///    H = K + U
///         + N k_BT_\mathrm{setpoint} \eta
///         +  \frac{1}{2} (N k_BT_\mathrm{setpoint} \tau^2)(\xi)^2
/// ```
/// Where $`K`$ is the kinetic energy of the system, $`U`$ is the
/// potential energy of the system, $`N`$ is the degrees-of-freedom,
/// $`k_BT_\mathrm{setpoint}`$ is the temperature setpoint.
///
/// Following the Trotter decomposition of Liouvillian,
/// [`MTTKThermostat`] integrate the $`\eta`$ and
/// $`\xi`$ forward by half time step $`\frac{\delta t}{2}`$
/// via the following procedure:
///
/// ```math
/// \begin{align}
///
/// &G_\mathrm{old} = \frac{1}{\tau^2} \left( \frac{k_B T_\mathrm{old}}{k_BT_\mathrm{setpoint}} - 1 \right) \\ \nonumber \\ 
/// 
/// &\xi \left\{ t+\frac{\delta t} {4} \right\} = \xi \{ t \} + G_\mathrm{old}\frac{\delta t}{4} \\ \nonumber \\ 
/// 
/// &\alpha = \exp\left[ -\xi \left\{ t+\frac{\delta t} {4} \right\} \frac{dt}{2} \right]  \quad\; \text{calculate rescaling factor} \\ \nonumber \\ 
/// 
/// &k_B T_\mathrm{new} = k_B T_\mathrm{old} \times \alpha^2 \quad\quad\quad\; \text{adjust temperature} \\ \nonumber \\ 
/// 
/// &\eta \left\{ t+\frac{\delta t} {2} \right\} = \eta \{ t \} + \xi \left\{ t+\frac{\delta t} {4} \right\} \frac{\delta t}{2} \\ \nonumber \\ 
/// 
/// &G_\mathrm{new} = \frac{1}{\tau^2} \left( \frac{k_B T_\mathrm{new} }{k_BT_\mathrm{setpoint}} - 1 \right) \\ \nonumber \\ 
/// 
/// &\xi \left\{ t+\frac{\delta t} {2} \right\} = \xi \left\{ t+\frac{\delta t} {4} \right\} + G_\mathrm{new} \frac{\delta t}{4}
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
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let dt = 0.001;
/// let tau = 100.0*dt;
/// let thermostat = MTTKThermostat::new(tau.try_into()?);
/// # Ok(())
/// # }
/// ```
pub struct MTTKThermostat {
    /// Thermostat time constant (`[time]`).
    tau: PositiveReal,
    /// Thermostat velocity.
    xi: f64,
    /// Thermostat position. Refer to the log(s) in Nose-Hoover's EOS.
    eta: f64,
    /// Energy the thermostat contributes to the Hamiltonian. Useful for checking energy conservation.
    energy: f64,
}

impl MTTKThermostat {
    /// Constrcut MTTKThermostat.
    pub fn new(tau: PositiveReal) -> Self {
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
        let sigma = 1.0 / *dof / self.tau.get().powi(2);

        self.xi = Normal::new(0.0, sigma.sqrt()).unwrap().sample(&mut rng);
        self.energy = self.thermostat_energy(kT_setpoint, dof)
    }

    /// Calculate thermostat energy.
    pub fn thermostat_energy(&self, kT_setpoint: &f64, dof: &f64) -> f64 {
        dof * kT_setpoint * (self.eta + 0.5 * (self.xi * self.tau.get()).powi(2))
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

impl<B, S, C, M> Thermostat<B, S, C, M> for MTTKThermostat
where
    B: Clone,
    M: Temperature,
{
    /// Integrate extra degrees-of-freedom and
    /// return the velocity rescaling factor, following
    /// Tuckerman's work <https://doi.org/10.1088/0305-4470/39/19/S18>.
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
        let G = (kT_instantaneous / kT_setpoint - 1.0) / self.tau.get().powi(2);

        // Update thermostat velocity
        let xi_quater = self.xi + 0.25 * G * dt;

        // Calculate rescaling factor at 0.25*dt
        let rescaling_factor = (-0.5 * xi_quater * dt).exp();

        // Expected update
        let kT_instantaneous_new = kT_instantaneous * (rescaling_factor).powi(2);

        // Update thermostat position
        self.eta += 0.5 * xi_quater * dt;

        // New thermostat acceleration
        let G_new = (kT_instantaneous_new / kT_setpoint - 1.0) / self.tau.get().powi(2);

        // Update thermostat velocity
        self.xi = xi_quater + 0.25 * G_new * dt;

        // Update thermostat energy
        self.energy = self.thermostat_energy(kT_setpoint, &dof);
        rescaling_factor
    }

    /// Call [`integrate_step_one`](MTTKThermostat::integrate_step_one) internally.
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

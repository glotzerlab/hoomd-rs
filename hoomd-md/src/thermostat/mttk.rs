// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![allow(non_snake_case)]

use crate::thermostat::Thermostat;
use hoomd_microstate::Microstate;
use hoomd_simulation::macrostate::Temperature;
use hoomd_utility::valid::PositiveReal;
use rand::Rng;
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
    pub fn thermalize<B, S, X, C, M>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        degrees_of_freedom: usize,
    ) where
        M: Temperature,
    {
        let temperature_set_point = *macrostate.temperature();
        let mut rng = microstate.counter().make_rng();
        let sigma = 1.0 / (degrees_of_freedom as f64) / self.tau.get().powi(2);

        self.xi = Normal::new(0.0, sigma.sqrt()).unwrap().sample(&mut rng);
        self.energy = self.thermostat_energy(temperature_set_point, degrees_of_freedom);

        microstate.increment_substep();
    }

    /// Calculate thermostat energy.
    pub fn thermostat_energy(&self, temperature_set_point: f64, degrees_of_freedom: usize) -> f64 {
        (degrees_of_freedom as f64) * temperature_set_point * (self.eta + 0.5 * (self.xi * self.tau.get()).powi(2))
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

impl<M> Thermostat<M> for MTTKThermostat
where
    M: Temperature,
{
    /// Integrate extra degrees-of-freedom and
    /// return the velocity rescaling factor, following
    /// Tuckerman's work <https://doi.org/10.1088/0305-4470/39/19/S18>.
    #[inline]
    fn integrate_step_one<R: Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64
    {
        let temperature_set_point = *macrostate.temperature();

        // Calculate current temperature
        let kinetic_temperature = 2.0 * kinetic_energy / (degrees_of_freedom as f64);

        // Thermostat acceleration
        let G = (kinetic_temperature / temperature_set_point - 1.0) / self.tau.get().powi(2);

        // Update thermostat velocity
        let xi_quater = self.xi + 0.25 * G * delta_t;

        // Calculate rescaling factor at 0.25*delta_t
        let rescaling_factor = (-0.5 * xi_quater * delta_t).exp();

        // Expected update
        let kT_instantaneous_new = kinetic_temperature * (rescaling_factor).powi(2);

        // Update thermostat position
        self.eta += 0.5 * xi_quater * delta_t;

        // New thermostat acceleration
        let G_new = (kT_instantaneous_new / temperature_set_point - 1.0) / self.tau.get().powi(2);

        // Update thermostat velocity
        self.xi = xi_quater + 0.25 * G_new * delta_t;

        // Update thermostat energy
        self.energy = self.thermostat_energy(temperature_set_point, degrees_of_freedom);
        rescaling_factor
    }

    /// Call [`integrate_step_one`](MTTKThermostat::integrate_step_one) internally.
    #[inline]
    fn integrate_step_two<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64
    {
        self.integrate_step_one(rng, macrostate, delta_t, kinetic_energy, degrees_of_freedom)
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
        let mttk = MTTKThermostat::new(tau.try_into()?);

        assert_eq!(tau, mttk.tau.get());
        assert_eq!(0.0, *mttk.get_velocity());
        assert_eq!(0.0, *mttk.get_position());
        assert_eq!(0.0, *mttk.get_energy());

        // Instantiation
        let custom_mttk = MTTKThermostat {
            tau: tau.try_into()?,
            xi: 1.0,
            eta: 2.0,
            energy: 3.0,
        };

        assert_eq!(tau, custom_mttk.tau.get());
        assert_eq!(1.0, custom_mttk.xi);
        assert_eq!(2.0, custom_mttk.eta);
        assert_eq!(3.0, custom_mttk.energy);

        Ok(())
    }

    #[rstest]
    #[should_panic(expected = "tau should be positive: NotPositive(-1.0)")]
    fn test_invalid_tau() {
        let tau = -1.0;
        let _ = MTTKThermostat::new(tau.try_into().expect("tau should be positive"));
    }
}

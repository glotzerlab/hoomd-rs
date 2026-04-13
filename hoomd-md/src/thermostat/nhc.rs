// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![allow(non_snake_case)]

use crate::thermostat::Thermostat;
use arrayvec::ArrayVec;
use hoomd_microstate::Microstate;
use hoomd_simulation::macrostate::Temperature;
use hoomd_utility::valid::PositiveReal;
use rand_distr::{Distribution, Normal};

/// [`NHCThermostat`] implement the Nos$`\text{\'e}`$-Hoover chain thermostat
/// that adjsut temperture using non-Hamiltonian dynamics
/// given a time constant $`\tau`$.
///
/// The generic type names are:
/// * `N`: The length of thermostat chain. Must be larger than zero.
/// When the `N` is one, the behaviour reduces to
/// [`MTTKThermostat`](crate::thermostat::MTTKThermostat).
///
/// It perform time integration on the
/// extra degrees-of-freedom in the non-Hamiltonian
/// equations of motion,
/// which are designed to sample the canonical ensemble (nvt).
///
/// # Reference
/// [Tuckerman et al. 2006]
///
/// [Tuckerman et al. 2006]: <https://doi.org/10.1088/0305-4470/39/19/S18>
///
/// # Examples
///
/// ```
/// use hoomd_md::{thermostat::NHCThermostat};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// const N_CHAIN_LENGTH: usize = 10;
/// let dt = 0.001;
/// let tau = 100.0*dt;
/// let thermostat = NHCThermostat::<N_CHAIN_LENGTH>::new(tau.try_into()?);
/// # Ok(())
/// # }
/// ```
pub struct NHCThermostat<const N: usize> {
    /// Thermostat time constant (`[time]`).
    tau: PositiveReal,
    /// Chain of thermostat velocity.
    xi_arr: ArrayVec<f64, N>,
    /// Chain of thermostat position. Refer to the log(s) in Nose-Hoover's EOS.
    eta_arr: ArrayVec<f64, N>,
    /// Chain of thermostat acceleration.
    g_arr: ArrayVec<f64, N>,
    /// Chain of thermostat mass.
    q_arr: ArrayVec<f64, N>,
    /// Energy the thermostat contributes to the Hamiltonian. Useful for checking energy conservation.
    energy: f64,
}

impl<const N: usize> NHCThermostat<N> {
    /// Constrcut NHCThermostat.
    pub fn new(tau: PositiveReal) -> Self {
        Self {
            tau: tau,
            xi_arr: ArrayVec::from([0.0; N]),
            eta_arr: ArrayVec::from([0.0; N]),
            g_arr: ArrayVec::from([0.0; N]),
            q_arr: ArrayVec::from([0.0; N]),
            energy: 0.0,
        }
    }

    /// Choose random initial values for the thermostat momentum.
    pub fn thermalize<B, S, X, C, M>(
        &mut self,
        microstate: &Microstate<B, S, X, C>,
        macrostate: &M,
        dof: &f64,
    ) where
        M: Temperature,
    {
        let kT_setpoint = macrostate.temperature();
        let mut rng = microstate.counter().make_rng();
        let sigma0 = 1.0 / (*dof).sqrt() / self.tau.get();
        let sigma_other = 1.0 / self.tau.get();

        self.xi_arr[0] = Normal::new(0.0, sigma0).unwrap().sample(&mut rng);
        for idx in 1..N {
            self.xi_arr[idx] = Normal::new(0.0, sigma_other).unwrap().sample(&mut rng);
        }
        self.energy = self.thermostat_energy(kT_setpoint, dof)
    }

    /// Calculate thermostat energy.
    pub fn thermostat_energy(&self, kT_setpoint: &f64, dof: &f64) -> f64 {
        let mut energy = 0.0;
        energy +=
            dof * kT_setpoint * self.eta_arr[0] + 0.5 * self.q_arr[0] * (self.xi_arr[0]).powi(2);
        for idx in 1..N {
            energy += kT_setpoint * self.eta_arr[idx]
                + 0.5 * self.q_arr[idx] * (self.xi_arr[idx]).powi(2);
        }
        energy
    }

    /// Get the energy of thermalstat.
    pub fn get_energy(&self) -> &f64 {
        &self.energy
    }

    /// Get the chain of position.
    pub fn get_position_arr(&self) -> &ArrayVec<f64, N> {
        &self.eta_arr
    }

    /// Get the chain of velocity.
    pub fn get_velocity_arr(&self) -> &ArrayVec<f64, N> {
        &self.xi_arr
    }
}

impl<const N: usize, B, S, X, C, M> Thermostat<B, S, X, C, M> for NHCThermostat<N>
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
        microstate: &Microstate<B, S, X, C>,
        macrostate: &M,
        dt: &f64,
        mut compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, X, C>) -> (f64, f64),
    {
        // Get current temperature setpoint.
        let kT_setpoint = macrostate.temperature();

        // Calculate current kinetic energy and dof of the system.
        let (ke, dof) = compute_properties(&microstate);

        // Get current kinetic energy setpoint.
        let nkT_setpoint = dof * kT_setpoint;

        // Update chain of mass
        self.q_arr[0] = nkT_setpoint * self.tau.get().powi(2);
        for idx in 1..N {
            self.q_arr[idx] = kT_setpoint * self.tau.get().powi(2);
        }

        // Update the thermostat acceleration coupled to the real system
        self.g_arr[0] = (2.0 * ke - nkT_setpoint) / self.q_arr[0];

        // Update the chain of velocity
        // start from the last one
        self.xi_arr[N - 1] += 0.25 * dt * self.g_arr[N - 1];
        // update the rest
        for idx in (0..N - 1).rev() {
            let xi_rescaling_factor = (-0.125 * dt * self.xi_arr[idx + 1]).exp();
            self.xi_arr[idx] *= xi_rescaling_factor;
            self.xi_arr[idx] += 0.25 * dt * self.g_arr[idx];
            self.xi_arr[idx] *= xi_rescaling_factor;
        }

        // calculate real velocity rescaling factor
        let rescaling_factor = (-0.5 * dt * self.xi_arr[0]).exp();

        // Expected temperature update
        let ke_new = ke * (rescaling_factor).powi(2);

        // Update the thermostat acceleration coupled to the real system
        self.g_arr[0] = (2.0 * ke_new - nkT_setpoint) / self.q_arr[0];

        // Update the chain of position
        for idx in 0..N {
            self.eta_arr[idx] += 0.5 * dt * self.xi_arr[idx];
        }

        // Update the chain of velocity
        // start from the first one
        if N > 1 {
            let xi_rescaling_factor = (-0.125 * dt * self.xi_arr[1]).exp();
            self.xi_arr[0] *= xi_rescaling_factor;
            self.xi_arr[0] += 0.25 * dt * self.g_arr[0];
            self.xi_arr[0] *= xi_rescaling_factor;
        } else {
            self.xi_arr[0] += 0.25 * dt * self.g_arr[0];
        }
        // update the rest
        // the chain of acceleration need to be updated here (have done the first one)
        for idx in 1..N - 1 {
            let xi_rescaling_factor = (-0.125 * dt * self.xi_arr[idx + 1]).exp();
            self.xi_arr[idx] *= xi_rescaling_factor;
            self.g_arr[idx] = (self.q_arr[idx - 1] * (self.xi_arr[idx - 1]).powi(2) - kT_setpoint)
                / self.q_arr[idx];
            self.xi_arr[idx] += 0.25 * dt * self.g_arr[idx];
            self.xi_arr[idx] *= xi_rescaling_factor;
        }
        // special for the last one
        if N > 1 {
            self.g_arr[N - 1] = (self.q_arr[N - 2] * (self.xi_arr[N - 2]).powi(2) - kT_setpoint)
                / self.q_arr[N - 1];
            self.xi_arr[N - 1] += 0.25 * dt * self.g_arr[N - 1];
        }

        self.energy = self.thermostat_energy(kT_setpoint, &dof);
        rescaling_factor
    }

    /// Call [`integrate_step_one`](NHCThermostat::integrate_step_one) internally.
    #[inline]
    fn integrate_step_two<P>(
        &mut self,
        microstate: &Microstate<B, S, X, C>,
        macrostate: &M,
        dt: &f64,
        mut compute_properties: P,
    ) -> f64
    where
        P: FnMut(&Microstate<B, S, X, C>) -> (f64, f64),
    {
        self.integrate_step_one(microstate, macrostate, &dt, &mut compute_properties)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn test_init() -> anyhow::Result<()> {
        // Blanket Implementation
        const N_CHAINS: usize = 5;
        let tau = 1.0;
        let nhc = NHCThermostat::<N_CHAINS>::new(tau.try_into()?);

        assert_eq!(tau, nhc.tau.get());
        assert_eq!([0.0; N_CHAINS], nhc.get_position_arr().as_slice());
        assert_eq!([0.0; N_CHAINS], nhc.get_velocity_arr().as_slice());
        assert_eq!(0.0, nhc.energy);

        // Instantiation
        let custom_nhc = NHCThermostat::<N_CHAINS> {
            tau: tau.try_into()?,
            xi_arr: ArrayVec::from([1.0; N_CHAINS]),
            eta_arr: ArrayVec::from([2.0; N_CHAINS]),
            g_arr: ArrayVec::from([3.0; N_CHAINS]),
            q_arr: ArrayVec::from([4.0; N_CHAINS]),
            energy: 5.0,
        };

        assert_eq!(tau, custom_nhc.tau.get());
        assert_eq!([1.0; N_CHAINS], custom_nhc.xi_arr.as_slice());
        assert_eq!([2.0; N_CHAINS], custom_nhc.eta_arr.as_slice());
        assert_eq!([3.0; N_CHAINS], custom_nhc.g_arr.as_slice());
        assert_eq!([4.0; N_CHAINS], custom_nhc.q_arr.as_slice());
        assert_eq!(5.0, custom_nhc.energy);

        Ok(())
    }

    #[rstest]
    #[should_panic(expected = "tau should be positive: NotPositive(-1.0)")]
    fn test_invalid_tau() {
        let tau = -1.0;
        let _ = NHCThermostat::<1>::new(tau.try_into().expect("tau should be positive"));
    }
}

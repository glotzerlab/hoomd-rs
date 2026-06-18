// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::Thermostat;
use hoomd_simulation::macrostate::Temperature;
use hoomd_utility::valid::PositiveReal;

/// Chain of Nosé-Hoover thermostats.
///
/// [`NoséHooverChain`] adds new degrees of freedom ($`\eta_i`$)
/// to a molecular dynamics simulation in such a way that the existing
/// degrees of freedom sample a constant temperature ensemble. Each
/// [`NoséHooverChain`] instance stores the $`\eta_i`$ and their momenta,
/// $`\xi_i`$, internally.
///
/// The dynamics of each $`\eta_i, \xi_i`$ are similar to that in
/// [`MartynaTuckermanTobiasKlein`], but they are also chained together.
/// See [Martyna et al. 1992] for details.
///
/// [`MartynaTuckermanTobiasKlein`]: crate::thermostat::MartynaTuckermanTobiasKlein
///
/// # Reference
///
/// * [Martyna et al. 1992]
///
/// [Martyna et al. 1992]: https://doi.org/10.1063/1.463940
///
/// # Examples
///
/// ```
/// use hoomd_md::{thermostat::NoséHooverChain};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// const N_CHAIN_LENGTH: usize = 10;
/// let dt = 0.001;
/// let tau = 100.0*dt;
/// let thermostat = NoséHooverChain::<N_CHAIN_LENGTH>::zero(tau.try_into()?);
/// # Ok(())
/// # }
/// ```
#[serde_as]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoséHooverChain<const N: usize> {
    /// Thermostat time constant.
    tau: PositiveReal,
    
    /// Chain of thermostat momenta.
    #[serde_as(as = "[_; N]")]
    xi: [f64; N],

    /// Chain of thermostat positions.
    #[serde_as(as = "[_; N]")]
    eta: [f64; N],

    /// Chain of thermostat accelerations.
    #[serde_as(as = "[_; N]")]
    g: [f64; N],

    /// Energy the thermostat contributes to the Hamiltonian.
    energy: f64,
}

impl<const N: usize> NoséHooverChain<N> {
    /// Construct a new `NoséHooverChain` thermostat with the given time constant,
    /// $` \xi_i = 0 `$, and $` \eta_i = 0 `$ .
    ///
    /// This initial condition is likely to be very far from equilibrium which
    /// will result in wild kinetic energy oscillations for the first hundred to
    /// thousand time steps. Use [`thermalized`] to choose the initial position
    /// and momentum from a thermal distribution.
    ///
    /// [`thermalized`]: Self::thermalized
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::thermostat::NoséHooverChain;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let thermostat = NoséHooverChain::<3>::zero(0.5.try_into()?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn zero(tau: PositiveReal) -> Self {
        Self {
            tau,
            xi: [0.0; N],
            eta: [0.0; N],
            g: [0.0; N],
            energy: 0.0,
        }
    }

    /// Construct a new `NoséHooverChain` thermostat with random $` \xi_i `$
    /// drawn from a thermal distribution.
    ///
    /// # Panics
    ///
    /// This method will panic when `degrees_of_freedom` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Body, Microstate, property::{DynamicPoint, Point}};
    /// use hoomd_vector::Cartesian;
    /// use hoomd_md::{TranslationalKineticEnergy, thermostat::NoséHooverChain};
    /// use hoomd_simulation::macrostate::Isothermal;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::builder()
    ///     .bodies([
    ///         Body { properties: DynamicPoint {
    ///           position: Cartesian::from([1.0, 2.0]),
    ///           ..Default::default()
    ///           },
    ///           sites: vec![Point::default()],
    ///           },
    ///         Body { properties: DynamicPoint {
    ///           position: Cartesian::from([-2.0, 3.0]),
    ///           ..Default::default()
    ///           },
    ///           sites: vec![Point::default()],
    ///           },
    ///     ])
    ///     .try_build()?;
    ///
    /// let macrostate = Isothermal { temperature: 1.5 };
    /// let mut rng = microstate.counter().make_rng();
    /// let translational_thermostat = NoséHooverChain::<3>::thermalized(&mut rng, 0.5.try_into()?, &macrostate, microstate.translational_kinetic_energy().1);
    /// microstate.increment_substep();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn thermalized<M, R: Rng + ?Sized>(
        rng: &mut R,
        tau: PositiveReal,
        macrostate: &M,
        degrees_of_freedom: usize,
    ) -> Self
    where
        M: Temperature,
    {
        let sigma_0 = 1.0 / (degrees_of_freedom as f64).sqrt() / tau.get();
        let sigma_other = 1.0 / tau.get();

        let mut xi = [0.0; N];

        xi[0] = Normal::new(0.0, sigma_0).expect("Normal distribution should be valid").sample(rng);
        
        for xi_i in xi.iter_mut().skip(1) {
            *xi_i = Normal::new(0.0, sigma_other).expect("Normal distribution should be valid").sample(rng);
        }

        let mut result = Self {
            tau,
            xi,
            eta: [0.0; N],
            g: [0.0; N],
            energy: 0.0,
        };

        let q = result.q(*macrostate.temperature(), degrees_of_freedom);
        result.energy = result.thermostat_energy(*macrostate.temperature(), degrees_of_freedom, &q);

        result
    }

    /// Calculate q.
    #[inline]
    fn q(&self, temperature: f64, degrees_of_freedom: usize) -> [f64; N] {
        let n_k_t = (degrees_of_freedom as f64) * temperature;
        let mut result = [temperature * self.tau.get().powi(2); N];
        
        result[0] = n_k_t * self.tau.get().powi(2);

        result
    }

    /// Calculate thermostat energy.
    #[inline]
    fn thermostat_energy(&self, temperature_set_point: f64, degrees_of_freedom: usize, q: &[f64; N]) -> f64 {
        let mut energy = 0.0;
        energy +=
            (degrees_of_freedom as f64) * temperature_set_point * self.eta[0] + 0.5 * q[0] * (self.xi[0]).powi(2);
    
        for (eta_i, (q_i, xi_i)) in self.eta.iter().zip(q.iter().zip(self.xi)) {
            energy += temperature_set_point * eta_i
                + 0.5 * q_i * (xi_i).powi(2);
        }
        energy
    }

    /// The total energy of the thermostat.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::thermostat::NoséHooverChain;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let thermostat = NoséHooverChain::<3>::zero(0.5.try_into()?);
    ///
    /// let energy = thermostat.energy();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn energy(&self) -> f64 {
        self.energy
    }

    /// The thermostat's position.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::thermostat::NoséHooverChain;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let thermostat = NoséHooverChain::<3>::zero(0.5.try_into()?);
    ///
    /// let eta = thermostat.eta();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn eta(&self) -> &[f64; N] {
        &self.eta
    }

    /// The thermostat's momentum.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::thermostat::NoséHooverChain;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let thermostat = NoséHooverChain::<3>::zero(0.5.try_into()?);
    ///
    /// let xi = thermostat.xi();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn xi(&self) -> &[f64; N] {
        &self.xi
    }
}

impl<const N: usize, M> Thermostat<M> for NoséHooverChain<N>
where
    M: Temperature,
{
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
        // Integrate extra degrees-of-freedom and
        // return the velocity rescaling factor, following
        // Tuckerman's work <https://doi.org/10.1088/0305-4470/39/19/S18>.

        let n_k_t = (degrees_of_freedom as f64) * *macrostate.temperature();
        let q = self.q(*macrostate.temperature(), degrees_of_freedom);

        // Update the thermostat acceleration coupled to the real system
        self.g[0] = (2.0 * kinetic_energy - n_k_t) / q[0];

        // Update the chain of velocity
        // start from the last one
        self.xi[N - 1] += 0.25 * delta_t * self.g[N - 1];
        // update the rest
        for idx in (0..N - 1).rev() {
            let xi_rescaling_factor = (-0.125 * delta_t * self.xi[idx + 1]).exp();
            self.xi[idx] *= xi_rescaling_factor;
            self.xi[idx] += 0.25 * delta_t * self.g[idx];
            self.xi[idx] *= xi_rescaling_factor;
        }

        // calculate real velocity rescaling factor
        let rescaling_factor = (-0.5 * delta_t * self.xi[0]).exp();

        // Expected temperature update
        let kinetic_energy_new = kinetic_energy * rescaling_factor.powi(2);

        // Update the thermostat acceleration coupled to the real system
        self.g[0] = (2.0 * kinetic_energy_new - n_k_t) / q[0];

        // Update the chain of position
        for idx in 0..N {
            self.eta[idx] += 0.5 * delta_t * self.xi[idx];
        }

        // Update the chain of velocity
        // start from the first one
        if N > 1 {
            let xi_rescaling_factor = (-0.125 * delta_t * self.xi[1]).exp();
            self.xi[0] *= xi_rescaling_factor;
            self.xi[0] += 0.25 * delta_t * self.g[0];
            self.xi[0] *= xi_rescaling_factor;
        } else {
            self.xi[0] += 0.25 * delta_t * self.g[0];
        }
        // update the rest
        // the chain of acceleration need to be updated here (have done the first one)
        for idx in 1..N - 1 {
            let xi_rescaling_factor = (-0.125 * delta_t * self.xi[idx + 1]).exp();
            self.xi[idx] *= xi_rescaling_factor;
            self.g[idx] = (q[idx - 1] * (self.xi[idx - 1]).powi(2) - *macrostate.temperature())
                / q[idx];
            self.xi[idx] += 0.25 * delta_t * self.g[idx];
            self.xi[idx] *= xi_rescaling_factor;
        }
        // special for the last one
        if N > 1 {
            self.g[N - 1] = (q[N - 2] * (self.xi[N - 2]).powi(2) - *macrostate.temperature())
                / q[N - 1];
            self.xi[N - 1] += 0.25 * delta_t * self.g[N - 1];
        }

        self.energy = self.thermostat_energy(*macrostate.temperature(), degrees_of_freedom, &q);
        rescaling_factor
    }

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
    use assert2::check;

    use crate::TranslationalKineticEnergy;
    use hoomd_microstate::{Body, Microstate, property::{DynamicPoint, Point}};
    use hoomd_vector::Cartesian;
    use hoomd_simulation::macrostate::Isothermal;

    #[test]
    fn test_zero() -> anyhow::Result<()> {
        let thermostat = NoséHooverChain::<10>::zero(0.5.try_into()?);

        check!(thermostat.tau.get() == 0.5);
        check!(thermostat.xi() == &[0.0; 10]);
        check!(thermostat.eta() == &[0.0; 10]);
        check!(thermostat.energy() == 0.0);

        Ok(())
    }

    #[test]
    fn test_thermalized() -> anyhow::Result<()> {
        let microstate = Microstate::builder()
            .bodies([
                Body { properties: DynamicPoint {
                  position: Cartesian::from([1.0, 2.0]),
                  ..Default::default()
                  },
                  sites: vec![Point::default()],
                  },
                Body { properties: DynamicPoint {
                  position: Cartesian::from([-2.0, 3.0]),
                  ..Default::default()
                  },
                  sites: vec![Point::default()],
                  },
            ])
            .try_build()?;

        let macrostate = Isothermal { temperature: 1.5 };
        let mut rng = microstate.counter().make_rng();
        let thermostat = NoséHooverChain::<10>::thermalized(&mut rng, 0.5.try_into()?, &macrostate, microstate.translational_kinetic_energy().1);

        check!(thermostat.tau.get() == 0.5);
        check!(thermostat.xi() != &[0.0; 10]);
        check!(thermostat.eta() == &[0.0; 10]);
        check!(thermostat.energy() != 0.0);

        Ok(())
    }
}

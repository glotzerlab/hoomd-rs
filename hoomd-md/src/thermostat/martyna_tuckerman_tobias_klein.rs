// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `MartynaTuckermanTobiasKlein`

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::Thermostat;
use hoomd_simulation::macrostate::Temperature;
use hoomd_utility::valid::PositiveReal;

/// Nosé-Hoover thermostat.
///
/// [`MartynaTuckermanTobiasKlein`] adds a new degree of freedom ($`\eta`$)
/// to a molecular dynamics simulation in such a way that the existing
/// degrees of freedom sample a constant temperature ensemble. Each
/// [`MartynaTuckermanTobiasKlein`] instance stores $`\eta`$ and its momentum,
/// $`\xi`$, internally.
///
/// The extended Hamiltonian $`H`$ is:
/// ```math
///    H = K + U
///         + N kT \eta
///         +  \frac{1}{2} N kT \tau^2\xi^2
/// ```
/// Where $`K`$ is the kinetic energy of the system, $`U`$ is the potential
/// energy of the system, $`N`$ is the number of degrees of freedom, and $`kT`$
/// is the temperature.
///
/// Following the Trotter decomposition of Liouvillian,
/// [`MartynaTuckermanTobiasKlein`] integrates $`\eta`$ and $`\xi`$ forward
/// by half time step $`\frac{\delta t}{2}`$ via the following procedure:
///
/// ```math
/// \begin{align*}
///
/// G_\mathrm{old} &= \frac{1}{\tau^2} \left( \frac{2 K}{N kT} - 1 \right) \\
///
/// \xi \left\{ t+\frac{\delta t} {4} \right\} &= \xi \{ t \} + G_\mathrm{old}\frac{\delta t}{4} \\
///
/// \alpha &= \exp\left[ -\xi \left\{ t+\frac{\delta t} {4} \right\} \frac{dt}{2} \right]  \\
///
/// K_{new} &= K \alpha^2 \\
///
/// \eta \left\{ t+\frac{\delta t} {2} \right\} &= \eta \{ t \} + \xi \left\{ t+\frac{\delta t} {4} \right\} \frac{\delta t}{2} \\
///
/// G_\mathrm{new} &= \frac{1}{\tau^2} \left( \frac{2 K_\mathrm{new} }{kT} - 1 \right) \\
///
/// \xi \left\{ t+\frac{\delta t} {2} \right\} &= \xi \left\{ t+\frac{\delta t} {4} \right\} + G_\mathrm{new} \frac{\delta t}{4}
///
/// \end{align*}
/// ```
///
/// # Warning
///
/// [`MartynaTuckermanTobiasKlein`] fails to sample the correct distribution when there are
/// strong harmonic interactions in the system. In such situations, use
/// [`Bussi`] or [`NoséHooverChain`] instead.
///
/// [`Bussi`]: crate::thermostat::Bussi
/// [`NoséHooverChain`]: crate::thermostat::NoséHooverChain
///
/// # References
/// * [Tuckerman et al. 2006](https://doi.org/10.1088/0305-4470/39/19/S18)
/// * [Martyna et al. 1994](https://doi.org/10.1063/1.467468)
///
/// # Example
///
/// ```
/// use hoomd_md::thermostat::MartynaTuckermanTobiasKlein;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let thermostat = MartynaTuckermanTobiasKlein::zero(0.5.try_into()?);
/// # Ok(())
/// # }
/// ```
#[doc(alias = "mttk")]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MartynaTuckermanTobiasKlein {
    /// Thermostat time constant.
    tau: PositiveReal,
    /// Thermostat velocity.
    xi: f64,
    /// Thermostat position.
    eta: f64,
    /// Energy the thermostat contributes to the Hamiltonian.
    energy: f64,
}

impl MartynaTuckermanTobiasKlein {
    /// Construct a new `MartynaTuckermanTobiasKlein` thermostat with the given time constant,
    /// $` \xi = 0 `$, and $` \eta = 0 `$ .
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
    /// use hoomd_md::thermostat::MartynaTuckermanTobiasKlein;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let thermostat = MartynaTuckermanTobiasKlein::zero(0.5.try_into()?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn zero(tau: PositiveReal) -> Self {
        Self {
            tau,
            xi: 0.0,
            eta: 0.0,
            energy: 0.0,
        }
    }

    /// Construct a new `MartynaTuckermanTobiasKlein` thermostat with a random $` \xi `$
    /// drawn from a thermal distribution.
    ///
    /// # Panics
    ///
    /// This method will panic when `degrees_of_freedom` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::{
    ///     TranslationalKineticEnergy, thermostat::MartynaTuckermanTobiasKlein,
    /// };
    /// use hoomd_microstate::{
    ///     Body, Microstate,
    ///     property::{DynamicPoint, Point},
    /// };
    /// use hoomd_simulation::macrostate::Isothermal;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::builder()
    ///     .bodies([
    ///         Body::single_site(
    ///             DynamicPoint {
    ///                 position: Cartesian::from([1.0, 2.0]),
    ///                 ..Default::default()
    ///             },
    ///             Point::default(),
    ///         ),
    ///         Body::single_site(
    ///             DynamicPoint {
    ///                 position: Cartesian::from([-2.0, 3.0]),
    ///                 ..Default::default()
    ///             },
    ///             Point::default(),
    ///         ),
    ///     ])
    ///     .try_build()?;
    ///
    /// let macrostate = Isothermal { temperature: 1.5 };
    /// let mut rng = microstate.counter().make_rng();
    /// let translational_thermostat = MartynaTuckermanTobiasKlein::thermalized(
    ///     &mut rng,
    ///     0.5.try_into()?,
    ///     &macrostate,
    ///     microstate.translational_kinetic_energy().1,
    /// );
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
        let sigma = 1.0 / (degrees_of_freedom as f64 * tau.get().powi(2));

        let xi = Normal::new(0.0, sigma.sqrt())
            .expect("Normal distribution should be valid")
            .sample(rng);

        let mut result = Self {
            tau,
            xi,
            eta: 0.0,
            energy: 0.0,
        };

        result.energy = result.thermostat_energy(*macrostate.temperature(), degrees_of_freedom);

        result
    }

    /// Calculate the thermostats energy.
    #[inline]
    fn thermostat_energy(&self, temperature_set_point: f64, degrees_of_freedom: usize) -> f64 {
        (degrees_of_freedom as f64)
            * temperature_set_point
            * (self.eta + 0.5 * (self.xi * self.tau.get()).powi(2))
    }

    /// The total energy of the thermostat.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::thermostat::MartynaTuckermanTobiasKlein;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let thermostat = MartynaTuckermanTobiasKlein::zero(0.5.try_into()?);
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
    /// use hoomd_md::thermostat::MartynaTuckermanTobiasKlein;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let thermostat = MartynaTuckermanTobiasKlein::zero(0.5.try_into()?);
    ///
    /// let eta = thermostat.eta();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn eta(&self) -> f64 {
        self.eta
    }

    /// The thermostat's momentum.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::thermostat::MartynaTuckermanTobiasKlein;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let thermostat = MartynaTuckermanTobiasKlein::zero(0.5.try_into()?);
    ///
    /// let xi = thermostat.xi();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn xi(&self) -> f64 {
        self.xi
    }
}

impl<M> Thermostat<M> for MartynaTuckermanTobiasKlein
where
    M: Temperature,
{
    #[inline]
    fn integrate_half_step_one<R: Rng + ?Sized>(
        &mut self,
        _rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64 {
        // Integrate extra degrees-of-freedom and return the
        // velocity rescaling factor, following Tuckerman's work
        // https://doi.org/10.1088/0305-4470/39/19/S18.

        let kinetic_temperature = 2.0 * kinetic_energy / (degrees_of_freedom as f64);
        let g = (kinetic_temperature / *macrostate.temperature() - 1.0) / self.tau.get().powi(2);
        let xi_quarter = self.xi + 0.25 * g * delta_t;
        let rescaling_factor = (-0.5 * xi_quarter * delta_t).exp();

        let kinetic_temperature_new = kinetic_temperature * (rescaling_factor).powi(2);
        self.eta += 0.5 * xi_quarter * delta_t;
        let g_new =
            (kinetic_temperature_new / *macrostate.temperature() - 1.0) / self.tau.get().powi(2);
        self.xi = xi_quarter + 0.25 * g_new * delta_t;

        // Cache the thermostat energy so that users do not have the opportunity
        // to provide incorrect temperature or degree of freedom values when
        // logging the thermostat's energy.
        self.energy = self.thermostat_energy(*macrostate.temperature(), degrees_of_freedom);
        rescaling_factor
    }

    #[inline]
    fn integrate_half_step_two<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64 {
        self.integrate_half_step_one(rng, macrostate, delta_t, kinetic_energy, degrees_of_freedom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::check;

    use crate::TranslationalKineticEnergy;
    use hoomd_microstate::{
        Body, Microstate,
        property::{DynamicPoint, Point},
    };
    use hoomd_simulation::macrostate::Isothermal;
    use hoomd_vector::Cartesian;

    #[test]
    fn test_zero() -> anyhow::Result<()> {
        let thermostat = MartynaTuckermanTobiasKlein::zero(0.5.try_into()?);

        check!(thermostat.tau.get() == 0.5);
        check!(thermostat.xi() == 0.0);
        check!(thermostat.eta() == 0.0);
        check!(thermostat.energy() == 0.0);

        Ok(())
    }

    #[test]
    fn test_thermalized() -> anyhow::Result<()> {
        let microstate = Microstate::builder()
            .bodies([
                Body {
                    properties: DynamicPoint {
                        position: Cartesian::from([1.0, 2.0]),
                        ..Default::default()
                    },
                    sites: vec![Point::default()],
                },
                Body {
                    properties: DynamicPoint {
                        position: Cartesian::from([-2.0, 3.0]),
                        ..Default::default()
                    },
                    sites: vec![Point::default()],
                },
            ])
            .try_build()?;

        let macrostate = Isothermal { temperature: 1.5 };
        let mut rng = microstate.counter().make_rng();
        let thermostat = MartynaTuckermanTobiasKlein::thermalized(
            &mut rng,
            0.5.try_into()?,
            &macrostate,
            microstate.translational_kinetic_energy().1,
        );

        check!(thermostat.tau.get() == 0.5);
        check!(thermostat.xi() != 0.0);
        check!(thermostat.eta() == 0.0);
        check!(thermostat.energy() != 0.0);

        Ok(())
    }
}

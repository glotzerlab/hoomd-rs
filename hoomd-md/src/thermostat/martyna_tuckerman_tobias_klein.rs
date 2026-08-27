// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `MartynaTuckermanTobiasKlein`

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::Thermostat;
use hoomd_simulation::macrostate::Temperature;
use hoomd_utility::valid::PositiveReal;

/// Stochastic momentum rescaling based on an extended ensemble.
/// 
///
/// `MartynaTuckermanTobiasKlein` implements the Nosé-Hoover thermostat
/// ([Nosé 1984], [Hoover 1985]) following [Martyna et al. 1994] and
/// [Tuckerman et al. 2006]. This  algorithm adds a new degree of freedom
/// $` \eta `$ whose dynamics are tuned to constrain the system's evolution such
/// that its other degrees of freedom sample a constant temperature ensemble.
/// A `MartynaTuckermanTobiasKlein` instance stores its extended "position"
/// $` \eta `$ and its corresponding extended "momentum" $` \xi `$.
/// 
/// [Nosé 1984]: https://doi.org/10.1063/1.447334
/// [Hoover 1985]: https://doi.org/10.1103/PhysRevA.31.1695
/// [Martyna et al. 1994]: https://doi.org/10.1063/1.467468
/// [Tuckerman et al. 2006]: https://doi.org/10.1088/0305-4470/39/19/S18
///
/// The dynamics of the new degree of freedom are tuned through the parameter
/// `tau` ($` \tau `$), which represents a coupling constant, somewhat analagous
/// to the spring constant in a system of a piston attached to a spring. Values
/// that are too high can cause abrupt fluctuations in the kinetic temperature,
/// while values that are too low can cause excessive equilibration time. The
/// recommended value for most systems is $` 1000 \Delta t `$.
/// 
/// The extended Hamiltonian $`H`$ is given by
/// 
/// ```math
/// H = K + U + N kT \eta + \frac{1}{2} N kT \tau^2\xi^2
/// ```
/// 
/// where $`N`$ is the number of degrees of freedom.
/// 
/// # Integrating the extra degree of freedom
/// 
/// The thermostat's extra degree of freedom is integrated in half steps, with
/// each half step following the same procedure. Consequently, in the equations
/// below, $`t`$ refers to the time at the start of the half step, *not* at the
/// start of the full step.
/// 
/// 1. Momentum $`\xi`$ is integrated forward a quarter step, and the rescaling
///    factor $`\alpha`$ is calculated from its new value. The equations are
/// 
///    ```math
///    \begin{align*}
///    
///    T_K(t) &= \frac{2 K(t)}{N} \\
///    
///    G(t) &= \frac{1}{\tau^2} \bigg( \frac{T_K(t)}{kT} - 1 \bigg) \\
///    
///    \xi \bigg( t + \frac{\Delta t}{4} \bigg) &= \xi(t) + G(t) \frac{\Delta t}{4} \\
///    
///    \alpha &= \exp\bigg[ - \xi \bigg( t + \frac{\Delta t}{4} \bigg) \frac{\Delta t}{2} \bigg]
///    
///    \end{align*}
///    ```
/// 
///    where $`T_K`$ is the instantaneous kinetic temperature and $`G`$
///    represents the thermodynamic driving force in the new degree of freedom.
/// 
/// 2. Position $`\eta`$ is integrated forward a half step using the new value
///    of $`\xi`$, and then $`\xi`$ is integrated forward another quarter step.
/// 
///    ```math
///    \begin{align*}
/// 
///    \eta \bigg( t + \frac{\Delta t}{2} \bigg) &= \eta(t) + \xi \bigg( t + \frac{\Delta t}{4} \bigg) \frac{\Delta t}{2} \\
/// 
///    T_K(t)' &= T_K(t) \alpha^2 \\
/// 
///    G(t)' &= \frac{1}{\tau^2} \bigg( \frac{T_K(t)'}{kT} - 1 \bigg) \\
///    
///    \xi \bigg( t + \frac{\Delta t}{2} \bigg) &= \xi \bigg( t + \frac{\Delta t}{4} \bigg) + G(t)' \frac{\Delta t}{4} \\
///    
///    \end{align*}
///    ```
/// 
/// # Warning
/// 
/// When there are strong harmonic forces in your interaction model,
/// `MartynaTuckermanTobiasKlein` will fail to sample the correct distribution;
/// use [`Bussi`] or [`NoséHooverChain`] instead.
///
/// [`Bussi`]: crate::thermostat::Bussi
/// [`NoséHooverChain`]: crate::thermostat::NoséHooverChain
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
    /// Construct a new `MartynaTuckermanTobiasKlein` thermostat with a given `tau` and a zeroed initial condition.
    /// 
    /// The resulting thermostat has `eta = 0` and `xi = 0`. This initial
    /// condition is likely to be very far from equilibrium, which will result
    /// in wild kinetic energy oscillations for the first hundred to thousand
    /// time steps. Use [`thermalized`] to choose the initial position and
    /// momentum from a thermal distribution.
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

    /// Construct a new thermostat with a given `tau` and an initial condition drawn from the thermal distribution.
    /// 
    /// The resulting thermostat has `eta = 0` and `xi` that is randomly chosen
    /// from the thermal distribution encoded in a macrostat's temperature
    /// set point.
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

    /// Calculate the thermostat's energy.
    /// 
    /// See above for the hamiltonian.
    #[inline]
    fn thermostat_energy(&self, temperature_set_point: f64, degrees_of_freedom: usize) -> f64 {
        (degrees_of_freedom as f64)
            * temperature_set_point
            * (self.eta + 0.5 * (self.xi * self.tau.get()).powi(2))
    }

    /// The energy contribution from the extra degree of freedom.
    /// 
    /// ```math
    /// N kT \eta + \frac{1}{2} N kT \tau^2\xi^2
    /// ```
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

    /// The extended position.
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

    /// The extended momentum.
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
    /// Integrate `xi` and `eta` forward a half step and return the rescaling factor.
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

    /// Integrate `xi` and `eta` forward a half step and return the rescaling factor.
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

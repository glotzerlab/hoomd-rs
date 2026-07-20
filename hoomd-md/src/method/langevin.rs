// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Langevin`.

use hoomd_microstate::Site;
use crate::{method::{Gamma, GammaR}, thermostat::NoThermostat};

/// Integrate bodies' degrees of freedom in the microstate according to
/// Langevin equations of motion, modelling the NVE or NVT ensemble.
/// 
/// The `Langevin` implementation follows the same symplectic integration scheme
/// used in [`ConstantVolume`](crate::ConstantVolume), but with drag and random
/// forces and torques applied to each body *i*:
/// 
/// ```math
/// \begin{align*}
/// \vec{F}_i &= \vec{F}_C - \gamma \cdot \vec{v}_i + \vec{F}_R \\
/// \vec{\tau}_i &= \vec{\tau}_C - \vec{\gamma}_R \cdot \vec{\omega}_i + \vec{\tau}_R \\
/// \end{align*}
/// ```
/// 
/// where $` \vec{F}_C `$ and $` \vec{\tau}_C `$ are the force and torque on the
/// body from all other bodies and external interactions, $` \gamma `$ and
/// $` \vec{\gamma}_R `$ are the translational and rotational drag coefficients, and
/// $` \vec{F}_R `$ and $` \vec{\tau}_R `$ are random forces and torques. These
/// random forces and torques are uniform
/// 
/// ```math
/// \begin{align*}
/// \left< \vec{F}_R \right> &= 0 \\
/// \left< \vec{\tau}_R \right> &= 0 \\
/// \end{align*}
/// ```
/// 
/// and their magnitudes are chosen via the [fluctuation-dissipation theorem]
/// to be consistent with the specified drag and temperature
/// 
/// ```math
/// \begin{align*}
/// \left< \left| \vec{F}_R \right|^2 \right> &= \frac{2 d k T \gamma}{\Delta t} \\
/// \left< \left| \vec{\tau}_R \right|^2 \right> &= \frac{2 d_R k T \gamma_R}{\Delta t} \\
/// \end{align*}
/// ```
/// 
/// where $` d `$ and $` d_R `$ are the number of translational and rotational
/// degrees of freedom. Note that $` d_R `$ is determined by the number of
/// non-zero components of the body's moment of inertia.
/// 
/// [fluctuation-dissipation theorem]: https://en.wikipedia.org/wiki/Fluctuation%E2%80%93dissipation_theorem
/// 
/// To create a `Langevin`, use [`Langevin::builder`].
/// 
/// TODO: example
pub struct Langevin<const N: usize, B, G, GR, TT, TR = TT>
where
    B: NetForce + NetTorque + Momentum + AngularMomentum,
    G: Gamma<B>,
    GR: GammaR<B>,
{
    /// The time step size.
    pub delta_t: f64,

    /// Translational drag coefficient.
    pub gamma: G,

    /// Rotational drag coefficients.
    pub gamma_r: GR,

    /// Translational thermostat.
    pub translational_thermostat: TT,

    /// Rotational thermostat.
    pub rotational_thermostat: TR,

    /// Mark the type of the body properties from which to determine gamma and gamma_r.
    marker: PhantomData<B>,
}

impl<const N: usize, B, G, GR, TT, TR> Langevin<N, B, G, GR, TT, TR>
where
    G: Gamma<BodyProperties = B>,
    GR: GammaR<N, BodyProperties = B>,
{
    /// Access the translational thermostat.
    #[inline]
    pub fn translational_thermostat(&self) -> &TT {
        &self.translational_thermostat
    }

    /// Access the translational thermostat (mutable).
    #[inline]
    pub fn translational_thermostat_mut(&mut self) -> &mut TT {
        &mut self.translational_thermostat
    }

    /// Access the rotational thermostat.
    #[inline]
    pub fn rotational_thermostat(&self) -> &TR {
        &self.rotational_thermostat
    }

    /// Access the rotational thermostat (mutable).
    #[inline]
    pub fn rotational_thermostat_mut(&mut self) -> &mut TR {
        &mut self.rotational_thermostat
    }

    /// Access the translational drag coefficient.
    #[inline]
    pub fn gamma(&self) -> &G {
        &self.gamma
    }

    /// Access the translational drag coefficient (mutable).
    #[inline]
    pub fn gamma_mut(&mut self) -> &mut G {
        &mut self.gamma
    }

    /// Access the rotational drag coefficients.
    #[inline]
    pub fn gamma_r(&self) -> &GR {
        &self.gamma_r
    }

    /// Access the rotational drag coefficients (mutable).
    #[inline]
    pub fn gamma_r_mut(&mut self) -> &mut GR {
        &mut self.gamma_r
    }
}

/// Builder that constructs [`Langevin`].
///
/// Call [`Langevin::builder`] to start building a new [`Langevin`].
pub struct LangevinBuilder<const N: usize, B, G, GR, TT, TR = TT>
    /// The time step size.
    delta_t: f64,

    /// Translational drag coefficient.
    gamma: G,

    /// Rotational drag coefficients.
    gamma_r: GR,

    /// Translational thermostat.
    translational_thermostat: TT,

    /// Rotational thermostat.
    rotational_thermostat: TR,

    /// Mark the type of the body properties from which to determine gamma and gamma_r.
    marker: PhantomData<B>,

}


impl<const N: usize, B, G, GR, TT, TR,> LangevinBuilder<N, B, G, GR, TT, TR,>
where
    G: Gamma<BodyProperties = B>,
    GR: GammaR<N, BodyProperties = B>,
{
    /// Set the thermostat that applies to the translational degrees of freedom.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::{method::Langevin, thermostat::Bussi};
    ///
    /// let delta_t = 0.001;
    /// let constant_volume = Langevin::builder(delta_t)
    ///     .translational_thermostat(Bussi::default())
    ///     .build();
    /// ```
    #[inline]
    pub fn translational_thermostat<T>(
        self,
        translational_thermostat: T,
    ) -> LangevinBuilder<N, B, G, GR, T, TR> {
        LangevinBuilder {
            delta_t: self.delta_t,
            gamma: self.gamma,
            gamma_r: self.gamma_r,
            translational_thermostat,
            rotational_thermostat: self.rotational_thermostat,
            marker: PhantomData,
        }
    }

    /// Set the thermostat that applies to the rotational degrees of freedom.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::{method::Langevin, thermostat::Bussi};
    ///
    /// let delta_t = 0.001;
    /// let langevin = Langevin::builder(delta_t)
    ///     .rotational_thermostat(Bussi::default())
    ///     .build();
    /// ```
    #[inline]
    pub fn rotational_thermostat<T>(
        self,
        rotational_thermostat: T,
    ) -> LangevinBuilder<N, B, G, GR, TT, T> {
        LangevinBuilder {
            delta_t: self.delta_t,
            gamma: self.gamma,
            gamma_r: self.gamma_r,
            translational_thermostat: self.translational_thermostat,
            rotational_thermostat,
            marker: PhantomData,
        }
    }

    /// Set the thermostat that applies to both translational and rotational degrees of freedom.
    ///
    /// The given thermostat is cloned. The translational and rotational thermostats evolve
    /// independently.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::{method::Langevin, thermostat::Bussi};
    ///
    /// let delta_t = 0.001;
    /// let langevin = Langevin::builder(delta_t)
    ///     .thermostat(Bussi::default())
    ///     .build();
    /// ```
    #[inline]
    pub fn thermostat<T: Clone>(
        self,
        thermostat: T
    ) -> LangevinBuilder<N, B, G, GR, T, T> {
        LangevinBuilder {
            delta_t: self.delta_t,
            gamma: self.gamma,
            gamma_r: self.gamma_r,
            translational_thermostat: thermostat.clone(),
            rotational_thermostat: thermostat,
            marker: PhantomData,
        }
    }

    /// Set the drag coefficient that applies to translational degrees of freedom.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::{method::Langevin, thermostat::Bussi};
    ///
    /// let delta_t = 0.001;
    /// let langevin = Langevin::builder(delta_t)
    ///     .gamma(2.0)
    ///     .build();
    /// ```
    #[inline]
    pub fn gamma<T: Gamma<B>>(
        self,
        gamma: T
    ) -> LangevinBuilder<N, B, T, GR, TT, TR> {
        LangevinBuilder {
            delta_t: self.delta_t,
            gamma,
            gamma_r: self.gamma_r,
            translational_thermostat: self.translational_thermostat,
            rotational_thermostat: self.rotational_thermostat,
            marker: PhantomData,
        }
    }

    /// Set the drag coefficients that apply to the rotational degrees of freedom.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::{method::Langevin, thermostat::Bussi};
    ///
    /// let delta_t = 0.001;
    /// let langevin = Langevin::builder(delta_t)
    ///     .gamma_r([2.0, 2.0, 2.0])
    ///     .build();
    /// ```
    #[inline]
    pub fn gamma_r<T: GammaR<B>>(
        self,
        gamma_r: T
    ) -> LangevinBuilder<N, B, G, T, TT, TR> {
        LangevinBuilder {
            delta_t: self.delta_t,
            gamma: self.gamma,
            gamma_r,
            translational_thermostat: self.translational_thermostat,
            rotational_thermostat: self.rotational_thermostat,
            marker: PhantomData,
        }
    }

    /// Complete building a new [`Langevin`].
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::method::Langevin;
    ///
    /// let delta_t = 0.001;
    /// let langevin = Langevin::builder(delta_t).build();
    /// ```
    #[inline]
    pub fn build(self) -> Langevin<N, B, G, GR, TT, TR> {
        Langevin {
            delta_t: self.delta_t,
            gamma: self.gamma,
            gamma_r: self.gamma_r,
            translational_thermostat: self.translational_thermostat,
            rotational_thermostat: self.rotational_thermostat,
            marker: PhantomData,
        }
    }
}

impl<const N: usize, B> Langevin<N, B, NoThermostat, NoThermostat, f64, [f64; N]> {
    #[inline]
    /// Start building a new `Langevin`.
    ///
    /// The default builder uses the given value for `delta_t` and [`NoThermostat`]
    /// for both the translational and rotational thermostats. Call zero or more
    /// of the [`LangevinBuilder`] methods to set the thermostats.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::method::Langevin;
    ///
    /// let delta_t = 0.001;
    /// let constant_volume = Langevin::builder(delta_t).build();
    /// ```
    /// [`NoThermostat`]: crate::thermostat::NoThermostat
    pub fn builder(
        delta_t: f64,
    ) -> LangevinBuilder<N, B, NoThermostat, NoThermostat, f64, [f64; N]> {
        LangevinBuilder::<N, B, NoThermostat, NoThermostat, f64, [f64; N]> {
            delta_t,
            translational_thermostat: NoThermostat,
            rotational_thermostat: NoThermostat,
            gamma: 1.0,
            gamma_r: [1.0_f64; N]
        }
    }
}

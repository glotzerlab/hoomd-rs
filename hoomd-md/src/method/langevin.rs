// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Langevin`.
use std::{marker::PhantomData};

use hoomd_simulation::macrostate::Temperature;
use rand::{Rng, distr::Distribution};

use hoomd_microstate::{Body, Microstate, SiteKey, Tagged, Transform, boundary::{GenerateGhosts, Wrap}, property::{AngularMomentum, DynamicOrientedPoint, Mass, MomentOfInertia, Momentum, NetForce, NetTorque, NetVirial, Position}};
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Angle, Cartesian, Outer, Versor};
use rand_distr::Uniform;
use crate::{RotationalMotion, Thermostat, TranslationalKineticEnergy, TranslationalMotion, UpdateNetForceAndVirial, UpdateNetForceVirialAndTorque, method::{Gamma, GammaR}, thermostat::NoThermostat};

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
    B: NetForce + NetTorque + Momentum + AngularMomentum,
    G: Gamma<B>,
    GR: GammaR<B, GammaR = <B as AngularMomentum>::AngularMomentum>,
    TT: Copy,
    TR: Copy,
{
    /// Access the time step size.
    #[inline]
    pub fn delta_t(&self) -> &f64 {
        &self.delta_t
    }

    /// Access the time step size (mutable).
    #[inline]
    pub fn delta_t_mut(&mut self) -> &mut f64 {
        &mut self.delta_t
    }

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

/// Dampen and add noise to forces and virials, on bodies in the microstate.
trait DragAndRandomTranslation<B, S, X, C, M, R: Rng + ?Sized> {
    fn apply_drag_and_random_forces_and_virials<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        rng: &mut R,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    );
}

/// Dampen and add noise to torques on bodies in the microstate.
trait DragAndRandomRotation<B, S, X, C, M, R: Rng + ?Sized> {
    fn apply_drag_and_random_torques<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        rng: &mut R,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    );
}

/// Langevin forces and virials in N-dimensional cartesian space.
impl<const N: usize, B, S, X, C, G, GR, TT, TR, M, R> DragAndRandomTranslation<B, S, X, C, M, R>
    for Langevin<N, B, G, GR, TT, TR>
where
    B: Position<Position = Cartesian<N>>
        + Momentum<Momentum = Cartesian<N>>
        + NetForce<NetForce = Cartesian<N>>
        + NetVirial<NetVirial = <Cartesian<N> as Outer>::Tensor>
        + NetTorque
        + AngularMomentum
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<N>> + Default,
    X: PointUpdate<Cartesian<N>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    G: Gamma<B>,
    GR: GammaR<B, GammaR = <B as AngularMomentum>::AngularMomentum>,
    TT: Thermostat<M>,
    M: Temperature,
    R: Rng + ?Sized,
{
    /// Apply drag and random forces and virials to selected bodies in the microstate.
    /// 
    /// Drag forces are parameterized by [`Langevin.gamma`] and oppose the
    /// direction of motion. Random forces are uniform and have magnitudes that
    /// are consistent with the drag and system temperature in accordance with
    /// the fluctuation-dissipation theorem. Drag and random virials are
    /// calculated directly from these forces. For details, see
    /// [above](crate::method::langevin).
    /// 
    /// TODO: check whether virials should be updated
    /// TODO: communicate that this is defined only for CARTESIAN and is not
    /// generic across VECTOR
    #[inline]
    fn apply_drag_and_random_forces_and_virials<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        rng: &mut R,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue;
            }
            let mut body_properties = body.item.properties.clone();

            // Calculate the drag force
            let g = self.gamma.value(&body_properties);
            let v = *body_properties.momentum() / body_properties.mass();
            let f_drag = v * g * -1.0;
            
            // Pick a random force
            let magnitude = (6.0 * macrostate.temperature() * g / self.delta_t).sqrt();
            let uniform = Uniform::new_inclusive(-1.0, 1.0).unwrap();
            let f_rand = Cartesian::<N>::from(
                core::array::from_fn(|_| magnitude * uniform.sample(rng))
            );

            // Apply drag and random forces and update virial accordingly
            *body_properties.net_force_mut() += f_drag + f_rand;
            let position = *body_properties.position();
            *body_properties.net_virial_mut() += (f_drag + f_rand).outer(&position);
        }
    }
}

/// Langevin torques in 3-dimensional cartesian space.
impl<B, S, X, C, G, GR, TT, TR, M, R> DragAndRandomRotation<B, S, X, C, M, R>
    for Langevin<3, B, G, GR, TT, TR>
where
    B: Transform<S>
        + Clone
        + Momentum
        + NetForce
        + AngularMomentum<AngularMomentum = Cartesian<3>>
        + MomentOfInertia<MomentOfInertia = [f64; 3]>
        + NetTorque<NetTorque = Cartesian<3>>,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<B>
        + Wrap<S>
        + GenerateGhosts<S>,
    G: Gamma<B>,
    GR: GammaR<B, GammaR = [f64; 3]>,
    TR: Thermostat<M>,
    M: Temperature,
    R: Rng + ?Sized,
{
    /// Apply drag and random torques to selected bodies in the microstate.
    /// 
    /// Drag torques are parameterized by [`Langevin.gamma_r`]. Random torques
    /// are uniform and have magnitudes that are consistent with the drag and
    /// system temperature in accordance with the fluctuation-dissipation
    /// theorem. For details, see [above](crate::method::langevin).
    #[inline]
    fn apply_drag_and_random_torques<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        rng: &mut R,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {       
        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue;
            }
            let mut body_properties = body.item.properties.clone();

            // Calculate the drag torque
            let g_r = self.gamma_r.value(&body_properties);
            let moi = body_properties.moment_of_inertia();
            let angular_momentum = body_properties.angular_momentum();
            let w = Cartesian::<3>::from(
                core::array::from_fn(|i| angular_momentum[i] / moi[i])
            );
            let t_drag = Cartesian::<3>::from(
                core::array::from_fn(|i| w[i] * g_r[i] * -1.0)
            );

            // Pick a random torque
            let uniform = Uniform::new_inclusive(-1.0, 1.0).unwrap();
            let t_rand = Cartesian::<3>::from(
                core::array::from_fn(|i| {
                    let magnitude = 
                        if moi[i] > 0.0 {
                            (6.0 * macrostate.temperature() * g_r[i] / self.delta_t).sqrt()
                        } else {
                            0.0
                        };
                    magnitude * uniform.sample(rng)
                })
            );

            // Apply drag and random torques
            *body_properties.net_torque_mut() += t_drag + t_rand;
        }
    }
}

/// Langevin torques in 2-dimensional cartesian space.
/// 
/// TODO: discuss how we link the return type of GammaR with the system's vector-space.
impl<B, S, X, C, G, GR, TT, TR, M, R> DragAndRandomRotation<B, S, X, C, M, R>
    for Langevin<2, B, G, GR, TT, TR>
where
    B: Transform<S>
        + Clone
        + Momentum
        + NetForce
        + AngularMomentum<AngularMomentum = f64>
        + MomentOfInertia<MomentOfInertia = f64>
        + NetTorque<NetTorque = f64>,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<B>
        + Wrap<S>
        + GenerateGhosts<S>,
    G: Gamma<B>,
    GR: GammaR<B, GammaR = f64>,
    TR: Thermostat<M>,
    M: Temperature,
    R: Rng + ?Sized,
{
    /// Apply drag and random torques to selected bodies in the microstate.
    /// 
    /// Drag torques are parameterized by [`Langevin.gamma_r`]. Random torques
    /// are uniform and have magnitudes that are consistent with the drag and
    /// system temperature in accordance with the fluctuation-dissipation
    /// theorem. For details, see [above](crate::method::langevin).
    #[inline]
    fn apply_drag_and_random_torques<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        rng: &mut R,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {       
        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue;
            }
            let mut body_properties = body.item.properties.clone();

            // Calculate the drag torque
            let g_r = self.gamma_r.value(&body_properties);
            let moi = body_properties.moment_of_inertia();
            let angular_momentum = body_properties.angular_momentum();
            let w = angular_momentum / moi;
            let t_drag = w * g_r * -1.0;

            // Pick a random torque
            let uniform = Uniform::new_inclusive(-1.0, 1.0).unwrap();
            let magnitude = 
                if *moi > 0.0 {
                    (6.0 * macrostate.temperature() * g_r / self.delta_t).sqrt()
                } else {
                    0.0
                };
            let t_rand = magnitude * uniform.sample(rng);

            // Apply drag and random torques
            *body_properties.net_torque_mut() += t_drag + t_rand;
        }
    }
}

/// Builder that constructs [`Langevin`].
///
/// Call [`Langevin::builder`] to start building a new [`Langevin`].
pub struct LangevinBuilder<const N: usize, B, G, GR, TT, TR = TT>
where
    B: NetForce + NetTorque + Momentum + AngularMomentum,
    G: Gamma<B>,
    GR: GammaR<B>,
{
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
    B: NetForce + NetTorque + Momentum + AngularMomentum,
    G: Gamma<B>,
    GR: GammaR<B>,
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

impl<B> Langevin<2, B, f64, f64, NoThermostat, NoThermostat>
where
    B: NetForce + NetTorque + Momentum + AngularMomentum<AngularMomentum = f64>,
{
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
    ) -> LangevinBuilder<2, B, f64, f64, NoThermostat, NoThermostat> {
        LangevinBuilder {
            delta_t,
            gamma: 1.0,
            gamma_r: 1.0,
            translational_thermostat: NoThermostat,
            rotational_thermostat: NoThermostat,
            marker: PhantomData,
        }
    }
}

impl<B> Langevin<3, B, f64, [f64; 3], NoThermostat, NoThermostat>
where
    B: NetForce
        + NetTorque
        + Momentum
        + AngularMomentum<AngularMomentum = Cartesian<3>>,
{
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
    ) -> LangevinBuilder<3, B, f64, [f64; 3], NoThermostat, NoThermostat> {
        LangevinBuilder {
            delta_t,
            gamma: 1.0,
            gamma_r: [1.0; 3],
            translational_thermostat: NoThermostat,
            rotational_thermostat: NoThermostat,
            marker: PhantomData,
        }
    }
}

impl<const N: usize, B, S, X, C, G, GR, TT, TR, M> TranslationalMotion<B, S, X, C, M> for Langevin<N, B, G, GR, TT, TR>
where
    B: Position<Position = Cartesian<N>>
        + Momentum<Momentum = Cartesian<N>>
        + NetForce<NetForce = Cartesian<N>>
        + NetVirial<NetVirial = <Cartesian<N> as Outer>::Tensor>
        + NetTorque
        + AngularMomentum
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<N>> + Default,
    X: PointUpdate<Cartesian<N>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    M: Temperature,
    TT: Thermostat<M>,
    G: Gamma<B>,
    GR: GammaR<B, GammaR = <B as AngularMomentum>::AngularMomentum>,
    {
    /// Integrate selected body positions forward a full step and their momenta forward a half step.
    ///
    /// This method is identical to [`ConstantVolume::integrate_translation_half_step_one_with_filter`].
    fn integrate_translation_half_step_one_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let (kinetic_energy, degrees_of_freedom) =
            microstate.translational_kinetic_energy_with_filter(&should_integrate_body);

        crate::method::constant_volume::integrate_translation_half_step_one_with_filter(
            self.delta_t,
            microstate,
            &mut self.translational_thermostat,
            kinetic_energy,
            degrees_of_freedom,
            macrostate,
            should_integrate_body
        );
    }

    /// Apply drag and random forces and virials to bodies, then integrate
    /// selected body momenta forward a half step.
    ///
    /// Aside from the application of drag and random forces and virials, this
    /// method is identical to
    /// [`ConstantVolume::integrate_translation_half_step_two_with_filter`].
    fn integrate_translation_half_step_two_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        self.apply_drag_and_random_forces_and_virials(
            &mut rng,
            microstate,
            macrostate,
            &should_integrate_body,
        );

        crate::method::constant_volume::integrate_translation_half_step_two_with_filter(
            self.delta_t,
            microstate,
            &mut self.translational_thermostat,
            macrostate,
            should_integrate_body
        );
    }
}

/// Rotational motion in 3-dimensional cartesian space.
impl<S, X, C, M, G, GR, TT, TR> RotationalMotion<DynamicOrientedPoint<Cartesian<3>, Versor>, S, X, C, M>
    for Langevin<3, DynamicOrientedPoint<Cartesian<3>, Versor>, G, GR, TT, TR>
where
    DynamicOrientedPoint<Cartesian<3>, Versor>: Transform<S>,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<DynamicOrientedPoint<Cartesian<3>, Versor>>
        + Wrap<S>
        + GenerateGhosts<S>,
    TR: Thermostat<M>,
    G: Gamma<DynamicOrientedPoint<Cartesian<3>, Versor>>,
    GR: GammaR<DynamicOrientedPoint<Cartesian<3>, Versor>, GammaR = [f64; 3]>,
    M: Temperature,
{
    /// Integrate selected body orientations forward a full step and their angular momenta forward a half step.
    ///
    /// This method is identical to [`ConstantVolume::integrate_rotation_half_step_one_with_filter`].
    fn integrate_rotation_half_step_one_with_filter<
        F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<3>, Versor>, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<DynamicOrientedPoint<Cartesian<3>, Versor>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        crate::method::constant_volume::integrate_rotation_half_step_one_with_filter_3d(
            self.delta_t,
            microstate,
            &mut self.rotational_thermostat,
            macrostate,
            should_integrate_body
        );
    }

    /// Apply drag and random torques to bodies, then integrate selected body
    /// angular momenta forward a half step.
    ///
    /// Aside from the application of drag and random torques, this method is
    /// identical to [`ConstantVolume::integrate_rotation_half_step_two_with_filter`].
    fn integrate_rotation_half_step_two_with_filter<
        F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<3>, Versor>, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<DynamicOrientedPoint<Cartesian<3>, Versor>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        self.apply_drag_and_random_torques(
            &mut rng,
            microstate,
            macrostate,
            &should_integrate_body,
        );
        crate::method::constant_volume::integrate_rotation_half_step_two_with_filter_3d(
            self.delta_t,
            microstate,
            &mut self.rotational_thermostat,
            macrostate,
            should_integrate_body
        );
    }
}

/// Rotational motion in 2-dimensional cartesian space.
impl<S, X, C, G, GR, TT, TR, M> RotationalMotion<DynamicOrientedPoint<Cartesian<2>, Angle>, S, X, C, M>
    for Langevin<2, DynamicOrientedPoint<Cartesian<2>, Angle>, G, GR, TT, TR>
where
    DynamicOrientedPoint<Cartesian<2>, Angle>: Transform<S>,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<DynamicOrientedPoint<Cartesian<2>, Angle>>
        + Wrap<S>
        + GenerateGhosts<S>,
    G: Gamma<DynamicOrientedPoint<Cartesian<2>, Angle>>,
    GR: GammaR<DynamicOrientedPoint<Cartesian<2>, Angle>, GammaR = f64>,
    TR: Thermostat<M>,
    M: Temperature,
{
    /// Integrate selected body orientations forward a full step and their angular momenta forward a half step.
    ///
    /// This method is identical to [`ConstantVolume::integrate_rotation_half_step_one_with_filter`].
    fn integrate_rotation_half_step_one_with_filter<
        F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<2>, Angle>, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<DynamicOrientedPoint<Cartesian<2>, Angle>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        crate::method::constant_volume::integrate_rotation_half_step_one_with_filter_2d(
            self.delta_t,
            microstate,
            &mut self.rotational_thermostat,
            macrostate,
            should_integrate_body
        );
    }

    /// Apply drag and random torques to bodies, then integrate selected body
    /// angular momenta forward a half step.
    ///
    /// Aside from the application of drag and random torques, this method is
    /// identical to [`ConstantVolume::integrate_rotation_half_step_two_with_filter`].
    fn integrate_rotation_half_step_two_with_filter<
        F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<2>, Angle>, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<DynamicOrientedPoint<Cartesian<2>, Angle>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        self.apply_drag_and_random_torques(
            &mut rng,
            microstate,
            macrostate,
            &should_integrate_body,
        );

        crate::method::constant_volume::integrate_rotation_half_step_two_with_filter_2d(
            self.delta_t,
            microstate,
            &mut self.rotational_thermostat,
            macrostate,
            should_integrate_body
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_langevin() {
        type B2 = DynamicOrientedPoint<Cartesian<2>, Angle>;
        type B3 = DynamicOrientedPoint<Cartesian<3>, Versor>;

        let dt = 2.0;
        
        let lan = Langevin::<2, B2, _, _, _>::builder(dt).build();
        assert_eq!(lan.delta_t, dt);

        let lan = Langevin::<3, B3, _, _, _>::builder(dt).build();
        assert_eq!(lan.delta_t, dt);
    }

    fn test_langevin_integration_2d() {}
}
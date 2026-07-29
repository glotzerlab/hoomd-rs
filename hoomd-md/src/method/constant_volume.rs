// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ConstantVolume`.

use serde::{Deserialize, Serialize};
use std::{array, ops::MulAssign};

use crate::{
    RotationalKineticEnergy, RotationalMotion, Thermostat, TranslationalKineticEnergy, TranslationalMotion, method::IntegrateRotation, thermostat::NoThermostat,
};
use hoomd_microstate::{
    Body, Microstate, SiteKey, Tagged, Transform, boundary::{GenerateGhosts, Wrap}, property::{
        AngularMomentum, DynamicOrientedPoint, Mass, MomentOfInertia, Momentum, NetForce, NetTorque, Orientation, Position, RotationalMotionTypes,
    },
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Angle, Cartesian, InnerProduct, Quaternion, Rotate, Rotation, Versor, Wedge};

/// Integrate bodies' degrees of freedom in the microstate, modelling the NVE or NVT ensemble.
///
/// The `ConstantVolume` implementation follows the symplectic integration
/// scheme by [Tuckerman et al. 2006] for translational motion and [Miller et
/// al. 2002] for rotational motion.
///
/// Use [`NoThermostat`] to integrate trajectories that sample the microcanonical ensemble:
/// ```
/// use hoomd_md::method::ConstantVolume;
///
/// let delta_t = 0.001;
/// let constant_volume = ConstantVolume::builder(delta_t).build();
/// ```
///
/// Use [`Bussi`] (or one of the other thermostats) to integrate trajectories that sample
/// the canonical ensemble:
/// ```
/// use hoomd_md::{method::ConstantVolume, thermostat::Bussi};
///
/// let delta_t = 0.001;
/// let constant_volume = ConstantVolume::builder(delta_t)
///     .thermostat(Bussi::default())
///     .build();
/// ```
///
/// # Reference
///
/// * [Tuckerman et al. 2006]
/// * [Miller et al. 2002]
///
/// [`NoThermostat`]: crate::thermostat::NoThermostat
/// [`Bussi`]: crate::thermostat::Bussi
/// [Tuckerman et al. 2006]: https://doi.org/10.1088/0305-4470/39/19/S18
/// [Miller et al. 2002]: https://doi.org/10.1063/1.1473654
#[doc(alias = "nvt")]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantVolume<TT, TR = TT> {
    /// The time step size.
    pub delta_t: f64,

    /// Translational thermostat.
    pub translational_thermostat: TT,

    /// Rotational thermostat.
    pub rotational_thermostat: TR,
}

/// Builder that constructs [`ConstantVolume`].
///
/// Call [`ConstantVolume::builder`] to start building a new [`ConstantVolume`].
pub struct ConstantVolumeBuilder<TT, TR> {
    /// The time step size.
    delta_t: f64,

    /// Translational thermostat.
    translational_thermostat: TT,

    /// Rotational thermostat.
    rotational_thermostat: TR,
}

impl<TT, TR> ConstantVolumeBuilder<TT, TR> {
    /// Set the thermostat that applies to the translational degrees of freedom.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::{method::ConstantVolume, thermostat::Bussi};
    ///
    /// let delta_t = 0.001;
    /// let constant_volume = ConstantVolume::builder(delta_t)
    ///     .translational_thermostat(Bussi::default())
    ///     .build();
    /// ```
    #[inline]
    pub fn translational_thermostat<T>(
        self,
        translational_thermostat: T,
    ) -> ConstantVolumeBuilder<T, TR> {
        ConstantVolumeBuilder {
            delta_t: self.delta_t,
            translational_thermostat,
            rotational_thermostat: self.rotational_thermostat,
        }
    }

    /// Set the thermostat that applies to the rotational degrees of freedom.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::{method::ConstantVolume, thermostat::Bussi};
    ///
    /// let delta_t = 0.001;
    /// let constant_volume = ConstantVolume::builder(delta_t)
    ///     .rotational_thermostat(Bussi::default())
    ///     .build();
    /// ```
    #[inline]
    pub fn rotational_thermostat<T>(
        self,
        rotational_thermostat: T,
    ) -> ConstantVolumeBuilder<TT, T> {
        ConstantVolumeBuilder {
            delta_t: self.delta_t,
            translational_thermostat: self.translational_thermostat,
            rotational_thermostat,
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
    /// use hoomd_md::{method::ConstantVolume, thermostat::Bussi};
    ///
    /// let delta_t = 0.001;
    /// let constant_volume = ConstantVolume::builder(delta_t)
    ///     .thermostat(Bussi::default())
    ///     .build();
    /// ```
    #[inline]
    pub fn thermostat<T: Clone>(self, thermostat: T) -> ConstantVolumeBuilder<T, T> {
        ConstantVolumeBuilder {
            delta_t: self.delta_t,
            translational_thermostat: thermostat.clone(),
            rotational_thermostat: thermostat,
        }
    }

    /// Complete building a new [`ConstantVolume`].
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::method::ConstantVolume;
    ///
    /// let delta_t = 0.001;
    /// let constant_volume = ConstantVolume::builder(delta_t).build();
    /// ```
    #[inline]
    pub fn build(self) -> ConstantVolume<TT, TR> {
        ConstantVolume {
            delta_t: self.delta_t,
            translational_thermostat: self.translational_thermostat,
            rotational_thermostat: self.rotational_thermostat,
        }
    }
}

impl ConstantVolume<NoThermostat, NoThermostat> {
    #[inline]
    /// Start building a new `ConstantVolume`.
    ///
    /// The default builder uses the given value for `delta_t` and [`NoThermostat`]
    /// for both the translational and rotational thermostats. Call zero or more
    /// of the [`ConstantVolumeBuilder`] methods to set the thermostats.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_md::method::ConstantVolume;
    ///
    /// let delta_t = 0.001;
    /// let constant_volume = ConstantVolume::builder(delta_t).build();
    /// ```
    /// [`NoThermostat`]: crate::thermostat::NoThermostat
    pub fn builder(delta_t: f64) -> ConstantVolumeBuilder<NoThermostat, NoThermostat> {
        ConstantVolumeBuilder {
            delta_t,
            translational_thermostat: NoThermostat,
            rotational_thermostat: NoThermostat,
        }
    }
}

impl<TT, TR> ConstantVolume<TT, TR> {
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
}

/// First half step of the symplectic volume-preserving integration scheme for
/// translational degrees of freedom.
/// 
/// Selected bodies' positions are integrated forward a full step and momenta
/// are integrated forward a half step.
/// 
/// This function is defined outside of [`ConstantVolume`] because it
/// is also used by [`Langevin`](crate::method::Langevin).
/// 
/// The system's number of  degrees of freedom, which is used for integrating
/// the translational thermostat, is tabulated differently by different methods,
/// so it must be passed directly to this function (along with kinetic energy)
/// by the calling method. Note that this is *not* the case for the second
/// half step.
/// 
/// For details, see the documentation for
/// [`ConstantVolume::integrate_translation_half_step_one_with_filter`].
pub(crate) fn integrate_translation_half_step_one_with_filter<V, B, S, X, C, TT, M, F>(
    delta_t: f64,
    microstate: &mut Microstate<B, S, X, C>,
    translational_thermostat: &mut TT,
    kinetic_energy: f64,
    degrees_of_freedom: usize,
    macrostate: &M,
    should_integrate_body: F,
)
where
    V: Default + InnerProduct,
    B: Position<Position = V>
        + Momentum<Momentum = V>
        + NetForce<NetForce = V>
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = V> + Default,
    X: PointUpdate<V, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    TT: Thermostat<M>,
    F: Fn(&Tagged<Body<B, S>>) -> bool,
{
    let mut rng = microstate.counter().make_rng();

    let rescaling_factor = translational_thermostat.integrate_half_step_one(
        &mut rng,
        macrostate,
        delta_t,
        kinetic_energy,
        degrees_of_freedom,
    );

    for body_index in 0..microstate.bodies().len() {
        let body = &microstate.bodies()[body_index];
        if !should_integrate_body(body) {
            continue;
        }
        let mut body_properties = body.item.properties.clone();

        let net_force = *body_properties.net_force();
        let mass = body_properties.mass();
        let mut momentum = *body_properties.momentum();

        momentum *= rescaling_factor;
        momentum += net_force * 0.5 * delta_t;
        *body_properties.position_mut() += momentum / mass * delta_t;
        *body_properties.momentum_mut() = momentum;

        microstate
            .update_body_properties(body_index, body_properties)
            .expect(
                "Bodies and sites should remain in simulation boundary.\n
            Add interactions that prevent sites from moving outside the boundary.",
            );
    }

    microstate.increment_substep();
}

/// Second half step of the symplectic volume-preserving integration scheme for
/// translational degrees of freedom.
/// 
/// Selected bodies' momenta are integrated forward a half step.
/// 
/// This function is defined outside of [`ConstantVolume`] because it
/// is also used by [`Langevin`](crate::method::Langevin).
/// 
/// For details, see the documentation for
/// [`ConstantVolume::integrate_translation_half_step_two_with_filter`].
pub(crate) fn integrate_translation_half_step_two_with_filter<V, B, S, X, C, TT, M, F>(
    delta_t: f64,
    microstate: &mut Microstate<B, S, X, C>,
    translational_thermostat: &mut TT,
    macrostate: &M,
    should_integrate_body: F,
)
where
    V: Default + InnerProduct,
    B: Position<Position = V>
        + Momentum<Momentum = V>
        + NetForce<NetForce = V>
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = V> + Default,
    X: PointUpdate<V, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    TT: Thermostat<M>,
    F: Fn(&Tagged<Body<B, S>>) -> bool,
{
    let mut rng = microstate.counter().make_rng();

    for body_index in 0..microstate.bodies().len() {
        let body = &microstate.bodies()[body_index];
        if !should_integrate_body(body) {
            continue;
        }
        let mut body_properties = body.item.properties.clone();
        let net_force = *body_properties.net_force();

        *body_properties.momentum_mut() += net_force * delta_t * 0.5;

        microstate
            .update_body_properties(body_index, body_properties)
            .expect(
                "Bodies and sites should remain in simulation boundary.\n
            Add interactions that prevent sites from moving outside the boundary.",
            );
    }

    let (kinetic_energy, degrees_of_freedom) = microstate.translational_kinetic_energy();
    let rescaling_factor = translational_thermostat.integrate_half_step_two(
        &mut rng,
        macrostate,
        delta_t,
        kinetic_energy,
        degrees_of_freedom - microstate.conserved_degrees_of_freedom(),
    );

    if rescaling_factor != 1.0 {
        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue;
            }
            let mut body_properties = body.item.properties.clone();

            *body_properties.momentum_mut() *= rescaling_factor;

            microstate
                .update_body_properties(body_index, body_properties)
                .expect(
                    "Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.",
                );
        }
    }

    microstate.increment_substep();
}

impl<V, B, S, X, C, TT, TR, M> TranslationalMotion<B, S, X, C, M> for ConstantVolume<TT, TR>
where
    V: Default + InnerProduct,
    B: Position<Position = V>
        + Momentum<Momentum = V>
        + NetForce<NetForce = V>
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = V> + Default,
    X: PointUpdate<V, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    TT: Thermostat<M>,
{
    /// Integrate selected body positions forward a full step and their momenta forward a half step.
    ///
    /// The first half step of the symplectic integration procedure is given by the equations below, which are
    /// applied to each selected body *i*. In each step, the marker $`'`$ is used when a variable's value changes
    /// during a step to distinguish the value before ( $`'`$ is present) from the value after ( $`'`$ is absent).
    ///
    /// 1. The translational thermostat is integrated forward a half-step and then momentum is rescaled accordingly:
    ///
    ///    ```math
    ///    \vec{p}_i\left( t \right) = \vec{p'}_i\left( t \right) \cdot \mathrm{translational\_thermostat.integrate\_half\_step\_one}\left( \sum_{j \in \mathrm{selection}} K'_{trans,j} \left( t \right) \right)
    ///    ```
    ///    where the summation represents the total [translational kinetic energy](crate::compute::TranslationalKineticEnergy)
    ///    of the selected bodies at the start of the step, and `translational_thermostat.integrate_half_step_one()` is the
    ///    first half step method implemented by `TT`.
    ///
    /// 2. Momentum is integrated forward a half step.
    ///
    ///    ```math
    ///    \vec{p}_i\left( t + \frac{\Delta t}{2} \right) = \vec{p}_i\left( t \right) + \vec{F}_i(t) \frac{\Delta t}{2}
    ///    ```
    ///
    /// 3. Position is integrated forward a full step using the new momentum.
    ///
    ///    ```math
    ///    \vec{r}_i\left( t + \Delta t \right) = \vec{r}_i\left( t \right) + \frac{\vec{p}_i\left( t + \frac{\Delta t}{2} \right)}{m_i} \Delta t
    ///    ```
    #[inline]
    fn integrate_translation_half_step_one_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let (kinetic_energy, degrees_of_freedom) =
            microstate.translational_kinetic_energy_with_filter(&should_integrate_body);

        let conserved_degrees_of_freedom =
            if degrees_of_freedom == V::n_dimensions() * microstate.bodies().len() {
                V::n_dimensions()
            } else {
                0
            };
        *microstate.conserved_degrees_of_freedom_mut() = conserved_degrees_of_freedom;

        integrate_translation_half_step_one_with_filter(
            self.delta_t,
            microstate,
            &mut self.translational_thermostat,
            kinetic_energy,
            degrees_of_freedom - conserved_degrees_of_freedom,
            macrostate,
            should_integrate_body,
        );
    }

    /// Integrate selected body momenta forward a half step.
    ///
    /// The second half step of the symplectic integration procedure is given by the equations below, which are
    /// applied to each selected body *i*. In each step, the marker $`'`$ is used when a variable's value changes
    /// during a step to distinguish the value before ( $`'`$ is present) from the value after ( $`'`$ is absent).
    ///
    /// 1. Momentum is integrated forward a half step.
    ///
    ///    ```math
    ///    \vec{p}_i\left( t + \Delta t \right) = \vec{p}_i\left( t + \frac{\Delta t}{2} \right) + \vec{F}_i\left( t + \frac{\Delta t}{2} \right) \frac{\Delta t}{2}
    ///    ```
    ///
    /// 2. The translational thermostat is integrated forward a half step and then momentum is rescaled accordingly.
    ///
    ///    ```math
    ///    \vec{p}_i\left( t + \Delta t \right) = \vec{p'}_i\left( t + \Delta t \right) \cdot \mathrm{translational\_thermostat.integrate\_half\_step\_two}\left( \sum_{j \in \mathrm{selection}} K'_{trans,j} \left( t + \Delta t \right) \right)
    ///    ```
    ///
    ///    where the summation represents the total [translational kinetic energy](crate::compute::TranslationalKineticEnergy)
    ///    of the selected bodies at the start of the step, and `translational_thermostat.integrate_half_step_two()` is the
    ///    second half step method implemented by `TT`.
    #[inline]
    fn integrate_translation_half_step_two_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        integrate_translation_half_step_two_with_filter(
            self.delta_t,
            microstate,
            &mut self.translational_thermostat,
            macrostate,
            should_integrate_body,
        );
    }
}

/// First half step of the symplectic volume-preserving integration scheme for
/// rotational degrees of freedom in 3D Cartesian space.
/// 
/// Selected bodies' orientations are integrated forward a full step and
/// angular momenta are integrated forward a half step.
/// 
/// This function is defined outside of [`ConstantVolume`] because it
/// is also used by [`Langevin`](crate::method::Langevin).
/// 
/// Note: The actual algorithm is used in [`crate::method::IntegrateRotation`]
/// in order to provide separate implementations for 2D and 3D.
pub(crate) fn integrate_rotation_half_step_one_with_filter<V, R, B, S, X, C, TR, M, F> (
    delta_t: f64,
    microstate: &mut Microstate<B, S, X, C>,
    rotational_thermostat: &mut TR,
    macrostate: &M,
    should_integrate_body: F,
)
where
    V: Wedge + Copy,
    R: IntegrateRotation<Rotation = R> + RotationalMotionTypes + Clone,
    B: Copy
        + Transform<S>
        + Position<Position = V>
        + Orientation<Rotation = <R as IntegrateRotation>::Rotation>
        + AngularMomentum<AngularMomentum = <R as RotationalMotionTypes>::AngularMomentum>
        + MomentOfInertia<MomentOfInertia = <R as RotationalMotionTypes>::MomentOfInertia>
        + NetTorque<NetTorque = <R as IntegrateRotation>::NetTorque>,
    S: Position<Position = V> + Default,
    X: PointUpdate<V, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    TR: Thermostat<M>,
    F: Fn(&Tagged<Body<B, S>>) -> bool,
    Microstate<B, S, X, C>: RotationalKineticEnergy<B, S>,
    <R as IntegrateRotation>::Rotation: Clone,
    <R as RotationalMotionTypes>::AngularMomentum: MulAssign<f64> + Clone,
{
    let mut rng = microstate.counter().make_rng();
    let (kinetic_energy, degrees_of_freedom) =
        microstate.rotational_kinetic_energy_with_filter(&should_integrate_body);
    let rescaling_factor = rotational_thermostat.integrate_half_step_one(
        &mut rng,
        macrostate,
        delta_t,
        kinetic_energy,
        degrees_of_freedom,
    );

    for body_index in 0..microstate.bodies().len() {
        let body = &microstate.bodies()[body_index];
        if !should_integrate_body(body) {
            continue;
        }
        let mut body_properties = body.item.properties;
        
        *body_properties.angular_momentum_mut() *= rescaling_factor;

        let mut orientation = body_properties.orientation().clone();
        let mut angular_momentum = body_properties.angular_momentum().clone();
        <R as IntegrateRotation>::step1(
            delta_t,
            body_properties.net_torque(),
            &mut angular_momentum,
            &mut orientation,
            body_properties.moment_of_inertia(),
        );

        *body_properties.orientation_mut() = orientation;           // TODO: check clone ok
        *body_properties.angular_momentum_mut() = angular_momentum; // TODO: check clone ok

        microstate
            .update_body_properties(body_index, body_properties)
            .expect(
                "Bodies and sites should remain in simulation boundary.\n
            Add interactions that prevent sites from moving outside the boundary.",
            );
    }

    microstate.increment_substep();
}

/// Second half step of the symplectic volume-preserving integration scheme for
/// rotational degrees of freedom in 3D Cartesian space.
/// 
/// Selected bodies' angular momenta are integrated forward a half step.
/// 
/// This function is defined outside of [`ConstantVolume`] because it
/// is also used by [`Langevin`](crate::method::Langevin).
/// 
/// Note: The actual algorithm is used in [`crate::method::IntegrateRotation`]
/// in order to provide separate implementations for 2D and 3D.
pub(crate) fn integrate_rotation_half_step_two_with_filter<V, R, B, S, X, C, TR, M, F> (
    delta_t: f64,
    microstate: &mut Microstate<B, S, X, C>,
    rotational_thermostat: &mut TR,
    macrostate: &M,
    should_integrate_body: F,
)
where
    V: Wedge + Copy,
    R: IntegrateRotation<Rotation = R> + RotationalMotionTypes + Clone,
    B: Copy
        + Transform<S>
        + Position<Position = V>
        + Orientation<Rotation = <R as IntegrateRotation>::Rotation>
        + AngularMomentum<AngularMomentum = <R as RotationalMotionTypes>::AngularMomentum>
        + MomentOfInertia<MomentOfInertia = <R as RotationalMotionTypes>::MomentOfInertia>
        + NetTorque<NetTorque = <R as IntegrateRotation>::NetTorque>,
    S: Position<Position = V> + Default,
    X: PointUpdate<V, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    TR: Thermostat<M>,
    F: Fn(&Tagged<Body<B, S>>) -> bool,
    Microstate<B, S, X, C>: RotationalKineticEnergy<B, S>,
    <R as RotationalMotionTypes>::AngularMomentum: MulAssign<f64> + Clone,
{
    let mut rng = microstate.counter().make_rng();

    for body_index in 0..microstate.bodies().len() {
        let body = &microstate.bodies()[body_index];
        if !should_integrate_body(body) {
            continue;
        }

        let mut body_properties = body.item.properties;

        let mut angular_momentum = body_properties.angular_momentum().clone();
        <R as IntegrateRotation>::step2(
            delta_t,
            body_properties.net_torque(),
            &mut angular_momentum,
            body_properties.orientation(),
            body_properties.moment_of_inertia(),
        );

        // let q = *body_properties.orientation().get();
        // let s = *body_properties.angular_momentum();
        //
        // let mut p = q * Quaternion::pure(s) * 2.0;
        //
        // p += (q * Quaternion::pure(net_torque)) * delta_t;
        // 
        // *body_properties.angular_momentum_mut() = ((q.conjugate() * p) * 0.5).vector;

        *body_properties.angular_momentum_mut() = angular_momentum; // TODO: check clone ok

        microstate
            .update_body_properties(body_index, body_properties)
            .expect(
                "Bodies and sites should remain in simulation boundary.\n
            Add interactions that prevent sites from moving outside the boundary.",
            );
    }

    let (kinetic_energy, degrees_of_freedom) =
        microstate.rotational_kinetic_energy_with_filter(&should_integrate_body);
    let rescaling_factor = rotational_thermostat.integrate_half_step_two(
        &mut rng,
        macrostate,
        delta_t,
        kinetic_energy,
        degrees_of_freedom,
    );

    if rescaling_factor != 1.0 {
        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue;
            }
            let mut body_properties = body.item.properties;

            *body_properties.angular_momentum_mut() *= rescaling_factor;

            microstate
                .update_body_properties(body_index, body_properties)
                .expect(
                    "Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.",
                );
        }
    }

    microstate.increment_substep();
}

impl<V, R, B, S, X, C, TT, TR, M> RotationalMotion<R, B, S, X, C, M>
    for ConstantVolume<TT, TR>
where
    V: Wedge + Copy,
    R: IntegrateRotation<Rotation = R> + RotationalMotionTypes + Clone,
    B: Copy
        + Transform<S>
        + Position<Position = V>
        + Orientation<Rotation = R>
        + AngularMomentum<AngularMomentum = <R as RotationalMotionTypes>::AngularMomentum>
        + MomentOfInertia<MomentOfInertia = <R as RotationalMotionTypes>::MomentOfInertia>
        + NetTorque<NetTorque = <R as IntegrateRotation>::NetTorque>,
    S: Position<Position = V> + Default,
    X: PointUpdate<V, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    TR: Thermostat<M>,
    Microstate<B, S, X, C>: RotationalKineticEnergy<B, S>,
    <R as RotationalMotionTypes>::AngularMomentum: MulAssign<f64> + Clone,
{
    /// Integrate selected body orientations forward a full step and their angular momenta forward a half step.
    /// 
    /// If a selected body has no active rotational degrees of freedom, it is
    /// skipped.
    /// 
    /// For details, see the documentation for the implementations of
    /// [`IntegrateRotation`](crate::method::IntegrateRotation).
    #[inline]
    fn integrate_rotation_half_step_one_with_filter<
        F: Fn(&Tagged<Body<B, S>>) -> bool,
    >(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        integrate_rotation_half_step_one_with_filter::<V, R, B, S, X, C, TR, M, F>(
            self.delta_t,
            microstate,
            &mut self.rotational_thermostat,
            macrostate,
            should_integrate_body
        );
    }

    /// Integrate selected body angular momenta forward a half step.
    /// 
    /// If a selected body has no active rotational degrees of freedom, it is
    /// skipped.
    /// 
    /// For details, see the documentation for the implementations of
    /// [`IntegrateRotation`](crate::method::IntegrateRotation).
    #[inline]
    fn integrate_rotation_half_step_two_with_filter<
        F: Fn(&Tagged<Body<B, S>>) -> bool,
    >(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        integrate_rotation_half_step_two_with_filter::<V, R, B, S, X, C, TR, M, F>(
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
    use crate::{UpdateNetForceAndVirial, UpdateNetForceVirialAndTorque};
    use hoomd_interaction::{
        External, Rigid, Zero,
        external::{ConstantForce, ConstantTorque},
    };
    use hoomd_microstate::{
        Body,
        property::{DynamicOrientedPoint, DynamicPoint, Point},
    };
    use hoomd_vector::Outer;

    use approxim::assert_relative_eq;

    fn dynamics_body<const N: usize>(
        mass: f64,
    ) -> Body<DynamicPoint<Cartesian<N>>, Point<Cartesian<N>>> {
        Body {
            properties: DynamicPoint {
                mass,
                ..Default::default()
            },
            sites: vec![Point::new(Cartesian::default())],
        }
    }

    fn oriented_dynamics_body_2d(
        mass: f64,
        moment_of_inertia: f64,
    ) -> Body<DynamicOrientedPoint<Cartesian<2>, Angle>, Point<Cartesian<2>>> {
        Body {
            properties: DynamicOrientedPoint {
                position: Cartesian::<2>::default(),
                orientation: Angle::default(),
                momentum: Cartesian::<2>::default(),
                net_force: Cartesian::<2>::default(),
                net_virial: Cartesian::<2>::default().outer(&Cartesian::<2>::default()),
                moment_of_inertia,
                angular_momentum: 0.0,
                net_torque: 0.0,
                mass,
                ..Default::default()
            },
            sites: vec![Point::new(Cartesian::from([0.0, 0.0]))],
        }
    }

    #[test]
    fn test_constant_volume() {
        let dt = 2.0;
        let cv = ConstantVolume::builder(dt).build();
        assert_eq!(cv.delta_t, dt);
    }

    #[test]
    fn test_translational_integration() -> anyhow::Result<()> {
        // Ensure translational integration of a simple external force in 3D
        // yields the correct position and momentum at the half step and the
        // full step.
        let mass = 1.0;
        let dt = 0.1;
        let force = Cartesian::<3>::from([
            1.0 / 3.0_f64.sqrt(),
            1.0 / 3.0_f64.sqrt(),
            1.0 / 3.0_f64.sqrt(),
        ]);

        let mut microstate = Microstate::builder()
            .bodies([dynamics_body(mass)])
            .try_build()?;
        let rigid = Rigid(External(ConstantForce {
            force,
            r_0: [0.0, 0.0, 0.0].into(),
        }));
        let mut method = ConstantVolume::builder(dt).build();
        let macrostate = ();

        // Update force first so that the particles can move
        microstate.update_net_force_and_virial(&rigid);

        // Check the first half step
        method.integrate_translation_half_step_one(&mut microstate, &macrostate);
        let mut expected_momentum = Cartesian::<3>::default() + (force * dt * 0.5);
        let expected_position = Cartesian::<3>::default() + expected_momentum * dt / mass;

        assert_relative_eq!(
            expected_momentum,
            microstate.bodies()[0].item.properties.momentum
        );
        assert_relative_eq!(
            expected_position,
            microstate.bodies()[0].item.properties.position
        );

        // Update force again
        microstate.update_net_force_and_virial(&rigid);

        // Check the second half step
        method.integrate_translation_half_step_two(&mut microstate, &macrostate);
        expected_momentum += force * dt * 0.5;
        assert_relative_eq!(
            expected_momentum,
            microstate.bodies()[0].item.properties.momentum
        );
        assert_relative_eq!(
            expected_position,
            microstate.bodies()[0].item.properties.position
        );

        assert_eq!(microstate.conserved_degrees_of_freedom(), 3);

        Ok(())
    }

    #[test]
    fn test_conserved_degrees_of_freedom() -> anyhow::Result<()> {
        let mass = 1.0;
        let dt = 0.1;

        let mut microstate = Microstate::builder()
            .bodies([
                dynamics_body::<4>(mass),
                dynamics_body(mass),
                dynamics_body(mass),
            ])
            .try_build()?;

        let mut method = ConstantVolume::builder(dt).build();
        let macrostate = ();
        let rigid = Rigid(Zero);

        assert_eq!(microstate.conserved_degrees_of_freedom(), 0);

        method
            .integrate_translation_with_filter(&mut microstate, &macrostate, &rigid, |b| b.tag < 2);

        assert_eq!(microstate.conserved_degrees_of_freedom(), 0);

        method
            .integrate_translation_with_filter(&mut microstate, &macrostate, &rigid, |b| b.tag < 3);

        assert_eq!(microstate.conserved_degrees_of_freedom(), 4);

        Ok(())
    }

    #[test]
    fn test_rotational_integration_2d() -> anyhow::Result<()> {
        // Ensure rotational integration of a simple external torque in 2D
        // yields the correct orientation and angular momentum at the half step
        // and the full step
        let mass = 1.0;
        let moi = 1.0;
        let dt = 0.1;
        let t_mag = 1.0;
        let t_dir = 1.0;

        let mut microstate = Microstate::builder()
            .bodies([oriented_dynamics_body_2d(mass, moi)])
            .try_build()?;
        let torque = Rigid(External(ConstantTorque {
            torque: t_dir * t_mag,
        }));
        let mut method = ConstantVolume::builder(dt).build();
        let macrostate = ();

        // Update torque first so that the particles can move
        microstate.update_net_force_virial_and_torque(&torque);

        // Check the first half step
        method.integrate_rotation_half_step_one(&mut microstate, &macrostate);
        // RotationalMotion::<Angle, _, _, _, _, _>::integrate_rotation_half_step_one(&mut method, &mut microstate, &macrostate);
        let mut expected_angular_momentum = t_dir * t_mag * 0.5 * dt;
        let expected_orientation = Angle::default().theta + expected_angular_momentum / moi * dt;

        assert_eq!(
            expected_angular_momentum,
            microstate.bodies()[0].item.properties.angular_momentum
        );
        assert_eq!(
            expected_orientation,
            microstate.bodies()[0].item.properties.orientation.theta
        );

        // Update torque again
        microstate.update_net_force_virial_and_torque(&torque);

        // Check the second half step
        method.integrate_rotation_half_step_two(&mut microstate, &macrostate);
        // RotationalMotion::<Angle, _, _, _, _, _>::integrate_rotation_half_step_two(&mut method, &mut microstate, &macrostate);
        expected_angular_momentum += t_dir * t_mag * 0.5 * dt;
        assert_eq!(
            expected_angular_momentum,
            microstate.bodies()[0].item.properties.angular_momentum
        );
        assert_eq!(
            expected_orientation,
            microstate.bodies()[0].item.properties.orientation.theta
        );

        Ok(())
    }
}

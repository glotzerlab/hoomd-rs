// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ConstantVolume`.

use std::array;
use serde::{Deserialize, Serialize};

use hoomd_microstate::{
    Body, Microstate, SiteKey, Tagged, Transform, boundary::{GenerateGhosts, Wrap}, property::{
        AngularMomentum, DynamicOrientedPoint, Mass, MomentOfInertia, Momentum, NetForce, NetTorque, Orientation, Position
    }
};
use hoomd_vector::{
    Angle, Cartesian, InnerProduct, Quaternion, Rotate, Rotation, Versor
};
use crate::{RotationalKineticEnergy, RotationalMotion, Thermostat, TranslationalKineticEnergy, TranslationalMotion, thermostat::NoThermostat};
use hoomd_spatial::PointUpdate;

/// Integrate bodies' translational and rotational degrees of freedom in the microstate.
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
///    .thermostat(Bussi::default())
///    .build();
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
pub struct ConstantVolume<TT, TR=TT> {
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
    ///    .translational_thermostat(Bussi::default())
    ///    .build();
    /// ```
    #[inline]
    pub fn translational_thermostat<T>(self, translational_thermostat: T) -> ConstantVolumeBuilder<T, TR> {
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
    ///    .rotational_thermostat(Bussi::default())
    ///    .build();
    /// ```
    #[inline]
    pub fn rotational_thermostat<T>(self, rotational_thermostat: T) -> ConstantVolumeBuilder<TT, T> {
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
    ///    .thermostat(Bussi::default())
    ///    .build();
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
    ///     ```math
    ///     \vec{p}_i\left( t \right) = \vec{p'}_i\left( t \right) \cdot \mathrm{translational\_thermostat.integrate\_half\_step\_one}\left( \sum_{j \in \mathrm{selection}} K'_{trans,j} \left( t \right) \right)
    ///     ```
    ///     where the summation represents the total [translational kinetic energy](crate::compute::TranslationalKineticEnergy)
    ///     of the selected bodies at the start of the step, and `translational_thermostat.integrate_half_step_one()` is the
    ///     first half step method implemented by `TT`.
    /// 
    /// 2. Momentum is integrated forward a half step.
    /// 
    ///     ```math
    ///     \vec{p}_i\left( t + \frac{\Delta t}{2} \right) = \vec{p}_i\left( t \right) + \vec{F}_i(t) \frac{\Delta t}{2}
    ///     ```
    /// 
    /// 3. Position is integrated forward a full step using the new momentum.
    /// 
    ///     ```math
    ///     \vec{r}_i\left( t + \Delta t \right) = \vec{r}_i\left( t \right) + \frac{\vec{p}_i\left( t + \frac{\Delta t}{2} \right)}{m_i} \Delta t
    ///     ```
    #[inline]
    fn integrate_translation_half_step_one_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        let (kinetic_energy, degrees_of_freedom) = microstate.translational_kinetic_energy_with_filter(&should_integrate_body);
        let rescaling_factor = self.translational_thermostat.integrate_half_step_one(
            &mut rng,
            macrostate,
            self.delta_t,
            kinetic_energy,
            degrees_of_freedom,
        );

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue
            }
            let mut body_properties = body.item.properties.clone();

            let net_force = *body_properties.net_force();
            let mass = body_properties.mass();
            let mut momentum = *body_properties.momentum();

            momentum *= rescaling_factor;
            momentum += net_force * 0.5 * self.delta_t;
            *body_properties.position_mut() += momentum / mass * self.delta_t;
            *body_properties.momentum_mut() = momentum;

            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.");
        }

        microstate.increment_substep();
    }

    /// Integrate selected body momenta forward a half step.
    /// 
    /// The second half step of the symplectic integration procedure is given by the equations below, which are
    /// applied to each selected body *i*. In each step, the marker $`'`$ is used when a variable's value changes
    /// during a step to distinguish the value before ( $`'`$ is present) from the value after ( $`'`$ is absent).
    /// 
    /// 1. Momentum is integrated forward a half step.
    /// 
    ///     ```math
    ///     \vec{p}_i\left( t + \Delta t \right) = \vec{p}_i\left( t + \frac{\Delta t}{2} \right) + \vec{F}_i\left( t + \frac{\Delta t}{2} \right) \frac{\Delta t}{2}
    ///     ```
    /// 
    /// 2. The translational thermostat is integrated forward a half step and then momentum is rescaled accordingly.
    /// 
    ///     ```math
    ///     \vec{p}_i\left( t + \Delta t \right) = \vec{p'}_i\left( t + \Delta t \right) \cdot \mathrm{translational\_thermostat.integrate\_half\_step\_two}\left( \sum_{j \in \mathrm{selection}} K'_{trans,j} \left( t + \Delta t \right) \right)
    ///     ```
    /// 
    ///     where the summation represents the total [translational kinetic energy](crate::compute::TranslationalKineticEnergy)
    ///     of the selected bodies at the start of the step, and `translational_thermostat.integrate_half_step_two()` is the
    ///     second half step method implemented by `TT`.
    #[inline]
    fn integrate_translation_half_step_two_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue
            }
            let mut body_properties = body.item.properties.clone();
            let net_force = *body_properties.net_force();

            *body_properties.momentum_mut() += net_force * self.delta_t * 0.5;

            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.");
        }

        let (kinetic_energy, degrees_of_freedom) = microstate.translational_kinetic_energy();
        let rescaling_factor = self.translational_thermostat.integrate_half_step_two(
            &mut rng,
            macrostate,
            self.delta_t,
            kinetic_energy,
            degrees_of_freedom,
        );

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue
            }
            let mut body_properties = body.item.properties.clone();

            *body_properties.momentum_mut() *= rescaling_factor;

            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.");
        }

        microstate.increment_substep();
    }
}


/// Compute the net torque in the body frame.
///
/// Also determine which of the three rotational degrees of freedom are active.
fn body_net_torque_and_active_degrees_of_freedom(body_properties: &DynamicOrientedPoint<Cartesian<3>, Versor>) -> (Cartesian<3>, [bool; 3]) {
    let q = body_properties.orientation();
    let moment_of_inertia = body_properties.moment_of_inertia();

    let mut net_torque = q.inverted().rotate(body_properties.net_torque());

    let active = array::from_fn(|i| moment_of_inertia[i] != 0.0);

    // Limited numerical precision might lead to non-zero torques about axes that should
    // not be integrated. Zeroing these out improves the stability of the integration.
    for i in 0..3 {
        if !active[i] {
            net_torque[i] = 0.0;
        }
    }

    (net_torque, active)
}

/// Rotational motion in 3-dimensional cartesian space.
impl<S, X, C, TT, TR, M> RotationalMotion<DynamicOrientedPoint<Cartesian<3>, Versor>, S, X, C, M> for ConstantVolume<TT, TR>
where
    DynamicOrientedPoint<Cartesian<3>, Versor>: Transform<S>,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<DynamicOrientedPoint<Cartesian<3>, Versor>> + Wrap<S> + GenerateGhosts<S>,
    TR: Thermostat<M>,
{
    /// Integrate selected body orientations forward a full step and their angular momenta forward a half step.
    /// 
    /// The first half step of the symplectic integration procedure is given by the equations below, which are
    /// applied to each selected body *i*. In each step, the marker $`'`$ is used when a variable's value changes
    /// during a step to distinguish the value before ( $`'`$ is present) from the value after ( $`'`$ is absent).
    /// Rotational degrees of freedom with a moment of inertia component of zero are skipped.
    /// 
    /// 1. The rotational thermostat is integrated forward a half-step and then angular momentum is rescaled
    /// accordingly:
    /// 
    ///     ```math
    ///     \vec{L}_i(t) = \vec{L}'_i(t) \cdot \mathrm{rotational\_thermostat.integrate\_half\_step\_one}\left(\sum_{j \in \mathrm{selection}} K'_{rot,j}(t) \right)
    ///     ```
    /// 
    ///     where the summation represents the total [rotational kinetic energy](crate::compute::RotationalKineticEnergy)
    ///     of the selected bodies at the start of the step, and `rotational_thermostat.integrate_half_step_one()` is the
    ///     first half step method implemented by `TR`.
    ///
    /// 2. Angular momentum $`\vec{L}`$ and orientation $`\mathbf{q}`$ are integrated forward. These integrations
    /// follow a complex, multistep process, so a fuller explanation is provided below. In each step, the body
    /// index *i* and time *t* are implicit on every variable unless otherwise specified.
    /// 
    ///     1. Angular momentum and net torque are converted to quaternions $`\mathbf{p}`$ and
    ///     $`\mathbf{f}`$, respectively:
    /// 
    ///         ```math
    ///         \begin{align*}
    ///         
    ///         \mathbf{p} &= 2\mathbf{S}(\mathbf{q}) \mathbf{L} \\
    ///         \mathbf{f} &= 2\mathbf{S}(\mathbf{q}) \boldsymbol{\tau} \\
    ///             
    ///         \end{align*}
    ///         ```
    ///         
    ///         where
    ///         
    ///         ```math
    ///         \begin{align*}
    ///         
    ///         \mathbf{L} &= (0, L_x, L_y, L_z) \\
    ///         \boldsymbol{\tau} &= (0, \tau_x, \tau_y, \tau_z) \\
    ///         
    ///         \mathbf{S}(\mathbf{q}) &= 
    ///         \begin{pmatrix}
    ///         q_0 & -q_1 & -q_2 & -q_3\\
    ///         q_1 & q_0 & -q_3 & q_2\\
    ///         q_2 & q_3 & q_0 & -q_1\\
    ///         q_3 & -q_2 & q_1 & q_0
    ///         \end{pmatrix}
    ///             
    ///         \end{align*}
    ///         ```
    ///
    ///     2. $`\mathbf{p}`$ and $`\mathbf{q}`$ are integrated forward using the NOvel Symplectic
    ///     QUaternIon ScHeme (NO_SQUISH) algorithm, which ensures the integration is both symplctic
    ///     and preserves orientation quaternion unity. There are several steps to this algorithm, whose
    ///     equations are given below.
    /// 
    ///         1. $`\mathbf{p}`$ is partially integrated forward a half step.
    /// 
    ///             ```math
    ///             \mathbf{p} = \mathbf{p}' + \frac{\Delta t}{2} \mathbf{f}
    ///             ```
    /// 
    ///         2. $`\mathbf{p}`$ is integrated forward the remainder of the half step and $`\mathbf{q}`$ is integrated
    ///         forward a full step. Properties of quaternion algebra are used to decompose the Liouvillian into a
    ///         sum over permutation matrices applied to $`\mathbf{q}`$ and $`\mathbf{p}`$. There are five steps
    ///         to this decomposition:
    /// 
    ///             ```math
    ///             \begin{align*}
    ///             
    ///             \phi_3 &= \frac{1}{4 I_{33}} \mathrm{dot} \left( \mathbf{p}, P_3 \mathbf{q} \right) \\
    ///             \mathbf{q} &= \cos{(\phi_3 \frac{\Delta t}{2})} \mathbf{q}^{'} +  \sin{(\phi_3 \frac{\Delta t}{2})} P_3 \mathbf{q}^{'} \nonumber \\
    ///             \mathbf{p} &= \cos{(\phi_3 \frac{\Delta t}{2})} \mathbf{p}' +  \sin{(\phi_3 \frac{\Delta t}{2})} P_3 \mathbf{p}' \nonumber \\ \nonumber \\
    ///             
    ///             \phi_2 &= \frac{1}{4 I_{22}} \mathrm{dot} \left( \mathbf{p}, P_2 \mathbf{q} \right) \\
    ///             \mathbf{q} &= \cos{(\phi_2 \frac{\Delta t}{2})} \mathbf{q}^{'} +  \sin{(\phi_2 \frac{\Delta t}{2})} P_2 \mathbf{q}^{'} \nonumber \\
    ///             \mathbf{p} &= \cos{(\phi_2 \frac{\Delta t}{2})} \mathbf{p}' +  \sin{(\phi_2 \frac{\Delta t}{2})} P_2 \mathbf{p}' \nonumber \\ \nonumber \\
    ///             
    ///             \phi_1 &= \frac{1}{4 I_{11}} \mathrm{dot} \left( \mathbf{p}, P_1 \mathbf{q} \right) \\
    ///             \mathbf{q} &= \cos{(\phi_1 \Delta t)} \mathbf{q}^{'} +  \sin{(\phi_1 \Delta t)} P_1 \mathbf{q}^{'} \nonumber \\
    ///             \mathbf{p} &= \cos{(\phi_1 \Delta t)} \mathbf{p}' +  \sin{(\phi_1 \Delta t)} P_1 \mathbf{p}' \nonumber  \nonumber \\ \nonumber \\
    ///             
    ///             \phi_2 &= \frac{1}{4 I_{22}} \mathrm{dot} \left( \mathbf{p}, P_2 \mathbf{q} \right) \\
    ///             \mathbf{q} &= \cos{(\phi_2 \frac{\Delta t}{2})} \mathbf{q}^{'} +  \sin{(\phi_2 \frac{\Delta t}{2})} P_2 \mathbf{q}^{'} \nonumber \\
    ///             \mathbf{p} &= \cos{(\phi_2 \frac{\Delta t}{2})} \mathbf{p}' +  \sin{(\phi_2 \frac{\Delta t}{2})} P_2 \mathbf{p}' \nonumber  \nonumber \\ \nonumber \\
    ///             
    ///             \phi_3 &= \frac{1}{4 I_{33}} \mathrm{dot} \left( \mathbf{p}, P_3 \mathbf{q} \right) \\
    ///             \mathbf{q} \left( t + \Delta t \right) &= \cos{(\phi_3 \frac{\Delta t}{2})} \mathbf{q}^{'} +  \sin{(\phi_3 \frac{\Delta t}{2})} P_3 \mathbf{q}^{'} \nonumber \\
    ///             \mathbf{p} \left( t + \frac{\Delta t}{2} \right) &= \cos{(\phi_3 \frac{\Delta t}{2})} \mathbf{p}' +  \sin{(\phi_3 \frac{\Delta t}{2})} P_3 \mathbf{p}' \nonumber    \nonumber \\ \nonumber \\
    ///             
    ///             \end{align*} 
    ///             ```
    /// 
    ///             where $`I_{kk}`$ is the component of the moment of inertia for $`k = 1, 2, 3`$ and $`P_k`$ is the corresponding
    ///             permuation matrix such that
    /// 
    ///             ```math
    ///             \begin{align*}
    ///             
    ///             P_0\mathbf{q} &= (q_0, q_1, q_2, q_3) \\
    ///             P_1\mathbf{q} &= (-q_1, q_0, q_3, -q_2) \\
    ///             P_2\mathbf{q} &= (-q_2, -q_3, q_0, q_1) \\
    ///             P_3\mathbf{q} &= (-q_3, q_2, -q_1, q_0) \\
    ///             (PP^T)_{\alpha \beta} &= \delta_{\alpha \beta} \\
    ///             
    ///             \end{align*}
    ///             ```
    /// 
    ///     3. $`\mathbf{p}`$ is converted back into vector-form angular momentum:
    /// 
    ///         ```math
    ///         \mathbf{L} \left( t + \frac{\Delta t}{2} \right) = \frac{1}{2} \mathbf{S}(\mathbf{q})^T \mathbf{p} \left( t + \frac{\Delta t}{2} \right)
    ///         ```
    /// 
    ///         where
    ///         
    ///         ```math
    ///         \begin{align*}
    ///         \mathbf{L} &= (0, L_x, L_y, L_z) \\
    ///         \vec{L} &= (L_x, L_y, L_z)
    ///         \end{align*}
    ///         ```
    #[inline]
    fn integrate_rotation_half_step_one_with_filter<F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<3>, Versor>, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<DynamicOrientedPoint<Cartesian<3>, Versor>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        let (kinetic_energy, degrees_of_freedom) = microstate.rotational_kinetic_energy_with_filter(&should_integrate_body);
        let rescaling_factor = self.rotational_thermostat.integrate_half_step_one(
            &mut rng,
            macrostate,
            self.delta_t,
            kinetic_energy,
            degrees_of_freedom,
        );

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue
            }
            let mut body_properties = body.item.properties;

            let (net_torque, active) = body_net_torque_and_active_degrees_of_freedom(&body_properties);
            let mut q = *body_properties.orientation().get();
            let moment_of_inertia = body_properties.moment_of_inertia();

            // DynamicOrientedPoint stores angular momentum in vector form. Convert it
            // into a quaternion, integrate the quaternion, then store it back as a vector.
            let s = *body_properties.angular_momentum();
            let mut p = (q * Quaternion::pure(s)) * 2.0;
            
            p = p * rescaling_factor + q * Quaternion::pure(net_torque) * self.delta_t;

            if active[2] {
                let p3 = Quaternion::from([-p.vector[2], p.vector[1], -p.vector[0], p.scalar]);
                let q3 = Quaternion::from([
                    -q.vector[2],
                    q.vector[1],
                    -q.vector[0],
                    q.scalar,
                ]);
                let phi3 = (1. / (4. * moment_of_inertia[2])) * ((p.scalar * q3.scalar) + p.vector.dot(&q3.vector));
                let c_phi3 = (0.5 * self.delta_t * phi3).cos();
                let s_phi3 = (0.5 * self.delta_t * phi3).sin();

                p = p * c_phi3 + p3 * s_phi3;
                q = q * c_phi3 + q3 * s_phi3;
            }

            if active[1] {
                let p2 = Quaternion::from([-p.vector[1], -p.vector[2], p.scalar, p.vector[0]]);
                let q2 = Quaternion::from([
                    -q.vector[1],
                    -q.vector[2],
                    q.scalar,
                    q.vector[0],
                ]);
                let phi2 = (1. / (4. * moment_of_inertia[1])) * ((p.scalar * q2.scalar) + p.vector.dot(&q2.vector));
                let c_phi2 = (0.5 * self.delta_t * phi2).cos();
                let s_phi2 = (0.5 * self.delta_t * phi2).sin();

                p = p * c_phi2 + p2 * s_phi2;
                q = q * c_phi2 + q2 * s_phi2;
            }

            if active[0] {
                let p1 = Quaternion::from([-p.vector[0], p.scalar, p.vector[2], -p.vector[1]]);
                let q1 = Quaternion::from([
                    -q.vector[0],
                    q.scalar,
                    q.vector[2],
                    -q.vector[1],
                ]);
                let phi1 = (1. / (4. * moment_of_inertia[0])) * ((p.scalar * q1.scalar) + p.vector.dot(&q1.vector));
                let c_phi1 = (self.delta_t * phi1).cos();
                let s_phi1 = (self.delta_t * phi1).sin();

                p = p * c_phi1 + p1 * s_phi1;
                q = q * c_phi1 + q1 * s_phi1;
            }

            if active[1] {
                let p2 = Quaternion::from([-p.vector[1], -p.vector[2], p.scalar, p.vector[0]]);
                let q2 = Quaternion::from([
                    -q.vector[1],
                    -q.vector[2],
                    q.scalar,
                    q.vector[0],
                ]);
                let phi2 = (1. / (4. * moment_of_inertia[1])) * ((p.scalar * q2.scalar) + p.vector.dot(&q2.vector));
                let c_phi2 = (0.5 * self.delta_t * phi2).cos();
                let s_phi2 = (0.5 * self.delta_t * phi2).sin();

                p = p * c_phi2 + p2 * s_phi2;
                q = q * c_phi2 + q2 * s_phi2;
            }

            if active[2] {
                let p3 = Quaternion::from([-p.vector[2], p.vector[1], -p.vector[0], p.scalar]);
                let q3 = Quaternion::from([
                    -q.vector[2],
                    q.vector[1],
                    -q.vector[0],
                    q.scalar,
                ]);
                let phi3 = (1. / (4. * moment_of_inertia[2])) * ((p.scalar * q3.scalar) + p.vector.dot(&q3.vector));
                let c_phi3 = (0.5 * self.delta_t * phi3).cos();
                let s_phi3 = (0.5 * self.delta_t * phi3).sin();

                p = p * c_phi3 + p3 * s_phi3;
                q = q * c_phi3 + q3 * s_phi3;
            }

            *body_properties.orientation_mut() = q.to_versor().expect("body orientation should be non-zero");
            *body_properties.angular_momentum_mut() = ((q.conjugate() * p) * 0.5).vector;

            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.");
        }

        microstate.increment_substep();
    }

    /// Integrate selected body angular momenta forward a half step.
    /// 
    /// The second half step of the symplectic integration procedure is given by the equations below, which are
    /// applied to each selected body *i*. In each step, the marker $`'`$ is used when a variable's value changes
    /// during a step to distinguish the value before ( $`'`$ is present) from the value after ( $`'`$ is absent).
    /// The time $`t + \frac{\Delta t}{2}`$ is implicit on every variable unless otherwise specified.
    /// Rotational degrees of freedom with a moment of inertia component of zero are skipped.
    ///
    /// 1. Angular momentum and net torque are converted to quaternions $`\mathbf{p}`$ and
    /// $`\mathbf{f}`$, respectively:
    ///
    ///     ```math
    ///     \begin{align*}
    ///     
    ///     \mathbf{p} &= 2\mathbf{S}(\mathbf{q}) \mathbf{L} \\
    ///     \mathbf{f} &= 2\mathbf{S}(\mathbf{q}) \boldsymbol{\tau} \\
    ///         
    ///     \end{align*}
    ///     ```
    ///     
    ///     where
    ///     
    ///     ```math
    ///     \begin{align*}
    ///     
    ///     \mathbf{L} &= (0, L_x, L_y, L_z) \\
    ///     \boldsymbol{\tau} &= (0, \tau_x, \tau_y, \tau_z) \\
    ///     
    ///     \mathbf{S}(\mathbf{q}) &= 
    ///     \begin{pmatrix}
    ///     q_0 & -q_1 & -q_2 & -q_3\\
    ///     q_1 & q_0 & -q_3 & q_2\\
    ///     q_2 & q_3 & q_0 & -q_1\\
    ///     q_3 & -q_2 & q_1 & q_0
    ///     \end{pmatrix}
    ///         
    ///     \end{align*}
    ///      ```
    /// 
    /// 2. $`\mathbf{p}`$ is integrated forward a half step.
    ///
    ///     ```math
    ///     \mathbf{p}\left( t + \Delta t \right) = \mathbf{p}\left( t + \frac{\Delta t}{2} \right) + \frac{\Delta t}{2} \mathbf{f}
    ///     ```
    /// 
    /// 3. $`\mathbf{p}`$ is converted back into vector-form angular momentum:
    ///
    ///     ```math
    ///     \mathbf{L} \left( t + \Delta t \right) = \frac{1}{2} \mathbf{S}(\mathbf{q})^T \mathbf{p} \left( t + \Delta t \right)
    ///     ```
    ///
    ///     where
    ///     
    ///     ```math
    ///     \begin{align*}
    ///     \mathbf{L} &= (0, L_x, L_y, L_z) \\
    ///     \vec{L} &= (L_x, L_y, L_z)
    ///     \end{align*}
    ///     ```
    /// 
    /// 4. The rotational thermostat is integrated forward a half-step and then angular momentum is rescaled
    /// accordingly. (Note: `rotational_thermostat.integrate_half_step_two()` is the first half step method
    /// implemented by `TR`.)
    /// 
    ///     ```math
    ///     \vec{L}_i(t + \Delta t) = \vec{L}'_i(t + \Delta t) \cdot \mathrm{rotational\_thermostat.integrate\_half\_step\_two}\left(\sum_{i \in \mathrm{selection}} K'_{rot,j}(t + \Delta t) \right)
    ///     ```
    /// 
    ///     where the summation represents the total [rotational kinetic energy](crate::compute::RotationalKineticEnergy)
    ///     of the selected bodies at the start of the step, and `rotational_thermostat.integrate_half_step_two()` is the
    ///     second half step method implemented by `TR`.
    #[inline]
    fn integrate_rotation_half_step_two_with_filter<F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<3>, Versor>, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<DynamicOrientedPoint<Cartesian<3>, Versor>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue
            }
            let mut body_properties = body.item.properties;

            let (net_torque, _) = body_net_torque_and_active_degrees_of_freedom(&body_properties);
            let q = *body_properties.orientation().get();
            let s = *body_properties.angular_momentum();

            let mut p = q * Quaternion::pure(s) * 2.0;

            p += (q * Quaternion::pure(net_torque)) * self.delta_t;

            *body_properties.angular_momentum_mut() = ((q.conjugate() * p) * 0.5).vector;

            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.");
        }

        let (kinetic_energy, degrees_of_freedom) = microstate.rotational_kinetic_energy_with_filter(&should_integrate_body);
        let rescaling_factor = self.rotational_thermostat.integrate_half_step_two(
            &mut rng,
            macrostate,
            self.delta_t,
            kinetic_energy,
            degrees_of_freedom,
        );

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue
            }
            let mut body_properties = body.item.properties;


            *body_properties.angular_momentum_mut() *= rescaling_factor;

            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.");
        }

        microstate.increment_substep();
    }
}

/// Rotational motion in 2-dimensional cartesian space.
impl<S, X, C, TT, TR, M> RotationalMotion<DynamicOrientedPoint<Cartesian<2>, Angle>, S, X, C, M> for ConstantVolume<TT, TR>
where
    DynamicOrientedPoint<Cartesian<2>, Angle>: Transform<S>,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<DynamicOrientedPoint<Cartesian<2>, Angle>> + Wrap<S> + GenerateGhosts<S>,
    TR: Thermostat<M>,
{
    /// Integrate selected body orientations forward a full step and their angular momenta forward a half step.
    /// 
    /// The first half step of the symplectic integration procedure is given by the equations below, which are
    /// applied to each selected body *i*. In each step, the marker $`'`$ is used when a variable's value changes
    /// during a step to distinguish the value before ( $`'`$ is present) from the value after ( $`'`$ is absent).
    /// Selected bodies which have ``moment_of_inertia = 0.0`` are skipped.
    /// 
    /// 1. The rotational thermostat is integrated forward a half-step and then angular momentum is rescaled
    /// accordingly:
    ///
    ///     ```math
    ///     L_i(t) = L'_i(t) \cdot \mathrm{rotational\_thermostat.integrate\_half\_step\_one}\left(\sum_{j \in \mathrm{selection}} K'_{rot,j}(t) \right)
    ///     ```
    /// 
    ///     where the summation represents the total [rotational kinetic energy](crate::compute::RotationalKineticEnergy)
    ///     of the selected bodies at the start of the step, and `rotational_thermostat.integrate_half_step_one()` is the
    ///     first half step method implemented by `TR`.
    /// 
    /// 2. Angular momentum is integrated forward a half step.
    /// 
    ///     ```math
    ///     L_i\left(t + \frac{\Delta t}{2} \right) = L_i(t) + \tau_i(t) \frac{\Delta t}{2}
    ///     ```
    /// 
    /// 3. Orientation is integrated forward a full step using the new angular momentum.
    /// 
    ///     ```math
    ///     \theta_i(t + \Delta t) = \theta_i(t) + \frac{L_i\left( t + \frac{\Delta t}{2} \right)}{I_i} \Delta t
    ///     ```
    #[inline]
    fn integrate_rotation_half_step_one_with_filter<F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<2>, Angle>, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<DynamicOrientedPoint<Cartesian<2>, Angle>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        let (kinetic_energy, degrees_of_freedom) = microstate.rotational_kinetic_energy_with_filter(&should_integrate_body);
        let rescaling_factor = self.rotational_thermostat.integrate_half_step_one(
            &mut rng,
            macrostate,
            self.delta_t,
            kinetic_energy,
            degrees_of_freedom,
        );

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue
            }

            let mut body_properties = body.item.properties;

            let moment_of_inertia = *body_properties.moment_of_inertia();
            if moment_of_inertia == 0.0 {
                continue
            }

            let net_torque = *body_properties.net_torque();

            *body_properties.angular_momentum_mut() *= rescaling_factor;
            *body_properties.angular_momentum_mut() += net_torque * 0.5 * self.delta_t;
            body_properties.orientation_mut().theta +=
                *body_properties.angular_momentum() / moment_of_inertia * self.delta_t;

            *body_properties.orientation_mut() = body_properties.orientation_mut().to_reduced();

            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.");
        }

        microstate.increment_substep();
    }

    /// Integrate selected body angular momenta forward a half step.
    /// 
    /// The second half step of the symplectic integration procedure is given by the equations below, which are
    /// applied to each selected body *i*. In each step, the marker $`'`$ is used when a variable's value changes
    /// during a step to distinguish the value before ( $`'`$ is present) from the value after ( $`'`$ is absent).
    /// Selected bodies which have ``moment_of_inertia = 0.0`` are skipped.
    /// 
    /// 1. Angular momentum is integrated forward a half step.
    /// 
    ///     ```math
    ///     L_i(t + \Delta t) = L_i\left( t + \frac{\Delta t}{2} \right) + \tau_i \left(t + \frac{\Delta t}{2} \right) \frac{\Delta t}{2}
    ///     ```
    /// 
    /// 2. The rotational thermostat is integrated forward a half step and then angular momentum
    /// is rescaled accordingly.
    /// 
    ///     ```math
    ///     L_i(t + \Delta t) = L'_i(t + \Delta t) \cdot \mathrm{rotational\_thermostat.integrate\_half\_step\_two}\left(\sum_{j \in \mathrm{selection}}K'_{rot,j}(t + \Delta t) \right)
    ///     ```
    /// 
    ///     where the summation represents the total [rotational kinetic energy](crate::compute::RotationalKineticEnergy)
    ///     of the selected bodies at the start of the step, and `rotational_thermostat.integrate_half_step_two()` is the
    ///     second half step method implemented by `TR`.

    #[inline]
    fn integrate_rotation_half_step_two_with_filter<F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<2>, Angle>, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<DynamicOrientedPoint<Cartesian<2>, Angle>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue
            }

            let mut body_properties = body.item.properties;

            let moment_of_inertia = *body_properties.moment_of_inertia();
            if moment_of_inertia == 0.0 {
                continue
            }

            let net_torque = *body_properties.net_torque();

            *body_properties.angular_momentum_mut() += net_torque * 0.5 * self.delta_t;

            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.");
        }

        let (kinetic_energy, degrees_of_freedom) = microstate.rotational_kinetic_energy();
        let rescaling_factor = self.rotational_thermostat.integrate_half_step_two(
            &mut rng,
            macrostate,
            self.delta_t,
            kinetic_energy,
            degrees_of_freedom,
        );

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue
            }
            let mut body_properties = body.item.properties;


            *body_properties.angular_momentum_mut() *= rescaling_factor;

            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.");
        }

        microstate.increment_substep();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_interaction::{External, external::{ConstantForce, ConstantTorque}, Rigid};
    use hoomd_microstate::{Body, property::{DynamicPoint, DynamicOrientedPoint, Point}};
    use crate::{UpdateNetForce, UpdateNetForceAndTorque};

    use approxim::assert_relative_eq;

    fn dynamics_body_3d(mass: f64) -> Body<DynamicPoint<Cartesian<3>>, Point<Cartesian<3>>> {
        Body {
            properties: DynamicPoint {
                position: Cartesian::<3>::default(),
                momentum: Cartesian::<3>::default(),
                net_force: Cartesian::<3>::default(),
                mass,
            },
            sites: vec![Point::new(Cartesian::from([0.0, 0.0, 0.0]))],
        }
    }

    fn oriented_dynamics_body_2d(mass: f64, moment_of_inertia: f64) -> Body<DynamicOrientedPoint<Cartesian<2>, Angle>, Point<Cartesian<2>>> {
        Body {
            properties: DynamicOrientedPoint {
                position: Cartesian::<2>::default(),
                orientation: Angle::default(),
                momentum: Cartesian::<2>::default(),
                net_force: Cartesian::<2>::default(),
                moment_of_inertia,
                angular_momentum: 0.0,
                net_torque: 0.0,
                mass,
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
        let force = Cartesian::<3>::from(
            [1.0 / 3.0_f64.sqrt(), 1.0 / 3.0_f64.sqrt(), 1.0 / 3.0_f64.sqrt()]
        );

        let mut microstate = Microstate::builder()
            .bodies([dynamics_body_3d(mass)])
            .try_build()?;
        let rigid = Rigid(External(ConstantForce {
            force,
            r_0: [0.0, 0.0, 0.0].into(),
        }));
        let mut method = ConstantVolume::builder(dt).build();
        let macrostate = ();

        // Update force first so that the particles can move
        microstate.update_net_force(&rigid);
        
        // Check the first half step
        method.integrate_translation_half_step_one(
            &mut microstate,
            &macrostate
        );
        let mut expected_momentum = Cartesian::<3>::default()
            + (force * dt * 0.5);
        let expected_position = Cartesian::<3>::default()
            + expected_momentum * dt / mass;

        assert_relative_eq!(expected_momentum, microstate.bodies()[0].item.properties.momentum);
        assert_relative_eq!(expected_position, microstate.bodies()[0].item.properties.position);

        // Update force again
        microstate.update_net_force(&rigid);

        // Check the second half step
         method.integrate_translation_half_step_two(
            &mut microstate,
            &macrostate
        );
        expected_momentum += force * dt * 0.5;
        assert_relative_eq!(expected_momentum, microstate.bodies()[0].item.properties.momentum);
        assert_relative_eq!(expected_position, microstate.bodies()[0].item.properties.position);

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
        microstate.update_net_force_and_torque(&torque);
        
        // Check the first half step
        method.integrate_rotation_half_step_one(
            &mut microstate,
            &macrostate
        );
        let mut expected_angular_momentum = t_dir * t_mag * 0.5 * dt;
        let expected_orientation = Angle::default().theta
            + expected_angular_momentum / moi * dt;

        assert_eq!(expected_angular_momentum, microstate.bodies()[0].item.properties.angular_momentum);
        assert_eq!(expected_orientation, microstate.bodies()[0].item.properties.orientation.theta);

        // Update torque again
        microstate.update_net_force_and_torque(&torque);

        // Check the second half step
         method.integrate_rotation_half_step_two(
            &mut microstate,
            &macrostate
        );
        expected_angular_momentum += t_dir * t_mag * 0.5 * dt;
        assert_eq!(expected_angular_momentum, microstate.bodies()[0].item.properties.angular_momentum);
        assert_eq!(expected_orientation, microstate.bodies()[0].item.properties.orientation.theta);

        Ok(())
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::array;

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

/// Perform time integration on the [`Microstate`] with the volume constraining
/// to a constant using Velocity Verlet algorithm.
/// 
/// When [`NoThermostat`](crate::thermostat::NoThermostat) is provided to the methods:
/// [`integrate_translation_half_step_one`](ConstantVolume::integrate_translation_half_step_one), 
/// [`integrate_translation_half_step_two`](ConstantVolume::integrate_translation_half_step_two), 
/// [`integrate_rotation_half_step_one`](ConstantVolume::integrate_rotation_half_step_one), and 
/// [`integrate_rotation_half_step_two`](ConstantVolume::integrate_rotation_half_step_two), it 
/// samples the microcanonical (NVE) ensemble. Otherwise, It samples the 
/// canonical (NVT) ensemble using the given [`macrostate`](hoomd_simulation::macrostate::Isothermal)
/// and [`Thermostat`].
/// 
/// The force and torque updates are separated from the four integration methods mentioned
/// above. To perform the integration correctly, force and torque must be updated in between
/// the first (step_one) and second-half (step_two) of the integration by using the methods:
/// [`update_force`](ConstantVolume::update_force), [`update_torque`](ConstantVolume::update_torque)
/// , or [`update_force_and_torque`](ConstantVolume::update_force_and_torque).
/// 
/// The imeplementation follows the sympletic integration scheme by [Tuckerman et al. 2006] 
/// for translational motion and [Miller et al. 2002] for rotational motion.
/// 
/// # Example
///
/// ```
/// use hoomd_md::method::ConstantVolume;
///
/// // Create a constant-volume integrator
/// let dt = 0.001;
/// let integrator = ConstantVolume::new(dt);
/// ```
/// 
/// # Reference
/// 
/// [Tuckerman et al. 2006]
/// 
/// [Miller et al. 2002]
/// 
/// [Tuckerman et al. 2006]: <https://doi.org/10.1088/0305-4470/39/19/S18>
/// [Miller et al. 2002]: <https://doi.org/10.1063/1.1473654>
#[doc(alias = "nvt")]
#[derive(Clone, Debug, PartialEq)]
pub struct ConstantVolume<TT, TR> {
    /// The time step size.
    pub delta_t: f64,

    /// Translational thermostat.
    pub translational_thermostat: TT,

    /// Rotational thermostat.
    pub rotational_thermostat: TR,

    // TODO: Should `ConstantVolume` track the last updated step and panic when:
    // - Step two is not called after step one?
    // - Step one is called twice for the same step?
}

pub struct ConstantVolumeBuilder<TT, TR> {
    /// The time step size.
    delta_t: f64,

    /// Translational thermostat.
    translational_thermostat: TT,

    /// Rotational thermostat.
    rotational_thermostat: TR,
}

impl<TT, TR> ConstantVolumeBuilder<TT, TR> {
    pub fn translational_thermostat<T>(self, translational_thermostat: T) -> ConstantVolumeBuilder<T, TR> {
        ConstantVolumeBuilder {
            delta_t: self.delta_t,
            translational_thermostat,
            rotational_thermostat: self.rotational_thermostat,
        }
    }
    pub fn rotational_thermostat<T>(self, rotational_thermostat: T) -> ConstantVolumeBuilder<TT, T> {
        ConstantVolumeBuilder {
            delta_t: self.delta_t,
            translational_thermostat: self.translational_thermostat,
            rotational_thermostat,
        }
    }
    pub fn thermostat<T: Clone>(self, thermostat: T) -> ConstantVolumeBuilder<T, T> {
        ConstantVolumeBuilder {
            delta_t: self.delta_t,
            translational_thermostat: thermostat.clone(),
            rotational_thermostat: thermostat,
        }
    }

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
    pub fn builder(delta_t: f64) -> ConstantVolumeBuilder<NoThermostat, NoThermostat> {
        ConstantVolumeBuilder {
            delta_t,
            translational_thermostat: NoThermostat,
            rotational_thermostat: NoThermostat,
        }
    } 

    /// Construct a new [`ConstantVolume`] with the given time step size and no thermostats.
    #[inline]
    pub fn new(delta_t: f64) -> Self {
        Self {
            delta_t,
            translational_thermostat: NoThermostat,
            rotational_thermostat: NoThermostat,
        }
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
    /// Perform the first-half integration on translational degrees-of-freedom
    /// , advancing the [`Microstate`] and possibly the [`Thermostat`] state forward as 
    /// :
    /// ```math
    /// \begin{align}
    ///
    /// &\mathbf{p}\left\{ t \right\} = \mathrm{Adjust\_temperature\_update\_thermostat}() \\
    /// &\mathbf{p}\left\{ t + \frac{\delta t}{2} \right\} = \mathbf{p}\left\{ t \right\} + \frac{1}{2} \delta t \mathbf{f} \\
    /// &\mathbf{r}\left\{ t + \delta t \right\} = \mathbf{r}\left\{ t \right\} + \delta t \frac{\mathbf{p}\left\{ t + \frac{\delta t}{2} \right\}}{m}
    ///         
    /// \end{align}
    /// ```
    /// 
    /// Where $`\mathbf{r}`$ is the position, $`\mathbf{p}`$ is the momentum, $`\mathbf{f}`$ is the force,
    /// $`m`$ is the mass of each [`Body`](hoomd_microstate::Body::properties), and $`t`$ is the time,
    /// $`\delta t`$ is the timestep dt. 
    #[inline]
    fn integrate_translation_half_step_one_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        let (kinetic_energy, degrees_of_freedom) = microstate.translational_kinetic_energy_with_filter(&should_integrate_body);
        let rescaling_factor = self.translational_thermostat.integrate_step_one(
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

    /// Perform the second-half integration on translational degrees-of-freedom
    /// , continuing from the last step in [`integrate_translation_half_step_one`](ConstantVolume::integrate_translation_half_step_one)
    /// and advancing the [`Microstate`] and possibly the [`Thermostat`] state 
    /// forward as:
    /// ```math
    /// \begin{align}
    ///
    /// &\mathbf{p}\left\{ t + \delta t \right\} = \mathbf{p}\left\{ t + \frac{1}{2} \delta t \right\} + \frac{1}{2} \delta t \mathbf{f} \\
    /// &\mathbf{p}\left\{ t + \delta t \right\} = \mathrm{Adjust\_temperature\_update\_thermostat}() \\
    ///         
    /// \end{align}
    /// ```
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
        let rescaling_factor = self.translational_thermostat.integrate_step_two(
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

impl<S, X, C, TT, TR, M> RotationalMotion<DynamicOrientedPoint<Cartesian<3>, Versor>, S, X, C, M> for ConstantVolume<TT, TR>
where
    DynamicOrientedPoint<Cartesian<3>, Versor>: Transform<S>,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<DynamicOrientedPoint<Cartesian<3>, Versor>> + Wrap<S> + GenerateGhosts<S>,
    TR: Thermostat<M>,
{
    /// Perform the first-half integration on rotational degrees-of-freedom
    /// for three-dimensional system, advancing the [`Microstate`] and possibly 
    /// the [`Thermostat`] state forward as follows
    /// 
    /// First, perform temperature adjustment and update [`Thermostat`] state due
    /// to rotational degrees-of-freedom:
    /// ```math
    /// \begin{equation}
    ///
    /// \mathbf{l}\left\{ t \right\} = \mathrm{Adjust\_temperature\_update\_thermostat}()
    ///         
    /// \end{equation}
    /// ```
    /// 
    /// Next, the [`AngularMomentum`] $`\mathbf{l}`$ and [`NetTorque`] $`\boldsymbol{\tau}`$ in 
    /// the body frame are converted into the quaternion form: $`\mathbf{p}^{(4)}`$ and $`\mathbf{f}^{(4)}`$ , 
    /// using [`Orientation`] quaternion $`\mathbf{q}=(q_0, q_1, q_2, q_3)`$. To clarify, starting from this
    /// section, variables being labeled by a superscript $`^{'}`$ represents the old varialbes that
    /// being produce in the previous step. Besides, we express the calculation as matrix-vector 
    /// algebra instead of quaternion algebra for simpilicity:
    /// ```math
    /// \begin{align}
    ///
    /// &\mathbf{p}^{(4)} = 2S(\mathbf{q}) \mathbf{l}^{(4) '},\; \mathbf{l}^{(4) '}=(0, l_x^{'}, l_y^{'}, l_z^{'}) \\
    /// &\mathbf{f}^{(4)} = 2S(\mathbf{q}) \boldsymbol{\tau}^{(4)},\; \boldsymbol{\tau}^{(4)}=(0, \tau_x, \tau_y, \tau_z)\\
    ///         
    /// \end{align}
    /// ```
    /// Where, expressed as real $`4\times4`$ matrices
    /// ```math
    /// \begin{align*}
    ///
    /// S(\mathbf{q}) = 
    ///     \begin{pmatrix}
    ///     q_0 & -q_1 & -q_2 & -q_3\\
    ///     q_1 & q_0 & -q_3 & q_2\\
    ///     q_2 & q_3 & q_0 & -q_1\\
    ///     q_3 & -q_2 & q_1 & q_0
    ///     \end{pmatrix}
    ///         
    /// \end{align*}
    /// ```
    /// Then, we start the NOvel Symplectic QUaternIon ScHeme (NO_SQUISH) algorithm that 
    /// integrate $`( \mathbf{q},  \mathbf{p}^{(4)})`$
    /// forward that is sympletic, phase space volume preserving, and unit orientation 
    /// quaternion preserving, i.e., $`|\mathbf{q}|=1`$:
    /// 
    /// First, we translate the angular momentum foward using torque $`\mathbf{f}^{(4)}`$
    /// ```math
    /// \begin{align}
    ///
    /// \mathbf{p}^{(4)} = \mathbf{p}^{(4) '} + \frac{\delta t}{2} \mathbf{f}^{(4)}
    ///         
    /// \end{align} 
    /// ```
    /// 
    /// Second, we use the properties of quaternion algebra that decompose the 
    /// Liovillian into a sum over permutation matrices applying on $`(\mathbf{q}, \mathbf{p}^{(4)})`$,
    /// resulting in a five-steps updates:
    /// 
    /// ```math
    /// \begin{align}
    /// 
    /// &\phi_3 = \frac{1}{4 I_{33}} \mathrm{dot} \left( \mathbf{p}^{(4) '}, P_3 \mathbf{q}^{'} \right) \\
    /// &\mathbf{q} = \cos{(\phi_3 \delta t / 2)} \mathbf{q}^{'} +  \sin{(\phi_3 \delta t / 2)} P_3 \mathbf{q}^{'} \nonumber \\
    /// &\mathbf{p}^{(4)} = \cos{(\phi_3 \delta t / 2)} \mathbf{p}^{(4) '} +  \sin{(\phi_3 \delta t / 2)} P_3 \mathbf{p}^{(4) '} \nonumber \\ \nonumber \\
    /// 
    /// &\phi_2 = \frac{1}{4 I_{22}} \mathrm{dot} \left( \mathbf{p}^{(4) '}, P_2 \mathbf{q}^{'} \right) \\
    /// &\mathbf{q} = \cos{(\phi_2 \delta t / 2)} \mathbf{q}^{'} +  \sin{(\phi_2 \delta t / 2)} P_2 \mathbf{q}^{'} \nonumber \\
    /// &\mathbf{p}^{(4)} = \cos{(\phi_2 \delta t / 2)} \mathbf{p}^{(4) '} +  \sin{(\phi_2 \delta t / 2)} P_2 \mathbf{p}^{(4) '} \nonumber \\ \nonumber \\
    /// 
    /// &\phi_1 = \frac{1}{4 I_{11}} \mathrm{dot} \left( \mathbf{p}^{(4) '}, P_1 \mathbf{q}^{'} \right) \\
    /// &\mathbf{q} = \cos{(\phi_1 \delta t)} \mathbf{q}^{'} +  \sin{(\phi_1 \delta t)} P_1 \mathbf{q}^{'} \nonumber \\
    /// &\mathbf{p}^{(4)} = \cos{(\phi_1 \delta t)} \mathbf{p}^{(4) '} +  \sin{(\phi_1 \delta t)} P_1 \mathbf{p}^{(4) '} \nonumber  \nonumber \\ \nonumber \\
    ///
    /// &\phi_2 = \frac{1}{4 I_{22}} \mathrm{dot} \left( \mathbf{p}^{(4) '}, P_2 \mathbf{q}^{'} \right) \\
    /// &\mathbf{q} = \cos{(\phi_2 \delta t / 2)} \mathbf{q}^{'} +  \sin{(\phi_2 \delta t / 2)} P_2 \mathbf{q}^{'} \nonumber \\
    /// &\mathbf{p}^{(4)} = \cos{(\phi_2 \delta t / 2)} \mathbf{p}^{(4) '} +  \sin{(\phi_2 \delta t / 2)} P_2 \mathbf{p}^{(4) '} \nonumber  \nonumber \\ \nonumber \\
    ///
    /// &\phi_3 = \frac{1}{4 I_{33}} \mathrm{dot} \left( \mathbf{p}^{(4) '}, P_3 \mathbf{q}^{'} \right) \\
    /// &\mathbf{q} \left\{ t + \delta t \right\} = \cos{(\phi_3 \delta t / 2)} \mathbf{q}^{'} +  \sin{(\phi_3 \delta t / 2)} P_3 \mathbf{q}^{'} \nonumber \\
    /// &\mathbf{p}^{(4)} \left\{ t + \frac{\delta t}{2} \right\} = \cos{(\phi_3 \delta t / 2)} \mathbf{p}^{(4) '} +  \sin{(\phi_3 \delta t / 2)} P_3 \mathbf{p}^{(4) '} \nonumber    \nonumber \\ \nonumber \\
    /// \end{align} 
    /// ```
    /// Where $`I_{kk}`$ are the principal compoenets of moment of inertia, and $`P_k`$ are the permuation matrices, such that $`P_1q=(-q_1, q_0, q_3, -q_2)`$, $`P_2q=(-q_2, -q_3, q_0, q_1)`$, 
    /// $`P_3q=(-q_3, q_2, -q_1, q_0)`$, $`P_0q=(q_0, q_1, q_2, q_3)`$, and $`(PP^T)_{\alpha \beta}=\delta_{\alpha \beta}`$.
    /// 
    /// Finally, the quaternion form of final angular momentum $`\mathbf{p}^{(4)} \left\{ t + \frac{\delta t}{2} \right\}`$ can be converted back to
    /// the vector form as:
    /// 
    /// ```math
    /// \begin{align}
    ///
    /// &\mathbf{l}^{(4)} = \frac{1}{2}S(\mathbf{q}^{'})^T \mathbf{p}^{(4) '},\; \mathbf{l}^{(4)}=(0, l_x, l_y, l_z)
    ///         
    /// \end{align}
    /// ```
    #[inline]
    fn integrate_rotation_half_step_one_with_filter<F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<3>, Versor>, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<DynamicOrientedPoint<Cartesian<3>, Versor>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        let (kinetic_energy, degrees_of_freedom) = microstate.rotational_kinetic_energy_with_filter(&should_integrate_body);
        let rescaling_factor = self.rotational_thermostat.integrate_step_one(
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

    /// Perform the second-half integration on rotational degrees-of-freedom
    /// for three-dimensional system, advancing the [`Microstate`] and possibly 
    /// the [`Thermostat`] state forward as follows
    /// 
    /// Continue from the last step in [`integrate_rotation_half_step_one`](ConstantVolume::integrate_rotation_half_step_one).
    /// Convert [`AngularMomentum`] and [`NetTorque`] into thier quaternion forms and translate the 
    /// angular momentum $`\mathbf{p}^{(4)}`$ forward:
    /// 
    /// ```math
    /// \begin{align}
    ///
    /// \mathbf{p}^{(4)}\left\{ t + \delta t \right\} = \mathbf{p}^{(4)}\left\{ t + \frac{\delta t}{2} \right\} + \frac{\delta t}{2} \mathbf{f}^{(4)}
    ///         
    /// \end{align} 
    /// ```
    /// 
    /// Then, convert angular momentum back to its vector form $`\mathbf{l}`$, perform 
    /// temperature adjustment, and update [`Thermostat`] state due
    /// to rotational degrees-of-freedom:
    /// ```math
    /// \begin{equation}
    ///
    /// \mathbf{l}\left\{ t + \delta t \right\} = \mathrm{Adjust\_temperature\_update\_thermostat}()
    ///         
    /// \end{equation}
    /// ``` 
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
        let rescaling_factor = self.rotational_thermostat.integrate_step_two(
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

/// Integrate rotational degrees of freedom in 2-dimensional Cartesian space.
///
/// [`ConstantVolume`] integration is only defined for macrostates with [`Temperature`].
///
/// The generic type names are:
/// * `B`: The [`Body::properties`](crate::Body) type.
/// * `S`: The [`Site::properties`](crate::Site) type.
/// * `C`: The [`boundary`](crate::boundary) condition type.
/// * `E`: The interaction [`evaluator`]() type.
/// * `T`: The [`Thermostat`]() type.
/// * `M`: The [`macrostate`](crate::macrostate) type.
impl<S, X, C, TT, TR, M> RotationalMotion<DynamicOrientedPoint<Cartesian<2>, Angle>, S, X, C, M> for ConstantVolume<TT, TR>
where
    DynamicOrientedPoint<Cartesian<2>, Angle>: Transform<S>,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<DynamicOrientedPoint<Cartesian<2>, Angle>> + Wrap<S> + GenerateGhosts<S>,
    TR: Thermostat<M>,
{
    /// Perform the first-half integration on rotational 
    /// degrees-of-freedom for two-dimensional system, 
    /// advancing the [`Microstate`] and possibly the 
    /// [`Thermostat`] state forward as:
    /// ```math
    /// \begin{align}
    ///
    /// &p\left\{ t \right\} = \mathrm{Adjust\_temperature\_update\_thermostat}() \\
    /// &p\left\{ t + \frac{\delta t}{2} \right\} = p\left\{ t \right\} + \frac{1}{2} \delta t f \\
    /// &\theta\left\{ t + \delta t \right\} = \theta\left\{ t \right\} + \delta t \frac{p\left\{ t + \frac{\delta t}{2} \right\}}{I}
    ///         
    /// \end{align}
    /// ```
    /// 
    /// Where $`\theta`$ is the orientation, $`p`$ is the angular momentum, $`f`$ is the toruqe, and
    /// $`I`$ is the moment of inertia on each [`Body`](hoomd_microstate::Body::properties), and $`t`$ is the time,
    /// $`\delta t`$ is the timestep dt. Note that in two-dimension, every particle only has 
    /// one degrees-of-freedom contributed from their rotational motion.
    #[inline]
    fn integrate_rotation_half_step_one_with_filter<F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<2>, Angle>, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<DynamicOrientedPoint<Cartesian<2>, Angle>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        let (kinetic_energy, degrees_of_freedom) = microstate.rotational_kinetic_energy_with_filter(&should_integrate_body);
        let rescaling_factor = self.rotational_thermostat.integrate_step_one(
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

            let net_torque = *body_properties.net_torque();
            let moment_of_inertia = *body_properties.moment_of_inertia();

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

    /// Perform the second-half integration on rotational degrees-of-freedon
    /// for two-dimensional system, continuing from the last step in [`integrate_rotation_half_step_one`](ConstantVolume::integrate_rotation_half_step_one)
    /// and advancing the [`Microstate`] and possibly the [`Thermostat`] state 
    /// forward as:
    /// ```math
    /// \begin{align}
    ///
    /// &p\left\{ t + \delta t \right\} = p\left\{ t + \frac{1}{2} \delta t \right\} + \frac{1}{2} \delta t f \\
    /// &p\left\{ t + \delta t \right\} = \mathrm{Adjust\_temperature\_update\_thermostat}() \\
    ///         
    /// \end{align}
    /// ```
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

            let net_torque = *body_properties.net_torque();

            *body_properties.angular_momentum_mut() += net_torque * 0.5 * self.delta_t;

            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.");
        }

        let (kinetic_energy, degrees_of_freedom) = microstate.rotational_kinetic_energy();
        let rescaling_factor = self.rotational_thermostat.integrate_step_two(
            &mut rng,
            macrostate,
            self.delta_t,
            kinetic_energy,
            degrees_of_freedom,
        );

        // Update velocity
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
    use hoomd_interaction::{External, MaximumInteractionRange, external::{ConstantForce, ConstantTorque}, Rigid};
    use hoomd_microstate::{Body, property::{DynamicPoint, DynamicOrientedPoint, Point}};

    use crate::{UpdateNetForce, thermostat::NoThermostat, UpdateNetForceAndTorque};

    use super::*;

    /// A simple 2d dynamics point body
    fn dynamics_body_2d() -> Body<DynamicPoint<Cartesian<2>>, Point<Cartesian<2>>> {
        Body {
            properties: DynamicPoint {
                position: Cartesian::<2>::default(),
                momentum: Cartesian::<2>::default(),
                net_force: Cartesian::<2>::default(),
                mass: 1.0,
            },
            sites: vec![Point::new(Cartesian::<2>::default())],
        }
    }

    /// A simple 3d dynamics point body
    fn dynamics_body_3d(mass: f64) -> Body<DynamicPoint<Cartesian<3>>, Point<Cartesian<3>>> {
        Body {
            properties: DynamicPoint {
                position: Cartesian::<3>::default(),
                momentum: Cartesian::<3>::default(),
                net_force: Cartesian::<3>::default(),
                mass: mass,
            },
            sites: vec![Point::new(Cartesian::from([0.0, 0.0, 0.0]))],
        }
    }

    /// A simple 2d oriented dynamics point body
    fn oriented_dynamics_body_2d(mass: f64, moi: f64) -> Body<DynamicOrientedPoint<Cartesian<2>, Angle>, Point<Cartesian<2>>> {
        Body {
            properties: DynamicOrientedPoint {
                position: Cartesian::<2>::default(),
                orientation: Angle::default(),
                momentum: Cartesian::<2>::default(),
                net_force: Cartesian::<2>::default(),
                moment_of_inertia: moi,
                angular_momentum: 0.0,
                net_torque: 0.0,
                mass: mass,
            },
            sites: vec![Point::new(Cartesian::from([0.0, 0.0]))],
        }
    }

    /// A simple 2d oriented dynamics point body
    fn oriented_dynamics_body_3d(mass: f64, moi: [f64; 3]) -> Body<DynamicOrientedPoint<Cartesian<3>, Versor>, Point<Cartesian<3>>> {
        Body {
            properties: DynamicOrientedPoint {
                position: Cartesian::<3>::default(),
                orientation: Versor::default(),
                momentum: Cartesian::<3>::default(),
                net_force: Cartesian::<3>::default(),
                moment_of_inertia: moi,
                angular_momentum: Cartesian::<3>::default(),
                net_torque: Cartesian::<3>::default(),
                mass: mass,
            },
            sites: vec![Point::new(Cartesian::from([0.0, 0.0, 0.0]))],
        }
    }

    #[test]
    fn test_constant_volume() -> anyhow::Result<()> {
        let dt = 2.0;
        let cv = ConstantVolume::new(dt);
        assert_eq!(cv.delta_t, dt);

        Ok(())
    }

    #[test]
    fn test_translational_integration() -> anyhow::Result<()> {
        // Ensure translational integration of a simple external force in 3D
        // yields the correct position and momentum at the halfstep and the
        // fullstep.
        let mass = 1.0;
        let dt = 0.1;
        let f_mag = 1.0;
        let f_dir = Cartesian::<3>::from(
            [1.0 / 3.0_f64.sqrt(), 1.0 / 3.0_f64.sqrt(), 1.0 / 3.0_f64.sqrt()]
        );

        let mut microstate = Microstate::builder()
            .bodies([dynamics_body_3d(mass)])
            .try_build()?;
        let force = Rigid(External(ConstantForce {
            force: f_dir * f_mag,
            r_0: [0.0, 0.0, 0.0].into(),
        }));
        let mut method = ConstantVolume::new(dt);
        struct Isoenergy {}
        let mut macrostate = Isoenergy {};

        // Update force first so that the particles can move
        microstate.update_net_force(&force);
        
        // Check the first halfstep
        method.integrate_translation_half_step_one(
            &mut microstate,
            &mut macrostate
        );
        let mut expected_momentum = Cartesian::<3>::default()
            + (f_dir * f_mag * dt * 0.5 * -1.0);
        let expected_position = Cartesian::<3>::default()
            + expected_momentum * dt / mass;

        assert_eq!(expected_momentum, microstate.bodies()[0].item.properties.momentum);
        assert_eq!(expected_position, microstate.bodies()[0].item.properties.position);

        // Update force again
        microstate.update_net_force(&force);

        // Check the second halfstep
         method.integrate_translation_half_step_two(
            &mut microstate,
            &mut macrostate
        );
        expected_momentum += f_dir * f_mag * dt * 0.5 * -1.0;
        assert_eq!(expected_momentum, microstate.bodies()[0].item.properties.momentum);
        assert_eq!(expected_position, microstate.bodies()[0].item.properties.position);

        Ok(())
    }

    #[test]
    fn test_rotational_integration_2d() -> anyhow::Result<()> {
        // Ensure rotational integration of a simple external torque in 2D
        // yields the correct orientation and angular momentum at the halfstep
        // and the fullstep
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
        let mut method = ConstantVolume::new(dt);
        struct Isoenergy {}
        let mut macrostate = Isoenergy {};

        // Update torque first so that the particles can move
        microstate.update_net_force_and_torque(&torque);
        
        // Check the first halfstep
        method.integrate_rotation_half_step_one(
            &mut microstate,
            &mut macrostate
        );
        let mut expected_angular_momentum = t_dir * t_mag * 0.5 * dt;
        let expected_orientation = Angle::default().theta
            + expected_angular_momentum / moi * dt;

        assert_eq!(expected_angular_momentum, microstate.bodies()[0].item.properties.angular_momentum);
        assert_eq!(expected_orientation, microstate.bodies()[0].item.properties.orientation.theta);

        // Update torque again
        microstate.update_net_force_and_torque(&torque);

        // Check the second halfstep
         method.integrate_rotation_half_step_two(
            &mut microstate,
            &mut macrostate
        );
        expected_angular_momentum += t_dir * t_mag * 0.5 * dt;
        assert_eq!(expected_angular_momentum, microstate.bodies()[0].item.properties.angular_momentum);
        assert_eq!(expected_orientation, microstate.bodies()[0].item.properties.orientation.theta);

        Ok(())
    }

    // TODO: uncomment and fix tests
    // #[test]
    // fn test_rotational_integration_3d() -> anyhow::Result<()> {
    //     // Ensure rotational integration of a simple external torque in 3D
    //     // yields the correct orientation and angular momentum at the halfstep
    //     // and the fullstep
    //     let mass = 1.0;
    //     let moi = [1.0, 1.0, 1.0];
    //     let dt = 0.1;
    //     let t_mag = 1.0;
    //     let t_dir = Cartesian::<3>::from([0.0, 0.0, 1.0]);

    //     let mut microstate = Microstate::builder()
    //         .bodies([oriented_dynamics_body_3d(mass, moi)])
    //         .try_build()?;

    //     let torque = Rigid(External(ConstantTorque::<Cartesian<3>> {    // TODO: why does this not permit a Unit vector?
    //         torque: t_dir * t_mag,
    //     }));
    //     let mut method = ConstantVolume::new(dt);
    //     struct Isoenergy {}
    //     let mut macrostate = Isoenergy {};
    //     let mut thermostat = NoThermostat;  // TODO: use an actual thermostat

    //     // Update torque first so that the particles can move
    //     method.update_force_and_torque(&mut microstate, &torque);
        
    //     // Check the first halfstep
    //     method.integrate_rotation_half_step_one(
    //         &mut microstate,
    //         &mut thermostat,
    //         &mut macrostate
    //     );

    //     // TODO: return here
    //     // Calculate expected angular momentum

    //     // // Calculate expected orientation
    //     // let mut expected_angular_momentum = t_dir * t_mag * 0.5 * dt;
    //     // let expected_orientation = Angle::default().theta
    //     //     + expected_angular_momentum / moi * dt;

    //     // assert_eq!(expected_angular_momentum, microstate.bodies()[0].item.properties.angular_momentum);
    //     // assert_eq!(expected_orientation, microstate.bodies()[0].item.properties.orientation.theta);

    //     // // Update torque again
    //     // method.update_force_and_torque(&mut microstate, &torque);

    //     // // Check the second halfstep
    //     //  method.integrate_rotation_half_step_two(
    //     //     &mut microstate,
    //     //     &mut thermostat,
    //     //     &mut macrostate
    //     // );
    //     // expected_angular_momentum += t_dir * t_mag * 0.5 * dt;
    //     // assert_eq!(expected_angular_momentum, microstate.bodies()[0].item.properties.angular_momentum);
    //     // assert_eq!(expected_orientation, microstate.bodies()[0].item.properties.orientation.theta);

    //     Ok(())
    // }

    #[test]
    fn test_force_update_2d() -> anyhow::Result<()> {
        let mut microstate = Microstate::builder()
            .bodies([dynamics_body_2d()])
            .try_build()?;
        let evaluator = Rigid(External(ConstantForce {
            r_0: Cartesian::from([0.0, 1.0]),
            force: Cartesian::from([0.0, 1.0]),
        }));
        
        microstate.update_net_force(&evaluator);        
        assert_eq!(microstate.bodies()[0].item.properties.net_force, Cartesian::<2>::from([0.0, -1.0]));

        Ok(())
    }

    #[test]
    fn test_force_update_3d() -> anyhow::Result<()> {
        let mut microstate = Microstate::builder()
            .bodies([dynamics_body_3d(1.0)])
            .try_build()?;
        let evaluator = Rigid(External(ConstantForce {
            r_0: Cartesian::from([0.0, 1.0, 0.0]),
            force: Cartesian::from([0.0, 1.0, 0.0]),
        }));
        
        microstate.update_net_force(&evaluator);        
        assert_eq!(microstate.bodies()[0].item.properties.net_force, Cartesian::<3>::from([0.0, -1.0, 0.0]));

        Ok(())
    }

    #[test]
    fn test_torque_update_2d() -> anyhow::Result<()> {
        let mut microstate = Microstate::builder()
            .bodies([oriented_dynamics_body_2d(1.0, 1.0)])
            .try_build()?;

        let evaluator = Rigid(External(ConstantTorque {
            torque: 1.0,
        }));
        
        microstate.update_net_force_and_torque(&evaluator);        
        assert_eq!(microstate.bodies()[0].item.properties.net_torque, 1.0);

        Ok(())
    }

    #[test]
    fn test_torque_update_3d() -> anyhow::Result<()> {
        let mut microstate = Microstate::builder()
            .bodies([oriented_dynamics_body_3d(1.0, [1.0, 1.0, 1.0])])
            .try_build()?;

        let evaluator = Rigid(External(ConstantTorque {
            torque: Cartesian::<3>::from([0.0, 0.0, 1.0])
        }));
        
        microstate.update_net_force_and_torque(&evaluator);        
        assert_eq!(microstate.bodies()[0].item.properties.net_torque, Cartesian::<3>::from([0.0, 0.0, 1.0]));

        Ok(())
    }
    
    // TODO: return here, and start by creating derive macros for NetForce, NetTorque, Momentum, AngularMomentum, etc.
    // #[test]
    // fn test_force_and_torque_update_2d() -> anyhow::Result<()> {
    //     let mut microstate = Microstate::builder()
    //         .bodies([oriented_dynamics_body_2d(1.0, 1.0)])
    //         .try_build()?;

    //     let torque_evaluator = Rigid(External(ConstantTorque {
    //         torque: 1.0
    //     }));
    //     let force_evaluator = Rigid(External(ConstantForce {
    //         force: [0.0, 1.0].into(),
    //         r_0: [0.0, 1.0].into(),
    //     }));
    //     let evaluator = (torque_evaluator, force_evaluator);
    //     let method = ConstantVolume::new(0.1);
        
    //     method.update_force_and_torque(&mut microstate, &evaluator);
    //     assert_eq!(microstate.bodies()[0].item.properties.net_force, Cartesian::<2>::from([0.0, 1.0]));
    //     assert_eq!(microstate.bodies()[0].item.properties.net_torque, 1.0);
        
    //     Ok(())
    // }

    // #[test]
    // fn test_force_and_torque_update_3d() {
    //     // #[derive(MaximumInteractionRange)]
    //     struct OverallInteraction {
    //         force: ConstantForce<Cartesian<3>>,
    //         torque: ConstantTorque<Cartesian<3>>
    //     }

    //     let interaction = OverallInteraction {
    //         force: ConstantForce {
    //             alpha: f_mag,
    //             plane_origin: [0.0, 0.0, 0.0].into(),
    //             plane_normal: f_dir.to_unit()?.0,
    //         },
    //         torque: ConstantTorque { alpha: 1.0, direction: 1.0 }
    //     };
    // }
}

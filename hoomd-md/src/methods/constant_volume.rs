// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![allow(non_snake_case)]

use hoomd_interaction::{NetBodyForce, NetBodyForceAndTorque};
use hoomd_microstate::{
    Microstate, SiteKey, Transform, boundary::{GenerateGhosts, Wrap}, property::{
        AngularMomentum, Mass, MomentOfInertia, Momentum, NetForce, NetTorque, Orientation,
        Position,
    }
};
use hoomd_vector::{
    Angle, Cartesian, InnerProduct, Quaternion, Rotate, Rotation, Vector, Versor
};
use crate::{thermostat::Thermostat, methods::{TranslationalMotion, RotationalMotion, ForceUpdate, ForceAndTorqueUpdate}};
use hoomd_spatial::PointUpdate;

/// Perform time integration on the [`Microstate`] with the volume constraining
/// to a constant using Velocity Verlet algorithm.
/// 
/// When [`NoThermostat`](crate::thermostat::NoThermostat) is provided to the methods:
/// [`integrate_translation_step_one`](ConstantVolume::integrate_translation_step_one), 
/// [`integrate_translation_step_two`](ConstantVolume::integrate_translation_step_two), 
/// [`integrate_rotation_step_one`](ConstantVolume::integrate_rotation_step_one), and 
/// [`integrate_rotation_step_two`](ConstantVolume::integrate_rotation_step_two), it 
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
/// use hoomd_md::{
///     methods::{ConstantVolume, ForceAndTorqueUpdate, RotationalMotion, TranslationalMotion},
///     thermostat::NoThermostat
/// };
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
#[doc(alias("nve", "nvt"))]
#[derive(Clone, Debug, PartialEq)]
pub struct ConstantVolume {
    /// The size of a timestep.
    dt: f64,

    /// The instantaneous kinetic energy of translational degrees of freedom.
    translational_kinetic_energy: f64,

    /// The instantaneous kinetic energy of rotatioanl degrees of freedom.
    rotational_kinetic_energy: f64,

    /// The number of translational degrees of freedom.
    translational_dof: f64,

    /// The number of rotational degrees of freedom.
    rotational_dof: f64,
}

impl ConstantVolume {
    /// Construct a new [`ConstantVolume`] given timestep dt.
    #[inline]
    pub fn new(dt: f64) -> Self {
        Self {
            dt,
            translational_kinetic_energy: 0.0,
            translational_dof: 0.0,
            rotational_kinetic_energy: 0.0,
            rotational_dof: 0.0,
        }
    }

    /// Access the current translational kinetic energy.
    #[inline]
    pub fn get_translational_kinetic_energy(&self) -> &f64 {
        &self.translational_kinetic_energy
    }

    /// Access the current translational degrees of freedom.
    #[inline]
    pub fn get_translational_dof(&self) -> &f64 {
        &self.translational_dof
    }

    /// Access the current rotatioanl energy.
    #[inline]
    pub fn get_rotational_kinetic_energy(&self) -> &f64 {
        &self.rotational_kinetic_energy
    }

    /// Access the current kinetic degrees of freedom.
    #[inline]
    pub fn get_rotational_dof(&self) -> &f64 {
        &self.rotational_dof
    }
}

impl<V, B, S, X, C, T, M> TranslationalMotion<B, S, X, C, T, M> for ConstantVolume
where
    V: Default + Vector + InnerProduct,
    B: Position<Position = V>
        + Momentum<Vector = V>
        + NetForce<Vector = V>
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = V> + Default,
    X: PointUpdate<V, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    T: Thermostat<B, S, X, C, M>,
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
    ///
    /// TODO: Do we want to allow users to set a displacement limit?
    #[inline]
    fn integrate_translation_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        thermostat: &mut T,
        macrostate: &M,
    ) {
        // Closure for calculating (ke, dof) in the Thermostat
        let mut compute_properties = |microstate: &Microstate<B, S, X, C>| -> (f64, f64) {
            let integrator_ke = &mut self.translational_kinetic_energy;
            let integrator_dof = &mut self.translational_dof;
            let mut ke = 0.0;
            // use the first body to determine the dimension
            let nd = microstate.bodies()[0]
                .item
                .properties
                .position()
                .n_dimensions() as f64;
            let dof = nd * (microstate.bodies().len() as f64 - 1.0);

            for body_index in 0..microstate.bodies().len() {
                // Get the the body information
                let body_properties = microstate.bodies()[body_index].item.properties.clone();

                // calculate m * v^2 part
                let momentum = body_properties.momentum();
                ke += momentum.norm_squared() / body_properties.mass();
            }
            ke *= 0.5;
            *integrator_ke = ke.clone();
            *integrator_dof = dof.clone();

            (ke, dof)
        };

        // Advance thermostat and get rescaling factor
        let rescaling_factor = thermostat.integrate_step_one(
            microstate,
            macrostate,
            &self.dt,
            &mut compute_properties,
        );
        self.translational_kinetic_energy *= rescaling_factor.powi(2);

        // Integrate position and momentum forward
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Perform the integration step
            // TODO: should we use the momentum methods here?
            let net_force = body_properties.net_force().clone();
            let mass = body_properties.mass().clone();
            let mut momentum = body_properties.momentum().clone();

            // Apply thermostat
            momentum *= rescaling_factor;
            momentum += net_force * 0.5 * self.dt;
            *body_properties.position_mut() += momentum / mass * self.dt;

            *body_properties.momentum_mut() = momentum;
            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        microstate.increment_substep();
    }

    /// Perform the second-half integration on translational degrees-of-freedom
    /// , continuing from the last step in [`integrate_translation_step_one`](ConstantVolume::integrate_translation_step_one)
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
    fn integrate_translation_step_two(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        thermostat: &mut T,
        macrostate: &M,
    ) {
        // Closure for calculating (ke, dof) in the Thermostat
        let mut compute_properties = |microstate: &Microstate<B, S, X, C>| -> (f64, f64) {
            let integrator_ke = &mut self.translational_kinetic_energy;
            let integrator_dof = &mut self.translational_dof;
            let mut ke = 0.0;
            // use the first body to determine the dimension
            let nd = microstate.bodies()[0]
                .item
                .properties
                .position()
                .n_dimensions() as f64;
            let dof = nd * (microstate.bodies().len() as f64 - 1.0);

            for body_index in 0..microstate.bodies().len() {
                // Get the the body information
                let body_properties = microstate.bodies()[body_index].item.properties.clone();

                // calculate m * v^2 part
                let momentum = body_properties.momentum();
                ke += momentum.norm_squared() / body_properties.mass();
            }
            ke *= 0.5;
            *integrator_ke = ke.clone();
            *integrator_dof = dof.clone();

            (ke, dof)
        };

        // Integrate momentum forward
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Get net force on body
            let net_force = body_properties.net_force().clone();

            // Perform the integration step
            *body_properties.momentum_mut() += net_force * self.dt * 0.5;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        // Advance thermostat and get rescaling factor
        let rescaling_factor = thermostat.integrate_step_two(
            microstate,
            macrostate,
            &self.dt,
            &mut compute_properties,
        );
        self.translational_kinetic_energy *= rescaling_factor.powi(2);

        // Apply thermostat
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Rescale velocity
            *body_properties.momentum_mut() *= rescaling_factor;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        microstate.increment_substep();
    }
}

impl<B, S, X, C, T, M> RotationalMotion<3, B, S, X, C, T, M> for ConstantVolume
where
    B: Orientation<Rotation = Versor>
        + AngularMomentum<AngularMomentum = Cartesian<3>>
        + NetTorque<NetTorque = Cartesian<3>>
        + MomentOfInertia<Vector = Cartesian<3>>
        + Transform<S>
        + Position<Position = Cartesian<3>> // TODO: should this be required?
        + Clone,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    T: Thermostat<B, S, X, C, M>,
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
    #[allow(clippy::too_many_lines)]
    fn integrate_rotation_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        thermostat: &mut T,
        macrostate: &M,
    ) {
        // Closure for calculating (ke, dof) in the Thermostat
        let mut compute_properties = |microstate: &Microstate<B, S, X, C>| -> (f64, f64) {
            let integrator_ke = &mut self.rotational_kinetic_energy;
            let integrator_dof = &mut self.rotational_dof;
            let mut ke = 0.0;
            let mut dof = 0.0;
            for body_index in 0..microstate.bodies().len() {
                // Get the important information from the body
                let body_properties = microstate.bodies()[body_index].item.properties.clone();

                // Shorthand variables
                // s is the vector representation of angular momentum
                // I is the 3-vector diagonal values of the moment of inertia
                let s = body_properties.angular_momentum();
                let I = body_properties.moment_of_inertia();

                // Ignore the ke which have zero contribution to inertia
                let x_nonzero = I[0] > 0.0;
                let y_nonzero = I[1] > 0.0;
                let z_nonzero = I[2] > 0.0;

                // angular momentum vector in global frame, s.scalar should be zero.
                // let s = (q.conjugate() * *p) * 0.5;
                if x_nonzero {
                    ke += s[0].powi(2) / I[0];
                    dof += 1.0
                };
                if y_nonzero {
                    ke += s[1].powi(2) / I[1];
                    dof += 1.0
                };
                if z_nonzero {
                    ke += s[2].powi(2) / I[2];
                    dof += 1.0
                };
            }
            ke *= 0.5;
            *integrator_ke = ke.clone();
            *integrator_dof = dof.clone();

            (ke, dof)
        };
        // Advance thermostat and get rescaling factor
        let rescaling_factor = thermostat.integrate_step_one(
            microstate,
            macrostate,
            &self.dt,
            &mut compute_properties,
        );
        self.rotational_kinetic_energy *= rescaling_factor.powi(2);

        // Integrate orientation and angular momentum forward
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Shorthand variables
            // q is the versor (unit quaternion) representation of orientation
            // s is the vector representation of angular momentum
            // t is the 3-vector if torque at t, calculated at previous integrate_rotation_step_two
            //   or initialized at t=0
            // I is the 3-vector diagonal values of the moment of inertia
            let mut q = *body_properties.orientation_mut();
            let mut s = *body_properties.angular_momentum_mut();
            let t = *body_properties.net_torque();
            let I = *body_properties.moment_of_inertia();

            // Rotate torque into body frame based on principal axes
            // TODO: check that this is correct
            let mut t_inframe = q.inverted().rotate(&t);

            let mut q_quaternion = *q.get();
            // convert angular momentum from a vector to qauternion.
            let mut p = (q_quaternion
                * Quaternion {
                    scalar: 0.0,
                    vector: s.coordinates.into(),
                })
                * 2.0;

            // Ignore torque along axes which have zero contribution to inertia
            let x_zero = I[0] == 0.0;
            let y_zero = I[1] == 0.0;
            let z_zero = I[2] == 0.0;

            if x_zero {
                t_inframe[0] = 0.0
            };
            if y_zero {
                t_inframe[1] = 0.0
            };
            if z_zero {
                t_inframe[2] = 0.0
            };

            // Apply thermostat
            p = p * rescaling_factor;
            // Advance p and q by half a timestep following Trotter
            // factorization of Liouvillian rotation
            p += q_quaternion
                * Quaternion {
                    scalar: 0.0,
                    vector: t_inframe.coordinates.into(),
                }
                * self.dt;

            // TODO: what do we call these steps?
            if !z_zero {
                let p3 = Quaternion::from([-p.vector[2], p.vector[1], -p.vector[0], p.scalar]);
                let q3 = Quaternion::from([
                    -q_quaternion.vector[2],
                    q_quaternion.vector[1],
                    -q_quaternion.vector[0],
                    q_quaternion.scalar,
                ]);
                let phi3 = (1. / (4. * I[2])) * ((p.scalar * q3.scalar) + p.vector.dot(&q3.vector));
                let cphi3 = (0.5 * self.dt * phi3).cos();
                let sphi3 = (0.5 * self.dt * phi3).sin();

                p = p * cphi3 + p3 * sphi3;
                q_quaternion = q_quaternion * cphi3 + q3 * sphi3;
            }

            if !y_zero {
                let p2 = Quaternion::from([-p.vector[1], -p.vector[2], p.scalar, p.vector[0]]);
                let q2 = Quaternion::from([
                    -q_quaternion.vector[1],
                    -q_quaternion.vector[2],
                    q_quaternion.scalar,
                    q_quaternion.vector[0],
                ]);
                let phi2 = (1. / (4. * I[1])) * ((p.scalar * q2.scalar) + p.vector.dot(&q2.vector));
                let cphi2 = (0.5 * self.dt * phi2).cos();
                let sphi2 = (0.5 * self.dt * phi2).sin();

                p = p * cphi2 + p2 * sphi2;
                q_quaternion = q_quaternion * cphi2 + q2 * sphi2;
            }

            if !x_zero {
                let p1 = Quaternion::from([-p.vector[0], p.scalar, p.vector[2], -p.vector[1]]);
                let q1 = Quaternion::from([
                    -q_quaternion.vector[0],
                    q_quaternion.scalar,
                    q_quaternion.vector[2],
                    -q_quaternion.vector[1],
                ]);
                let phi1 = (1. / (4. * I[0])) * ((p.scalar * q1.scalar) + p.vector.dot(&q1.vector));
                let cphi1 = (self.dt * phi1).cos();
                let sphi1 = (self.dt * phi1).sin();

                p = p * cphi1 + p1 * sphi1;
                q_quaternion = q_quaternion * cphi1 + q1 * sphi1;
            }

            if !y_zero {
                let p2 = Quaternion::from([-p.vector[1], -p.vector[2], p.scalar, p.vector[0]]);
                let q2 = Quaternion::from([
                    -q_quaternion.vector[1],
                    -q_quaternion.vector[2],
                    q_quaternion.scalar,
                    q_quaternion.vector[0],
                ]);
                let phi2 = (1. / (4. * I[1])) * ((p.scalar * q2.scalar) + p.vector.dot(&q2.vector));
                let cphi2 = (0.5 * self.dt * phi2).cos();
                let sphi2 = (0.5 * self.dt * phi2).sin();

                p = p * cphi2 + p2 * sphi2;
                q_quaternion = q_quaternion * cphi2 + q2 * sphi2;
            }

            if !z_zero {
                let p3 = Quaternion::from([-p.vector[2], p.vector[1], -p.vector[0], p.scalar]);
                let q3 = Quaternion::from([
                    -q_quaternion.vector[2],
                    q_quaternion.vector[1],
                    -q_quaternion.vector[0],
                    q_quaternion.scalar,
                ]);
                let phi3 = (1. / (4. * I[2])) * ((p.scalar * q3.scalar) + p.vector.dot(&q3.vector));
                let cphi3 = (0.5 * self.dt * phi3).cos();
                let sphi3 = (0.5 * self.dt * phi3).sin();

                p = p * cphi3 + p3 * sphi3;
                q_quaternion = q_quaternion * cphi3 + q3 * sphi3;
            }

            // Renormalize for improved stability
            q = q_quaternion.to_versor().unwrap();

            // Update the particle data
            *body_properties.orientation_mut() = q;

            // convert angular momentum from a quaternion to vector.
            // ((q.conjugate() * p) * 0.5).scalar should be 0.
            s = ((q_quaternion.conjugate() * p) * 0.5).vector;
            *body_properties.angular_momentum_mut() = s;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        microstate.increment_substep();
    }

    /// Perform the second-half integration on rotational degrees-of-freedom
    /// for three-dimensional system, advancing the [`Microstate`] and possibly 
    /// the [`Thermostat`] state forward as follows
    /// 
    /// Continue from the last step in [`integrate_rotation_step_one`](ConstantVolume::integrate_rotation_step_one).
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
    fn integrate_rotation_step_two(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        thermostat: &mut T,
        macrostate: &M,
    ) {
        // Closure for calculating (ke, dof) in the Thermostat
        let mut compute_properties = |microstate: &Microstate<B, S, X, C>| -> (f64, f64) {
            let integrator_ke = &mut self.rotational_kinetic_energy;
            let integrator_dof = &mut self.rotational_dof;
            let mut ke = 0.0;
            let mut dof = 0.0;
            for body_index in 0..microstate.bodies().len() {
                // Get the important information from the body
                let body_properties = microstate.bodies()[body_index].item.properties.clone();

                // Shorthand variables
                // s is the vector representation of angular momentum
                // I is the 3-vector diagonal values of the moment of inertia
                let s = body_properties.angular_momentum();
                let I = body_properties.moment_of_inertia();

                // Ignore the ke which have zero contribution to inertia
                let x_nonzero = I[0] > 0.0;
                let y_nonzero = I[1] > 0.0;
                let z_nonzero = I[2] > 0.0;

                if x_nonzero {
                    ke += s[0].powi(2) / I[0];
                    dof += 1.0
                };
                if y_nonzero {
                    ke += s[1].powi(2) / I[1];
                    dof += 1.0
                };
                if z_nonzero {
                    ke += s[2].powi(2) / I[2];
                    dof += 1.0
                };
            }
            ke *= 0.5;
            *integrator_ke = ke.clone();
            *integrator_dof = dof.clone();

            (ke, dof)
        };

        // Integrate angular momentum forward
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Shorthand variables
            // q is the versor (unit quaternion) representation of orientation
            // s is the vector representation of angular momentum
            // I is the diagonal values of the moment of inertia
            let q = *body_properties.orientation_mut();
            let t = *body_properties.net_torque();
            let mut s = *body_properties.angular_momentum_mut();
            let I = *body_properties.moment_of_inertia();

            // Rotate torque into body frame based on principal axes
            // TODO: check that this is correct
            let mut t_inframe = q.inverted().rotate(&t);

            // convert orientation from versor to quaternion
            let q_quaternion = *q.get();
            // convert angular momentum from a vector to qauternion.
            let mut p = (q_quaternion
                * Quaternion {
                    scalar: 0.0,
                    vector: s.coordinates.into(),
                })
                * 2.0;

            // Ignore torque along axes which have zero contribution to inertia
            let x_zero = I[0] == 0.0;
            let y_zero = I[1] == 0.0;
            let z_zero = I[2] == 0.0;

            if x_zero {
                t_inframe[0] = 0.0
            };
            if y_zero {
                t_inframe[1] = 0.0
            };
            if z_zero {
                t_inframe[2] = 0.0
            };

            // Advance p by half a timestep following Trotter
            // factorization of Liouvillian rotation
            p += q_quaternion
                * Quaternion {
                    scalar: 0.0,
                    vector: t_inframe.coordinates.into(),
                }
                * self.dt;

            // convert angular momentum from a quaternion to vector.
            // ((q.conjugate() * p) * 0.5).scalar should be 0.
            s = ((q_quaternion.conjugate() * p) * 0.5).vector;
            *body_properties.angular_momentum_mut() = s;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        // Advance thermostat and get rescaling factor
        let rescaling_factor = thermostat.integrate_step_two(
            microstate,
            macrostate,
            &self.dt,
            &mut compute_properties,
        );
        self.rotational_kinetic_energy *= rescaling_factor.powi(2);

        // Apply thermostat
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let mut s = *body_properties.angular_momentum_mut();

            // Apply thermostat
            s = s * rescaling_factor;

            // Update the angular momentum in particle data
            *body_properties.angular_momentum_mut() = s;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
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
impl<B, S, X, C, T, M> RotationalMotion<2, B, S, X, C, T, M> for ConstantVolume
where
    B: Orientation<Rotation = Angle>
        + AngularMomentum<AngularMomentum = f64>
        + NetTorque<NetTorque = f64>
        + MomentOfInertia<Vector = f64>
        + Transform<S>
        + Position<Position = Cartesian<2>> // TODO: should this be required?
        + Clone,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    T: Thermostat<B, S, X, C, M>,
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
    fn integrate_rotation_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        thermostat: &mut T,
        macrostate: &M,
    ) {
        // Closure for calculating (ke, dof) in the Thermostat
        let mut compute_properties = |microstate: &Microstate<B, S, X, C>| -> (f64, f64) {
            let integrator_ke = &mut self.rotational_kinetic_energy;
            let integrator_dof = &mut self.rotational_dof;
            let mut ke = 0.0;
            let mut dof = 0.0;
            for body_index in 0..microstate.bodies().len() {
                // Get the important information from the body
                let body_properties = microstate.bodies()[body_index].item.properties.clone();

                // Shorthand variables
                // p is the z-component of angular momentum
                // I is the z-compoenet of the moment of inertia
                let p = body_properties.angular_momentum();
                let I = body_properties.moment_of_inertia();

                // Ignore the ke which have zero contribution to inertia
                let z_nonzero = *I > 0.0;

                if z_nonzero {
                    ke += p.powi(2) / I;
                    dof += 1.0
                };
            }
            ke *= 0.5;
            *integrator_ke = ke.clone();
            *integrator_dof = dof.clone();

            (ke, dof)
        };

        // Advance thermostat and get rescaling factor
        let rescaling_factor = thermostat.integrate_step_one(
            microstate,
            macrostate,
            &self.dt,
            &mut compute_properties,
        );
        self.rotational_kinetic_energy *= rescaling_factor.powi(2);

        // Integrate orientation and angular momentum forward
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Shorthand variables
            // t is the z-compoenet of net torque
            // I is the z-compoenet of the moment of inertia
            let t = *body_properties.net_torque();
            let I = *body_properties.moment_of_inertia();

            // Apply thermostat
            // Advance p by half a timestep and q by a full timestep following Trotter
            // factorization of Liouvillian rotation
            *body_properties.angular_momentum_mut() *= rescaling_factor;
            *body_properties.angular_momentum_mut() += t * 0.5 * self.dt;
            body_properties.orientation_mut().theta +=
                *body_properties.angular_momentum() / I * self.dt;

            // wrap angle back into [0, 2pi] to improve stability
            *body_properties.orientation_mut() = body_properties.orientation_mut().to_reduced();

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        microstate.increment_substep();
    }

    /// Perform the second-half integration on rotational degrees-of-freedon
    /// for two-dimensional system, continuing from the last step in [`integrate_rotation_step_one`](ConstantVolume::integrate_rotation_step_one)
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
    fn integrate_rotation_step_two(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        thermostat: &mut T,
        macrostate: &M,
    ) {
        // Closure for calculating (ke, dof) in the Thermostat
        let mut compute_properties = |microstate: &Microstate<B, S, X, C>| -> (f64, f64) {
            let integrator_ke = &mut self.rotational_kinetic_energy;
            let integrator_dof = &mut self.rotational_dof;
            let mut ke = 0.0;
            let mut dof = 0.0;
            for body_index in 0..microstate.bodies().len() {
                // Get the important information from the body
                let body_properties = microstate.bodies()[body_index].item.properties.clone();

                // Shorthand variables
                // p is the z-component of angular momentum
                // I is the z-compoenet of the moment of inertia
                let p = body_properties.angular_momentum();
                let I = body_properties.moment_of_inertia();

                // Ignore the ke which have zero contribution to inertia
                let z_nonzero = *I > 0.0;

                if z_nonzero {
                    ke += p.powi(2) / I;
                    dof += 1.0
                };
            }
            ke *= 0.5;
            *integrator_ke = ke.clone();
            *integrator_dof = dof.clone();

            (ke, dof)
        };

        // Integrate angular momentum forward
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Get net torque on body.
            let t = *body_properties.net_torque();

            // Advance p by half a timestep following Trotter
            // factorization of Liouvillian rotation
            *body_properties.angular_momentum_mut() += t * 0.5 * self.dt;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        // Advance thermostat and get rescaling factor
        let rescaling_factor = thermostat.integrate_step_two(
            microstate,
            macrostate,
            &self.dt,
            &mut compute_properties,
        );
        self.rotational_kinetic_energy *= rescaling_factor.powi(2);

        // Update velocity
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Apply thermostat
            *body_properties.angular_momentum_mut() *= rescaling_factor;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        microstate.increment_substep();
    }
}

impl<V, B, S, X, C, E> ForceUpdate<B, S, X, C, E> for ConstantVolume
where
    V: Default + Vector + InnerProduct,
    B: Position<Position = V> + NetForce<Vector = V> + Transform<S> + Clone,
    S: Position<Position = V> + Default,
    X: PointUpdate<V, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    E: NetBodyForce<V, B, S, X, C>,
{
    /// Perform the [`NetForce`] update in [`Microstate`] using its 
    /// [`Body`](hoomd_microstate::Body::properties) and [`rigid`](hoomd_interaction::rigid).
    /// 
    /// # Note
    /// This method should be called in between, 
    /// [`integrate_translation_step_one`](ConstantVolume::integrate_translation_step_one), 
    /// [`integrate_rotation_step_one`](ConstantVolume::integrate_rotation_step_one) and 
    /// [`integrate_translation_step_two`](ConstantVolume::integrate_translation_step_two), 
    /// [`integrate_rotation_step_two`](ConstantVolume::integrate_rotation_step_two), to enable
    /// correct time integration. 
    #[inline]
    fn update_force(&self, microstate: &mut Microstate<B, S, X, C>, evaluator: &E) {
        for body_index in 0..microstate.bodies().len() {
            // Get a copy of the body properties to modify
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Calculate the net force and update the properties copy
            let net_force_new = evaluator.net_body_force(microstate, body_index);
            *body_properties.net_force_mut() = net_force_new;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}

// Impl for both translation and rotation
impl<B, S, X, C, E> ForceAndTorqueUpdate<2, B, S, X, C, E> for ConstantVolume
where
    B: Orientation<Rotation = Angle>
        + NetForce<Vector = Cartesian<2>>
        + NetTorque<NetTorque = f64>
        + Transform<S>
        + Position<Position = Cartesian<2>> // TODO: should this be required?
        + Clone,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    E: NetBodyForceAndTorque<2, Cartesian<2>, B, S, X, C>,
{
    /// Perform the [`NetForce`] and [`NetTorque`] update in [`Microstate`] using its 
    /// [`Body`](hoomd_microstate::Body::properties) and [`rigid`](hoomd_interaction::rigid).
    /// 
    /// # Note
    /// This method should be called in between, 
    /// [`integrate_translation_step_one`](ConstantVolume::integrate_translation_step_one), 
    /// [`integrate_rotation_step_one`](ConstantVolume::integrate_rotation_step_one) and 
    /// [`integrate_translation_step_two`](ConstantVolume::integrate_translation_step_two), 
    /// [`integrate_rotation_step_two`](ConstantVolume::integrate_rotation_step_two), to enable
    /// correct time integration. 
    #[inline]
    fn update_force_and_torque(&self, microstate: &mut Microstate<B, S, X, C>, evaluator: &E) {
        for body_index in 0..microstate.bodies().len() {
            // Get a copy of the body properties to modify
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Calculate the net force and update the properties copy
            let (net_force_new, net_torque_new) =
                evaluator.net_body_force_and_torque(microstate, body_index);
            *body_properties.net_force_mut() = net_force_new;
            *body_properties.net_torque_mut() = net_torque_new;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}

// Impl for both translation and rotation
impl<B, S, X, C, E> ForceAndTorqueUpdate<3, B, S, X, C, E> for ConstantVolume
where
    B: Orientation<Rotation = Versor>
        + NetForce<Vector = Cartesian<3>>
        + NetTorque<NetTorque = Cartesian<3>>
        + Transform<S>
        + Position<Position = Cartesian<3>> // TODO: should this be required?
        + Clone,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    E: NetBodyForceAndTorque<3, Cartesian<3>, B, S, X, C>,
{
    /// Perform the [`NetForce`] and [`NetTorque`] update in [`Microstate`] using its 
    /// [`Body`](hoomd_microstate::Body::properties) and [`rigid`](hoomd_interaction::rigid).
    /// 
    /// # Note
    /// This method should be called in between, 
    /// [`integrate_translation_step_one`](ConstantVolume::integrate_translation_step_one), 
    /// [`integrate_rotation_step_one`](ConstantVolume::integrate_rotation_step_one) and 
    /// [`integrate_translation_step_two`](ConstantVolume::integrate_translation_step_two), 
    /// [`integrate_rotation_step_two`](ConstantVolume::integrate_rotation_step_two), to enable
    /// correct time integration. 
    #[inline]
    fn update_force_and_torque(&self, microstate: &mut Microstate<B, S, X, C>, evaluator: &E) {
        for body_index in 0..microstate.bodies().len() {
            // Get a copy of the body properties to modify
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Calculate the net force and update the properties copy
            let (net_force_new, net_torque_new) =
                evaluator.net_body_force_and_torque(microstate, body_index);
            *body_properties.net_force_mut() = net_force_new;
            *body_properties.net_torque_mut() = net_torque_new;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}

#[cfg(test)]
mod tests {
    use hoomd_interaction::{External, MaximumInteractionRange, external::{ConstantForce, ConstantTorque}, rigid::Rigid};
    use hoomd_microstate::{Body, property::{DynamicsPoint, OrientedDynamicsPoint, Point}};

    use crate::thermostat::NoThermostat;

    use super::*;

    /// A simple 2d dynamics point body
    fn dynamics_body_2d() -> Body<DynamicsPoint<Cartesian<2>>, Point<Cartesian<2>>> {
        Body {
            properties: DynamicsPoint {
                position: Cartesian::<2>::default(),
                momentum: Cartesian::<2>::default(),
                net_force: Cartesian::<2>::default(),
                mass: 1.0,
            },
            sites: vec![Point::new(Cartesian::<2>::default())],
        }
    }

    /// A simple 3d dynamics point body
    fn dynamics_body_3d(mass: f64) -> Body<DynamicsPoint<Cartesian<3>>, Point<Cartesian<3>>> {
        Body {
            properties: DynamicsPoint {
                position: Cartesian::<3>::default(),
                momentum: Cartesian::<3>::default(),
                net_force: Cartesian::<3>::default(),
                mass: mass,
            },
            sites: vec![Point::new(Cartesian::from([0.0, 0.0, 0.0]))],
        }
    }

    /// A simple 2d oriented dynamics point body
    fn oriented_dynamics_body_2d() -> Body<OrientedDynamicsPoint<Cartesian<2>, Angle>, Point<Cartesian<2>>> {
        Body {
            properties: OrientedDynamicsPoint {
                position: Cartesian::<2>::default(),
                orientation: Angle::default(),
                momentum: Cartesian::<2>::default(),
                net_force: Cartesian::<2>::default(),
                moment_of_inertia: 1.0,
                angular_momentum: 0.0,
                net_torque: 0.0,
                mass: 1.0,
            },
            sites: vec![Point::new(Cartesian::from([0.0, 0.0]))],
        }
    }

    /// A simple 2d oriented dynamics point body
    fn oriented_dynamics_body_3d() -> Body<OrientedDynamicsPoint<Cartesian<3>, Versor>, Point<Cartesian<3>>> {
        Body {
            properties: OrientedDynamicsPoint {
                position: Cartesian::<3>::default(),
                orientation: Versor::default(),
                momentum: Cartesian::<3>::default(),
                net_force: Cartesian::<3>::default(),
                moment_of_inertia: Cartesian::<3>::from([1.0, 1.0, 1.0]),
                angular_momentum: Cartesian::<3>::default(),
                net_torque: Cartesian::<3>::default(),
                mass: 1.0,
            },
            sites: vec![Point::new(Cartesian::from([0.0, 0.0, 0.0]))],
        }
    }

    #[test]
    fn test_constant_volume() -> anyhow::Result<()> {
        // Instantiation
        let custom_cv = ConstantVolume {
            dt: 1.0,
            translational_kinetic_energy: 2.0,
            rotational_kinetic_energy: 3.0,
            translational_dof: 4.0,
            rotational_dof: 5.0,
        };

        // Blanket Implementation
        let dt = 2.0;
        let new_cv = ConstantVolume::new(dt);
        assert_eq!(new_cv.dt, dt);
        assert_eq!(new_cv.translational_kinetic_energy, 0.0);
        assert_eq!(new_cv.rotational_kinetic_energy, 0.0);
        assert_eq!(new_cv.translational_dof, 0.0);
        assert_eq!(new_cv.rotational_dof, 0.0);

        assert_eq!(*custom_cv.get_translational_kinetic_energy(), 2.0);
        assert_eq!(*custom_cv.get_rotational_kinetic_energy(), 3.0);
        assert_eq!(*custom_cv.get_translational_dof(), 4.0);
        assert_eq!(*custom_cv.get_rotational_dof(), 5.0);

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
        let mut thermostat = NoThermostat;  // TODO: use an actual thermostat

        // Update force first so that the particles can move
        method.update_force(&mut microstate, &force);
        
        // Check the first halfstep
        method.integrate_translation_step_one(
            &mut microstate,
            &mut thermostat,
            &mut macrostate
        );
        let mut expected_momentum = Cartesian::<3>::default()
            + (f_dir * (f_mag ) * (dt / 2.0) * -1.0);
        let expected_position = Cartesian::<3>::default()
            + expected_momentum * dt / mass;

        assert_eq!(expected_momentum, microstate.bodies()[0].item.properties.momentum);
        assert_eq!(expected_position, microstate.bodies()[0].item.properties.position);

        // Update force again
        method.update_force(&mut microstate, &force);

        // Check the second halfstep
         method.integrate_translation_step_two(
            &mut microstate,
            &mut thermostat,
            &mut macrostate
        );
        expected_momentum += f_dir * (f_mag ) * (dt / 2.0) * -1.0;
        assert_eq!(expected_momentum, microstate.bodies()[0].item.properties.momentum);


        Ok(())
    }

    #[test]
    fn test_rotational_integration_2d() {}

    #[test]
    fn test_rotational_integration_3d() {}

    #[test]
    fn test_force_update_2d() -> anyhow::Result<()> {
        let mut microstate = Microstate::builder()
            .bodies([dynamics_body_2d()])
            .try_build()?;
        let evaluator = Rigid(External(ConstantForce {
            r_0: Cartesian::from([0.0, 1.0]),
            force: Cartesian::from([0.0, 1.0]),
        }));
        let method = ConstantVolume::new(0.1);
        
        method.update_force(&mut microstate, &evaluator);        
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
        let method = ConstantVolume::new(0.1);
        
        method.update_force(&mut microstate, &evaluator);        
        assert_eq!(microstate.bodies()[0].item.properties.net_force, Cartesian::<3>::from([0.0, -1.0, 0.0]));

        Ok(())
    }

    // TODO: return here, and start by creating derive macros for NetForce, NetTorque, Momentum, AngularMomentum, etc.
    // #[test]
    // fn test_force_and_torque_update_2d() -> anyhow::Result<()> {
    //     let mut microstate = Microstate::builder()
    //         .bodies([oriented_dynamics_body_2d()])
    //         .try_build()?;

    //     let torque_evaluator = Rigid(External(ConstantTorque {
    //         alpha: 1.0,
    //         direction: 1.0
    //     }));
    //     let force_evaluator = Rigid(External(ConstantForce {
    //         alpha: 1.0,
    //         plane_origin: [0.0, 1.0].into(),
    //         plane_normal: [0.0, 1.0].try_into()?,
    //     }));
    //     let evaluator = (torque_evaluator, force_evaluator);
    //     let method = ConstantVolume::new(0.1);
        
    //     method.update_force_and_torque(&mut microstate, &evaluator);
    //     assert_eq!(microstate.bodies()[0].item.properties.net_force, Cartesian::<2>::from([0.0, 1.0]));
    //     assert_eq!(microstate.bodies()[0].item.properties.net_torque, 1.0);

        
    //     Ok(())
    // }

    #[test]
    fn test_force_and_torque_update_3d() {
        #[derive(MaximumInteractionRange)]
        struct OverallInteraction {
            force: ConstantForce,
            torque: ConstantTorque
        }

        let interaction = OverallInteraction {
            force: ConstantForce {
                alpha: f_mag,
                plane_origin: [0.0, 0.0, 0.0].into(),
                plane_normal: f_dir.to_unit()?.0,
            },
            torque: ConstantTorque { alpha: 1.0, direction: 1.0 }
        };
    }
}

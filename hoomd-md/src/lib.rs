// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

//! Simulate molecular dynamics in systems of particles.

pub mod thermostat;

use std::array;

use hoomd_interaction::{NetBodyForce, NetBodyTorque};
use hoomd_microstate::{
    Microstate, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{
        Position, Mass, Momentum, NetForce, Orientation, MomentOfInertia, AngularMomentum, NetTorque
    },
};
use hoomd_vector::{Angle, Cartesian, InnerProduct, Quaternion, Rotate, Rotation, Vector};
use thermostat::Thermostat;
use hoomd_simulation::macrostate::Temperature;

/// Integrate over translational degrees of freedom.
/// 
/// Conceptually, integration changes a system's [`Microstate`] according to
/// equations of motion that are determined by the system's metric-space and
/// degrees of freedom.
/// 
/// In translational integration, the equations of motion allow a body's
/// [`Position`], [`Velocity`], and [`Acceleration`] to evolve over time.
/// `Acceleration` only changes if the interaction model includes force, and
/// particle properties and interactions may be additionally modulated by a
/// [`Thermostat`] that maintains system temperature according to the setpoint
/// stored in a `macrostate`.
/// 
/// Mathematically, integration is accomplished using an adaptation of the
/// symplectic and time-reversible two-step Verlet integration schemes published
/// in Miller et al. (2002) and Kamberaj et al. (2005). Jens Glaser adapted
/// these derivations to also accommodate constant pressure integration.
pub trait TranslationalMotion<B, S, C, E, T, M> {
    /// Perform the first integration half-step, mutating the microstate and possibly the thermostat.
    fn integrate_translation_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, C>,
        force: &E,
        thermostat: &mut T,
        macrostate: &M,
    );

    /// Perform the second integration half-step, mutating the microstate and possibly the thermostat.
    fn integrate_translation_step_two(
        &self,
        microstate: &mut Microstate<B, S, C>,
        force: &E,
        thermostat: &mut T,
        macrostate: &M,
    );
}

/// Integrate over rotational degrees of freedom.
/// 
/// Conceptually, integration changes a system's [`Microstate`] according to
/// equations of motion that are determined by the system's metric-space and
/// degrees of freedom.
/// 
/// In rotational integration, the equations of motion allow a body's
/// [`Orientation`] and [`AngularVelocity`] to evolve over time. `AngularVelocity`
/// only changes if the interaction model includes torque, and particle properties
/// and interactions may be additionally modulated by a [`Thermostat`] that
/// maintains system temperature according to the setpoint stored in a `macrostate`.
/// 
/// Mathematically, integration is accomplished using an adaptation of the
/// symplectic and time-reversible two-step Verlet integration schemes published
/// in Miller et al. (2002) and Kamberaj et al. (2005). Jens Glaser adapted
/// these derivations to also accommodate constant pressure integration.
pub trait RotationalMotion<const N: usize, B, S, C, E, T, M> {
    /// Perform the first integration half-step, mutating the microstate and possibly the thermostat.
    #[must_use]
    fn integrate_rotation_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, C>,
        torque: &E,
        thermostat: &mut T,
        macrostate: &M,
    );

    /// Perform the second integration half-step, mutating the microstate and possibly the thermostat.
    #[must_use]
    fn integrate_rotation_step_two(
        &self,
        microstate: &mut Microstate<B, S, C>,
        torque: &E,
        thermostat: &mut T,
        macrostate: &M,
    );
}

/// Evolve a system that is constrained to a constant volume.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstantVolume {
    /// The size of a timestep.
    dt: f64,

    kinetic_energy: f64,

    dof: f64,
}

impl ConstantVolume {
    pub fn new(dt: f64) -> Self {
        Self { dt: dt, kinetic_energy: 0.0, dof: 0.0 }
    }

    pub fn get_kinetic_energy(&self) -> &f64 {
        &self.kinetic_energy
    }
}

/// TODO: add documentation
pub struct ConstantPressure;

/// Integrate over translational degrees of freedom for a system with constant
/// volume in any vector space.
/// 
/// [`ConstantVolume`] integration is only defined for macrostates with [`Temperature`].
/// 
/// The generic type names are:
/// * `V`: The [`Vector`]() type.
/// * `B`: The [`Body::properties`](crate::Body) type.
/// * `S`: The [`Site::properties`](crate::Site) type.
/// * `C`: The [`boundary`](crate::boundary) condition type.
/// * `E`: The interaction [`evaluator`]() type.
/// * `T`: The [`Thermostat`]() type.
/// * `M`: The [`macrostate`](crate::macrostate) type.
impl<V, B, S, C, E, T, M> TranslationalMotion<B, S, C, E, T, M> for ConstantVolume
where
    V: Default + Vector + InnerProduct,
    B: Position<Vector = V>
        + Momentum<Vector = V>
        + NetForce<Vector = V>
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Vector = V> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    E: NetBodyForce<V, B, S, C>,
    T: Thermostat<B, S, C, M>,
    M: Temperature,
{
    /// Perform the first integration half-step, mutating the microstate and possibly the thermostat.
    /// 
    /// `microstate` holds the system configuration that will be changed,
    /// `torque` is the evaluator that is used to calculate the net torque on every body,
    /// `thermostat` is the thermostat,
    /// `macrostate` holds the temperature setpoint (used by the thermostat),
    /// `dof` is the number of degrees of degrees of freedom (used by the thermostat),
    /// `kinetic_energy` is the kinetic energy of the system (used by the thermostat)
    /// 
    /// TODO: Do we want to allow users to set a displacement limit?
    #[inline]
    fn integrate_translation_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, C>,
        _force: &E,
        thermostat: &mut T,
        macrostate: &M,
    ) {
        let mut compute_properties = |microstate: &Microstate<B, S, C>| -> (f64, f64) {
            let integrator_ke = &mut self.kinetic_energy;
            let integrator_dof = &mut self.dof;
            let mut ke = 0.0;
            let dof = 3.0 * microstate.bodies().len() as f64;

            for body_index in 0..microstate.bodies().len() {
                // Get the the body information
                let body_properties = microstate.bodies()[body_index].item.properties.clone();

                // calculate m * v^2 part
                let velocity = body_properties.velocity();
                ke += velocity.norm_squared() * body_properties.mass();
            }
            ke *= 0.5;
            *integrator_ke = ke.clone();
            *integrator_dof = dof.clone();

            (ke, dof)
        };

        // Calculate temperature scaling factor
        let rescaling_factor = thermostat.rescaling_factor_step_one(
            microstate,
            macrostate,
            &self.dt,
            &mut compute_properties,
        );

        // For loop over a range instead of bodies().iter() since the latter holds an immutable borrow.
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Perform the integration step
            // TODO: should we use the momentum methods here?
            let acceleration = *body_properties.net_force() / *body_properties.mass();
            let velocity = body_properties.velocity();
            body_properties.set_velocity(
                velocity + ((acceleration * 0.5 * self.dt) * rescaling_factor)
            );
            *body_properties.position_mut() += velocity * self.dt;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        thermostat.advance(
            microstate,
            macrostate,
            &self.dt,
            &mut compute_properties,
        );

        microstate.increment_substep();
    }

    /// Perform the second integration half-step, mutating the microstate and possibly the thermostat.
    /// 
    /// `microstate` holds the system configuration that will be changed,
    /// `torque` is the evaluator that is used to calculate the net torque on every body,
    /// `thermostat` is the thermostat,
    /// `macrostate` holds the temperature setpoint (used by the thermostat),
    /// `dof` is the number of degrees of degrees of freedom (used by the thermostat),
    /// `kinetic_energy` is the kinetic energy of the system (used by the thermostat)
    /// 
    /// TODO: Do we want to allow users to set a displacement limit?
    #[inline]
    fn integrate_translation_step_two(
        &self,
        microstate: &mut Microstate<B, S, C>,
        force: &E,
        thermostat: &mut T,
        macrostate: &M,
    ) {
        // Calculate temperature scaling factor
        let rescaling_factor =
            thermostat.rescaling_factor_step_two(microstate, macrostate, &self.dt);

        // For loop over a range instead of bodies().iter() since the latter holds an immutable borrow.
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Calculate the net force on the body
            let net_force_new = force.net_force_on_body(microstate, body_index);
            let acceleration_new = net_force_new / *body_properties.mass();

            // Perform the integration step
            // TODO: should we use the momentum methods here?
            let velocity = body_properties.velocity();
            *body_properties.net_force_mut() = net_force_new;
            body_properties.set_velocity(
                velocity + ((acceleration_new * 0.5 * self.dt) * rescaling_factor)
            );

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        microstate.increment_substep();
    }
}

/// Integrate over rotational degrees of freedom for a system with constant
/// volume in 3-dimensional Cartesian space.
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
impl<B, S, C, E, T, M> RotationalMotion<3, B, S, C, E, T, M> for ConstantVolume
where
    B: Orientation<Rotation = Quaternion>
        + AngularMomentum<Rotation = Quaternion>
        + NetTorque<Vector=Cartesian<3>>
        + MomentOfInertia<Vector = Cartesian<3>>
        + Transform<S>
        + Position<Vector = Cartesian<3>> // TODO: should this be required?
        + Clone,
    S: Position<Vector = Cartesian<3>> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    E: NetBodyTorque<Cartesian<3>, B, S, C>,
    T: Thermostat<B, S, C, M>,
    M: Temperature,
{
    /// Perform the first integration half-step, mutating the microstate and
    /// possibly the thermostat.
    /// 
    /// `microstate` holds the system configuration that will be changed,
    /// `torque` is the evaluator that is used to calculate the net torque on every body,
    /// `thermostat` is the thermostat,
    /// `macrostate` holds the temperature setpoint (used by the thermostat),
    /// `dof` is the number of degrees of degrees of freedom (used by the thermostat),
    /// `kinetic_energy` is the kinetic energy of the system (used by the thermostat)
    #[inline]
    #[allow(clippy::too_many_lines)]
    fn integrate_rotation_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, C>,
        _torque: &E,  // do not update toruqe in this step.
        thermostat: &mut T,
        macrostate: &M,
    ) {
        let mut compute_properties = |microstate: &Microstate<B, S, C>| -> (f64, f64) {
            let integrator_ke = &mut self.kinetic_energy;
            let integrator_dof = &mut self.dof;
            let mut ke = 0.0;
            let mut dof = 0.0;
            for body_index in 0..microstate.bodies().len() {
                // Get the important information from the body
                let body_properties = microstate.bodies()[body_index].item.properties.clone();

                // Shorthand variables
                // q is the quaternion representation of orientation
                // p is the quaternion representation of angular momentum
                // I is the 3-vector diagonal values of the moment of inertia
                let q = body_properties.orientation();
                let p = body_properties.angular_momentum();
                let I = body_properties.moment_of_inertia();

                // Ignore the ke which have zero contribution to inertia
                let x_nonzero = I[0] > 0.0;
                let y_nonzero = I[1] > 0.0;
                let z_nonzero = I[2] > 0.0;

                let s = (q.conjugate() * *p) * 0.5;
                if x_nonzero {
                    ke += s.vector[0].powi(2) / I[0];
                    dof += 1.0
                };
                if y_nonzero {
                    ke += s.vector[1].powi(2) / I[1];
                    dof += 1.0
                };
                if z_nonzero {
                    ke += s.vector[2].powi(2) / I[2];
                    dof += 1.0
                };
            }
            ke *= 0.5;
            *integrator_ke = ke.clone();
            *integrator_dof = dof.clone();

            (ke, dof)
        };
        // Calculate temperature scaling factor
        let rescaling_factor = thermostat.rescaling_factor_step_one(
            microstate,
            macrostate,
            &self.dt,
            &mut compute_properties,
        );

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Shorthand variables
            // q is the quaternion representation of orientation
            // p is the quaternion representation of angular momentum
            // t is the 3-vector if torque at t, calculated at previous integrate_rotation_step_two
            //   or initialized at t=0
            // I is the 3-vector diagonal values of the moment of inertia
            let mut q = *body_properties.orientation_mut();
            let mut p = *body_properties.angular_momentum_mut();
            let t = *body_properties.net_torque();
            let I = *body_properties.moment_of_inertia();

            // Rotate torque into principal frame
            // TODO: check that this is correct
            let mut t_inframe = q.conjugate().to_versor().unwrap().rotate(&t);

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

            // Advance p and q by half a timestep following Trotter
            // factorization of Liouvillian rotation
            p +=
                q * Quaternion {
                    scalar: 0.0,
                    vector: t_inframe.coordinates.into(),
                } * self.dt;

            // Apply thermostat
            p = p * rescaling_factor;

            // TODO: what do we call these steps?
            if !z_zero {
                let p3 = Quaternion::from([-p.vector[2], p.vector[1], -p.vector[0], p.scalar]);
                let q3 = Quaternion::from([-q.vector[2], q.vector[1], -q.vector[0], q.scalar]);
                let phi3 =
                    (1. / (4. * I[2])) * (p.scalar + q3.scalar) * p.vector.dot(&q3.vector);
                let cphi3 = (0.5 * self.dt * phi3).cos();
                let sphi3 = (0.5 * self.dt * phi3).sin();

                p = p * cphi3 + p3 * sphi3;
                q = q * cphi3 + q3 * sphi3;
            }

            if !y_zero {
                let p2 = Quaternion::from([-p.vector[1], -p.vector[2], p.scalar, p.vector[0]]);
                let q2 = Quaternion::from([-q.vector[1], -q.vector[2], q.scalar, q.vector[0]]);
                let phi2 =
                    (1. / (4. * I[1])) * (p.scalar + q2.scalar) * p.vector.dot(&q2.vector);
                let cphi2 = (0.5 * self.dt * phi2).cos();
                let sphi2 = (0.5 * self.dt * phi2).sin();

                p = p * cphi2 + p2 * sphi2;
                q = q * cphi2 + q2 * sphi2;
            }

            if !x_zero {
                let p1 = Quaternion::from([-p.vector[0], p.scalar, p.vector[2], -p.vector[1]]);
                let q1 = Quaternion::from([-q.vector[0], q.scalar, q.vector[2], -q.vector[1]]);
                let phi1 =
                    (1. / (4. * I[0])) * (p.scalar + q1.scalar) * p.vector.dot(&q1.vector);
                let cphi1 = (self.dt * phi1).cos();
                let sphi1 = (self.dt * phi1).sin();

                p = p * cphi1 + p1 * sphi1;
                q = q * cphi1 + q1 * sphi1;
            }

            if !y_zero {
                let p2 = Quaternion::from([-p.vector[1], -p.vector[2], p.scalar, p.vector[0]]);
                let q2 = Quaternion::from([-q.vector[1], -q.vector[2], q.scalar, q.vector[0]]);
                let phi2 =
                    (1. / (4. * I[1])) * (p.scalar + q2.scalar) * p.vector.dot(&q2.vector);
                let cphi2 = (0.5 * self.dt * phi2).cos();
                let sphi2 = (0.5 * self.dt * phi2).sin();

                p = p * cphi2 + p2 * sphi2;
                q = q * cphi2 + q2 * sphi2;
            }

            if !z_zero {
                let p3 = Quaternion::from([-p.vector[2], p.vector[1], -p.vector[0], p.scalar]);
                let q3 = Quaternion::from([-q.vector[2], q.vector[1], -q.vector[0], q.scalar]);
                let phi3 =
                    (1. / (4. * I[2])) * (p.scalar + q3.scalar) * p.vector.dot(&q3.vector);
                let cphi3 = (0.5 * self.dt * phi3).cos();
                let sphi3 = (0.5 * self.dt * phi3).sin();

                p = p * cphi3 + p3 * sphi3;
                q = q * cphi3 + q3 * sphi3;
            }

            // Renormalize for improved stability
            q = *q.to_versor().unwrap().get();

            // Update the particle data
            *body_properties.orientation_mut() = q;
            *body_properties.angular_momentum_mut() = p;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        thermostat.advance(
            microstate,
            macrostate,
            &self.dt,
            &mut compute_properties,
        );

        microstate.increment_substep();
    }

    /// Perform the second integration half-step, mutating the microstate and
    /// possibly the thermostat.
    /// 
    /// `microstate` holds the system configuration that will be changed,
    /// `torque` is the evaluator that is used to calculate the net torque on every body,
    /// `thermostat` is the thermostat,
    /// `macrostate` holds the temperature setpoint (used by the thermostat),
    /// `dof` is the number of degrees of degrees of freedom (used by the thermostat),
    /// `kinetic_energy` is the kinetic energy of the system (used by the thermostat)
    #[inline]
    fn integrate_rotation_step_two(
        &self,
        microstate: &mut Microstate<B, S, C>,
        torque: &E,
        thermostat: &mut T,
        macrostate: &M,
    ) {
        // Calculate temperature scaling factor
        let rescaling_factor = 
            thermostat.rescaling_factor_step_two(microstate, macrostate, &self.dt);

        // Integration Step One
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Shorthand variables
            // q is the quaternion representation of orientation
            // p is angular momentum
            // I is the diagonal values of the moment of inertia
            let q = *body_properties.orientation_mut();
            let mut p = *body_properties.angular_momentum_mut();
            let I = *body_properties.moment_of_inertia();
            
            // calculate the net torque since position has been updated at integrate_rotation_step_one
            let net_t_new = torque.net_torque_on_body(microstate, body_index);
            // Update the torque in particle data
            *body_properties.net_torque_mut() = net_t_new;

            // Rotate torque into principal frame
            // TODO: check that this is correct
            let mut t_new_inframe = q.conjugate().to_versor().unwrap().rotate(&net_t_new);

            // Ignore torque along axes which have zero contribution to inertia
            let x_zero = I[0] == 0.0;
            let y_zero = I[1] == 0.0;
            let z_zero = I[2] == 0.0;

            if x_zero {
                t_new_inframe[0] = 0.0
            };
            if y_zero {
                t_new_inframe[1] = 0.0
            };
            if z_zero {
                t_new_inframe[2] = 0.0
            };


            // Apply thermostat
            p = p * rescaling_factor;

            // Advance p by half a timestep following Trotter
            // factorization of Liouvillian rotation
            p +=
                q * Quaternion {
                    scalar: 0.0,
                    vector: t_new_inframe.coordinates.into(),
                } * self.dt;

            // Update the angular momentum in particle data
            *body_properties.angular_momentum_mut() = p;

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
impl<B, S, C, E, T, M> RotationalMotion<2, B, S, C, E, T, M> for ConstantVolume
where
    B: Orientation<Rotation = Angle>
        + AngularMomentum<Rotation = f64>
        + NetTorque<Vector = f64>
        + MomentOfInertia<Vector = f64>
        + Transform<S>
        + Position<Vector = Cartesian<2>> // TODO: should this be required?
        + Clone,
    S: Position<Vector = Cartesian<2>> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    E: NetBodyTorque<f64, B, S, C>,
    T: Thermostat<B, S, C, M>,
    M: Temperature,
{
    /// Perform the first integration half-step, mutating the microstate and
    /// possibly the thermostat.
    /// 
    /// `microstate` holds the system configuration that will be changed,
    /// `torque` is the evaluator that is used to calculate the net torque on every body,
    /// `thermostat` is the thermostat,
    /// `macrostate` holds the temperature setpoint (used by the thermostat),
    /// `dof` is the number of degrees of degrees of freedom (used by the thermostat),
    /// `kinetic_energy` is the kinetic energy of the system (used by the thermostat)
    #[inline]
    fn integrate_rotation_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, C>,
        torque: &E,
        thermostat: &mut T,
        macrostate: &M,
    ) {
        let mut compute_properties = |microstate: &Microstate<B, S, C>| -> (f64, f64) {
            let integrator_ke = &mut self.kinetic_energy;
            let integrator_dof = &mut self.dof;
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
        // Calculate temperature scaling factor
        let rescaling_factor = thermostat.rescaling_factor_step_one(
            microstate,
            macrostate,
            &self.dt,
            &mut compute_properties,
        );
        // 
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Shorthand variables
            // p is the z-component of angular momentum
            // t is the z-compoenet of net torque
            // I is the z-compoenet of the moment of inertia
            let p = *body_properties.angular_momentum();
            let t = *body_properties.net_torque();
            let I = *body_properties.moment_of_inertia();
            
            // Advance p by half a timestep and q by a full timestep following Trotter
            // factorization of Liouvillian rotation
            *body_properties.angular_momentum_mut() += t * 0.5 * self.dt;
            *body_properties.angular_momentum_mut() *= rescaling_factor;
            body_properties.orientation_mut().theta += p/I * self.dt;

            // wrap angle back into [0, 2pi] to improve stability
            *body_properties.orientation_mut() = body_properties.orientation_mut().to_reduced();

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        thermostat.advance(
            microstate,
            macrostate,
            &self.dt,
            &mut compute_properties,
        );

        microstate.increment_substep();
    }

    /// Perform the first integration half-step, mutating the microstate and
    /// possibly the thermostat.
    /// 
    /// `microstate` holds the system configuration that will be changed,
    /// `torque` is the evaluator that is used to calculate the net torque on every body,
    /// `thermostat` is the thermostat,
    /// `macrostate` holds the temperature setpoint (used by the thermostat),
    /// `dof` is the number of degrees of degrees of freedom (used by the thermostat),
    /// `kinetic_energy` is the kinetic energy of the system (used by the thermostat)
    #[inline]
    fn integrate_rotation_step_two(
        &self,
        microstate: &mut Microstate<B, S, C>,
        torque: &E,
        thermostat: &mut T,
        macrostate: &M,
    ) {
        // Calculate temperature scaling factor
        let rescaling_factor = thermostat.rescaling_factor_step_two(
            microstate,
            macrostate,
            &self.dt,
        );
        // 
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // calculate the net torque since position has been updated at integrate_rotation_step_one
            let net_t_new = torque.net_torque_on_body(microstate, body_index);
            // Update the torque in particle data
            *body_properties.net_torque_mut() = net_t_new;
            
            // Advance p by half a timestep following Trotter
            // factorization of Liouvillian rotation
            *body_properties.angular_momentum_mut() *= rescaling_factor;
            *body_properties.angular_momentum_mut() += net_t_new * 0.5 * self.dt;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
        // TODO
    }
}

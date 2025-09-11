// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Simulate molecular dynamics in systems of particles.
TODO: Add documentation.
*/

pub mod thermostat;
use std::{array, marker::PhantomData, ops::{AddAssign, Index}};

use hoomd_interaction::{NetBodyForce, NetBodyTorque};
use hoomd_vector::{Cartesian, InnerProduct, Quaternion, Rotate, Vector};
use thermostat::Thermostat;
use hoomd_microstate::{boundary::{GenerateGhosts, Wrap}, property::{Acceleration, AngularVelocity, Mass, MomentOfInertia, Orientation, Position, Velocity}, Microstate, Transform};
use hoomd_simulation::macrostate::Isochoric;


/// TODO: add documentation
pub struct ConstantVolume<T, M, B, S, C>
where
    T: Thermostat<M, B, S, C>,
{
    /// The size of a timestep.
    pub dt: f64,

    /// The thermostat.
    pub thermostat: T,

    _marker: PhantomData<(M, B, S, C)>
}

/// TODO: add documentation
pub struct ConstantPressure;

/// Integrate translational degrees of freedom in N-dimensional Cartesian space.
impl<V, T, M, B, S, C, F> ConstantVolume<T, M, B, S, C>
where
    V: Default + Vector,
    T: Thermostat<M, B, S, C>,
    M: Isochoric,
    B: Position<Vector = V> + Velocity<Vector = V> + Acceleration<Vector = V> + Mass + Transform<S> + Clone,
    S: Position<Vector = V> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    F: NetBodyForce<V, B, S, C>
{
    /** Perform first half-step of the Verlet algorithm following Kamberaj 2005.
    TODO: Do we want to allow users to set a displacement limit?
    */
    #[inline]
    fn integrate_translation_step_one(
        &self,
        macrostate: M,
        microstate: &mut Microstate<B, S, C>,
        force: &F,
        dof: f64,
        kinetic_energy: f64,
    ) {
        // Calculate temperature scaling factor
        let mut rng = microstate.counter().make_rng();
        let rescaling_factor = self.thermostat.rescaling_factor_step_one(
            macrostate,
            microstate,
            self.dt,
            dof,
            kinetic_energy
        );

        // For loop over a range instead of bodies().iter() since the latter holds an immutable borrow.
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Perform the integration step
            let acceleration = *body_properties.acceleration();
            let velocity = *body_properties.velocity();
            *body_properties.velocity_mut() += (acceleration * 0.5 * self.dt) 
                * rescaling_factor;
            *body_properties.position_mut() += velocity * self.dt;

            // Update body properties accordingly, wrapping automatically
            microstate.update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        self.thermostat.advance();

        microstate.increment_substep();
    }

    /** Perform second half-step of the Verlet algorithm following Kamberaj 2005.
    TODO: Do we want to allow users to set a displacement limit?
    */
    #[inline]
    fn integrate_translation_step_two(
        &self,
        macrostate: M,
        microstate: &mut Microstate<B, S, C>,
        force: &F,
        dof: f64,
        kinetic_energy: f64,
    ) {
        // Calculate temperature scaling factor
        let mut rng = microstate.counter().make_rng();
        let rescaling_factor = self.thermostat.rescaling_factor_step_two(
            macrostate,
            microstate,
            self.dt,
            dof,
            kinetic_energy
        );

        // For loop over a range instead of bodies().iter() since the latter holds an immutable borrow.
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();
        
            // Calculate the net force on the body
            let net_force = force.net_force_on_body(microstate, body_index);    // should bundle both pairwise and non-pairwise
            
            // Perform the integration step
            *body_properties.acceleration_mut() = net_force / *body_properties.mass();
            let acceleration = *body_properties.acceleration();
            *body_properties.velocity_mut() += (acceleration * 0.5 * self.dt) * rescaling_factor;

            // Update body properties accordingly
            microstate.update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        microstate.increment_substep();
    }
}

/// Integrate rotational degrees of freedom in 3-D Cartesian space.
impl<T, M, B, S, C, F> ConstantVolume<T, M, B, S, C>
where
    T: Thermostat<M, B, S, C>,
    M: Isochoric,
    B: Orientation<Rotation = Quaternion>
        + AngularVelocity<Vector = Cartesian<3>>
        + MomentOfInertia<Vector = Cartesian<3>>
        + Clone,
    S: Position<Vector = Cartesian<3>> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    F: NetBodyTorque<Cartesian<3>, B, S, C>,
{
    /** Perform the first integration half-step, mutating the system configuration.

    `microstate` holds the system configuration that will be changed, and
    `torque` is the evaluator that is used to calculate torques used in the
    integration.
    */
    #[inline]
    fn integrate_rotation_step_one(
        &self,
        macrostate: M,
        microstate: &mut Microstate<B, S, C>,
        torque: &F,
        dof: f64,
        kinetic_energy: f64,
    ) {
        // Calculate temperature scaling factor
        let mut rng = microstate.counter().make_rng();
        let rescaling_factor = self.thermostat.rescaling_factor_step_one(
            macrostate,
            microstate,
            self.dt,
            dof,
            kinetic_energy
        );

        // Integration Step One
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Perform the integration step
            // Shorthand variables
            let mut q = *body_properties.orientation_mut();                     // quaternion representation of orientation
            let mut p = Quaternion::from(array::from_fn(|i| // angular momentum
                body_properties
                .angular_velocity()
                .coordinates[i] * body_properties.moment_of_inertia()[i]
            ));
            let mut t = torque.net_torque_on_body(microstate, body_index);  // net torque
            let I = *body_properties.moment_of_inertia();

            // Rotate torque into principal frame
            t = q.conjugate().to_versor().unwrap().rotate(&t);  // TODO: do not use unpack (should orientation always return a versor?)
            
            // Ignore torque along axes which have zero contribution to inertia
            let x_zero = I[0] == 0.0;
            let y_zero = I[1] == 0.0;
            let z_zero = I[2] == 0.0;

            if x_zero {
                t[0] = 0.0
            };
            if y_zero {
                t[0] = 0.0
            };
            if z_zero {
                t[0] = 0.0
            };

            // Advance p and q by half a timestep following Trotter factorization of Liouvillian rotation
            p += q * Quaternion {scalar: 0.0, vector: t.coordinates.into()} * self.dt;

            // Apply thermostat
            p = p * rescaling_factor;

            // TODO: what do we call these steps?
            if !z_zero {
                let mut p3 = Quaternion::from([-p.vector[2], p.vector[1], -p.vector[0], p.scalar]);
                let mut q3 = Quaternion::from([-q.vector[2], q.vector[1], -q.vector[0], q.scalar]);
                let mut phi3 = (1. / (4. * I[2])) * (p.scalar + q3.scalar) * p.vector.dot(&q3.vector);
                let cphi3 = (0.5 * self.dt * phi3).cos();
                let sphi3 = (0.5 * self.dt * phi3).sin();

                p = p * cphi3 + p3 * sphi3;
                q = q * cphi3 + q3 * sphi3;
            }

            if !y_zero {
                let mut p2 = Quaternion::from([-p.vector[1], -p.vector[2], p.scalar, p.vector[0]]);
                let mut q2 = Quaternion::from([-q.vector[1], -q.vector[2], q.scalar, q.vector[0]]);
                let mut phi2 = (1. / (4. * I[1])) * (p.scalar + q2.scalar) * p.vector.dot(&q2.vector);
                let cphi2 = (0.5 * self.dt * phi2).cos();
                let sphi2 = (0.5 * self.dt * phi2).sin();

                p = p * cphi2 + p2 * sphi2;
                q = q * cphi2 + q2 * sphi2;
            }

            if !x_zero {
                let mut p1 = Quaternion::from([-p.vector[0], p.scalar, p.vector[2], -p.vector[1]]);
                let mut q1 = Quaternion::from([-q.vector[0], q.scalar, q.vector[2], -q.vector[1]]);
                let mut phi1 = (1. / (4. * I[0])) * (p.scalar + q1.scalar) * p.vector.dot(&q1.vector);
                let cphi1 = (self.dt * phi1).cos();
                let sphi1 = (self.dt * phi1).sin();

                p = p * cphi1 + p1 * sphi1;
                q = q * cphi1 + q1 * sphi1;
            }

            if !y_zero {
                let mut p2 = Quaternion::from([-p.vector[1], -p.vector[2], p.scalar, p.vector[0]]);
                let mut q2 = Quaternion::from([-q.vector[1], -q.vector[2], q.scalar, q.vector[0]]);
                let mut phi2 = (1. / (4. * I[1])) * (p.scalar + q2.scalar) * p.vector.dot(&q2.vector);
                let cphi2 = (0.5 * self.dt * phi2).cos();
                let sphi2 = (0.5 * self.dt * phi2).sin();

                p = p * cphi2 + p2 * sphi2;
                q = q * cphi2 + q2 * sphi2;
            }

            if !z_zero {
                let mut p3 = Quaternion::from([-p.vector[2], p.vector[1], -p.vector[0], p.scalar]);
                let mut q3 = Quaternion::from([-q.vector[2], q.vector[1], -q.vector[0], q.scalar]);
                let mut phi3 = (1. / (4. * I[2])) * (p.scalar + q3.scalar) * p.vector.dot(&q3.vector);
                let cphi3 = (0.5 * self.dt * phi3).cos();
                let sphi3 = (0.5 * self.dt * phi3).sin();

                p = p * cphi3 + p3 * sphi3;
                q = q * cphi3 + q3 * sphi3;
            }

            // Renormalize for improved stability
            q = q * (1.0 / q.norm());

            // Update the particle data
            // TODO: make angular velocity calculation cleaner
            // TODO: RESUME HERE
            *body_properties.orientation_mut() = q;
            vec = p.into_iter()
                .zip(*body_properties.moment_of_inertia())
                .map(|(i1, i2)| i1 / i2)
                .collect::<Vec<f64>>();
            arr: [f64; 4] = new_angular_velocity.try_into()
                .expect("There should be exactly 4 elements");
            let new_angular_velocity = Quaternion::from(arr);
            *body_properties.angular_velocity_mut() =  new_angular_velocity;
        }

        self.thermostat.advance();
        
        microstate.increment_substep();
    }

    #[inline]
    fn integrate_rotation_step_two(&self, microstate: &mut Microstate<B, S, C>, torque: &F) {
        // TODO
    }
}
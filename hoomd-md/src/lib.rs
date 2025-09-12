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
pub struct ConstantVolume
{
    /// The size of a timestep.
    pub dt: f64,
}

/// TODO: add documentation
pub struct ConstantPressure;

/// Integrate translational degrees of freedom in N-dimensional Cartesian space.
impl<V, T, M, B, S, C, F> ConstantVolume
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
        thermostat: T,
    ) {
        // Calculate temperature scaling factor
        let mut rng = microstate.counter().make_rng();
        let rescaling_factor = thermostat.rescaling_factor_step_one(
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

        thermostat.advance(self.dt);

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

/** Integrate rotational degrees of freedom in 3-dimensional Cartesian space.

Conceptually, integration changes a system's [`Microstate`] according to
equations of motion that are determined by the system's metric-space and
degrees of freedom.

In rotational integration, the equations of motion allow a body's
[`Orientation`] and [`AngularVelocity`] to evolve over time. `AngularVelocity`
only changes if the interaction model includes torque, and particle properties
and interactions may be additionally modulated by a [`Thermostat`] that
maintains system temperature according to the setpoint stored in a `macrostate`.
[`ConstantVolume`] integration is only defined for [`Isochoric`] macrostates.

Mathematically, integration is accomplished using an adaptation of the
symplectic and time-reversible two-step Verlet integration schemes published
in Miller et al. (2002) and Kamberaj et al. (2005). Jens Glaser adapted
these derivations to also accommodate constant pressure integration.

The generic type names are:
* `B`: The [`Body::properties`](crate::Body) type.
* `S`: The [`Site::properties`](crate::Site) type.
* `C`: The [`boundary`](crate::boundary) condition type.
* `E`: The interaction [`evaluator`]() type.
* `T`: The [`Thermostat`]() type.
* `M`: The [`macrostate`](crate::macrostate) type.
*/
impl ConstantVolume
{
    /** Perform the first integration half-step, mutating the microstate and
    possibly the thermostat.
    
    `microstate` holds the system configuration that will be changed,
    `torque` is the evaluator that is used to calculate the net torque on every body,
    `thermostat` is the thermostat,
    `macrostate` holds the temperature setpoint (used by the thermostat),
    `dof` is the number of degrees of degrees of freedom (used by the thermostat),
    `kinetic_energy` is the kinetic energy of the system (used by the thermostat)
    */
    #[inline]
    fn integrate_rotation_step_one<B, S, C, E, T, M>(
        &self,
        microstate: &mut Microstate<B, S, C>,
        torque: &E,
        thermostat: &mut T,
        macrostate: &M,
        dof: u32,
        kinetic_energy: f64,
    )
    where
        B: Orientation<Rotation = Quaternion>
            + AngularVelocity<Vector = Cartesian<3>>
            + MomentOfInertia<Vector = Cartesian<3>>
            + Transform<S>
            + Position<Vector = Cartesian<3>>   // TODO: should this be required?
            + Clone,
        S: Position<Vector = Cartesian<3>> + Default,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
        E: NetBodyTorque<Cartesian<3>, B, S, C>,
        T: Thermostat<B, S, C, M>,
        M: Isochoric,
    {
        // Calculate temperature scaling factor
        let mut rng = microstate.counter().make_rng();
        let rescaling_factor = thermostat.rescaling_factor_step_one(
            microstate,
            macrostate,
            self.dt,
            dof,
            kinetic_energy
        );

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Shorthand variables
            // q is the quaternion representation of orientation
            // p is the quaternion representation of angular momentum
            // t is the 3-vector representation of net torque
            // I is the 3-vector diagonal values of the moment of inertia
            let mut q = *body_properties.orientation_mut();
            let mut p = Quaternion::from(array::from_fn(|i|
                body_properties.angular_velocity().coordinates[i]
                * body_properties.moment_of_inertia()[i]
            ));
            let mut t = torque.net_torque_on_body(microstate, body_index);
            let I = *body_properties.moment_of_inertia();

            // Rotate torque into principal frame
            // TODO: check that this is correct
            t = q.conjugate().to_versor().unwrap().rotate(&t);
            
            // Ignore torque along axes which have zero contribution to inertia
            let x_zero = I[0] == 0.0;
            let y_zero = I[1] == 0.0;
            let z_zero = I[2] == 0.0;

            if x_zero { t[0] = 0.0 };
            if y_zero { t[1] = 0.0 };
            if z_zero { t[2] = 0.0 };

            // Advance p and q by half a timestep following Trotter
            // factorization of Liouvillian rotation
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
            *body_properties.orientation_mut() = q;
            *body_properties.angular_velocity_mut() = Cartesian::from(
                array::from_fn(|i|
                    p.vector[i] / body_properties.moment_of_inertia()[i]
                )
            );

            // Update the microstate with new body properties, wrapping automatically
            microstate.update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        thermostat.advance(self.dt);

        microstate.increment_substep();
    }

    /** Perform the second integration half-step, mutating the microstate and
    possibly the thermostat.
    
    `microstate` holds the system configuration that will be changed,
    `torque` is the evaluator that is used to calculate the net torque on every body,
    `thermostat` is the thermostat,
    `macrostate` holds the temperature setpoint (used by the thermostat),
    `dof` is the number of degrees of degrees of freedom (used by the thermostat),
    `kinetic_energy` is the kinetic energy of the system (used by the thermostat)
    */
    #[inline]
    fn integrate_rotation_step_two<B, S, C, E, T, M>(
        &self,
        microstate: &mut Microstate<B, S, C>,
        torque: &E,
        thermostat: &mut T,
        macrostate: &M,
        dof: u32,
        kinetic_energy: f64,
    )
    where
        B: Orientation<Rotation = Quaternion>
            + AngularVelocity<Vector = Cartesian<3>>
            + MomentOfInertia<Vector = Cartesian<3>>
            + Transform<S>
            + Position<Vector = Cartesian<3>>   // TODO: should this be required?
            + Clone,
        S: Position<Vector = Cartesian<3>> + Default,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
        E: NetBodyTorque<Cartesian<3>, B, S, C>,
        T: Thermostat<B, S, C, M>,
        M: Isochoric,
    {
        // Calculate temperature scaling factor
        let rescaling_factor = thermostat.rescaling_factor_step_one(
            microstate,
            macrostate,
            self.dt,
            dof,
            kinetic_energy
        );

        // Integration Step One
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Shorthand variables
            // q is the quaternion representation of orientation
            // p is angular momentum
            // t is net torque
            // I is the diagonal values of the moment of inertia
            let mut q = *body_properties.orientation_mut();
            let mut p = Quaternion::from(array::from_fn(|i|
                body_properties.angular_velocity().coordinates[i]
                * body_properties.moment_of_inertia()[i]
            ));
            let mut t = torque.net_torque_on_body(microstate, body_index);
            let I = *body_properties.moment_of_inertia();

            // Rotate torque into principal frame
            // TODO: check that this is correct
            t = q.conjugate().to_versor().unwrap().rotate(&t);
            
            // Ignore torque along axes which have zero contribution to inertia
            let x_zero = I[0] == 0.0;
            let y_zero = I[1] == 0.0;
            let z_zero = I[2] == 0.0;

            if x_zero { t[0] = 0.0 };
            if y_zero { t[1] = 0.0 };
            if z_zero { t[2] = 0.0 };

            // Apply thermostat
            p = p * rescaling_factor;

            // Advance p and q by half a timestep following Trotter
            // factorization of Liouvillian rotation
            p += q * Quaternion {scalar: 0.0, vector: t.coordinates.into()} * self.dt;

            // Update the particle data
            *body_properties.angular_velocity_mut() = Cartesian::from(
                array::from_fn(|i|
                    p.vector[i] / body_properties.moment_of_inertia()[i]
                )
            );

            // Update the microstate with new body properties, wrapping automatically
            microstate.update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        thermostat.advance(self.dt);

        microstate.increment_substep();
    }
}
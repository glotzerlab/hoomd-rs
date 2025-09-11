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
use std::{array, ops::{AddAssign, Index}};

use hoomd_interaction::{NetBodyForce, NetBodyTorque};
use hoomd_vector::{Cartesian, InnerProduct, Quaternion, Rotate, Vector};
use thermostat::{Thermostat, NoThermostat};
use hoomd_microstate::{boundary::{GenerateGhosts, Wrap}, property::{Acceleration, AngularVelocity, Mass, MomentOfInertia, Orientation, Position, Velocity}, Microstate, Transform};

/** Integrate over translational degrees of freedom. 
TODO: Add example.
*/
pub trait TranslationalMotion<B, S, C, F> {
    /** Perform the first integration half-step, mutating the system configuration.
    
    `microstate` holds the system configuration that will be changed, and
    `force` is the evaluator that is used to calculate forces used in the
    integration.
    */
    fn integrate_translation_step_one(
        &self,
        microstate: &mut Microstate<B, S, C>,
        force: &F,
        kT_setpoint: Option<&f64>
    );

    /** Perform the second integration half-step, mutating the system configuration.
    
    `microstate` holds the system configuration that will be changed, and
    `force` is the evaluator that is used to calculate forces used in the
    integration.
    */
    fn integrate_translation_step_two(
        &self,
        microstate: &mut Microstate<B, S, C>,
        force: &F,
        kT_setpoint: Option<&f64>
    );
}

/** Integrate over rotational degrees of freedom. 
TODO: Add example.
*/
pub trait RotationalMotion<B, S, C, T> {
    /** Perform integration, mutating the system configuration.

    `microstate` holds the system configuration that will be changed, and
    `torque` is the evaluator that is used to calculate torques used in the
    integration.
    */
    fn integrate_rotation(&self, microstate: &mut Microstate<B, S, C>, torque: &T);
}

/// TODO: add documentation
pub struct ConstantVolume<T: Thermostat> {
    /// The size of a timestep.
    pub dt: f64,

    /// The thermostat.
    pub thermostat: T,
}

/// TODO: add documentation
pub struct ConstantPressure;

impl<V, B, S, C, T, F> TranslationalMotion<B, S, C, F> for ConstantVolume<T>
where
    V: Default + Vector,
    B: Position<Vector = V> + Velocity<Vector = V> + Acceleration<Vector = V> + Mass + Transform<S> + Clone,
    S: Position<Vector = V> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    T: Thermostat,
    F: NetBodyForce<V, B, S, C>
{
    /** Perform first half-step of the Verlet algorithm following Kamberaj 2005.
    TODO: Do we want to allow users to set a displacement limit?
    */
    #[inline]
    fn integrate_translation_step_one(
        &self,
        microstate: &mut Microstate<B, S, C>,
        force: &F,
        kT_setpoint: &T::Macrostate,
    ) {
        let mut rng = microstate.counter().make_rng();
        let degrees_of_freedom = 2;
        let kinetic_energy = 3.0;

        let rescaling_factor = self.thermostat.rescaling_factor_step_one(
            kT_setpoint,
            &rng,
            self.dt,
            degrees_of_freedom,
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

        microstate.increment_substep();
    }

    /** Perform second half-step of the Verlet algorithm following Kamberaj 2005.
    TODO: Do we want to allow users to set a displacement limit?
    */
    #[inline]
    fn integrate_translation_step_two(
        &self,
        microstate: &mut Microstate<B, S, C>,
        force: &F,
    ) {
        let rescaling_factor = self.thermostat.rescaling_factor_step_one();

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

impl<B, S, C, T> RotationalMotion<B, S, C, T> for ConstantVolume<T>
where 
    T: Thermostat
{
    #[inline]
    fn integrate_rotation(&self, microstate: &mut Microstate<B, S, C>, torque: &T) {
        // TODO
    }
}
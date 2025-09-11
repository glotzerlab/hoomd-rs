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
use thermostat::{Thermostat, NoThermostat};
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

impl<B, S, C, T> RotationalMotion<B, S, C, T> for ConstantVolume<T>
where 
    T: Thermostat
{
    #[inline]
    fn integrate_rotation(&self, microstate: &mut Microstate<B, S, C>, torque: &T) {
        // TODO
    }
}
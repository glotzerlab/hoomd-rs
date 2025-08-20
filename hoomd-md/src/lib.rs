// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Simulate molecular dynamics in systems of particles.

TODO: Expand documentation.
 */

pub mod thermostat;
use std::ops::AddAssign;

use hoomd_interaction::NetBodyForce;
use hoomd_vector::{Vector, Cartesian};
use thermostat::{Thermostat, NoThermostat};
use hoomd_microstate::{boundary::Boundary, property::{Acceleration, Mass, Position, Velocity}, Microstate, Transform};

/** Integrate over translational degrees of freedom. 
 */
pub trait TranslationalMotion<B, S, C, F> {
    /// Integrate over translational degrees of freedom. 
    fn integrate_translation(&self, microstate: &mut Microstate<B, S, C>, force: &F);
}

/** Integrate over rotational degrees of freedom. 
 */
pub trait RotationalMotion<B, S, C> {
    /// Integrate over rotational degrees of freedom.
    fn integrate_rotation(&self, microstate: &mut Microstate<B, S, C>);
}

/// TODO: add documentation
pub struct ConstantVolume<T: Thermostat> {
    /// The size of a timestep.
    pub dt: f64,

    /// The temperature in units of kT.
    pub kT: f64,

    /// The thermostat.
    pub thermostat: T,
}

/// TODO: add documentation
pub struct ConstantPressure;

/// TODO: add documentation
impl<V, B, S, C, T, F> TranslationalMotion<B, S, C, F> for ConstantVolume<T>
where 
    // B: Position<Vector = Cartesian<N>>
    V: Default + Vector,
    B: Position<Vector = V> + Velocity<Vector = V> + Acceleration<Vector = V> + Mass + Transform<S> + Clone,
    S: Position<Vector = V>,
    C: Boundary<V, B, S>,
    T: Thermostat,
    F: NetBodyForce<V, B, S, C>
{
    // TODO: Do we need to allow users to set a displacement limit?
    #[inline]
    fn integrate_translation(
        &self,
        microstate: &mut Microstate<B, S, C>,
        force: &F,
    ) {
        let rescaling_factor = self.thermostat.temperature_factor();

        // Integration Step One
        // For loop over a range instead of bodies().iter() since the latter holds an immutable borrow.
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Perform the integration step
            let acceleration = *body_properties.acceleration();
            *body_properties.velocity_mut() += (acceleration * 0.5 * self.dt) 
                * rescaling_factor;
            *body_properties.position_mut() = *body_properties.velocity() * self.dt;

            // Update body properties accordingly, wrapping automatically
            microstate.update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        // Integration Step Two
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

impl<B, S, C, T> RotationalMotion<B, S, C> for ConstantVolume<T>
where 
    T: Thermostat
{
    #[inline]
    fn integrate_rotation(&self, microstate: &mut Microstate<B, S, C>) {
        // TODO
    }
}
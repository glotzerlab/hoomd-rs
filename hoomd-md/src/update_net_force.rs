//! Define `UpdateNetForce`

use hoomd_interaction::{NetBodyForce, NetBodyForceAndTorque};
use hoomd_microstate::{Microstate, property::{NetForce, NetTorque}};
use hoomd_vector::{Vector, Wedge};

/// Compute the net force given by an interaction model and apply it to each body in the
/// microstate.
///
/// Given an interaction model that implements [`NetBodyForce`], [`UpdateNetForce`]
/// sets the [`NetForce`] property of each body in the microstate to the one computed
/// by the interaction model.
///
/// [`NetBodyForce`]: hoomd_interaction::NetBodyForce
/// [`NetForce`]: hoomd_microstate::property::NetForce
pub trait UpdateNetForce<E> {
    /// Compute and set the net force on each body.
    fn update_net_force(&mut self, interaction_model: &E);
}

/// Compute the net force and torque given by an interaction model and apply them
/// to each body in the microstate.
///
/// Given an interaction model that implements [`NetBodyForceAndTorque`], [`UpdateNetForce`]
/// sets the [`NetForce`] and [`NetTorque`] properties of each body in the microstate to
/// the ones computed by the interaction model.
///
/// [`NetBodyForceAndTorque`]: hoomd_interaction::NetBodyForceAndTorque
/// [`NetForce`]: hoomd_microstate::property::NetForce
/// [`NetTorque`]: hoomd_microstate::property::NetTorque
pub trait UpdateNetForceAndTorque<E> {
    /// Compute and set the net force and torque on each body.
    fn update_net_force_and_torque(&mut self, interaction_model: &E);
}

impl<V, B, S, X, C, E> UpdateNetForce<E> for Microstate<B, S, X, C>
where
    V: Default + Vector,
    B: NetForce<NetForce = V>,
    E: NetBodyForce<B, S, X, C, Force=V>
{
    #[inline]
    fn update_net_force(&mut self, interaction_model: &E) {
        // TODO: rayon parallelization and benchmarks
        for body_index in 0..self.bodies().len() {
            let net_force = interaction_model.net_body_force(self, body_index);
            self.set_body_net_force(body_index, net_force);
        }
    }
}

impl<V, B, S, X, C, E> UpdateNetForceAndTorque<E> for Microstate<B, S, X, C>
where
    V: Default + Vector + Wedge,
    B: NetForce<NetForce = V> + NetTorque<NetTorque = V::Bivector>,
    E: NetBodyForceAndTorque<B, S, X, C, Force=V>
{
    #[inline]
    fn update_net_force_and_torque(&mut self, interaction_model: &E) {
        // TODO: rayon parallelization and benchmarks
        for body_index in 0..self.bodies().len() {
            let (net_force, net_torque) = interaction_model.net_body_force_and_torque(self, body_index);
            self.set_body_net_force(body_index, net_force);
            self.set_body_net_torque(body_index, net_torque);
        }
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

//! Simulate molecular dynamics in systems of particles.

pub mod thermalize;
pub mod thermostat;

use std::array;

use hoomd_interaction::{NetBodyForce, NetBodyForceAndTorque, NetBodyTorque};
use hoomd_microstate::{
    Microstate, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{
        AngularMomentum, Mass, MomentOfInertia, Momentum, NetForce, NetTorque, Orientation,
        Position,
    },
};
use hoomd_simulation::macrostate::Temperature;
use hoomd_vector::{
    Angle, Cartesian, InnerProduct, Quaternion, Rotate, Vector, Versor, WedgeProduct,
};
use thermostat::Thermostat;

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
pub trait TranslationalMotion<B, S, C, T, M> {
    /// Perform the first integration half-step, mutating the microstate and possibly the thermostat.
    fn integrate_translation_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, C>,
        thermostat: &mut T,
        macrostate: &M,
    );

    /// Perform the second integration half-step, mutating the microstate and possibly the thermostat.
    fn integrate_translation_step_two(
        &mut self,
        microstate: &mut Microstate<B, S, C>,
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
pub trait RotationalMotion<const N: usize, B, S, C, T, M> {
    /// Perform the first integration half-step, mutating the microstate and possibly the thermostat.
    fn integrate_rotation_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, C>,
        thermostat: &mut T,
        macrostate: &M,
    );

    /// Perform the second integration half-step, mutating the microstate and possibly the thermostat.
    fn integrate_rotation_step_two(
        &mut self,
        microstate: &mut Microstate<B, S, C>,
        thermostat: &mut T,
        macrostate: &M,
    );
}

pub trait ForceUpdate<B, S, C, E> {
    fn update_force(&self, microstate: &mut Microstate<B, S, C>, evaluator: &E);
}

pub trait TorqueUpdate<const N: usize, B, S, C, E> {
    fn update_torque(&self, microstate: &mut Microstate<B, S, C>, evaluator: &E);
}

pub trait ForceAndTorqueUpdate<const N: usize, B, S, C, E> {
    fn update_force_and_torque(&self, microstate: &mut Microstate<B, S, C>, evaluator: &E);
}

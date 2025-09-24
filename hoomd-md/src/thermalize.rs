// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Thermalize microstat's momentum and velocity.
//! 
use hoomd_microstate::{
    Microstate, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{
        Position, Mass, Momentum, NetForce, Orientation, MomentOfInertia, AngularMomentum, NetTorque
    },
};
use hoomd_vector::{Angle, Cartesian, InnerProduct, Quaternion, Rotate, Rotation, Vector};
use rand_distr::{Distribution, Normal};
use hoomd_simulation::macrostate::Temperature;

pub trait Thermalizer <const N: usize, B, S, C> {
    /// Randomize the translational mometum of microstate.
    fn translational_motion(
        &self, 
        microstate: &mut Microstate<B, S, C>
    );
}

/// Construct the [Thermalizer].
#[derive(Clone, Debug, PartialEq)]
pub struct Thermalize {
    /// The desired temperature
    pub kT: f64,
}

impl<B, S, C> Thermalizer<2, B, S, C> for Thermalize 
    where
        B: Position<Vector = Cartesian<2>>
            + Momentum<Vector = Cartesian<2>>
            + NetForce<Vector = Cartesian<2>>
            + Mass
            + Transform<S>
            + Clone,
        S: Position<Vector = Cartesian<2>> + Default,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    /// Randomize the mometum of microstate, by drwan from
    /// a Gaussian distribution, for two-dimension system.
    /// Autometically zero the center-of-mass momentum.
    fn translational_motion(
        &self, 
        microstate: &mut Microstate<B, S, C>)

    {
        let mut rng = microstate.counter().make_rng();
        // use the first body to determine the dimension
        let mut total_momentum = Cartesian::<2>::default();

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Calculate the net force on the body
            let mass = body_properties.mass();

            let sigma_momentum = (self.kT * mass).sqrt();

            let normal = Normal::new(0.0, sigma_momentum).unwrap();
            let random_momentum: Cartesian::<2> = Cartesian::from(
                [
                    normal.sample(&mut rng),
                    normal.sample(&mut rng)             
                ]
            );
            *body_properties.momentum_mut() = random_momentum.clone();
            total_momentum += random_momentum;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        let center_of_mass_momentum = total_momentum / microstate.bodies().len() as f64;

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            *body_properties.momentum_mut() -= center_of_mass_momentum;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}

impl<B, S, C> Thermalizer<3, B, S, C> for Thermalize 
    where
        B: Position<Vector = Cartesian<3>>
            + Momentum<Vector = Cartesian<3>>
            + NetForce<Vector = Cartesian<3>>
            + Mass
            + Transform<S>
            + Clone,
        S: Position<Vector = Cartesian<3>> + Default,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    /// Randomize the mometum of microstate, by drwan from
    /// a Gaussian distribution, for three-dimension system.
    /// Autometically zero the center-of-mass momentum.
    fn translational_motion(
        &self, 
        microstate: &mut Microstate<B, S, C>)

    {
        let mut rng = microstate.counter().make_rng();
        // use the first body to determine the dimension
        let mut total_momentum = Cartesian::<3>::default();

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Calculate the net force on the body
            let mass = body_properties.mass();

            let sigma_momentum = (self.kT * mass).sqrt();

            let normal = Normal::new(0.0, sigma_momentum).unwrap();
            let random_momentum: Cartesian::<3> = Cartesian::from(
                [
                    normal.sample(&mut rng),
                    normal.sample(&mut rng),
                    normal.sample(&mut rng),           
                ]
            );
            *body_properties.momentum_mut() = random_momentum.clone();
            total_momentum += random_momentum;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        let center_of_mass_momentum = total_momentum / microstate.bodies().len() as f64;

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            *body_properties.momentum_mut() -= center_of_mass_momentum;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}
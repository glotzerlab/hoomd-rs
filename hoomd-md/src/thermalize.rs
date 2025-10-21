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
use hoomd_vector::{Angle, Cartesian, InnerProduct, Cross, Quaternion, Rotate, Rotation, Vector};
use rand_distr::{Distribution, Normal};
use hoomd_simulation::macrostate::Temperature;

pub trait TranslationalThermalizer <const N: usize, B, S, C> {
    /// Randomize the translational mometum of microstate.
    fn translational_motion(
        &self,
        microstate: &mut Microstate<B, S, C>
    );
}

pub trait RotationalThermalizer <const N: usize, B, S, C> {
    /// Randomize the angular mometum of microstate.
    fn rotational_motion(
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

impl Thermalize
{
    /// Remove the center-of-mass momentum.
    pub fn remove_com_momentum<V, B, S, C>(
        &self,
        microstate: &mut Microstate<B, S, C>)
    where
        V: Default + Vector + InnerProduct,
        B: Position<Position = V>
            + Momentum<Vector = V>
            + NetForce<Vector = V>
            + Mass
            + Transform<S>
            + Clone,
        S: Position<Position = V> + Default,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,

    {
        let mut total_mass = 0.0;
        let mut total_momentum = V::default();

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let body_properties = microstate.bodies()[body_index].item.properties.clone();

            let mass = body_properties.mass();
            let momentum = body_properties.momentum();
            total_mass += mass;
            total_momentum += *momentum;
        }

        let center_of_mass_velocity= total_momentum / total_mass;
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();
            let mass = body_properties.mass();
            let mut momentum = body_properties.momentum().clone();

            momentum -= center_of_mass_velocity * *mass;

            *body_properties.momentum_mut() = momentum;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }

    /// Remove the center-of-mass angular momentum resulting from translational DOF.
    /// TODO: Finish the implementation.
    pub fn remove_com_angular_momentum<V, B, S, C>(
        &self,
        microstate: &mut Microstate<B, S, C>)
    where
        V: Default + Vector + InnerProduct + Cross,
        B: Position<Position = V>
            + Momentum<Vector = V>
            + NetForce<Vector = V>
            + Mass
            + Transform<S>
            + Clone,
        S: Position<Position = V> + Default,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,

    {
        let mut com = V::default();
        let mut total_mass = 0.0;

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let body_properties = microstate.bodies()[body_index].item.properties.clone();

            let position = body_properties.position();
            let mass = body_properties.mass();
            com += *position * *mass;
            total_mass += *mass;
        }
        com /= total_mass;

        // Need to calculate com angular momentum and moment of inertia to
        // get angular velocity, which invloves matrix inversion
        // wait for linalg crate to complete?
        let mut com_angular_momentum = V::default();
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let position = body_properties.position();
            let velocity = body_properties.velocity();
            let mass = body_properties.mass();

            let p_to_com = *position - com;

            com_angular_momentum += p_to_com.cross(&velocity) * *mass;
        }
    }
}

impl<const N: usize, B, S, C> TranslationalThermalizer<N, B, S, C> for Thermalize
    where
        B: Position<Position = Cartesian<N>>
            + Momentum<Vector = Cartesian<N>>
            + NetForce<Vector = Cartesian<N>>
            + Mass
            + Transform<S>
            + Clone,
        S: Position<Position = Cartesian<N>> + Default,
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

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let mass = body_properties.mass();

            let sigma_momentum = (self.kT * mass).sqrt();

            let normal = Normal::new(0.0, sigma_momentum).unwrap();

            let random_momentum: Cartesian::<N> = Cartesian::from(
                [(); N].map(|_| normal.sample(&mut rng))
            );
            *body_properties.momentum_mut() = random_momentum;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}

impl<B, S, C> RotationalThermalizer<2, B, S, C> for Thermalize
    where
    B: Orientation<Rotation = Angle>
        + AngularMomentum<AngularMomentum = f64>
        + NetTorque<NetTorque=f64>
        + MomentOfInertia<Vector = f64>
        + Position<Position = Cartesian<2>>
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<2>> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,

{
    /// Randomize the mometum of microstate, by drwan from
    /// a Gaussian distribution, for two-dimension system.
    /// Autometically zero the center-of-mass momentum.
    fn rotational_motion(
        &self,
        microstate: &mut Microstate<B, S, C>)

    {
        let mut rng = microstate.counter().make_rng();

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let Iz = body_properties.moment_of_inertia();

            let sigma_angmom = (self.kT * Iz).sqrt();

            *body_properties.angular_momentum_mut() = Normal::new(0.0, sigma_angmom).unwrap().sample(&mut rng);
            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}

impl<B, S, C> RotationalThermalizer<3, B, S, C> for Thermalize
    where
    B: Orientation<Rotation = Quaternion>
        + AngularMomentum<AngularMomentum = Quaternion>
        + NetTorque<NetTorque=Cartesian<3>>
        + MomentOfInertia<Vector=Cartesian<3>>
        + Position<Position = Cartesian<3>>
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<3>> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,

{
    /// Randomize the mometum of microstate, by drwan from
    /// a Gaussian distribution, for two-dimension system.
    /// Autometically zero the center-of-mass momentum.
    fn rotational_motion(
        &self,
        microstate: &mut Microstate<B, S, C>)

    {
        let mut rng = microstate.counter().make_rng();

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let q = body_properties.orientation();
            let I = body_properties.moment_of_inertia();

            let x_nonzero = I[0] > 0.0;
            let y_nonzero = I[1] > 0.0;
            let z_nonzero = I[2] > 0.0;
            let sigma_angmom_x = (self.kT * I[0]).sqrt();
            let sigma_angmom_y = (self.kT * I[1]).sqrt();
            let sigma_angmom_z = (self.kT * I[2]).sqrt();

            // Randomize angular momentum as a vector
            let mut random_angmom_vec = Cartesian::<3>::default();

            if x_nonzero {
                random_angmom_vec[0] = Normal::new(0.0, sigma_angmom_x).unwrap().sample(&mut rng);
            };
            if y_nonzero {
                random_angmom_vec[1] = Normal::new(0.0, sigma_angmom_y).unwrap().sample(&mut rng);
            };
            if z_nonzero {
                random_angmom_vec[2] = Normal::new(0.0, sigma_angmom_z).unwrap().sample(&mut rng);
            };

            // Convert angular momoentum from a vector to a quaternion
            let random_angmom = *q * Quaternion {
                    scalar: 0.0,
                    vector: random_angmom_vec.coordinates.into(),
                } * 2.0;

            *body_properties.angular_momentum_mut() = random_angmom;
            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}

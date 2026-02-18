// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![allow(non_snake_case)]

use hoomd_microstate::{
    Microstate, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{
        AngularMomentum, MomentOfInertia, NetTorque, Orientation,
        Position,
    },
};
use hoomd_vector::{Angle, Cartesian, Versor};
use rand_distr::{Distribution, Normal};
use crate::thermalizer::{RotationalThermalizer, Thermalizer};


impl<B, S, C> RotationalThermalizer<2, B, S, C> for Thermalizer
where
    B: Orientation<Rotation = Angle>
        + AngularMomentum<AngularMomentum = f64>
        + NetTorque<NetTorque = f64>
        + MomentOfInertia<Vector = f64>
        + Position<Position = Cartesian<2>>
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<2>> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    /// Draw random angular momentum from Gaussian.
    ///
    /// The function thermalizes the two-dimensional
    /// system's rotational montion given $`k_BT`$ by
    /// drawing random angular momentum from Gaussians.
    /// Note that, in 2D, angular momentum is a scalar.
    ///
    /// According to the Maxwell–Boltzmann statistics, angular momentum
    /// $`l`$ with the moment of inertia $`I`$ distributes
    /// as a Gaussian with the probability density function
    /// with mean of 0 and the standard deviation of $`\sqrt{I k_B}`$ as:
    /// ```math
    ///    f(l) = \sqrt{ \frac{1}{2 \pi I k_B T} } \exp{\left( -\frac{l^2}{2 I k_B T} \right)}
    /// ```
    fn thermalize_rotation(&self, microstate: &mut Microstate<B, S, C>) {
        let mut rng = microstate.counter().make_rng();

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let Iz = body_properties.moment_of_inertia();

            let sigma_angmom = (self.kT * Iz).sqrt();

            *body_properties.angular_momentum_mut() =
                Normal::new(0.0, sigma_angmom).unwrap().sample(&mut rng);
            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}

impl<B, S, C> RotationalThermalizer<3, B, S, C> for Thermalizer
where
    B: Orientation<Rotation = Versor>
        + AngularMomentum<AngularMomentum = Cartesian<3>>
        + NetTorque<NetTorque = Cartesian<3>>
        + MomentOfInertia<Vector = Cartesian<3>>
        + Position<Position = Cartesian<3>>
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<3>> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    /// Draw random angular momentum from Gaussian.
    ///
    /// The function thermalizes the three-dimensional system's rotational montion given $`k_BT`$ by
    /// drawing random angular momentum from Gaussians.
    ///
    /// According to the Maxwell–Boltzmann statistics, angular momentum
    /// $`l_i,\; i=x,y,z`$ with the moment of inertia $`I_{ij}`$ that carrys
    /// the pricipal components $`I_{ii},\; i=x,y,z`$ distributes
    /// as a Gaussian with the probability density function
    /// with mean of 0 and the standard deviation of $`\sqrt{I_{ii} k_B}`$ as:
    /// ```math
    ///    f(l_i) = \sqrt{ \frac{1}{2 \pi I_{ii} k_B T} } \exp{\left( -\frac{l_i^2}{2 I_{ii} k_B T} \right)}
    /// ```
    fn thermalize_rotation(&self, microstate: &mut Microstate<B, S, C>) {
        let mut rng = microstate.counter().make_rng();

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let I = body_properties.moment_of_inertia();

            let x_nonzero = I[0] > 0.0;
            let y_nonzero = I[1] > 0.0;
            let z_nonzero = I[2] > 0.0;
            let sigma_angmom_x = (self.kT * I[0]).sqrt();
            let sigma_angmom_y = (self.kT * I[1]).sqrt();
            let sigma_angmom_z = (self.kT * I[2]).sqrt();

            // Randomize angular momentum as a vector
            let mut random_angmom = Cartesian::<3>::default();

            if x_nonzero {
                random_angmom[0] = Normal::new(0.0, sigma_angmom_x).unwrap().sample(&mut rng);
            };
            if y_nonzero {
                random_angmom[1] = Normal::new(0.0, sigma_angmom_y).unwrap().sample(&mut rng);
            };
            if z_nonzero {
                random_angmom[2] = Normal::new(0.0, sigma_angmom_z).unwrap().sample(&mut rng);
            };

            *body_properties.angular_momentum_mut() = random_angmom;
            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}
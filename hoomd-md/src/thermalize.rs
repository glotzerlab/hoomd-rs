// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Thermalize microstat's momentum and velocity.
//!
use hoomd_linear_algebra::{GeneralMatrix, MatMul, matrix::Matrix};
use hoomd_microstate::{
    Microstate, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{
        AngularMomentum, Mass, MomentOfInertia, Momentum, NetForce, NetTorque, Orientation,
        Position,
    },
};
use hoomd_vector::{Angle, Cartesian, InnerProduct, Versor, WedgeProduct};
use rand_distr::{Distribution, Normal};

pub trait TranslationalModifier<const N: usize, B, S, C> {
    /// Randomize the translational mometum of microstate.
    fn thermalize_translation(&self, microstate: &mut Microstate<B, S, C>);

    fn remove_com_momentum(&self, microstate: &mut Microstate<B, S, C>);
}

pub trait TranslationalAngularMomentumModifier<const N: usize, B, S, C> {
    /// Randomize the translational mometum of microstate.
    fn remove_com_angular_momentum(&self, microstate: &mut Microstate<B, S, C>);
}

pub trait RotationalModifier<const N: usize, B, S, C> {
    /// Randomize the angular mometum of microstate.
    fn thermalize_rotation(&self, microstate: &mut Microstate<B, S, C>);
}

/// Construct the [Thermalizer].
#[derive(Clone, Debug, PartialEq)]
pub struct Thermalize {
    /// The desired temperature
    pub kT: f64,
}

impl<B, S, C> TranslationalAngularMomentumModifier<3, B, S, C> for Thermalize
where
    B: Position<Position = Cartesian<3>>
        + Momentum<Vector = Cartesian<3>>
        + NetForce<Vector = Cartesian<3>>
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<3>> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    /// Remove the center-of-mass angular momentum resulting from translational DOF.
    /// TODO: Finish the implementation.
    fn remove_com_angular_momentum(&self, microstate: &mut Microstate<B, S, C>) {
        let mut com = Cartesian::default();
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

        let mut com_angular_momentum = Cartesian::default();
        let mut com_moment_of_inertia = Matrix::<3, 3>::zeros();
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let position = body_properties.position();
            let momentum = body_properties.momentum();
            let mass = body_properties.mass();

            let p_to_com = *position - com;

            com_angular_momentum += p_to_com.wedge_product(&momentum); // r x p

            // cast it to Matrix, resulting in a 1x3 matrix
            let p_to_com_matrix = p_to_com.to_row_matrix();
            let p_to_com_lengthsq = p_to_com.norm_squared();
            com_moment_of_inertia += (Matrix::with_diagonal([(); 3].map(|_| p_to_com_lengthsq))
                - p_to_com_matrix.transpose().matmul(&p_to_com_matrix))
                * *mass; // m * [||r||^2 x delta_ij - r_i (outer prodcut) r_j]
        }

        let com_angular_momentum_matrix = com_angular_momentum.to_row_matrix();
        // use svd to solve the omega in L = I * omega
        let (u, s, vt) = com_moment_of_inertia.svd();
        // If the system do not rotate w. r. t. the principle axis (I_principal=0),
        // set the omega component to 0 by setting the corresponding s^-1 to 0.
        let mut s_inv_dense = Matrix::<3, 3>::zeros();
        if s[0] > 0.0 {
            s_inv_dense.rows[0][0] = 1.0 / s[0];
        }
        if s[1] > 0.0 {
            s_inv_dense.rows[1][1] = 1.0 / s[1];
        }
        if s[2] > 0.0 {
            s_inv_dense.rows[2][2] = 1.0 / s[2];
        }
        // omega = L * u * s^-1 * vt (omage and L are row matrix)
        let omega_tmp = com_angular_momentum_matrix
            .matmul(&u)
            .matmul(&s_inv_dense)
            .matmul(&vt);
        let com_angular_velocity = Cartesian::from(omega_tmp.rows[0]);

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let position = body_properties.position();
            let mut momentum = body_properties.momentum().clone();
            let mass = body_properties.mass();

            let p_to_com = *position - com;

            // p_new = p_old - omega x r
            momentum -= com_angular_velocity.wedge_product(&p_to_com) * *mass;

            *body_properties.momentum_mut() = momentum;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}

impl<B, S, C> TranslationalAngularMomentumModifier<2, B, S, C> for Thermalize
where
    B: Position<Position = Cartesian<2>>
        + Momentum<Vector = Cartesian<2>>
        + NetForce<Vector = Cartesian<2>>
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<2>> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    /// Remove the center-of-mass angular momentum resulting from translational DOF.
    /// TODO: Finish the implementation.
    fn remove_com_angular_momentum(&self, microstate: &mut Microstate<B, S, C>) {
        let mut com = Cartesian::default();
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

        let mut com_angular_momentum = 0.0;
        let mut com_moment_of_inertia = 0.0;

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let position = body_properties.position();
            let momentum = body_properties.momentum();
            let mass = body_properties.mass();

            let p_to_com = *position - com;

            com_angular_momentum += p_to_com.wedge_product(&momentum);

            let p_to_com_lengthsq = p_to_com.norm_squared();
            com_moment_of_inertia += p_to_com_lengthsq * *mass;
        }

        if com_moment_of_inertia > 0.0 {
            let com_angular_velocity = com_angular_momentum / com_moment_of_inertia;

            for body_index in 0..microstate.bodies().len() {
                // Get the important information from the body
                let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

                let position = body_properties.position();
                let mut momentum = body_properties.momentum().clone();
                let mass = body_properties.mass();

                let p_to_com = *position - com;

                momentum -=
                    Cartesian::from([-p_to_com[1], p_to_com[0]]) * com_angular_velocity * *mass;

                *body_properties.momentum_mut() = momentum;

                // Update the microstate with new body properties, wrapping automatically
                microstate
                    .update_body_properties(body_index, body_properties)
                    .expect("Bodies and sites should remain in simulation boundary.");
            }
        }
    }
}

impl<const N: usize, B, S, C> TranslationalModifier<N, B, S, C> for Thermalize
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
    fn thermalize_translation(&self, microstate: &mut Microstate<B, S, C>) {
        let mut rng = microstate.counter().make_rng();

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let mass = body_properties.mass();

            let sigma_momentum = (self.kT * mass).sqrt();

            let normal = Normal::new(0.0, sigma_momentum).unwrap();

            let random_momentum: Cartesian<N> =
                Cartesian::from([(); N].map(|_| normal.sample(&mut rng)));
            *body_properties.momentum_mut() = random_momentum;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }

    /// Remove the center-of-mass momentum.
    fn remove_com_momentum(&self, microstate: &mut Microstate<B, S, C>) {
        let mut total_mass = 0.0;
        let mut total_momentum = Cartesian::<N>::default();

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let body_properties = microstate.bodies()[body_index].item.properties.clone();

            let mass = body_properties.mass();
            let momentum = body_properties.momentum();
            total_mass += mass;
            total_momentum += *momentum;
        }

        let center_of_mass_velocity = total_momentum / total_mass;
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
}

impl<B, S, C> RotationalModifier<2, B, S, C> for Thermalize
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
    /// Randomize the mometum of microstate, by drwan from
    /// a Gaussian distribution, for two-dimension system.
    /// Autometically zero the center-of-mass momentum.
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

impl<B, S, C> RotationalModifier<3, B, S, C> for Thermalize
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
    /// Randomize the mometum of microstate, by drwan from
    /// a Gaussian distribution, for two-dimension system.
    /// Autometically zero the center-of-mass momentum.
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

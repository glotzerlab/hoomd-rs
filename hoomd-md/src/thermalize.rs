// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Methods for thermalize or modify the momenta
//! of system.
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

/// Thermalize the translational motion of [`Microstate`].
///
/// Implement [`TranslationalThermalizer`] on a custom type
/// or use one of the provide method in
/// [`thermalize`](crate::thermalize) in MD simulations.
pub trait TranslationalThermalizer<const N: usize, B, S, C> {
    /// Thermalize the rotational motion.
    fn thermalize_translation(&self, microstate: &mut Microstate<B, S, C>);
}

/// Thermalize the rotational motion of [`Microstate`].
///
/// Implement [`RotationalThermalizer`] on a custom type
/// or use one of the provide method in
/// [`thermalize`](crate::thermalize) in MD simulations.
pub trait RotationalThermalizer<const N: usize, B, S, C> {
    /// Thermalize the rotational motion.
    fn thermalize_rotation(&self, microstate: &mut Microstate<B, S, C>);
}

/// Modify the translational momenta of [`Microstate`].
///
/// Implement [`TranslationalMomentumModifier`] on a custom type
/// or use one of the provide method in
/// [`thermalize`](crate::thermalize) in MD simulations.
pub trait TranslationalMomentumModifier<const N: usize, B, S, C> {
    /// Modify the translational momenta.
    fn modify(&self, microstate: &mut Microstate<B, S, C>);
}

/// Modify the angular momenta contributed from translational
/// degrees-of-freedom of [`Microstate`].
///
/// Implement [`TranslationalAngularMomentumModifier`] on a custom type
/// or use one of the provide method in
/// [`thermalize`](crate::thermalize) in MD simulations.
pub trait TranslationalAngularMomentumModifier<const N: usize, B, S, C> {
    /// Modify the angular momenta contributed from translational
    /// degrees-of-freedom.
    fn modify(&self, microstate: &mut Microstate<B, S, C>);
}

/// Construct the [Thermalizer].
#[derive(Clone, Debug, PartialEq)]
pub struct Thermalizer {
    /// The desired temperature
    pub kT: f64,
}

/// `thermalize_translation` thermalize the system's translational montion given $`k_BT`$ by
/// drawing random momentum from Gaussians.
///
/// According to the Maxwell–Boltzmann statistics, each
/// component of momentum $`p_i,\; i=x,y,z`$ with the mass $`m`$ distributes
/// as a Gaussian with the probability density function
/// with mean of 0 and the standard deviation of $`\sqrt{m k_B}`$ as:
/// ```math
///    f(p_i) = \sqrt{ \frac{1}{2 \pi m k_B T} } \exp{\left( -\frac{p_i^2}{2 m k_B T} \right)}
/// ```
///
/// It is equivalent to sample the Maxwell-Boltzmann distribution $`f_\mathrm{Maxwell-Boltzmann}(p)`$,
/// which can be obtained from the relation to the joint Gaussian
/// $`f_\mathrm{Maxwell-Boltzmann}(p) dp = f(p_x)f(p_y)f(p_z) dp_x dp_y dp_z`$ and express it in terms of the
/// magnitude of momentum $`p = \sqrt{(p_x^2+p_y^2+p_z^2)}`$ as:
/// ```math
///    f_\mathrm{Maxwell-Boltzmann}(p) = \left[ \frac{1}{2 \pi k_B T} \right]^\frac{3}{2} (\frac{4 \pi p^2}{\sqrt{m}}) \exp{\left( -\frac{p^2}{2 m k_B T} \right)}
/// ```
impl<const N: usize, B, S, C> TranslationalThermalizer<N, B, S, C> for Thermalizer
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
    /// Draw random momentum from Gaussian.
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
}

/// `thermalize_rotation` thermalize two-dimensional
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

/// `thermalize_rotation` thermalize three-dimensional system's rotational montion given $`k_BT`$ by
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

/// Remove the center-of-mass angular momentum.
pub struct ComAngularMomentumRemover;
/// `remove_com_angular_momentum` modify the three-dimensional system's momentum by zeroing the
/// center-of-mass (COM) angular momentum as
/// ```math
/// \mathbf{p}_{k,\; \mathrm{new}} = \mathbf{p}_{k,\; \mathrm{old}} - \left( \mathbf{\omega}_\mathrm{com} \times \mathbf{r}_{k,\; \mathrm{com}} \right) m_k
/// ```
/// where $`k`$ is the index of each body in a system,
/// $`\mathbf{\omega}_\mathrm{com}`$ is the COM angular velocity vector,
/// $`\mathbf{r}_{k,\; \mathrm{com}}`$ is the relative position vector
/// point from COM to $`k`$-th body, $`\mathbf{p}_{k,\; \mathrm{old}}`$
/// and $`\mathbf{p}_{k,\; \mathrm{new}}`$ are the momentum vector before and after
/// modification of $`k`$-th body, and $`m_k`$ is the mass of $`k`$-th body.
///
/// The $`\mathbf{\omega}_\mathrm{com}`$ is obtained by solving
/// the following linear system:
/// ```math
/// \mathbf{I}_\mathrm{com} \mathbf{\omega}_\mathrm{com} = \mathbf{L}_\mathrm{com}
/// ```
/// where $`\mathbf{I}_\mathrm{com}`$ is the COM moment-of-inertia matrix, and
/// $`\mathbf{L}_\mathrm{com}`$ is the COM angular momentum.
/// If the algorithm found one pricipal component of $`\mathbf{I}_\mathrm{com}`$
/// is 0, it will set the corresponding $`\mathbf{\omega}_\mathrm{com}`$ component
/// to 0, by assuming the system do not rotate with respect to the corresponding
/// principal axis.
impl<B, S, C> TranslationalAngularMomentumModifier<3, B, S, C> for ComAngularMomentumRemover
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
    fn modify(&self, microstate: &mut Microstate<B, S, C>) {
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

/// `remove_com_angular_momentum` modify the two-dimensional system's momentum by zeroing the
/// center-of-mass (COM) angular momentum as
/// ```math
/// \mathbf{p}_{k,\; \mathrm{new}} = \mathbf{p}_{k,\; \mathrm{old}} - \left( [-r_{k,\; \mathrm{com}}^{y}, r_{k,\; \mathrm{com}}^{x}] \right) \frac{l_\mathrm{com}}{I_\mathrm{com}} m_k
/// ```
/// where $`k`$ is the index of each body in a system,
/// $`l_\mathrm{com}`$ is the COM angular momentum, $`I_\mathrm{com}`$ is the COM moment of inertia,
/// $`r_{k,\; \mathrm{com}}^{i}`$ is the relative position vector component $`i`$
/// pointing from COM to $`k`$-th body, $`\mathbf{p}_{k,\; \mathrm{old}}`$
/// and $`\mathbf{p}_{k,\; \mathrm{new}}`$ are the momentum vector before and after
/// modification of $`k`$-th body, and $`m_k`$ is the mass of $`k`$-th body.
///
impl<B, S, C> TranslationalAngularMomentumModifier<2, B, S, C> for ComAngularMomentumRemover
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
    fn modify(&self, microstate: &mut Microstate<B, S, C>) {
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

/// Remove the center-of-mass momentum.
pub struct ComMomentumRemover;

/// `remove_com_momentum` modify the system's momentum by zeroing the
/// center-of-mass momentum as
/// ```math
/// \mathbf{p}_{k,\; \mathrm{new}} = \mathbf{p}_{k,\; \mathrm{old}} - \frac{\sum_k \mathbf{p}_{k,\; \mathrm{old}}}{\sum_k m_k} m_k
/// ```
/// where $`k`$ is the index of each body in a system, $`\mathbf{p}_{k,\; \mathrm{old}}`$
/// and $`\mathbf{p}_{k,\; \mathrm{new}}`$ are the momentum vector before and after
/// modification of $`k`$-th body, and $`m_k`$ is the mass of $`k`$-th body.
impl<const N: usize, B, S, C> TranslationalMomentumModifier<N, B, S, C> for ComMomentumRemover
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
    /// Remove the center-of-mass momentum.
    fn modify(&self, microstate: &mut Microstate<B, S, C>) {
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

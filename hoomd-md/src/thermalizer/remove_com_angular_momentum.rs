// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
use hoomd_linear_algebra::{GeneralMatrix, MatMul, matrix::Matrix};
use hoomd_microstate::{
    Microstate, SiteKey, Transform, boundary::{GenerateGhosts, Wrap}, property::{
        Mass, Momentum, NetForce, Position,
    }
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Cartesian, InnerProduct, TensorProduct, WedgeProduct};
use crate::thermalizer::TranslationalMomentumModifier;


/// Remove the center-of-mass angular momentum.
pub struct ComAngularMomentumRemover;

impl<B, S, X, C> TranslationalMomentumModifier<3, B, S, X, C> for ComAngularMomentumRemover
where
    B: Position<Position = Cartesian<3>>
        + Momentum<Vector = Cartesian<3>>
        + NetForce<Vector = Cartesian<3>>
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    /// Remove the center-of-mass angular momentum resulting from translational DOF.
    /// 
    /// The function modifies the three-dimensional system's momentum by zeroing the
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
    fn modify(&self, microstate: &mut Microstate<B, S, X, C>) {
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
            let body_properties = microstate.bodies()[body_index].item.properties.clone();

            let position = body_properties.position();
            let momentum = body_properties.momentum();
            let mass = body_properties.mass();

            let p_to_com = *position - com;

            com_angular_momentum += p_to_com.wedge_product(&momentum); // r x p

            let p_to_com_lengthsq = p_to_com.norm_squared();
            com_moment_of_inertia += (Matrix::with_diagonal([(); 3].map(|_| p_to_com_lengthsq))
                - p_to_com.tensor_product(&p_to_com))
                * *mass; // m * [||r||^2 x delta_ij - r_i (tensor prodcut) r_j]
        }

        let com_angular_momentum_matrix = com_angular_momentum.to_row_matrix();
        // use svd to solve the omega in L = omega * I
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
        // omega = L * v * s^-1 * u^t (omage and L are row matrix)
        let omega_tmp = com_angular_momentum_matrix
            .matmul(&vt.transpose())
            .matmul(&s_inv_dense)
            .matmul(&u.transpose());
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

impl<B, S, X, C> TranslationalMomentumModifier<2, B, S, X, C> for ComAngularMomentumRemover
where
    B: Position<Position = Cartesian<2>>
        + Momentum<Vector = Cartesian<2>>
        + NetForce<Vector = Cartesian<2>>
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    /// Remove the center-of-mass angular momentum resulting from translational DOF.
    /// 
    /// The function modifies the two-dimensional system's momentum by zeroing the
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
    fn modify(&self, microstate: &mut Microstate<B, S, X, C>) {
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
            let body_properties = microstate.bodies()[body_index].item.properties.clone();

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
                    p_to_com.perpendicular() * com_angular_velocity * *mass;

                *body_properties.momentum_mut() = momentum;

                // Update the microstate with new body properties, wrapping automatically
                microstate
                    .update_body_properties(body_index, body_properties)
                    .expect("Bodies and sites should remain in simulation boundary.");
            }
        }
    }
}
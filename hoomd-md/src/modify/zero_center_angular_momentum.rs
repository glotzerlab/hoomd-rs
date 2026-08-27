// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ZeroCenterAngularMomentum`

use super::ZeroCenterAngularMomentum;
use hoomd_linear_algebra::{GeneralMatrix, MatMul, matrix::Matrix};
use hoomd_microstate::{
    Body,
    Microstate,
    SiteKey,
    Tagged,
    Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{
        Mass,
        Momentum,
        Position,
    },
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Cartesian, InnerProduct, Outer, Wedge};

/// Negate the overall rotational motion of the system's center of mass.
/// 
/// This trait binds the center of mass rotational negation scheme to the type
/// that represents position and momentum. Implement this trait on a type that
/// represents body position to make a [`Microstate`] containing such bodies
/// compatible with [`ZeroCenterAngularMomentum`].
///
/// [`Microstate`]: hoomd_microstate::Microstate
pub trait ZeroCenterRotation {
    /// Type that represents a body's position.
    type Position;

    /// Type that represents a body's momentum.
    type Momentum;

    /// Type that represents the system's angular momentum about its center of mass.
    type AngularMomentum;

    /// Type that represents the system's moment of inertia.
    type MomentOfInertia;

    /// Type that represents the system's full (non-diagonalized) moment of inertia.
    type FullMomentOfInertia;

    /// Calculate the contribution of a body to the system's moment of inertia.
    fn body_contribution_to_center_moment_of_inertia(
        body_position_relative_to_center: &Self::Position,
        mass: &f64,
    ) -> Self::FullMomentOfInertia;

    /// Calculate the system's angular velocity.
    fn system_center_angular_velocity(
        center_angular_momentum: &Self::AngularMomentum,
        center_moment_of_inertia: &Self::FullMomentOfInertia,
    ) -> Self::AngularMomentum;

    /// Calculate the correction term for a body's momentum.
    /// 
    /// Cumulatively, once all bodies are corrected, the system's
    /// angular momentum will be zero.
    fn body_momentum_correction(
        body_position_relative_to_center: &Self::Position,
        center_angular_velocity: &Self::AngularMomentum,
        mass: &f64,
    ) -> Self::Momentum;
}

/// Rotational motion negation for systems in 2-dimensional cartesian space.
impl ZeroCenterRotation for Cartesian<2> {
    type Position = Cartesian<2>;
    type Momentum = Cartesian<2>;
    type AngularMomentum = f64;
    type MomentOfInertia = f64;
    type FullMomentOfInertia = f64;

    /// Calculate the contribution of a body to the system's moment of inertia.
    /// 
    /// The contribution of body $` i `$ is given by
    /// 
    /// ```math
    /// \Delta I_{sys} = m_i \cdot \left| \vec{r}_i' \right|^2
    /// ```
    /// 
    /// where $` \vec{r}_i' = \vec{r}_i - \vec{r}_{com} `$ is the position of
    /// the body relative to the system's center of mass.
    fn body_contribution_to_center_moment_of_inertia(
        body_position_relative_to_center: &Self::Position,
        mass: &f64,
    ) -> Self::FullMomentOfInertia {
        body_position_relative_to_center.norm_squared() * mass
    }

    /// Calculate the system's angular velocity about its center of mass.
    /// 
    /// The system's angular velocity is given by
    /// 
    /// ```math
    /// \omega_{sys} = L_{sys} / I_{sys}
    /// ```
    fn system_center_angular_velocity(
        center_angular_momentum: &Self::AngularMomentum,
        center_moment_of_inertia: &Self::MomentOfInertia,
    ) -> Self::AngularMomentum {
        center_angular_momentum / center_moment_of_inertia
    }

    /// Calculate the correction term for a body's momentum.
    /// 
    /// Cumulatively, once all bodies are corrected, the system's
    /// angular momentum will be zero.
    /// 
    /// The correction term is given by
    /// 
    /// ```math
    /// \Delta \vec{p}_i = m_i \omega_{sys} \cdot \vec{r}_i' \begin{bmatrix} 0 & -1 \\ 1 & 0\end{bmatrix}
    /// ```
    /// 
    /// where $` \vec{r}_i' = \vec{r}_i - \vec{r}_{com} `$ is the position of
    /// the body relative to the system's center of mass and the matrix rotates
    /// the vector counterclockwise by $` \pi/2 `$ radians.
    fn body_momentum_correction(
        body_position_relative_to_center: &Self::Position,
        center_angular_velocity: &Self::AngularMomentum,
        mass: &f64,
    ) -> Self::Momentum {
        body_position_relative_to_center.perpendicular() * *center_angular_velocity * *mass
    }
}

/// Rotational motion negation for systems in 3-dimensional cartesian space.
impl ZeroCenterRotation for Cartesian<3> {
    type Position = Cartesian<3>;
    type Momentum = Cartesian<3>;
    type AngularMomentum = Cartesian<3>;
    type MomentOfInertia = [f64; 3];
    type FullMomentOfInertia = Matrix<3,3>;

    /// Calculate the contribution of a body to the system's moment of inertia.
    /// 
    /// The contribution of body $` i `$ is given by
    /// 
    /// ```math
    /// \Delta \mathbf{I}_{sys} = m_i \cdot \left[
    /// \begin{bmatrix}
    /// \left|\vec{r}_i'\right|^2 &                         0 &                         0 \\
    ///                         0 & \left|\vec{r}_i'\right|^2 &                         0 \\
    ///                         0 &                         0 & \left|\vec{r}_i'\right|^2
    /// \end{bmatrix}
    /// - \left( \vec{r}_i' \otimes \vec{r}_i' \right)
    /// \right]
    /// ```
    /// 
    /// where $` \vec{r}_i' = \vec{r}_i - \vec{r}_{com} `$ is the position of
    /// the body relative to the system's center of mass.
    fn body_contribution_to_center_moment_of_inertia(
        body_position_relative_to_center: &Self::Position,
        mass: &f64,
    ) -> Self::FullMomentOfInertia {
        let r = *body_position_relative_to_center;
        (Matrix::with_diagonal([r.norm_squared(); 3]) - r.outer(&r)) * *mass
    }

    /// Calculate the system's angular velocity about its center of mass.
    /// 
    /// Given the [singular value decomposition] of the system's moment of
    /// inertia
    /// 
    /// ```math
    /// \mathbf{I}_{sys} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^*
    /// ```
    /// 
    /// the system's angular velocity is given by
    /// 
    /// ```math
    /// \vec{\omega}_{sys} = \mathbf{L}_{sys}^T \mathbf{V}^{*T} \mathbf{\Sigma}^{-1} \mathbf{U}^T
    /// ```
    /// 
    /// where $` \mathbf{L}_{sys}^T `$ is the row-matrix form of the system's
    /// angular momentum vector.
    /// 
    /// [singular value decomposition]: https://en.wikipedia.org/wiki/Singular_value_decomposition
    fn system_center_angular_velocity(
        center_angular_momentum: &Self::AngularMomentum,
        center_moment_of_inertia: &Self::FullMomentOfInertia,
    ) -> Self::AngularMomentum {
        let (u, s, vt) = center_moment_of_inertia.svd();

        // If the system does not rotate w. r. t. the principle axis (I_principal=0),
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

        // omega = L * v * s^-1 * u^t (omega and L are row matrices)
        let omega = center_angular_momentum
            .to_row_matrix()
            .matmul(&vt.transpose())
            .matmul(&s_inv_dense)
            .matmul(&u.transpose());
        let center_angular_velocity = Cartesian::from(omega.rows[0]);

        center_angular_velocity
    }

    /// Calculate the correction term for a body's momentum.
    /// 
    /// Cumulatively, once all bodies are corrected, the system's
    /// angular momentum will be zero.
    /// 
    /// The correction term is given by
    /// 
    /// ```math
    /// \Delta \vec{p}_i = - m_i \cdot (\vec{\omega}_{sys} \wedge \vec{r}_i')
    /// ```
    /// 
    /// where $` \vec{r}_i' = \vec{r}_i - \vec{r}_{com} `$ is the position of
    /// the body relative to the system's center of mass.
    fn body_momentum_correction(
        body_position_relative_to_center: &Self::Position,
        center_angular_velocity: &Self::AngularMomentum,
        mass: &f64,
    ) -> Self::Momentum {
        center_angular_velocity.wedge(body_position_relative_to_center) * *mass * -1.0
    }
}

/// Remove collective rotational motion about the system's center of mass.
/// 
/// The momentum of each body is adjusted to zero out the overall angular
/// momentum of the system about its center of mass, following a procedure that
/// is bound to the type representing position and momentum through the trait
/// [`ZeroCenterRotation`].
impl<V, B, S, X, C> ZeroCenterAngularMomentum<B, S> for Microstate<B, S, X, C>
where
    V: Default
        + Copy
        + Wedge
        + ZeroCenterRotation<Position = V, Momentum = V, AngularMomentum = V::Bivector>
        + std::ops::AddAssign
        + std::ops::DivAssign<f64>
        + std::ops::Mul<f64, Output = V>
        + std::ops::Sub<Output = V>,
    B: Clone
        + Transform<S>
        + Position<Position = V>
        + Momentum<Momentum = V>
        + Mass,
    S: Default + Position<Position = V>,
    X: PointUpdate<V, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    V::Bivector: std::ops::AddAssign,
    <V as ZeroCenterRotation>::FullMomentOfInertia: Default
        + std::ops::AddAssign,
    <V as ZeroCenterRotation>::AngularMomentum: Default,
{
    fn zero_center_angular_momentum_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        should_zero_body: F,
    ) {
        // Calculate the system's total mass and center of mass
        let mut center_of_mass = V::default();
        let mut total_mass = 0.0;

        for body in self.bodies() {
            if !should_zero_body(body) {
                continue;
            }

            let position = body.item.properties.position();
            let mass = body.item.properties.mass();

            center_of_mass += *position * mass;
            total_mass += mass;
        }
        center_of_mass /= total_mass;

        // Calculate the system's overall moment of inertia and angular momentum
        let mut center_angular_momentum = <V as ZeroCenterRotation>::AngularMomentum::default();
        let mut center_moment_of_inertia = <V as ZeroCenterRotation>::FullMomentOfInertia::default();

        for body in self.bodies() {
            if !should_zero_body(body) {
                continue;
            }

            let body_position_relative_to_center = *body.item.properties.position()
                - center_of_mass;
            
            center_angular_momentum += body_position_relative_to_center.wedge(
                body.item.properties.momentum()
            );
            
            center_moment_of_inertia += <V as ZeroCenterRotation>::body_contribution_to_center_moment_of_inertia(
                &body_position_relative_to_center,
                &body.item.properties.mass()
            );
        }

        // Calculate the system's overall angular velocity
        let center_angular_velocity = <V as ZeroCenterRotation>::system_center_angular_velocity(
            &center_angular_momentum,
            &center_moment_of_inertia
        );

        // Using the system's overall angular velocity, modify each body's momentum
        // to effectively zero out the system's angular momentum
        for body_index in 0..self.bodies().len() {
            let body = &self.bodies()[body_index];
            if !should_zero_body(body) {
                continue;
            }

            let mut body_properties = body.item.properties.clone();

            let position = body_properties.position();
            let mass = body_properties.mass();

            let body_position_relative_to_center = *position - center_of_mass;

            *body_properties.momentum_mut() += <V as ZeroCenterRotation>::body_momentum_correction(
                &body_position_relative_to_center,
                &center_angular_velocity,
                &mass,
            );

            // Update the microstate
            self
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}

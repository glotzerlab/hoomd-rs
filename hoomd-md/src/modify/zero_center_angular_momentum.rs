// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ZeroCenterAngularMomentum`

use std::ops::{AddAssign, DivAssign, Mul, Sub};

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
        Orientation,
        Position,
        RotationalMotionTypes,
    },
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Angle, Cartesian, InnerProduct, Outer, Versor, Wedge};

/// When we require this trait in the bounds for the impl block for
/// ZeroCenterAngularMomentum on Microstate, there is a problem if we use the
/// same pattern as before, where this trait is implemented on rotation types
/// that are then bound to body orientation. If we do that, then this
/// functionality is only available for microstates with oriented bodies. What
/// else can we do?
/// 
/// Options:
/// 1. Add R to this trait's generics
/// 2. Create a binding between Vector type and Orientation type. Cartesian<2>
/// would then always be bound to Angle, likewise for Cartesian<3> and Versor.
/// The orientation type would then be inferred from the position type.
/// 3. Design this trait to instead be implemented on vector types, rather than
/// orientation types.
/// 
/// I'll try option 3 first.

/// Negate the overall rotational motion of the system's center of mass.
/// 
/// This trait binds the center of mass rotational negation scheme to the type
/// that represents position. Implement this trait on a type that represents
/// body position to make a [`Microstate`] containing such bodies compatible
/// with [`ZeroCenterAngularMomentum`].
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

    /// Calculate the contribution of a body to the system's overall moment of inertia.
    fn body_contribution_to_center_moment_of_inertia(
        body_position_relative_to_center: &Self::Position,
        mass: &f64,
    ) -> Self::FullMomentOfInertia;

    /// Calculate the system's overall angular velocity.
    fn system_center_angular_velocity(
        center_angular_momentum: &Self::AngularMomentum,
        center_moment_of_inertia: &Self::FullMomentOfInertia,
    ) -> Self::AngularMomentum;

    /// Calculate the correction term for a body's momentum.
    /// 
    /// Cumulatively, once all bodies are corrected, the system's overall
    /// angular momentum will be zero.
    fn body_momentum_correction(
        body_position_relative_to_center: &Self::Position,
        center_angular_velocity: &Self::AngularMomentum,
        mass: &f64,
    ) -> Self::Momentum;
}

impl ZeroCenterRotation for Cartesian<2> {
    type Position = Cartesian<2>;
    type Momentum = Cartesian<2>;
    type AngularMomentum = f64;
    type MomentOfInertia = f64;
    type FullMomentOfInertia = f64;

    fn body_contribution_to_center_moment_of_inertia(
        body_position_relative_to_center: &Self::Position,
        mass: &f64,
    ) -> Self::FullMomentOfInertia {
        body_position_relative_to_center.norm_squared() * mass
    }

    fn system_center_angular_velocity(
        center_angular_momentum: &Self::AngularMomentum,
        center_moment_of_inertia: &Self::MomentOfInertia,
    ) -> Self::AngularMomentum {
        center_angular_momentum / center_moment_of_inertia
    }

    fn body_momentum_correction(
        body_position_relative_to_center: &Self::Position,
        center_angular_velocity: &Self::AngularMomentum,
        mass: &f64,
    ) -> Self::Momentum {
        body_position_relative_to_center.perpendicular() * *center_angular_velocity * *mass
    }
}

impl ZeroCenterRotation for Cartesian<3> {
    type Position = Cartesian<3>;
    type Momentum = Cartesian<3>;
    type AngularMomentum = Cartesian<3>;
    type MomentOfInertia = [f64; 3];
    type FullMomentOfInertia = Matrix<3,3>;

    fn body_contribution_to_center_moment_of_inertia(
        body_position_relative_to_center: &Self::Position,
        mass: &f64,
    ) -> Self::FullMomentOfInertia {
        let r = *body_position_relative_to_center;
        (Matrix::with_diagonal([r.norm_squared(); 3]) - r.outer(&r)) * *mass
    }

    fn system_center_angular_velocity(
        center_angular_momentum: &Self::AngularMomentum,
        center_moment_of_inertia: &Self::FullMomentOfInertia,
    ) -> Self::AngularMomentum {
        let (u, s, vt) = center_moment_of_inertia.svd();

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

        // omega = L * v * s^-1 * u^t (omega and L are row matrices)
        let omega = center_angular_momentum
            .to_row_matrix()
            .matmul(&vt.transpose())
            .matmul(&s_inv_dense)
            .matmul(&u.transpose());
        let center_angular_velocity = Cartesian::from(omega.rows[0]);

        center_angular_velocity
    }

    fn body_momentum_correction(
        body_position_relative_to_center: &Self::Position,
        center_angular_velocity: &Self::AngularMomentum,
        mass: &f64,
    ) -> Self::Momentum {
        center_angular_velocity.wedge(body_position_relative_to_center) * *mass * -1.0
    }
}

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
    V::Bivector: AddAssign,
    <V as ZeroCenterRotation>::FullMomentOfInertia: Default + AddAssign,
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

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ZeroAngularMomentum`

use super::ZeroAngularMomentum;
use hoomd_linear_algebra::{GeneralMatrix, MatMul, matrix::Matrix};
use hoomd_microstate::{
    Microstate, SiteKey, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{DynamicOrientedPoint, DynamicPoint, Mass, Momentum, Position},
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Angle, Cartesian, InnerProduct, Outer, Versor, Wedge};

/// Zero a 3D microstate's angular momentum.
#[inline]
fn zero_angular_momentum_3d<B, S, X, C>(microstate: &mut Microstate<B, S, X, C>)
where
    B: Position<Position = Cartesian<3>>
        + Mass
        + Momentum<Momentum = Cartesian<3>>
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
        let mut center_of_mass = Cartesian::default();
        let mut total_mass = 0.0;

        for body in microstate.bodies() {
            let position = body.item.properties.position();
            let mass = body.item.properties.mass();

            center_of_mass += *position * mass;
            total_mass += mass;
        }
        center_of_mass /= total_mass;

        let mut angular_momentum_center = Cartesian::default();
        let mut moment_of_inertia_center = Matrix::<3, 3>::zeros();
        for body in microstate.bodies() {
            let position = body.item.properties.position();
            let momentum = body.item.properties.momentum();
            let mass = body.item.properties.mass();

            let r = *position - center_of_mass;
            angular_momentum_center += r.wedge(momentum);

            moment_of_inertia_center += (Matrix::with_diagonal([r.norm_squared(); 3])
                - r.outer(&r))
                * mass;
        }

        let center_angular_momentum_matrix = angular_momentum_center.to_row_matrix();
        let (u, s, vt) = moment_of_inertia_center.svd();

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
        let omega = center_angular_momentum_matrix
            .matmul(&vt.transpose())
            .matmul(&s_inv_dense)
            .matmul(&u.transpose());
        let center_angular_velocity = Cartesian::from(omega.rows[0]);

        for body_index in 0..microstate.bodies().len() {
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let position = body_properties.position();
            let mass = body_properties.mass();

            let r = *position - center_of_mass;

            *body_properties.momentum_mut() -= center_angular_velocity.wedge(&r) * mass;

            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
    

/// Zero a 2D microstate's angular momentum.
#[inline]
fn zero_angular_momentum_2d<B, S, X, C>(microstate: &mut Microstate<B, S, X, C>)
where
    B: Position<Position = Cartesian<2>>
        + Mass
        + Momentum<Momentum = Cartesian<2>>
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{        let mut center_of_mass = Cartesian::default();
        let mut total_mass = 0.0;

        for body in microstate.bodies() {
            let position = body.item.properties.position();
            let mass = body.item.properties.mass();

            center_of_mass += *position * mass;
            total_mass += mass;
        }
        center_of_mass /= total_mass;

        let mut angular_momentum_center = 0.0;
        let mut moment_of_inertia_center = 0.0;

        for body in microstate.bodies() {
            let position = body.item.properties.position();
            let momentum = body.item.properties.momentum();
            let mass = body.item.properties.mass();

            let r = *position - center_of_mass;

            angular_momentum_center += r.wedge(momentum);

            moment_of_inertia_center += r.norm_squared() * mass;
        }

        if moment_of_inertia_center > 0.0 {
            let angular_velocity_center = angular_momentum_center / moment_of_inertia_center;

            for body_index in 0..microstate.bodies().len() {
                let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

                let position = body_properties.position();
                let mass = body_properties.mass();

                let r = *position - center_of_mass;

                *body_properties.momentum_mut() -= r.perpendicular() * angular_velocity_center * mass;

                microstate
                    .update_body_properties(body_index, body_properties)
                    .expect("Bodies and sites should remain in simulation boundary.");
            }
        }
}

impl<S, X, C> ZeroAngularMomentum for Microstate<DynamicOrientedPoint<Cartesian<3>, Versor>, S, X, C> 
where
    DynamicOrientedPoint<Cartesian<3>, Versor>: Transform<S>,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<DynamicOrientedPoint<Cartesian<3>, Versor>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn zero_angular_momentum(&mut self) {
        zero_angular_momentum_3d(self);
    }
}

impl<S, X, C> ZeroAngularMomentum for Microstate<DynamicPoint<Cartesian<3>>, S, X, C> 
where
    DynamicPoint<Cartesian<3>>: Transform<S>,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<DynamicPoint<Cartesian<3>>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn zero_angular_momentum(&mut self) {
        zero_angular_momentum_3d(self);
    }
}

impl<S, X, C> ZeroAngularMomentum for Microstate<DynamicOrientedPoint<Cartesian<2>, Angle>, S, X, C>
where
    DynamicOrientedPoint<Cartesian<2>, Angle>: Transform<S>,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<DynamicOrientedPoint<Cartesian<2>, Angle>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn zero_angular_momentum(&mut self) {
        zero_angular_momentum_2d(self);
    }
}

impl<S, X, C> ZeroAngularMomentum for Microstate<DynamicPoint<Cartesian<2>>, S, X, C>
where
    DynamicPoint<Cartesian<2>>: Transform<S>,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<DynamicPoint<Cartesian<2>>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn zero_angular_momentum(&mut self) {
        zero_angular_momentum_2d(self);
    }
}

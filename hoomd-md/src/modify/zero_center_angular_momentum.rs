// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ZeroCenterAngularMomentum`

use super::ZeroCenterAngularMomentum;
use hoomd_linear_algebra::{GeneralMatrix, MatMul, matrix::Matrix};
use hoomd_microstate::{
    Body, Microstate, SiteKey, Tagged, Transform, boundary::{GenerateGhosts, Wrap}, property::{
        CustomBodyCartesian2, CustomBodyCartesian3, DynamicOrientedPoint, DynamicPoint, Mass, Momentum, Position, RotationalMotionTypes,
    },
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Cartesian, InnerProduct, Outer, Wedge};

/// Zero a 3D microstate's angular momentum.
#[inline]
fn zero_angular_momentum_cartesian3<B, S, X, C, F>(
    microstate: &mut Microstate<B, S, X, C>,
    should_zero_body: F,
) where
    B: Position<Position = Cartesian<3>>
        + Mass
        + Momentum<Momentum = Cartesian<3>>
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    F: Fn(&Tagged<Body<B, S>>) -> bool,
{
    let mut center_of_mass = Cartesian::default();
    let mut total_mass = 0.0;

    for body in microstate.bodies() {
        if !should_zero_body(body) {
            continue;
        }

        let position = body.item.properties.position();
        let mass = body.item.properties.mass();

        center_of_mass += *position * mass;
        total_mass += mass;
    }
    center_of_mass /= total_mass;

    let mut angular_momentum_center = Cartesian::default();
    let mut moment_of_inertia_center = Matrix::<3, 3>::zeros();
    for body in microstate.bodies() {
        if !should_zero_body(body) {
            continue;
        }

        let position = body.item.properties.position();
        let momentum = body.item.properties.momentum();
        let mass = body.item.properties.mass();

        let r = *position - center_of_mass;
        angular_momentum_center += r.wedge(momentum);

        moment_of_inertia_center +=
            (Matrix::with_diagonal([r.norm_squared(); 3]) - r.outer(&r)) * mass;
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
        let body = &microstate.bodies()[body_index];
        if !should_zero_body(body) {
            continue;
        }

        let mut body_properties = body.item.properties.clone();

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
fn zero_angular_momentum_cartesian2<B, S, X, C, F>(
    microstate: &mut Microstate<B, S, X, C>,
    should_zero_body: F,
) where
    B: Position<Position = Cartesian<2>>
        + Mass
        + Momentum<Momentum = Cartesian<2>>
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    F: Fn(&Tagged<Body<B, S>>) -> bool,
{
    let mut center_of_mass = Cartesian::default();
    let mut total_mass = 0.0;

    for body in microstate.bodies() {
        if !should_zero_body(body) {
            continue;
        }

        let position = body.item.properties.position();
        let mass = body.item.properties.mass();

        center_of_mass += *position * mass;
        total_mass += mass;
    }
    center_of_mass /= total_mass;

    let mut angular_momentum_center = 0.0;
    let mut moment_of_inertia_center = 0.0;

    for body in microstate.bodies() {
        if !should_zero_body(body) {
            continue;
        }

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
            let body = &microstate.bodies()[body_index];
            if !should_zero_body(body) {
                continue;
            }

            let mut body_properties = body.item.properties.clone();

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

impl<R, S, X, C> ZeroCenterAngularMomentum<DynamicOrientedPoint<Cartesian<3>, R>, S>
    for Microstate<DynamicOrientedPoint<Cartesian<3>, R>, S, X, C>
where
    R: RotationalMotionTypes,
    DynamicOrientedPoint<Cartesian<3>, R>: Transform<S> + Clone,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<DynamicOrientedPoint<Cartesian<3>, R>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn zero_center_angular_momentum_with_filter<
        F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<3>, R>, S>>) -> bool,
    >(
        &mut self,
        should_zero_body: F,
    ) {
        zero_angular_momentum_cartesian3(self, should_zero_body);
    }
}

impl<S, X, C> ZeroCenterAngularMomentum<DynamicPoint<Cartesian<3>>, S>
    for Microstate<DynamicPoint<Cartesian<3>>, S, X, C>
where
    DynamicPoint<Cartesian<3>>: Transform<S>,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<DynamicPoint<Cartesian<3>>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn zero_center_angular_momentum_with_filter<
        F: Fn(&Tagged<Body<DynamicPoint<Cartesian<3>>, S>>) -> bool,
    >(
        &mut self,
        should_zero_body: F,
    ) {
        zero_angular_momentum_cartesian3(self, should_zero_body);
    }
}

impl<R, E, S, X, C> ZeroCenterAngularMomentum<CustomBodyCartesian3<R, E>, S>
    for Microstate<CustomBodyCartesian3<R, E>, S, X, C>
where
    CustomBodyCartesian3<R, E>: Clone
        + Transform<S>
        + Position<Position = Cartesian<3>>
        + Mass
        + Momentum<Momentum = Cartesian<3>>,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<CustomBodyCartesian3<R, E>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn zero_center_angular_momentum_with_filter<
        F: Fn(&Tagged<Body<CustomBodyCartesian3<R, E>, S>>) -> bool,
    >(
        &mut self,
        should_zero_body: F,
    ) {
        zero_angular_momentum_cartesian3(self, should_zero_body);
    }
}

impl<R, S, X, C> ZeroCenterAngularMomentum<DynamicOrientedPoint<Cartesian<2>, R>, S>
    for Microstate<DynamicOrientedPoint<Cartesian<2>, R>, S, X, C>
where
    R: RotationalMotionTypes,
    DynamicOrientedPoint<Cartesian<2>, R>: Transform<S> + Clone,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<DynamicOrientedPoint<Cartesian<2>, R>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn zero_center_angular_momentum_with_filter<
        F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<2>, R>, S>>) -> bool,
    >(
        &mut self,
        should_zero_body: F,
    ) {
        zero_angular_momentum_cartesian2(self, should_zero_body);
    }
}

impl<S, X, C> ZeroCenterAngularMomentum<DynamicPoint<Cartesian<2>>, S>
    for Microstate<DynamicPoint<Cartesian<2>>, S, X, C>
where
    DynamicPoint<Cartesian<2>>: Transform<S>,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<DynamicPoint<Cartesian<2>>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn zero_center_angular_momentum_with_filter<
        F: Fn(&Tagged<Body<DynamicPoint<Cartesian<2>>, S>>) -> bool,
    >(
        &mut self,
        should_zero_body: F,
    ) {
        zero_angular_momentum_cartesian2(self, should_zero_body);
    }
}

impl<R, E, S, X, C> ZeroCenterAngularMomentum<CustomBodyCartesian2<R, E>, S>
    for Microstate<CustomBodyCartesian2<R, E>, S, X, C>
where
    CustomBodyCartesian2<R, E>: Clone
        + Transform<S>
        + Position<Position = Cartesian<2>>
        + Mass
        + Momentum<Momentum = Cartesian<2>>,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<CustomBodyCartesian2<R, E>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn zero_center_angular_momentum_with_filter<
        F: Fn(&Tagged<Body<CustomBodyCartesian2<R, E>, S>>) -> bool,
    >(
        &mut self,
        should_zero_body: F,
    ) {
        zero_angular_momentum_cartesian2(self, should_zero_body);
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ThermalizeAngularMomentum`

// TODO: add documentation

use super::ThermalizeAngularMomentum;
use hoomd_microstate::{
    Body, Microstate, SiteKey, Tagged, Transform, boundary::{GenerateGhosts, Wrap}, property::{AngularMomentum, DynamicOrientedPoint, MomentOfInertia, Position, RotationalMotionTypes},
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Angle, Cartesian, Versor};
use rand_distr::{Distribution, Normal};

#[inline]
fn thermalize_angular_momentum_with_filter_cartesian2<
    B, S, X, C, F: Fn(&Tagged<Body<B, S>>) -> bool,
>(
    microstate: &mut Microstate<B, S, X, C>,
    temperature: f64,
    should_thermalize_body: F,
)
where
    B: Clone
        + Transform<S>
        + Position<Position = Cartesian<2>>
        + MomentOfInertia<MomentOfInertia = <Angle as RotationalMotionTypes>::MomentOfInertia>
        + AngularMomentum<AngularMomentum = <Angle as RotationalMotionTypes>::AngularMomentum>,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    let mut rng = microstate.counter().make_rng();

    for body_index in 0..microstate.bodies().len() {
        let body = &microstate.bodies()[body_index];
        if !should_thermalize_body(body) {
            continue;
        }

        let mut body_properties = body.item.properties.clone();

        let moment_of_inertia = body_properties.moment_of_inertia();
        let sigma = (temperature * moment_of_inertia).sqrt();
        let normal = Normal::new(0.0, sigma).expect("Normal distribution should be valid");

        *body_properties.angular_momentum_mut() = normal.sample(&mut rng);

        microstate.update_body_properties(body_index, body_properties)
            .expect("Bodies and sites should remain in simulation boundary.");
    }

    microstate.increment_substep();
}

#[inline]
fn thermalize_angular_momentum_with_filter_cartesian3<
    B, S, X, C, F: Fn(&Tagged<Body<B, S>>) -> bool,
>(
    microstate: &mut Microstate<B, S, X, C>,
    temperature: f64,
    should_thermalize_body: F,
)
where
    B: Clone
        + Transform<S>
        + Position<Position = Cartesian<3>>
        + MomentOfInertia<MomentOfInertia = <Versor as RotationalMotionTypes>::MomentOfInertia>
        + AngularMomentum<AngularMomentum = <Versor as RotationalMotionTypes>::AngularMomentum>,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    let mut rng = microstate.counter().make_rng();

    for body_index in 0..microstate.bodies().len() {
        let body = &microstate.bodies()[body_index];
        if !should_thermalize_body(body) {
            continue;
        }

        let mut body_properties = body.item.properties.clone();

        let moment_of_inertia = body_properties.moment_of_inertia();

        let x_nonzero = moment_of_inertia[0] > 0.0;
        let y_nonzero = moment_of_inertia[1] > 0.0;
        let z_nonzero = moment_of_inertia[2] > 0.0;
        let sigma_x = (temperature * moment_of_inertia[0]).sqrt();
        let sigma_y = (temperature * moment_of_inertia[1]).sqrt();
        let sigma_z = (temperature * moment_of_inertia[2]).sqrt();
        let normal_x = Normal::new(0.0, sigma_x).expect("Normal distribution should be valid.");
        let normal_y = Normal::new(0.0, sigma_y).expect("Normal distribution should be valid.");
        let normal_z = Normal::new(0.0, sigma_z).expect("Normal distribution should be valid.");

        let mut angular_momentum_new = Cartesian::<3>::default();

        if x_nonzero {
            angular_momentum_new[0] = normal_x.sample(&mut rng);
        }
        if y_nonzero {
            angular_momentum_new[1] = normal_y.sample(&mut rng);
        }
        if z_nonzero {
            angular_momentum_new[2] = normal_z.sample(&mut rng);
        }

        *body_properties.angular_momentum_mut() = angular_momentum_new;
        microstate.update_body_properties(body_index, body_properties)
            .expect("Bodies and sites should remain in simulation boundary.");
    }

    microstate.increment_substep();
}

/// Thermalize angular momentum for bodies in 2-dimensional cartesian space.
impl<S, X, C> ThermalizeAngularMomentum<DynamicOrientedPoint<Cartesian<2>, Angle>, S>
    for Microstate<DynamicOrientedPoint<Cartesian<2>, Angle>, S, X, C>
where
    DynamicOrientedPoint<Cartesian<2>, Angle>: Clone + Transform<S>,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<DynamicOrientedPoint<Cartesian<2>, Angle>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn thermalize_angular_momentum_with_filter<
        F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<2>, Angle>, S>>) -> bool,
    >(
        &mut self,
        temperature: f64,
        should_thermalize_body: F,
    ) {
        thermalize_angular_momentum_with_filter_cartesian2(
            self,
            temperature,
            should_thermalize_body
        );
    }
}

/// Thermalize angular momentum for bodies in 3-dimensional cartesian space.
impl<S, X, C> ThermalizeAngularMomentum<DynamicOrientedPoint<Cartesian<3>, Versor>, S>
    for Microstate<DynamicOrientedPoint<Cartesian<3>, Versor>, S, X, C>
where
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<DynamicOrientedPoint<Cartesian<3>, Versor>> + Wrap<S> + GenerateGhosts<S>,
    DynamicOrientedPoint<Cartesian<3>, Versor>: Transform<S>,
{
    #[inline]
    fn thermalize_angular_momentum_with_filter<
        F: Fn(&Tagged<Body<DynamicOrientedPoint<Cartesian<3>, Versor>, S>>) -> bool,
    >(
        &mut self,
        temperature: f64,
        should_thermalize_body: F,
    ) {
        thermalize_angular_momentum_with_filter_cartesian3(
            self,
            temperature,
            should_thermalize_body
        );
    }
}

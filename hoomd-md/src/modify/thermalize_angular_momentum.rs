// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ThermalizeAngularMomentum`

use super::ThermalizeAngularMomentum;
use hoomd_microstate::{
    Body, Microstate, SiteKey, Tagged, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{AngularMomentum, DynamicOrientedPoint, MomentOfInertia, Position},
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Angle, Cartesian, Outer, Versor, Wedge};
use rand_distr::{Distribution, Normal};

impl<P, S, X, C> ThermalizeAngularMomentum<DynamicOrientedPoint<P, Angle>, S>
    for Microstate<DynamicOrientedPoint<P, Angle>, S, X, C>
where
    P: Copy + Wedge + Outer,
    DynamicOrientedPoint<P, Angle>: Clone + Transform<S>,
    S: Position<Position = P> + Default,
    X: PointUpdate<P, SiteKey>,
    C: Wrap<DynamicOrientedPoint<P, Angle>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn thermalize_angular_momentum_with_filter<
        F: Fn(&Tagged<Body<DynamicOrientedPoint<P, Angle>, S>>) -> bool,
    >(
        &mut self,
        temperature: f64,
        should_thermalize_body: F,
    ) {
        let mut rng = self.counter().make_rng();

        for body_index in 0..self.bodies().len() {
            let body = &self.bodies()[body_index];
            if !should_thermalize_body(body) {
                continue;
            }

            let mut body_properties = body.item.properties.clone();

            let moment_of_inertia = body_properties.moment_of_inertia();
            let sigma = (temperature * moment_of_inertia).sqrt();
            let normal = Normal::new(0.0, sigma).expect("Normal distribution should be valid");

            *body_properties.angular_momentum_mut() = normal.sample(&mut rng);

            self.update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        self.increment_substep();
    }
}

impl<P, S, X, C> ThermalizeAngularMomentum<DynamicOrientedPoint<P, Versor>, S>
    for Microstate<DynamicOrientedPoint<P, Versor>, S, X, C>
where
    P: Copy + Wedge + Outer,
    DynamicOrientedPoint<P, Versor>: Clone + Transform<S>,
    S: Position<Position = P> + Default,
    X: PointUpdate<P, SiteKey>,
    C: Wrap<DynamicOrientedPoint<P, Versor>> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn thermalize_angular_momentum_with_filter<
        F: Fn(&Tagged<Body<DynamicOrientedPoint<P, Versor>, S>>) -> bool,
    >(
        &mut self,
        temperature: f64,
        should_thermalize_body: F,
    ) {
        let mut rng = self.counter().make_rng();

        for body_index in 0..self.bodies().len() {
            let body = &self.bodies()[body_index];
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
            self.update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        self.increment_substep();
    }
}

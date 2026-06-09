// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `RotationalKineticEnergy`

use super::RotationalKineticEnergy;
use hoomd_microstate::{
    Body, Microstate, Tagged, property::DynamicOrientedPoint
};
use hoomd_vector::{Angle, Versor, Wedge};

impl<P, S, X, C> RotationalKineticEnergy<DynamicOrientedPoint<P, Angle>, S> for Microstate<DynamicOrientedPoint<P, Angle>, S, X, C>
where P: Wedge
{
    #[inline]
    fn rotational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<DynamicOrientedPoint<P, Angle>, S>>) -> bool>(&self, should_sum_body: F)
        -> (f64, usize) {
        self.bodies()
            .iter()
            .filter(|&body| should_sum_body(body))
            .fold((0.0, 0), |(total, count), body| {
                let moment_of_inertia = body.item.properties.moment_of_inertia;
                let angular_momentum = body.item.properties.angular_momentum;

                if moment_of_inertia > 0.0 {
                    (angular_momentum.powi(2) / (2.0 * moment_of_inertia), count + 1)
                } else {
                    (total, count)
                }
            })
    }
}

impl<P, S, X, C> RotationalKineticEnergy<DynamicOrientedPoint<P, Versor>, S> for Microstate<DynamicOrientedPoint<P, Versor>, S, X, C>
where P: Wedge
{
    #[inline]
    fn rotational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<DynamicOrientedPoint<P, Versor>, S>>) -> bool>(&self, should_sum_body: F)
        -> (f64, usize) {
        self.bodies()
            .iter()
            .filter(|&body| should_sum_body(body))
            .fold((0.0, 0), |(mut total, mut count), body| {
                let moment_of_inertia = body.item.properties.moment_of_inertia;
                let angular_momentum = body.item.properties.angular_momentum;

                for i in 0..3 {
                    if moment_of_inertia[i] > 0.0 {
                        total += angular_momentum[i].powi(2) / (2.0 * moment_of_inertia[0]);
                        count += 1;
                    }
                }

                (total, count)
            })
    }
}

// TODO: Test.

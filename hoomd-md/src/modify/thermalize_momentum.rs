// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ThermalizeMomentum`

use std::array;

use super::ThermalizeMomentum;
use hoomd_microstate::{
    Body, Microstate, SiteKey, Tagged, Transform, boundary::{GenerateGhosts, Wrap}, property::{Mass, Momentum, Position}
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::Cartesian;
use rand_distr::{Distribution, Normal};

impl<const N: usize, B, S, X, C> ThermalizeMomentum<B, S> for Microstate<B, S, X, C>
where
    B: Position<Position = Cartesian<N>>
        + Momentum<Momentum = Cartesian<N>>
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<N>> + Default,
    X: PointUpdate<Cartesian<N>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn thermalize_momentum_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(&mut self, temperature: f64, should_thermalize_body: F) {
        let mut rng = self.counter().make_rng();

        for body_index in 0..self.bodies().len() {
            let body = &self.bodies()[body_index];
            if !should_thermalize_body(body) {
                continue;
            }
            
            let mut body_properties = body.item.properties.clone();
            let mass = body_properties.mass();
            let sigma = (temperature * mass).sqrt();
            let normal = Normal::new(0.0, sigma).expect("Normal distribution should be valid");

            if mass > 0.0 {
                let random_momentum: Cartesian<N> =
                    array::from_fn(|_| normal.sample(&mut rng)).into();
                *body_properties.momentum_mut() = random_momentum;

                self
                    .update_body_properties(body_index, body_properties)
                    .expect("Bodies and sites should remain in simulation boundary.");
            }
        }

        self.increment_substep();
    }
}

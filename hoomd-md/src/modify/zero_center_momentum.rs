// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ZeroCenterMomentum`

use super::ZeroCenterMomentum;
use hoomd_microstate::{
    Body, Microstate, SiteKey, Tagged, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{Momentum, Position},
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::Cartesian;

impl<const N: usize, B, S, X, C> ZeroCenterMomentum<B, S> for Microstate<B, S, X, C>
where
    B: Position<Position = Cartesian<N>> + Momentum<Momentum = Cartesian<N>> + Transform<S> + Clone,
    S: Position<Position = Cartesian<N>> + Default,
    X: PointUpdate<Cartesian<N>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn zero_center_momentum_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        should_zero_body: F,
    ) {
        let (total_momentum, count) = self
            .bodies()
            .iter()
            .filter(|&body| should_zero_body(body))
            .fold((Cartesian::default(), 0), |(total, count), body| {
                (total + *body.item.properties.momentum(), count + 1)
            });
        let average_momentum = total_momentum / f64::from(count);

        for body_index in 0..self.bodies().len() {
            let body = &self.bodies()[body_index];
            if !should_zero_body(body) {
                continue;
            }

            let mut body_properties = body.item.properties.clone();

            *body_properties.momentum_mut() -= average_momentum;

            self.update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}

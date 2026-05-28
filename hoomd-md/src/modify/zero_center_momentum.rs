// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ZeroCenterMomentum`

use super::ZeroCenterMomentum;
use hoomd_microstate::{
    Microstate, SiteKey, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{Momentum, Position},
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::Cartesian;

impl<const N: usize, B, S, X, C> ZeroCenterMomentum for Microstate<B, S, X, C>
where
    B: Position<Position = Cartesian<N>>
        + Momentum<Momentum = Cartesian<N>>
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<N>> + Default,
    X: PointUpdate<Cartesian<N>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    #[inline]
    fn zero_center_momentum(&mut self) {
        let total_momentum = self.bodies().iter().fold(Cartesian::default(), |total, body| total + *body.item.properties.momentum());
        let average_momentum = total_momentum / (self.bodies().len() as f64);
        
        for body_index in 0..self.bodies().len() {
            let mut body_properties = self.bodies()[body_index].item.properties.clone();

            *body_properties.momentum_mut()-= average_momentum;

            self
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `TranslationalKineticEnergy`

use super::TranslationalKineticEnergy;
use hoomd_microstate::{
    Body, Microstate, Tagged, property::{Mass, Momentum}
};
use hoomd_vector::{InnerProduct};

impl<V, B, S, X, C> TranslationalKineticEnergy<B, S> for Microstate<B, S, X, C>
where
    V: InnerProduct,
    B: Momentum<Momentum = V>
       + Mass,
{
    #[inline]
    fn translational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(&self, should_sum: F)
        -> (f64, usize) {
        self.bodies()
            .iter()
            .filter(|&body| should_sum(body))
            .fold((0.0, 0), |(total, count), body| {
                let p = body.item.properties.momentum();
                (total + p.norm_squared() / (2.0 * body.item.properties.mass()), count + p.n_dimensions())
            })
    }
}

// TODO: Test

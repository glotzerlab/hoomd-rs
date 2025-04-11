// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Translate
*/

use super::LocalTrial;
use hoomd_microstate::property::Position;
use hoomd_vector::Vector;

use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use std::marker::PhantomData;

pub struct Translate<V> {
    pub maximum_distance: f64,
    vector_type: PhantomData<V>,
}

impl<V> Translate<V> {
    pub fn with_maximum_distance(maximum_distance: f64) -> Self {
        Self {
            maximum_distance,
            vector_type: PhantomData,
        }
    }
}

impl<B, V> LocalTrial<B> for Translate<V>
where
    B: Position<V>,
    V: Vector,
    StandardUniform: Distribution<V>,
{
    #[inline]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B {
        let mut trial = body_properties;

        // TODO: Replace draw vector from the ball with radius maximum_distance
        // Implement the Ball distribution in hoomd-vector
        let delta_r = rng.random::<V>() * self.maximum_distance;
        *trial.position_mut() += delta_r;

        trial
    }
}

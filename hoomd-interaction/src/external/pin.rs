// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`PinOrientations`]

use hoomd_microstate::{
    Body, Microstate,
    boundary::GenerateGhosts,
    property::{Orientation, Position},
};
use hoomd_vector::{InnerProduct, RotationalDistance};

use crate::{DeltaEnergyOne, TotalEnergy};

pub struct PinOrientations<R> {
    pub k: f64,
    pub reference: Vec<Vec<R>>,
}


impl<R> PinOrientations<R>
{
    #[inline]
    pub fn body_energy<B, S>(&self, body_tag: usize, body: &Body<B, S>) -> f64
    where
        R: RotationalDistance,
        B: Orientation<Rotation = R>,
    {
        self.reference[body_tag].iter()
            .map(|reference_orientation| 0.5 * self.k * body.properties.orientation().rotational_distance(reference_orientation).powi(2))
            .reduce(f64::min)
            .unwrap_or(0.0)
    }
}

impl<B, S, X, C, R> DeltaEnergyOne<B, S, X, C> for PinOrientations<R>
where
    R: RotationalDistance,
    B: Orientation<Rotation = R>,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let tagged_body = &initial_microstate.bodies()[body_index];
        let body_tag = tagged_body.tag;
        
        let energy_initial = self.body_energy(body_tag, &tagged_body.item);
        let energy_final = self.body_energy(body_tag, final_body);

        energy_final - energy_initial
    }
}

impl<B, S, X, C, R> TotalEnergy<Microstate<B, S, X, C>> for PinOrientations<R>
where
    R: RotationalDistance,
    B: Orientation<Rotation = R>,
{
    /// Total energy over all bodies.
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, X, C>) -> f64 {
        microstate.bodies()
            .iter()
            .map(|tagged_body| self.body_energy(tagged_body.tag, &tagged_body.item))
            .sum()
    }
}

pub struct PinPositions<V> {
    pub k: f64,
    pub reference: Vec<V>,
}

impl<V> PinPositions<V>
where
    V: InnerProduct,
{
    #[inline]
    pub fn body_energy<B, S, C>(
        &self,
        body_tag: usize,
        body: &Body<B, S>,
        boundary: &C,
    ) -> f64
    where
        B: Position<Position = V>,
        C: GenerateGhosts<B>,
    {
        let reference = &self.reference[body_tag];
        let ghosts = boundary.generate_ghosts(&body.properties);
        std::iter::once(body.properties.position())
            .chain(ghosts.iter().map(hoomd_microstate::property::Position::position))
            .map(|position| 0.5 * self.k * (*position - *reference).norm_squared())
            .reduce(f64::min)
            .unwrap_or(0.0)
    }
}

impl<B, S, X, C, V> DeltaEnergyOne<B, S, X, C> for PinPositions<V>
where
    B: Position<Position = V>,
    C: GenerateGhosts<B>,
    V: InnerProduct,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let tagged_body = &initial_microstate.bodies()[body_index];
        let body_tag = tagged_body.tag;
        let boundary = initial_microstate.boundary();

        let energy_initial = self.body_energy(body_tag, &tagged_body.item, boundary);
        let energy_final = self.body_energy(body_tag, final_body, boundary);

        energy_final - energy_initial
    }
}

impl<B, S, X, C, V> TotalEnergy<Microstate<B, S, X, C>> for PinPositions<V>
where
    B: Position<Position = V>,
    C: GenerateGhosts<B>,
    V: InnerProduct,
{
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, X, C>) -> f64 {
        let boundary = microstate.boundary();
        microstate.bodies()
            .iter()
            .map(|tagged_body| self.body_energy(tagged_body.tag, &tagged_body.item, boundary))
            .sum()
    }
}

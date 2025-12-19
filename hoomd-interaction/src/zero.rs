// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Zero`

use super::{DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, TotalEnergy};

use hoomd_microstate::{Body, Microstate};

/// Set the energy of any system to 0.
///
/// *hoomd-rs* uses [`Zero`] in minimal examples that demonstrate MC simulations.
/// It returns 0 for all energies and delta energies.
pub struct Zero;

impl<M> TotalEnergy<M> for Zero {
    #[inline]
    fn total_energy(&self, _microstate: &M) -> f64 {
        0.0
    }
}

impl<B, S, X, C> DeltaEnergyOne<B, S, X, C> for Zero {
    #[inline]
    fn delta_energy_one(
        &self,
        _initial_microstate: &Microstate<B, S, X, C>,
        _body_index: usize,
        _final_body: &Body<B, S>,
    ) -> f64 {
        0.0
    }
}

impl<B, S, X, C> DeltaEnergyInsert<B, S, X, C> for Zero {
    #[inline]
    fn delta_energy_insert(
        &self,
        _initial_microstate: &Microstate<B, S, X, C>,
        _new_body: &Body<B, S>,
    ) -> f64 {
        0.0
    }
}

impl<B, S, X, C> DeltaEnergyRemove<B, S, X, C> for Zero {
    #[inline]
    fn delta_energy_remove(
        &self,
        _initial_microstate: &Microstate<B, S, X, C>,
        _body_index: usize,
    ) -> f64 {
        0.0
    }
}

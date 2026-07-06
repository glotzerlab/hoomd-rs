// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Zero`

use serde::{Deserialize, Serialize};

use hoomd_vector::{Outer, Wedge};
use hoomd_microstate::{Body, Microstate, property::Position};

use super::{NetSiteForceAndVirial, NetSiteForceVirialAndTorque, DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, TotalEnergy};

/// Hamiltonian with H = 0.
///
/// *hoomd-rs* uses [`Zero`] in minimal examples. It evaluates to 0 for all
/// forces, virials, torques, energies, and delta energies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

impl<V, B, S, X, C> NetSiteForceVirialAndTorque<B, S, X, C> for Zero
where
    V: Default + Wedge + Outer,
    V::Bivector: Default,
    S: Position<Position = V>,
    V::Tensor: Default
{
    type Force = V;

    #[inline]
    fn net_site_force_virial_and_torque(
        &self,
        _microstate: &Microstate<B, S, X, C>,
        _site_index: usize
    ) -> (V, V::Tensor, V::Bivector) {
        (V::default(), V::Tensor::default(), V::Bivector::default())
    }
}

impl<V, B, S, X, C> NetSiteForceAndVirial<B, S, X, C> for Zero
where
    V: Default + Outer,
    S: Position<Position = V>,
    V::Tensor: Default
{
    type Force = V;

    #[inline]
    fn net_site_force_and_virial(
        &self,
        _microstate: &Microstate<B, S, X, C>,
        _site_index: usize
    ) -> (V, V::Tensor) {
        (V::default(), V::Tensor::default())
    }
}

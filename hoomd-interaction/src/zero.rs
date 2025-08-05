// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `Zero`
*/

use super::DeltaEnergyOne;

use hoomd_microstate::{Body, Microstate};

/** Set the energy of any system to 0.

*hoomd-rs* uses [`Zero`] in minimal examples that demonstrate MC simulations.
It returns 0 for all delta energies.
*/
pub struct Zero;

impl<B, S, C> DeltaEnergyOne<B, S, C> for Zero {
    #[inline]
    fn delta_energy_one(
        &self,
        _initial_microstate: &Microstate<B, S, C>,
        _body_index: usize,
        _final_body: &Body<B, S>,
    ) -> f64 {
        0.0
    }
}

// TODO: Implement Zero for other traits in hoomd-interaction

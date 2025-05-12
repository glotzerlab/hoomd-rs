// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `DeltaEnergy*` for external potentials.
*/

use super::DeltaEnergyOne;
use hoomd_interaction::{Single, SiteEnergy};
use hoomd_microstate::{Body, Microstate, Transform};

impl<V, B, S, C, E> DeltaEnergyOne<V, B, S, C> for Single<E>
where
    E: SiteEnergy<S>,
    B: Transform<S>,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<V, B, S, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let energy_final = final_body.sites.iter().fold(0.0, |total, s| {
            let new_site = final_body.properties.transform(s);
            // TODO: boundary conditions
            // TODO: What is the energy if a site cannot be wrapped? infinite?
            total + self.site_energy(&new_site)
        });

        let energy_initial = initial_microstate
            .iter_body_sites(body_index)
            .fold(0.0, |total, s| total + self.site_energy(&s.properties));

        energy_final - energy_initial
    }
}

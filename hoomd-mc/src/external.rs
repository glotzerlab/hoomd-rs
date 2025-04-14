// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `DeltaEnergy*` for external potentials.
*/

use super::DeltaEnergyOne;
use hoomd_interaction::{Single, SiteEnergy};
use hoomd_microstate::{Body, Microstate, Tagged, Transform};

impl<B, S, C, E> DeltaEnergyOne<B, S, C> for Single<E>
where
    E: SiteEnergy<S>,
    B: Transform<S>,
{
    #[inline]
    fn delta_energy_one(
        &self,
        microstate: &Microstate<B, S, C>,
        new_body: &Tagged<Body<B, S>>,
    ) -> f64 {
        let energy_final = new_body.item.sites.iter().fold(0.0, |total, s| {
            let new_site = new_body.item.properties.transform(s);
            // TODO: boundary conditions
            total + self.inner.site_energy(&new_site)
        });

        let energy_initial = 0.0; // TODO
        // - self.inner.site_energy(&old_body.item.properties);
        energy_final - energy_initial
    }
}

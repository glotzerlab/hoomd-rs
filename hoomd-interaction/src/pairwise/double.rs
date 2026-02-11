// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `SitePairEnergy` for tuples

use crate::{MaximumInteractionRange, SitePairEnergy};

impl <A, B, S> SitePairEnergy<S> for (A, B) where
A: SitePairEnergy<S>,
B: SitePairEnergy<S>,
{
    /// TODO: Docs and example
    #[inline]
    fn site_pair_energy(&self, site_properties_i: &S, site_properties_j: &S) -> f64 {
        let site_pair_energy_a = self.0.site_pair_energy(site_properties_i, site_properties_j);
        if site_pair_energy_a == f64::INFINITY {
            return site_pair_energy_a;
        }

        let site_pair_energy_b = self.1.site_pair_energy(site_properties_i, site_properties_j);
        site_pair_energy_a + site_pair_energy_b
    }

    #[inline]
    fn site_pair_energy_initial(&self, site_properties_i: &S, site_properties_j: &S) -> f64 {
        let site_pair_energy_initial_a = self.0.site_pair_energy_initial(site_properties_i, site_properties_j);
        if site_pair_energy_initial_a == f64::INFINITY {
            return site_pair_energy_initial_a;
        }

        let site_pair_energy_initial_b = self.1.site_pair_energy_initial(site_properties_i, site_properties_j);
        site_pair_energy_initial_a + site_pair_energy_initial_b
    }

    #[inline]
    fn is_only_infinite_or_zero() -> bool {
        A::is_only_infinite_or_zero() && B::is_only_infinite_or_zero()
    }

}

impl<A, B> MaximumInteractionRange for (A,B)
where
    A: MaximumInteractionRange,
    B: MaximumInteractionRange,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.0.maximum_interaction_range().max(self.1.maximum_interaction_range())
    }
}

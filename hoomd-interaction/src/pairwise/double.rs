// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `SitePairEnergy` for tuples

use crate::{MaximumInteractionRange, SitePairEnergy};

/// Sum two site pair energy terms.
///
/// Use a tuple to combine two site pair energy terms and evaluate them efficiently
/// in one loop over neighbors by `PairwiseCutoff`.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{PairwiseCutoff, pairwise::{AngularMask, Anisotropic, HardSphere, angular_mask::Patch},
/// univariate::Boxcar};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
///
/// let hard_disk = HardSphere { diameter: 1.0 };
///
/// let patch_interaction_range = 1.12;
/// let boxcar = Boxcar {
///     epsilon: -5.8,
///     left: 0.0,
///     right: patch_interaction_range,
/// };
/// let masks = [Patch {
///     director: [0.0, 1.0].try_into()?,
///     cos_delta: 37.0_f64.cos(),
/// },Patch {
///     director: [0.0, -1.0].try_into()?,
///     cos_delta: 37.0_f64.cos(),
/// }];
/// let angular_mask = Anisotropic { interaction: AngularMask::new(boxcar, masks), r_cut: patch_interaction_range };
///
/// let hamiltonian = PairwiseCutoff((hard_disk, angular_mask));
/// # Ok(())
/// # }
/// ```
impl <A, B, S> SitePairEnergy<S> for (A, B) where
A: SitePairEnergy<S>,
B: SitePairEnergy<S>,
{
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

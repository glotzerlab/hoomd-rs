// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement CurvedIsotropic
*/

use hoomd_interaction::{pairwise::IsotropicEnergy, SitePairEnergy};
use hoomd_microstate::property::Position;
use hoomd_vector::Vector;
use crate::CurvedManifold;

/** Compute isotropic properties from a pair of sites on a curved manifold
*/
pub struct CurvedIsotropic<E>(pub E, pub f64);

impl<V, S, E> SitePairEnergy<S> for CurvedIsotropic<E>
where
    S: Position<Vector = V>,
    V: Vector + CurvedManifold,
    E: IsotropicEnergy,
{
    #[inline]
    fn site_pair_energy(&self, a: &S, b: &S) -> f64 {
        self.0.energy((a.position()).geodesic_distance(b.position(), self.1))
    }
}

impl<E> IsotropicEnergy for CurvedIsotropic<E>
where
    E: IsotropicEnergy,
{
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.0.energy(r)
    }
}

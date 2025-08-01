// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Curved Isotropic
*/

use crate::CurvedManifold;
use hoomd_interaction::{SitePairEnergy, pairwise::IsotropicEnergy};
use hoomd_microstate::property::Position;
use hoomd_vector::Vector;

/** Compute isotropic properties from a pair of sites on a curved manifold, embedded
in some vector space. Function is similar to [`Isotropic`],
but the metric passed to the energy function is from the embedded manifold.
*/
pub struct CurvedIsotropic<E, M> {
    /// The isotropic potential
    pub isotropic: E,
    /// the manifold type
    pub manifold: M,
}

impl<V, S, E, M> SitePairEnergy<S> for CurvedIsotropic<E, M>
where
    S: Position<Vector = V>,
    V: Vector,
    E: IsotropicEnergy,
    M: CurvedManifold,
{
    #[inline]
    fn site_pair_energy(&self, a: &S, b: &S) -> f64 {
        let site_a: M = CurvedManifold::to_manifold(a.position().to_vec());
        let site_b: M = CurvedManifold::to_manifold(b.position().to_vec());
        self.isotropic.energy(site_a.geodesic_distance(&site_b))
    }
}

impl<E, M> IsotropicEnergy for CurvedIsotropic<E, M>
where
    E: IsotropicEnergy,
    M: CurvedManifold,
{
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.isotropic.energy(r)
    }
}

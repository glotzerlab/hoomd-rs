// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Isotropic

use serde::{Deserialize, Serialize};

use crate::{MaximumInteractionRange, SitePairEnergy, univariate::UnivariateEnergy};
use hoomd_microstate::property::Position;
use hoomd_vector::{InnerProduct, Vector, Metric};

/// Compute isotropic interactions between a pair of sites.
///
/// [`Isotropic`] provides a single implementation that computes pairwise
/// interactions that are a function only of the distance between sites. It
/// fills the gap between traits like [`SitePairEnergy`] which operates on
/// site properties and [`UnivariateEnergy`] which is a function only of the
/// separation distance.
///
/// Use [`Isotropic`] with [`PairwiseCutoff`] in MD and MC simulations.
///
/// [`PairwiseCutoff`]: crate::PairwiseCutoff
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     SitePairEnergy, pairwise::Isotropic, univariate::LennardJones,
/// };
/// use hoomd_microstate::property::Point;
/// use hoomd_vector::Cartesian;    
/// use approxim::assert_relative_eq;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let a = Point {
///     position: Cartesian::from([0.0, 0.0]),
/// };
/// let b = Point {
///     position: Cartesian::from([0.0, 2.0 * 2.0_f64.powf(1.0 / 6.0)]),
/// };
///
/// let lennard_jones: LennardJones = LennardJones {
///     epsilon: 1.5,
///     sigma: 2.0,
/// };
/// let lennard_jones = Isotropic {
///     interaction: lennard_jones,
///     r_cut: 2.5,
/// };
///
/// let energy = lennard_jones.site_pair_energy(&a, &b);
/// let force_ab = lennard_jones.site_pair_force(&a, &b);
/// let force_ba = lennard_jones.site_pair_force(&b, &a);
/// 
/// assert_eq!(energy, -1.5);
/// assert_eq!(force_ab, -force_ba);
/// assert_relative_eq!(force_ab, Cartesian::from([0.0, 0.0]), epsilon = 1e-14);
/// 
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Isotropic<E> {
    /// The site-site interaction.
    pub interaction: E,
    /// Maximum distance between two interacting sites.
    pub r_cut: f64,
}

impl<P, S, E> SitePairEnergy<S> for Isotropic<E>
where
    S: Position<Position = P>,
    P: Metric,
    E: UnivariateEnergy,
{
    #[inline]
    fn site_pair_energy(&self, site_properties_i: &S, site_properties_j: &S) -> f64 {
        let r = site_properties_i
            .position()
            .distance(site_properties_j.position());
        if r >= self.r_cut {
            return 0.0;
        }

        self.interaction.energy(r)
    }
}

impl<E> MaximumInteractionRange for Isotropic<E> {
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.r_cut
    }
}

impl<P, S, E> SitePairForce<P, S> for Isotropic<E>
where
    E: IsotropicForce,
    P: Vector + InnerProduct + Metric,
    S: Position<Position = P>
{
    /// Calculate the pairwise force 
    /// 
    /// ```math
    /// \begin{equation}
    /// \mathbf{f}_{\alpha\beta} = -\nabla_{r_{\alpha\beta}} U(r_{\alpha\beta})
    /// \end{equation}
    /// ```
    /// 
    /// on [`Site`](hoomd_microstate::Site) $`\alpha`$ exerting by $`\beta`$.
    #[inline]
    fn site_pair_force(&self, a: &S, b: &S) -> P {
        let r = *a.position() - *b.position();
        let distance = r.norm();
        r * self.0.force(distance) / distance
    }
}

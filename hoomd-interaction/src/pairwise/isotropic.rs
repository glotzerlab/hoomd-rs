// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Isotropic

use serde::{Deserialize, Serialize};

use crate::{MaximumInteractionRange, SitePairEnergy, SitePairForce, univariate::{UnivariateEnergy, UnivariateForce}};
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

impl<V, S, E> SitePairForce<S> for Isotropic<E>
where
    E: UnivariateForce,
    V: InnerProduct,
    S: Position<Position = V>
{
    type Force = V;

    /// Calculate the pairwise force on site `a` exerted by site `b`.
    ///
    ///
    /// Isotropic forces always act along the radial direction:
    /// ```math
    /// \begin{equation}
    /// \vec{F} = -\frac{\mathrm{d} U}{\mathrm{d} r} \biggr\rvert_{r=r_{ab}} \hat{r}_{ab}
    /// \end{equation}
    /// ```
    ///
    // TODO: Signs differ between equation and implementation. Should this instead compute
    // the force on `b`?
    #[inline]
    fn site_pair_force(&self, a: &S, b: &S) -> V {
        let r = *a.position() - *b.position();
        let distance = r.norm();
        r * self.interaction.force(distance) / distance
    }
}

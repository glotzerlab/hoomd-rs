// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Isotropic

use serde::{Deserialize, Serialize};

use crate::{MaximumInteractionRange, SitePairEnergy, SitePairForceAndVirial, SitePairForceVirialAndTorque, univariate::{UnivariateEnergy, UnivariateForce}};
use hoomd_microstate::property::Position;
use hoomd_vector::{InnerProduct, Metric, Outer, Wedge};

/// Compute isotropic interactions between a pair of sites.
///
/// [`Isotropic`] provides a single implementation that computes pairwise
/// interactions that are a function only of the distance between sites. It
/// fills the gap between traits like [`SitePairEnergy`] / [`SitePairForce`]
/// which operate on site properties and [`UnivariateEnergy`] /
/// [`UnivariateForce`] which is a function only of the separation distance.
///
/// [`Isotropic`] cuts all interactions off when the distance between sites
/// is greater than or equal to `r_cut`:
/// ```math
/// U_{ij} =
/// \begin{cases}
/// U(r_{ij}) & r_{ij} < r_\mathrm{cut} \\
/// 0 & r_{ij} \ge r_\mathrm{cut}
/// \end{cases}
/// ```
/// ```math
/// \vec{F_{ij}} =
/// \begin{cases}
/// -\frac{\mathrm{d} U}{\mathrm{d} r} \biggr\rvert_{r=r_{ji}} \hat{r}_{ji} & r_{ij} < r_\mathrm{cut} \\
/// \vec{0} & r_{ij} \ge r_\mathrm{cut}
/// \end{cases}
/// ```
/// where $` U `$ is given by `E`'s [`UnivariateEnergy`] implementation and 
/// $` -\frac{\mathrm{d} U}{\mathrm{d} r} `$ is given by `E`'s [`UnivariateForce`]
/// implementation. 
///
/// Use [`Isotropic`] with [`PairwiseCutoff`] in MD and MC simulations.
///
/// [`PairwiseCutoff`]: crate::PairwiseCutoff
/// # Example
///
/// ```
/// use approxim::assert_relative_eq;
/// use hoomd_interaction::{
///     SitePairEnergy, SitePairForce, pairwise::Isotropic, univariate::LennardJones,
/// };
/// use hoomd_microstate::property::Point;
/// use hoomd_vector::Cartesian;    
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
    /// Compute the energy contribution from a pair of sites.
    ///
    /// ```math
    /// U_{ij} =
    /// \begin{cases}
    /// U(r_{ij}) & r_{ij} < r_\mathrm{cut} \\
    /// 0 & r_{ij} \ge r_\mathrm{cut}
    /// \end{cases}
    /// ```
    /// where $` U `$ is given by `E`'s [`UnivariateEnergy`] implementation. 
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_interaction::{
    ///     SitePairEnergy, pairwise::Isotropic, univariate::LennardJones,
    /// };
    /// use hoomd_microstate::property::Point;
    /// use hoomd_vector::Cartesian;    
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
    /// 
    /// assert_eq!(energy, -1.5);
    /// 
    /// # Ok(())
    /// # }
    /// ```
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
    /// The maximum interaction range for `Isotropic` is the given `r_cut`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::{MaximumInteractionRange,
    ///     pairwise::Isotropic, univariate::LennardJones,
    /// };
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lennard_jones: LennardJones = LennardJones {
    ///     epsilon: 1.5,
    ///     sigma: 2.0,
    /// };
    /// let lennard_jones = Isotropic {
    ///     interaction: lennard_jones,
    ///     r_cut: 2.5,
    /// };
    ///
    /// assert_eq!(lennard_jones.maximum_interaction_range(), 2.5);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.r_cut
    }
}

impl<V, S, E> SitePairForceAndVirial<S> for Isotropic<E>
where
    V: Default + InnerProduct + Outer,
    S: Position<Position = V>,
    E: UnivariateForce,
    <V as Outer>::Output: Default
{
    type Force = V;

    /// Evaluate the force and virial on site `i` caused by site `j`.
    ///
    /// Isotropic forces always act along the radial direction:
    /// ```math
    /// \vec{F_{ij}} =
    /// \begin{cases}
    /// -\frac{\mathrm{d} U}{\mathrm{d} r} \biggr\rvert_{r=r_{ji}} \hat{r}_{ji} & r_{ij} < r_\mathrm{cut} \\
    /// \vec{0} & r_{ij} \ge r_\mathrm{cut}
    /// \end{cases}
    /// ```
    /// where $` -\frac{\mathrm{d} U}{\mathrm{d} r} `$ is given by `E`'s [`UnivariateForce`]
    /// implementation. 
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_interaction::{
    ///     SitePairForce, pairwise::Isotropic, univariate::LennardJones,
    /// };
    /// use hoomd_microstate::property::Point;
    /// use hoomd_vector::Cartesian;    
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
    /// let (force_ab, virial_ab) = lennard_jones.site_pair_force_and_virial(&a, &b);
    /// let (force_ba, virial_ba) = lennard_jones.site_pair_force_and_virial(&b, &a);
    /// 
    /// assert_eq!(force_ab, -force_ba);
    /// assert_relative_eq!(force_ab, Cartesian::from([0.0, 0.0]), epsilon = 1e-14);
    /// todo!() // add virial check
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn site_pair_force_and_virial(
        &self,
        site_properties_i: &S,
        site_properties_j: &S
    ) -> (Self::Force, <Self::Force as Outer>::Output) {
        let r_ji = *site_properties_i.position() - *site_properties_j.position();
        let distance = r_ji.norm();

        if distance >= self.r_cut {
            (V::default(), <V as Outer>::Output::default())
        } else {
            let force = (r_ji / distance) * self.interaction.force(distance);
            let virial = force.outer(&r_ji);
            (force, virial)
        }

    }
}

impl<V, S, E> SitePairForceVirialAndTorque<S> for Isotropic<E>
where
    V: Default + InnerProduct + Wedge + Outer,
    V::Bivector: Default,
    S: Position<Position = V>,
    E: UnivariateForce,
    <V as Outer>::Output: Default,
{
    type Force = V;

    /// Evaluate the force, virial, and torque on site `i` caused by site `j`.
    ///
    /// Isotropic forces always act along the radial direction:
    /// ```math
    /// \vec{F_{ij}} =
    /// \begin{cases}
    /// -\frac{\mathrm{d} U}{\mathrm{d} r} \biggr\rvert_{r=r_{ji}} \hat{r}_{ji} & r_{ij} < r_\mathrm{cut} \\
    /// \vec{0} & r_{ij} \ge r_\mathrm{cut}
    /// \end{cases}
    /// ```
    /// where $` -\frac{\mathrm{d} U}{\mathrm{d} r} `$ is given by `E`'s [`UnivariateForce`]
    /// implementation. 
    ///
    /// Radial forces produce zero torque.
    #[inline]
    fn site_pair_force_virial_and_torque(
        &self,
        site_properties_i: &S,
        site_properties_j: &S
    ) -> (V, <Self::Force as Outer>::Output, V::Bivector) {
        let r_ji = *site_properties_i.position() - *site_properties_j.position();
        let distance = r_ji.norm();

        if distance >= self.r_cut {
            (V::default(), <V as Outer>::Output::default(), V::Bivector::default())
        } else {
            let force = (r_ji / distance) * self.interaction.force(distance);
            let virial = force.outer(&r_ji);
            let torque = V::Bivector::default();
            (force, virial, torque)
        }
    }
}

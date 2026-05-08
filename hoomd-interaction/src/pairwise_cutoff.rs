// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `PairwiseCutoff`

use std::ops::AddAssign;

use serde::{Deserialize, Serialize};

use hoomd_microstate::{
    Body, Microstate, Site, SiteKey, Transform, boundary::Wrap, property::Position,
};
use hoomd_spatial::PointsNearBall;
use hoomd_vector::{InnerProduct, Metric, Vector, Wedge};
use crate::{
    DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, MaximumInteractionRange, NetSiteForce, NetSiteForceAndTorque, SitePairEnergy, SitePairForce, SitePairForceAndTorque, TotalEnergy
};

/// Short-ranged pairwise interactions between sites.
///
/// A [`PairwiseCutoff`] newtype wrapping a type that implements
/// [`SitePairEnergy`] represents:
/// ```math
/// U_\mathrm{total} = \sum_{i=0}^{N-1}\sum_{j=i+1}^{N-1} U\left(s_i, s_j \right) \left[ \left|\vec{r}_j - \vec{r}_i\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
/// ```
/// where $`U(s_i, s_j)`$ is the potential computed by [`SitePairEnergy`],
/// $`s_i`$ is the full set of site properties for site i, $`\vec{r}_i`$ is
/// the position of site i, $`b_i`$ is the body tag that holds site *i*, and
/// $`\left[ \  \right]`$ denotes the Iverson bracket.
///
/// In other words, [`PairwiseCutoff`] sums the energy for all pairs that are
/// separated by a distance less than the maximum interaction range `r_cut` and
/// belong to different bodies.
///
/// A [`PairwiseCutoff`] newtype wrapping a type that implements
/// [`SitePairForce`] and/or [`SitePairForceAndTorque`] represents:
/// ```math
/// \vec{F}_i = \sum_{j \ne i} \vec{F}\left(s_i, s_j \right) \left[ \left|\vec{r}_j - \vec{r}_i\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
/// ```
/// ```math
/// \vec{\tau}_i = \sum_{j \ne i} \vec{\tau}\left(s_i, s_j \right) \left[ \left|\vec{r}_j - \vec{r}_i\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
/// ```
/// where $`\vec{F}(s_i, s_j)`$ is the force computed by [`SitePairForce`]
/// (or [`SitePairForceAndTorque`]) and $`\vec{\tau}(s_i, s_j)`$ is the torque computed by
/// [`SitePairForceAndTorque`].
///
/// A type that implements *both* [`SitePairEnergy`] and [`SitePairForce`]
/// (or [`SitePairForceAndTorque`]) *must* compute forces and torques that are
/// derivatives of the energy.
///
/// Use [`PairwiseCutoff`] with [`Anisotropic`], [`Isotropic`], [`HardShape`], or
/// your own custom type that implements [`SitePairEnergy`], [`SitePairForce`] and/or
/// [`SitePairForceAndTorque`].
///
/// [`Anisotropic`]: crate::pairwise::Anisotropic
/// [`Isotropic`]: crate::pairwise::Isotropic
/// [`HardShape`]: crate::pairwise::HardShape
///
/// # Example
///
/// Basic usage:
/// ```
/// use hoomd_interaction::{
///     PairwiseCutoff, pairwise::Isotropic, univariate::LennardJones,
/// };
///
/// let lennard_jones: LennardJones = LennardJones {
///     epsilon: 1.5,
///     sigma: 2.0,
/// };
/// let pairwise_cutoff = PairwiseCutoff(Isotropic {
///     interaction: lennard_jones,
///     r_cut: 5.0,
/// });
/// ```
///
/// Set a custom potential using a closure (implements only [`SitePairEnergy`]):
/// ```
/// use hoomd_interaction::{PairwiseCutoff, pairwise::Isotropic};
///
/// let pairwise_cutoff = PairwiseCutoff(Isotropic {
///     interaction: |r: f64| 1.0 / (r.powi(12)),
///     r_cut: 3.0,
/// });
/// ```
///
/// Implement a custom potential via a type:
/// ```
/// use hoomd_interaction::{
///     PairwiseCutoff, pairwise::Isotropic, univariate::UnivariateEnergy,
/// };
///
/// struct Custom {
///     a: f64,
/// }
///
/// impl UnivariateEnergy for Custom {
///     fn energy(&self, r: f64) -> f64 {
///         self.a / r.powi(12)
///     }
/// }
///
/// let custom = Custom { a: 2.0 };
/// let pairwise_cutoff = PairwiseCutoff(Isotropic {
///     interaction: custom,
///     r_cut: 2.0,
/// });
/// ```
///
/// Hard sphere:
/// ```
/// use hoomd_interaction::{PairwiseCutoff, pairwise::HardSphere};
/// use hoomd_microstate::property::Point;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let hard_sphere = PairwiseCutoff(HardSphere { diameter: 1.0 });
/// # Ok(())
/// # }
/// ```
///
/// Hard ellipse:
///
/// ```
/// use hoomd_geometry::shape::Ellipse;
/// use hoomd_interaction::{PairwiseCutoff, pairwise::HardShape};
/// use hoomd_microstate::property::Point;
/// use hoomd_vector::Cartesian;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let ellipse = Ellipse::with_semi_axes([4.0.try_into()?, 1.0.try_into()?]);
/// let hard_ellipse = PairwiseCutoff(HardShape(ellipse));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairwiseCutoff<E>(pub E);

impl<E> PairwiseCutoff<E> {
    /// Calculate the pairwise force on site `i` caused by site `j`.
    ///
    /// Use this method to compute an individual term in the net force on site `i`,
    /// subject to the the maximum interaction range `r_cut` and inter-body checks:
    /// ```math
    /// \vec{F}_i = \sum_{j \ne i} \vec{F}\left(s_i, s_j \right) \left[ \left|\vec{r}_j - \vec{r}_i\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
    /// ```
    /// 
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_interaction::{
    ///     PairwiseCutoff, pairwise::Isotropic, univariate::LennardJones,
    /// };
    /// use hoomd_microstate::{
    ///     Body, Microstate, Site, property::Point
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0, 0.0])),
    ///     Body::point(Cartesian::from([1.0, 0.0, 0.0])),
    /// ])?;
    ///
    ///  let lennard_jones: LennardJones = LennardJones {
    ///             epsilon: 1.0,
    ///             sigma: 1.0};
    ///
    /// let force = PairwiseCutoff(
    ///     Isotropic{
    ///         interaction: lennard_jones,
    ///         r_cut: 2.5
    /// });
    ///
    /// let sites = microstate.sites();
    /// let force_0 = force.site_pair_force(&sites[0], &sites[1]);
    /// let force_1 = force.site_pair_force(&sites[1], &sites[0]);
    ///
    /// assert_relative_eq!(force_0, Cartesian::from([-24.0, 0.0, 0.0]));
    /// assert_relative_eq!(force_1, Cartesian::from([24.0, 0.0, 0.0]));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn site_pair_force<V, S>(&self, site_i: &Site<S>, site_j: &Site<S>) -> V
    where
        E: SitePairForce<S, Force = V>,
        V: Default,
    {
        if site_i.body_tag == site_j.body_tag {
            V::default()
        } else {
            self.0.site_pair_force(&site_i.properties, &site_j.properties)
        }
    }

    /// Calculate the pairwise force and torque on site `i` caused by site `j`.
    /// 
    /// Use this method to compute an individual term in the net force on site `i`,
    /// subject to the the maximum interaction range `r_cut` and inter-body checks:
    ///
    /// ```math
    /// \vec{F}_i = \sum_{j \ne i} \vec{F}\left(s_i, s_j \right) \left[ \left|\vec{r}_j - \vec{r}_i\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
    /// ```
    /// ```math
    /// \vec{\tau}_i = \sum_{j \ne i} \vec{\tau}\left(s_i, s_j \right) \left[ \left|\vec{r}_j - \vec{r}_i\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
    /// ```
    ///
    /// # Return value
    ///
    /// `net_site_force_and_torque` returns the force and torque in a tuple:
    /// `(force, torque)`.
    #[inline]
    pub fn site_pair_force_and_torque<V, S>(&self, site_i: &Site<S>, site_j: &Site<S>) -> (V, V::Bivector)
    where
        E: SitePairForceAndTorque<S, Force = V>,
        V: Default + Wedge,
        V::Bivector: Default,
    {
        if site_i.body_tag == site_j.body_tag {
            (V::default(), V::Bivector::default())
        } else {
            self.0.site_pair_force_and_torque(&site_i.properties, &site_j.properties)
        }
    }

    /// Compute the pair energy between two sites.
    ///
    /// Use this method to compute an individual term in the total pair energy,
    /// subject to the the maximum interaction range `r_cut` and inter-body checks:
    ///
    /// ```math
    /// U\left(s_i, s_j \right) \left[ \left|\vec{r}_j - \vec{r}_i\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
    /// ```
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_interaction::{
    ///     PairwiseCutoff, pairwise::Isotropic, univariate::LennardJones,
    /// };
    /// use hoomd_microstate::{Body, Microstate, Site};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lennard_jones: LennardJones = LennardJones {
    ///     epsilon: 1.0,
    ///     sigma: 1.0,
    /// };
    /// let pairwise_cutoff = PairwiseCutoff(Isotropic {
    ///     interaction: lennard_jones,
    ///     r_cut: 2.5,
    /// });
    ///
    /// let body_a = Body::point(Cartesian::from([0.0, 0.0]));
    /// let body_b = Body::point(Cartesian::from([0.0, 3.0]));
    /// let body_c = Body::point(Cartesian::from([0.0, -2.0_f64.powf(1.0 / 6.0)]));
    ///
    /// let microstate = Microstate::builder()
    ///     .bodies([body_a, body_b, body_c])
    ///     .try_build()?;
    ///
    /// let sites = microstate.sites();
    /// let energy_ab = pairwise_cutoff.site_pair_energy(&sites[0], &sites[1]);
    /// let energy_ac = pairwise_cutoff.site_pair_energy(&sites[0], &sites[2]);
    ///
    /// assert_eq!(energy_ab, 0.0);
    /// assert_relative_eq!(energy_ac, -1.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn site_pair_energy<S>(&self, site_i: &Site<S>, site_j: &Site<S>) -> f64
    where
        E: SitePairEnergy<S>,
    {
        if site_i.body_tag == site_j.body_tag {
            return 0.0;
        }

        self.0
            .site_pair_energy(&site_i.properties, &site_j.properties)
    }

    /// Compute the filtered energy contribution of a single site (`AllPairs` specialization)
    #[inline(always)]
    fn filtered_site_energy_all<B, S, X, C, F, F2>(
        &self,
        microstate: &Microstate<B, S, X, C>,
        site_i_properties: &S,
        filter: F,
        site_pair_energy: F2,
    ) -> f64
    where
        E: SitePairEnergy<S>,
        F: Fn(&Site<S>) -> bool,
        F2: Fn(&E, &S, &S) -> f64,
    {
        let mut energy = 0.0;

        for site_j in microstate.sites().iter().chain(microstate.ghosts()) {
            if filter(site_j) {
                let one = site_pair_energy(&self.0, site_i_properties, &site_j.properties);
                if one == f64::INFINITY {
                    return one;
                }

                energy += one;
            }
        }

        energy
    }

    /// Compute the filtered energy contribution of a single site (spatial data specialization)
    #[inline(always)]
    fn filtered_site_energy_spatial<P, B, S, X, C, F, F2>(
        &self,
        microstate: &Microstate<B, S, X, C>,
        site_i_properties: &S,
        filter: F,
        site_pair_energy: F2,
    ) -> f64
    where
        E: SitePairEnergy<S> + MaximumInteractionRange,
        S: Position<Position = P>,
        X: PointsNearBall<P, SiteKey>,
        F: Fn(&Site<S>) -> bool,
        F2: Fn(&E, &S, &S) -> f64,
    {
        let mut energy = 0.0;

        for site_j in microstate.iter_sites_near(
            site_i_properties.position(),
            self.0.maximum_interaction_range(),
        ) {
            if filter(site_j) {
                let one = site_pair_energy(&self.0, site_i_properties, &site_j.properties);
                if one == f64::INFINITY {
                    return one;
                }

                energy += one;
            }
        }

        energy
    }

    /// Compute the filtered energy contribution of a single site.
    #[inline(always)]
    fn filtered_site_energy<P, B, S, X, C, F, F2>(
        &self,
        microstate: &Microstate<B, S, X, C>,
        site_i_properties: &S,
        filter: F,
        site_pair_energy: F2,
    ) -> f64
    where
        E: SitePairEnergy<S> + MaximumInteractionRange,
        S: Position<Position = P>,
        X: PointsNearBall<P, SiteKey>,
        F: Fn(&Site<S>) -> bool,
        F2: Fn(&E, &S, &S) -> f64,
    {
        if X::is_all_pairs() {
            self.filtered_site_energy_all(microstate, site_i_properties, filter, site_pair_energy)
        } else {
            self.filtered_site_energy_spatial(
                microstate,
                site_i_properties,
                filter,
                site_pair_energy,
            )
        }
    }

    /// Compute the final energy of a body in the microstate.
    #[inline(always)]
    fn filtered_body_energy_final<P, B, S, X, C, F>(
        &self,
        microstate: &Microstate<B, S, X, C>,
        body: &Body<B, S>,
        filter: F,
    ) -> f64
    where
        E: SitePairEnergy<S> + MaximumInteractionRange,
        B: Transform<S>,
        S: Position<Position = P>,
        X: PointsNearBall<P, SiteKey>,
        C: Wrap<B> + Wrap<S>,
        F: Fn(&Site<S>) -> bool,
    {
        let mut energy_final = 0.0;
        for s in &body.sites {
            match microstate.boundary().wrap(body.properties.transform(s)) {
                Err(_) => return f64::INFINITY,
                Ok(site_i_properties) => {
                    let one = self.filtered_site_energy(
                        microstate,
                        &site_i_properties,
                        &filter,
                        E::site_pair_energy,
                    );
                    if one == f64::INFINITY {
                        return one;
                    }

                    energy_final += one;
                }
            }
        }
        energy_final
    }

    /// Compute the initial energy of a body in the microstate.
    #[inline(always)]
    fn filtered_body_energy_initial<P, B, S, X, C, F>(
        &self,
        microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        filter: F,
    ) -> f64
    where
        E: SitePairEnergy<S> + MaximumInteractionRange,
        S: Position<Position = P>,
        X: PointsNearBall<P, SiteKey>,
        F: Fn(&Site<S>) -> bool,
    {
        let mut energy_initial = 0.0;
        if !E::is_only_infinite_or_zero() {
            for site_i in microstate.iter_body_sites(body_index) {
                let one = self.filtered_site_energy(
                    microstate,
                    &site_i.properties,
                    &filter,
                    E::site_pair_energy_initial,
                );
                if one == f64::INFINITY {
                    return one;
                }

                energy_initial += one;
            }
        }
        energy_initial
    }
}

impl<V, B, S, X, C, E> NetSiteForce<B, S, X, C> for PairwiseCutoff<E>
where
    V: Vector + Default + InnerProduct + Metric,
    B: Transform<S>,
    S: Position<Position = V>,
    E: MaximumInteractionRange + SitePairForce<S, Force = V>,
    X: PointsNearBall<V, SiteKey>,
{
    type Force = V;
    
    /// Compute the net force and torque on a given site.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_interaction::{
    ///     NetSiteForce, PairwiseCutoff, pairwise::Isotropic, univariate::LennardJones,
    /// };
    /// use hoomd_microstate::{
    ///     Body, Microstate, Site, property::Point
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0, 0.0])),
    ///     Body::point(Cartesian::from([1.0, 0.0, 0.0])),
    /// ])?;
    ///
    ///  let lennard_jones: LennardJones = LennardJones {
    ///             epsilon: 1.0,
    ///             sigma: 1.0};
    ///
    /// let force = PairwiseCutoff(
    ///     Isotropic{
    ///         interaction: lennard_jones,
    ///         r_cut: 2.5
    /// });
    ///
    /// let force_0 = force.net_site_force(&microstate, 0);
    /// let force_1 = force.net_site_force(&microstate, 1);
    ///
    /// assert_relative_eq!(force_0, Cartesian::from([-24.0, 0.0, 0.0]));
    /// assert_relative_eq!(force_1, Cartesian::from([24.0, 0.0, 0.0]));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn net_site_force(&self, microstate: &Microstate<B, S, X, C>, site_index: usize) -> V {
        let site = &microstate.sites()[site_index];
        let mut total_force = V::default();
        
        for other_site in microstate
            .iter_sites_near(site.properties.position(), self.maximum_interaction_range())
            .filter(|s| site.body_tag != s.body_tag)
        {
            total_force += self.0.site_pair_force(&site.properties, &other_site.properties);
        }

        total_force
    }
}

impl<V, B, S, X, C, E> NetSiteForceAndTorque<B, S, X, C> for PairwiseCutoff<E>
where
    V: Vector + Default + InnerProduct + Metric + Wedge,
    B: Transform<S>,
    S: Position<Position = V>,
    E: MaximumInteractionRange + SitePairForceAndTorque<S, Force = V>,
    V::Bivector: AddAssign + Default,
    X: PointsNearBall<V, SiteKey>,
{
    type Force = V;
    
    /// Compute the net force and torque on a given site.
    /// 
    /// # Return value
    ///
    /// `net_site_force_and_torque` returns the force and torque in a tuple:
    /// `(force, torque)`.
    #[inline]
    fn net_site_force_and_torque(&self, microstate: &Microstate<B, S, X, C>, site_index: usize) -> (V, V::Bivector) {
        let site = &microstate.sites()[site_index];
        let mut total_force = V::default();
        let mut total_torque = V::Bivector::default();
        
        
        for other_site in microstate
            .iter_sites_near(site.properties.position(), self.maximum_interaction_range())
            .filter(|s| site.body_tag != s.body_tag)
        {
            let (force, torque) = self.0.site_pair_force_and_torque(&site.properties, &other_site.properties);
            total_force += force;
            total_torque += torque;
        }

        (total_force, total_torque)
    }
}

impl<P, B, S, X, C, E> TotalEnergy<Microstate<B, S, X, C>> for PairwiseCutoff<E>
where
    E: SitePairEnergy<S> + MaximumInteractionRange,
    S: Position<Position = P>,
    X: PointsNearBall<P, SiteKey>,
{
    /// Compute the total energy of the microstate contributed by functions on pairs of sites.
    ///
    /// # Examples
    ///
    /// Lennard-Jones:
    /// ```
    /// use hoomd_interaction::{
    ///     PairwiseCutoff, SitePairEnergy, TotalEnergy, pairwise::Isotropic,
    ///     univariate::LennardJones,
    /// };
    /// use hoomd_microstate::{
    ///     Body, Microstate,
    ///     property::{Point, Position},
    /// };
    /// use hoomd_vector::{Cartesian, InnerProduct};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([1.0, 0.0])),
    ///     Body::point(Cartesian::from([0.0, 5.0])),
    ///     Body::point(Cartesian::from([-1.0, 5.0])),
    /// ])?;
    ///
    /// let lennard_jones: LennardJones = LennardJones {
    ///     epsilon: 1.5,
    ///     sigma: 1.0 / 2.0_f64.powf(1.0 / 6.0),
    /// };
    /// let pairwise_cutoff = PairwiseCutoff(Isotropic {
    ///     interaction: lennard_jones,
    ///     r_cut: 2.5,
    /// });
    ///
    /// let total_energy = pairwise_cutoff.total_energy(&microstate);
    /// assert_eq!(total_energy, -3.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Hard circle:
    /// ```
    /// use hoomd_geometry::shape::Circle;
    /// use hoomd_interaction::{
    ///     PairwiseCutoff, TotalEnergy, pairwise::HardSphere,
    /// };
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::{Angle, Cartesian};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([0.4, 0.0])),
    /// ])?;
    ///
    /// let hard_circle = PairwiseCutoff(HardSphere { diameter: 1.0 });
    ///
    /// let total_energy = hard_circle.total_energy(&microstate);
    /// assert_eq!(total_energy, f64::INFINITY);
    ///
    /// microstate.update_body_properties(0, Point::new([0.0, -2.0].into()));
    /// let total_energy = hard_circle.total_energy(&microstate);
    /// assert_eq!(total_energy, 0.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, X, C>) -> f64 {
        let mut total = 0.0;

        // If needed, total_energy could specialize further in the all-pairs
        // code path. The current implementation performs many unneeded
        // site_i.site_tag < site_j.site_tag checks. However, the solution is
        // non-trivial. It would need to loop over the j sites *by tag*
        // to avoid looping over unneeded sites. The ghost loop would need
        // to be separate and include the tag filter.

        for site_i in microstate.sites() {
            let one = self.filtered_site_energy(
                microstate,
                &site_i.properties,
                |site_j| site_i.site_tag < site_j.site_tag && site_i.body_tag != site_j.body_tag,
                E::site_pair_energy,
            );
            if one == f64::INFINITY {
                return one;
            }

            total += one;
        }

        total
    }

    /// Compute the difference in energy between two microstates.
    ///
    /// Returns `$ E_\mathrm{final} - E_\mathrm{initial} $`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::{
    ///     PairwiseCutoff, SitePairEnergy, TotalEnergy, pairwise::Isotropic,
    ///     univariate::LennardJones,
    /// };
    /// use hoomd_microstate::{
    ///     Body, Microstate,
    ///     property::{Point, Position},
    /// };
    /// use hoomd_vector::{Cartesian, InnerProduct};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate_a = Microstate::new();
    /// microstate_a.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([1.0, 0.0])),
    /// ])?;
    ///
    /// let mut microstate_b = Microstate::new();
    /// microstate_b.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([5.0, 0.0])),
    /// ])?;
    ///
    /// let lennard_jones: LennardJones = LennardJones {
    ///     epsilon: 1.5,
    ///     sigma: 1.0 / 2.0_f64.powf(1.0 / 6.0),
    /// };
    /// let pairwise_cutoff = PairwiseCutoff(Isotropic {
    ///     interaction: lennard_jones,
    ///     r_cut: 2.5,
    /// });
    ///
    /// let delta_energy_total =
    ///     pairwise_cutoff.delta_energy_total(&microstate_a, &microstate_b);
    /// assert_eq!(delta_energy_total, 1.5);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn delta_energy_total(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        final_microstate: &Microstate<B, S, X, C>,
    ) -> f64 {
        let mut energy_final = 0.0;

        for site_i in final_microstate.sites() {
            let one = self.filtered_site_energy(
                final_microstate,
                &site_i.properties,
                |site_j| site_i.site_tag < site_j.site_tag && site_i.body_tag != site_j.body_tag,
                E::site_pair_energy,
            );
            if one == f64::INFINITY {
                return one;
            }
            energy_final += one;
        }

        let mut energy_initial = 0.0;
        if !E::is_only_infinite_or_zero() {
            for site_i in initial_microstate.sites() {
                let one = self.filtered_site_energy(
                    initial_microstate,
                    &site_i.properties,
                    |site_j| {
                        site_i.site_tag < site_j.site_tag && site_i.body_tag != site_j.body_tag
                    },
                    E::site_pair_energy_initial,
                );
                if one == f64::INFINITY {
                    return -one;
                }
                energy_initial += one;
            }
        }

        energy_final - energy_initial
    }
}

impl<P, B, S, X, C, E> DeltaEnergyOne<B, S, X, C> for PairwiseCutoff<E>
where
    E: SitePairEnergy<S> + MaximumInteractionRange,
    B: Transform<S>,
    S: Position<Position = P>,
    X: PointsNearBall<P, SiteKey>,
    C: Wrap<B> + Wrap<S>,
{
    /// Evaluate the change in energy contributed by `PairwiseCutoff` when one body is updated.
    ///
    /// # Examples
    ///
    /// Boxcar:
    /// ```
    /// use hoomd_interaction::{
    ///     DeltaEnergyOne, PairwiseCutoff, pairwise::Isotropic, univariate::Boxcar,
    /// };
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([1.0, 0.0])),
    /// ])?;
    ///
    /// let epsilon = 2.0;
    /// let (left, right) = (0.0, 1.5);
    /// let boxcar = Boxcar {
    ///     epsilon,
    ///     left,
    ///     right,
    /// };
    /// let pairwise_cutoff = PairwiseCutoff(Isotropic {
    ///     interaction: boxcar,
    ///     r_cut: 1.5,
    /// });
    ///
    /// let delta_energy = pairwise_cutoff.delta_energy_one(
    ///     &microstate,
    ///     0,
    ///     &Body::point([-1.0, 0.0].into()),
    /// );
    /// assert_eq!(delta_energy, -2.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Hard circle:
    /// ```
    /// use hoomd_geometry::shape::Circle;
    /// use hoomd_interaction::{
    ///     DeltaEnergyOne, PairwiseCutoff, pairwise::HardSphere,
    /// };
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::{Angle, Cartesian};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([2.0, 0.0])),
    /// ])?;
    ///
    /// let hard_circle = PairwiseCutoff(HardSphere { diameter: 1.0 });
    ///
    /// let delta_energy = hard_circle.delta_energy_one(
    ///     &microstate,
    ///     1,
    ///     &Body::point([0.4, 0.0].into()),
    /// );
    /// assert_eq!(delta_energy, f64::INFINITY);
    ///
    /// let delta_energy = hard_circle.delta_energy_one(
    ///     &microstate,
    ///     1,
    ///     &Body::point([1.5, 0.0].into()),
    /// );
    /// assert_eq!(delta_energy, 0.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let body_tag = initial_microstate.bodies()[body_index].tag;

        let energy_final =
            self.filtered_body_energy_final(initial_microstate, final_body, |site_j| {
                body_tag != site_j.body_tag
            });
        if energy_final == f64::INFINITY {
            return energy_final;
        }

        let energy_initial =
            self.filtered_body_energy_initial(initial_microstate, body_index, |site_j| {
                body_tag != site_j.body_tag
            });

        energy_final - energy_initial
    }
}

impl<P, B, S, X, C, E> DeltaEnergyInsert<B, S, X, C> for PairwiseCutoff<E>
where
    E: SitePairEnergy<S> + MaximumInteractionRange,
    B: Transform<S>,
    S: Position<Position = P>,
    X: PointsNearBall<P, SiteKey>,
    C: Wrap<B> + Wrap<S>,
{
    /// Evaluate the change in energy contributed by `PairwiseCutoff` when one body is inserted.
    ///
    /// # Example
    ///
    /// Boxcar:
    /// ```
    /// use hoomd_interaction::{
    ///     DeltaEnergyInsert, PairwiseCutoff, pairwise::Isotropic,
    ///     univariate::Boxcar,
    /// };
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([1.0, 0.0])),
    /// ])?;
    ///
    /// let epsilon = 2.0;
    /// let (left, right) = (0.0, 1.5);
    /// let boxcar = Boxcar {
    ///     epsilon,
    ///     left,
    ///     right,
    /// };
    /// let pairwise_cutoff = PairwiseCutoff(Isotropic {
    ///     interaction: boxcar,
    ///     r_cut: 1.5,
    /// });
    ///
    /// let delta_energy = pairwise_cutoff
    ///     .delta_energy_insert(&microstate, &Body::point([-1.0, 0.0].into()));
    /// assert_eq!(delta_energy, 2.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Hard circle:
    /// ```
    /// use hoomd_geometry::shape::Circle;
    /// use hoomd_interaction::{
    ///     DeltaEnergyInsert, PairwiseCutoff, pairwise::HardSphere,
    /// };
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::{Angle, Cartesian};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([Body::point(Cartesian::from([0.0, 0.0]))])?;
    ///
    /// let hard_circle = PairwiseCutoff(HardSphere { diameter: 1.0 });
    ///
    /// let delta_energy = hard_circle
    ///     .delta_energy_insert(&microstate, &Body::point([0.4, 0.0].into()));
    /// assert_eq!(delta_energy, f64::INFINITY);
    ///
    /// let delta_energy = hard_circle
    ///     .delta_energy_insert(&microstate, &Body::point([1.5, 0.0].into()));
    /// assert_eq!(delta_energy, 0.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn delta_energy_insert(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        new_body: &Body<B, S>,
    ) -> f64 {
        // The new body is not yet in the microstate, so there is no need to
        // filter matching body tags. The new body does not yet have a tag.
        self.filtered_body_energy_final(initial_microstate, new_body, |_| true)
    }
}

impl<P, B, S, X, C, E> DeltaEnergyRemove<B, S, X, C> for PairwiseCutoff<E>
where
    E: SitePairEnergy<S> + MaximumInteractionRange,
    S: Position<Position = P>,
    X: PointsNearBall<P, SiteKey>,
{
    /// Evaluate the change in energy contributed by `PairwiseCutoff` when one body is removed.
    ///
    /// # Example
    ///
    /// Boxcar:
    /// ```
    /// use hoomd_interaction::{
    ///     DeltaEnergyRemove, PairwiseCutoff, pairwise::Isotropic,
    ///     univariate::Boxcar,
    /// };
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([1.0, 0.0])),
    /// ])?;
    ///
    /// let epsilon = 2.0;
    /// let (left, right) = (0.0, 1.5);
    /// let boxcar = Boxcar {
    ///     epsilon,
    ///     left,
    ///     right,
    /// };
    /// let pairwise_cutoff = PairwiseCutoff(Isotropic {
    ///     interaction: boxcar,
    ///     r_cut: 1.5,
    /// });
    ///
    /// let delta_energy = pairwise_cutoff.delta_energy_remove(&microstate, 0);
    /// assert_eq!(delta_energy, -2.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Hard circle:
    /// ```
    /// use hoomd_geometry::shape::Circle;
    /// use hoomd_interaction::{
    ///     DeltaEnergyRemove, PairwiseCutoff, pairwise::HardSphere,
    /// };
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::{Angle, Cartesian};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([2.0, 0.0])),
    /// ])?;
    ///
    /// let hard_circle = PairwiseCutoff(HardSphere { diameter: 1.0 });
    ///
    /// let delta_energy = hard_circle.delta_energy_remove(&microstate, 1);
    /// assert_eq!(delta_energy, 0.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn delta_energy_remove(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
    ) -> f64 {
        let body_tag = initial_microstate.bodies()[body_index].tag;
        let energy_initial =
            self.filtered_body_energy_initial(initial_microstate, body_index, |site_j| {
                body_tag != site_j.body_tag
            });

        -energy_initial
    }
}

#[cfg(test)]
mod tests_finite {
    use super::*;
    use crate::{TotalEnergy, pairwise::Isotropic, univariate::HarmonicRepulsion};
    use assert2::check;
    use hoomd_geometry::shape::Hypercuboid;
    use hoomd_microstate::{
        boundary::{Closed, Open},
        property::Point,
    };
    use hoomd_spatial::{AllPairs, VecCell};
    use hoomd_vector::{Cartesian, distribution::Ball};

    use approxim::assert_relative_eq;
    use rand::{
        RngExt, SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };
    use rstest::*;
    use std::f64::consts::PI;

    #[fixture]
    fn square() -> Closed<Hypercuboid<2>> {
        let cuboid = Hypercuboid {
            edge_lengths: [
                4.0.try_into()
                    .expect("hard-coded constant should be positive"),
                4.0.try_into()
                    .expect("hard-coded constant should be positive"),
            ],
        };
        Closed(cuboid)
    }

    mod pairwise_cutoff {
        use super::*;
        use crate::pairwise::Isotropic;

        #[fixture]
        fn microstate()
        -> Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, AllPairs<SiteKey>, Open> {
            let mut microstate = Microstate::new();
            microstate
                .extend_bodies([
                    Body::point(Cartesian::from([0.0, 0.0])),
                    Body::point(Cartesian::from([1.0, 0.0])),
                    Body::point(Cartesian::from([0.0, 5.0])),
                    Body::point(Cartesian::from([1.0, 5.0])),
                ])
                .expect("hard-coded bodies should be in the boundary");
            microstate
        }

        #[rstest]
        fn blanket_fn(
            microstate: Microstate<
                Point<Cartesian<2>>,
                Point<Cartesian<2>>,
                AllPairs<SiteKey>,
                Open,
            >,
        ) {
            // Ensure that closures can be used as UnivariateEnergy
            let pairwise_cutoff = PairwiseCutoff(Isotropic {
                interaction: |r| 1.0 / (r * 2.0),
                r_cut: 2.0,
            });

            // Two pairs at a distance of 1.0 each with energy 1/2.
            assert_eq!(pairwise_cutoff.total_energy(&microstate), 1.0);

            let sites = microstate.sites();
            check!(pairwise_cutoff.site_pair_energy(&sites[0], &sites[0]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[0], &sites[1]) == 0.5);
            check!(pairwise_cutoff.site_pair_energy(&sites[0], &sites[2]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[0], &sites[3]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[1], &sites[1]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[1], &sites[2]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[1], &sites[3]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[2], &sites[2]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[2], &sites[3]) == 0.5);
            check!(pairwise_cutoff.site_pair_energy(&sites[3], &sites[3]) == 0.0);
        }

        #[rstest]
        fn large_r_cut(
            microstate: Microstate<
                Point<Cartesian<2>>,
                Point<Cartesian<2>>,
                AllPairs<SiteKey>,
                Open,
            >,
        ) {
            // Ensure that PairwiseCutoff respects the r_cut value set.
            let pairwise_cutoff = PairwiseCutoff(Isotropic {
                interaction: |r| 1.0 / (r * 2.0),
                r_cut: 5.0_f64.next_up(),
            });

            // Two pairs at a distance of 1.0 each with energy 1/2.
            // Plus two pairs at a distance of 5.0 with energy 1/10
            check!(pairwise_cutoff.total_energy(&microstate) == 1.2);
        }

        #[test]
        fn body_exclusion() -> anyhow::Result<()> {
            // Ensure that PairwiseCutoff excludes pairs in the same body.
            let body_a = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [
                    Point::new(Cartesian::from([1.0, 1.0])),
                    Point::new(Cartesian::from([1.0, -1.0])),
                    Point::new(Cartesian::from([-1.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, -1.0])),
                ]
                .into(),
            };
            let body_b = Body {
                properties: Point::new(Cartesian::from([3.0, 0.0])),
                sites: body_a.sites.clone(),
            };

            let mut microstate = Microstate::new();
            microstate.extend_bodies([body_a, body_b])?;

            let pairwise_cutoff = PairwiseCutoff(Isotropic {
                interaction: |_r| 1.0,
                r_cut: 1.0_f64.next_up(),
            });

            // Of all the pairs a distance 1.0 apart, only 2 are interbody pairs.
            check!(pairwise_cutoff.total_energy(&microstate) == 2.0);

            let sites = microstate.sites();
            check!(pairwise_cutoff.site_pair_energy(&sites[0], &sites[0]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[0], &sites[1]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[0], &sites[2]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[0], &sites[3]) == 0.0);

            check!(pairwise_cutoff.site_pair_energy(&sites[4], &sites[4]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[4], &sites[5]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[4], &sites[6]) == 0.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[4], &sites[7]) == 0.0);

            check!(pairwise_cutoff.site_pair_energy(&sites[0], &sites[6]) == 1.0);
            check!(pairwise_cutoff.site_pair_energy(&sites[1], &sites[7]) == 1.0);

            Ok(())
        }
    }

    mod delta_energy_one {
        use super::*;

        #[rstest]
        fn site_outside(square: Closed<Hypercuboid<2>>) -> anyhow::Result<()> {
            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };
            let mut final_body = body.clone();
            final_body.properties.position[0] = 1.0;

            let microstate = Microstate::builder()
                .boundary(square)
                .bodies([body])
                .try_build()?;

            let energy = PairwiseCutoff(Isotropic {
                interaction: |_r| 0.0,
                r_cut: 0.0,
            });

            check!(energy.delta_energy_one(&microstate, 0, &final_body) == f64::INFINITY);

            Ok(())
        }

        #[test]
        fn body_exclusion() -> anyhow::Result<()> {
            // Ensure that PairwiseCutoff.delta_energy_one excludes pairs in the same body.
            let body_a = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [
                    Point::new(Cartesian::from([1.0, 1.0])),
                    Point::new(Cartesian::from([1.0, -1.0])),
                    Point::new(Cartesian::from([-1.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, -1.0])),
                ]
                .into(),
            };
            let body_b = Body {
                properties: Point::new(Cartesian::from([3.0, 0.0])),
                sites: body_a.sites.clone(),
            };
            let body_a_final = Body {
                properties: Point::new(Cartesian::from([-1.0, 0.0])),
                sites: body_a.sites.clone(),
            };

            let mut microstate = Microstate::new();
            microstate.extend_bodies([body_a, body_b])?;

            let pairwise_cutoff = PairwiseCutoff(Isotropic {
                interaction: |_r| 1.0,
                r_cut: 1.0_f64.next_up(),
            });

            // Of all the pairs a distance 1.0 apart, only 2 are interbody pairs.
            // Moving body 0 to the left results in a -2.0 energy difference.
            check!(pairwise_cutoff.delta_energy_one(&microstate, 0, &body_a_final) == -2.0);

            Ok(())
        }

        #[test]
        fn random_moves() -> anyhow::Result<()> {
            // Ensure that PairwiseCutoff.delta_energy_one is consistent with TotalEnergy
            let body_template = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [
                    Point::new(Cartesian::from([0.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, -1.0])),
                ]
                .into(),
            };
            let body_a = Body {
                properties: Point::new(Cartesian::from([3.0, 0.0])),
                sites: body_template.sites.clone(),
            };
            let body_b = body_template.clone();

            let microstate_initial = Microstate::builder().bodies([body_a, body_b]).try_build()?;

            let mut microstate_final = microstate_initial.clone();
            let harmonic_repulsion: HarmonicRepulsion = HarmonicRepulsion { a: 5.0, r_cut: 5.0 };
            let pairwise_cutoff = PairwiseCutoff(Isotropic {
                interaction: harmonic_repulsion,
                r_cut: 5.0,
            });

            check!(pairwise_cutoff.total_energy(&microstate_initial) != 0.0);

            // Use `HarmonicRepulsion` for validation because it is a varies
            // with r and will therefore show some changes for any moves (unlike
            // `BoxCar`). HarmonicRepulsion avoids numerical errors when two
            // sites get too close.
            let mut rng = StdRng::seed_from_u64(0);
            let r_distribution = Uniform::new(3.0, 6.0)?;
            let theta_distribution = Uniform::new(0.0, 2.0 * PI)?;

            let mut new_body = body_template.clone();
            for _ in 0..1024 {
                let r = rng.sample(r_distribution);
                let theta = rng.sample(theta_distribution);
                new_body.properties.position = [r * theta.cos(), r * theta.sin()].into();

                let delta_energy_one =
                    pairwise_cutoff.delta_energy_one(&microstate_initial, 0, &new_body);
                microstate_final.update_body_properties(0, new_body.properties)?;
                let delta_energy_total = pairwise_cutoff.total_energy(&microstate_final)
                    - pairwise_cutoff.total_energy(&microstate_initial);

                assert_relative_eq!(delta_energy_one, delta_energy_total, epsilon = 1e-10);
                assert_relative_eq!(
                    pairwise_cutoff.delta_energy_total(&microstate_initial, &microstate_final),
                    delta_energy_total,
                    epsilon = 1e-10
                );
            }

            Ok(())
        }
    }

    mod delta_energy_insert_remove {
        use super::*;

        #[rstest]
        fn site_outside(square: Closed<Hypercuboid<2>>) -> anyhow::Result<()> {
            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };
            let mut new_body = body.clone();
            new_body.properties.position[0] = 1.0;

            let microstate = Microstate::builder()
                .boundary(square)
                .bodies([body])
                .try_build()?;

            let energy = PairwiseCutoff(Isotropic {
                interaction: |_r| 0.0,
                r_cut: 0.0,
            });

            check!(energy.delta_energy_insert(&microstate, &new_body) == f64::INFINITY);

            Ok(())
        }

        #[test]
        fn body_exclusion() -> anyhow::Result<()> {
            // Ensure that PairwiseCutoff.delta_energy_insert excludes pairs in the same body.
            let body_a_new = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [
                    Point::new(Cartesian::from([1.0, 1.0])),
                    Point::new(Cartesian::from([1.0, -1.0])),
                    Point::new(Cartesian::from([-1.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, -1.0])),
                ]
                .into(),
            };
            let body_b = Body {
                properties: Point::new(Cartesian::from([3.0, 0.0])),
                sites: body_a_new.sites.clone(),
            };

            let mut microstate = Microstate::new();
            microstate.extend_bodies([body_b])?;

            let pairwise_cutoff = PairwiseCutoff(Isotropic {
                interaction: |_r| 1.0,
                r_cut: 1.0_f64.next_up(),
            });

            // Of all the pairs a distance 1.0 apart, only 2 are interbody pairs.
            // Moving body 0 to the left results in a -2.0 energy difference.
            check!(pairwise_cutoff.delta_energy_insert(&microstate, &body_a_new) == 2.0);

            microstate.add_body(body_a_new)?;
            check!(pairwise_cutoff.delta_energy_remove(&microstate, 1) == -2.0);

            Ok(())
        }

        #[test]
        fn random_moves() -> anyhow::Result<()> {
            // Ensure that PairwiseCutoff.delta_energy_insert is consistent with TotalEnergy
            let body_template = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [
                    Point::new(Cartesian::from([0.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, -1.0])),
                ]
                .into(),
            };
            let body_a = Body {
                properties: Point::new(Cartesian::from([3.0, 0.0])),
                sites: body_template.sites.clone(),
            };

            let microstate_initial = Microstate::builder().bodies([body_a]).try_build()?;

            let mut microstate_final = microstate_initial.clone();
            let harmonic_repulsion: HarmonicRepulsion = HarmonicRepulsion { a: 5.0, r_cut: 5.0 };
            let pairwise_cutoff = PairwiseCutoff(Isotropic {
                interaction: harmonic_repulsion,
                r_cut: 5.0,
            });

            // Use `HarmonicRepulsion` for validation because it is a varies
            // with r and will therefore show some changes for any moves (unlike
            // `BoxCar`). HarmonicRepulsion avoids numerical errors when two
            // sites get too close.
            let mut rng = StdRng::seed_from_u64(2);
            let r_distribution = Uniform::new(3.0, 6.0)?;
            let theta_distribution = Uniform::new(0.0, 2.0 * PI)?;

            for _ in 0..1024 {
                let r = rng.sample(r_distribution);
                let theta = rng.sample(theta_distribution);
                let mut new_body = body_template.clone();
                new_body.properties.position = [r * theta.cos(), r * theta.sin()].into();

                let delta_energy_insert =
                    pairwise_cutoff.delta_energy_insert(&microstate_initial, &new_body);
                let tag = microstate_final.add_body(new_body)?;
                let delta_energy_total = pairwise_cutoff.total_energy(&microstate_final)
                    - pairwise_cutoff.total_energy(&microstate_initial);

                assert_relative_eq!(delta_energy_insert, delta_energy_total, epsilon = 1e-6);

                let delta_energy_remove = pairwise_cutoff.delta_energy_remove(&microstate_final, 1);
                assert_relative_eq!(delta_energy_remove, -delta_energy_total, epsilon = 1e-6);

                microstate_final.remove_body(
                    microstate_final.body_indices()[tag].expect("tag should be present"),
                );
            }

            Ok(())
        }
    }

    #[rstest]
    fn spatial_data_consistency(square: Closed<Hypercuboid<2>>) -> anyhow::Result<()> {
        const N_BODIES: usize = 2_000;
        let r_cut = 0.5;
        let mut rng = StdRng::seed_from_u64(0);

        let mut microstate_all_pairs = Microstate::builder()
            .spatial_data(AllPairs::<SiteKey>::default())
            .boundary(square.clone())
            .try_build()?;

        let cell_list = VecCell::builder()
            .nominal_search_radius(r_cut.try_into()?)
            .build();
        let mut microstate_vec_cell = Microstate::builder()
            .boundary(square.clone())
            .spatial_data(cell_list)
            .try_build()?;

        let body_template = Body {
            properties: Point::new(Cartesian::from([0.0, 0.0])),
            sites: [Point::default()].into(),
        };
        for _ in 0..N_BODIES {
            let mut new_body = body_template.clone();
            new_body.properties.position = square.sample(&mut rng);
            microstate_all_pairs.add_body(new_body.clone())?;
            microstate_vec_cell.add_body(new_body)?;
        }

        let harmonic_repulsion: HarmonicRepulsion = HarmonicRepulsion { a: 5.0, r_cut };
        let pairwise_cutoff = PairwiseCutoff(Isotropic {
            interaction: harmonic_repulsion,
            r_cut,
        });

        assert_relative_eq!(
            pairwise_cutoff.total_energy(&microstate_all_pairs),
            pairwise_cutoff.total_energy(&microstate_vec_cell)
        );

        let move_distribution = Ball {
            radius: 0.2.try_into()?,
        };
        for i in (0..N_BODIES).step_by(4) {
            assert_relative_eq!(
                pairwise_cutoff.delta_energy_remove(&microstate_all_pairs, i),
                pairwise_cutoff.delta_energy_remove(&microstate_vec_cell, i),
                epsilon = 1e-10
            );

            let mut final_body = microstate_all_pairs.bodies()[i].clone();
            final_body.item.properties.position += move_distribution.sample(&mut rng);

            assert_relative_eq!(
                pairwise_cutoff.delta_energy_one(&microstate_all_pairs, i, &final_body.item),
                pairwise_cutoff.delta_energy_one(&microstate_vec_cell, i, &final_body.item),
                epsilon = 1e-10
            );
        }

        for _ in 0..N_BODIES / 4 {
            let mut new_body = body_template.clone();
            new_body.properties.position = square.sample(&mut rng);

            assert_relative_eq!(
                pairwise_cutoff.delta_energy_insert(&microstate_all_pairs, &new_body),
                pairwise_cutoff.delta_energy_insert(&microstate_vec_cell, &new_body),
                epsilon = 1e-10
            );
        }

        Ok(())
    }
}

impl<E> MaximumInteractionRange for PairwiseCutoff<E>
where
    E: MaximumInteractionRange,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.0.maximum_interaction_range()
    }
}

#[cfg(test)]
mod test_infinite {
    use super::*;
    use crate::{
        TotalEnergy,
        pairwise::{HardShape, HardSphere},
    };
    use assert2::check;
    use hoomd_geometry::shape::{Ellipse, Hypercuboid};
    use hoomd_microstate::{
        boundary::Closed,
        property::{OrientedPoint, Point},
    };
    use hoomd_vector::{Angle, Cartesian};

    use rstest::*;

    #[fixture]
    fn square() -> Closed<Hypercuboid<2>> {
        let cuboid = Hypercuboid {
            edge_lengths: [
                4.0.try_into()
                    .expect("hard-coded constant should be positive"),
                4.0.try_into()
                    .expect("hard-coded constant should be positive"),
            ],
        };
        Closed(cuboid)
    }

    mod pairwise_cutoff {
        use super::*;

        #[test]
        fn large_r_cut() -> anyhow::Result<()> {
            let mut microstate = Microstate::new();
            microstate.extend_bodies([
                Body::point(Cartesian::from([0.0, 0.0])),
                Body::point(Cartesian::from([0.0, 5.0])),
            ])?;

            // Ensure that PairwiseCutoff respects the r_cut value set.
            let r_cut = 5.0_f64.next_up();
            let pairwise_cutoff = PairwiseCutoff(HardSphere { diameter: r_cut });

            check!(pairwise_cutoff.total_energy(&microstate) == f64::INFINITY);

            let r_cut = 5.0_f64;
            let pairwise_cutoff = PairwiseCutoff(HardSphere { diameter: r_cut });

            check!(pairwise_cutoff.total_energy(&microstate) == 0.0);

            Ok(())
        }

        #[test]
        fn body_exclusion() -> anyhow::Result<()> {
            // Ensure that PairwiseCutoff excludes pairs in the same body.
            let body_a = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [
                    Point::new(Cartesian::from([1.0, 1.0])),
                    Point::new(Cartesian::from([1.0, -1.0])),
                    Point::new(Cartesian::from([-1.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, -1.0])),
                ]
                .into(),
            };
            let body_b = Body {
                properties: Point::new(Cartesian::from([4.0, 0.0])),
                sites: body_a.sites.clone(),
            };

            let mut microstate = Microstate::new();
            microstate.extend_bodies([body_a, body_b])?;

            let r_cut = 1.0_f64.next_up();
            let pairwise_cutoff = PairwiseCutoff(HardSphere { diameter: r_cut });

            check!(pairwise_cutoff.total_energy(&microstate) == 0.0);

            let r_cut = 2.0_f64.next_up();
            let pairwise_cutoff = PairwiseCutoff(HardSphere { diameter: r_cut });

            check!(pairwise_cutoff.total_energy(&microstate) == f64::INFINITY);

            Ok(())
        }
    }

    mod delta_energy_one {
        use super::*;

        #[rstest]
        fn site_outside(square: Closed<Hypercuboid<2>>) -> anyhow::Result<()> {
            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };
            let mut final_body = body.clone();
            final_body.properties.position[0] = 1.0;

            let microstate = Microstate::builder()
                .boundary(square)
                .bodies([body])
                .try_build()?;

            let energy = PairwiseCutoff(HardSphere { diameter: 0.0 });

            check!(energy.delta_energy_one(&microstate, 0, &final_body) == f64::INFINITY);

            Ok(())
        }

        #[test]
        fn body_exclusion() -> anyhow::Result<()> {
            // Ensure that PairwiseCutoff.delta_energy_one excludes pairs in the same body.
            let body_a = Body {
                properties: Point::new(Cartesian::from([-1.0, 0.0])),
                sites: [
                    Point::new(Cartesian::from([1.0, 1.0])),
                    Point::new(Cartesian::from([1.0, -1.0])),
                    Point::new(Cartesian::from([-1.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, -1.0])),
                ]
                .into(),
            };
            let body_b = Body {
                properties: Point::new(Cartesian::from([3.0, 0.0])),
                sites: body_a.sites.clone(),
            };
            let body_a_overlap = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: body_a.sites.clone(),
            };
            let body_a_no_overlap = Body {
                properties: Point::new(Cartesian::from([-1.0, -1.0])),
                sites: body_a.sites.clone(),
            };

            let mut microstate = Microstate::new();
            microstate.extend_bodies([body_a, body_b])?;

            let r_cut = 1.0_f64.next_up();
            let pairwise_cutoff = PairwiseCutoff(HardSphere { diameter: r_cut });

            // moving body a to the right generates overlaps
            check!(
                pairwise_cutoff.delta_energy_one(&microstate, 0, &body_a_overlap) == f64::INFINITY
            );

            // moving body away results in no overlaps
            check!(pairwise_cutoff.delta_energy_one(&microstate, 0, &body_a_no_overlap) == 0.0);

            Ok(())
        }
    }

    mod delta_energy_insert_remove {
        use super::*;

        #[rstest]
        fn site_outside(square: Closed<Hypercuboid<2>>) -> anyhow::Result<()> {
            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };
            let mut new_body = body.clone();
            new_body.properties.position[0] = 1.0;

            let microstate = Microstate::builder()
                .boundary(square)
                .bodies([body])
                .try_build()?;

            let energy = PairwiseCutoff(HardSphere { diameter: 0.0 });

            check!(energy.delta_energy_insert(&microstate, &new_body) == f64::INFINITY);

            Ok(())
        }

        #[test]
        fn body_exclusion() -> anyhow::Result<()> {
            // Ensure that PairwiseCutoff.delta_energy_insert excludes pairs in the same body.
            let body_a_new = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [
                    Point::new(Cartesian::from([1.0, 1.0])),
                    Point::new(Cartesian::from([1.0, -1.0])),
                    Point::new(Cartesian::from([-1.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, -1.0])),
                ]
                .into(),
            };
            let body_b = Body {
                properties: Point::new(Cartesian::from([3.0, 0.0])),
                sites: body_a_new.sites.clone(),
            };

            let mut microstate = Microstate::new();
            microstate.extend_bodies([body_b])?;

            let r_cut = 1.0_f64.next_up();
            let pairwise_cutoff = PairwiseCutoff(HardSphere { diameter: r_cut });

            check!(pairwise_cutoff.delta_energy_insert(&microstate, &body_a_new) == f64::INFINITY);

            microstate.add_body(body_a_new)?;
            check!(pairwise_cutoff.delta_energy_remove(&microstate, 1) == 0.0);

            Ok(())
        }

        #[test]
        fn hard_shape_initial() -> anyhow::Result<()> {
            // Ensure that HardShape always evaluates 0 initial energies.
            let a = OrientedPoint {
                position: Cartesian::from([0.0, 0.0]),
                orientation: Angle::default(),
            };
            let b = OrientedPoint {
                position: Cartesian::from([1.5, 0.0]),
                orientation: Angle::default(),
            };
            let body_a = Body {
                properties: a,
                sites: [a].into(),
            };
            let body_b = Body {
                properties: b,
                sites: [a].into(),
            };
            let mut microstate = Microstate::new();
            microstate.extend_bodies([body_a, body_b.clone()])?;

            let ellipse = Ellipse::with_semi_axes([1.0.try_into()?, 2.0.try_into()?]);
            let hard_ellipse = PairwiseCutoff(HardShape(ellipse));

            // The initial configuration should have infinite energy.
            check!(hard_ellipse.total_energy(&microstate) == f64::INFINITY);
            check!(hard_ellipse.delta_energy_one(&microstate, 1, &body_b) == f64::INFINITY);

            let mut new_body_b = body_b;
            new_body_b.properties.position.coordinates = [2.1, 0.0];

            // That infinity should be ignored, resulting in a delta E of 0
            // when the body is moved into a non-overlapping state.
            check!(hard_ellipse.delta_energy_one(&microstate, 1, &new_body_b) == 0.0);

            Ok(())
        }

        #[test]
        fn delta_energy_total() -> anyhow::Result<()> {
            let mut microstate_0 = Microstate::new();
            microstate_0.extend_bodies([
                Body::point(Cartesian::from([0.0, 0.0])),
                Body::point(Cartesian::from([0.0, 1.125])),
            ])?;

            let mut microstate_inf = Microstate::new();
            microstate_inf.extend_bodies([
                Body::point(Cartesian::from([0.0, 0.0])),
                Body::point(Cartesian::from([0.0, 0.875])),
            ])?;

            let pairwise_cutoff = PairwiseCutoff(HardSphere { diameter: 1.0 });

            check!(pairwise_cutoff.delta_energy_total(&microstate_0, &microstate_0) == 0.0);
            check!(
                pairwise_cutoff.delta_energy_total(&microstate_0, &microstate_inf) == f64::INFINITY
            );
            check!(pairwise_cutoff.delta_energy_total(&microstate_inf, &microstate_0) == 0.0);
            check!(
                pairwise_cutoff.delta_energy_total(&microstate_inf, &microstate_inf)
                    == f64::INFINITY
            );

            Ok(())
        }
    }
}

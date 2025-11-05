// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `PairwiseCutoff`

use crate::{DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, SitePairEnergy, TotalEnergy};
use hoomd_microstate::{
    Body, Microstate, Site, SiteKey, Transform, boundary::Wrap, property::Position,
};
use hoomd_spatial::PointsNearBall;
use hoomd_vector::Metric;

/// Short-ranged pairwise interactions between sites.
///
/// Given an evaluator that implements [`SitePairEnergy`], [`PairwiseCutoff`] represents:
///
/// ```math
/// U_\mathrm{total} = \sum_{i=0}^{N-1}\sum_{j=i+1}^{N-1} U\left(s_i, s_j \right) \left[ \left|\vec{r}_j - \vec{r}_i\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
/// ```
/// where $`U(s_i, s_j)`$ is the potential computed by [`PairwiseCutoff::evaluator`],
/// $`s_i`$ is the full set of site properties for site i, $`\vec{r}_i`$ is
/// the position of site i, $`b_i`$ is the body tag that holds site *i*, and
/// $`\left[ \  \right]`$ denotes the Iverson bracket.
///
/// In other words, [`PairwiseCutoff`] sums the energy for all pairs that are separated
/// by a distance less than `r_cut` and belong to different bodies.
///
/// For the evaluator, use [`Anisotropic`], [`Isotropic`], [`HardShape`], or your own custom type.
///
/// TODO: Reword this when [`PairwiseCutoff`] also implements `SitePairForce`.
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
///     PairwiseCutoff,
///     pairwise::{Isotropic, LennardJones},
/// };
///
/// let lennard_jones: LennardJones = LennardJones {
///     epsilon: 1.5,
///     sigma: 2.0,
/// };
/// let evaluator = Isotropic(lennard_jones);
/// let cutoff_pair = PairwiseCutoff {
///     r_cut: 5.0,
///     evaluator,
/// };
/// ```
///
/// Set a custom potential using a closure:
/// ```
/// use hoomd_interaction::{PairwiseCutoff, pairwise::Isotropic};
///
/// let cutoff_pair = PairwiseCutoff {
///     r_cut: 3.0,
///     evaluator: Isotropic(|r: f64| 1.0 / (r.powi(12))),
/// };
/// ```
///
/// Implement a custom potential via a type:
/// ```
/// use hoomd_interaction::{
///     PairwiseCutoff,
///     pairwise::{Isotropic, IsotropicEnergy},
/// };
///
/// struct Custom {
///     a: f64,
/// }
///
/// impl IsotropicEnergy for Custom {
///     fn energy(&self, r: f64) -> f64 {
///         self.a / r.powi(12)
///     }
/// }
///
/// let custom = Custom { a: 2.0 };
/// let cutoff_pair = PairwiseCutoff {
///     r_cut: 2.0,
///     evaluator: Isotropic(custom),
/// };
/// ```
///
/// Hard sphere:
/// ```
/// use hoomd_interaction::{PairwiseCutoff, pairwise::HardSphere};
/// use hoomd_microstate::property::Point;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let hard_sphere = PairwiseCutoff {
///     r_cut: 1.0,
///     evaluator: HardSphere,
/// };
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
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let ellipse = Ellipse {
///     semi_axes: [4.0.try_into()?, 1.0.try_into()?],
/// };
/// let hard_ellipse = PairwiseCutoff {
///     r_cut: 8.0,
///     evaluator: HardShape(ellipse),
/// };
/// # Ok(())
/// # }
/// ```
pub struct PairwiseCutoff<E> {
    /// The distance beyond which all pairwise interactions evaluate to 0.
    pub r_cut: f64,

    /// Computes the pairwise energies and forces.
    pub evaluator: E,
}

impl<E> PairwiseCutoff<E> {
    /// Compute the pair energy between two sites.
    ///
    /// Use this method to compute an individual term in the total pair energy,
    /// subject to the `r_cut` and inter-body checks:
    ///
    /// ```math
    /// U\left(s_i, s_j \right) \left[ \left|\vec{r}_j - \vec{r}_i\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
    /// ```
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    ///
    /// use hoomd_interaction::{
    ///     PairwiseCutoff,
    ///     pairwise::{Isotropic, LennardJones},
    /// };
    /// use hoomd_microstate::{Body, Microstate, Site};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lennard_jones: LennardJones = LennardJones {
    ///     epsilon: 1.0,
    ///     sigma: 1.0,
    /// };
    /// let evaluator = Isotropic(lennard_jones);
    /// let cutoff_pair = PairwiseCutoff {
    ///     r_cut: 2.5,
    ///     evaluator,
    /// };
    ///
    /// let body_a = Body::point(Cartesian::from([0.0, 0.0]));
    /// let body_b = Body::point(Cartesian::from([0.0, 3.0]));
    /// let body_c = Body::point(Cartesian::from([0.0, -2.0f64.powf(1.0 / 6.0)]));
    ///
    /// let microstate = Microstate::builder()
    ///     .bodies([body_a, body_b, body_c])
    ///     .try_build()?;
    ///
    /// let sites = microstate.sites();
    /// let energy_ab = cutoff_pair.site_pair_energy(&sites[0], &sites[1]);
    /// let energy_ac = cutoff_pair.site_pair_energy(&sites[0], &sites[2]);
    ///
    /// assert_eq!(energy_ab, 0.0);
    /// assert_relative_eq!(energy_ac, -1.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn site_pair_energy<P, S>(
        &self,
        site_properties_i: &Site<S>,
        site_properties_j: &Site<S>,
    ) -> f64
    where
        E: SitePairEnergy<S>,
        S: Position<Position = P>,
        P: Metric,
    {
        let r = (site_properties_i.properties.position())
            .distance(site_properties_j.properties.position());
        if r < self.r_cut && site_properties_i.body_tag != site_properties_j.body_tag {
            self.evaluator
                .site_pair_energy(&site_properties_i.properties, &site_properties_j.properties)
        } else {
            0.0
        }
    }
}

impl<P, B, S, X, C, E> TotalEnergy<Microstate<B, S, X, C>> for PairwiseCutoff<E>
where
    E: SitePairEnergy<S>,
    S: Position<Position = P>,
    P: Metric,
    X: PointsNearBall<P, SiteKey>,
{
    /// Compute the total energy of the microstate contributed by functions on pairs of sites.
    ///
    /// # Examples
    ///
    /// Lennard-Jones:
    /// ```
    /// use hoomd_interaction::{PairwiseCutoff, SitePairEnergy, TotalEnergy,
    /// pairwise::{Isotropic, LennardJones}};
    /// use hoomd_microstate::{Microstate, Body};
    /// use hoomd_microstate::property::{Point, Position};
    /// use hoomd_vector::{Cartesian, InnerProduct};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([Body::point(Cartesian::from([0.0, 0.0])),
    /// Body::point(Cartesian::from([1.0, 0.0])),
    /// Body::point(Cartesian::from([0.0, 5.0])),
    /// Body::point(Cartesian::from([-1.0, 5.0])),
    /// ])?;
    ///
    /// let lennard_jones: LennardJones = LennardJones { epsilon: 1.5,
    /// sigma: 1.0 / 2.0_f64.powf(1.0 / 6.0) };
    /// let evaluator = Isotropic(lennard_jones);
    /// let cutoff_pair = PairwiseCutoff { r_cut: 2.5, evaluator };
    ///
    /// // The potential energy is set to 0 beyond r_cut when computed by `PairwiseCutoff`.
    /// let total_energy = cutoff_pair.total_energy(&microstate);
    /// assert_eq!(total_energy, -3.0);
    ///
    /// // However, individual pairwise `site_pair_energy` evaluations are always computed.
    /// let a = &microstate.sites()[0].properties;
    /// let b = &microstate.sites()[2].properties;
    /// assert_eq!((*a.position() - *b.position()).norm(), 5.0);
    /// assert!(cutoff_pair.evaluator.site_pair_energy(a, b) < 0.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Hard circle:
    /// ```
    /// use hoomd_interaction::{
    ///     PairwiseCutoff, TotalEnergy, pairwise::HardSphere,
    /// };
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::{Angle, Cartesian};
    /// use hoomd_geometry::shape::Circle;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([0.4, 0.0])),
    /// ])?;
    ///
    /// let hard_circle = PairwiseCutoff {
    ///     r_cut: 1.0,
    ///     evaluator: HardSphere,
    /// };
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
        for site_i in microstate.sites() {
            for site_j in microstate.iter_sites_near(site_i.properties.position(), self.r_cut) {
                if site_i.site_tag < site_j.site_tag
                    && site_i.body_tag != site_j.body_tag
                    && site_i
                        .properties
                        .position()
                        .distance_squared(site_j.properties.position())
                        < self.r_cut.powi(2)
                {
                    let one = self
                        .evaluator
                        .site_pair_energy(&site_i.properties, &site_j.properties);
                    if one == f64::INFINITY {
                        return one;
                    }

                    total += one;
                }
            }
        }

        total
    }

    // TODO: TotalEnergy needs a delta_energy method that takes an initial and final microstate
    // so that it can call `site_pair_energy_initial` (or skip the evaluation entirely
    // for infinite-only potentials) on the initial state.
}

impl<P, B, S, X, C, E> DeltaEnergyOne<B, S, X, C> for PairwiseCutoff<E>
where
    E: SitePairEnergy<S>,
    B: Transform<S>,
    S: Position<Position = P>,
    X: PointsNearBall<P, SiteKey>,
    C: Wrap<B> + Wrap<S>,
    P: Metric,
    {
    /// Evaluate the change in energy contributed by `PairwiseCutoff` when one body is updated.
    ///
    /// # Examples
    ///
    /// Boxcar:
    /// ```
    /// use hoomd_interaction::{
    ///     PairwiseCutoff, DeltaEnergyOne,
    ///     pairwise::{Boxcar, Isotropic},
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
    /// let evaluator = Isotropic(boxcar);
    /// let cutoff_pair = PairwiseCutoff {
    ///     r_cut: 1.5,
    ///     evaluator,
    /// };
    ///
    /// let delta_energy = cutoff_pair.delta_energy_one(
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
    /// use hoomd_interaction::{
    ///     PairwiseCutoff, DeltaEnergyOne, pairwise::HardSphere,
    /// };
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::{Angle, Cartesian};
    /// use hoomd_geometry::shape::Circle;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([2.0, 0.0])),
    /// ])?;
    ///
    /// let hard_circle = PairwiseCutoff {
    ///     r_cut: 1.0,
    ///     evaluator: HardSphere,
    /// };
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

        let mut energy_final = 0.0;
        for s in &final_body.sites {
            match initial_microstate
                .boundary()
                .wrap(final_body.properties.transform(s))
            {
                Err(_) => return f64::INFINITY,
                Ok(site_i_properties) => {
                    for site_j in initial_microstate.iter_sites_near(site_i_properties.position(), self.r_cut) {
                        if body_tag != site_j.body_tag
                            && site_i_properties
                                .position()
                                .distance_squared(site_j.properties.position())
                                < self.r_cut.powi(2)
                        {
                            let one = self
                                    .evaluator
                                    .site_pair_energy(&site_i_properties, &site_j.properties);
                            if one == f64::INFINITY {
                                return one;
                            }                                    
                        
                            energy_final += one;
                        }
                    }
                }
            }
        }

        let mut energy_initial = 0.0;
        if !E::is_only_infinite_or_zero() {
            for site_i in initial_microstate.iter_body_sites(body_index) {
                for site_j in initial_microstate.iter_sites_near(site_i.properties.position(), self.r_cut) {
                    if body_tag != site_j.body_tag
                        && site_i.properties
                            .position()
                            .distance_squared(site_j.properties.position())
                            < self.r_cut.powi(2)
                    {
                        let one = self
                                .evaluator
                                .site_pair_energy_initial(&site_i.properties, &site_j.properties);
                        if one == f64::INFINITY {
                            return one;
                        }                                    
                    
                        energy_initial += one;
                    }
                }
            }
        }

        energy_final - energy_initial
    }
}

impl<P, B, S, X, C, E> DeltaEnergyInsert<B, S, X, C> for PairwiseCutoff<E>
where
    E: SitePairEnergy<S>,
    B: Transform<S>,
    S: Position<Position = P>,
    X: PointsNearBall<P, SiteKey>,
    C: Wrap<B> + Wrap<S>,
    P: Metric,
{
    /// Evaluate the change in energy contributed by `PairwiseCutoff` when one body is inserted.
    ///
    /// # Example
    ///
    /// Boxcar:
    /// ```
    /// use hoomd_interaction::{
    ///     PairwiseCutoff, DeltaEnergyInsert,
    ///     pairwise::{Boxcar, Isotropic},
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
    /// let evaluator = Isotropic(boxcar);
    /// let cutoff_pair = PairwiseCutoff {
    ///     r_cut: 1.5,
    ///     evaluator,
    /// };
    ///
    /// let delta_energy = cutoff_pair
    ///     .delta_energy_insert(&microstate, &Body::point([-1.0, 0.0].into()));
    /// assert_eq!(delta_energy, 2.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Hard circle:
    /// ```
    /// use hoomd_interaction::{
    ///     PairwiseCutoff, DeltaEnergyInsert, pairwise::HardSphere,
    /// };
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::{Angle, Cartesian};
    /// use hoomd_geometry::shape::Circle;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([Body::point(Cartesian::from([0.0, 0.0]))])?;
    ///
    /// let hard_circle = PairwiseCutoff {
    ///     r_cut: 1.0,
    ///     evaluator: HardSphere,
    /// };
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
        let mut energy_final = 0.0;
        for s in &new_body.sites {
            match initial_microstate
                .boundary()
                .wrap(new_body.properties.transform(s))
            {
                Err(_) => return f64::INFINITY,
                Ok(site_i_properties) => {
                    for site_j in initial_microstate.iter_sites_near(site_i_properties.position(), self.r_cut) {
                        if site_i_properties
                                .position()
                                .distance_squared(site_j.properties.position())
                                < self.r_cut.powi(2)
                        {
                            let one = self
                                    .evaluator
                                    .site_pair_energy(&site_i_properties, &site_j.properties);
                            if one == f64::INFINITY {
                                return one;
                            }                                    
                        
                            energy_final += one;
                        }
                    }
                }
            }
        }

        energy_final
    }
}

impl<P, B, S, X, C, E> DeltaEnergyRemove<B, S, X, C> for PairwiseCutoff<E>
where
    E: SitePairEnergy<S>,
    S: Position<Position = P>,
    X: PointsNearBall<P, SiteKey>,
    P: Metric,
    {
    /// Evaluate the change in energy contributed by `PairwiseCutoff` when one body is removed.
    ///
    /// # Example
    ///
    /// Boxcar:
    /// ```
    /// use hoomd_interaction::{
    ///     PairwiseCutoff, DeltaEnergyRemove,
    ///     pairwise::{Boxcar, Isotropic},
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
    /// let evaluator = Isotropic(boxcar);
    /// let cutoff_pair = PairwiseCutoff {
    ///     r_cut: 1.5,
    ///     evaluator,
    /// };
    ///
    /// let delta_energy = cutoff_pair.delta_energy_remove(&microstate, 0);
    /// assert_eq!(delta_energy, -2.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Hard circle:
    /// ```
    /// use hoomd_interaction::{
    ///     PairwiseCutoff, DeltaEnergyRemove, pairwise::HardSphere,
    /// };
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::{Angle, Cartesian};
    /// use hoomd_geometry::shape::Circle;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([2.0, 0.0])),
    /// ])?;
    ///
    /// let hard_circle = PairwiseCutoff {
    ///     r_cut: 1.0,
    ///     evaluator: HardSphere,
    /// };
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

        let mut energy_initial = 0.0;
        if !E::is_only_infinite_or_zero() {
            for site_i in initial_microstate.iter_body_sites(body_index) {
                for site_j in initial_microstate.iter_sites_near(site_i.properties.position(), self.r_cut) {
                    if body_tag != site_j.body_tag
                        && site_i.properties
                            .position()
                            .distance_squared(site_j.properties.position())
                            < self.r_cut.powi(2)
                    {
                        let one = self
                                .evaluator
                                .site_pair_energy_initial(&site_i.properties, &site_j.properties);
                        if one == f64::INFINITY {
                            return one;
                        }                                    
                    
                        energy_initial += one;
                    }
                }
            }
        }

        -energy_initial
    }
}

#[cfg(test)]
mod tests_finite {
    use super::*;
    use assert2::check;
    use crate::{
        TotalEnergy,
        pairwise::{Isotropic, LennardJones},
    };
    use hoomd_geometry::shape::Hypercuboid;
    use hoomd_microstate::{
        boundary::{Closed, Open},
        property::Point,
    };
    use hoomd_spatial::AllPairs;
    use hoomd_vector::Cartesian;

    use approxim::assert_relative_eq;
    use rand::{Rng, SeedableRng, distr::Uniform, rngs::StdRng};
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

    mod cutoff_pair {
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
            // Ensure that closures can be used as IsotropicEnergy
            let cutoff_pair = PairwiseCutoff {
                r_cut: 2.0,
                evaluator: Isotropic(|r| 1.0 / (r * 2.0)),
            };

            // Two pairs at a distance of 1.0 each with energy 1/2.
            assert_eq!(cutoff_pair.total_energy(&microstate), 1.0);

            let sites = microstate.sites();
            check!(cutoff_pair.site_pair_energy(&sites[0], &sites[0]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[0], &sites[1]) == 0.5);
            check!(cutoff_pair.site_pair_energy(&sites[0], &sites[2]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[0], &sites[3]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[1], &sites[1]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[1], &sites[2]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[1], &sites[3]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[2], &sites[2]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[2], &sites[3]) == 0.5);
            check!(cutoff_pair.site_pair_energy(&sites[3], &sites[3]) == 0.0);
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
            let cutoff_pair = PairwiseCutoff {
                r_cut: 5.0_f64.next_up(),
                evaluator: Isotropic(|r| 1.0 / (r * 2.0)),
            };

            // Two pairs at a distance of 1.0 each with energy 1/2.
            // Plus two pairs at a distance of 5.0 with energy 1/10
            check!(cutoff_pair.total_energy(&microstate) == 1.2);
        }

        #[test]
        fn body_exclusion() {
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
            microstate
                .extend_bodies([body_a, body_b])
                .expect("hard-coded bodies should be in the boundary");

            let cutoff_pair = PairwiseCutoff {
                r_cut: 1.0_f64.next_up(),
                evaluator: Isotropic(|_r| 1.0),
            };

            // Of all the pairs a distance 1.0 apart, only 2 are interbody pairs.
            check!(cutoff_pair.total_energy(&microstate) == 2.0);

            let sites = microstate.sites();
            check!(cutoff_pair.site_pair_energy(&sites[0], &sites[0]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[0], &sites[1]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[0], &sites[2]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[0], &sites[3]) == 0.0);

            check!(cutoff_pair.site_pair_energy(&sites[4], &sites[4]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[4], &sites[5]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[4], &sites[6]) == 0.0);
            check!(cutoff_pair.site_pair_energy(&sites[4], &sites[7]) == 0.0);

            check!(cutoff_pair.site_pair_energy(&sites[0], &sites[6]) == 1.0);
            check!(cutoff_pair.site_pair_energy(&sites[1], &sites[7]) == 1.0);
        }
    }

    mod delta_energy_one {
        use super::*;

        #[rstest]
        fn site_outside(square: Closed<Hypercuboid<2>>) {
            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };
            let mut final_body = body.clone();
            final_body.properties.position[0] = 1.0;

            let microstate = Microstate::builder()
                .boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            let energy = PairwiseCutoff {
                r_cut: 0.0,
                evaluator: Isotropic(|_r| 0.0),
            };

            check!(
                energy.delta_energy_one(&microstate, 0, &final_body) ==
                f64::INFINITY
            );
        }

        #[test]
        fn body_exclusion() {
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
            microstate
                .extend_bodies([body_a, body_b])
                .expect("hard-coded bodies should be in the boundary");

            let cutoff_pair = PairwiseCutoff {
                r_cut: 1.0_f64.next_up(),
                evaluator: Isotropic(|_r| 1.0),
            };

            // Of all the pairs a distance 1.0 apart, only 2 are interbody pairs.
            // Moving body 0 to the left results in a -2.0 energy difference.
            check!(
                cutoff_pair.delta_energy_one(&microstate, 0, &body_a_final) ==
                -2.0
            );
        }

        #[test]
        fn random_moves() {
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

            let microstate_initial = Microstate::builder()
                .bodies([body_a, body_b])
                .try_build()
                .expect("hard-coded bodies should be in the boundary");

            let mut microstate_final = microstate_initial.clone();
            let lennard_jones: LennardJones = LennardJones {
                epsilon: 1.5,
                sigma: 1.25,
            };
            let cutoff_pair = PairwiseCutoff {
                r_cut: 5.0,
                evaluator: Isotropic(lennard_jones),
            };

            check!(cutoff_pair.total_energy(&microstate_initial) != 0.0);

            // Use `LennardJones` for validation because it is a varies with r and
            // will therefore show some changes for any moves (unlike `BoxCar`).
            // However, we need to avoid numerical errors when two sites get
            // too close. Randomly move the 2nd particle around within a
            // well-defined space where there will be no such overlaps.
            let mut rng = StdRng::seed_from_u64(0);
            let r_distribution =
                Uniform::new(3.0, 6.0).expect("hard-coded constants should be valid");
            let theta_distribution =
                Uniform::new(0.0, 2.0 * PI).expect("hard-coded constants should be valid");

            let mut new_body = body_template.clone();
            for _ in 0..1024 {
                let r = rng.sample(r_distribution);
                let theta = rng.sample(theta_distribution);
                new_body.properties.position = [r * theta.cos(), r * theta.sin()].into();

                let delta_energy_one =
                    cutoff_pair.delta_energy_one(&microstate_initial, 0, &new_body);
                microstate_final
                    .update_body_properties(0, new_body.properties)
                    .expect("generated bodies should be inside open boundaries");
                let delta_energy_total = cutoff_pair.total_energy(&microstate_final)
                    - cutoff_pair.total_energy(&microstate_initial);

                assert_relative_eq!(delta_energy_one, delta_energy_total, epsilon = 1e-10);
            }
        }
    }

    mod delta_energy_insert_remove {
        use super::*;

        #[rstest]
        fn site_outside(square: Closed<Hypercuboid<2>>) {
            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };
            let mut new_body = body.clone();
            new_body.properties.position[0] = 1.0;

            let microstate = Microstate::builder()
                .boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            let energy = PairwiseCutoff {
                r_cut: 0.0,
                evaluator: Isotropic(|_r| 0.0),
            };

            check!(
                energy.delta_energy_insert(&microstate, &new_body) ==
                f64::INFINITY
            );
        }

        #[test]
        fn body_exclusion() {
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
            microstate
                .extend_bodies([body_b])
                .expect("hard-coded bodies should be in the boundary");

            let cutoff_pair = PairwiseCutoff {
                r_cut: 1.0_f64.next_up(),
                evaluator: Isotropic(|_r| 1.0),
            };

            // Of all the pairs a distance 1.0 apart, only 2 are interbody pairs.
            // Moving body 0 to the left results in a -2.0 energy difference.
            check!(
                cutoff_pair.delta_energy_insert(&microstate, &body_a_new) ==
                2.0
            );

            microstate
                .add_body(body_a_new)
                .expect("hard-coded bodies should be in the boundary");
            check!(cutoff_pair.delta_energy_remove(&microstate, 1) == -2.0);
        }

        #[test]
        fn random_moves() {
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

            let microstate_initial = Microstate::builder()
                .bodies([body_a])
                .try_build()
                .expect("hard-coded bodies should be in the boundary");

            let mut microstate_final = microstate_initial.clone();
            let lennard_jones: LennardJones = LennardJones {
                epsilon: 1.5,
                sigma: 1.25,
            };
            let cutoff_pair = PairwiseCutoff {
                r_cut: 5.0,
                evaluator: Isotropic(lennard_jones),
            };

            // Use `LennardJones` for validation because it is a varies with r and
            // will therefore show some changes for any moves (unlike `BoxCar`).
            // However, we need to avoid numerical errors when two sites get
            // too close. Randomly insert the 2nd body in a well-defined area
            // to avoid this.
            let mut rng = StdRng::seed_from_u64(2);
            let r_distribution =
                Uniform::new(3.0, 6.0).expect("hard-coded constants should be valid");
            let theta_distribution =
                Uniform::new(0.0, 2.0 * PI).expect("hard-coded constants should be valid");

            for _ in 0..1024 {
                let r = rng.sample(r_distribution);
                let theta = rng.sample(theta_distribution);
                let mut new_body = body_template.clone();
                new_body.properties.position = [r * theta.cos(), r * theta.sin()].into();

                let delta_energy_insert =
                    cutoff_pair.delta_energy_insert(&microstate_initial, &new_body);
                let tag = microstate_final
                    .add_body(new_body)
                    .expect("generated bodies should be inside open boundaries");
                let delta_energy_total = cutoff_pair.total_energy(&microstate_final)
                    - cutoff_pair.total_energy(&microstate_initial);

                assert_relative_eq!(delta_energy_insert, delta_energy_total, epsilon = 1e-4);

                let delta_energy_remove = cutoff_pair.delta_energy_remove(&microstate_final, 1);
                assert_relative_eq!(delta_energy_remove, -delta_energy_total, epsilon = 1e-6);

                microstate_final.remove_body(
                    microstate_final.body_indices()[tag].expect("tag should be present"),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::check;
    use crate::{TotalEnergy, pairwise::HardSphere};
    use hoomd_geometry::shape::Hypercuboid;
    use hoomd_microstate::{boundary::Closed, property::Point};
    use hoomd_vector::Cartesian;

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

    mod cutoff_pair {
        use super::*;

        #[test]
        fn large_r_cut() {
            let mut microstate = Microstate::new();
            microstate
                .extend_bodies([
                    Body::point(Cartesian::from([0.0, 0.0])),
                    Body::point(Cartesian::from([0.0, 5.0])),
                ])
                .expect("hard-coded bodies should be in the boundary");

            // Ensure that PairwiseCutoff respects the r_cut value set.
            let r_cut = 5.0_f64.next_up();
            let cutoff_pair = PairwiseCutoff{
                r_cut,
                evaluator: HardSphere,
            };

            check!(cutoff_pair.total_energy(&microstate) == f64::INFINITY);

            let r_cut = 5.0_f64;
            let cutoff_pair = PairwiseCutoff{
                r_cut,
                evaluator: HardSphere,
            };

            check!(cutoff_pair.total_energy(&microstate) == 0.0);
        }

        #[test]
        fn body_exclusion() {
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
            microstate
                .extend_bodies([body_a, body_b])
                .expect("hard-coded bodies should be in the boundary");

            let r_cut = 1.0_f64.next_up();
            let cutoff_pair = PairwiseCutoff{
                r_cut,
                evaluator: HardSphere,
            };

            check!(cutoff_pair.total_energy(&microstate) == 0.0);

            let r_cut = 2.0_f64.next_up();
            let cutoff_pair = PairwiseCutoff{
                r_cut,
                evaluator: HardSphere,
            };

            check!(cutoff_pair.total_energy(&microstate) == f64::INFINITY);
        }
    }

    mod delta_energy_one {
        use super::*;

        #[rstest]
        fn site_outside(square: Closed<Hypercuboid<2>>) {
            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };
            let mut final_body = body.clone();
            final_body.properties.position[0] = 1.0;

            let microstate = Microstate::builder()
                .boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            let energy = PairwiseCutoff{
                r_cut: 0.0,
                evaluator: HardSphere,
            };

            check!(
                energy.delta_energy_one(&microstate, 0, &final_body) ==
                f64::INFINITY
            );
        }

        #[test]
        fn body_exclusion() {
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
            microstate
                .extend_bodies([body_a, body_b])
                .expect("hard-coded bodies should be in the boundary");

            let r_cut = 1.0_f64.next_up();
            let cutoff_pair = PairwiseCutoff{
                r_cut,
                evaluator: HardSphere,
            };

            // moving body a to the right generates overlaps
            check!(
                cutoff_pair.delta_energy_one(&microstate, 0, &body_a_overlap) ==
                f64::INFINITY
            );

            // moving body away results in no overlaps
            check!(
                cutoff_pair.delta_energy_one(&microstate, 0, &body_a_no_overlap) ==
                0.0
            );
        }
    }

    mod delta_energy_insert_remove {
        use super::*;

        #[rstest]
        fn site_outside(square: Closed<Hypercuboid<2>>) {
            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };
            let mut new_body = body.clone();
            new_body.properties.position[0] = 1.0;

            let microstate = Microstate::builder()
                .boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            let energy = PairwiseCutoff{
                r_cut: 0.0,
                evaluator: HardSphere,
            };

            check!(
                energy.delta_energy_insert(&microstate, &new_body) ==
                f64::INFINITY
            );
        }

        #[test]
        fn body_exclusion() {
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
            microstate
                .extend_bodies([body_b])
                .expect("hard-coded bodies should be in the boundary");

            let r_cut = 1.0_f64.next_up();
            let cutoff_pair = PairwiseCutoff{
                r_cut,
                evaluator: HardSphere,
            };

            check!(
                cutoff_pair.delta_energy_insert(&microstate, &body_a_new) ==
                f64::INFINITY
            );

            microstate
                .add_body(body_a_new)
                .expect("hard-coded bodies should be in the boundary");
            check!(cutoff_pair.delta_energy_remove(&microstate, 1) == 0.0);
        }
    }
}

// TODO: Test HardShape

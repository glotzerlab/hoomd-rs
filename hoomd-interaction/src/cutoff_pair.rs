// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `CutoffPair`
use crate::{DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, pairwise::IsotropicForce, NetBodyForce, NetSiteForce, SitePairEnergy, SitePairForce, TotalEnergy};
use hoomd_microstate::{Body, Microstate, Site, Transform, boundary::Wrap, property::Position};
use hoomd_vector::{Cartesian, InnerProduct, Vector, Metric};

/// Short-ranged pairwise interactions between sites.
///
/// Given an evaluator that implements [`SitePairEnergy`], [`CutoffPair`] represents:
///
/// ```math
/// U_\mathrm{total} = \sum_{i=0}^{N-1}\sum_{j=i+1}^{N-1} U\left(s_i, s_j \right) \left[ \left|\vec{r}_j - \vec{r}_i\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
/// ```
/// where $`U(s_i, s_j)`$ is the potential computed by [`CutoffPair::evaluator`],
/// $`s_i`$ is the full set of site properties for site i, $`\vec{r}_i`$ is
/// the position of site i, $`b_i`$ is the body tag that holds site *i*, and
/// $`\left[ \  \right]`$ denotes the Iverson bracket.
///
/// In other words, [`CutoffPair`] sums the energy for all pairs that are separated
/// by a distance less than `r_cut` and belong to different bodies.
///
/// For the evaluator, use [`Anisotropic`], [`Isotropic`] or your own custom type.
///
/// TODO: Reword this when [`CutoffPair`] also implements `SitePairForce`.
///
/// [`Anisotropic`]: crate::pairwise::Anisotropic
/// [`Isotropic`]: crate::pairwise::Isotropic
///
/// # Example
///
/// Basic usage:
/// ```
/// use hoomd_interaction::{
///     CutoffPair,
///     pairwise::{Isotropic, LennardJones},
/// };
///
/// let lennard_jones: LennardJones = LennardJones {
///     epsilon: 1.5,
///     sigma: 2.0,
/// };
/// let evaluator = Isotropic(lennard_jones);
/// let cutoff_pair = CutoffPair {
///     r_cut: 5.0,
///     evaluator,
/// };
/// ```
///
/// Set a custom potential using a closure:
/// ```
/// use hoomd_interaction::{CutoffPair, pairwise::Isotropic};
///
/// let cutoff_pair = CutoffPair {
///     r_cut: 3.0,
///     evaluator: Isotropic(|r: f64| 1.0 / (r.powi(12))),
/// };
/// ```
///
/// Implement a custom potential via a type:
/// ```
/// use hoomd_interaction::{
///     CutoffPair,
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
/// let cutoff_pair = CutoffPair {
///     r_cut: 2.0,
///     evaluator: Isotropic(custom),
/// };
/// ```
pub struct CutoffPair<E> {
    /// The distance beyond which all pairwise interactions evaluate to 0.
    pub r_cut: f64,

    /// Computes the pairwise energies and forces.
    pub evaluator: E,
}

impl<E> CutoffPair<E> {
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
    ///     CutoffPair,
    ///     pairwise::{Isotropic, LennardJones},
    /// };
    /// use hoomd_microstate::{Body, MicrostateBuilder, Site};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let lennard_jones: LennardJones = LennardJones {
    ///     epsilon: 1.0,
    ///     sigma: 1.0,
    /// };
    /// let evaluator = Isotropic(lennard_jones);
    /// let cutoff_pair = CutoffPair {
    ///     r_cut: 2.5,
    ///     evaluator,
    /// };
    ///
    /// let body_a = Body::point(Cartesian::from([0.0, 0.0]));
    /// let body_b = Body::point(Cartesian::from([0.0, 3.0]));
    /// let body_c = Body::point(Cartesian::from([0.0, -2.0f64.powf(1.0 / 6.0)]));
    ///
    /// let microstate = MicrostateBuilder::new()
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


    /// Compute the cutoff pairwise force on one site from another site.
    /// TODO: Add example.
    ///
    #[inline]
    pub fn site_pair_force<V, S>(&self, a: &Site<S>, b: &Site<S>) -> V
    where
        E: SitePairForce<V, S>,
        S: Position<Position = V>,
        V: Vector + Default + InnerProduct + Metric,
    {
        let r = (a.properties.position()).distance(b.properties.position());
        if r < self.r_cut && a.body_tag != b.body_tag {
            self.evaluator
                .site_pair_force(&a.properties, &b.properties)
        } else {
            V::default()
        }

    }
}

impl<P, B, S, C, E> TotalEnergy<Microstate<B, S, C>> for CutoffPair<E>
where
    E: SitePairEnergy<S>,
    S: Position<Position = P>,
    P: Metric,
{
    /// Compute the total energy of the microstate contributed by functions on pairs of sites.
    ///
    /// # Example
    /// ```
    /// use hoomd_interaction::{CutoffPair, SitePairEnergy, TotalEnergy,
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
    /// let cutoff_pair = CutoffPair { r_cut: 2.5, evaluator };
    ///
    /// // The potential energy is set to 0 beyond r_cut when computed by `CutoffPair`.
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
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, C>) -> f64 {
        let mut total = 0.0;
        for site_i in microstate.sites() {
            for site_j in microstate
                .iter_sites_near(site_i.properties.position(), self.r_cut)
                .filter(|s| site_i.site_tag < s.site_tag && site_i.body_tag != s.body_tag)
            {
                total += self
                    .evaluator
                    .site_pair_energy(&site_i.properties, &site_j.properties);
            }
        }

        total
    }
}

/// Evaluate the change in energy contributed by `CutoffPair` when one body is updated.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     CutoffPair, DeltaEnergyOne,
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
/// let cutoff_pair = CutoffPair {
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
impl<P, B, S, C, E> DeltaEnergyOne<B, S, C> for CutoffPair<E>
where
    E: SitePairEnergy<S>,
    B: Transform<S>,
    S: Position<Position = P>,
    C: Wrap<B> + Wrap<S>,
    P: Metric,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let body_tag = initial_microstate.bodies()[body_index].tag;

        // CutoffPair cannot implement site_energy to centrally calculate the
        // energy of one site with the rest of the system because the resulting
        // TotalEnergy calculation would double count interactions. Therefore,
        // this (and other codes) that need to sum over specific pairs implement
        // the necessary loops and call `site_pair_energy` directly.
        let site_energy = |site_properties: &S| {
            initial_microstate
                .iter_sites_near(site_properties.position(), self.r_cut)
                .filter(|s| body_tag != s.body_tag)
                .fold(0.0, |total, site_j| {
                    total
                        + self
                            .evaluator
                            .site_pair_energy(site_properties, &site_j.properties)
                })
        };

        let mut energy_final = 0.0;
        for s in &final_body.sites {
            match initial_microstate
                .boundary()
                .wrap(final_body.properties.transform(s))
            {
                Err(_) => return f64::INFINITY,
                Ok(wrapped_site) => energy_final += site_energy(&wrapped_site),
            }
        }

        let energy_initial = initial_microstate
            .iter_body_sites(body_index)
            .fold(0.0, |total, site_i| total + site_energy(&site_i.properties));

        energy_final - energy_initial
    }
}

/// Evaluate the change in energy contributed by `CutoffPair` when one body is inserted.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     CutoffPair, DeltaEnergyInsert,
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
/// let cutoff_pair = CutoffPair {
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
impl<P, B, S, C, E> DeltaEnergyInsert<B, S, C> for CutoffPair<E>
where
    E: SitePairEnergy<S>,
    B: Transform<S>,
    S: Position<Position = P>,
    C: Wrap<B> + Wrap<S>,
    P: Metric,
{
    #[inline]
    fn delta_energy_insert(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        new_body: &Body<B, S>,
    ) -> f64 {
        // The new body is not yet in the microstate, so there is no need to
        // filter matching body tags. The new body does not yet have a tag.
        let site_energy = |site_properties: &S| {
            initial_microstate
                .iter_sites_near(site_properties.position(), self.r_cut)
                .fold(0.0, |total, site_j| {
                    total
                        + self
                            .evaluator
                            .site_pair_energy(site_properties, &site_j.properties)
                })
        };

        let mut energy_final = 0.0;
        for s in &new_body.sites {
            match initial_microstate
                .boundary()
                .wrap(new_body.properties.transform(s))
            {
                Err(_) => return f64::INFINITY,
                Ok(wrapped_site) => energy_final += site_energy(&wrapped_site),
            }
        }

        energy_final
    }
}

/// Evaluate the change in energy contributed by `CutoffPair` when one body is removed.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     CutoffPair, DeltaEnergyRemove,
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
/// let cutoff_pair = CutoffPair {
///     r_cut: 1.5,
///     evaluator,
/// };
///
/// let delta_energy = cutoff_pair.delta_energy_remove(&microstate, 0);
/// assert_eq!(delta_energy, -2.0);
/// # Ok(())
/// # }
/// ```
impl<P, B, S, C, E> DeltaEnergyRemove<B, S, C> for CutoffPair<E>
where
    E: SitePairEnergy<S>,
    S: Position<Position = P>,
    P: Metric,
{
    #[inline]
    fn delta_energy_remove(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        body_index: usize,
    ) -> f64 {
        let body_tag = initial_microstate.bodies()[body_index].tag;

        // CutoffPair cannot implement site_energy to centrally calculate the
        // energy of one site with the rest of the system because the resulting
        // TotalEnergy calculation would double count interactions. Therefore,
        // this (and other codes) that need to sum over specific pairs implement
        // the necessary loops and call `site_pair_energy` directly.
        let site_energy = |site_properties: &S| {
            initial_microstate
                .iter_sites_near(site_properties.position(), self.r_cut)
                .filter(|s| body_tag != s.body_tag)
                .fold(0.0, |total, site_j| {
                    total
                        + self
                            .evaluator
                            .site_pair_energy(site_properties, &site_j.properties)
                })
        };

        let energy_initial = initial_microstate
            .iter_body_sites(body_index)
            .fold(0.0, |total, site_i| total + site_energy(&site_i.properties));

        -energy_initial
    }
}

/** Compute the net cutoff pairwise force on a single body.
TODO: Add example.
*/
impl<V, B, S, C, E> NetBodyForce<V, B, S, C> for CutoffPair<E>
where
    V: Vector + Default + InnerProduct + Metric,
    B: Transform<S>,
    S: Position<Position = V>,
    E: SitePairForce<V, S>,
{
    #[inline]
    fn net_force_on_body(&self, microstate: &Microstate<B, S, C>, body_index: usize) -> V {
        let mut total = V::default();
        for site in microstate.iter_body_sites(body_index) {
            total += self.net_force_on_site(microstate, site);
        }
        total
    }

    // TODO: consider whether to add state here to track which body/sites have
    // been calculated already, which would prevent double-calculations.
}

/** Compute the net cutoff pairwise force on a single site.
TODO: Add example.
*/
impl<V, B, S, C, E> NetSiteForce<V, B, S, C> for CutoffPair<E>
where
    V: Vector + Default + InnerProduct + Metric,
    B: Transform<S>,
    S: Position<Position = V>,
    E: SitePairForce<V, S>,
{
    #[inline]
    fn net_force_on_site(&self, microstate: &Microstate<B, S, C>, site: &Site<S>) -> V {
        let mut total = V::default();
        for other_site in microstate
            .iter_sites_near(site.properties.position(), self.r_cut)
            .filter(|s| site.body_tag != s.body_tag)
        {
            total += self
                .evaluator
                .site_pair_force(&site.properties, &other_site.properties);
        }
        total

    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TotalEnergy,
        pairwise::{Isotropic, LennardJones},
    };
    use hoomd_geometry::shape::Hypercuboid;
    use hoomd_microstate::{
        MicrostateBuilder,
        boundary::{Closed, Open},
        property::Point,
    };
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
        fn microstate() -> Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open> {
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
        fn blanket_fn(microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>) {
            // Ensure that closures can be used as IsotropicEnergy
            let cutoff_pair = CutoffPair {
                r_cut: 2.0,
                evaluator: Isotropic(|r| 1.0 / (r * 2.0)),
            };

            // Two pairs at a distance of 1.0 each with energy 1/2.
            assert_eq!(cutoff_pair.total_energy(&microstate), 1.0);

            let sites = microstate.sites();
            assert_eq!(cutoff_pair.site_pair_energy(&sites[0], &sites[0]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[0], &sites[1]), 0.5);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[0], &sites[2]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[0], &sites[3]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[1], &sites[1]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[1], &sites[2]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[1], &sites[3]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[2], &sites[2]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[2], &sites[3]), 0.5);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[3], &sites[3]), 0.0);
        }

        #[rstest]
        fn large_r_cut(microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>) {
            // Ensure that CutoffPair respects the r_cut value set.
            let cutoff_pair = CutoffPair {
                r_cut: 5.0_f64.next_up(),
                evaluator: Isotropic(|r| 1.0 / (r * 2.0)),
            };

            // Two pairs at a distance of 1.0 each with energy 1/2.
            // Plus two pairs at a distance of 5.0 with energy 1/10
            assert_eq!(cutoff_pair.total_energy(&microstate), 1.2);
        }

        #[test]
        fn body_exclusion() {
            // Ensure that CutoffPair excludes pairs in the same body.
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

            let cutoff_pair = CutoffPair {
                r_cut: 1.0_f64.next_up(),
                evaluator: Isotropic(|_r| 1.0),
            };

            // Of all the pairs a distance 1.0 apart, only 2 are interbody pairs.
            assert_eq!(cutoff_pair.total_energy(&microstate), 2.0);

            let sites = microstate.sites();
            assert_eq!(cutoff_pair.site_pair_energy(&sites[0], &sites[0]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[0], &sites[1]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[0], &sites[2]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[0], &sites[3]), 0.0);

            assert_eq!(cutoff_pair.site_pair_energy(&sites[4], &sites[4]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[4], &sites[5]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[4], &sites[6]), 0.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[4], &sites[7]), 0.0);

            assert_eq!(cutoff_pair.site_pair_energy(&sites[0], &sites[6]), 1.0);
            assert_eq!(cutoff_pair.site_pair_energy(&sites[1], &sites[7]), 1.0);
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

            let microstate = MicrostateBuilder::with_boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            let energy = CutoffPair {
                r_cut: 0.0,
                evaluator: Isotropic(|_r| 0.0),
            };

            assert_eq!(
                energy.delta_energy_one(&microstate, 0, &final_body),
                f64::INFINITY
            );
        }

        #[test]
        fn body_exclusion() {
            // Ensure that CutoffPair.delta_energy_one excludes pairs in the same body.
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

            let cutoff_pair = CutoffPair {
                r_cut: 1.0_f64.next_up(),
                evaluator: Isotropic(|_r| 1.0),
            };

            // Of all the pairs a distance 1.0 apart, only 2 are interbody pairs.
            // Moving body 0 to the left results in a -2.0 energy difference.
            assert_eq!(
                cutoff_pair.delta_energy_one(&microstate, 0, &body_a_final),
                -2.0
            );
        }

        #[test]
        fn random_moves() {
            // Ensure that CutoffPair.delta_energy_one is consistent with TotalEnergy
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

            let microstate_initial = MicrostateBuilder::new()
                .bodies([body_a, body_b])
                .try_build()
                .expect("hard-coded bodies should be in the boundary");

            let mut microstate_final = microstate_initial.clone();
            let lennard_jones: LennardJones = LennardJones {
                epsilon: 1.5,
                sigma: 1.25,
            };
            let cutoff_pair = CutoffPair {
                r_cut: 5.0,
                evaluator: Isotropic(lennard_jones),
            };

            assert!(cutoff_pair.total_energy(&microstate_initial) != 0.0);

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

            let microstate = MicrostateBuilder::with_boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            let energy = CutoffPair {
                r_cut: 0.0,
                evaluator: Isotropic(|_r| 0.0),
            };

            assert_eq!(
                energy.delta_energy_insert(&microstate, &new_body),
                f64::INFINITY
            );
        }

        #[test]
        fn body_exclusion() {
            // Ensure that CutoffPair.delta_energy_insert excludes pairs in the same body.
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

            let cutoff_pair = CutoffPair {
                r_cut: 1.0_f64.next_up(),
                evaluator: Isotropic(|_r| 1.0),
            };

            // Of all the pairs a distance 1.0 apart, only 2 are interbody pairs.
            // Moving body 0 to the left results in a -2.0 energy difference.
            assert_eq!(
                cutoff_pair.delta_energy_insert(&microstate, &body_a_new),
                2.0
            );

            microstate
                .add_body(body_a_new)
                .expect("hard-coded bodies should be in the boundary");
            assert_eq!(cutoff_pair.delta_energy_remove(&microstate, 1), -2.0);
        }

        #[test]
        fn random_moves() {
            // Ensure that CutoffPair.delta_energy_insert is consistent with TotalEnergy
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

            let microstate_initial = MicrostateBuilder::new()
                .bodies([body_a])
                .try_build()
                .expect("hard-coded bodies should be in the boundary");

            let mut microstate_final = microstate_initial.clone();
            let lennard_jones: LennardJones = LennardJones {
                epsilon: 1.5,
                sigma: 1.25,
            };
            let cutoff_pair = CutoffPair {
                r_cut: 5.0,
                evaluator: Isotropic(lennard_jones),
            };

            // Use `LennardJones` for validation because it is a varies with r and
            // will therefore show some changes for any moves (unlike `BoxCar`).
            // However, we need to avoid numerical errors when two sites get
            // too close. Randomly insert the 2nd body in a well-defined area
            // to avoid this.
            let mut rng = StdRng::seed_from_u64(0);
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

                assert_relative_eq!(delta_energy_insert, delta_energy_total, epsilon = 1e-6);

                let delta_energy_remove = cutoff_pair.delta_energy_remove(&microstate_final, 1);
                assert_relative_eq!(delta_energy_remove, -delta_energy_total, epsilon = 1e-6);

                microstate_final.remove_body(
                    microstate_final.body_indices()[tag].expect("tag should be present"),
                );
            }
        }
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `External`

use serde::{Deserialize, Serialize};
use crate::{
    DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, MaximumInteractionRange, NetSiteForceAndTorque, SiteEnergy, SiteForce, TotalEnergy
};
use hoomd_microstate::{boundary::Wrap, property::Position, Body, Microstate, Site, Transform};
use hoomd_vector::Wedge;

/// Interactions between sites and external fields.
///
/// Given an inner type that implements [`SiteEnergy`], [`External`] represents:
///
/// ```math
/// U_\mathrm{total} = \sum_{i=0}^{N-1} U\left( s_i \right)
/// ```
/// where $`s_i`$ is the full set of site properties for site i.
///
/// For the inner type, use one from [`external`] or your own custom type.
///
/// [`external`]: crate::external
///
/// # Examples
///
/// A linear external potential:
/// ```
/// use hoomd_interaction::{External, TotalEnergy, external::Linear};
/// use hoomd_microstate::{Body, Microstate, property::Point};
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.extend_bodies([
///     Body::point(Cartesian::from([1.0, 0.0])),
///     Body::point(Cartesian::from([-1.0, 2.0])),
/// ])?;
///
/// let linear = External(Linear {
///     alpha: 1.0,
///     plane_origin: Cartesian::default(),
///     plane_normal: [0.0, 1.0].try_into()?,
/// });
///
/// let total_energy = linear.total_energy(&microstate);
/// assert_eq!(total_energy, 2.0);
/// # Ok(())
/// # }
/// ```
///
/// Infinite interaction with a wall:
/// ```
/// use hoomd_interaction::{External, SiteEnergy, TotalEnergy};
/// use hoomd_microstate::{Body, Microstate, property::Point};
/// use hoomd_vector::Cartesian;
///
/// struct Wall;
///
/// impl SiteEnergy<Point<Cartesian<2>>> for Wall {
///     fn site_energy(&self, site_properties: &Point<Cartesian<2>>) -> f64 {
///         if site_properties.position[1].abs() < 1.0 {
///             f64::INFINITY
///         } else {
///             0.0
///         }
///     }
///
///     fn is_only_infinite_or_zero() -> bool {
///         true
///     }
/// }
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut microstate = Microstate::new();
///     microstate.extend_bodies([
///         Body::point(Cartesian::from([1.0, 1.25])),
///         Body::point(Cartesian::from([-1.0, 2.0])),
///     ])?;
///
///     let wall = External(Wall);
///
///     let total_energy = wall.total_energy(&microstate);
///     assert_eq!(total_energy, 0.0);
///     Ok(())
/// }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct External<E>(pub E);

impl<B, S, X, C, E> TotalEnergy<Microstate<B, S, X, C>> for External<E>
where
    E: SiteEnergy<S>,
{
    /// Compute the total energy of the microstate contributed by functions of a single site.
    ///
    /// The sum over sites differs from HOOMD-blue where external energies are
    /// evaluated only at the body centers. In general, hoomd-rs interactions apply
    /// to sites. Use a custom implementation to compute energies over body centers.
    ///
    /// # Examples
    ///
    /// A linear external potential:
    /// ```
    /// use hoomd_interaction::{External, TotalEnergy, external::Linear};
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([1.0, 0.0])),
    ///     Body::point(Cartesian::from([-1.0, 2.0])),
    /// ])?;
    ///
    /// let linear = External(Linear {
    ///     alpha: 1.0,
    ///     plane_origin: Cartesian::default(),
    ///     plane_normal: [0.0, 1.0].try_into()?,
    /// });
    ///
    /// let total_energy = linear.total_energy(&microstate);
    /// assert_eq!(total_energy, 2.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Infinite interaction with a wall:
    /// ```
    /// use hoomd_interaction::{External, SiteEnergy, TotalEnergy};
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// struct Wall;
    ///
    /// impl SiteEnergy<Point<Cartesian<2>>> for Wall {
    ///     fn site_energy(&self, site_properties: &Point<Cartesian<2>>) -> f64 {
    ///         if site_properties.position[1].abs() < 1.0 {
    ///             f64::INFINITY
    ///         } else {
    ///             0.0
    ///         }
    ///     }
    ///
    ///     fn is_only_infinite_or_zero() -> bool {
    ///         true
    ///     }
    /// }
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let mut microstate = Microstate::new();
    ///     microstate.extend_bodies([
    ///         Body::point(Cartesian::from([1.0, 1.25])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])?;
    ///
    ///     let wall = External(Wall);
    ///
    ///     let total_energy = wall.total_energy(&microstate);
    ///     assert_eq!(total_energy, 0.0);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, X, C>) -> f64 {
        let mut total = 0.0;
        for site in microstate.sites() {
            let one = self.0.site_energy(&site.properties);
            if one == f64::INFINITY {
                return one;
            }
            total += one;
        }

        total
    }

    /// Compute the difference in energy between two microstates.
    ///
    /// Returns $` E_\mathrm{final} - E_\mathrm{initial} `$.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::{External, TotalEnergy, external::Linear};
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate_a = Microstate::new();
    /// microstate_a.extend_bodies([
    ///     Body::point(Cartesian::from([1.0, 0.0])),
    ///     Body::point(Cartesian::from([-1.0, 2.0])),
    /// ])?;
    ///
    /// let mut microstate_b = Microstate::new();
    /// microstate_b.extend_bodies([
    ///     Body::point(Cartesian::from([1.0, 1.0])),
    ///     Body::point(Cartesian::from([-1.0, 2.0])),
    /// ])?;
    ///
    /// let linear = External(Linear {
    ///     alpha: 1.0,
    ///     plane_origin: Cartesian::default(),
    ///     plane_normal: [0.0, 1.0].try_into()?,
    /// });
    ///
    /// let delta_energy_total =
    ///     linear.delta_energy_total(&microstate_a, &microstate_b);
    /// assert_eq!(delta_energy_total, 1.0);
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
        for site in final_microstate.sites() {
            let one = self.0.site_energy(&site.properties);
            if one == f64::INFINITY {
                return one;
            }
            energy_final += one;
        }

        let mut energy_initial = 0.0;
        if !E::is_only_infinite_or_zero() {
            for site in initial_microstate.sites() {
                let one = self.0.site_energy_initial(&site.properties);
                if one == f64::INFINITY {
                    return -one;
                }
                energy_initial += one;
            }
        }

        energy_final - energy_initial
    }
}

impl<P, B, S, X, C, E> DeltaEnergyOne<B, S, X, C> for External<E>
where
    E: SiteEnergy<S>,
    B: Transform<S>,
    S: Position<Position = P>,
    C: Wrap<B> + Wrap<S>,
{
    /// Evaluate the change in energy contributed by `External` when a single body is updated.
    ///
    /// # Examples
    ///
    /// A linear external potential:
    /// ```
    /// use hoomd_interaction::{DeltaEnergyOne, External, external::Linear};
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;
    ///
    /// let linear = External(Linear {
    ///     alpha: 1.0,
    ///     plane_origin: Cartesian::default(),
    ///     plane_normal: [0.0, 1.0].try_into()?,
    /// });
    ///
    /// let delta_energy = linear.delta_energy_one(
    ///     &microstate,
    ///     0,
    ///     &Body::point([0.0, -1.0].into()),
    /// );
    /// assert_eq!(delta_energy, -1.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Infinite interaction with a wall:
    /// ```
    /// use hoomd_interaction::{DeltaEnergyOne, External, SiteEnergy};
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// struct Wall;
    ///
    /// impl SiteEnergy<Point<Cartesian<2>>> for Wall {
    ///     fn site_energy(&self, site_properties: &Point<Cartesian<2>>) -> f64 {
    ///         if site_properties.position[1].abs() < 1.0 {
    ///             f64::INFINITY
    ///         } else {
    ///             0.0
    ///         }
    ///     }
    ///
    ///     fn is_only_infinite_or_zero() -> bool {
    ///         true
    ///     }
    /// }
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let mut microstate = Microstate::new();
    ///     microstate.extend_bodies([
    ///         Body::point(Cartesian::from([1.0, 1.25])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])?;
    ///
    ///     let wall = External(Wall);
    ///
    ///     let delta_energy = wall.delta_energy_one(
    ///         &microstate,
    ///         0,
    ///         &Body::point([0.0, -0.5].into()),
    ///     );
    ///     assert_eq!(delta_energy, f64::INFINITY);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let mut energy_final = 0.0;
        for s in &final_body.sites {
            match initial_microstate
                .boundary()
                .wrap(final_body.properties.transform(s))
            {
                Ok(wrapped_site) => {
                    let one = self.0.site_energy(&wrapped_site);
                    if one == f64::INFINITY {
                        return one;
                    }
                    energy_final += one;
                }
                Err(_) => return f64::INFINITY,
            }
        }

        let energy_initial = if E::is_only_infinite_or_zero() {
            0.0
        } else {
            initial_microstate
                .iter_body_sites(body_index)
                .fold(0.0, |total, s| {
                    total + self.0.site_energy_initial(&s.properties)
                })
        };

        energy_final - energy_initial
    }
}

impl<P, B, S, X, C, E> DeltaEnergyInsert<B, S, X, C> for External<E>
where
    E: SiteEnergy<S>,
    B: Transform<S>,
    S: Position<Position = P>,
    C: Wrap<B> + Wrap<S>,
{
    /// Evaluate the change in energy contributed by `External` when a single body is inserted.
    ///
    /// # Examples
    ///
    /// A linear external potential:
    /// ```
    /// use hoomd_interaction::{DeltaEnergyInsert, External, external::Linear};
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;
    ///
    /// let linear = External(Linear {
    ///     alpha: 1.0,
    ///     plane_origin: Cartesian::default(),
    ///     plane_normal: [0.0, 1.0].try_into()?,
    /// });
    ///
    /// let delta_energy = linear
    ///     .delta_energy_insert(&microstate, &Body::point([0.0, -1.0].into()));
    /// assert_eq!(delta_energy, -1.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Infinite interaction with a wall:
    /// ```
    /// use hoomd_interaction::{DeltaEnergyInsert, External, SiteEnergy};
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// struct Wall;
    ///
    /// impl SiteEnergy<Point<Cartesian<2>>> for Wall {
    ///     fn site_energy(&self, site_properties: &Point<Cartesian<2>>) -> f64 {
    ///         if site_properties.position[1].abs() < 1.0 {
    ///             f64::INFINITY
    ///         } else {
    ///             0.0
    ///         }
    ///     }
    ///
    ///     fn is_only_infinite_or_zero() -> bool {
    ///         true
    ///     }
    /// }
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let mut microstate = Microstate::new();
    ///     microstate.extend_bodies([
    ///         Body::point(Cartesian::from([1.0, 1.25])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])?;
    ///
    ///     let wall = External(Wall);
    ///
    ///     let delta_energy = wall
    ///         .delta_energy_insert(&microstate, &Body::point([0.0, -0.5].into()));
    ///     assert_eq!(delta_energy, f64::INFINITY);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    fn delta_energy_insert(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        new_body: &Body<B, S>,
    ) -> f64 {
        let mut energy_final = 0.0;
        for s in &new_body.sites {
            match initial_microstate
                .boundary()
                .wrap(new_body.properties.transform(s))
            {
                Ok(wrapped_site) => {
                    let one = self.0.site_energy(&wrapped_site);
                    if one == f64::INFINITY {
                        return one;
                    }
                    energy_final += one;
                }
                Err(_) => return f64::INFINITY,
            }
        }

        energy_final
    }
}

impl<B, S, X, C, E> DeltaEnergyRemove<B, S, X, C> for External<E>
where
    E: SiteEnergy<S>,
{
    /// Evaluate the change in energy contributed by `External` when a single body is removed.
    ///
    /// # Examples
    ///
    /// A linear external potential:
    /// ```
    /// use hoomd_interaction::{DeltaEnergyRemove, External, external::Linear};
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.add_body(Body::point(Cartesian::from([0.0, 1.0])))?;
    ///
    /// let linear = External(Linear {
    ///     alpha: 1.0,
    ///     plane_origin: Cartesian::default(),
    ///     plane_normal: [0.0, 1.0].try_into()?,
    /// });
    ///
    /// let delta_energy = linear.delta_energy_remove(&microstate, 0);
    /// assert_eq!(delta_energy, -1.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Infinite interaction with a wall:
    /// ```
    /// use hoomd_interaction::{DeltaEnergyRemove, External, SiteEnergy};
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// struct Wall;
    ///
    /// impl SiteEnergy<Point<Cartesian<2>>> for Wall {
    ///     fn site_energy(&self, site_properties: &Point<Cartesian<2>>) -> f64 {
    ///         if site_properties.position[1].abs() < 1.0 {
    ///             f64::INFINITY
    ///         } else {
    ///             0.0
    ///         }
    ///     }
    ///
    ///     fn is_only_infinite_or_zero() -> bool {
    ///         true
    ///     }
    /// }
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let mut microstate = Microstate::new();
    ///     microstate.extend_bodies([
    ///         Body::point(Cartesian::from([1.0, 1.25])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])?;
    ///
    ///     let wall = External(Wall);
    ///
    ///     let delta_energy = wall.delta_energy_remove(&microstate, 0);
    ///     assert_eq!(delta_energy, 0.0);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    fn delta_energy_remove(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
    ) -> f64 {
        if E::is_only_infinite_or_zero() {
            return 0.0;
        }

        let energy_initial = initial_microstate
            .iter_body_sites(body_index)
            .fold(0.0, |total, s| {
                total + self.0.site_energy_initial(&s.properties)
            });

        -energy_initial
    }
}

impl<V, B, S, X, C, E> NetSiteForceAndTorque<V, B, S, X, C> for External<E>
where
    V: Wedge,
    V::Bivector: Default,
    E: SiteForce<S, Force = V>
{
    /// Calculate the net force and torque.
    /// 
    /// `microstate` describes the system configuration and the target `site` 
    /// within the system for which the net force and torque
    /// $`\mathbf{f}_\alpha`$, $`\boldsymbol{\tau}_{\alpha}`$ are calculated.
    #[inline]
    fn net_site_force_and_torque(&self, _microstate: &Microstate<B, S, X, C>, site: &Site<S>) -> (V, V::Bivector) {
        let force = self.0.site_force(&site.properties);
        let torque = V::Bivector::default();

        (force, torque)
    }
}

impl<E> MaximumInteractionRange for External<E> {
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        // External interactions are not applied between pairs of particles.
        0.0
    }
}

#[cfg(test)]
mod test_finite {
    use super::*;

    use crate::external::ConstantForce;
    use assert2::check;
    use hoomd_geometry::shape::Rectangle;
    use hoomd_microstate::{
        Body, Microstate,
        boundary::{Closed, Open},
        property::{Point, Position},
    };
    use hoomd_vector::{Cartesian, Unit};
    use rstest::*;

    struct TestSE;

    impl<S> SiteEnergy<S> for TestSE
    where
        S: Position<Position = Cartesian<2>>,
    {
        fn site_energy(&self, site_properties: &S) -> f64 {
            site_properties.position()[0] + site_properties.position()[1]
        }
    }

    mod site_energy {
        use super::*;
        use hoomd_microstate::SiteKey;
        use hoomd_spatial::AllPairs;

        #[fixture]
        fn microstate()
        -> Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, AllPairs<SiteKey>, Open> {
            let mut microstate = Microstate::new();
            microstate
                .extend_bodies([
                    Body::point(Cartesian::from([1.0, 0.0])),
                    Body::point(Cartesian::from([-1.0, 3.0])),
                ])
                .expect("hard-coded bodies should be in the boundary");
            microstate
        }

        #[rstest]
        fn single_total(
            microstate: Microstate<
                Point<Cartesian<2>>,
                Point<Cartesian<2>>,
                AllPairs<SiteKey>,
                Open,
            >,
        ) {
            let test_se = TestSE;
            let single = External(test_se);

            check!(single.total_energy(&microstate) == 3.0);
        }

        #[rstest]
        fn single_site(
            microstate: Microstate<
                Point<Cartesian<2>>,
                Point<Cartesian<2>>,
                AllPairs<SiteKey>,
                Open,
            >,
        ) {
            let test_se = TestSE;
            let single = External(test_se);

            check!(single.0.site_energy(&microstate.sites()[0].properties) == 1.0);
            check!(single.0.site_energy(&microstate.sites()[1].properties) == 2.0);
        }
    }
    mod delta_energy {
        use super::*;

        struct Zero;

        impl SiteEnergy<Point<Cartesian<2>>> for Zero {
            fn site_energy(&self, _site_properties: &Point<Cartesian<2>>) -> f64 {
                0.0
            }
        }

        #[test]
        fn site_outside() {
            let cuboid = Rectangle::with_equal_edges(
                4.0.try_into()
                    .expect("hard-coded constant should be positive"),
            );
            let square = Closed(cuboid);

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

            let energy = External(Zero);

            check!(energy.delta_energy_one(&microstate, 0, &final_body) == f64::INFINITY);
            check!(energy.delta_energy_insert(&microstate, &final_body) == f64::INFINITY);
        }

        #[test]
        fn delta_energy() -> anyhow::Result<()> {
            let cuboid = Rectangle::with_equal_edges(
                4.0.try_into()
                    .expect("hard-coded constant should be positive"),
            );
            let square = Closed(cuboid);

            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([0.0, 0.0]))].into(),
            };
            let mut final_body = body.clone();
            final_body.properties.position[1] = 0.5;

            let microstate = Microstate::builder()
                .boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            let plane_normal = Unit::<Cartesian<2>>::try_from([0.0, 1.0])
                .expect("the hard-coded vector is not zero");
            let energy = External(ConstantForce {
                plane_origin: [0.0, -1.0].into(),
                plane_normal,
                alpha: 4.0,
            });

            check!(energy.delta_energy_one(&microstate, 0, &final_body) == 2.0);
            check!(energy.delta_energy_insert(&microstate, &final_body) == 6.0);
            check!(energy.delta_energy_remove(&microstate, 0) == -4.0);

            let mut microstate_final = microstate.clone();
            microstate_final.update_body_properties(0, final_body.properties)?;

            check!(energy.delta_energy_total(&microstate, &microstate_final) == 2.0);

            Ok(())
        }
    }
}

#[cfg(test)]
mod test_infinite {
    use super::*;
    use assert2::check;
    use hoomd_geometry::shape::Rectangle;
    use hoomd_microstate::{
        Body, Microstate,
        boundary::{Closed, Open},
        property::{Point, Position},
    };
    use hoomd_vector::Cartesian;
    use rstest::*;

    struct TestSO;

    impl<S> SiteEnergy<S> for TestSO
    where
        S: Position<Position = Cartesian<2>>,
    {
        fn site_energy(&self, site_properties: &S) -> f64 {
            if site_properties.position()[1].abs() < 1.0 {
                f64::INFINITY
            } else {
                0.0
            }
        }

        fn is_only_infinite_or_zero() -> bool {
            true
        }
    }

    mod site_energy {
        use super::*;
        use hoomd_microstate::SiteKey;
        use hoomd_spatial::AllPairs;

        #[fixture]
        fn microstate()
        -> Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, AllPairs<SiteKey>, Open> {
            let mut microstate = Microstate::new();
            microstate
                .extend_bodies([
                    Body::point(Cartesian::from([1.0, -2.0])),
                    Body::point(Cartesian::from([-1.0, 3.0])),
                ])
                .expect("hard-coded bodies should be in the boundary");
            microstate
        }

        #[fixture]
        fn overlapping_microstate()
        -> Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, AllPairs<SiteKey>, Open> {
            let mut microstate = Microstate::new();
            microstate
                .extend_bodies([
                    Body::point(Cartesian::from([1.0, 0.75])),
                    Body::point(Cartesian::from([-1.0, 3.0])),
                ])
                .expect("hard-coded bodies should be in the boundary");
            microstate
        }

        #[rstest]
        fn single_total_0(
            microstate: Microstate<
                Point<Cartesian<2>>,
                Point<Cartesian<2>>,
                AllPairs<SiteKey>,
                Open,
            >,
        ) {
            let single = External(TestSO);

            check!(single.total_energy(&microstate) == 0.0);
        }

        #[rstest]
        fn single_total_inf(
            overlapping_microstate: Microstate<
                Point<Cartesian<2>>,
                Point<Cartesian<2>>,
                AllPairs<SiteKey>,
                Open,
            >,
        ) {
            let single = External(TestSO);

            check!(single.total_energy(&overlapping_microstate) == f64::INFINITY);
        }

        #[rstest]
        fn single_site_0(
            microstate: Microstate<
                Point<Cartesian<2>>,
                Point<Cartesian<2>>,
                AllPairs<SiteKey>,
                Open,
            >,
        ) {
            let single = External(TestSO);

            check!(single.0.site_energy(&microstate.sites()[0].properties) == 0.0);
            check!(single.0.site_energy(&microstate.sites()[1].properties) == 0.0);
        }

        #[rstest]
        fn single_site_inf(
            overlapping_microstate: Microstate<
                Point<Cartesian<2>>,
                Point<Cartesian<2>>,
                AllPairs<SiteKey>,
                Open,
            >,
        ) {
            let single = External(TestSO);

            check!(
                single
                    .0
                    .site_energy(&overlapping_microstate.sites()[0].properties)
                    == f64::INFINITY
            );
            check!(
                single
                    .0
                    .site_energy(&overlapping_microstate.sites()[1].properties)
                    == 0.0
            );
        }
    }
    mod delta_energy {
        use super::*;

        struct Zero;

        impl SiteEnergy<Point<Cartesian<2>>> for Zero {
            fn site_energy(&self, _site_properties: &Point<Cartesian<2>>) -> f64 {
                0.0
            }

            fn is_only_infinite_or_zero() -> bool {
                true
            }
        }

        #[test]
        fn site_outside() {
            let cuboid = Rectangle::with_equal_edges(
                4.0.try_into()
                    .expect("hard-coded constant should be positive"),
            );
            let square = Closed(cuboid);

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

            let energy = External(Zero);

            check!(energy.delta_energy_one(&microstate, 0, &final_body) == f64::INFINITY);
            check!(energy.delta_energy_insert(&microstate, &final_body) == f64::INFINITY);
        }

        #[test]
        fn delta_energy() -> anyhow::Result<()> {
            let cuboid = Rectangle::with_equal_edges(
                4.0.try_into()
                    .expect("hard-coded constant should be positive"),
            );
            let square = Closed(cuboid);

            let body = Body {
                properties: Point::new(Cartesian::from([1.5, 1.5])),
                sites: [Point::new(Cartesian::from([0.0, 0.0]))].into(),
            };
            let mut final_body_inf = body.clone();
            final_body_inf.properties.position[1] = 0.5;

            let mut final_body_0 = body.clone();
            final_body_0.properties.position[1] = -1.5;

            let microstate = Microstate::builder()
                .boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            let energy = External(TestSO);

            check!(energy.delta_energy_one(&microstate, 0, &final_body_0) == 0.0);
            check!(energy.delta_energy_one(&microstate, 0, &final_body_inf) == f64::INFINITY);
            check!(energy.delta_energy_insert(&microstate, &final_body_0) == 0.0);
            check!(energy.delta_energy_insert(&microstate, &final_body_inf) == f64::INFINITY);
            check!(energy.delta_energy_remove(&microstate, 0) == 0.0);

            let mut microstate_inf = microstate.clone();
            microstate_inf.update_body_properties(0, final_body_inf.properties)?;

            let mut microstate_0 = microstate.clone();
            microstate_0.update_body_properties(0, final_body_0.properties)?;

            check!(energy.delta_energy_total(&microstate_0, &microstate_0) == 0.0);
            check!(energy.delta_energy_total(&microstate_0, &microstate_inf) == f64::INFINITY);
            check!(energy.delta_energy_total(&microstate_inf, &microstate_0) == 0.0);
            check!(energy.delta_energy_total(&microstate_inf, &microstate_inf) == f64::INFINITY);

            Ok(())
        }
    }
}

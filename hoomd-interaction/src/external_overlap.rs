// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ExternalOverlap`

use crate::{DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, SiteOverlap, TotalEnergy};
use hoomd_microstate::{Body, Microstate, Transform, boundary::Wrap, property::Position};

/// Hard overlaps between sites and external objects.
///
/// Use [`ExternalOverlap`] instead of [`External`] for hard interactions.
/// [`ExternalOverlap`] does not need to compute the initial energy and it can
/// short-circuit energy evaluations when the first overlap is detected. Both of
/// these lead to improved performance.
///
/// Given an inner type that implements [`SiteOverlap`], [`ExternalOverlap`] represents:
///
/// ```math
/// U_\mathrm{total} = \sum_{i=0}^{N-1} U\left( s_i \right)
/// ```
/// where $`s_i`$ is the full set of site properties for site i and
/// $`U\left( s_i \right)`$ is $`\infty`$ when the site overlaps with an external
/// object and 0 when it does not.
///
/// **hoomd-rs** currently does not provide any types that implement
/// [`SiteOverlap`]. Provide your own custom type.
///
/// [`External`]: crate::External
///
/// # Example
///
/// ```
/// use hoomd_interaction::{ExternalOverlap, SiteOverlap, TotalEnergy};
/// use hoomd_microstate::{Body, Microstate, property::Point};
/// use hoomd_vector::Cartesian;
///
/// struct Wall;
///
/// impl SiteOverlap<Point<Cartesian<2>>> for Wall {
///     fn site_overlap(&self, site_properties: &Point<Cartesian<2>>) -> bool {
///         site_properties.position[1].abs() < 1.0
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
///     let wall = ExternalOverlap(Wall);
///
///     let total_energy = wall.total_energy(&microstate);
///     assert_eq!(total_energy, 0.0);
///     Ok(())
/// }
/// ```
pub struct ExternalOverlap<E>(pub E);

impl<B, S, C, E> TotalEnergy<Microstate<B, S, C>> for ExternalOverlap<E>
where
    E: SiteOverlap<S>,
{
    /// Compute the total energy of the microstate contributed by functions of a single site.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::{ExternalOverlap, SiteOverlap, TotalEnergy};
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// struct Wall;
    ///
    /// impl SiteOverlap<Point<Cartesian<2>>> for Wall {
    ///     fn site_overlap(&self, site_properties: &Point<Cartesian<2>>) -> bool {
    ///         site_properties.position[1].abs() < 1.0
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
    ///     let wall = ExternalOverlap(Wall);
    ///
    ///     let total_energy = wall.total_energy(&microstate);
    ///     assert_eq!(total_energy, 0.0);
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, C>) -> f64 {
        for site in microstate.sites() {
            if self.0.site_overlap(&site.properties) {
                return f64::INFINITY;
            }
        }

        0.0
    }
}

/// Evaluate the change in energy contributed by `ExternalOverlap` when a single body is updated.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{DeltaEnergyOne, ExternalOverlap, SiteOverlap};
/// use hoomd_microstate::{Body, Microstate, property::Point};
/// use hoomd_vector::Cartesian;
///
/// struct Wall;
///
/// impl SiteOverlap<Point<Cartesian<2>>> for Wall {
///     fn site_overlap(&self, site_properties: &Point<Cartesian<2>>) -> bool {
///         site_properties.position[1].abs() < 1.0
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
///     let wall = ExternalOverlap(Wall);
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
impl<V, B, S, C, E> DeltaEnergyOne<B, S, C> for ExternalOverlap<E>
where
    E: SiteOverlap<S>,
    B: Transform<S>,
    S: Position<Position = V>,
    C: Wrap<B> + Wrap<S>,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        _body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        for s in &final_body.sites {
            match initial_microstate
                .boundary()
                .wrap(final_body.properties.transform(s))
            {
                Ok(wrapped_site) => {
                    if self.site_overlap(&wrapped_site) {
                        return f64::INFINITY;
                    }
                }
                Err(_) => return f64::INFINITY,
            }
        }

        0.0
    }
}

/// Evaluate the change in energy contributed by `ExternalOverlap` when a single body is inserted.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{DeltaEnergyInsert, ExternalOverlap, SiteOverlap};
/// use hoomd_microstate::{Body, Microstate, property::Point};
/// use hoomd_vector::Cartesian;
///
/// struct Wall;
///
/// impl SiteOverlap<Point<Cartesian<2>>> for Wall {
///     fn site_overlap(&self, site_properties: &Point<Cartesian<2>>) -> bool {
///         site_properties.position[1].abs() < 1.0
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
///     let wall = ExternalOverlap(Wall);
///
///     let delta_energy = wall
///         .delta_energy_insert(&microstate, &Body::point([0.0, -0.5].into()));
///     assert_eq!(delta_energy, f64::INFINITY);
///     Ok(())
/// }
/// ```
impl<V, B, S, C, E> DeltaEnergyInsert<B, S, C> for ExternalOverlap<E>
where
    E: SiteOverlap<S>,
    B: Transform<S>,
    S: Position<Position = V>,
    C: Wrap<B> + Wrap<S>,
{
    #[inline]
    fn delta_energy_insert(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        new_body: &Body<B, S>,
    ) -> f64 {
        for s in &new_body.sites {
            match initial_microstate
                .boundary()
                .wrap(new_body.properties.transform(s))
            {
                Ok(wrapped_site) => {
                    if self.site_overlap(&wrapped_site) {
                        return f64::INFINITY;
                    }
                }
                Err(_) => return f64::INFINITY,
            }
        }

        0.0
    }
}

/// Evaluate the change in energy contributed by `ExternalOverlap` when a single body is removed.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{DeltaEnergyRemove, ExternalOverlap, SiteOverlap};
/// use hoomd_microstate::{Body, Microstate, property::Point};
/// use hoomd_vector::Cartesian;
///
/// struct Wall;
///
/// impl SiteOverlap<Point<Cartesian<2>>> for Wall {
///     fn site_overlap(&self, site_properties: &Point<Cartesian<2>>) -> bool {
///         site_properties.position[1].abs() < 1.0
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
///     let wall = ExternalOverlap(Wall);
///
///     let delta_energy = wall.delta_energy_remove(&microstate, 0);
///     assert_eq!(delta_energy, 0.0);
///     Ok(())
/// }
/// ```
impl<B, S, C, E> DeltaEnergyRemove<B, S, C> for ExternalOverlap<E>
where
    E: SiteOverlap<S>,
{
    #[inline]
    fn delta_energy_remove(
        &self,
        _initial_microstate: &Microstate<B, S, C>,
        _body_index: usize,
    ) -> f64 {
        0.0
    }
}

impl<E, S> SiteOverlap<S> for ExternalOverlap<E>
where
    E: SiteOverlap<S>,
{
    #[inline]
    fn site_overlap(&self, site_properties: &S) -> bool {
        self.0.site_overlap(site_properties)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_geometry::shape::Rectangle;
    use hoomd_microstate::{
        Body, Microstate, MicrostateBuilder,
        boundary::{Closed, Open},
        property::{Point, Position},
    };
    use hoomd_vector::Cartesian;
    use rstest::*;

    struct TestSO;

    impl<S> SiteOverlap<S> for TestSO
    where
        S: Position<Position = Cartesian<2>>,
    {
        fn site_overlap(&self, site_properties: &S) -> bool {
            site_properties.position()[1].abs() < 1.0
        }
    }

    mod site_energy {
        use super::*;

        #[fixture]
        fn microstate() -> Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open> {
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
        fn overlapping_microstate() -> Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open> {
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
        fn single_total_0(microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>) {
            let single = ExternalOverlap(TestSO);

            assert_eq!(single.total_energy(&microstate), 0.0);
        }

        #[rstest]
        fn single_total_inf(
            overlapping_microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>,
        ) {
            let single = ExternalOverlap(TestSO);

            assert_eq!(single.total_energy(&overlapping_microstate), f64::INFINITY);
        }

        #[rstest]
        fn single_site_0(microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>) {
            let single = ExternalOverlap(TestSO);

            assert!(!single.site_overlap(&microstate.sites()[0].properties));
            assert!(!single.site_overlap(&microstate.sites()[1].properties));
        }

        #[rstest]
        fn single_site_inf(
            overlapping_microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>,
        ) {
            let single = ExternalOverlap(TestSO);

            assert!(single.site_overlap(&overlapping_microstate.sites()[0].properties));
            assert!(!single.site_overlap(&overlapping_microstate.sites()[1].properties));
        }
    }
    mod delta_energy {
        use super::*;

        struct Zero;

        impl SiteOverlap<Point<Cartesian<2>>> for Zero {
            fn site_overlap(&self, _site_properties: &Point<Cartesian<2>>) -> bool {
                false
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

            let microstate = MicrostateBuilder::with_boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            let energy = ExternalOverlap(Zero);

            assert_eq!(
                energy.delta_energy_one(&microstate, 0, &final_body),
                f64::INFINITY
            );
            assert_eq!(
                energy.delta_energy_insert(&microstate, &final_body),
                f64::INFINITY
            );
        }

        #[test]
        fn delta_energy() {
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

            let microstate = MicrostateBuilder::with_boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            let energy = ExternalOverlap(TestSO);

            assert_eq!(energy.delta_energy_one(&microstate, 0, &final_body_0), 0.0);
            assert_eq!(
                energy.delta_energy_one(&microstate, 0, &final_body_inf),
                f64::INFINITY
            );
            assert_eq!(energy.delta_energy_insert(&microstate, &final_body_0), 0.0);
            assert_eq!(
                energy.delta_energy_insert(&microstate, &final_body_inf),
                f64::INFINITY
            );
            assert_eq!(energy.delta_energy_remove(&microstate, 0), 0.0);
        }
    }
}

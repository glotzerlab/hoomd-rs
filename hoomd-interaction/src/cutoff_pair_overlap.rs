// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `CutoffPairOverlap`

use crate::{DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, SitePairOverlap, TotalEnergy};
use hoomd_microstate::{Body, Microstate, SiteKey, Transform, boundary::Wrap, property::Position};
use hoomd_spatial::PointsInBall;
use hoomd_vector::Vector;

/// Short-ranged hard overlaps between pairs of sites.
///
/// Use [`CutoffPairOverlap`] instead of [`CutoffPair`] for hard interactions.
/// [`CutoffPairOverlap`] does not need to compute the initial energy and it can
/// short-circuit energy evaluations when the first overlap is detected. Both of
/// these lead to improved performance.
///
/// Given an evaluator that implements [`SitePairOverlap`], [`CutoffPairOverlap`]
/// represents:
///
/// ```math
/// U_\mathrm{total} = \sum_{i=0}^{N-1}\sum_{j=i+1}^{N-1} U\left(s_i, s_j \right) \left[ \left|\vec{r}_j - \vec{r}_i\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
/// ```
/// where $`U(s_i, s_j)`$ is $`\infty`$ when [`CutoffPairOverlap::evaluator`] finds an
/// overlap and 0 when it does not,
/// $`s_i`$ is the full set of site properties for site i, $`\vec{r}_i`$ is
/// the position of site i, $`b_i`$ is the body tag that holds site *i*, and
/// $`\left[ \  \right]`$ denotes the Iverson bracket.
///
/// In other words, [`CutoffPairOverlap`] checks for overlaps between all site pairs that
/// are separated by a distance less than `r_cut` and belong to different bodies.
///
/// For the evaluator, use [`AlwaysTrue`], [`HardShape`] or your own custom type.
///
/// [`CutoffPair`]: crate::CutoffPair
/// [`AlwaysTrue`]: crate::pairwise::AlwaysTrue
/// [`HardShape`]: crate::pairwise::HardShape
///
/// # Example
///
/// Hard sphere:
/// ```
/// use hoomd_interaction::{CutoffPairOverlap, pairwise::AlwaysTrue};
/// use hoomd_microstate::property::Point;
/// use hoomd_vector::Cartesian;
///
/// let hard_sphere = CutoffPairOverlap {
///     r_cut: 1.0,
///     evaluator: AlwaysTrue,
/// };
/// ```
///
/// Hard shape:
///
/// ```
/// use hoomd_geometry::shape::Ellipse;
/// use hoomd_interaction::{CutoffPairOverlap, pairwise::HardShape};
/// use hoomd_microstate::property::Point;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let ellipse = Ellipse {
///     semi_axes: [4.0.try_into()?, 1.0.try_into()?],
/// };
/// let hard_ellipse = CutoffPairOverlap {
///     r_cut: 8.0,
///     evaluator: HardShape(ellipse),
/// };
/// # Ok(())
/// # }
/// ```
pub struct CutoffPairOverlap<E> {
    /// The distance beyond which all pairwise interactions evaluate to 0.
    pub r_cut: f64,

    /// Check the pairwise overlaps.
    pub evaluator: E,
}

impl<V, B, S, X, C, E> TotalEnergy<Microstate<B, S, X, C>> for CutoffPairOverlap<E>
where
    E: SitePairOverlap<S>,
    S: Position<Position = V>,
    X: PointsInBall<V, SiteKey>,
    V: Vector,
{
    /// Compute the total energy of the microstate contributed by functions on pairs of sites.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::{
    ///     CutoffPairOverlap, TotalEnergy, pairwise::AlwaysTrue,
    /// };
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([0.0, 0.0])),
    ///     Body::point(Cartesian::from([0.5, 0.0])),
    /// ])?;
    ///
    /// let hard_sphere = CutoffPairOverlap {
    ///     r_cut: 1.0,
    ///     evaluator: AlwaysTrue,
    /// };
    ///
    /// let total_energy = hard_sphere.total_energy(&microstate);
    /// assert_eq!(total_energy, f64::INFINITY);
    ///
    /// microstate.update_body_properties(0, Point::new([0.0, -2.0].into()));
    /// let total_energy = hard_sphere.total_energy(&microstate);
    /// assert_eq!(total_energy, 0.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, X, C>) -> f64 {
        for site_i in microstate.sites() {
            for site_j in
                microstate.iter_sites_potentially_near(site_i.properties.position(), self.r_cut)
            {
                if site_i.site_tag < site_j.site_tag
                    && site_i.body_tag != site_j.body_tag
                    && site_i
                        .properties
                        .position()
                        .distance_squared(site_j.properties.position())
                        < self.r_cut.powi(2)
                    && self
                        .evaluator
                        .site_pair_overlap(&site_i.properties, &site_j.properties)
                {
                    return f64::INFINITY;
                }
            }
        }

        0.0
    }
}

/// Evaluate the change in energy contributed by `CutoffPairOverlap` when one body is updated.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     CutoffPairOverlap, DeltaEnergyOne, pairwise::AlwaysTrue,
/// };
/// use hoomd_microstate::{Body, Microstate, property::Point};
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.extend_bodies([
///     Body::point(Cartesian::from([0.0, 0.0])),
///     Body::point(Cartesian::from([2.0, 0.0])),
/// ])?;
///
/// let hard_sphere = CutoffPairOverlap {
///     r_cut: 1.0,
///     evaluator: AlwaysTrue,
/// };
///
/// let delta_energy = hard_sphere.delta_energy_one(
///     &microstate,
///     1,
///     &Body::point([0.5, 0.0].into()),
/// );
/// assert_eq!(delta_energy, f64::INFINITY);
///
/// let delta_energy = hard_sphere.delta_energy_one(
///     &microstate,
///     1,
///     &Body::point([1.5, 0.0].into()),
/// );
/// assert_eq!(delta_energy, 0.0);
/// # Ok(())
/// # }
/// ```
impl<V, B, S, X, C, E> DeltaEnergyOne<B, S, X, C> for CutoffPairOverlap<E>
where
    E: SitePairOverlap<S>,
    B: Transform<S>,
    S: Position<Position = V>,
    X: PointsInBall<V, SiteKey>,
    C: Wrap<B> + Wrap<S>,
    V: Vector,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let body_tag = initial_microstate.bodies()[body_index].tag;

        let site_overlap = |site_properties: &S| {
            for site_j in initial_microstate
                .iter_sites_potentially_near(site_properties.position(), self.r_cut)
            {
                if body_tag != site_j.body_tag
                    && site_properties
                        .position()
                        .distance_squared(site_j.properties.position())
                        < self.r_cut.powi(2)
                    && self
                        .evaluator
                        .site_pair_overlap(site_properties, &site_j.properties)
                {
                    return true;
                }
            }

            false
        };

        for s in &final_body.sites {
            match initial_microstate
                .boundary()
                .wrap(final_body.properties.transform(s))
            {
                Err(_) => return f64::INFINITY,
                Ok(wrapped_site) => {
                    if site_overlap(&wrapped_site) {
                        return f64::INFINITY;
                    }
                }
            }
        }

        0.0
    }
}

/// Evaluate the change in energy contributed by `CutoffPairOverlap` when one body is inserted.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     CutoffPairOverlap, DeltaEnergyInsert, pairwise::AlwaysTrue,
/// };
/// use hoomd_microstate::{Body, Microstate, property::Point};
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.extend_bodies([Body::point(Cartesian::from([0.0, 0.0]))])?;
///
/// let hard_sphere = CutoffPairOverlap {
///     r_cut: 1.0,
///     evaluator: AlwaysTrue,
/// };
///
/// let delta_energy = hard_sphere
///     .delta_energy_insert(&microstate, &Body::point([0.5, 0.0].into()));
/// assert_eq!(delta_energy, f64::INFINITY);
///
/// let delta_energy = hard_sphere
///     .delta_energy_insert(&microstate, &Body::point([1.5, 0.0].into()));
/// assert_eq!(delta_energy, 0.0);
/// # Ok(())
/// # }
/// ```
impl<V, B, S, X, C, E> DeltaEnergyInsert<B, S, X, C> for CutoffPairOverlap<E>
where
    E: SitePairOverlap<S>,
    B: Transform<S>,
    S: Position<Position = V>,
    X: PointsInBall<V, SiteKey>,
    C: Wrap<B> + Wrap<S>,
    V: Vector,
{
    #[inline]
    fn delta_energy_insert(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        new_body: &Body<B, S>,
    ) -> f64 {
        // The new body is not yet in the microstate, so there is no need to
        // filter matching body tags. The new body does not yet have a tag.
        let site_overlap = |site_properties: &S| {
            for site_j in initial_microstate
                .iter_sites_potentially_near(site_properties.position(), self.r_cut)
            {
                if site_properties
                    .position()
                    .distance_squared(site_j.properties.position())
                    < self.r_cut.powi(2)
                    && self
                        .evaluator
                        .site_pair_overlap(site_properties, &site_j.properties)
                {
                    return true;
                }
            }

            false
        };

        for s in &new_body.sites {
            match initial_microstate
                .boundary()
                .wrap(new_body.properties.transform(s))
            {
                Err(_) => return f64::INFINITY,
                Ok(wrapped_site) => {
                    if site_overlap(&wrapped_site) {
                        return f64::INFINITY;
                    }
                }
            }
        }

        0.0
    }
}

/// Evaluate the change in energy contributed by `CutoffPair` when one body is removed.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     CutoffPairOverlap, DeltaEnergyRemove, pairwise::AlwaysTrue,
/// };
/// use hoomd_microstate::{Body, Microstate, property::Point};
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.extend_bodies([
///     Body::point(Cartesian::from([0.0, 0.0])),
///     Body::point(Cartesian::from([2.0, 0.0])),
/// ])?;
///
/// let hard_sphere = CutoffPairOverlap {
///     r_cut: 1.0,
///     evaluator: AlwaysTrue,
/// };
///
/// let delta_energy = hard_sphere.delta_energy_remove(&microstate, 1);
/// assert_eq!(delta_energy, 0.0);
/// # Ok(())
/// # }
/// ```
impl<V, B, S, X, C, E> DeltaEnergyRemove<B, S, X, C> for CutoffPairOverlap<E>
where
    E: SitePairOverlap<S>,
    S: Position<Position = V>,
    V: Vector,
{
    #[inline]
    fn delta_energy_remove(
        &self,
        _initial_microstate: &Microstate<B, S, X, C>,
        _body_index: usize,
    ) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TotalEnergy, pairwise::AlwaysTrue};
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

            // Ensure that CutoffPairOverlap respects the r_cut value set.
            let cutoff_pair = CutoffPairOverlap {
                r_cut: 5.0_f64.next_up(),
                evaluator: AlwaysTrue,
            };

            assert_eq!(cutoff_pair.total_energy(&microstate), f64::INFINITY);

            let cutoff_pair = CutoffPairOverlap {
                r_cut: 5.0,
                evaluator: AlwaysTrue,
            };

            assert_eq!(cutoff_pair.total_energy(&microstate), 0.0);
        }

        #[test]
        fn body_exclusion() {
            // Ensure that CutoffPairOverlap excludes pairs in the same body.
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

            let cutoff_pair = CutoffPairOverlap {
                r_cut: 1.0_f64.next_up(),
                evaluator: AlwaysTrue,
            };

            assert_eq!(cutoff_pair.total_energy(&microstate), 0.0);

            let cutoff_pair = CutoffPairOverlap {
                r_cut: 2.0_f64.next_up(),
                evaluator: AlwaysTrue,
            };

            assert_eq!(cutoff_pair.total_energy(&microstate), f64::INFINITY);
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

            let energy = CutoffPairOverlap {
                r_cut: 0.0,
                evaluator: AlwaysTrue,
            };

            assert_eq!(
                energy.delta_energy_one(&microstate, 0, &final_body),
                f64::INFINITY
            );
        }

        #[test]
        fn body_exclusion() {
            // Ensure that CutoffPairOverlap.delta_energy_one excludes pairs in the same body.
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

            let cutoff_pair = CutoffPairOverlap {
                r_cut: 1.0_f64.next_up(),
                evaluator: AlwaysTrue,
            };

            // moving body a to the right generates overlaps
            assert_eq!(
                cutoff_pair.delta_energy_one(&microstate, 0, &body_a_overlap),
                f64::INFINITY
            );

            // moving body away results in no overlaps
            assert_eq!(
                cutoff_pair.delta_energy_one(&microstate, 0, &body_a_no_overlap),
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

            let energy = CutoffPairOverlap {
                r_cut: 0.0,
                evaluator: AlwaysTrue,
            };

            assert_eq!(
                energy.delta_energy_insert(&microstate, &new_body),
                f64::INFINITY
            );
        }

        #[test]
        fn body_exclusion() {
            // Ensure that CutoffPairOverlap.delta_energy_insert excludes pairs in the same body.
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

            let cutoff_pair = CutoffPairOverlap {
                r_cut: 1.0_f64.next_up(),
                evaluator: AlwaysTrue,
            };

            assert_eq!(
                cutoff_pair.delta_energy_insert(&microstate, &body_a_new),
                f64::INFINITY
            );

            microstate
                .add_body(body_a_new)
                .expect("hard-coded bodies should be in the boundary");
            assert_eq!(cutoff_pair.delta_energy_remove(&microstate, 1), 0.0);
        }
    }
}

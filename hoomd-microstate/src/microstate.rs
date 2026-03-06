// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Microstate`] and related types.

use arrayvec::ArrayVec;
use serde::{Deserialize, Serialize};
use std::{cmp::Reverse, collections::BinaryHeap, fmt};

use crate::{
    Body, Error, Site, Transform,
    boundary::{GenerateGhosts, MAX_GHOSTS, Open, Wrap},
    property::Position,
};

use hoomd_geometry::MapPoint;
use hoomd_rand::Counter;
use hoomd_spatial::{AllPairs, PointUpdate, PointsNearBall};

/// Either a primary site index or a ghost site index.
#[derive(Clone, Copy, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[expect(
    clippy::exhaustive_enums,
    reason = "There will only ever be primary and ghost sites."
)]
pub enum SiteKey {
    /// Index to a primary site.
    Primary(usize),

    /// Index to a ghost site.
    Ghost(usize),
}

/// Track a unique identifier for an item in [`Microstate`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tagged<T> {
    /// The unique identifier.
    pub tag: usize,
    /// The tagged item.
    pub item: T,
}

/// A dense vector with O(1) remove complexity.
///
/// Each item pushed to the vector is given a tag (in monotonically increasing
/// order). Access items by tag when identity matters and by index order when it
/// doesn't.
///
/// Items are removed using `swap_remove`. Removed tags are reused when adding new
/// items.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct VecWithTags<T> {
    /// Items in index order.
    items: Vec<T>,

    /// Tags of the items, in index order.
    tags: Vec<usize>,

    /// Indices of the items, in tag order.
    indices: Vec<Option<usize>>,

    /// Tags that can be reused.
    free_tags: BinaryHeap<Reverse<usize>>,
}

impl<T> VecWithTags<T> {
    /// Construct an empty vector of tagged items.
    fn new() -> Self {
        Self {
            items: Vec::new(),
            tags: Vec::new(),
            indices: Vec::new(),
            free_tags: BinaryHeap::new(),
        }
    }

    /// Remove all items from the vector.
    fn clear(&mut self) {
        self.items.clear();
        self.tags.clear();
        self.indices.clear();
        self.free_tags.clear();
    }

    /// The tag that will be assigned to the next item added.
    fn next_tag(&self) -> usize {
        self.free_tags.peek().map_or(self.indices.len(), |t| t.0)
    }

    /// Add a new item and return the tag added.
    fn push(&mut self, item: T) -> usize {
        let tag = self.free_tags.pop().map_or(self.indices.len(), |t| t.0);
        let index = self.items.len();

        self.items.push(item);
        self.tags.push(tag);

        if tag == self.indices.len() {
            self.indices.push(Some(index));
        } else {
            debug_assert_eq!(self.indices[tag], None);
            self.indices[tag] = Some(index);
        }

        tag
    }

    /// Remove an item identified *by index*
    fn remove(&mut self, index: usize) {
        let removed_tag = self.tags[index];

        self.items.swap_remove(index);
        self.tags.swap_remove(index);

        if index < self.items.len() {
            let replaced_tag = self.tags[index];
            self.indices[replaced_tag] = Some(index);
        }
        self.indices[removed_tag] = None;
        self.free_tags.push(Reverse(removed_tag));
    }

    /// Number of items stored.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.items.len()
    }

    /// True when any items are stored.
    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Iterate over items in tag order.
    fn iter_tag_order(&self) -> impl Iterator<Item = &T> {
        self.indices
            .iter()
            .filter_map(|opt_i| opt_i.map(|i| &self.items[i]))
    }
}

/// Store and manage all the degrees of freedom of a single microstate in phase space.
///
/// [`Microstate`] implements the main logic of the crate. See the [crate-level
/// documentation](crate) for a full overview and the method-specific documentation
/// for additional details.
///
/// The generic type names are:
/// * `B`: The [`Body::properties`](crate::Body) type.
/// * `S`: The [`Site::properties`](crate::Site) type.
/// * `X`: The [`spatial data structure`](hoomd_spatial) type.
/// * `C`: The [`boundary`](crate::boundary) condition type.
///
/// ## Constructing Microstate
///
/// You will find many examples in this documentation using [`Microstate::new`]. It
/// is designed to be terse, and is inflexible as a consequence. [`Microstate::new`]
/// always sets [`Open`](crate::boundary::Open) boundary conditions and initializes
/// the seed and step to 0.
/// ```
/// use hoomd_microstate::Microstate;
/// # use hoomd_microstate::{Body, property::Point};
/// # use hoomd_vector::Cartesian;
///
/// let mut microstate = Microstate::new();
/// # microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
/// ```
///
/// When you need more control, use [`MicrostateBuilder`] to set the boundary conditions,
/// use a different seed or starting step:
///
/// ```
/// use hoomd_geometry::shape::Rectangle;
/// use hoomd_microstate::{Body, Microstate, boundary::Closed};
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let square = Closed(Rectangle::with_equal_edges(10.0.try_into()?));
///
/// let microstate = Microstate::builder()
///     .boundary(square)
///     .seed(0x43abf1)
///     .step(100_000)
///     .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
///     .try_build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Microstate<B, S = B, X = AllPairs<SiteKey>, C = Open> {
    /// Total number of steps that this microstate has been advanced in a simulation model.
    step: u64,

    /// Number of substeps that the simulation has taken during the current simulation step.
    substep: u32,

    /// User chosen random number seed.
    seed: u32,

    /// Bodies in the microstate, stored in index order.
    bodies: VecWithTags<Tagged<Body<B, S>>>,

    /// Sites in the system reference frame.
    sites: VecWithTags<Site<S>>,

    /// Tags of the sites associated with the bodies (in body index order).
    bodies_sites: Vec<Vec<usize>>,

    /// Ghost sites in the system reference frame.
    ghosts: VecWithTags<Site<S>>,

    /// Tags of the ghosts associated with a given site (in site index order).
    sites_ghosts: Vec<ArrayVec<usize, MAX_GHOSTS>>,

    /// The range of allowed particle positions and a description of any periodicity.
    boundary: C,

    /// Spatial data structure.
    spatial_data: X,
}

impl<B, S> Default for Microstate<B, S, AllPairs<SiteKey>, Open> {
    /// Construct an empty microstate with open boundary conditions.
    ///
    /// See [`Microstate::new`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S> Microstate<B, S, AllPairs<SiteKey>, Open> {
    /// Construct an empty microstate with open boundary conditions.
    ///
    /// The microstate starts at step 0, substep 0, random number seed 0,
    /// and has no bodies. Use the [`AllPairs`] spatial search algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::Microstate;
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// let mut microstate = Microstate::new();
    /// assert_eq!(microstate.step(), 0);
    /// assert_eq!(microstate.substep(), 0);
    /// assert_eq!(microstate.seed(), 0);
    /// assert_eq!(microstate.bodies().len(), 0);
    /// assert_eq!(microstate.sites().len(), 0);
    /// # microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
    /// ```
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Microstate {
            step: 0,
            substep: 0,
            seed: 0,
            bodies: VecWithTags::new(),
            sites: VecWithTags::new(),
            bodies_sites: Vec::new(),
            ghosts: VecWithTags::new(),
            sites_ghosts: Vec::new(),
            boundary: Open,
            spatial_data: AllPairs::default(),
        }
    }

    /// Set microstate parameters before construction.
    ///
    /// The builder defaults to:
    /// * `step = 0`
    /// * `seed = 0`
    /// * `spatial_data` = [`AllPairs`]
    /// * `boundary` = [`Open`]
    /// * No bodies.
    ///
    /// Call [`MicrostateBuilder`] methods in a chain to set these parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::{
    ///     Body, Microstate, boundary::Closed, property::Point,
    /// };
    /// use hoomd_spatial::VecCell;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let cell_list = VecCell::builder()
    ///     .nominal_search_radius(2.5.try_into()?)
    ///     .build();
    /// let square = Closed(Rectangle::with_equal_edges(10.0.try_into()?));
    ///
    /// let microstate = Microstate::builder()
    ///     .boundary(square)
    ///     .spatial_data(cell_list)
    ///     .step(100_000)
    ///     .seed(0x1234abcd)
    ///     .bodies([
    ///         Body::point(Cartesian::from([1.0, 0.0])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])
    ///     .try_build()?;
    ///
    /// assert_eq!(microstate.boundary().0.edge_lengths[0].get(), 10.0);
    /// assert_eq!(microstate.step(), 100_000);
    /// assert_eq!(microstate.seed(), 0x1234abcd);
    /// assert_eq!(microstate.bodies().len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn builder() -> MicrostateBuilder<B, S, AllPairs<SiteKey>, Open> {
        MicrostateBuilder {
            step: 0,
            seed: 0,
            bodies: Vec::new(),
            spatial_data: AllPairs::default(),
            boundary: Open,
        }
    }
}

/// Access and manage the simulation step, substep, RNG seeds.
impl<B, S, X, C> Microstate<B, S, X, C> {
    /// Get the simulation step.
    ///
    /// # Examples
    ///
    /// Get the step:
    /// ```
    /// use hoomd_microstate::Microstate;
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// let mut microstate = Microstate::new();
    /// # microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
    /// assert_eq!(microstate.step(), 0);
    /// ```
    ///
    /// Initialize a microstate with a given step:
    /// ```
    /// use hoomd_microstate::Microstate;
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = Microstate::builder()
    ///     .step(100_000)
    /// # .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
    ///     .try_build()?;
    /// assert_eq!(microstate.step(), 100_000);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Increment the simulation step.
    ///
    /// Also set the substep to 0.
    ///
    /// # Examples
    ///
    /// Increment the simulation step:
    /// ```
    /// use hoomd_microstate::Microstate;
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// let mut microstate = Microstate::new();
    /// # microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
    /// microstate.increment_step();
    ///
    /// assert_eq!(microstate.step(), 1);
    /// ```
    ///
    /// Confirm that `substep` resets to 0:
    /// ```
    /// use hoomd_microstate::Microstate;
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// let mut microstate = Microstate::new();
    /// # microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
    ///
    /// microstate.increment_substep();
    /// microstate.increment_substep();
    /// microstate.increment_substep();
    /// assert_eq!(microstate.substep(), 3);
    ///
    /// microstate.increment_step();
    ///
    /// assert_eq!(microstate.step(), 1);
    /// assert_eq!(microstate.substep(), 0);
    /// ```
    #[inline]
    pub fn increment_step(&mut self) {
        self.step += 1;
        self.substep = 0;
    }

    /// Get the simulation substep.
    ///
    /// # Example
    /// ```
    /// use hoomd_microstate::Microstate;
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// let mut microstate = Microstate::new();
    /// # microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
    /// microstate.increment_substep();
    ///
    /// assert_eq!(microstate.substep(), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn substep(&self) -> u32 {
        self.substep
    }

    /// Increment the simulation substep.
    ///
    /// # Example
    /// ```
    /// use hoomd_microstate::Microstate;
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// let mut microstate = Microstate::new();
    /// # microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
    /// microstate.increment_substep();
    ///
    /// assert_eq!(microstate.substep(), 1);
    /// ```
    #[inline]
    pub fn increment_substep(&mut self) {
        self.substep += 1;
    }

    /// Get the simulation seed.
    ///
    /// # Examples:
    ///
    /// Get the simulation seed.
    /// ```
    /// use hoomd_microstate::Microstate;
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// let mut microstate = Microstate::new();
    /// # microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
    ///
    /// assert_eq!(microstate.seed(), 0);
    /// ```
    ///
    /// Initialize a microstate with a given seed:
    /// ```
    /// use hoomd_microstate::Microstate;
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// # type BodyProperties = Point<Cartesian<2>>;
    /// # type SiteProperties = Point<Cartesian<2>>;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = Microstate::<BodyProperties, SiteProperties>::builder()
    ///     .seed(0x1234abcd)
    /// # .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
    ///     .try_build()?;
    /// assert_eq!(microstate.seed(), 0x1234abcd);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// Create a partially constructed [`Counter`] from the current step, substep, and seed.
    ///
    /// Use the produced [`Counter`] to make a independent random number generator at each
    /// substep. Call additional methods on the [`Counter`] first to further differentiate
    /// the stream.
    ///
    /// # Example
    ///
    /// Make a random number generator unique to this substep:
    /// ```
    /// use hoomd_microstate::Microstate;
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// let mut microstate = Microstate::new();
    /// # microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
    ///
    /// let rng = microstate.counter().make_rng();
    /// ```
    ///
    /// Make a random number generator unique to a particular particle on this substep:
    ///
    /// ```
    /// use hoomd_microstate::Microstate;
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// let mut microstate = Microstate::new();
    /// # microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
    ///
    /// let tag = 10;
    /// let rng = microstate.counter().index(tag).make_rng();
    /// ```
    #[inline]
    pub fn counter(&self) -> Counter {
        Counter::new(self.step, self.substep, self.seed)
    }
}

/// Access and manage the boundary condition.
impl<B, S, X, C> Microstate<B, S, X, C> {
    /// Get the boundary condition.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::{Microstate, boundary::Closed};
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// let square = Closed(Rectangle::with_equal_edges(10.0.try_into()?));
    /// let microstate = Microstate::builder()
    ///     .boundary(square)
    /// # .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
    ///     .try_build()?;
    ///
    /// assert_eq!(microstate.boundary().0.edge_lengths[0].get(), 10.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn boundary(&self) -> &C {
        &self.boundary
    }
}

/// Manage bodies in the microstate.
impl<P, B, S, X, C> Microstate<B, S, X, C>
where
    P: Copy,
    B: Transform<S> + Position<Position = P>,
    S: Position<Position = P> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    X: PointUpdate<P, SiteKey>,
{
    /// Update the ghosts of a site.
    ///
    /// Given a site in the boundary, update that site's ghosts to be consistent
    /// with that site's properties. This may require adding or removing ghosts.
    fn update_site_ghosts(
        sites: &VecWithTags<Site<S>>,
        site_index: usize,
        boundary: &C,
        sites_ghosts: &mut [ArrayVec<usize, MAX_GHOSTS>],
        ghosts: &mut VecWithTags<Site<S>>,
        spatial_data: &mut X,
    ) {
        let site = &sites.items[site_index];
        let new_ghosts = boundary.generate_ghosts(&site.properties);
        let ghost_tags = &mut sites_ghosts[site_index];

        match ghost_tags.len().cmp(&new_ghosts.len()) {
            std::cmp::Ordering::Less => {
                let ghosts_to_add = new_ghosts.len() - ghost_tags.len();
                for _ in 0..ghosts_to_add {
                    let ghost_tag = ghosts.push(Site {
                        site_tag: site.site_tag,
                        body_tag: site.body_tag,
                        properties: S::default(),
                    });
                    ghost_tags.push(ghost_tag);
                }
            }
            std::cmp::Ordering::Greater => {
                let ghosts_to_remove = ghost_tags.len() - new_ghosts.len();
                for ghost_tag in ghost_tags.iter().rev().take(ghosts_to_remove) {
                    let ghost_index = ghosts.indices[*ghost_tag]
                        .expect("sites_ghosts and ghost.indices should be consistent");
                    ghosts.remove(ghost_index);
                    spatial_data.remove(&SiteKey::Ghost(*ghost_tag));
                }

                ghost_tags.truncate(new_ghosts.len());
            }
            std::cmp::Ordering::Equal => {}
        }

        debug_assert_eq!(ghost_tags.len(), new_ghosts.len());

        for (new_ghost, ghost_tag) in new_ghosts.into_iter().zip(ghost_tags) {
            let ghost_index = ghosts.indices[*ghost_tag]
                .expect("sites_ghosts and ghost.indices should be consistent");
            spatial_data.insert(SiteKey::Ghost(*ghost_tag), *new_ghost.position());
            ghosts.items[ghost_index].properties = new_ghost;
        }
    }

    /// Update ghosts for all the sites of a given body (by index).
    fn update_body_site_ghosts(&mut self, body_index: usize) {
        for site_tag in &self.bodies_sites[body_index] {
            let site_index = self.sites.indices[*site_tag]
                .expect("bodies_sites and site_indices should be consistent");
            Self::update_site_ghosts(
                &self.sites,
                site_index,
                &self.boundary,
                &mut self.sites_ghosts,
                &mut self.ghosts,
                &mut self.spatial_data,
            );
        }
    }

    /// Add a new body to the microstate.
    ///
    /// Each body is assigned a unique tag. The first body is given tag 0,
    /// the second is given tag 1, and so on. When a body is removed (see
    /// [`Microstate::remove_body()`]), its tag becomes unused. The next call to
    /// `add_body` will assign the smallest unused tag.
    ///
    /// `add_body` also adds the body's sites to the microstate's
    /// [`sites`](Microstate::sites) (in system coordinates) and assigns unique
    /// tags to the sites similarly. It wraps the body's position (and the
    /// positions of its sites in system coordinates) into the boundary (see
    /// [`boundary`]).
    ///
    /// [`boundary`]: crate::boundary
    ///
    /// # Cost
    ///
    /// The cost of adding a body is proportional to the number of sites in the
    /// body.
    ///
    /// # Returns
    ///
    /// [`Ok(tag)`](Result::Ok) with the tag of the added body on success.
    ///
    /// # Errors
    ///
    /// [`Error::AddBody`] when the body cannot be added to the microstate because
    /// the body position or any site position cannot be wrapped into the boundary
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// let first_tag =
    ///     microstate.add_body(Body::point(Cartesian::from([1.0, 0.0])))?;
    /// let second_tag =
    ///     microstate.add_body(Body::point(Cartesian::from([-1.0, 2.0])))?;
    ///
    /// assert_eq!(microstate.bodies().len(), 2);
    /// assert_eq!(first_tag, 0);
    /// assert_eq!(second_tag, 1);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    pub fn add_body(&mut self, body: Body<B, S>) -> Result<usize, Error> {
        // Find the tag of the new body.
        let body_tag = self.bodies.next_tag();

        let mut body = body;
        body.properties = self
            .boundary
            .wrap(body.properties)
            .map_err(|e| Error::AddBody(body_tag, e))?;

        // An unknown site in the body might not wrap into the boundary.
        // Check that they do first before starting to modify internal data
        // structures. This wraps every site twice on add. Should that prove to
        // be a performance bottleneck, we could alternately implement rollback
        // (complicated) or a staging Vec (would require additional allocations
        // or a reusable scratch storage).
        for s in &body.sites {
            self.boundary
                .wrap(body.properties.transform(s))
                .map_err(|e| Error::AddBody(body_tag, e))?;
        }

        // Now that all errors have been checked, it is safe to start mutating the
        // microstate.

        // Add the body's sites first.
        // Should the Vec allocation prove a bottleneck, we could recycle the body_sites
        // vecs along with the tags.
        let mut body_sites = Vec::with_capacity(body.sites.len());
        for s in &body.sites {
            let site_tag = self.sites.next_tag();

            let site = Site {
                site_tag,
                properties: self
                    .boundary
                    .wrap(body.properties.transform(s))
                    .expect("sites should be validated as wrappable prior to this loop"),
                body_tag,
            };
            self.spatial_data
                .insert(SiteKey::Primary(site.site_tag), *site.properties.position());
            self.sites.push(site);
            self.sites_ghosts.push(ArrayVec::new());

            body_sites.push(site_tag);
        }

        // Add body
        self.bodies.push(Tagged {
            tag: body_tag,
            item: body,
        });
        self.bodies_sites.push(body_sites);

        self.update_body_site_ghosts(self.bodies().len() - 1);

        Ok(body_tag)
    }

    /// Add multiple bodies to the microstate.
    ///
    /// See [`Microstate::add_body()`] for details.
    ///
    /// # Errors
    ///
    /// [`Error::AddBody`] when any of the bodies cannot be added to the microstate.
    /// `extend_bodies` adds each body one by one. When an error occurs, it
    /// short-circuits and does not attempt to add any further bodies. The bodies
    /// added before the error will remain in the microstate.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::point(Cartesian::from([1.0, 0.0])),
    ///     Body::point(Cartesian::from([-1.0, 2.0])),
    /// ])?;
    ///
    /// assert_eq!(microstate.bodies().len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn extend_bodies<T>(&mut self, bodies: T) -> Result<(), Error>
    where
        T: IntoIterator<Item = Body<B, S>>,
    {
        for body in bodies {
            self.add_body(body)?;
        }

        Ok(())
    }

    /// Remove a body at the given *index* from the microstate.
    ///
    /// Also remove all the body's sites. The body's tag (and the tags of its
    /// sites) are then free to be reused by [`Microstate::add_body`].
    ///
    /// Removing a body will change the index order of the
    /// [`bodies`](Microstate::bodies) and [`sites`](Microstate::sites) arrays.
    /// [`Microstate`] does not guarantee any specific ordering in these arrays
    /// after calling `remove_body`.
    ///
    /// # Cost
    ///
    /// The cost of removing a body is proportional to the number of sites in the
    /// body.
    ///
    /// # Panics
    ///
    /// Panics when `index` is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::builder()
    ///     .bodies([
    ///         Body::point(Cartesian::from([1.0, 0.0])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])
    ///     .try_build()?;
    ///
    /// microstate.remove_body(0);
    ///
    /// assert_eq!(microstate.bodies().len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn remove_body(&mut self, body_index: usize) {
        let body_tag = self.bodies.items[body_index].tag;
        debug_assert_eq!(self.bodies.indices[body_tag], Some(body_index));

        // Remove sites and their associated ghosts. `add_body` adds sites in
        // increasing index order, so remove them in reverse order to avoid keep
        // the other bodies' sites in increasing order.
        let body_sites = self.bodies_sites.swap_remove(body_index);
        for site_tag in body_sites.iter().rev() {
            let site_index = self.sites.indices[*site_tag]
                .expect("bodies_sites and sites.indices should be consistent");

            let site_ghosts = self.sites_ghosts.swap_remove(site_index);
            for ghost_tag in site_ghosts.iter().rev() {
                let ghost_index = self.ghosts.indices[*ghost_tag]
                    .expect("sites_ghosts and ghosts.indices should be consistent");
                self.spatial_data.remove(&SiteKey::Ghost(*ghost_tag));
                self.ghosts.remove(ghost_index);
            }

            self.spatial_data.remove(&SiteKey::Primary(*site_tag));
            self.sites.remove(site_index);
        }

        // Remove body
        self.bodies.remove(body_index);
    }

    /// Sets the properties of the given body.
    ///
    /// `update_body_properties` also updates the properties of the sites (in the
    /// system frame) associated with the body accordingly.
    ///
    /// # Errors
    ///
    /// [`Error::UpdateBody`] the body properties cannot be updated because the body
    /// position or any site position cannot be wrapped into the boundary. When an
    /// error occurs, `update_body_properties` makes no change to the [`Microstate`].
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::builder()
    ///     .bodies([Body::point(Cartesian::from([1.0, 0.0]))])
    ///     .try_build()?;
    ///
    /// microstate
    ///     .update_body_properties(0, Point::new(Cartesian::from([-2.0, 3.0])))?;
    /// assert_eq!(
    ///     microstate.bodies()[0].item.properties.position,
    ///     [-2.0, 3.0].into()
    /// );
    /// assert_eq!(
    ///     microstate.sites()[0].properties.position,
    ///     [-2.0, 3.0].into()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    pub fn update_body_properties(&mut self, body_index: usize, properties: B) -> Result<(), Error>
    where
        B: Transform<S> + Position<Position = P>,
        S: Position<Position = P>,
        C: Wrap<B> + Wrap<S>,
    {
        let body = &mut self.bodies.items[body_index];

        let new_body_properties = self
            .boundary
            .wrap(properties)
            .map_err(|e| Error::UpdateBody(body.tag, e))?;

        // An unknown site in the body might not wrap into the boundary.
        // Check that they do first before starting to modify internal data
        // structures. This wraps every site twice on update. Should that prove
        // to be a performance bottleneck, we could alternately implement a
        // staging Vec (would require allocation/deallocation per update or a
        // reusable scratch storage).
        for s in &body.item.sites {
            self.boundary
                .wrap(new_body_properties.transform(s))
                .map_err(|e| Error::UpdateBody(body.tag, e))?;
        }

        body.item.properties = new_body_properties;

        // Update site properties
        for (i, site_tag) in self.bodies_sites[body_index].iter().enumerate() {
            let site_index = self.sites.indices[*site_tag]
                .expect("bodies_sites and site_indices should be consistent");
            let site_properties = self
                .boundary
                .wrap(body.item.properties.transform(&body.item.sites[i]))
                .expect("sites should be validated as wrappable prior to this loop");
            self.spatial_data
                .insert(SiteKey::Primary(*site_tag), *site_properties.position());
            self.sites.items[site_index].properties = site_properties;
        }

        self.update_body_site_ghosts(body_index);

        Ok(())
    }

    /// Remove all bodies from the microstate.
    ///
    /// The step, substep, seed, and boundary are left unchanged.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Body, Microstate, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::builder()
    ///     .bodies([Body::point(Cartesian::from([1.0, 0.0]))])
    ///     .try_build()?;
    ///
    /// microstate.clear();
    /// assert_eq!(microstate.bodies().len(), 0);
    /// assert_eq!(microstate.sites().len(), 0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.bodies.clear();
        self.sites.clear();
        self.bodies_sites.clear();
        self.ghosts.clear();
        self.sites_ghosts.clear();
        self.spatial_data.clear();
    }
}

/// Access contents of the microstate.
impl<B, S, X, C> Microstate<B, S, X, C> {
    /// Access the microstate's tagged bodies in index order.
    ///
    /// [`Microstate`] stores bodies in a flat memory region. The [`Tagged`] type
    /// holds the unique identifier for each body in [`Tagged::tag`] and the
    /// [`Body`] itself in [`Tagged::item`].
    ///
    /// [`bodies`](Microstate::bodies) provides direct immutable access
    /// to this slice. To mutate a body (and by extension, its sites), see
    /// [`Microstate::update_body_properties()`].
    ///
    /// # Examples
    ///
    /// Identify the tag of a body at a given index:
    ///
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = Microstate::builder()
    ///     .bodies([
    ///         Body::point(Cartesian::from([1.0, 0.0])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])
    ///     .try_build()?;
    ///
    /// assert_eq!(microstate.bodies()[0].tag, 0);
    /// assert_eq!(microstate.bodies()[1].tag, 1);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Compute system-wide properties that are order-independent:
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_vector::{Cartesian, Vector};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = Microstate::builder()
    ///     .bodies([
    ///         Body::point(Cartesian::from([1.0, 0.0])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])
    ///     .try_build()?;
    ///
    /// let average_position = microstate
    ///     .bodies()
    ///     .iter()
    ///     .map(|tagged_body| tagged_body.item.properties.position)
    ///     .sum::<Cartesian<2>>()
    ///     / (microstate.bodies().len() as f64);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn bodies(&self) -> &[Tagged<Body<B, S>>] {
        &self.bodies.items
    }

    /// Identify the index of a body given a tag.
    ///
    /// Use [`body_indices`](Microstate::body_indices) to locate a specific body in
    /// [`Microstate::bodies`].
    ///
    /// `body_indices()[tag]` is:
    /// * [`None`] when there is no body with the given tag in the microstate.
    /// * [`Some(index)`](Option::Some) when the body with the given tag is in the
    ///   microstate. `index` is the index of the body in [`Microstate::bodies`].
    ///
    /// # Example
    ///
    /// ```
    /// use anyhow::anyhow;
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::builder()
    ///     .bodies([
    ///         Body::point(Cartesian::from([1.0, 2.0])),
    ///         Body::point(Cartesian::from([3.0, 4.0])),
    ///         Body::point(Cartesian::from([5.0, 6.0])),
    ///         Body::point(Cartesian::from([7.0, 8.0])),
    ///     ])
    ///     .try_build()?;
    ///
    /// let index =
    ///     microstate.body_indices()[0].ok_or(anyhow!("body 0 is not present"))?;
    /// microstate.remove_body(index);
    ///
    /// assert_eq!(microstate.body_indices()[0], None);
    /// assert!(matches!(microstate.body_indices()[3], Some(_)));
    ///
    /// let index =
    ///     microstate.body_indices()[2].ok_or(anyhow!("body 2 is not present"))?;
    /// assert_eq!(
    ///     microstate.bodies()[index].item.properties.position,
    ///     [5.0, 6.0].into()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn body_indices(&self) -> &[Option<usize>] {
        &self.bodies.indices
    }

    /// Access the microstate's sites (in the system frame) in index order.
    ///
    /// [`Microstate`] stores sites twice. Each body in
    /// [`bodies`](Microstate::bodies) stores its sites in the body frame of
    /// reference. [`Microstate`] also stores a flat vector of sites that have been
    /// transformed (see [`Transform`]) to the system reference frame. The [`Site`]
    /// type holds the unique identifier for each site in [`Site::site_tag`],
    /// the associated body tag in [`Site::body_tag`] and the site's properties in
    /// [`Site::properties`].
    ///
    /// [`sites`](Microstate::sites) provides direct immutable access to the
    /// slice of all sites. To mutate a body (and by extension, its sites), see
    /// [`Microstate::update_body_properties()`].
    ///
    /// # Examples
    ///
    /// Identify the site and body tags of a site at a given index:
    ///
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = Microstate::builder()
    ///     .bodies([
    ///         Body::point(Cartesian::from([1.0, 0.0])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])
    ///     .try_build()?;
    ///
    /// assert_eq!(microstate.sites()[0].site_tag, 0);
    /// assert_eq!(microstate.sites()[0].body_tag, 0);
    ///
    /// assert_eq!(microstate.sites()[1].body_tag, 1);
    /// assert_eq!(microstate.sites()[1].body_tag, 1);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Compute system-wide properties that are order-independent:
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_vector::{Cartesian, Vector};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = Microstate::builder()
    ///     .bodies([
    ///         Body::point(Cartesian::from([1.0, 0.0])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])
    ///     .try_build()?;
    ///
    /// let average_position = microstate
    ///     .sites()
    ///     .iter()
    ///     .map(|site| site.properties.position)
    ///     .sum::<Cartesian<2>>()
    ///     / (microstate.sites().len() as f64);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn sites(&self) -> &[Site<S>] {
        &self.sites.items
    }

    /// Access the ghost sites in the system frame.
    ///
    /// Each ghost site shares a `site_tag` and `body_tag` with a primary site
    /// (in [`sites`]). Ghost sites are only placed when using periodic boundary
    /// conditions and are outside the edges of the boundary.
    ///
    /// [`sites`]: Self::sites
    #[inline]
    pub fn ghosts(&self) -> &[Site<S>] {
        &self.ghosts.items
    }

    /// Identify the index of a site given a tag.
    ///
    /// Use [`site_indices`](Microstate::site_indices) to locate a specific site in
    /// [`Microstate::sites`].
    ///
    /// See [`body_indices`](Microstate::body_indices) for details.
    #[inline]
    pub fn site_indices(&self) -> &[Option<usize>] {
        &self.sites.indices
    }

    /// Iterate over all the sites (in the system reference frame) associated with a body.
    ///
    /// Use [`iter_body_sites`](Microstate::iter_body_sites) to perform computations
    /// in the system reference frame on all sites that are associated with a given
    /// body *index*. The borrowed sites are immutable. Call
    /// [`Microstate::update_body_properties()`] to mutate a body.
    ///
    /// `iter_body_sites` always iterates over *primary sites*. In periodic boundary
    /// conditions, these sites may be split across one or more parts of the
    /// boundary.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_vector::{Cartesian, Vector};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = Microstate::builder()
    ///     .bodies([
    ///         Body::point(Cartesian::from([1.0, 0.0])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])
    ///     .try_build()?;
    ///
    /// let average_position = microstate
    ///     .iter_body_sites(0)
    ///     .map(|site| site.properties.position)
    ///     .sum::<Cartesian<2>>()
    ///     / (microstate.bodies()[0].item.sites.len() as f64);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    pub fn iter_body_sites(&self, body_index: usize) -> impl Iterator<Item = &Site<S>> {
        self.bodies_sites[body_index].iter().map(|site_tag| {
            &self.sites.items[self.sites.indices[*site_tag]
                .expect("bodies_sites and site_indices should be consistent")]
        })
    }

    /// Iterate over all sites in monotonically increasing tag order.
    ///
    /// `iter_sites_tag_order` is especially useful when implementing
    /// [`AppendMicrostate`], as GSD files must be written in tag order.
    ///
    /// [`AppendMicrostate`]: crate::AppendMicrostate
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::builder()
    ///     .bodies([
    ///         Body::point(Cartesian::from([1.0, 0.0])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])
    ///     .try_build()?;
    ///
    /// microstate.remove_body(0);
    /// microstate.add_body(Body::point(Cartesian::from([3.0, 1.0])))?;
    ///
    /// let positions_tag_order: Vec<_> = microstate
    ///     .iter_sites_tag_order()
    ///     .map(|s| s.properties.position)
    ///     .collect();
    /// assert_eq!(
    ///     positions_tag_order,
    ///     vec![[3.0, 1.0].into(), [-1.0, 2.0].into()]
    /// );
    ///
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn iter_sites_tag_order(&self) -> impl Iterator<Item = &Site<S>> {
        self.sites.iter_tag_order()
    }

    /// Get the spatial data structure.
    #[inline]
    pub fn spatial_data(&self) -> &X {
        &self.spatial_data
    }
}

impl<P, B, S, X, C> Microstate<B, S, X, C>
where
    S: Position<Position = P>,
    X: PointsNearBall<P, SiteKey>,
{
    /// Find sites near a point in space.
    ///
    /// Iterate over all sites and ghost sites within a distance `r` of the
    /// given `point`, *and possibly other sites as well*. All sites produced
    /// by this iterator will be in the system reference frame. No wrapping is
    /// required for ghost sites, which will be slightly outside the boundary
    /// condition. When a ghost site is provided by the iterator, its `site_tag`
    /// and `body_tag` will match that of the actual site.
    ///
    /// The iterator does not filter on distance to avoid duplicating effort
    /// as many callers already perform circumsphere checks.
    ///
    /// The caller *may* provide a value for `r` that is larger than the maximum
    /// interaction range. In the current implementation, this is not an error.
    /// However, in such cases `iter_sites_near` will only iterate over the placed
    /// ghosts which are within the boundary's `maximum_interaction_range`.
    ///
    /// In other words, `iter_sites_near` is meant for use with pairwise functions
    /// that follow the minimum image convention.
    #[inline]
    #[expect(
        clippy::missing_panics_doc,
        reason = "Will panic only due to a bug in hoomd-rs."
    )]
    pub fn iter_sites_near(&self, point: &P, r: f64) -> impl Iterator<Item = &Site<S>> {
        let potential_sites = self.spatial_data.points_near_ball(point, r);
        potential_sites.map(|k| match k {
            SiteKey::Primary(tag) => {
                let index =
                    self.sites.indices[tag].expect("sites and spatial data should be consistent");
                &self.sites.items[index]
            }
            SiteKey::Ghost(tag) => {
                let index =
                    self.ghosts.indices[tag].expect("ghosts and spatial data should be consistent");
                &self.ghosts.items[index]
            }
        })
    }
}

/// Manipulate the microstate as a whole.
impl<P, B, S, X, C> Microstate<B, S, X, C>
where
    P: Copy,
    B: Clone + Transform<S> + Position<Position = P>,
    S: Clone + Position<Position = P> + Default,
    C: Clone + Wrap<B> + Wrap<S> + GenerateGhosts<S> + MapPoint<P>,
    X: Clone + PointUpdate<P, SiteKey>,
{
    /// Clone the microstate, mapping or wrapping bodies into a new boundary.
    ///
    /// The resulting microstate contains the same bodies and sites as the source.
    /// All bodies and sites maintain the same index order and tags.
    ///
    /// `should_map_body` will be called on every body in the microstate. When
    /// it returns `true`, `clone_with_boundary` will map the body's position
    /// from `self.boundary` to `new_boundary` using [`MapPoint`]. When
    /// `should_map_body` returns `false`, the `clone_with_boundary` wraps
    /// the body's unmodified position into `new_boundary`. That wrap may fail,
    /// especially in closed (or partially closed) boundary conditions.
    ///
    /// [`MapPoint`]: hoomd_geometry::MapPoint
    ///
    /// # Errors
    ///
    /// [`Error::UpdateBody`] when some body or site cannot be wrapped into the
    /// new boundary.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::{
    ///     Body, Microstate,
    ///     boundary::Closed,
    ///     property::{Point, Position},
    /// };
    /// use hoomd_vector::Cartesian;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// let square = Closed(Rectangle::with_equal_edges(10.0.try_into()?));
    /// let microstate = Microstate::builder()
    ///     .boundary(square)
    ///     .bodies([Body::point(Cartesian::from([1.0, 2.0]))])
    ///     .bodies([Body::point(Cartesian::from([3.0, 4.0]))])
    ///     .try_build()?;
    ///
    /// let new_square = Closed(Rectangle::with_equal_edges(20.0.try_into()?));
    ///
    /// let new_microstate =
    ///     microstate.clone_with_boundary(new_square, |body| body.tag > 0)?;
    ///
    /// assert_eq!(
    ///     *new_microstate.bodies()[0].item.properties.position(),
    ///     Cartesian::from([1.0, 2.0])
    /// );
    /// assert_eq!(
    ///     *new_microstate.bodies()[1].item.properties.position(),
    ///     Cartesian::from([6.0, 8.0])
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[allow(
        clippy::missing_inline_in_public_items,
        reason = "extremely expensive methods should not be inlined"
    )]
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    pub fn clone_with_boundary<F>(
        &self,
        new_boundary: C,
        should_map_body: F,
    ) -> Result<Microstate<B, S, X, C>, Error>
    where
        F: Fn(&Tagged<Body<B, S>>) -> bool,
    {
        // clone_with_boundary is used in Monte Carlo methods, such as box trial
        // moves. Callers expect that any new microstate produced maintains
        // the same body/site tag associations, including the same set of free
        // tags as there may be external code that refers to specific bodies
        // by tag. Therefore, this method cannot construct a new microstate and
        // add bodies to it. A full clone is not strictly necessary to preserve
        // tags, but it is the lowest effort approach.
        let mut new_microstate = self.clone();

        // MC methods require the clone as they keep the old microstate for
        // rejected moves. MD methods do not need to clone.
        // TODO: Refactor this code into a method like `set_boundary_and_update_bodies`.
        // It would take a callable that updates the bodies so that MD methods could
        // scale both position and velocity appropriately.

        new_microstate.boundary = new_boundary;

        for body_index in 0..new_microstate.bodies().len() {
            let tagged_body = &new_microstate.bodies()[body_index];
            let mut new_properties = tagged_body.item.properties.clone();
            if should_map_body(tagged_body) {
                *new_properties.position_mut() = self
                    .boundary
                    .map_point(*new_properties.position(), &new_microstate.boundary)
                    .expect("body position should be inside the boundary");
            }

            new_microstate.update_body_properties(body_index, new_properties)?;
        }

        Ok(new_microstate)
    }
}

/// Choose parameters when constructing a [`Microstate`].
///
/// Use a [`MicrostateBuilder`] to choose the values of optional parameters when
/// constructing a [`Microstate`]. Some parameters, such as `seed` and `step`,
/// cannot be directly modified after building the [`Microstate`].
///
/// # Example
///
/// ```
/// use hoomd_microstate::{Body, Microstate};
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::builder()
///     .step(100_000)
///     .seed(0x1234abcd)
///     .bodies([
///         Body::point(Cartesian::from([1.0, 0.0])),
///         Body::point(Cartesian::from([-1.0, 2.0])),
///     ])
///     .try_build()?;
///
/// assert_eq!(microstate.step(), 100_000);
/// assert_eq!(microstate.seed(), 0x1234abcd);
/// assert_eq!(microstate.bodies().len(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MicrostateBuilder<B, S = B, X = AllPairs<SiteKey>, C = Open> {
    /// The initial value for step in the resulting [`Microstate`].
    step: u64,

    /// The random number seed to set in the resulting [`Microstate`].
    seed: u32,

    /// Bodies to add to the resulting [`Microstate`].
    bodies: Vec<Body<B, S>>,

    /// Spatial data structure to use in the resulting [`Microstate`].
    spatial_data: X,

    /// Boundary conditions to apply in the resulting [`Microstate`].
    boundary: C,
}

impl<B, S, X, C> MicrostateBuilder<B, S, X, C> {
    /// Choose the boundary conditions in the resulting [`Microstate`].
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::{Microstate, boundary::Closed};
    /// use hoomd_spatial::AllPairs;
    /// use hoomd_vector::Cartesian;
    ///
    /// # use hoomd_microstate::property::Point;
    /// # type BodyProperties = Point<Cartesian<2>>;
    /// # type SiteProperties = Point<Cartesian<2>>;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let square = Closed(Rectangle::with_equal_edges(10.0.try_into()?));
    ///
    /// let microstate = Microstate::<BodyProperties, SiteProperties>::builder()
    ///     .boundary(square)
    ///     .try_build()?;
    ///
    /// assert_eq!(microstate.boundary().0.edge_lengths[0].get(), 10.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn boundary<C2>(self, boundary: C2) -> MicrostateBuilder<B, S, X, C2> {
        MicrostateBuilder::<B, S, X, C2> {
            step: self.step,
            seed: self.seed,
            bodies: self.bodies,
            spatial_data: self.spatial_data,
            boundary,
        }
    }

    /// Set the spatial data structure in the resulting [`Microstate`].
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Microstate, SiteKey};
    /// use hoomd_spatial::VecCell;
    /// use hoomd_vector::Cartesian;
    ///
    /// # use hoomd_microstate::property::Point;
    /// # type BodyProperties = Point<Cartesian<2>>;
    /// # type SiteProperties = Point<Cartesian<2>>;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// let cell_list = VecCell::builder()
    ///     .nominal_search_radius(2.5.try_into()?)
    ///     .build();
    ///
    /// let microstate = Microstate::<BodyProperties, SiteProperties>::builder()
    ///     .spatial_data(cell_list)
    ///     .try_build()?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn spatial_data<X2>(self, spatial_data: X2) -> MicrostateBuilder<B, S, X2, C> {
        MicrostateBuilder::<B, S, X2, C> {
            step: self.step,
            seed: self.seed,
            bodies: self.bodies,
            spatial_data,
            boundary: self.boundary,
        }
    }

    /// Choose the initial step in the resulting [`Microstate`].
    ///
    /// The default `step` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Microstate, boundary::Open, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # type BodyProperties = Point<Cartesian<2>>;
    /// # type SiteProperties = Point<Cartesian<2>>;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = Microstate::<BodyProperties, SiteProperties>::builder()
    ///     .step(100_000)
    ///     .try_build()?;
    ///
    /// assert_eq!(microstate.step(), 100_000);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn step(mut self, step: u64) -> Self {
        self.step = step;
        self
    }

    /// Choose the random number seed in the resulting [`Microstate`].
    ///
    /// The default `seed` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Microstate, boundary::Open, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # type BodyProperties = Point<Cartesian<2>>;
    /// # type SiteProperties = Point<Cartesian<2>>;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = Microstate::<BodyProperties, SiteProperties>::builder()
    ///     .seed(0x1234abcd)
    ///     .try_build()?;
    ///
    /// assert_eq!(microstate.seed(), 0x1234abcd);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Add bodies to the resulting [`Microstate`].
    ///
    /// All bodies will be appended when this method is called multiple times.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::builder()
    ///     .bodies([
    ///         Body::point(Cartesian::from([1.0, 0.0])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])
    ///     .try_build()?;
    ///
    /// assert_eq!(microstate.bodies().len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn bodies<T>(mut self, bodies: T) -> Self
    where
        T: IntoIterator<Item = Body<B, S>>,
    {
        self.bodies.extend(bodies);
        self
    }

    /// Construct a [`Microstate`] with the chosen options.
    ///
    /// # Errors
    ///
    /// See [`Microstate::extend_bodies()`].
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::builder()
    ///     .step(100_000)
    ///     .seed(0x1234abcd)
    ///     .bodies([
    ///         Body::point(Cartesian::from([1.0, 0.0])),
    ///         Body::point(Cartesian::from([-1.0, 2.0])),
    ///     ])
    ///     .try_build()?;
    ///
    /// assert_eq!(microstate.step(), 100_000);
    /// assert_eq!(microstate.seed(), 0x1234abcd);
    /// assert_eq!(microstate.bodies().len(), 2);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn try_build<P>(self) -> Result<Microstate<B, S, X, C>, Error>
    where
        P: Copy,
        B: Transform<S> + Position<Position = P>,
        S: Position<Position = P> + Default,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
        X: PointUpdate<P, SiteKey>,
    {
        let mut microstate = Microstate {
            step: self.step,
            substep: 0,
            seed: self.seed,
            boundary: self.boundary,
            bodies: VecWithTags::new(),
            sites: VecWithTags::new(),
            bodies_sites: Vec::new(),
            ghosts: VecWithTags::new(),
            sites_ghosts: Vec::new(),
            spatial_data: self.spatial_data,
        };

        microstate.spatial_data.clear();

        microstate.extend_bodies(self.bodies)?;

        Ok(microstate)
    }
}

impl<B, S, X, C> fmt::Display for Microstate<B, S, X, C>
where
    X: fmt::Display,
{
    /// Summarize the contents of the microstate.
    ///
    /// This is a slow operation. It is meant to be printed to logs only
    /// occasionally, such as at the end of a benchmark or simulation.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::VecCell;
    /// use log::info;
    ///
    /// let vec_cell = VecCell::<usize, 3>::default();
    ///
    /// info!("{vec_cell}");
    /// ```
    #[allow(
        clippy::missing_inline_in_public_items,
        reason = "no need to inline display"
    )]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Microstate:")?;
        writeln!(f, "- step, substep: {}, {}.", self.step, self.substep)?;
        writeln!(f, "- {} bodies.", self.bodies().len())?;
        writeln!(
            f,
            "- {} sites / {} ghosts.",
            self.sites.items.len(),
            self.ghosts.items.len()
        )?;
        write!(f, "{}", self.spatial_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        boundary::{self, Closed, Periodic},
        property::Point,
    };
    use hoomd_geometry::shape::Hypercuboid;
    use hoomd_spatial::{HashCell, VecCell};
    use hoomd_vector::Cartesian;

    use approxim::assert_relative_eq;
    use rand::{Rng, SeedableRng, distr::Distribution, rngs::StdRng, seq::SliceRandom};
    use rstest::*;
    use std::collections::{HashMap, HashSet};

    // The doc tests above cover all the trivial cases for every method which
    // are not repeated here. The following tests perform self-consistency
    // checks on the internal data structures after calling many methods randomly.

    const N_STEPS: usize = 1024;
    const MAX_BODY_SIZE: usize = 20;
    const MAX_INITIAL_BODY_COORDINATE: f64 = 10.0;
    const MAX_SITE_COORDINATE: f64 = 5.0;
    const MAX_BODY_TRANSLATE: f64 = 0.125;

    mod open {
        use super::*;
        use assert2::{assert, check};
        use rand::RngExt;

        fn create_body<R: Rng>(rng: &mut R) -> Body<Point<Cartesian<2>>> {
            let mut body = Body::point(rng.random::<Cartesian<2>>() * MAX_INITIAL_BODY_COORDINATE);

            let n = rng.random_range(1..MAX_BODY_SIZE);
            body.sites = (0..n)
                .map(|_| Point::new(rng.random::<Cartesian<2>>() * MAX_SITE_COORDINATE))
                .collect();

            body
        }

        fn test_consistency<X>(seed: u64)
        where
            X: PointUpdate<Cartesian<2>, SiteKey> + PointsNearBall<Cartesian<2>, SiteKey> + Default,
        {
            // Rather than crafting many corner cases by hand, generate many
            // microstates randomly by adding, removing, and updating bodies.
            // Validate the internal consistency of the microstate when compared
            // to an alternate reference.

            let mut rng = StdRng::seed_from_u64(seed);
            let mut reference_bodies = HashMap::new();
            let mut microstate = Microstate::builder()
                .spatial_data(X::default())
                .try_build()
                .expect("default microstate should be valid");

            for _ in 0..N_STEPS {
                let move_type_r: f64 = rng.random();
                if move_type_r > 0.7 {
                    // Add bodies more often than removing bodies so that typical
                    // test executions will result in a non-empty microstate.
                    let body = create_body(&mut rng);
                    let tag = microstate
                        .add_body(body.clone())
                        .expect("all bodies should be allowed with open boundary conditions");
                    reference_bodies.insert(tag, body);
                } else if move_type_r > 0.5 && !microstate.bodies.is_empty() {
                    let index = rng.random_range(..microstate.bodies.len());
                    let tag = microstate.bodies()[index].tag;
                    microstate.remove_body(index);
                    reference_bodies.remove(&tag);
                } else if !microstate.bodies.is_empty() {
                    let index = rng.random_range(..microstate.bodies.len());
                    let tag = microstate.bodies()[index].tag;
                    let body = reference_bodies
                        .get_mut(&tag)
                        .expect("tags in the microstate should also be present in the reference");

                    body.properties.position += rng.random::<Cartesian<2>>() * MAX_BODY_TRANSLATE;
                    microstate
                        .update_body_properties(index, body.properties)
                        .expect("all bodies should be allowed with open boundary conditions");
                }
            }

            check!(microstate.bodies.len() == reference_bodies.len());
            check!(
                microstate.sites.len()
                    == reference_bodies.values().map(|body| body.sites.len()).sum()
            );

            for (tag, optional_index) in microstate.bodies.indices.iter().enumerate() {
                if let Some(index) = optional_index {
                    check!(microstate.bodies()[*index].tag == tag);
                    check!(reference_bodies.contains_key(&tag));
                } else {
                    check!(!reference_bodies.contains_key(&tag));
                }
            }

            for (tag, body) in &reference_bodies {
                let body_index = microstate.body_indices()[*tag]
                    .expect("tags in the reference should also be present in the microstate");
                check!(microstate.bodies()[body_index].item == *body);
            }

            for (tag, optional_index) in microstate.sites.indices.iter().enumerate() {
                if let Some(index) = optional_index {
                    check!(microstate.sites()[*index].site_tag == tag);
                }
            }

            check!(microstate.spatial_data().len() == microstate.sites.len());
            for site in microstate.sites() {
                let body_index = microstate.body_indices()[site.body_tag]
                    .expect("tags in the microstate should also be in the reference");
                check!(microstate.bodies_sites[body_index].contains(&site.site_tag));
                check!(
                    microstate
                        .spatial_data()
                        .contains_key(&SiteKey::Primary(site.site_tag))
                );
            }

            assert!(microstate.bodies().len() == microstate.bodies_sites.len());
            for (body, body_sites) in microstate
                .bodies()
                .iter()
                .zip(microstate.bodies_sites.iter())
            {
                assert!(body.item.sites.len() == body_sites.len());
                for site_tag in body_sites {
                    let site_index = microstate.site_indices()[*site_tag]
                        .expect("body_sites should be consistent with site_indices");
                    check!(microstate.sites()[site_index].body_tag == body.tag);
                }
            }

            for (body_index, body) in microstate.bodies().iter().enumerate() {
                for (system_site, local_site) in microstate
                    .iter_body_sites(body_index)
                    .zip(body.item.sites.iter())
                {
                    check!(system_site.body_tag == microstate.bodies()[body_index].tag);
                    check!(system_site.properties == body.item.properties.transform(local_site));
                }
            }
        }

        #[rstest]
        fn test_consistency_all_pairs(#[values(1, 2, 3, 4)] seed: u64) {
            test_consistency::<AllPairs<SiteKey>>(seed);
        }

        #[rstest]
        fn test_consistency_hash_cell(#[values(5, 6, 7, 8)] seed: u64) {
            test_consistency::<HashCell<SiteKey, 2>>(seed);
        }

        #[rstest]
        fn test_consistency_vec_cell(#[values(9, 10, 11, 12)] seed: u64) {
            test_consistency::<VecCell<SiteKey, 2>>(seed);
        }

        #[rstest]
        fn remove_all(#[values(1, 2, 3, 4)] seed: u64) {
            let mut microstate = Microstate::new();
            let mut rng = StdRng::seed_from_u64(seed);

            for _ in 0..N_STEPS {
                let body = create_body(&mut rng);
                microstate
                    .add_body(body)
                    .expect("all bodies should be allowed in open boundary conditions");
            }

            let mut removal_order = (0..N_STEPS).collect::<Vec<_>>();
            removal_order.shuffle(&mut rng);

            for body_tag in removal_order {
                let body_index = microstate.body_indices()[body_tag]
                    .expect("body tags should be assigned in order");
                microstate.remove_body(body_index);
            }

            check!(microstate.bodies().is_empty());
            check!(microstate.bodies_sites.is_empty());
            check!(microstate.sites().is_empty());
        }
    }

    mod closed {
        use super::*;
        use assert2::check;

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

        #[rstest]
        fn add_body_outside(square: Closed<Hypercuboid<2>>) {
            let mut microstate = Microstate::builder()
                .boundary(square)
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            check!(
                microstate.add_body(Body::point(Cartesian::from([2.0, 0.0])))
                    == Err(Error::AddBody(0, boundary::Error::CannotWrapProperties))
            );
        }

        #[rstest]
        fn update_body_outside(square: Closed<Hypercuboid<2>>) {
            let mut microstate = Microstate::builder()
                .boundary(square)
                .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            check!(
                microstate.update_body_properties(
                    0,
                    Point {
                        position: [2.0, 0.0].into()
                    }
                ) == Err(Error::UpdateBody(0, boundary::Error::CannotWrapProperties))
            );
        }

        #[rstest]
        fn add_site_outside(square: Closed<Hypercuboid<2>>) {
            let body = Body {
                properties: Point::new(Cartesian::from([1.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };

            let mut microstate = Microstate::builder()
                .boundary(square)
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            check!(
                microstate.add_body(body)
                    == Err(Error::AddBody(0, boundary::Error::CannotWrapProperties))
            );
        }

        #[rstest]
        fn update_site_outside(square: Closed<Hypercuboid<2>>) {
            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };

            let mut microstate = Microstate::builder()
                .boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            check!(
                microstate.update_body_properties(
                    0,
                    Point {
                        position: [1.0, 0.0].into()
                    }
                ) == Err(Error::UpdateBody(0, boundary::Error::CannotWrapProperties))
            );
        }
    }

    mod periodic {
        use super::*;
        use assert2::{assert, check};
        use rand::RngExt;

        fn create_body<R: Rng>(
            rng: &mut R,
            boundary: &Periodic<Hypercuboid<2>>,
        ) -> Body<Point<Cartesian<2>>> {
            let mut body = Body::point(boundary.sample(rng));

            let n = rng.random_range(1..MAX_BODY_SIZE);
            body.sites = (0..n)
                .map(|_| Point::new(rng.random::<Cartesian<2>>() * MAX_SITE_COORDINATE))
                .collect();

            body
        }

        #[fixture]
        fn rectangle() -> Periodic<Hypercuboid<2>> {
            let cuboid = Hypercuboid {
                edge_lengths: [
                    10.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    20.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };
            Periodic::new(1.0, cuboid)
                .expect("hard-coded interaction range is less than the box plane distance")
        }

        #[rstest]
        fn add_body_outside(rectangle: Periodic<Hypercuboid<2>>) {
            let mut microstate = Microstate::builder()
                .boundary(rectangle)
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            assert!(microstate.add_body(Body::point(Cartesian::from([11.0, -21.0]))) == Ok(0));

            let body = &microstate.bodies()[0].item;
            assert_relative_eq!(body.properties.position, [1.0, -1.0].into(), epsilon = 1e-6);
            check!(microstate.ghosts().len() == 0);
        }

        #[rstest]
        fn update_body_outside(rectangle: Periodic<Hypercuboid<2>>) {
            let mut microstate = Microstate::builder()
                .boundary(rectangle)
                .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            assert_eq!(
                microstate.update_body_properties(
                    0,
                    Point {
                        position: [11.0, -21.0].into()
                    }
                ),
                Ok(())
            );

            let body = &microstate.bodies()[0].item;
            assert_relative_eq!(body.properties.position, [1.0, -1.0].into(), epsilon = 1e-6);
            check!(microstate.ghosts().len() == 0);
        }

        #[rstest]
        fn add_site_outside(rectangle: Periodic<Hypercuboid<2>>) {
            let body = Body {
                properties: Point::new(Cartesian::from([4.5, 1.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };

            let mut microstate = Microstate::builder()
                .boundary(rectangle)
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            check!(microstate.add_body(body) == Ok(0));

            let body = &microstate.bodies()[0].item;
            assert_relative_eq!(body.properties.position, [4.5, 1.0].into(), epsilon = 1e-6);

            let site = &microstate.sites()[0];
            assert_relative_eq!(site.properties.position, [-4.5, 1.0].into(), epsilon = 1e-6);

            assert!(microstate.ghosts().len() == 1);
            let ghost = &microstate.ghosts()[0];
            assert_relative_eq!(ghost.properties.position, [5.5, 1.0].into(), epsilon = 1e-6);

            check!(ghost.site_tag == site.site_tag);
            check!(ghost.body_tag == site.body_tag);
        }

        #[rstest]
        fn update_site_outside(rectangle: Periodic<Hypercuboid<2>>) {
            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };

            let mut microstate = Microstate::builder()
                .boundary(rectangle)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            assert!(
                microstate.update_body_properties(
                    0,
                    Point {
                        position: [4.5, 1.0].into()
                    }
                ) == Ok(())
            );

            let body = &microstate.bodies()[0].item;
            assert_relative_eq!(body.properties.position, [4.5, 1.0].into(), epsilon = 1e-6);

            let site = &microstate.sites()[0];
            assert_relative_eq!(site.properties.position, [-4.5, 1.0].into(), epsilon = 1e-6);

            assert!(microstate.ghosts().len() == 1);
            let ghost = &microstate.ghosts()[0];
            assert_relative_eq!(ghost.properties.position, [5.5, 1.0].into(), epsilon = 1e-6);

            check!(ghost.site_tag == site.site_tag);
            check!(ghost.body_tag == site.body_tag);

            assert!(
                microstate.update_body_properties(
                    0,
                    Point {
                        position: [0.0, 0.0].into()
                    }
                ) == Ok(())
            );

            check!(microstate.ghosts().len() == 0);
        }

        fn test_consistency<X>(seed: u64, rectangle: Periodic<Hypercuboid<2>>)
        where
            X: PointUpdate<Cartesian<2>, SiteKey> + PointsNearBall<Cartesian<2>, SiteKey> + Default,
        {
            // The boundary-specific unit tests validate that the *right*
            // ghosts are created. This test throws random body insertions,
            // updates, and removals and ensures that the internal ghost/site
            // data structures remain consistent.

            let mut rng = StdRng::seed_from_u64(seed);
            let mut microstate = Microstate::builder()
                .boundary(rectangle)
                .spatial_data(X::default())
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            for _ in 0..N_STEPS {
                let move_type_r: f64 = rng.random();
                if move_type_r > 0.7 {
                    // Add bodies more often than removing bodies so that typical
                    // test executions will result in a non-empty microstate.
                    let body = create_body(&mut rng, microstate.boundary());
                    microstate
                        .add_body(body.clone())
                        .expect("all bodies should be wrapped into the boundary");
                } else if move_type_r > 0.5 && !microstate.bodies.is_empty() {
                    let index = rng.random_range(..microstate.bodies.len());
                    microstate.remove_body(index);
                } else if !microstate.bodies.is_empty() {
                    let index = rng.random_range(..microstate.bodies.len());
                    let mut body_properties = microstate.bodies()[index].item.properties;

                    body_properties.position += rng.random::<Cartesian<2>>() * MAX_BODY_TRANSLATE;
                    microstate
                        .update_body_properties(index, body_properties)
                        .expect("all bodies should be wrapped into the boundary");
                }
            }

            // open::consistency validates most of the internal data structures
            // in Microstate. periodic::consistency only needs to validate
            // the consistency of the ghosts.
            let mut sites_with_ghosts = HashSet::new();

            assert!(!microstate.ghosts().is_empty());
            check!(
                microstate.spatial_data().len() == microstate.sites.len() + microstate.ghosts.len()
            );
            for (ghost, ghost_tag) in microstate.ghosts().iter().zip(&microstate.ghosts.tags) {
                let parent_site_index = microstate.site_indices()[ghost.site_tag]
                    .expect("every ghost should have a parent site");
                sites_with_ghosts.insert(parent_site_index);
                let parent = &microstate.sites()[parent_site_index];

                check!(parent.site_tag == ghost.site_tag);
                check!(parent.body_tag == ghost.body_tag);
                check!(
                    microstate
                        .spatial_data()
                        .contains_key(&SiteKey::Ghost(*ghost_tag))
                );
            }

            for (site_index, site_ghosts) in microstate.sites_ghosts.iter().enumerate() {
                if sites_with_ghosts.contains(&site_index) {
                    for ghost_tag in site_ghosts {
                        let ghost_index = microstate.ghosts.indices[*ghost_tag]
                            .expect("ghost tag in sites_ghosts should be present");
                        let ghost = &microstate.ghosts()[ghost_index];
                        let site = &microstate.sites()[site_index];
                        check!(site.site_tag == ghost.site_tag);
                        check!(site.body_tag == ghost.body_tag);
                    }
                } else {
                    check!(site_ghosts.is_empty());
                }
            }
        }

        #[rstest]
        fn test_consistency_all_pairs(
            #[values(1, 2, 3, 4)] seed: u64,
            rectangle: Periodic<Hypercuboid<2>>,
        ) {
            test_consistency::<AllPairs<SiteKey>>(seed, rectangle);
        }

        #[rstest]
        fn test_consistency_hash_cell(
            #[values(5, 6, 7, 8)] seed: u64,
            rectangle: Periodic<Hypercuboid<2>>,
        ) {
            test_consistency::<HashCell<SiteKey, 2>>(seed, rectangle);
        }

        #[rstest]
        fn test_consistency_vec_cell(
            #[values(9, 10, 11, 12)] seed: u64,
            rectangle: Periodic<Hypercuboid<2>>,
        ) {
            test_consistency::<VecCell<SiteKey, 2>>(seed, rectangle);
        }

        #[rstest]
        fn remove_all(#[values(1, 2, 3, 4)] seed: u64, rectangle: Periodic<Hypercuboid<2>>) {
            let mut microstate = Microstate::builder()
                .boundary(rectangle)
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");
            let mut rng = StdRng::seed_from_u64(seed);

            for _ in 0..N_STEPS {
                let body = create_body(&mut rng, microstate.boundary());
                microstate
                    .add_body(body)
                    .expect("all bodies should be allowed in open boundary conditions");
            }

            let mut removal_order = (0..N_STEPS).collect::<Vec<_>>();
            removal_order.shuffle(&mut rng);

            for body_tag in removal_order {
                let body_index = microstate.body_indices()[body_tag]
                    .expect("body tags should be assigned in order");
                microstate.remove_body(body_index);
            }

            check!(microstate.bodies().is_empty());
            check!(microstate.bodies_sites.is_empty());
            check!(microstate.sites().is_empty());
            check!(microstate.ghosts().is_empty());
        }
    }

    // TODO: Test iter_sites_near: with and without periodic boundaries
}

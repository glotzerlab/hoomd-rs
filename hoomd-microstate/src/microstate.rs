// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Microstate`] and related types.

use std::{cmp::Reverse, collections::BinaryHeap};
use tinyvec::ArrayVec;

use crate::{
    Body, Error, Site, Transform,
    boundary::{GenerateGhosts, MAX_GHOSTS, Open, Wrap},
    property::Position,
};

use hoomd_utility::random::Counter;
use hoomd_vector::Vector;

/// Track a unique identifier for an item in [`Microstate`].
#[derive(Clone, Debug, Default, PartialEq)]
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
#[derive(Clone)]
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
    /// Construct an empty vector with tagged items.
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
/// use hoomd_microstate::{
///     Body, Microstate, MicrostateBuilder, boundary::Closed,
/// };
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let square = Closed(Rectangle::with_equal_edges(10.0.try_into()?));
///
/// let microstate = MicrostateBuilder::with_boundary(square)
///     .seed(0x43abf1)
///     .step(100_000)
///     .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
///     .try_build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Microstate<B, S = B, C = Open> {
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
    sites_ghosts: Vec<ArrayVec<[usize; MAX_GHOSTS]>>,

    /// The range of allowed particle positions and a description of any periodicity.
    boundary: C,
}

impl<B, S> Default for Microstate<B, S, Open> {
    /// Construct an empty microstate with open boundary conditions.
    ///
    /// See [`Microstate::new`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S> Microstate<B, S, Open> {
    /// Construct an empty microstate with open boundary conditions.
    ///
    /// The microstate starts at step 0, substep 0, random number seed 0,
    /// and has no bodies.
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
        }
    }
}

/// Access and manage the simulation step, substep, RNG seeds.
impl<B, S, C> Microstate<B, S, C> {
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
    /// use hoomd_microstate::{Microstate, MicrostateBuilder};
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = MicrostateBuilder::new()
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
    /// use hoomd_microstate::{Microstate, MicrostateBuilder};
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    ///
    /// # type BodyProperties = Point<Cartesian<2>>;
    /// # type SiteProperties = Point<Cartesian<2>>;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = MicrostateBuilder::<BodyProperties, SiteProperties>::new()
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
impl<B, S, C> Microstate<B, S, C> {
    /// Get the boundary condition.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::{Microstate, MicrostateBuilder, boundary::Closed};
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// let square = Closed(Rectangle::with_equal_edges(10.0.try_into()?));
    /// let microstate = MicrostateBuilder::with_boundary(square)
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

    /// Get the boundary condition (mutable).
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::{Microstate, MicrostateBuilder, boundary::Closed};
    /// # use hoomd_microstate::{Body, property::Point};
    /// # use hoomd_vector::Cartesian;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///
    /// let square = Closed(Rectangle::with_equal_edges(10.0.try_into()?));
    /// let mut microstate = MicrostateBuilder::with_boundary(square)
    /// # .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
    ///     .try_build()?;
    ///
    /// microstate.boundary_mut().0.edge_lengths[0] = 11.0.try_into()?;
    /// assert_eq!(microstate.boundary().0.edge_lengths[0].get(), 11.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// TODO: Replace with setter. `boundary_mut` allows the caller to create an
    /// invalid microstate by changing the boundary in such a way that sites may
    /// be outside. Changing the boundary will also require regenerating ghost
    /// sites. Just checking for a valid boundary on set will pose some difficulty
    /// to the caller. To increase the boundary, the caller will need to set
    /// the new boundary and then move the bodies. To decrease the boundary, the
    /// caller will need to move the bodies and then set the boundary. Perhaps a
    /// `set_boundary_and_update_bodies` method that does both simultaneously would
    /// solve this? It could take a function that updates the bodies along with the
    /// new boundary.
    #[inline]
    pub fn boundary_mut(&mut self) -> &mut C {
        &mut self.boundary
    }
}

/// Manage bodies in the microstate.
impl<V, B, S, C> Microstate<B, S, C>
where
    B: Transform<S> + Position<Vector = V>,
    S: Position<Vector = V> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    /// Update the ghosts of a site.
    ///
    /// Given a site in the boundary, update that site's ghosts to be consistent
    /// with that site's properties. This may require adding or removing ghosts.
    fn update_site_ghosts(
        sites: &VecWithTags<Site<S>>,
        site_index: usize,
        boundary: &C,
        sites_ghosts: &mut [ArrayVec<[usize; MAX_GHOSTS]>],
        ghosts: &mut VecWithTags<Site<S>>,
    ) {
        let site = &sites.items[site_index];
        let new_ghosts = boundary.generate_ghosts(&site.properties);
        let ghost_tags = &mut sites_ghosts[site_index];

        if ghost_tags.len() < new_ghosts.len() {
            let ghosts_to_add = new_ghosts.len() - ghost_tags.len();
            for _ in 0..ghosts_to_add {
                let ghost_tag = ghosts.push(Site {
                    site_tag: site.site_tag,
                    body_tag: site.body_tag,
                    properties: S::default(),
                });
                ghost_tags.push(ghost_tag);
            }
        } else if ghost_tags.len() > new_ghosts.len() {
            let ghosts_to_remove = ghost_tags.len() - new_ghosts.len();
            for ghost_tag in ghost_tags.iter().rev().take(ghosts_to_remove) {
                let ghost_index = ghosts.indices[*ghost_tag]
                    .expect("sites_ghosts and ghost.indices should be consistent");
                ghosts.remove(ghost_index);
            }

            ghost_tags.truncate(new_ghosts.len());
        }

        debug_assert_eq!(ghost_tags.len(), new_ghosts.len());

        for (new_ghost, ghost_tag) in new_ghosts.into_iter().zip(ghost_tags) {
            let ghost_index = ghosts.indices[*ghost_tag]
                .expect("sites_ghosts and ghost.indices should be consistent");
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

            self.sites.push(Site {
                site_tag,
                properties: self
                    .boundary
                    .wrap(body.properties.transform(s))
                    .expect("sites should be validated as wrappable prior to this loop"),
                body_tag,
            });
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
    /// use hoomd_microstate::{Body, Microstate, MicrostateBuilder};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = MicrostateBuilder::new()
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
                self.ghosts.remove(ghost_index);
            }

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
    /// use hoomd_microstate::{
    ///     Body, Microstate, MicrostateBuilder, property::Point,
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = MicrostateBuilder::new()
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
        B: Transform<S> + Position<Vector = V>,
        S: Position<Vector = V>,
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
            self.sites.items[site_index].properties = self
                .boundary
                .wrap(body.item.properties.transform(&body.item.sites[i]))
                .expect("sites should be validated as wrappable prior to this loop");
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
    /// use hoomd_microstate::{
    ///     Body, Microstate, MicrostateBuilder, property::Point,
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = MicrostateBuilder::new()
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
    }
}

/// Access contents of the microstate.
impl<B, S, C> Microstate<B, S, C> {
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
    /// use hoomd_microstate::{Body, Microstate, MicrostateBuilder};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = MicrostateBuilder::new()
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
    /// use hoomd_microstate::{Body, Microstate, MicrostateBuilder};
    /// use hoomd_vector::{Cartesian, Vector};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = MicrostateBuilder::new()
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
    /// use hoomd_microstate::{Body, Microstate, MicrostateBuilder};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = MicrostateBuilder::new()
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
    /// use hoomd_microstate::{Body, Microstate, MicrostateBuilder};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = MicrostateBuilder::new()
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
    /// use hoomd_microstate::{Body, Microstate, MicrostateBuilder};
    /// use hoomd_vector::{Cartesian, Vector};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = MicrostateBuilder::new()
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
    /// use hoomd_microstate::{Body, Microstate, MicrostateBuilder};
    /// use hoomd_vector::{Cartesian, Vector};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = MicrostateBuilder::new()
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
}

impl<V, B, S, C> Microstate<B, S, C>
where
    S: Position<Vector = V>,
    V: Vector,
{
    /// Find sites near a point in space.
    ///
    /// Iterate over all sites and ghost sites within a distance `r` of the given
    /// `point`. All sites produced by this iterator will be in the system reference
    /// frame and within the given distance metric. No wrapping is required for
    /// ghost sites, which will be slightly outside the boundary condition. When a
    /// ghost site is provided by the iterator, its `site_tag` and `body_tag` will
    /// match that of the actual site.
    ///
    /// The caller *may* provide a value for `r` that is larger than the maximum
    /// interaction range. In the current implementation, this is not an error.
    /// However, in such cases `iter_sites_near` will only iterate over the placed
    /// ghosts which are within the boundary's `maximum_interaction_range`.
    ///
    /// In other words, `iter_sites_near` is meant for use with pairwise functions
    /// that follow the minimum image convention.
    #[inline]
    pub fn iter_sites_near(&self, point: &V, r: f64) -> impl Iterator<Item = &Site<S>> {
        self.sites
            .items
            .iter()
            .chain(self.ghosts.items.iter())
            .filter(move |s| point.distance_squared(s.properties.position()) < r.powi(2))
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
/// use hoomd_microstate::{Body, Microstate, MicrostateBuilder};
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = MicrostateBuilder::new()
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
pub struct MicrostateBuilder<B, S = B, C = Open> {
    /// The initial value for step in the resulting [`Microstate`].
    step: u64,
    /// The random number seed to set in the resulting [`Microstate`].
    seed: u32,
    /// Bodies to add to the resulting [`Microstate`].
    bodies: Vec<Body<B, S>>,
    /// Boundary conditions to apply in the resulting [`Microstate`].
    boundary: C,
}

impl<B, S> MicrostateBuilder<B, S, Open> {
    /// Construct an empty [`MicrostateBuilder`] with open boundary conditions.
    ///
    /// The resulting microstate starts at step 0 and has a random seed of 0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{Microstate, MicrostateBuilder, property::Point};
    /// use hoomd_vector::Cartesian;
    ///
    /// # type BodyProperties = Point<Cartesian<2>>;
    /// # type SiteProperties = Point<Cartesian<2>>;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = MicrostateBuilder::<BodyProperties, SiteProperties>::new()
    ///     .try_build()?;
    ///
    /// assert_eq!(microstate.step(), 0);
    /// assert_eq!(microstate.seed(), 0);
    /// assert_eq!(microstate.bodies().len(), 0);
    /// assert_eq!(*microstate.boundary(), hoomd_microstate::boundary::Open);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        MicrostateBuilder::with_boundary(Open)
    }
}

impl<B, S> Default for MicrostateBuilder<B, S, Open> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S, C> MicrostateBuilder<B, S, C> {
    /// Construct an empty [`MicrostateBuilder`] with the given boundary conditions.
    ///
    /// The resulting microstate starts at step 0 and has a random seed of 0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::{Microstate, MicrostateBuilder, boundary::Closed};
    /// use hoomd_vector::Cartesian;
    ///
    /// # use hoomd_microstate::property::Point;
    /// # type BodyProperties = Point<Cartesian<2>>;
    /// # type SiteProperties = Point<Cartesian<2>>;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let square = Closed(Rectangle::with_equal_edges(10.0.try_into()?));
    ///
    /// let microstate = MicrostateBuilder::<
    ///     BodyProperties,
    ///     SiteProperties,
    ///     Closed<Rectangle>,
    /// >::with_boundary(square)
    /// .try_build()?;
    ///
    /// assert_eq!(microstate.step(), 0);
    /// assert_eq!(microstate.seed(), 0);
    /// assert_eq!(microstate.bodies().len(), 0);
    /// assert_eq!(microstate.boundary().0.edge_lengths[0].get(), 10.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn with_boundary(boundary: C) -> Self {
        Self {
            step: 0,
            seed: 0,
            bodies: Vec::new(),
            boundary,
        }
    }

    /// Choose the initial step in the resulting [`Microstate`].
    ///
    /// The default `step` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_microstate::{
    ///     Microstate, MicrostateBuilder, boundary::Open, property::Point,
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # type BodyProperties = Point<Cartesian<2>>;
    /// # type SiteProperties = Point<Cartesian<2>>;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = MicrostateBuilder::<BodyProperties, SiteProperties>::new()
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
    /// use hoomd_microstate::{
    ///     Microstate, MicrostateBuilder, boundary::Open, property::Point,
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # type BodyProperties = Point<Cartesian<2>>;
    /// # type SiteProperties = Point<Cartesian<2>>;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = MicrostateBuilder::<BodyProperties, SiteProperties>::new()
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
    /// use hoomd_microstate::{Body, Microstate, MicrostateBuilder};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = MicrostateBuilder::new()
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
    /// use hoomd_microstate::{Body, Microstate, MicrostateBuilder};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = MicrostateBuilder::new()
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
    pub fn try_build<V>(self) -> Result<Microstate<B, S, C>, Error>
    where
        B: Transform<S> + Position<Vector = V>,
        S: Position<Vector = V> + Default,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
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
        };

        microstate.extend_bodies(self.bodies)?;

        Ok(microstate)
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
    use hoomd_vector::Cartesian;

    use ::approx::assert_relative_eq;
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

        fn create_body<R: Rng>(rng: &mut R) -> Body<Point<Cartesian<2>>> {
            let mut body = Body::point(rng.random::<Cartesian<2>>() * MAX_INITIAL_BODY_COORDINATE);

            let n = rng.random_range(1..MAX_BODY_SIZE);
            body.sites = (0..n)
                .map(|_| Point::new(rng.random::<Cartesian<2>>() * MAX_SITE_COORDINATE))
                .collect();

            body
        }

        #[rstest]
        fn consistency(#[values(1, 2, 3, 4)] seed: u64) {
            // Rather than crafting many corner cases by hand, generate many
            // microstates randomly by adding, removing, and updating bodies.
            // Validate the internal consistency of the microstate when compared
            // to an alternate reference.

            let mut rng = StdRng::seed_from_u64(seed);
            let mut reference_bodies = HashMap::new();
            let mut microstate = Microstate::new();

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

            assert_eq!(microstate.bodies.len(), reference_bodies.len());
            assert_eq!(
                microstate.sites.len(),
                reference_bodies.values().map(|body| body.sites.len()).sum()
            );

            for (tag, optional_index) in microstate.bodies.indices.iter().enumerate() {
                if let Some(index) = optional_index {
                    assert_eq!(microstate.bodies()[*index].tag, tag);
                    assert!(reference_bodies.contains_key(&tag));
                } else {
                    assert!(!reference_bodies.contains_key(&tag));
                }
            }

            for (tag, body) in &reference_bodies {
                let body_index = microstate.body_indices()[*tag]
                    .expect("tags in the reference should also be present in the microstate");
                assert_eq!(microstate.bodies()[body_index].item, *body);
            }

            for (tag, optional_index) in microstate.sites.indices.iter().enumerate() {
                if let Some(index) = optional_index {
                    assert_eq!(microstate.sites()[*index].site_tag, tag);
                }
            }

            for site in microstate.sites() {
                let body_index = microstate.body_indices()[site.body_tag]
                    .expect("tags in the microstate should also be in the reference");
                assert!(microstate.bodies_sites[body_index].contains(&site.site_tag));
            }

            assert_eq!(microstate.bodies().len(), microstate.bodies_sites.len());
            for (body, body_sites) in microstate
                .bodies()
                .iter()
                .zip(microstate.bodies_sites.iter())
            {
                assert_eq!(body.item.sites.len(), body_sites.len());
                for site_tag in body_sites {
                    let site_index = microstate.site_indices()[*site_tag]
                        .expect("body_sites should be consistent with site_indices");
                    assert_eq!(microstate.sites()[site_index].body_tag, body.tag);
                }
            }

            for (body_index, body) in microstate.bodies().iter().enumerate() {
                for (system_site, local_site) in microstate
                    .iter_body_sites(body_index)
                    .zip(body.item.sites.iter())
                {
                    assert_eq!(system_site.body_tag, microstate.bodies()[body_index].tag);
                    assert_eq!(
                        system_site.properties,
                        body.item.properties.transform(local_site)
                    );
                }
            }
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

            assert!(microstate.bodies().is_empty());
            assert!(microstate.bodies_sites.is_empty());
            assert!(microstate.sites().is_empty());
        }
    }

    mod closed {
        use super::*;

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
            let mut microstate = MicrostateBuilder::with_boundary(square)
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            assert_eq!(
                microstate.add_body(Body::point(Cartesian::from([2.0, 0.0]))),
                Err(Error::AddBody(0, boundary::Error::CannotWrapProperties))
            );
        }

        #[rstest]
        fn update_body_outside(square: Closed<Hypercuboid<2>>) {
            let mut microstate = MicrostateBuilder::with_boundary(square)
                .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            assert_eq!(
                microstate.update_body_properties(
                    0,
                    Point {
                        position: [2.0, 0.0].into()
                    }
                ),
                Err(Error::UpdateBody(0, boundary::Error::CannotWrapProperties))
            );
        }

        #[rstest]
        fn add_site_outside(square: Closed<Hypercuboid<2>>) {
            let body = Body {
                properties: Point::new(Cartesian::from([1.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };

            let mut microstate = MicrostateBuilder::with_boundary(square)
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            assert_eq!(
                microstate.add_body(body),
                Err(Error::AddBody(0, boundary::Error::CannotWrapProperties))
            );
        }

        #[rstest]
        fn update_site_outside(square: Closed<Hypercuboid<2>>) {
            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };

            let mut microstate = MicrostateBuilder::with_boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            assert_eq!(
                microstate.update_body_properties(
                    0,
                    Point {
                        position: [1.0, 0.0].into()
                    }
                ),
                Err(Error::UpdateBody(0, boundary::Error::CannotWrapProperties))
            );
        }
    }

    mod periodic {
        use super::*;

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
            let mut microstate = MicrostateBuilder::with_boundary(rectangle)
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            assert_eq!(
                microstate.add_body(Body::point(Cartesian::from([11.0, -21.0]))),
                Ok(0)
            );

            let body = &microstate.bodies()[0].item;
            assert_relative_eq!(body.properties.position, [1.0, -1.0].into(), epsilon = 1e-6);
            assert_eq!(microstate.ghosts().len(), 0);
        }

        #[rstest]
        fn update_body_outside(rectangle: Periodic<Hypercuboid<2>>) {
            let mut microstate = MicrostateBuilder::with_boundary(rectangle)
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
            assert_eq!(microstate.ghosts().len(), 0);
        }

        #[rstest]
        fn add_site_outside(rectangle: Periodic<Hypercuboid<2>>) {
            let body = Body {
                properties: Point::new(Cartesian::from([4.5, 1.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };

            let mut microstate = MicrostateBuilder::with_boundary(rectangle)
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            assert_eq!(microstate.add_body(body), Ok(0));

            let body = &microstate.bodies()[0].item;
            assert_relative_eq!(body.properties.position, [4.5, 1.0].into(), epsilon = 1e-6);

            let site = &microstate.sites()[0];
            assert_relative_eq!(site.properties.position, [-4.5, 1.0].into(), epsilon = 1e-6);

            assert_eq!(microstate.ghosts().len(), 1);
            let ghost = &microstate.ghosts()[0];
            assert_relative_eq!(ghost.properties.position, [5.5, 1.0].into(), epsilon = 1e-6);

            assert!(ghost.site_tag == site.site_tag);
            assert!(ghost.body_tag == site.body_tag);
        }

        #[rstest]
        fn update_site_outside(rectangle: Periodic<Hypercuboid<2>>) {
            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };

            let mut microstate = MicrostateBuilder::with_boundary(rectangle)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            assert_eq!(
                microstate.update_body_properties(
                    0,
                    Point {
                        position: [4.5, 1.0].into()
                    }
                ),
                Ok(())
            );

            let body = &microstate.bodies()[0].item;
            assert_relative_eq!(body.properties.position, [4.5, 1.0].into(), epsilon = 1e-6);

            let site = &microstate.sites()[0];
            assert_relative_eq!(site.properties.position, [-4.5, 1.0].into(), epsilon = 1e-6);

            assert_eq!(microstate.ghosts().len(), 1);
            let ghost = &microstate.ghosts()[0];
            assert_relative_eq!(ghost.properties.position, [5.5, 1.0].into(), epsilon = 1e-6);

            assert!(ghost.site_tag == site.site_tag);
            assert!(ghost.body_tag == site.body_tag);

            assert_eq!(
                microstate.update_body_properties(
                    0,
                    Point {
                        position: [0.0, 0.0].into()
                    }
                ),
                Ok(())
            );

            assert_eq!(microstate.ghosts().len(), 0);
        }

        #[rstest]
        fn consistency(#[values(1, 2, 3, 4)] seed: u64, rectangle: Periodic<Hypercuboid<2>>) {
            // The boundary-specific unit tests validate that the *right*
            // ghosts are created. This test throws random body insertions,
            // updates, and removals and ensures that the internal ghost/site
            // data structures remain consistent.

            let mut rng = StdRng::seed_from_u64(seed);
            let mut microstate = MicrostateBuilder::with_boundary(rectangle)
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
            for ghost in microstate.ghosts() {
                let parent_site_index = microstate.site_indices()[ghost.site_tag]
                    .expect("every ghost should have a parent site");
                sites_with_ghosts.insert(parent_site_index);
                let parent = &microstate.sites()[parent_site_index];

                assert_eq!(parent.site_tag, ghost.site_tag);
                assert_eq!(parent.body_tag, ghost.body_tag);
            }

            for (site_index, site_ghosts) in microstate.sites_ghosts.iter().enumerate() {
                if sites_with_ghosts.contains(&site_index) {
                    for ghost_tag in site_ghosts {
                        let ghost_index = microstate.ghosts.indices[*ghost_tag]
                            .expect("ghost tag in sites_ghosts should be present");
                        let ghost = &microstate.ghosts()[ghost_index];
                        let site = &microstate.sites()[site_index];
                        assert_eq!(site.site_tag, ghost.site_tag);
                        assert_eq!(site.body_tag, ghost.body_tag);
                    }
                } else {
                    assert!(site_ghosts.is_empty());
                }
            }
        }

        #[rstest]
        fn remove_all(#[values(1, 2, 3, 4)] seed: u64, rectangle: Periodic<Hypercuboid<2>>) {
            let mut microstate = MicrostateBuilder::with_boundary(rectangle)
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

            assert!(microstate.bodies().is_empty());
            assert!(microstate.bodies_sites.is_empty());
            assert!(microstate.sites().is_empty());
            assert!(microstate.ghosts().is_empty());
        }
    }

    // TODO: Test iter_sites_near: with and without periodic boundaries
}

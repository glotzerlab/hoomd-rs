// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Microstate`] and related types.
 */

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::boundary::{Boundary, Open};
use crate::property::Position;
use crate::{Body, Error, Site, Transform};
use hoomd_utility::random::Counter;

/** Track a unique identifier for an item in [`Microstate`].
*/
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tagged<T> {
    /// The unique identifier.
    pub tag: usize,
    /// The tagged item.
    pub item: T,
}

/** Store and manage all the degrees of freedom of a single microstate in phase space.

[`Microstate`] implements the main logic of the crate. See the [crate-level
documentation](crate) for a full overview and the method-specific documentation
for additional details.

The generic type names are:
* `B`: The [`Body::properties`](crate::Body) type.
* `S`: The [`Site::properties`](crate::Site) type.
* `C`: The [`boundary`](crate::boundary) condition type.
*/
#[derive(Clone)]
pub struct Microstate<B, S = B, C = Open> {
    /// Total number of steps that this microstate has been advanced in a simulation model.
    step: u64,

    /// Number of substeps that the simulation has taken during the current simulation step.
    substep: u32,

    /// User chosen random number seed.
    seed: u32,

    /// Bodies in the microstate, stored in index order.
    bodies: Vec<Tagged<Body<B, S>>>,

    /// Indices of the bodies, in tag order.
    body_indices: Vec<Option<usize>>,

    /// Body tags that can be reused.
    free_body_tags: BinaryHeap<Reverse<usize>>,

    /// Sites in the system reference frame.
    sites: Vec<Site<S>>,

    /// Indices of the sites, in tag order.
    site_indices: Vec<Option<usize>>,

    /// Body tags that can be reused.
    free_site_tags: BinaryHeap<Reverse<usize>>,

    /// Tags of the sites associated with the bodies (in body index order).
    bodies_sites: Vec<Vec<usize>>,

    /// The range of allowed particle positions and a description of any periodicity.
    boundary: C,
}

impl<B, S> Default for Microstate<B, S, Open> {
    /** Construct an empty microstate with open boundary conditions.

    See [`Microstate::new`].
    */
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<B, S> Microstate<B, S, Open> {
    /** Construct an empty microstate with open boundary conditions.

    The microstate starts at step 0, substep 0, random number seed 0,
    and has no bodies.

    # Example

    ```
    use hoomd_microstate::{Microstate, property::Point};
    use hoomd_vector::Cartesian;

    let microstate = Microstate::<Point<Cartesian<2>>>::new();
    assert_eq!(microstate.step(), 0);
    assert_eq!(microstate.substep(), 0);
    assert_eq!(microstate.seed(), 0);
    assert_eq!(microstate.bodies().len(), 0);
    assert_eq!(microstate.sites().len(), 0);
    ```
    */
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Microstate {
            step: 0,
            substep: 0,
            seed: 0,
            bodies: Vec::new(),
            body_indices: Vec::new(),
            free_body_tags: BinaryHeap::new(),
            sites: Vec::new(),
            site_indices: Vec::new(),
            free_site_tags: BinaryHeap::new(),
            bodies_sites: Vec::new(),
            boundary: Open,
        }
    }
}

/// Access and manage the simulation step, substep, RNG seeds.
impl<B, S, C> Microstate<B, S, C> {
    /** Get the simulation step.

    # Examples

    Get the step:
    ```
    use hoomd_microstate::{Microstate, property::Point};
    use hoomd_vector::Cartesian;

    let microstate = Microstate::<Point<Cartesian<2>>>::new();
    assert_eq!(microstate.step(), 0);
    ```

    Initialize a microstate with a given step:
    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, property::Point};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::<Point<Cartesian<2>>>::new()
        .step(100_000)
        .try_build()?;
    assert_eq!(microstate.step(), 100_000);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn step(&self) -> u64 {
        self.step
    }

    /** Increment the simulation step.

    Also set the substep to 0.

    # Examples

    Increment the simulation step:
    ```
    use hoomd_microstate::{Microstate, property::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::<Point<Cartesian<2>>>::new();
    microstate.increment_step();

    assert_eq!(microstate.step(), 1);
    ```

    Confirm that `substep` resets to 0:
    ```
    use hoomd_microstate::{Microstate, property::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::<Point<Cartesian<2>>>::new();

    microstate.increment_substep();
    microstate.increment_substep();
    microstate.increment_substep();
    assert_eq!(microstate.substep(), 3);

    microstate.increment_step();

    assert_eq!(microstate.step(), 1);
    assert_eq!(microstate.substep(), 0);
    ```
    */
    #[inline]
    pub fn increment_step(&mut self) {
        self.step += 1;
        self.substep = 0;
    }

    /** Get the simulation substep.

    # Example
    ```
    use hoomd_microstate::{Microstate, property::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::<Point<Cartesian<2>>>::new();
    microstate.increment_substep();

    assert_eq!(microstate.substep(), 1);
    ```
    */
    #[inline]
    #[must_use]
    pub fn substep(&self) -> u32 {
        self.substep
    }

    /** Increment the simulation substep.

    # Example
    ```
    use hoomd_microstate::{Microstate, property::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::<Point<Cartesian<2>>>::new();
    microstate.increment_substep();

    assert_eq!(microstate.substep(), 1);
    ```
    */
    #[inline]
    pub fn increment_substep(&mut self) {
        self.substep += 1;
    }

    /** Get the simulation seed.

    # Examples:

    Get the simulation seed.
    ```
    use hoomd_microstate::{Microstate, property::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::<Point<Cartesian<2>>>::new();

    assert_eq!(microstate.seed(), 0);
    ```

    Initialize a microstate with a given seed:
    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, property::Point};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::<Point<Cartesian<2>>>::new()
        .seed(0x1234abcd)
        .try_build()?;
    assert_eq!(microstate.seed(), 0x1234abcd);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u32 {
        self.seed
    }

    /** Create a partially constructed [`Counter`] from the current step, substep, and seed.

    Use the produced [`Counter`] to make a independent random number generator at each
    substep. Call additional methods on the [`Counter`] first to further differentiate
    the stream.

    # Example

    Make a random number generator unique to this substep:
    ```
    use hoomd_microstate::{Microstate, property::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::<Point<Cartesian<2>>>::new();

    let rng = microstate.counter().make_rng();
    ```

    Make a random number generator unique to a particular particle on this substep:

    ```
    use hoomd_microstate::{Microstate, property::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::<Point<Cartesian<2>>>::new();

    let tag = 10;
    let rng = microstate.counter().index(tag).make_rng();
    ```
    */
    #[inline]
    pub fn counter(&self) -> Counter {
        Counter::new(self.step, self.substep, self.seed)
    }
}

/// Access and manage the boundary condition.
impl<B, S, C> Microstate<B, S, C> {
    /** Get the boundary condition.

    # Example

    TODO: Write once we have a non-trivial boundary type.
    */
    #[inline]
    pub fn boundary(&self) -> &C {
        &self.boundary
    }

    /** Get the boundary condition (mutable).

    # Example

    TODO: Write once we have a non-trivial boundary type.
    */
    #[inline]
    pub fn boundary_mut(&mut self) -> &mut C {
        &mut self.boundary
    }
}

/** Manage bodies in the microstate.
*/
impl<B, S, C> Microstate<B, S, C> {
    /** Add a new body to the microstate.

    Each body is assigned a unique tag. The first body is given tag 0,
    the second is given tag 1, and so on. When a body is removed (see
    [`Microstate::remove_body()`], its tag becomes unused. The next call to
    `add_body` will assign the smallest unused tag.

    `add_body` also adds the body's sites to the microstate's
    [`sites`](Microstate::sites) (in system coordinates) and assigns unique tags
    to the sites similarly.

    `add_body` wraps the body's position (and the positions of its sites in
    system coordinates) into the boundary (see `Boundary::wrap()`).

    # Cost

    The cost of adding a body is proportional to the number of sites in the
    body.

    # Returns

    [`Ok(tag)`] with the tag of the added body on success.

    # Errors

    [`Error::CannotWrapPosition`] when either the body position or one of the site
    positions cannot be wrapped into the boundary.

    # Example

    ```
    use hoomd_microstate::{Microstate, Body};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = Microstate::new();
    let first_tag = microstate.add_body(Body::point(Cartesian::from([1.0, 0.0])))?;
    let second_tag = microstate.add_body(Body::point(Cartesian::from([-1.0, 2.0])))?;

    assert_eq!(microstate.bodies().len(), 2);
    assert_eq!(first_tag, 0);
    assert_eq!(second_tag, 1);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    pub fn add_body<V>(&mut self, body: Body<B, S>) -> Result<usize, Error>
    where
        B: Transform<S> + Position<V>,
        S: Position<V>,
        C: Boundary<V>,
    {
        let mut body = body;
        body.properties = self.boundary.wrap(body.properties)?;

        // An unknown site in the body might not wrap into the boundary.
        // Check that they do first before starting to modify internal data
        // structures. This wraps every site twice on add. Should that prove to
        // be a performance bottleneck, we could alternately implement rollback
        // (complicated) or a staging Vec (would require additional allocations
        // or a reusable scratch storage).
        for s in &body.sites {
            self.boundary.wrap(body.properties.transform(s))?;
        }

        // Find body tag before adding sites
        let body_tag = match self.free_body_tags.pop() {
            None => self.body_indices.len(),
            Some(t) => t.0,
        };

        // Add sites.
        // Should the Vec allocation prove a bottleneck, we could recycle the body_sites
        // vecs along with the tags.
        let mut body_sites = Vec::with_capacity(body.sites.len());
        for s in &body.sites {
            let site_tag = match self.free_site_tags.pop() {
                None => self.site_indices.len(),
                Some(t) => t.0,
            };
            self.sites.push(Site {
                site_tag,
                properties: self
                    .boundary
                    .wrap(body.properties.transform(s))
                    .expect("can wrap site"),
                body_tag,
            });

            let index = self.sites.len() - 1;

            if site_tag == self.site_indices.len() {
                self.site_indices.push(Some(index));
            } else {
                debug_assert_eq!(self.site_indices[site_tag], None);
                self.site_indices[site_tag] = Some(index);
            }

            body_sites.push(site_tag);
        }

        // Add body
        self.bodies.push(Tagged {
            tag: body_tag,
            item: body,
        });
        self.bodies_sites.push(body_sites);

        let index = Some(self.bodies.len() - 1);

        if body_tag == self.body_indices.len() {
            self.body_indices.push(index);
        } else {
            debug_assert_eq!(self.body_indices[body_tag], None);
            self.body_indices[body_tag] = index;
        }

        Ok(body_tag)
    }

    /** Add multiple bodies to the microstate.

    See [`Microstate::add_body()`] for details.

    # Errors

    [`Error::CannotWrapPosition`] when any of the body positions or one of the
    site positions cannot be wrapped into the boundary. `try_extend_bodies` adds
    each body one by one. When an error occurs, it short-circuits and does not
    attempt to add any further bodies. The bodies added before the error will
    remain in the microstate.

    # Example

    ```
    use hoomd_microstate::{Microstate, Body};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = Microstate::new();
    microstate.try_extend_bodies([Body::point(Cartesian::from([1.0, 0.0])),
                                  Body::point(Cartesian::from([-1.0, 2.0]))])?;

    assert_eq!(microstate.bodies().len(), 2);
    # Ok(())
    # }
    ```
    */
    #[inline]
    pub fn try_extend_bodies<T, V>(&mut self, bodies: T) -> Result<(), Error>
    where
        T: IntoIterator<Item = Body<B, S>>,
        B: Transform<S> + Position<V>,
        S: Position<V>,
        C: Boundary<V>,
    {
        for body in bodies {
            self.add_body(body)?;
        }

        Ok(())
    }

    /** Remove a body at the given *index* from the microstate.

    Also remove all the body's sites. The body's tag (and the tags of its
    sites) are then free to be reused by [`Microstate::add_body`].

    Removing a body will change the index order of the
    [`bodies`](Microstate::bodies) and [`sites`](Microstate::sites) arrays.
    [`Microstate`] does not guarantee any specific ordering in these arrays.

    # Cost

    The cost of removing a body is proportional to the number of sites in the
    body.

    # Panics

    Panics when `index` is out of bounds.

    # Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([1.0, 0.0])),
                 Body::point(Cartesian::from([-1.0, 2.0]))])
        .try_build()?;

    microstate.remove_body(0);

    assert_eq!(microstate.bodies().len(), 1);
    # Ok(())
    # }
    ```
    */
    #[inline]
    pub fn remove_body(&mut self, body_index: usize) {
        let body_tag = self.bodies[body_index].tag;
        debug_assert_eq!(self.body_indices[body_tag], Some(body_index));

        // Remove sites. `add_body` adds sites in increasing index order, so
        // remove them in reverse order to avoid keep the other bodies' sites
        // in increasing order.
        let body_sites = self.bodies_sites.swap_remove(body_index);
        for site_tag in body_sites.iter().rev() {
            let site_index = self.site_indices[*site_tag].expect("A valid site.");
            let removed_site = self.sites.swap_remove(site_index);
            if site_index < self.sites.len() {
                self.site_indices[self.sites[site_index].site_tag] = Some(site_index);
            }
            self.site_indices[removed_site.site_tag] = None;
            self.free_site_tags.push(Reverse(removed_site.site_tag));
        }

        // Remove body
        self.bodies.swap_remove(body_index);
        if body_index < self.bodies.len() {
            self.body_indices[self.bodies[body_index].tag] = Some(body_index);
        }
        self.body_indices[body_tag] = None;
        self.free_body_tags.push(Reverse(body_tag));
    }

    /** Sets the properties of the given body.

    Also updates the properties of the sites (in the system frame) associated
    with the body.

    # Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
    use hoomd_microstate::property::Point;
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([1.0, 0.0]))])
        .try_build()?;

    microstate.update_body_properties(0, Point::new(Cartesian::from([-2.0, 3.0])));
    assert_eq!(microstate.bodies()[0].item.properties.position, [-2.0, 3.0].into());
    assert_eq!(microstate.sites()[0].properties.position, [-2.0, 3.0].into());
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    pub fn update_body_properties(&mut self, body_index: usize, properties: B)
    where
        B: Transform<S>,
    {
        let body = &mut self.bodies[body_index].item;
        body.properties = properties;

        // Update site properties
        for (i, site_tag) in self.bodies_sites[body_index].iter().enumerate() {
            let site_index = self.site_indices[*site_tag].expect("site_tag should be a valid tag");
            self.sites[site_index].properties = body.properties.transform(&body.sites[i]);
        }
    }
}

/** Access contents of the microstate.
*/
impl<B, S, C> Microstate<B, S, C> {
    /** Access the microstate's tagged bodies in index order.

    [`Microstate`] stores bodies in a flat memory region. The [`Tagged`] type
    holds the unique identifier for each body in [`Tagged::tag`] and the
    [`Body`] itself in [`Tagged::item`].

    [`bodies`](Microstate::bodies) provides direct immutable access
    to this slice. To mutate a body (and by extension, its sites), see
    [`Microstate::update_body_properties()`].

    # Examples

    Identify the tag of a body at a given index:

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([1.0, 0.0])),
                 Body::point(Cartesian::from([-1.0, 2.0]))])
        .try_build()?;

    // The initial index order is equivalent to the tag order.
    assert_eq!(microstate.bodies()[0].tag, 0);
    assert_eq!(microstate.bodies()[1].tag, 1);
    # Ok(())
    # }
    ```

    Compute system-wide properties that are order-independent:
    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
    use hoomd_vector::{Vector, Cartesian};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([1.0, 0.0])),
                 Body::point(Cartesian::from([-1.0, 2.0]))])
        .try_build()?;

    let average_position = microstate.bodies()
        .iter()
        .map(|tagged_body| tagged_body.item.properties.position)
        .sum::<Cartesian<2>>() / (microstate.bodies().len() as f64);
    # Ok(())
    # }
    ```
    */
    #[inline]
    pub fn bodies(&self) -> &[Tagged<Body<B, S>>] {
        &self.bodies
    }

    /** Identify the index of a body given a tag.

    Use [`body_indices`](Microstate::body_indices) to locate a specific body in
    [`Microstate::bodies`].

    `body_indices()[tag]` is:
    * `None` when there is no body with the given tag in the microstate.
    * `Some(index)` when the body with the given tag is in the microstate.
      `index` is the index of the body in [`Microstate::bodies`].

    # Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([1.0, 2.0])),
                 Body::point(Cartesian::from([3.0, 4.0])),
                 Body::point(Cartesian::from([5.0, 6.0])),
                 Body::point(Cartesian::from([7.0, 8.0]))])
        .try_build()?;

    microstate.remove_body(microstate.body_indices()[0].expect("valid tag"));

    assert_eq!(microstate.body_indices()[0], None);
    assert!(matches!(microstate.body_indices()[3], Some(_)));

    if let Some(index) = microstate.body_indices()[2] {
        assert_eq!(microstate.bodies()[index].item.properties.position, [5.0, 6.0].into());
    }
    # Ok(())
    # }
    ```
    */
    #[inline]
    pub fn body_indices(&self) -> &[Option<usize>] {
        &self.body_indices
    }

    /** Access the microstate's sites (in the system frame) in index order.

    [`Microstate`] stores sites twice. Each body in
    [`bodies`](Microstate::bodies) stores its sites in the body frame of
    reference. [`Microstate`] also stores a flat vector of sites that have been
    transformed (see [`Transform`]) to the system reference frame. The [`Site`]
    type holds the unique identifier for each site in [`Site::site_tag`],
    the associated body tag in [`Site::body_tag`] and the site's properties in
    [`Site::properties`].

    [`sites`](Microstate::sites) provides direct immutable access to
    this slice. To mutate a body (and by extension, its sites), see
    [`Microstate::update_body_properties()`].

    # Examples

    Identify the site and body tags of a site at a given index:

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([1.0, 0.0])),
                 Body::point(Cartesian::from([-1.0, 2.0]))])
        .try_build()?;

    // The initial index order is equivalent to the tag order.
    assert_eq!(microstate.sites()[0].site_tag, 0);
    assert_eq!(microstate.sites()[0].body_tag, 0);

    assert_eq!(microstate.sites()[1].body_tag, 1);
    assert_eq!(microstate.sites()[1].body_tag, 1);
    # Ok(())
    # }
    ```

    Compute system-wide properties that are order-independent:
    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
    use hoomd_vector::{Vector, Cartesian};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([1.0, 0.0])),
                 Body::point(Cartesian::from([-1.0, 2.0]))])
        .try_build()?;

    let average_position = microstate.sites()
        .iter()
        .map(|site| site.properties.position)
        .sum::<Cartesian<2>>() / (microstate.sites().len() as f64);
    # Ok(())
    # }
    ```
    */
    #[inline]
    pub fn sites(&self) -> &[Site<S>] {
        &self.sites
    }

    /** Identify the index of a site given a tag.

    Use [`site_indices`](Microstate::site_indices) to locate a specific site in
    [`Microstate::sites`].

    See [`body_indices`](Microstate::body_indices) for details.
    */
    #[inline]
    pub fn site_indices(&self) -> &[Option<usize>] {
        &self.site_indices
    }

    /** Iterate over all the sites (in the system reference frame) associated with a body.

    Use [`iter_body_sites`](Microstate::iter_body_sites) to perform computations
    in the system reference frame on all sites that are associated with a given
    body *index*. The borrowed sites are immutable. Call
    [`Microstate::update_body_properties()`] to mutate a body.

    # Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
    use hoomd_vector::{Vector, Cartesian};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([1.0, 0.0])),
                 Body::point(Cartesian::from([-1.0, 2.0]))])
        .try_build()?;

    let average_position = microstate.iter_body_sites(0)
        .map(|site| site.properties.position)
        .sum::<Cartesian<2>>() / (microstate.bodies()[0].item.sites.len() as f64);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    pub fn iter_body_sites(&self, body_index: usize) -> impl Iterator<Item = &Site<S>> {
        self.bodies_sites[body_index]
            .iter()
            .map(|site_tag| &self.sites[self.site_indices[*site_tag].expect("valid site tag")])
    }
}

/** Choose parameters when constructing a [`Microstate`].

Use a [`MicrostateBuilder`] to choose the values of optional parameters when
constructing a [`Microstate`]. Some parameters, such as `seed` and `step`,
cannot be directly modified after building the [`Microstate`].

# Example

```
use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = MicrostateBuilder::new()
    .step(100_000)
    .seed(0x1234abcd)
    .bodies([Body::point(Cartesian::from([1.0, 0.0])),
             Body::point(Cartesian::from([-1.0, 2.0]))])
    .try_build()?;

assert_eq!(microstate.step(), 100_000);
assert_eq!(microstate.seed(), 0x1234abcd);
assert_eq!(microstate.bodies().len(), 2);
# Ok(())
# }
```
*/
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
    /** Construct an empty [`MicrostateBuilder`] with open boundary conditions.

    The resulting microstate starts at step 0 and has a random seed of 0.

    # Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, property::Point};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::<Point<Cartesian<2>>>::new().try_build()?;

    assert_eq!(microstate.step(), 0);
    assert_eq!(microstate.seed(), 0);
    assert_eq!(microstate.bodies().len(), 0);
    assert_eq!(*microstate.boundary(), hoomd_microstate::boundary::Open);
    # Ok(())
    # }
    ```
    */
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
    /** Construct an empty [`MicrostateBuilder`] with the given boundary conditions.

    The resulting microstate starts at step 0 and has a random seed of 0.

    # Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, property::Point};
    use hoomd_microstate::boundary::Open;
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::<Point<Cartesian<2>>>::with_boundary(Open).try_build()?;

    assert_eq!(microstate.step(), 0);
    assert_eq!(microstate.seed(), 0);
    assert_eq!(microstate.bodies().len(), 0);
    assert_eq!(*microstate.boundary(), hoomd_microstate::boundary::Open);
    # Ok(())
    # }
    ```

    TODO: Show non-trivial boundary conditions.
    */
    #[inline]
    pub fn with_boundary(boundary: C) -> Self {
        Self {
            step: 0,
            seed: 0,
            bodies: Vec::new(),
            boundary,
        }
    }

    /** Choose the initial step in the resulting [`Microstate`].

    The default `step` is 0.

    # Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, property::Point};
    use hoomd_microstate::boundary::Open;
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::<Point<Cartesian<2>>>::new()
        .step(100_000)
        .try_build()?;

    assert_eq!(microstate.step(), 100_000);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn step(mut self, step: u64) -> Self {
        self.step = step;
        self
    }

    /** Choose the random number seed in the resulting [`Microstate`].

    The default `seed` is 0.

    # Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, property::Point};
    use hoomd_microstate::boundary::Open;
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::<Point<Cartesian<2>>>::new()
        .seed(0x1234abcd)
        .try_build()?;

    assert_eq!(microstate.seed(), 0x1234abcd);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /** Add bodies to the resulting [`Microstate`].

    All bodies will be appended when this method is called multiple times.

    # Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([1.0, 0.0])),
                 Body::point(Cartesian::from([-1.0, 2.0]))])
        .try_build()?;

    assert_eq!(microstate.bodies().len(), 2);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn bodies<T>(mut self, bodies: T) -> Self
    where
        T: IntoIterator<Item = Body<B, S>>,
    {
        self.bodies.extend(bodies);
        self
    }

    /** Construct a [`Microstate`] with the chosen options.

    # Errors

    See [`Microstate::try_extend_bodies()`].

    # Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = MicrostateBuilder::new()
        .step(100_000)
        .seed(0x1234abcd)
        .bodies([Body::point(Cartesian::from([1.0, 0.0])),
                 Body::point(Cartesian::from([-1.0, 2.0]))])
        .try_build()?;

    assert_eq!(microstate.step(), 100_000);
    assert_eq!(microstate.seed(), 0x1234abcd);
    assert_eq!(microstate.bodies().len(), 2);
    # Ok(())
    # }
    ```
    */
    #[inline]
    pub fn try_build<V>(self) -> Result<Microstate<B, S, C>, Error>
    where
        B: Transform<S> + Position<V>,
        S: Position<V>,
        C: Boundary<V>,
    {
        let mut microstate = Microstate {
            step: self.step,
            substep: 0,
            seed: self.seed,
            boundary: self.boundary,
            bodies: Vec::new(),
            body_indices: Vec::new(),
            free_body_tags: BinaryHeap::new(),
            sites: Vec::new(),
            site_indices: Vec::new(),
            free_site_tags: BinaryHeap::new(),
            bodies_sites: Vec::new(),
        };

        microstate.try_extend_bodies(self.bodies)?;

        Ok(microstate)
    }
}

// This might be useful in future tests. I'm not sure if it would be interesting for users...

// impl<B, S, C> PartialEq<Microstate<B, S, C>> for Microstate<B, S, C>
// where
//     B: PartialEq,
//     S: PartialEq,
//     C: PartialEq,
// {
//     #[inline]
//     fn eq(&self, other: &Microstate<B, S, C>) -> bool {
//         // `PartialEq` cannot be derived for Microstate, so implement it manually.
//         //
//         // Not all fields matter for equality. Check only those that do.
//         self.step == other.step
//             && self.substep == other.substep
//             && self.seed == other.seed
//             && self.bodies == other.bodies
//             && self.body_indices == other.body_indices
//             && self.sites == other.sites
//             && self.site_indices == other.site_indices
//             && self.boundary == other.boundary
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::Point;
    use hoomd_vector::Cartesian;

    use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
    use rstest::*;
    use std::collections::HashMap;

    // The doc tests above cover all the trivial cases for every method which
    // are not repeated here. The following tests perform self-consistency
    // checks on the internal data structures after calling many methods randomly.

    const N_STEPS: usize = 1024;
    const MAX_BODY_SIZE: usize = 20;
    const MAX_INITIAL_BODY_COORDINATE: f64 = 10.0;
    const MAX_SITE_COORDINATE: f64 = 5.0;
    const MAX_BODY_TRANSLATE: f64 = 0.125;

    fn create_body<R: Rng>(rng: &mut R) -> Body<Point<Cartesian<2>>> {
        let mut body = Body::point(rng.random::<Cartesian<2>>() * MAX_INITIAL_BODY_COORDINATE);

        let n = rng.random_range(1..MAX_BODY_SIZE);
        body.sites = (0..n)
            .map(|_| Point::new(rng.random::<Cartesian<2>>() * MAX_SITE_COORDINATE))
            .collect();

        body
    }

    #[rstest]
    fn consistency_open(#[values(1, 2, 3, 4)] seed: u64) {
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
                let tag = microstate.add_body(body.clone()).expect("valid body");
                reference_bodies.insert(tag, body);
            } else if move_type_r > 0.5 && !microstate.bodies.is_empty() {
                let index = rng.random_range(..microstate.bodies.len());
                let tag = microstate.bodies()[index].tag;
                microstate.remove_body(index);
                reference_bodies.remove(&tag);
            } else if !microstate.bodies.is_empty() {
                let index = rng.random_range(..microstate.bodies.len());
                let tag = microstate.bodies()[index].tag;
                let body = reference_bodies.get_mut(&tag).expect("valid body tag");

                body.properties.position += rng.random::<Cartesian<2>>() * MAX_BODY_TRANSLATE;
                microstate.update_body_properties(index, body.properties);
            }
        }

        assert_eq!(microstate.bodies.len(), reference_bodies.len());
        assert_eq!(
            microstate.sites.len(),
            reference_bodies.values().map(|body| body.sites.len()).sum()
        );

        for (tag, optional_index) in microstate.body_indices.iter().enumerate() {
            if let Some(index) = optional_index {
                assert_eq!(microstate.bodies()[*index].tag, tag);
                assert!(reference_bodies.contains_key(&tag));
            } else {
                assert!(!reference_bodies.contains_key(&tag));
            }
        }

        for (tag, body) in &reference_bodies {
            let body_index = microstate.body_indices()[*tag].expect("valid index");
            assert_eq!(microstate.bodies()[body_index].item, *body);
        }

        for (tag, optional_index) in microstate.site_indices.iter().enumerate() {
            if let Some(index) = optional_index {
                assert_eq!(microstate.sites()[*index].site_tag, tag);
            }
        }

        for site in microstate.sites() {
            let body_index = microstate.body_indices()[site.body_tag].expect("valid body");
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
                let site_index = microstate.site_indices()[*site_tag].expect("valid index");
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
            microstate.add_body(body).expect("valid body");
        }

        let mut removal_order = (0..N_STEPS).collect::<Vec<_>>();
        removal_order.shuffle(&mut rng);

        for body_tag in removal_order {
            let body_index = microstate.body_indices()[body_tag].expect("valid tag");
            microstate.remove_body(body_index);
        }

        assert!(microstate.bodies().is_empty());
        assert!(microstate.bodies_sites.is_empty());
        assert!(microstate.sites().is_empty());
    }

    // TODO: Test add_bodies and update_body_properties with boundaries that result in errors.
    // TODO: Test add_bodies and update_body_properties with periodic boundaries that result in wrapping.
}

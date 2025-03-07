// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Microstate`] and related types.
 */

use std::collections::BinaryHeap;
use std::cmp::Reverse;

use hoomd_vector::Vector;

use crate::{Body, Site, Transform, boundary::Open};

/** Track a unique identifier for an item in [`Microstate`].
        microstate.extend_bodies(self.bodies);
    */
#[derive(Clone, Debug, PartialEq)]
pub struct Tagged<T> {
    /// The unique identifier.
    pub tag: usize,
    /// The tagged item.
    pub item: T,
}

/** Store and manage all the degrees of freedom of a single microstate in phase space.

TODO: document this

TODO: After planning the RNG seed layout, consider reducing the width of step,
substep, and seed. For example, we could combine step and substep into 1 u64 if
there aren't enough seed bits.

TODO: Process boundary conditions
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

    // The range of allowed particle positions and a description of any periodicity.
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
    }}

/// Access and manage the simulation step, substep, and RNG seeds.
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

    let microstate = MicrostateBuilder::<Point<Cartesian<2>>>::new().step(100_000).build();
    assert_eq!(microstate.step(), 100_000);
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

    let microstate = MicrostateBuilder::<Point<Cartesian<2>>>::new().seed(0x1234abcd).build();
    assert_eq!(microstate.seed(), 0x1234abcd);
    ```
    */
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u32 {
        self.seed
    }
}

/* What we want to do is:

```
impl<P, B, V> Microstate<P, B> where
P: Particle<V>,
V: Vector
```

This is not possible due to compile error E0207.
* One workaround involves placing the bounds on the struct and using PhantomData.
* The other involves grouping the methods that require the use of V in a trait.

https://stackoverflow.com/questions/55519710/how-to-make-a-generic-of-generic-with-trait?noredirect=1&lq=1
https://stackoverflow.com/questions/50671177/specify-fn-trait-bound-on-struct-definition-without-fixing-one-of-the-fn-par?noredirect=1&lq=1

The PhantomData solution feels like more of a hack, so this code implements the trait solution.
*/

// /** Methods that operate on particles in a [`Microstate`].

// See [`Microstate`] for more information.
// */
// pub trait Particles<P, V> {
//     /// Add a new particle to the microstate.
//     fn add_particle(&mut self, particle: P);

//     // fn extend_particles( // TODO

//     /// Remove a particle at the given index from the microstate.
//     fn remove_particle(&mut self, index: usize);

//     /// Access all particles in the microstate.
//     fn particles(&self) -> &[P];

//     /// Update a single particle at the given index in the microstate.
//     fn update_particle(&mut self, index: usize, particle: P);

//     // TODO: how to efficiently update all particles? We could provide a method that calls
//     // a Fn with &mut [P], but then we have to assume that the caller may have reordered the
//     // particles. In MD, a full system update without reordering is common and would not
//     // require rebuilding the neighbor list.
// }

/** Manage bodies in the microstate.
*/
impl<B, S, C> Microstate<B, S, C>
{
    /** Add a new body to the microstate.

    Each body is assigned a unique tag. The first body is given tag 0,
    the second is given tag 1, and so on. When a body is removed (see
    [`remove_body`](Microstate::remove_body)), its tag becomes unused. The next
    call to [`add_body`](Microstate::add_body) will assign the smallest unused
    tag.

    [`add_body`] also adds the body's sites to the microstate's
    [`sites`](Microstate::sites) (in system coordinates) and assigns unique
    tags to the sites similarly.

    # Cost

    The cost of adding a body is proportional to the number of sites in the
    body.

    # Returns

    The tag of the added body.

    # Example

    ```
    use hoomd_microstate::{Microstate, Body};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::new();
    let first_tag = microstate.add_body(Body::point(Cartesian::from([1.0, 0.0])));
    let second_tag = microstate.add_body(Body::point(Cartesian::from([-1.0, 2.0])));

    assert_eq!(microstate.bodies().len(), 2);
    assert_eq!(first_tag, 0);
    assert_eq!(second_tag, 1);
    ```
    */
    #[inline]
    pub fn add_body(&mut self, body: Body<B, S>) -> usize where
B: Transform<S>{
        // Find body tag before adding sites
        let body_tag = match self.free_body_tags.pop() {
            None => self.body_indices.len(),
            Some(t) => t.0,
        };
    
        // Add sites
        let mut body_sites = Vec::with_capacity(body.sites.len());
        for s in &body.sites {
            let site_tag = match self.free_site_tags.pop() {
                None => self.site_indices.len(),
                Some(t) => t.0,
            };
            self.sites.push(Site { site_tag, properties: body.properties.transform(s), body_tag });

            let index = self.sites.len()-1;

            if site_tag == self.site_indices.len() {
                self.site_indices.push(Some(index));
            } else {
                debug_assert_eq!(self.site_indices[site_tag], None);
                self.site_indices[site_tag] = Some(index);
            }

            body_sites.push(site_tag);
        }

        // Add body
        self.bodies.push(Tagged { tag: body_tag, item: body });
        self.bodies_sites.push(body_sites);

        let index = Some(self.bodies.len()-1);

        if body_tag == self.body_indices.len() {
            self.body_indices.push(index);
        } else {
            debug_assert_eq!(self.body_indices[body_tag], None);
            self.body_indices[body_tag] = index;
        }

        body_tag
    }

    /** Add multiple bodies to the microstate.

    See [`add_body`](Microstate::add_body) for details.

    # Example

    ```
    use hoomd_microstate::{Microstate, Body};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::new();
    microstate.extend_bodies([Body::point(Cartesian::from([1.0, 0.0])),
                              Body::point(Cartesian::from([-1.0, 2.0]))]);

    assert_eq!(microstate.bodies().len(), 2);
    ```
    */
    #[inline]
    pub fn extend_bodies<T>(&mut self, bodies: T) where
    T: IntoIterator<Item = Body<B, S>>,
    B: Transform<S>{
        for body in bodies {
            self.add_body(body);
        }
    }

    /** Remove a body at the given *index* from the microstate.

    The body's tag (and the tags of its sites) are then free to be reused by
    [`add_body`](Microstate::add_body).

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

    let mut microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([1.0, 0.0])),
                 Body::point(Cartesian::from([-1.0, 2.0]))])
        .build();

    microstate.remove_body(0);

    assert_eq!(microstate.bodies().len(), 1);
    ```
    */
    #[inline]
    pub fn remove_body(&mut self, body_index: usize) {
        let body_tag = self.bodies[body_index].tag;
        debug_assert_eq!(self.body_indices[body_tag], Some(body_index));

        // Remove sites
        let body_sites = self.bodies_sites.swap_remove(body_index);
        for site_tag in body_sites {
            let site_index = self.site_indices[site_tag].expect("A valid site.");
            let removed_site = self.sites.swap_remove(site_index);
            self.site_indices[self.sites[site_index].site_tag] = Some(site_index);
            self.site_indices[removed_site.site_tag] = None;
            self.free_site_tags.push(Reverse(removed_site.site_tag));
        }

        // Remove body
        self.bodies.swap_remove(body_index);
        self.body_indices[self.bodies[body_index].tag] = Some(body_index);
        self.body_indices[body_tag] = None;
        self.free_body_tags.push(Reverse(body_tag));

    }

    #[inline]
    pub fn update_body_properties(&mut self, index: usize, properties: B) where
B: Transform<S>{
        self.bodies[index].item.properties = properties;

        // TODO: Update site properties
    }
}

/** Access contents of the microstate.
*/
impl<B, S, C> Microstate<B, S, C> {

    /** Access the microstate's tagged bodies in index order.

    # Examples

    Identify the tag of a body at a given index:

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
    use hoomd_vector::Cartesian;

    let microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([1.0, 0.0])),
                 Body::point(Cartesian::from([-1.0, 2.0]))])
        .build();

    assert_eq!(microstate.bodies()[0].tag, 0);
    assert_eq!(microstate.bodies()[1].tag, 1);
    ```

    Compute system-wide properties that are order-independent:
    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body};
    use hoomd_vector::{Vector, Cartesian};

    let microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([1.0, 0.0])),
                 Body::point(Cartesian::from([-1.0, 2.0]))])
        .build();

    let average_position = microstate.bodies()
        .iter()
        .map(|tagged_body| tagged_body.item.properties.position)
        .sum::<Cartesian<2>>() / (microstate.bodies().len() as f64);
    ```
    */
    #[inline]
    pub fn bodies(&self) -> &[Tagged<Body<B,S>>] {
        &self.bodies
    }

    #[inline]
    pub fn body_indices(&self) -> &[Option<usize>] {
        &self.body_indices
    }

    #[inline]
    pub fn sites(&self) -> &[Site<S>] {
        &self.sites
    }
}

/** Choose parameters when constructing a [`Microstate`].

By default, a [`Microstate`] 

# Examples

TODO
*/
pub struct MicrostateBuilder<B, S=B, C=Open> {
    step: u64,
    seed: u32,
    bodies: Vec<Body<B, S>>,
    boundary: C,
}

impl<B, S> MicrostateBuilder<B, S, Open> {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        MicrostateBuilder::with_boundary(Open)
    }
}

impl<B, S, C> MicrostateBuilder<B, S, C> {
    pub fn with_boundary(boundary: C) -> Self {
        Self { step: 0, seed: 0, bodies: Vec::new(), boundary }
    }

    pub fn step(mut self, step: u64) -> Self {
        self.step = step;
        self
    }

    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn bodies<T>(mut self, bodies: T) -> Self where
    T: IntoIterator<Item = Body<B, S>>,
    {
        self.bodies.extend(bodies);
        self
    }

    pub fn build(self) -> Microstate<B, S, C> where
    B: Transform<S> {
        let mut microstate = Microstate { step: self.step, substep: 0, seed: self.seed,
            boundary: self.boundary,
            bodies: Vec::new(),
            body_indices: Vec::new(),
            free_body_tags: BinaryHeap::new(),
            sites: Vec::new(),
            site_indices: Vec::new(),
            free_site_tags: BinaryHeap::new(),
            bodies_sites: Vec::new(),
        };

        microstate.extend_bodies(self.bodies);
        
        microstate
    }
}

// TODO: Tests

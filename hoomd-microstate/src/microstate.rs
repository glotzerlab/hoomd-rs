// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Microstate`] and related types.
 */

use std::collections::BinaryHeap;
use std::cmp::Reverse;

use hoomd_vector::Vector;

use crate::{Body, Site, Transform};

/** Track a unique identifier for items in a [`Microstate`]
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

TODO: Add boundary conditions
*/
#[derive(Clone)]
pub struct Microstate<B, S, /*, C*/> {
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

    /// Indices of the sites associated with the bodies (in index order).
    bodies_sites: Vec<Vec<usize>>, 

    // The range of allowed particle positions and a description of any periodicity.
    // boundary: C,
}

impl<B, S /*, C*/> Default for Microstate<B, S /*, C*/> {
    /** Create an empty microstate.

    The default microstate starts at step 0, substep 0, random number seed 0,
    and has no bodies.

    # Example

    ```
    use hoomd_microstate::{Microstate, Particles, particle::Point};
    use hoomd_vector::Cartesian;

    let microstate = Microstate::<Point<Cartesian<2>>>::default();
    assert_eq!(microstate.step(), 0);
    assert_eq!(microstate.substep(), 0);
    assert_eq!(microstate.seed(), 0);
    assert_eq!(microstate.particles().len(), 0);
    ```
    */
    #[inline]
    #[must_use]
    fn default() -> Self {
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
        }
    }
}

/// Access and manage the simulation step, substep, and RNG seeds.
impl<B, S /*, C*/> Microstate<B, S /*, C*/> {
    /** Get the simulation step.

    # Example
    ```
    use hoomd_microstate::{Microstate, particle::Point};
    use hoomd_vector::Cartesian;

    let microstate = Microstate::<Point<Cartesian<2>>>::default();
    assert_eq!(microstate.step(), 0);
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
    ```
    use hoomd_microstate::{Microstate, particle::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::<Point<Cartesian<2>>>::default();
    microstate.increment_step();

    assert_eq!(microstate.step(), 1);
    ```

    ```
    use hoomd_microstate::{Microstate, particle::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::<Point<Cartesian<2>>>::default();

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
    use hoomd_microstate::{Microstate, particle::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::<Point<Cartesian<2>>>::default();
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
    use hoomd_microstate::{Microstate, particle::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::<Point<Cartesian<2>>>::default();
    microstate.increment_substep();

    assert_eq!(microstate.substep(), 1);
    ```
    */
    #[inline]
    pub fn increment_substep(&mut self) {
        self.substep += 1;
    }

    /** Get the simulation seed.

    ```
    use hoomd_microstate::{Microstate, particle::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::<Point<Cartesian<2>>>::default();

    assert_eq!(microstate.seed(), 0);
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

impl<B, S, /* C */> Microstate<B, S, /* C*/>
{
    /** Add a new body to the microstate.

    # Example

    ```
    use hoomd_microstate::{Microstate, Particles, particle::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::default();
    microstate.add_particle(Point::<Cartesian<2>>::default());

    assert_eq!(microstate.particles().len(), 1);
    ```
    */
    #[inline]
    pub fn add_body(&mut self, body: Body<B, S>) {
        let tag = match self.free_body_tags.pop() {
            None => self.body_indices.len(),
            Some(t) => t.0,
        };
        self.bodies.push(Tagged { tag, item: body });

        let index = Some(self.bodies.len()-1);

        if tag == self.body_indices.len() {
            self.body_indices.push(index);
        } else {
            debug_assert_eq!(self.body_indices[tag], None);
            self.body_indices[tag] = index;
        }

        // TODO: add sites
    }

    // fn extend_bodies( // TODO

    /** Remove a body at the given index from the microstate.

    # Example

    ```
    use hoomd_microstate::{Microstate, Particles, particle::Point};
    use hoomd_vector::Cartesian;

    let mut microstate = Microstate::default();
    microstate.add_particle(Point::<Cartesian<2>>::default());
    microstate.remove_particle(0);

    assert_eq!(microstate.particles().len(), 0);
    ```

    # Panics

    Panics when `index` is out of bounds.
    */
    #[inline]
    pub fn remove_body(&mut self, index: usize) {
        let tag = self.bodies[index].tag;
        debug_assert_eq!(self.body_indices[tag], Some(index));

        self.bodies.swap_remove(index);
        self.body_indices[tag] = None;
        self.free_body_tags.push(Reverse(tag));

        // TODO: Remove sites
    }

    #[inline]
    pub fn update_body_properties(&mut self, index: usize, properties: B) {
        self.bodies[index].item.properties = properties;

        // TODO: Update site properties
    }

    #[inline]
    #[must_use]
    pub fn bodies(&self) -> &[Tagged<Body<B,S>>] {
        &self.bodies
    }

    #[inline]
    #[must_use]
    pub fn sites(&self) -> &[Site<S>] {
        &self.sites
    }
}

// TODO: Implement builder to initialize a microstate with parameters.

// TODO: Tests

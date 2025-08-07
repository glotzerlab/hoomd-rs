// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `UniformIn`
*/

use hoomd_microstate::{Body, property::Point};
use hoomd_vector::Cartesian;

use rand::{Rng, distr::Distribution};

/** Generate bodies uniformly in the given boundary condition.

Use [`UniformIn`] to randomly generate bodies inside the simulation boundary.

# Example

TODO: Write example. 

*/
pub struct UniformIn<S, C> {
    /// Generate bodies inside this boundary.
    pub boundary: C,

    /// Give each generated body these sites.
    pub template_sites: Vec<S>,
    }

impl<V, S, C> Distribution<Body<Point<V>, S>> for UniformIn<S, C> where
S: Clone,
C: Distribution<V>
 {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Body<Point<V>, S> {
        let properties = Point { position: self.boundary.sample(rng) };
        let sites = self.template_sites.clone();
        Body { properties, sites }
    }
}

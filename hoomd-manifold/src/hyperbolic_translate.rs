// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement HyperbolicTranslate
*/

use hoomd_mc::LocalTrial;
use hoomd_microstate::property::Position;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Vector;
use crate::{Minkowski, Hyperboloid, HyperbolicRotate, FundamentalDomain};

use rand::Rng;
use rand::distr::Distribution;

/** Move the position of a body in hyperbolic space by a small distance 

TODO: documentation, examples
*/
pub struct HyperbolicTranslate {
    // The max distance a body can be translated in one trial move
    pub maximum_distance: PositiveReal
}

impl<V,B> LocalTrial<B> for HyperbolicTranslate 
where
    B: Position<Vector = V>,
    V: Vector,
{
    /** TODO: documentation, examples
    */
    #[inline]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B {
        //temp code
        body_properties
    }
}

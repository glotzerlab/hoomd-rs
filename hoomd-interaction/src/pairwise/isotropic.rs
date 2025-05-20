// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Isotropic
*/

use crate::SitePairEnergy;
use super::IsotropicEnergy;
use hoomd_microstate::{Microstate, property::Position};
use hoomd_vector::Vector;

/** Compute isotropic properties from a pair of sites

[`Isotropic`] is a newtype that provides a single implementation of pairwise
properties. It fills the gap between traits like [`SitePairEnergy`] which
operates on site properties and [`IsotropicEnergy`] which is a function
only of the separation distance.

# Example

TODO
*/
pub struct Isotropic<E>(pub E);

// impl<S, E, V> SitePairEnergy<S> for Isotropic<E> where
// S: Position<V>,
// V: Vector{
    
//     fn site_pair_energy(&self, a: &S, b: &S) -> f64 {
//     0.0
//     }
// }

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//TODO: documentation 

/*! Implement Voronoi tesselations of a given point set 
*/
use hoomd_vector::{Vector, Cartesian, InnerProduct};
use hoomd_microstate::{Microstate, property::Point, boundary::Open};
use hoomd_manifold::{Minkowski, Hyperboloid};
use std::array;
use crate::{meshless_voro::Dimensionality};
use glam::DVec3;

/** Define generator for Hyperbolic space
TODO: documentation
*/

#[derive(Clone, Debug)]
pub struct GeneratorHyperbolic {
    /// Coordinates of point in hyperbolic space in hyperboloid coordinates
    pub loc: Vec<f64>,
    /// skirt width of the hyperboloic
    pub skirt: f64,
    /// site tag of point
    pub site_tag: usize,
}

impl GeneratorHyperbolic {
    pub(super) fn new(id: usize, skirt: f64, loc: Vec<f64>, dimensionality: Dimensionality) -> Self {
        let mut loc = loc;
        match dimensionality {
            Dimensionality::OneD => {
                loc[1] = 0.;
                loc[2] = 0.;
            }
            Dimensionality::TwoD => loc[2] = 0.,
            _ => (),
        }
        Self { loc: loc, skirt: skirt, site_tag: id }
    }
    /** Get the site tag number of this pd generator
    */
    pub fn id(&self) -> usize {
        self.site_tag
    }
    /** Get the position of this pd generator
    */
    pub fn loc(&self) -> DVec3 {
        let mut coords: Vec<f64> = self.loc.clone();
        coords.push(0.0);
        DVec3::from_array([coords[0], coords[1], coords[2]])
    }
    /** get a generator from a microstate site
    */
    pub fn from_microstate() -> GeneratorHyperbolic {
        GeneratorHyperbolic { 
            loc: vec![0.0,0.0,0.0],
            skirt: 1.0 as f64,
            site_tag: 1,
        }
    }
}

/** Define the neighbor list
TODO: documentation
*/
pub struct NeighborList<B> {
    /// ordered, nested vector of 2-tuples with nearest-neighbor pairs
    pub neighbors: Vec<(u32,u32)>
}

impl<B> NeighborList<B> {
    #[inline]
    pub fn nearest_neighbors(point: Microstate<B>) -> NeighborList<B> {
        NeighborList::<B>{neighbors : vec![(1,1)]}
    }
}


// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//TODO: documentation 

/*! Implement Voronoi tesselations of a given point set 
*/
use hoomd_vector::{Vector, Cartesian, InnerProduct};
use hoomd_microstate::{Microstate, property::Point, boundary::Open};
use hoomd_manifold::{Minkowski, Hyperboloid};
use std::array;
use crate::{PowerDiagram, meshless_voro::Dimensionality};
use glam::DVec3;

/** Define generator for Hyperbolic space
TODO: documentation
*/

#[derive(Clone, Copy, Debug)]
pub struct GeneratorHyperbolic {
    /// Coordinates of point in hyperbolic space in hyperboloid coordinates
    pub loc: Vec<f64>,
    /// skirt width of the hyperboloic
    pub skirt: f64,
    /// site tag of point
    pub site_tag: usize,
}

impl GeneratorHyperbolic {
    pub(super) fn new(id: usize, rad: f64, loc: Vec<f64>, dimensionality: Dimensionality) -> Self {
        let mut loc = loc;
        match dimensionality {
            Dimensionality::OneD => {
                loc.y = 0.;
                loc.z = 0.;
            }
            Dimensionality::TwoD => loc.z = 0.,
            _ => (),
        }
        Self { center: loc, radius: rad, site_tag: id }
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

    }
}

/** Define the neighbor list
TODO: documentation
*/
#[derive(Clone, Debug, PartialEq)]
pub struct NeighborList {
    /// ordered, nested vector of 2-tuples with nearest-neighbor pairs
    pub neighbors: Vec<(u32,u32)>
}

impl NeighborList {
    #[inline]
    pub fn nearest_neighbors(power_diagram: PowerDiagramCenters) -> NeighborList {
        NeighborList{neighbors : vec![(1,1)]}
    }
}


// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//TODO: documentation 

/*! Implement Voronoi tesselations of a given point set 
*/
use hoomd_vector::{Vector, Cartesian};
use hoomd_microstate::{Microstate, property::Point, boundary::Open};
use hoomd_manifold::{Minkowski, Hyperboloid};
use std::array;
use crate::{meshless_voronoi::Dimensionality, Voronoi};
use glam::DVec3;

/** Define generator for Hyperbolic space
TODO: documentation
*/

#[derive(Clone, Debug)]
pub struct GeneratorHyperbolic {
    /// Coordinates of point in hyperbolic space in poincare coordinates
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
pub struct NeighborList<B, S,C> {
    /// ordered, nested vector of 2-tuples with nearest-neighbor pairs
    pub neighbors: Vec<(usize,usize)>,
    /// Microstate 
    pub microstate: Microstate<B, S, C>
}

impl<B,S,C> NeighborList<B,S,C> {
    /// Get the neighbor list
    pub fn neighbors(&self) -> &Vec<(usize,usize)> {
        &self.neighbors
    }
}
/** Neighbor list from microstates in cartesian space

#Example

```
use hoomd_microstate::{Microstate, MicrostateBuilder, Body, property::Point};
use hoomd_vector::Cartesian;
use hoomd_order::NeighborList;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let microstate = MicrostateBuilder::new()
    .bodies([Body::point(Cartesian::from([0.5, 0.5])),
             Body::point(Cartesian::from([-1.0, 1.0])),
             Body::point(Cartesian::from([1.0, -1.0])),
             Body::point(Cartesian::from([-0.5, -0.5]))])
    .try_build()?;

let nlist = NeighborList::from_microstate(microstate);
assert_eq!(vec![(0 as usize, 1 as usize),
                (0 as usize, 2 as usize),
                (0 as usize, 3 as usize),
                (1 as usize, 3 as usize),
                (2 as usize, 3 as usize)], 
            *nlist.neighbors());
# Ok(())
# }
```
*/
impl<const N: usize, B> NeighborList<B, Point<Cartesian<N>>,Open> {
    #[inline]
    pub fn from_microstate(microstate: Microstate<B, Point<Cartesian<N>>,Open> ) -> NeighborList<B, Point<Cartesian<N>>,Open> {
        let mut nlist = vec![];
        let mut generators = vec![];
        for site in microstate.sites() {
            let mut pos_vec : Vec<f64> = Vec::from(site.properties.position.coordinates);
            pos_vec.push(0.0);
            let position = DVec3{x:pos_vec[0], y:pos_vec[1], z:pos_vec[2]};
            generators.push(position);
        }
        // TODO: find way of getting anchor and width for open boundary conditions
        let anchor = DVec3::splat(-2.);
        let width = DVec3::splat(4.);
        let _voronoi = Voronoi::build(&generators, anchor, width, N.try_into().unwrap(), false);
        let cells = _voronoi.cells();
        for site_index in microstate.site_indices().iter() {
            if let Some(site_tag) = site_index {
                let mut nn_list = cells[*site_tag].neighbour_ids(&_voronoi);
                let mut temp_list = vec![];
                for n in nn_list {
                    if n > *site_tag {
                        temp_list.push(n)
                    }
                }
                temp_list.sort();
                for n in temp_list {
                    nlist.push((*site_tag, n));
                }
            }
        }
        NeighborList { neighbors: nlist, microstate: microstate}
    }
}
   
/** Neighbor list from microstates in hyperbolic space

#Example

```
use hoomd_microstate::{Microstate, MicrostateBuilder, Body, property::Point};
use hoomd_manifold::Minkowski;
use hoomd_order::NeighborList;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = MicrostateBuilder::with_boundary(Open)
    .bodies([Body::point(Minkowski::from([1.0, -2.0, sqrt(5.0)])),
        Body::point(Minkowski::from([1.0, -1.0, sqrt(3.0)])),
        Body::point(Minkowski::from([-1.0, -2.0, sqrt(5.0)])),
        Body::point(Minkowski::from([-1.0, 1.0, sqrt(3.0)]))])
    .try_build()?;

let nlist = NeighborList::from_microstate(microstate);
assert_eq!(vec![(0 as usize, 1 as usize),
                (0 as usize, 2 as usize),
                (1 as usize, 2 as usize),
                (1 as usize, 3 as usize),
                (2 as usize, 3 as usize)], 
            *nlist.neighbors());
# Ok(())
# }
```
*/
impl<const N: usize> NeighborList<Point<Minkowski<N>>,Point<Minkowski<N>>,Open> {
    #[inline]
    pub fn from_microstate(microstate: Microstate<Point<Minkowski<N>>,Point<Minkowski<N>>,Open> ) -> NeighborList<Point<Minkowski<N>>,Point<Minkowski<N>>,Open> {
        let mut nlist = vec![];
        let mut generators = vec![];
        let rho = microstate.sites()[0].properties.position.get_skirt_width();
        for site in microstate.sites() {
            let mut pos_vec = site.properties.position.to_poincare(rho);
            pos_vec.push(0.0);
            generators.push(pos_vec);
        }
        let anchor = DVec3::splat(-1.);
        let width = DVec3::splat(2.);
        let _voronoi = Voronoi::build_hyperbolic(&generators, rho, anchor, width, (N-1 as usize).try_into().unwrap(), false);
        let cells = _voronoi.cells();
        for site_index in microstate.site_indices().iter() {
            if let Some(site_tag) = site_index {
                let mut nn_list = cells[*site_tag].neighbour_ids(&_voronoi);
                let mut temp_list = vec![];
                for n in nn_list {
                    if n > *site_tag {
                        temp_list.push(n)
                    }
                }
                temp_list.sort();
                for n in temp_list {
                    nlist.push((*site_tag, n));
                }
            }
        }
        NeighborList { neighbors: nlist, microstate: microstate}
    }
}



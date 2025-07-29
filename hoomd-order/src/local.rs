// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//TODO: documentation 

/*! Implement Voronoi tesselations of a given point set 
*/
use hoomd_vector::{Vector, Cartesian};
use hoomd_microstate::{Microstate, property::Position, property::Point, boundary::Open};
use hoomd_manifold::{Minkowski, Hyperboloid};
use std::array;
use crate::{meshless_voronoi::Dimensionality, Voronoi};
use glam::DVec3;

/** Define generator for making voronoi diagrams in Hyperbolic space
*/

#[derive(Clone, Debug)]
pub struct GeneratorHyperbolic {
    /// Coordinates of point in hyperbolic space in Poincare coordinates
    pub loc: Vec<f64>,
    /// skirt width of the hyperboloid (equivalently, the radius of the Poincare disk)
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
    pub fn skirt(&self) -> f64 {
        self.skirt
    }
}

/** Define the neighbor list

The neighborlist for a given microstate is a vector of two-element tuples giving the pair of nearest neighbors. Nearest 
neighbors are found using the voronoi diagram.
*/
pub struct NeighborList<'a, B, S,C> {
    /// ordered, nested vector of 2-tuples with nearest-neighbor pairs
    pub neighbors: Vec<(usize,usize)>,
    /// Microstate 
    pub microstate: &'a Microstate<B, S, C>
}

impl<B,S,C> NeighborList<'_,B,S,C> {
    /// Get the neighbor list
    pub fn neighbors(&self) -> &Vec<(usize,usize)> {
        &self.neighbors
    }
    /** Get the indices of the neighbors for a specific site

    #Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body, property::Point, boundary::Open};
    use hoomd_vector::Cartesian;
    use hoomd_order::NeighborList;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([0.5, 0.25])),
                Body::point(Cartesian::from([-1.0, 1.0])),
                Body::point(Cartesian::from([1.0, -0.75])),
                Body::point(Cartesian::from([-0.5, -0.5]))])
        .try_build()?;

    let nlist = NeighborList::from_microstate(&microstate);
    let nlist_for_0 = nlist.neighbors_of_site(microstate.site_indices()[0]);
    assert_eq!(vec![1 as usize, 2 as usize, 3 as usize], nlist_for_0);
    # Ok(())
    # }
    ```
    */
    pub fn neighbors_of_site(&self, site_index: Option<usize>) -> Vec<usize> {
        match site_index {
            Some(num) => {
                let filtered_nlist: Vec<usize> = self.neighbors.clone().into_iter()
                    .fold(Vec::new(), |mut nlist, c| {
                        if c.1 == num {
                            nlist.push(c.0);
                        } else if c.0 == num {
                            nlist.push(c.1)
                        }
                        nlist
                    });
                filtered_nlist
            }
            None => panic!("given site index is invalid"),
        }
    }
    /** Get the coordination numbers for each site in a microstate

    #Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body, property::Point, boundary::Open};
    use hoomd_vector::Cartesian;
    use hoomd_order::NeighborList;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([0.5, 0.25])),
                Body::point(Cartesian::from([-1.0, 1.0])),
                Body::point(Cartesian::from([1.0, -0.75])),
                Body::point(Cartesian::from([-0.5, -0.5]))])
        .try_build()?;

    let nlist = NeighborList::from_microstate(&microstate);
    let coordination_numbers = nlist.coordination_numbers();
    assert_eq!(vec![3 as usize, 2 as usize, 2 as usize, 3 as usize], coordination_numbers);
    # Ok(())
    # }
    ```
    */
    pub fn coordination_numbers(&self) -> Vec<usize> {
        let mut coord_number = vec![];
        for site_index in self.microstate.site_indices().iter() {
            coord_number.push(self.neighbors_of_site(*site_index).len());
        }
        coord_number
    }
}
/** Neighbor list from microstates in cartesian space

#Example

```
use hoomd_microstate::{Microstate, MicrostateBuilder, Body, property::Point, boundary::Open};
use hoomd_vector::Cartesian;
use hoomd_order::NeighborList;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let microstate = MicrostateBuilder::new()
    .bodies([Body::point(Cartesian::from([0.5, 0.25])),
             Body::point(Cartesian::from([-1.0, 1.0])),
             Body::point(Cartesian::from([1.0, -0.75])),
             Body::point(Cartesian::from([-0.5, -0.5]))])
    .try_build()?;

let nlist = NeighborList::from_microstate(&microstate);
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
impl<const N: usize, B, S> NeighborList<'_,B, S, Open> 
where S: Position<Vector = Cartesian<N>>
{
    #[inline]
    pub fn from_microstate(microstate: &Microstate<B, S,Open> ) -> NeighborList<B, S, Open> {
        let mut nlist = vec![];
        let mut generators = vec![];
        let mut coord_numbers = vec![];
        for site in microstate.sites() {
            let mut pos_vec : Vec<f64> = Vec::from(site.properties.position().coordinates);
            pos_vec.push(0.0);
            let position = DVec3{x:pos_vec[0], y:pos_vec[1], z:pos_vec[2]};
            generators.push(position);
            for n in 0..N {
                coord_numbers.push(site.properties.position()[n].floor() as i32);
            }
        }
        let max_coord = coord_numbers.iter().max();
        let min_coord = coord_numbers.iter().min();
        let mut anchor_num : f64 = 0.0;
        let mut width_num : f64 = 0.0;
        match max_coord {
            Some(max) => anchor_num = (*max) as f64 + 2.0,
            None => panic!("microstate is empty!"),
        }
        match min_coord {
            Some(min) => width_num = (*min as f64) - 1.0,
            None => panic!("microstate is empty!"),
        }
        let anchor = DVec3::splat(-2.0);
        let width = DVec3::splat(4.0);
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
use hoomd_microstate::{Microstate, MicrostateBuilder, Body, property::Point, boundary::Open};
use hoomd_manifold::Minkowski;
use hoomd_order::NeighborList;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = MicrostateBuilder::with_boundary(Open)
    .bodies([Body::point(Minkowski::from([1.0, -2.0, 5.0_f64.sqrt()])),
        Body::point(Minkowski::from([1.0, -1.0, 3.0_f64.sqrt()])),
        Body::point(Minkowski::from([-1.0, -2.0, 5.0_f64.sqrt()])),
        Body::point(Minkowski::from([-1.0, 1.0, 3.0_f64.sqrt()]))])
    .try_build()?;

let nlist = NeighborList::from_hyperbolic_microstate(&microstate);
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
impl<const N: usize, B, S> NeighborList<'_,B,S,Open>
where S: Position<Vector = Minkowski<N>> 
{
    #[inline]
    pub fn from_hyperbolic_microstate(microstate: &Microstate<B, S,Open> ) -> NeighborList<B,S,Open> {
        let mut nlist = vec![];
        let mut generators = vec![];
        let rho = microstate.sites()[0].properties.position().get_skirt_width();
        for site in microstate.sites() {
            let mut pos_vec = site.properties.position().to_poincare(rho);
            pos_vec.push(0.0);
            generators.push(pos_vec);
        }
        let _voronoi= Voronoi::build_hyperbolic(&generators, rho, DVec3::splat(-1.), DVec3::splat(2.), (N-1 as usize).try_into().unwrap(), false);
        let cells = _voronoi.cells();
        for site_index in microstate.site_indices().iter() {
            if let Some(site_tag) = site_index {
                let nn_list = cells[*site_tag].neighbour_ids(&_voronoi);
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



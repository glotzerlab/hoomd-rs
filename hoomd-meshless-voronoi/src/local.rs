// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Voronoi tesselations of a given point set
*/
#![allow(dead_code)]
use crate::{Dimensionality, Voronoi};
use glam::DVec3;
use hoomd_geometry::shape::{Cuboid, EightEight};
use hoomd_manifold::Hyperboloid;
use hoomd_microstate::{
    Microstate,
    boundary::{GenerateGhosts, Open, Periodic},
    property::Position,
};
use hoomd_vector::{Cartesian, Metric};
use num::complex::Complex;
use thiserror::Error;
use ndarray::prelude::*;

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
    pub(super) fn new(
        id: usize,
        skirt: f64,
        loc: Vec<f64>,
        dimensionality: Dimensionality,
    ) -> Self {
        let mut loc = loc;
        match dimensionality {
            Dimensionality::OneD => {
                loc[1] = 0.;
                loc[2] = 0.;
            }
            Dimensionality::TwoD => loc[2] = 0.,
            _ => (),
        }
        Self {
            loc,
            skirt,
            site_tag: id,
        }
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
pub struct NeighborList<'a, B, S, C> {
    /// ordered, nested vector of 2-tuples with nearest-neighbor pairs
    pub neighbors: Vec<(usize, usize)>,
    /// Microstate
    pub microstate: &'a Microstate<B, S, C>,
}

pub trait GenerateNeighborList<B, S, C, M> {
    /** Generate the neighbor list from a given microstate
     */
    fn from_microstate(microstate: &Microstate<B, S, C>) -> NeighborList<B, S, C>;
}

impl<B, S, C> NeighborList<'_, B, S, C> {
    /// Get the neighbor list
    pub fn neighbors(&self) -> &Vec<(usize, usize)> {
        &self.neighbors
    }
    /** Get the indices of the neighbors for a specific site

    #Example

    ```
    use hoomd_microstate::{Microstate, MicrostateBuilder, Body, property::Point, boundary::Open};
    use hoomd_vector::Cartesian;
    use hoomd_meshless_voronoi::{GenerateNeighborList, NeighborList};

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
                let filtered_nlist: Vec<usize> =
                    self.neighbors
                        .clone()
                        .into_iter()
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
            None => vec![0_usize],
        }
    }
    /** Get the coordination numbers for each site in a microstate

    #Example

    ```
    use hoomd_microstate::{MicrostateBuilder, Body, boundary::Open};
    use hoomd_vector::Cartesian;
    use hoomd_meshless_voronoi::{GenerateNeighborList, NeighborList};

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

pub trait DirectorField<B, S, C, M> {
    fn hexatic(
        &self,
        microstate: &Microstate<B, S, C>,
        site_index: Option<usize>,
    ) -> Result<Complex<f64>, Error>;
    /// TODO: add description
    fn orientational_order(
        &self,
        microstate: &Microstate<B, S, C>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<ComplexField, Error>;
}

/** TODO: documentation
 */
pub struct ComplexField {
    pub bin_edges: Array<f64, Dim<[usize; 1]>>,
    pub bounds: [f64; 2],
    pub field_value: Array<Complex<f64>, Dim<[usize; 1]>>,
    pub n_bins: usize,
}

impl<B, S, C> DirectorField<B, S, C, Cartesian<2>> for NeighborList<'_, B, S, C>
where
    S: Position<Metric = Cartesian<2>>,
{
    /// compute the hexatic director field at a point from the microstate
    fn hexatic(
        &self,
        microstate: &Microstate<B, S, C>,
        site_index: Option<usize>,
    ) -> Result<Complex<f64>, Error> {
        match site_index {
            Some(num) => {
                let site_neighbors = self.neighbors_of_site(site_index);
                if site_neighbors == vec![0_usize] {
                    return Err(Error::NoNearestNeighbors);
                }
                let point = microstate.sites()[num].properties.position();
                let neighbors_translated: Vec<Cartesian<2>> = site_neighbors
                    .iter()
                    .map(|s| *microstate.sites()[*s].properties.position() - *point)
                    .collect();
                let angles: Vec<f64> = neighbors_translated
                    .iter()
                    .map(|site| (site[1]).atan2(site[0]))
                    .collect();
                let hex: Complex<f64> = angles.iter().fold(Complex::new(0.0, 0.0), |sum, theta| {
                    sum + Complex::new(0.0, 6.0 * theta).exp()
                });
                Ok(hex.scale(1.0 / (site_neighbors.len() as f64)))
            }
            None => return Err(Error::InvalidSiteIndex),
        }
    }
    /// TODO: description
    #[inline]
    fn orientational_order(
        &self,
        microstate: &Microstate<B, S, C>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<ComplexField, Error> {
        let bin_size: f64 = (r_max - r_min) / (nbins as f64);
        let bin_edges = Array::from_vec(
            (0..=nbins)
                .collect::<Vec<usize>>()
                .iter()
                .map(|i| (*i as f64) * bin_size + r_min)
                .collect::<Vec<f64>>(),
        );
        // iterate through all pairs of sites and place director into correct spot
        // all directors are stored with an index marking where they belong
        let mut directors_tagged: Vec<(Complex<f64>, usize)> = vec![];
        for site_1 in microstate.site_indices() {
            for site_2 in microstate.site_indices() {
                match site_1 {
                    Some(site_1_index) => match site_2 {
                        Some(site_2_index) => {
                            let distance = microstate.sites()[*site_1_index]
                                .properties
                                .position()
                                .distance(
                                    microstate.sites()[*site_2_index].properties.position()
                                );
                            let index = bin_edges.iter()
                                .filter(|edge| **edge <= distance)
                                .count() - 1_usize;
                            let dir1 = self.hexatic(microstate, *site_1)?;
                            let dir2 = self.hexatic(microstate, *site_2)?;
                            directors_tagged.push((dir1.conj() * dir2, index));
                        }
                        // return error if microstate is empty
                        None => return Err(Error::InvalidSiteIndex),
                    },
                    None => return Err(Error::InvalidSiteIndex),
                }
            }
        }
        let mut directors: Vec<Complex<f64>> = vec![];
        for index in 0..=nbins {
            let dirs: Vec<&(Complex<f64>, usize)> = directors_tagged.iter().filter(|(_val, bin)| *bin == index).collect();
            let num = dirs.len();
            let avg_dirs = dirs.iter().fold(Complex::new(0.0,0.0), |sum, (val, _bin)| sum + val);
            directors.push(avg_dirs.scale(1.0/(num as f64)));
        };
        Ok (ComplexField { bin_edges, bounds: [r_min, r_max], field_value: Array::from_vec(directors), n_bins: nbins})
    }
}

impl<B, S, C> DirectorField<B, S, C, Hyperboloid<3>> for NeighborList<'_, B, S, C>
where
    S: Position<Metric = Hyperboloid<3>>,
{
    /// compute the hexatic director field at a point from the microstate
    fn hexatic(
        &self,
        microstate: &Microstate<B, S, C>,
        site_index: Option<usize>,
    ) -> Result<Complex<f64>, Error> {
        match site_index {
            Some(num) => {
                let site_neighbors = self.neighbors_of_site(site_index);
                if site_neighbors == vec![0_usize] {
                    return Err(Error::NoNearestNeighbors);
                }
                let point = microstate.sites()[num].properties.position();
                let boost = -(point.point.coordinates[2] / point.skirt()).acosh();
                let rot = -point.point.coordinates[1].atan2(point.point.coordinates[0]);
                let neighbors_translated: Vec<[f64; 2]> = site_neighbors
                    .iter()
                    .map(|s| {
                        let nn = microstate.sites()[*s]
                            .properties
                            .position()
                            .point
                            .coordinates;
                        [
                            nn[0] * (boost.cosh()) * (rot.cos())
                                - nn[1] * (boost.cosh()) * (rot.sin())
                                + nn[2] * (boost.sinh()),
                            nn[0] * (rot.sin()) + nn[1] * (rot.cos()),
                        ]
                    })
                    .collect();
                let angles: Vec<f64> = neighbors_translated
                    .iter()
                    .map(|site| (site[1]).atan2(site[0]))
                    .collect();
                let hex: Complex<f64> = angles.iter().fold(Complex::new(0.0, 0.0), |sum, theta| {
                    sum + Complex::new(0.0, 6.0 * theta).exp()
                });
                Ok(hex.scale(1.0 / (site_neighbors.len() as f64)))
            }
            None => return Err(Error::InvalidSiteIndex),
        }
    }
    #[inline]
    fn orientational_order(
        &self,
        microstate: &Microstate<B, S, C>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<ComplexField, Error> {
        let bin_size: f64 = (r_max - r_min) / (nbins as f64);
        let bin_edges = Array::from_vec(
            (0..=nbins)
                .collect::<Vec<usize>>()
                .iter()
                .map(|i| (*i as f64) * bin_size + r_min)
                .collect::<Vec<f64>>(),
        );
        // iterate through all pairs of sites and place director into correct spot
        // all directors are stored with an index marking where they belong
        let mut directors_tagged: Vec<(Complex<f64>, usize)> = vec![];
        for site_1 in microstate.site_indices() {
            for site_2 in microstate.site_indices() {
                match site_1 {
                    Some(site_1_index) => match site_2 {
                        Some(site_2_index) => {
                            let distance = microstate.sites()[*site_1_index]
                                .properties
                                .position()
                                .distance(
                                    microstate.sites()[*site_2_index].properties.position()
                                );
                            let index = bin_edges.iter()
                                .filter(|edge| **edge <= distance)
                                .count() - 1_usize;
                            let dir1 = self.hexatic(microstate, *site_1)?;
                            let dir2 = self.hexatic(microstate, *site_2)?;
                            directors_tagged.push((dir1.conj() * dir2, index));
                        }
                        // return error if microstate is empty
                        None => return Err(Error::InvalidSiteIndex),
                    },
                    None => return Err(Error::InvalidSiteIndex),
                }
            }
        }
        let mut directors: Vec<Complex<f64>> = vec![];
        for index in 0..=nbins {
            let dirs: Vec<&(Complex<f64>, usize)> = directors_tagged.iter().filter(|(_val, bin)| *bin == index).collect();
            let num = dirs.len();
            let avg_dirs = dirs.iter().fold(Complex::new(0.0,0.0), |sum, (val, _bin)| sum + val);
            directors.push(avg_dirs.scale(1.0/(num as f64)));
        };
        Ok (ComplexField { bin_edges, bounds: [r_min, r_max], field_value: Array::from_vec(directors), n_bins: nbins})
    }
}

/// Enumerate possible sources of error in fallible boundary methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Given microstate has no valid indices
    #[error("invalid site index")]
    InvalidSiteIndex,
    /// No nearest neighbors
    #[error("No nearest neighbors (likely an invalid site index)")]
    NoNearestNeighbors,
}

/** Neighbor list from microstates in cartesian space

#Example

```
use hoomd_microstate::{Microstate, MicrostateBuilder, Body, property::Point, boundary::Open};
use hoomd_vector::Cartesian;
use hoomd_meshless_voronoi::{GenerateNeighborList, NeighborList};

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
impl<const N: usize, B, S> GenerateNeighborList<B, S, Open, Cartesian<N>>
    for NeighborList<'_, B, S, Open>
where
    S: Position<Metric = Cartesian<N>>,
{
    /** Compute the neighbor list from a microstate embedded in Cartesian space
     */
    #[inline]
    fn from_microstate(microstate: &Microstate<B, S, Open>) -> NeighborList<B, S, Open> {
        let mut nlist = vec![];
        let mut generators = vec![];
        let mut coord_numbers = vec![];
        for site in microstate.sites() {
            let mut pos_vec: Vec<f64> = Vec::from(site.properties.position().coordinates);
            pos_vec.push(0.0);
            let position = DVec3 {
                x: pos_vec[0],
                y: pos_vec[1],
                z: pos_vec[2],
            };
            generators.push(position);
            for n in 0..N {
                coord_numbers.push(site.properties.position()[n].floor() as i32);
            }
        }
        let max_coord = coord_numbers.iter().max();
        let min_coord = coord_numbers.iter().min();
        let anchor_num: f64 = match max_coord {
            Some(max) => (*max) as f64 + 2.0,
            None => panic!("microstate is empty!"),
        };
        let width_num: f64 = match min_coord {
            Some(min) => (*min as f64) - 1.0,
            None => panic!("microstate is empty!"),
        };
        let anchor = DVec3::splat(anchor_num);
        let width = DVec3::splat(width_num - anchor_num);
        let _voronoi = Voronoi::build(&generators, anchor, width, N.try_into().unwrap());
        let cells = _voronoi.cells();
        for site_tag in microstate.site_indices().iter().flatten() {
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
        NeighborList {
            neighbors: nlist,
            microstate,
        }
    }
}

impl<B, S> GenerateNeighborList<B, S, Periodic<Cuboid<3>>, Cartesian<3>>
    for NeighborList<'_, B, S, Periodic<Cuboid<3>>>
where
    S: Position<Metric = Cartesian<3>> + Copy + Default,
{
    /** Compute the neighbor list from a microstate embedded in three-dimensional cartesian space with periodic cuboid boundary conditions
     */
    #[inline]
    fn from_microstate(
        microstate: &Microstate<B, S, Periodic<Cuboid<3>>>,
    ) -> NeighborList<B, S, Periodic<Cuboid<3>>> {
        // the periodic = true feature of voronoi build does not work.
        // Attempt: make the voronoi diagram for the microstate + ghosts
        let mut nlist = vec![];
        let mut generators_with_ghosts = vec![];
        let n_particles = microstate.sites().len();
        let boundary = microstate.boundary().shape();
        let (min_extent, max_extent) = (boundary.minimal_extents(), boundary.maximal_extents());
        let anchor_vec: Vec<f64> = Vec::from(min_extent);
        let max_vec = Vec::from(max_extent);
        let anchor = DVec3 {
            x: anchor_vec[0] - 2.0 * max_vec[0],
            y: anchor_vec[1] - 2.0 * max_vec[0],
            z: anchor_vec[2] - 2.0 * max_vec[0],
        };
        let width = DVec3 {
            x: max_vec[0] * 6.0,
            y: max_vec[1] * 6.0,
            z: max_vec[2] * 6.0,
        };
        // first n_particles elements in generators_with_ghosts are true particles
        for site in microstate.sites() {
            let pos_vec: Vec<f64> = Vec::from(site.properties.position().coordinates);
            let position = DVec3 {
                x: pos_vec[0],
                y: pos_vec[1],
                z: pos_vec[2],
            };
            generators_with_ghosts.push(position);
        }
        // all subsequent additions are ghost particles
        let mut ghost_list: Vec<usize> = vec![]; // vector of ghost particle indices
        for site in microstate.sites() {
            let ghosts = GenerateGhosts::generate_ghosts(microstate.boundary(), &site.properties);
            for ghost in ghosts {
                let ghost_pos_vec: Vec<f64> = Vec::from(ghost.position().coordinates);
                let ghost_position = DVec3 {
                    x: ghost_pos_vec[0],
                    y: ghost_pos_vec[1],
                    z: ghost_pos_vec[2],
                };
                generators_with_ghosts.push(ghost_position);
                ghost_list.push(site.site_tag);
            }
        }

        let _voronoi = Voronoi::build(
            &generators_with_ghosts,
            anchor,
            width,
            3.try_into().unwrap(),
        );
        let cells = _voronoi.cells();
        for site_tag in microstate.site_indices().iter().flatten() {
            let nn_list = cells[*site_tag].neighbour_ids(&_voronoi);
            let mut temp_list = vec![];
            for n in nn_list {
                if n > *site_tag && n < n_particles {
                    // case where n is tag for real site
                    temp_list.push(n);
                } else if n > *site_tag && n >= n_particles {
                    // case where n is tag for ghost site
                    let real_n = ghost_list[n - n_particles];
                    if real_n > *site_tag {
                        temp_list.push(real_n);
                    }
                }
            }
            temp_list.sort();
            for n in temp_list {
                nlist.push((*site_tag, n));
            }
        }
        NeighborList {
            neighbors: nlist,
            microstate,
        }
    }
}

impl<B, S> GenerateNeighborList<B, S, Periodic<Cuboid<2>>, Cartesian<2>>
    for NeighborList<'_, B, S, Periodic<Cuboid<2>>>
where
    S: Position<Metric = Cartesian<2>> + Copy + Default,
{
    /** Compute the neighbor list from a microstate embedded in two-dimensional Cartesian space with periodic cuboid boundary conditions
     */
    #[inline]
    fn from_microstate(
        microstate: &Microstate<B, S, Periodic<Cuboid<2>>>,
    ) -> NeighborList<B, S, Periodic<Cuboid<2>>> {
        // the periodic = true feature of voronoi build does not work.
        // Attempt: make the voronoi diagram for the microstate + ghosts
        let mut nlist = vec![];
        let mut generators_with_ghosts = vec![];
        let n_particles = microstate.sites().len();
        let boundary = microstate.boundary().shape();
        let (min_extent, max_extent) = (boundary.minimal_extents(), boundary.maximal_extents());
        let anchor_vec: Vec<f64> = Vec::from(min_extent);
        let max_vec = Vec::from(max_extent);
        let anchor = DVec3 {
            x: anchor_vec[0] - 2.0 * max_vec[0],
            y: anchor_vec[1] - 2.0 * max_vec[0],
            z: 0.0,
        };
        let width = DVec3 {
            x: max_vec[0] * 6.0,
            y: max_vec[1] * 6.0,
            z: 0.0,
        };
        // first n_particles elements in generators_with_ghosts are true particles
        for site in microstate.sites() {
            let pos_vec: Vec<f64> = Vec::from(site.properties.position().coordinates);
            let position = DVec3 {
                x: pos_vec[0],
                y: pos_vec[1],
                z: 0.0,
            };
            generators_with_ghosts.push(position);
        }
        // all subsequent additions are ghost particles
        let mut ghost_list: Vec<usize> = vec![]; // vector of ghost particle indices
        for site in microstate.sites() {
            let ghosts = GenerateGhosts::generate_ghosts(microstate.boundary(), &site.properties);
            for ghost in ghosts {
                let ghost_pos_vec: Vec<f64> = Vec::from(ghost.position().coordinates);
                let ghost_position = DVec3 {
                    x: ghost_pos_vec[0],
                    y: ghost_pos_vec[1],
                    z: 0.0,
                };
                generators_with_ghosts.push(ghost_position);
                ghost_list.push(site.site_tag);
            }
        }

        let _voronoi = Voronoi::build(
            &generators_with_ghosts,
            anchor,
            width,
            2.try_into().unwrap(),
        );
        let cells = _voronoi.cells();
        for site_tag in microstate.site_indices().iter().flatten() {
            let nn_list = cells[*site_tag].neighbour_ids(&_voronoi);
            let mut temp_list = vec![];
            for n in nn_list {
                if n > *site_tag && n < n_particles {
                    // case where n is tag for real site
                    temp_list.push(n);
                } else if n > *site_tag && n >= n_particles {
                    // case where n is tag for ghost site
                    let real_n = ghost_list[n - n_particles];
                    if real_n > *site_tag {
                        temp_list.push(real_n);
                    }
                }
            }
            temp_list.sort();
            for n in temp_list {
                nlist.push((*site_tag, n));
            }
        }
        NeighborList {
            neighbors: nlist,
            microstate,
        }
    }
}
/** Neighbor list from microstates in hyperbolic space

#Example

```
use hoomd_microstate::{Microstate, MicrostateBuilder, Body, property::Point, boundary::Open};
use hoomd_manifold::{Hyperboloid, Minkowski};
use hoomd_meshless_voronoi::{GenerateNeighborList, NeighborList};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let microstate = MicrostateBuilder::with_boundary(Open)
    .bodies([Body::point(Hyperboloid::from(&Minkowski::from([1.0, -2.0, 6.0_f64.sqrt()]))),
        Body::point(Hyperboloid::from(&Minkowski::from([1.0, -1.0, 3.0_f64.sqrt()]))),
        Body::point(Hyperboloid::from(&Minkowski::from([-1.0, -2.0, 6.0_f64.sqrt()]))),
        Body::point(Hyperboloid::from(&Minkowski::from([-1.0, 1.0, 3.0_f64.sqrt()])))])
    .try_build()?;

let nlist = NeighborList::from_microstate(&microstate);
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
impl<const N: usize, B, S> GenerateNeighborList<B, S, Open, Hyperboloid<N>>
    for NeighborList<'_, B, S, Open>
where
    S: Position<Metric = Hyperboloid<N>>,
{
    /** Compute the neighbor list from a microstate in hyperbolic space
     */
    #[inline]
    fn from_microstate(microstate: &Microstate<B, S, Open>) -> NeighborList<B, S, Open> {
        let mut nlist = vec![];
        let mut generators = vec![];
        let rho = microstate.sites()[0].properties.position().skirt();
        for site in microstate.sites() {
            let mut pos_vec = site.properties.position().to_poincare();
            pos_vec.push(0.0);
            generators.push(pos_vec);
        }
        let _voronoi = Voronoi::build_hyperbolic(
            &generators,
            rho,
            DVec3::splat(-1.),
            DVec3::splat(2.),
            (N - 1_usize).try_into().unwrap(),
        );
        let cells = _voronoi.cells();
        for site_tag in microstate.site_indices().iter().flatten() {
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
        NeighborList {
            neighbors: nlist,
            microstate,
        }
    }
}

impl<B, S> GenerateNeighborList<B, S, Periodic<EightEight>, Hyperboloid<3>>
    for NeighborList<'_, B, S, Periodic<EightEight>>
where
    S: Position<Metric = Hyperboloid<3>> + Copy + Default,
{
    /** Compute the neighbor list from a microstate in two-dimensional hyperbolic space with periodic boundary conditions
     */
    #[inline]
    fn from_microstate(
        microstate: &Microstate<B, S, Periodic<EightEight>>,
    ) -> NeighborList<B, S, Periodic<EightEight>> {
        let mut nlist = vec![];
        let mut generators_with_ghosts = vec![];
        let n_particles = microstate.sites().len();
        let rho = microstate.sites()[0].properties.position().skirt();
        // first n_particles elements in generators_with_ghosts are true particles
        for site in microstate.sites() {
            let mut pos_vec = site.properties.position().to_poincare();
            pos_vec.push(0.0);
            generators_with_ghosts.push(pos_vec);
        }
        //all subsequent additions are ghosts
        let mut ghost_list: Vec<usize> = vec![];
        for site in microstate.sites() {
            let ghosts = GenerateGhosts::generate_ghosts(microstate.boundary(), &site.properties);
            for ghost in ghosts {
                let mut ghost_vec = ghost.position().to_poincare();
                ghost_vec.push(0.0);
                generators_with_ghosts.push(ghost_vec);
                ghost_list.push(site.site_tag);
            }
        }

        let _voronoi = Voronoi::build_hyperbolic(
            &generators_with_ghosts,
            rho,
            DVec3::splat(-1.),
            DVec3::splat(2.),
            2.try_into().unwrap(),
        );
        let cells = _voronoi.cells();
        for site_tag in microstate.site_indices().iter().flatten() {
            let nn_list = cells[*site_tag].neighbour_ids(&_voronoi);
            let mut temp_list = vec![];
            for n in nn_list {
                if n > *site_tag && n < n_particles {
                    temp_list.push(n)
                } else if n > *site_tag && n >= n_particles {
                    let real_n = ghost_list[n - n_particles];
                    if real_n > *site_tag {
                        temp_list.push(real_n)
                    }
                }
            }
            temp_list.sort();
            for n in temp_list {
                nlist.push((*site_tag, n));
            }
        }
        NeighborList {
            neighbors: nlist,
            microstate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use hoomd_geometry::shape::Cuboid;
    use hoomd_manifold::{Hyperboloid, Minkowski};
    use hoomd_microstate::{Body, MicrostateBuilder, boundary::Open, boundary::Periodic};
    use hoomd_vector::Cartesian;

    #[test]
    fn nlist_cartesian() {
        let microstate = MicrostateBuilder::new()
            .bodies([
                Body::point(Cartesian::from([0.5, 0.25])),
                Body::point(Cartesian::from([-1.0, 1.0])),
                Body::point(Cartesian::from([1.0, -0.75])),
                Body::point(Cartesian::from([-0.5, -0.5])),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let nlist = NeighborList::from_microstate(&microstate);
        assert_eq!(
            vec![
                (0_usize, 1_usize),
                (0_usize, 2_usize),
                (0_usize, 3_usize),
                (1_usize, 3_usize),
                (2_usize, 3_usize)
            ],
            *nlist.neighbors()
        );
    }

    #[test]
    fn nlist_cartesian_periodic_3d() {
        let boundary = Periodic::new(
            1.0,
            Cuboid::<3>::with_equal_edges(2.0.try_into().expect("hard-coded positive number")),
        )
        .expect("no interactions");
        let microstate = MicrostateBuilder::with_boundary(boundary)
            .bodies([
                Body::point(Cartesian::from([-0.6, 0.01, 0.01])),
                Body::point(Cartesian::from([0.01, 0.01, 0.01])),
                Body::point(Cartesian::from([0.6, 0.01, 0.01])),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let nlist = NeighborList::from_microstate(&microstate);
        assert_eq!(
            vec![(0_usize, 1_usize), (0_usize, 2_usize), (1_usize, 2_usize),],
            *nlist.neighbors()
        );
    }

    #[test]
    fn nlist_cartesian_periodic_2d() {
        let boundary = Periodic::new(
            1.0,
            Cuboid::<2>::with_equal_edges(2.0.try_into().expect("hard-coded positive number")),
        )
        .expect("no interactions");
        let microstate = MicrostateBuilder::with_boundary(boundary)
            .bodies([
                Body::point(Cartesian::from([-0.6, 0.01])),
                Body::point(Cartesian::from([0.01, 0.01])),
                Body::point(Cartesian::from([0.6, 0.01])),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let nlist = NeighborList::from_microstate(&microstate);
        assert_eq!(
            vec![(0_usize, 1_usize), (0_usize, 2_usize), (1_usize, 2_usize),],
            *nlist.neighbors()
        );
    }

    #[test]
    fn nlist_hyperboloid() {
        let microstate = MicrostateBuilder::with_boundary(Open)
            .bodies([
                Body::point(Hyperboloid::from(&Minkowski::from([
                    1.0,
                    -2.0,
                    6.0_f64.sqrt(),
                ]))),
                Body::point(Hyperboloid::from(&Minkowski::from([
                    1.0,
                    -1.0,
                    3.0_f64.sqrt(),
                ]))),
                Body::point(Hyperboloid::from(&Minkowski::from([
                    -1.0,
                    -2.0,
                    6.0_f64.sqrt(),
                ]))),
                Body::point(Hyperboloid::from(&Minkowski::from([
                    -1.0,
                    1.0,
                    3.0_f64.sqrt(),
                ]))),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let nlist = NeighborList::from_microstate(&microstate);
        assert_eq!(
            vec![
                (0_usize, 1_usize),
                (0_usize, 2_usize),
                (1_usize, 2_usize),
                (1_usize, 3_usize),
                (2_usize, 3_usize)
            ],
            *nlist.neighbors()
        );
    }

    #[test]
    fn coordination_numbers_cartesian() {
        let microstate = MicrostateBuilder::with_boundary(Open)
            .bodies([
                Body::point(Cartesian::from([0.5, 0.25])),
                Body::point(Cartesian::from([-1.0, 1.0])),
                Body::point(Cartesian::from([1.0, -0.75])),
                Body::point(Cartesian::from([-0.5, -0.5])),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let nlist = NeighborList::from_microstate(&microstate);
        let coordination_numbers = nlist.coordination_numbers();
        assert_eq!(
            vec![3_usize, 2_usize, 2_usize, 3_usize],
            coordination_numbers
        );
    }

    #[test]
    fn hexatic_order_cartesian() -> Result<(), Box<dyn std::error::Error>> {
        let microstate = MicrostateBuilder::with_boundary(Open)
            .bodies([
                Body::point(Cartesian::from([1.0, 1.0])),
                Body::point(Cartesian::from([2.0, 1.0])),
                Body::point(Cartesian::from([0.0, 1.0])),
                Body::point(Cartesian::from([1.5, 1.0 + (3.0_f64).sqrt() / 2.0])),
                Body::point(Cartesian::from([1.5, 1.0 - (3.0_f64).sqrt() / 2.0])),
                Body::point(Cartesian::from([0.5, 1.0 + (3.0_f64).sqrt() / 2.0])),
                Body::point(Cartesian::from([0.5, 1.0 - (3.0_f64).sqrt() / 2.0])),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let nlist = NeighborList::from_microstate(&microstate);
        let hexatic_0 = nlist.hexatic(&microstate, microstate.site_indices()[0])?;
        assert_relative_eq!(1.0, hexatic_0.re, epsilon = 1e-12);
        assert_relative_eq!(0.0, hexatic_0.im, epsilon = 1e-12);
        Ok(())
    }
}

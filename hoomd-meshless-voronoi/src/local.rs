// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Voronoi tesselations of a given point set

use crate::{PDSeed, PowerDiagram, voronoi_neighborlist};
use hoomd_geometry::shape::{Hypercuboid, EightEight};
use hoomd_manifold::Hyperbolic;
use hoomd_microstate::{
    Microstate,
    boundary::{GenerateGhosts, Open, Periodic},
    property::{Point, Position},
};
use hoomd_vector::{Cartesian, Metric};
use ndarray::prelude::*;
use num::complex::Complex;
use thiserror::Error;

/// Define the neighbor list.
///
/// The neighbor list for a given microstate is a vector of two-element tuples
/// giving the pair of nearest neighbors. Nearest neighbors are found using
/// the voronoi diagram.
pub struct NeighborList<'a, B, S, X, C> {
    /// Ordered vector of 2-tuples with nearest-neighbor pairs.
    pub neighbors: Vec<(usize, usize)>,
    /// Microstate
    pub microstate: &'a Microstate<B, S, X, C>,
}

pub trait GenerateNeighborList<B, S, X, C, M> {
    /// Generate the neighbor list from a given microstate.
    fn from_microstate(microstate: &Microstate<B, S, X, C>) -> Result<NeighborList<'_, B, S, X, C>, Error>;
}

impl<B, S, X, C> NeighborList<'_, B, S, X, C> {
    /// Get the neighbor list.
    pub fn neighbors(&self) -> &Vec<(usize, usize)> {
        &self.neighbors
    }
    /// Get the indices of the neighbors for a specific site.
    ///
    /// #Example
    ///
    /// ```
    /// use hoomd_microstate::{Microstate, MicrostateBuilder, Body, property::Point, boundary::Open};
    /// use hoomd_vector::Cartesian;
    /// use hoomd_meshless_voronoi::{GenerateNeighborList, NeighborList};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = MicrostateBuilder::new()
    ///    .bodies([Body::point(Cartesian::from([0.5, 0.25])),
    ///            Body::point(Cartesian::from([-1.0, 1.0])),
    ///            Body::point(Cartesian::from([1.0, -0.75])),
    ///            Body::point(Cartesian::from([-0.5, -0.5]))])
    ///    .try_build()?;
    ///
    /// let nlist = NeighborList::from_microstate(&microstate)?;
    /// let nlist_for_0 = nlist.neighbors_of_site(microstate.site_indices()[0]);
    /// assert_eq!(vec![1 as usize, 2 as usize, 3 as usize], nlist_for_0);
    /// # Ok(())
    /// # }
    /// ```
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
    /// Get the coordination numbers for each site in a microstate.
    ///
    /// #Example
    ///
    /// ```
    /// use hoomd_microstate::{MicrostateBuilder, Body, boundary::Open};
    /// use hoomd_vector::Cartesian;
    /// use hoomd_meshless_voronoi::{GenerateNeighborList, NeighborList};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let microstate = MicrostateBuilder::new()
    ///     .bodies([Body::point(Cartesian::from([0.5, 0.25])),
    ///             Body::point(Cartesian::from([-1.0, 1.0])),
    ///             Body::point(Cartesian::from([1.0, -0.75])),
    ///             Body::point(Cartesian::from([-0.5, -0.5]))])
    ///     .try_build()?;
    ///
    /// let nlist = NeighborList::from_microstate(&microstate)?;
    /// let coordination_numbers = nlist.coordination_numbers();
    /// assert_eq!(vec![3 as usize, 2 as usize, 2 as usize, 3 as usize], coordination_numbers);
    /// # Ok(())
    /// # }
    /// ```
    pub fn coordination_numbers(&self) -> Vec<usize> {
        let mut coord_number = vec![];
        for site_index in self.microstate.site_indices().iter() {
            coord_number.push(self.neighbors_of_site(*site_index).len());
        }
        coord_number
    }
}

pub trait DirectorField<B, S, X, C, M> {
    /// Calculate the hexatic order $`\psi_6`$ for a given site index belonging
    /// to a microstate.
    fn hexatic(
        &self,
        microstate: &Microstate<B, S, X, C>,
        site_index: Option<usize>,
    ) -> Result<Complex<f64>, Error>;
    /// Get a histrogram of hexatic orders $`\psi_6`$ across all body sites in a 
    /// given microstate.
    fn orientational_order(
        &self,
        microstate: &Microstate<B, S, X, C>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<ComplexField, Error>;
}

/// A [`SpatialHistogram`] with data type `Complex<f64>`.
pub struct ComplexField {
    pub bin_edges: Array<f64, Dim<[usize; 1]>>,
    pub bounds: [f64; 2],
    pub field_value: Array<Complex<f64>, Dim<[usize; 1]>>,
    pub n_bins: usize,
}

impl<B, S, X, C> DirectorField<B, S, X, C, Cartesian<2>> for NeighborList<'_, B, S, X, C>
where
    S: Position<Position = Cartesian<2>>,
{
    /// Compute the hexatic director field at a point from the microstate.
    fn hexatic(
        &self,
        microstate: &Microstate<B, S, X, C>,
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
            None => Err(Error::InvalidSiteIndex),
        }
    }
    /// TODO: description.
    #[inline]
    fn orientational_order(
        &self,
        microstate: &Microstate<B, S, X, C>,
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
                                .distance(microstate.sites()[*site_2_index].properties.position());
                            let index = bin_edges.iter().filter(|edge| **edge <= distance).count()
                                - 1_usize;
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
            let dirs: Vec<&(Complex<f64>, usize)> = directors_tagged
                .iter()
                .filter(|(_val, bin)| *bin == index)
                .collect();
            let num = dirs.len();
            let avg_dirs = dirs
                .iter()
                .fold(Complex::new(0.0, 0.0), |sum, (val, _bin)| sum + val);
            directors.push(avg_dirs.scale(1.0 / (num as f64)));
        }
        Ok(ComplexField {
            bin_edges,
            bounds: [r_min, r_max],
            field_value: Array::from_vec(directors),
            n_bins: nbins,
        })
    }
}

impl<B, S, X, C> DirectorField<B, S, X, C, Hyperbolic<3>> for NeighborList<'_, B, S, X, C>
where
    S: Position<Position = Hyperbolic<3>>,
{
    /// Compute the hexatic director field at a point from the microstate.
    fn hexatic(
        &self,
        microstate: &Microstate<B, S, X, C>,
        site_index: Option<usize>,
    ) -> Result<Complex<f64>, Error> {
        match site_index {
            Some(num) => {
                let site_neighbors = self.neighbors_of_site(site_index);
                if site_neighbors == vec![0_usize] {
                    return Err(Error::NoNearestNeighbors);
                }
                let point = microstate.sites()[num].properties.position();
                let boost = -(point.coordinates()[2] / 1.0).acosh();
                let rot = -point.coordinates()[1].atan2(point.coordinates()[0]);
                let neighbors_translated: Vec<[f64; 2]> = site_neighbors
                    .iter()
                    .map(|s| {
                        let nn = microstate.sites()[*s]
                            .properties
                            .position()
                            .coordinates();
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
            None => Err(Error::InvalidSiteIndex),
        }
    }
    #[inline]
    fn orientational_order(
        &self,
        microstate: &Microstate<B, S, X, C>,
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
                                .distance(microstate.sites()[*site_2_index].properties.position());
                            let index = bin_edges.iter().filter(|edge| **edge <= distance).count()
                                - 1_usize;
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
            let dirs: Vec<&(Complex<f64>, usize)> = directors_tagged
                .iter()
                .filter(|(_val, bin)| *bin == index)
                .collect();
            let num = dirs.len();
            let avg_dirs = dirs
                .iter()
                .fold(Complex::new(0.0, 0.0), |sum, (val, _bin)| sum + val);
            directors.push(avg_dirs.scale(1.0 / (num as f64)));
        }
        Ok(ComplexField {
            bin_edges,
            bounds: [r_min, r_max],
            field_value: Array::from_vec(directors),
            n_bins: nbins,
        })
    }
}

/// Enumerate possible sources of error.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum Error {
    /// Given microstate has no valid indices
    #[error("invalid site index")]
    InvalidSiteIndex,
    /// No nearest neighbors
    #[error("No nearest neighbors (likely an invalid site index)")]
    NoNearestNeighbors,
    /// Error from power diagram construction
    #[error("Error while constructing power diagram")]
    PowerDiagramError(voronoi_neighborlist::Error),
}

impl From<voronoi_neighborlist::Error> for Error {
    fn from(err: voronoi_neighborlist::Error) -> Self {
        Error::PowerDiagramError(err)
    }
}

/// Compute the neighbor list from microstates in cartesian space.
///
/// #Example
///
/// ```
/// use hoomd_microstate::{Microstate, MicrostateBuilder, Body, property::Point, boundary::Open};
/// use hoomd_vector::Cartesian;
/// use hoomd_meshless_voronoi::{GenerateNeighborList, NeighborList};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let microstate = MicrostateBuilder::new()
///     .bodies([Body::point(Cartesian::from([0.5, 0.25])),
///              Body::point(Cartesian::from([-1.0, 1.0])),
///              Body::point(Cartesian::from([1.0, -0.75])),
///              Body::point(Cartesian::from([-0.5, -0.5]))])
///     .try_build()?;
///
/// let nlist = NeighborList::from_microstate(&microstate)?;
/// assert_eq!(vec![(0 as usize, 1 as usize),
///                 (0 as usize, 2 as usize),
///                 (0 as usize, 3 as usize),
///                 (1 as usize, 3 as usize),
///                 (2 as usize, 3 as usize)],
///             *nlist.neighbors());
/// # Ok(())
/// # }
/// ```
impl<B, S, X> GenerateNeighborList<B, S, X, Open, Cartesian<2>> for NeighborList<'_, B, S, X, Open>
where
    S: Position<Position = Cartesian<2>>,
{
    /// Compute the neighbor list from a microstate in `Cartesian<2>` with `Open`
    /// boundary conditions.
    #[inline]
    fn from_microstate(
        microstate: &Microstate<B, S, X, Open>,
    ) -> Result<NeighborList<'_, B, S, X, Open>, Error> {
        let mut nlist = vec![];
        let mut seeds: Vec<PDSeed<2>> = vec![];
        let mut coordinate_numbers = vec![];
        for site_index in microstate.site_indices().iter().flatten() {
            seeds.push(PDSeed {
                coordinate: microstate.sites()[*site_index]
                    .properties
                    .position()
                    .coordinates,
                weight: 0.01,
                index: *site_index,
            });
            for n in 0..2 {
                coordinate_numbers
                    .push(microstate.sites()[*site_index].properties.position()[n].floor() as i32);
            }
        }
        let max_coord = coordinate_numbers
            .iter()
            .max()
            .expect("non empty microstate");
        let min_coord = coordinate_numbers
            .iter()
            .min()
            .expect("non empty microstate");
        let simulation_box_vertices = vec![
            [*max_coord as f64, *max_coord as f64],
            [*min_coord as f64, *max_coord as f64],
            [*min_coord as f64, *min_coord as f64],
            [*max_coord as f64, *min_coord as f64],
        ];

        let power_diagram = PowerDiagram::<2>::build(&seeds, simulation_box_vertices, 14_usize)?;
        let nnlist = power_diagram.neighborlist();
        for elt in nnlist {
            let mut elt_nlist: Vec<(usize, usize)> = elt
                .1
                .iter()
                .filter(|nghbr| **nghbr > elt.0)
                .map(|gr_nghbr| (elt.0, *gr_nghbr))
                .collect();
            elt_nlist.sort();
            for edge in elt_nlist {
                nlist.push(edge);
            }
        }
        Ok(NeighborList {
            neighbors: nlist,
            microstate,
        })
    }
}
impl<B, S, X> GenerateNeighborList<B, S, X, Open, Cartesian<3>> for NeighborList<'_, B, S, X, Open>
where
    S: Position<Position = Cartesian<3>>,
{
    /// Compute the neighbor list from a microstate in `Cartesian<3>` with `Open`
    /// boundary conditions.
    #[inline]
    fn from_microstate(
        microstate: &Microstate<B, S, X, Open>,
    ) -> Result<NeighborList<'_, B, S, X, Open>, Error> {
        let mut nlist = vec![];
        let mut seeds: Vec<PDSeed<3>> = vec![];
        let mut coordinate_numbers = vec![];
        for site_index in microstate.site_indices().iter().flatten() {
            seeds.push(PDSeed {
                coordinate: microstate.sites()[*site_index]
                    .properties
                    .position()
                    .coordinates,
                weight: 0.01,
                index: *site_index,
            });
            for n in 0..2 {
                coordinate_numbers
                    .push(microstate.sites()[*site_index].properties.position()[n].floor() as i32);
            }
        }
        let max_coord = coordinate_numbers
            .iter()
            .max()
            .expect("non empty microstate");
        let min_coord = coordinate_numbers
            .iter()
            .min()
            .expect("non empty microstate");
        let simulation_box_vertices = vec![
            [*max_coord as f64, *max_coord as f64, *max_coord as f64],
            [*min_coord as f64, *max_coord as f64, *max_coord as f64],
            [*min_coord as f64, *min_coord as f64, *max_coord as f64],
            [*max_coord as f64, *min_coord as f64, *max_coord as f64],
            [*max_coord as f64, *max_coord as f64, *min_coord as f64],
            [*min_coord as f64, *max_coord as f64, *min_coord as f64],
            [*min_coord as f64, *min_coord as f64, *min_coord as f64],
            [*max_coord as f64, *min_coord as f64, *min_coord as f64],
        ];

        let power_diagram = PowerDiagram::<3>::build(&seeds, simulation_box_vertices, 14_usize)?;
        let nnlist = power_diagram.neighborlist();
        for elt in nnlist {
            let mut elt_nlist: Vec<(usize, usize)> = elt
                .1
                .iter()
                .filter(|nghbr| **nghbr > elt.0)
                .map(|gr_nghbr| (elt.0, *gr_nghbr))
                .collect();
            elt_nlist.sort();
            for edge in elt_nlist {
                nlist.push(edge);
            }
        }
        Ok(NeighborList {
            neighbors: nlist,
            microstate,
        })
    }
}

impl<B, S, X> GenerateNeighborList<B, S, X, Periodic<Hypercuboid<3>>, Cartesian<3>>
    for NeighborList<'_, B, S, X, Periodic<Hypercuboid<3>>>
where
    S: Position<Position = Cartesian<3>> + Copy + Default,
{
    /// Compute the neighbor list from a microstate in `Cartesian<3>` with periodic
    /// `Hypercuboid<3>` boundary conditions.
    #[inline]
    fn from_microstate(
        microstate: &Microstate<B, S, X, Periodic<Hypercuboid<3>>>,
    ) -> Result<NeighborList<'_, B, S, X, Periodic<Hypercuboid<3>>>, Error> {
        let mut nlist = vec![];
        let mut seeds_with_ghosts = vec![];
        let n_particles = microstate.sites().len();
        let boundary = microstate.boundary().shape();
        let (min_extent, max_extent) = (boundary.minimal_extents(), boundary.maximal_extents());
        let simulation_box = vec![
            [
                max_extent[0] * 1.5,
                max_extent[1] * 1.5,
                max_extent[2] * 1.5,
            ],
            [
                min_extent[0] * 1.5,
                max_extent[1] * 1.5,
                max_extent[2] * 1.5,
            ],
            [
                min_extent[0] * 1.5,
                min_extent[1] * 1.5,
                max_extent[2] * 1.5,
            ],
            [
                max_extent[0] * 1.5,
                min_extent[1] * 1.5,
                max_extent[2] * 1.5,
            ],
            [
                max_extent[0] * 1.5,
                max_extent[1] * 1.5,
                min_extent[2] * 1.5,
            ],
            [
                min_extent[0] * 1.5,
                max_extent[1] * 1.5,
                min_extent[2] * 1.5,
            ],
            [
                min_extent[0] * 1.5,
                min_extent[1] * 1.5,
                min_extent[2] * 1.5,
            ],
            [
                max_extent[0] * 1.5,
                min_extent[1] * 1.5,
                min_extent[2] * 1.5,
            ],
        ];
        // first n_particles elements in generators_with_ghosts are true particles
        for site_index in microstate.site_indices().iter().flatten() {
            seeds_with_ghosts.push(PDSeed {
                coordinate: microstate.sites()[*site_index]
                    .properties
                    .position()
                    .coordinates,
                weight: 0.01,
                index: *site_index,
            });
        }
        // all subsequent additions are ghost particles
        let mut ghost_list: Vec<usize> = vec![]; // vector of ghost particle indices
        let mut count = n_particles;
        for site in microstate.sites() {
            let ghosts = GenerateGhosts::generate_ghosts(microstate.boundary(), &site.properties);
            for ghost in ghosts {
                seeds_with_ghosts.push(PDSeed {
                    coordinate: ghost.position().coordinates,
                    weight: 0.01,
                    index: count,
                });
                ghost_list.push(site.site_tag);
                count += 1;
            }
        }
        let power_diagram = PowerDiagram::<3>::build(&seeds_with_ghosts, simulation_box, 14_usize)?;
        let nnlist = power_diagram.neighborlist();

        for elt in nnlist {
            let mut elt_nlist = vec![];
            for n in elt.1.iter() {
                if *n > elt.0 && *n < n_particles {
                    elt_nlist.push(n)
                } else if *n > elt.0 && *n >= n_particles && ghost_list[n - n_particles] > elt.0 {
                    elt_nlist.push(&ghost_list[n - n_particles]);
                }
            }
            elt_nlist.sort();
            elt_nlist.dedup();
            for n in elt_nlist {
                nlist.push((elt.0, *n));
            }
        }
        Ok(NeighborList {
            neighbors: nlist,
            microstate,
        })
    }
}

impl<B, S, X> GenerateNeighborList<B, S, X, Periodic<Hypercuboid<2>>, Cartesian<2>>
    for NeighborList<'_, B, S, X, Periodic<Hypercuboid<2>>>
where
    S: Position<Position = Cartesian<2>> + Copy + Default,
{
    /// Compute the neighbor list from a microstate in `Cartesian<2>` with periodic
    /// `Hypercuboid<2>` boundary conditions.
    #[inline]
    fn from_microstate(
        microstate: &Microstate<B, S, X, Periodic<Hypercuboid<2>>>,
    ) -> Result<NeighborList<'_, B, S, X, Periodic<Hypercuboid<2>>>, Error> {
        let mut nlist = vec![];
        let mut seeds_with_ghosts = vec![];
        let n_particles = microstate.sites().len();
        let boundary = microstate.boundary().shape();
        let (min_extent, max_extent) = (boundary.minimal_extents(), boundary.maximal_extents());
        let simulation_box = vec![
            [max_extent[0] * 1.5, max_extent[1] * 1.5],
            [min_extent[0] * 1.5, max_extent[1] * 1.5],
            [min_extent[0] * 1.5, min_extent[1] * 1.5],
            [max_extent[0] * 1.5, min_extent[1] * 1.5],
        ];
        // first n_particles elements in generators_with_ghosts are true particles
        for site_index in microstate.site_indices().iter().flatten() {
            seeds_with_ghosts.push(PDSeed {
                coordinate: microstate.sites()[*site_index]
                    .properties
                    .position()
                    .coordinates,
                weight: 0.01,
                index: *site_index,
            });
        }
        // all subsequent additions are ghost particles
        let mut ghost_list: Vec<usize> = vec![]; // vector of ghost particle indices
        let mut count = n_particles;
        for site in microstate.sites() {
            let ghosts = GenerateGhosts::generate_ghosts(microstate.boundary(), &site.properties);
            for ghost in ghosts {
                seeds_with_ghosts.push(PDSeed {
                    coordinate: ghost.position().coordinates,
                    weight: 0.01,
                    index: count,
                });
                ghost_list.push(site.site_tag);
                count += 1;
            }
        }
        let power_diagram = PowerDiagram::<2>::build(&seeds_with_ghosts, simulation_box, 14_usize)?; 
        let nnlist = power_diagram.neighborlist();

        for elt in nnlist {
            let mut elt_nlist = vec![];
            for n in elt.1.iter() {
                if *n > elt.0 && *n < n_particles {
                    elt_nlist.push(n)
                } else if *n > elt.0 && *n >= n_particles && ghost_list[n - n_particles] > elt.0 {
                    elt_nlist.push(&ghost_list[n - n_particles]);
                }
            }
            elt_nlist.sort();
            elt_nlist.dedup();
            for n in elt_nlist {
                nlist.push((elt.0, *n));
            }
        }
        Ok(NeighborList {
            neighbors: nlist,
            microstate,
        })
    }
}

/// Neighbor list from microstates in hyperbolic space.
///
/// #Example
///
/// ```
/// use hoomd_microstate::{Microstate, MicrostateBuilder, Body, property::Point, boundary::Open};
/// use hoomd_manifold::{Hyperbolic, Minkowski};
/// use hoomd_meshless_voronoi::{GenerateNeighborList, NeighborList};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let microstate = MicrostateBuilder::with_boundary(Open)
///     .bodies([Body::point(Hyperbolic::from_minkowski_coordinates(Minkowski::from([1.0, -2.0, 6.0_f64.sqrt()]),1.0)),
///         Body::point(Hyperbolic::from_minkowski_coordinates(Minkowski::from([1.0, -1.0, 3.0_f64.sqrt()]),1.0)),
///         Body::point(Hyperbolic::from_minkowski_coordinates(Minkowski::from([-1.0, -2.0, 6.0_f64.sqrt()]),1.0)),
///         Body::point(Hyperbolic::from_minkowski_coordinates(Minkowski::from([-1.0, 1.0, 3.0_f64.sqrt()]),1.0))])
///     .try_build()?;
///
/// let nlist = NeighborList::from_microstate(&microstate)?;
/// assert_eq!(vec![(0 as usize, 1 as usize),
///                 (0 as usize, 2 as usize),
///                 (1 as usize, 2 as usize),
///                 (1 as usize, 3 as usize),
///                 (2 as usize, 3 as usize)],
///             *nlist.neighbors());
/// # Ok(())
/// # }
/// ```
impl<B, S, X> GenerateNeighborList<B, S, X, Open, Hyperbolic<3>> for NeighborList<'_, B, S, X, Open>
where
    S: Position<Position = Hyperbolic<3>>,
{
    /// Compute the neighbor list from a microstate in `Hyperbolic<3>` with `Open`
    /// boundary conditions.
    #[inline]
    fn from_microstate(
        microstate: &Microstate<B, S, X, Open>,
    ) -> Result<NeighborList<'_, B, S, X, Open>, Error> {
        let mut nlist = vec![];
        let to_seed = |id: &usize| {
            let coords = microstate.sites()[*id].properties.position().coordinates();
            let klein: [f64; 2] = [coords[0] / coords[2], coords[1] / coords[2]];
            let prefactor = 1.0 / (2.0 * (1.0 - klein[0].powi(2) - klein[1].powi(2)).sqrt());
            let seed_coords = [prefactor * klein[0], prefactor * klein[1]];
            let radius = (klein[0].powi(2) + klein[1].powi(2))
                / (4.0 * (1.0 - klein[0].powi(2) - klein[1].powi(2)))
                - 1.0 / (1.0 - klein[0].powi(2) - klein[1].powi(2)).sqrt();
            PDSeed {
                coordinate: seed_coords,
                weight: radius.powi(2),
                index: *id,
            }
        };
        let seeds: Vec<PDSeed<2>> = microstate
            .site_indices()
            .iter()
            .flatten()
            .map(to_seed)
            .collect();
        let max_abs = microstate
            .sites()
            .iter()
            .map(|s| Vec::from(s.properties.position().coordinates()))
            .fold(-1.0, |x: f64, y| {
                let y_abs_max: f64 = y.iter().map(|val: &f64| val.abs()).fold(0.0, f64::max);
                f64::max(x, y_abs_max)
            });
        let simulation_box_vertices = vec![
            [max_abs + 1.0, max_abs + 1.0],
            [-max_abs - 1.0, max_abs + 1.0],
            [-max_abs - 1.0, -max_abs - 1.0],
            [max_abs + 1.0, -max_abs - 1.0],
        ];
        let power_diagram = PowerDiagram::<2>::build(&seeds, simulation_box_vertices, 14_usize)?;
        let nnlist = power_diagram.neighborlist();
        for elt in nnlist {
            let mut elt_nlist: Vec<(usize, usize)> = elt
                .1
                .iter()
                .filter(|nghbr| **nghbr > elt.0)
                .map(|gr_nghbr| (elt.0, *gr_nghbr))
                .collect();
            elt_nlist.sort();
            for edge in elt_nlist {
                nlist.push(edge);
            }
        }
        Ok(NeighborList {
            neighbors: nlist,
            microstate,
        })
    }
}

impl<B, X> GenerateNeighborList<B, Point<Hyperbolic<3>>, X, Periodic<EightEight>, Hyperbolic<3>>
    for NeighborList<'_, B, Point<Hyperbolic<3>>, X, Periodic<EightEight>>
{
    /// Compute the neighbor list from a microstate in `Hyperbolic` with periodic
    /// boundary conditions.
    #[inline]
    fn from_microstate(
        microstate: &Microstate<B, Point<Hyperbolic<3>>, X, Periodic<EightEight>>,
    ) -> Result<NeighborList<'_, B, Point<Hyperbolic<3>>, X, Periodic<EightEight>>, Error> {
        let mut nlist = vec![];
        let to_seed = |id: &usize| {
            let coords = microstate.sites()[*id].properties.position().coordinates();
            let klein: [f64; 2] = [coords[0] / coords[2], coords[1] / coords[2]];
            let prefactor = 1.0 / (2.0 * (f64::from(1.0 - klein[0].powi(2) - klein[1].powi(2))).sqrt());
            let seed_coords = [prefactor * klein[0], prefactor * klein[1]];
            let radius = (klein[0].powi(2) + klein[1].powi(2))
                / (4.0 * (1.0 - klein[0].powi(2) - klein[1].powi(2)))
                - 1.0 / (f64::from(1.0 - klein[0].powi(2) - klein[1].powi(2))).sqrt();
            PDSeed {
                coordinate: seed_coords,
                weight: radius.powi(2),
                index: *id,
            }
        };
        let mut seeds_with_ghosts: Vec<PDSeed<2>> = microstate
            .site_indices()
            .iter()
            .flatten()
            .map(to_seed)
            .collect();
        let n_particles = microstate.sites().len();
        //all subsequent additions are ghosts
        let mut ghost_list: Vec<usize> = vec![];
        let mut count = n_particles;
        for site in microstate.sites() {
            let ghosts = GenerateGhosts::generate_ghosts(microstate.boundary(), &site.properties);
            for ghost in ghosts {
                let ghost_coord = ghost.position().coordinates();
                let ghost_klein = [
                    ghost_coord[0] / ghost_coord[2],
                    ghost_coord[1] / ghost_coord[2],
                ];
                let prefactor =
                    1.0 / (2.0 * (f64::from(1.0 - ghost_klein[0].powi(2) - ghost_klein[1].powi(2))).sqrt());
                let ghost_seed_coords = [prefactor * ghost_klein[0], prefactor * ghost_klein[1]];
                let ghost_radius = (ghost_klein[0].powi(2) + ghost_klein[1].powi(2))
                    / (4.0 * (1.0 - ghost_klein[0].powi(2) - ghost_klein[1].powi(2)))
                    - 1.0 / (f64::from(1.0 - ghost_klein[0].powi(2) - ghost_klein[1].powi(2))).sqrt();
                seeds_with_ghosts.push(PDSeed {
                    coordinate: ghost_seed_coords,
                    weight: ghost_radius,
                    index: count,
                });
                ghost_list.push(site.site_tag);
                count += 1;
            }
        }
        let max_abs = seeds_with_ghosts
            .iter()
            .map(|s: &PDSeed<2>| Vec::from(s.coordinate()))
            .fold(-1.0, |x: f64, y| {
                let y_abs_max: f64 = y.iter().map(|val: &f64| val.abs()).fold(0.0, f64::max);
                f64::max(x, y_abs_max)
            });
        let simulation_box_vertices = vec![
            [max_abs + 1.0, max_abs + 1.0],
            [-max_abs - 1.0, max_abs + 1.0],
            [-max_abs - 1.0, -max_abs - 1.0],
            [max_abs + 1.0, -max_abs - 1.0],
        ];

        let power_diagram =
            PowerDiagram::<2>::build(&seeds_with_ghosts, simulation_box_vertices, 14_usize)?; 
        let nnlist = power_diagram.neighborlist();

        for elt in nnlist {
            let mut elt_nlist = vec![];
            for n in elt.1.iter() {
                if *n > elt.0 && *n < n_particles {
                    elt_nlist.push(n)
                } else if *n > elt.0 && *n >= n_particles && ghost_list[n - n_particles] > elt.0 {
                    elt_nlist.push(&ghost_list[n - n_particles]);
                }
            }
            elt_nlist.sort();
            elt_nlist.dedup();
            for n in elt_nlist {
                nlist.push((elt.0, *n));
            }
        }
        Ok(NeighborList {
            neighbors: nlist,
            microstate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use hoomd_geometry::shape::Hypercuboid;
    use hoomd_manifold::{Hyperbolic, Minkowski};
    use hoomd_microstate::{Body, MicrostateBuilder, boundary::Open, boundary::Periodic};
    use hoomd_vector::Cartesian;

    #[test]
    fn nlist_2d_cartesian_open() -> Result<(), Box<dyn std::error::Error>> {
        let microstate = Microstate::builder()
            .bodies([
                Body::point(Cartesian::from([0.5, 0.25])),
                Body::point(Cartesian::from([-1.01, 1.01])),
                Body::point(Cartesian::from([1.03, -0.75])),
                Body::point(Cartesian::from([-0.52, -0.52])),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let nlist = NeighborList::from_microstate(&microstate)?;
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
        Ok(())
    }

    #[test]
    fn nlist_cartesian_periodic_3d() -> Result<(), Box<dyn std::error::Error>> {
        let boundary = Periodic::new(
            1.0,
            Hypercuboid::<3>::with_equal_edges(2.0.try_into().expect("hard-coded positive number")),
        )
        .expect("no interactions");
        let microstate = Microstate::builder()
            .boundary(boundary)
            .bodies([
                Body::point(Cartesian::from([-0.6, 0.01, 0.01])),
                Body::point(Cartesian::from([0.01, 0.01, 0.01])),
                Body::point(Cartesian::from([0.6, 0.01, 0.01])),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let nlist = NeighborList::from_microstate(&microstate)?;
        assert_eq!(
            vec![(0_usize, 1_usize), (0_usize, 2_usize), (1_usize, 2_usize),],
            *nlist.neighbors()
        );
        Ok(())
    }

    #[test]
    fn nlist_cartesian_periodic_2d() -> Result<(), Box<dyn std::error::Error>> {
        let boundary = Periodic::new(
            1.0,
            Hypercuboid::<2>::with_equal_edges(2.0.try_into().expect("hard-coded positive number")),
        )
        .expect("no interactions");
        let microstate = Microstate::builder()
            .boundary(boundary)
            .bodies([
                Body::point(Cartesian::from([-0.6, 0.01])),
                Body::point(Cartesian::from([0.01, 0.01])),
                Body::point(Cartesian::from([0.6, 0.01])),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let nlist = NeighborList::from_microstate(&microstate)?;
        assert_eq!(
            vec![(0_usize, 1_usize), (0_usize, 2_usize), (1_usize, 2_usize),],
            *nlist.neighbors()
        );
        Ok(())
    }

    #[test]
    fn nlist_hyperbolic() -> Result<(), Box<dyn std::error::Error>> {
        let microstate = Microstate::builder()
            .boundary(Open)
            .bodies([
                Body::point(Hyperbolic::from_minkowski_coordinates([
                    1.0,
                    -2.0,
                    6.0_f64.sqrt(),
                ].into())),
                Body::point(Hyperbolic::from_minkowski_coordinates([
                    1.0,
                    -1.0,
                    3.0_f64.sqrt(),
                ].into())),
                Body::point(Hyperbolic::from_minkowski_coordinates([
                    -1.0,
                    -2.0,
                    6.0_f64.sqrt(),
                ].into())),
                Body::point(Hyperbolic::from_minkowski_coordinates([
                    -1.0,
                    1.0,
                    3.0_f64.sqrt(),
                ].into())),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let nlist = NeighborList::from_microstate(&microstate)?;
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
        Ok(())
    }

    #[test]
    fn coordination_numbers_cartesian() -> Result<(), Box<dyn std::error::Error>> {
        let microstate = Microstate::builder()
            .boundary(Open)
            .bodies([
                Body::point(Cartesian::from([0.5, 0.25])),
                Body::point(Cartesian::from([-1.0, 1.0])),
                Body::point(Cartesian::from([1.0, -0.75])),
                Body::point(Cartesian::from([-0.5, -0.5])),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let nlist = NeighborList::from_microstate(&microstate)?;
        let coordination_numbers = nlist.coordination_numbers();
        assert_eq!(
            vec![3_usize, 2_usize, 2_usize, 3_usize],
            coordination_numbers
        );
        Ok(())
    }

    #[test]
    fn hexatic_order_cartesian() -> Result<(), Box<dyn std::error::Error>> {
        let microstate = Microstate::builder()
            .boundary(Open)
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

        let nlist = NeighborList::from_microstate(&microstate)?;
        let hexatic_0 = nlist.hexatic(&microstate, microstate.site_indices()[0])?;
        assert_relative_eq!(1.0, hexatic_0.re, epsilon = 1e-12);
        assert_relative_eq!(0.0, hexatic_0.im, epsilon = 1e-12);
        Ok(())
    }
}

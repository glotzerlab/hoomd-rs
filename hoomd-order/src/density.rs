// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement various methods relating to the density of the system, namely radial distribution functions and correlation functions.
The methods are based on the struct `SpatialHistogram`.
*/

#![allow(dead_code)]
use hoomd_geometry::shape::{Cuboid, EightEight};
use hoomd_manifold::Hyperboloid;
use hoomd_microstate::{
    Microstate, MicrostateBuilder, Transform,
    boundary::{GenerateGhosts, MaximumAllowableInteractionRange, Open, Periodic},
    property::Position,
};
use hoomd_vector::{Cartesian, Metric};
use ndarray::prelude::*;
use thiserror::Error;

/** Struct for creating and manipulating histograms. `N` specifies the dimension of the histogram bins (may be 1, 2 or 3), `C`
is the boundary condition of the data (e.g. `Open`, periodic), and `A` is the type for the data itself. `A` must be able to implement `Add`
 and `PartialOrd`.

The default output `bin_counts` is an array of the frequencies for each of the bins and is stored as the type `usize`.

```
use hoomd_microstate::{Microstate, property::Position};
use hoomd_order::{SpatialHistogram, GenerateHistogram};
use hoomd_vector::Metric;
use ndarray::prelude::*;

let numbers = vec![[1],[2],[4],[10],[11],[12],[14],[20],[21],[22],[23]];
let bin_edges = array![
    [0_usize, 10_usize, 20_usize, 30_usize],
    [0_usize, 0_usize, 0_usize, 0_usize]
];
let bounds = [[0_usize, 30_usize]; 1];
let hist =
    SpatialHistogram::<1, usize>::histogram(&numbers, bin_edges, bounds, [3_usize]);
let ans = array![4_usize, 4_usize, 3_usize];
assert_eq!(ans, hist.bin_counts);
```
*/
pub struct SpatialHistogram<const N: usize, A> {
    /// a vector containing the bin edges of the histogram
    pub bin_edges: Array<A, Dim<[usize; 2]>>,
    /// an array containing the upper and lower bounds of the histogram
    pub bounds: [[A; 2]; N],
    /// the bin counts in the histogram
    pub bin_counts: Array<usize, Dim<[usize; N]>>,
    /// number of bins in each dimension
    pub n_bins: [usize; N],
}

/// TODO: documentation
pub struct NormalizedHistogram {
    /// a vector containing the bin edges of the histogram
    pub bin_edges: Array<f64, Dim<[usize; 1]>>,
    /// an array containing the upper and lower bounds of the histogram
    pub bounds: [f64; 2],
    /// the bin counts in the histogram
    pub bin_counts: Array<f64, Dim<[usize; 1]>>,
    /// number of bins in each dimension
    pub n_bins: usize,
}

impl NormalizedHistogram {
    /// normalize the 1D histogram
    #[inline]
    fn normalize(histogram: &SpatialHistogram<1, f64>) -> NormalizedHistogram {
        let sum = histogram
            .bin_counts
            .iter()
            .fold(0.0_f64, |sum, x| sum + *x as f64);
        let normed_counts: Vec<f64> = histogram
            .bin_counts
            .iter()
            .map(|c| (*c as f64) / sum)
            .collect();
        let n_bins = histogram.n_bins[0];
        let bounds = histogram.bounds[0];
        let bin_edges: Array<f64, Dim<[usize; 1]>> = histogram.bin_edges.row(0).to_owned();
        NormalizedHistogram {
            bin_edges,
            bounds,
            bin_counts: Array::from_vec(normed_counts),
            n_bins,
        }
    }
}

/** Compute a histogram with `N` dimensional data of type `A` which implements `Add` and `PartialOrd`
 */
pub trait GenerateHistogram<const N: usize, A> {
    /// generate a histogram from a given microstate
    fn histogram(
        data: &[[A; N]],
        bin_edges: Array<A, Dim<[usize; 2]>>,
        bounds: [[A; 2]; N],
        nbins: [usize; N],
    ) -> SpatialHistogram<N, A>;
}

/** Various correlation functions from microstate data.
 */
pub trait CorrelationFunction<B, S, C, M> {
    /// computes the radial distribution function g(r) from a given microstate
    /// TODO:
    /// # Errors
    fn rdf(
        microstate: &Microstate<B, S, C>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<SpatialHistogram<1, f64>, Error>;
    /// get the normalized rdf
    /// # Errors
    #[inline]
    fn normed_rdf(
        microstate: &Microstate<B, S, C>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<NormalizedHistogram, Error> {
        let rdf = Self::rdf(microstate, r_min, r_max, nbins)?;
        Ok(NormalizedHistogram::normalize(&rdf))
    }
}

/// Enumerate possible sources of error in fallible density methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Given microstate has no valid indices
    #[error("given microstate is empty")]
    EmptyMicrostate,
    /// The maximum interaction range is larger than the periodic boundary condition will allow.
    #[error("the requested RDF range ({0}) is larger than the boundary will allow ({1})")]
    RDFRangeTooLarge(f64, f64),
}

impl<const N: usize, A> SpatialHistogram<N, A> {
    /// A 2D array with the bin edges of the histogram. Each row gives the edges along one of the axes.
    #[inline]
    pub fn bin_edges(&self) -> &Array<A, Dim<[usize; 2]>> {
        &self.bin_edges
    }
    /// The lower and upper bounds of the histogram for each of the axes
    #[inline]
    pub fn bounds(&self) -> &[[A; 2]; N] {
        &self.bounds
    }
    /// The frequency counts for each of the bins
    #[inline]
    pub fn bin_counts(&self) -> &Array<usize, Dim<[usize; N]>> {
        &self.bin_counts
    }
}

impl<A> GenerateHistogram<1, A> for SpatialHistogram<1, A>
where
    A: std::ops::Add<Output = A> + std::cmp::PartialOrd,
{
    /// Create a one-dimensional histogram
    #[inline]
    fn histogram(
        data: &[[A; 1]],
        bin_edges: Array<A, Dim<[usize; 2]>>,
        bounds: [[A; 2]; 1],
        nbins: [usize; 1],
    ) -> Self {
        let mut counts = Array::zeros(nbins);
        for n in 0..nbins[0] {
            counts[n] = data
                .iter()
                .filter(|[a]| *a > bin_edges[[0, n]] && *a <= bin_edges[[0, n + 1]])
                .count();
        }
        SpatialHistogram {
            bin_edges,
            bounds,
            bin_counts: counts,
            n_bins: nbins,
        }
    }
}

impl<B, S, M> CorrelationFunction<B, S, Open, M> for SpatialHistogram<1, f64>
where
    S: Position<Metric = M>,
    M: Metric,
{
    /// Calculate the radial distribution function (RDF), g(r), for a given microstate.
    #[inline]
    fn rdf(
        microstate: &Microstate<B, S, Open>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<Self, Error> {
        let bin_size: f64 = (r_max - r_min) / (nbins as f64);
        let bin_edges_arr = Array::from_vec(
            (0..=nbins)
                .collect::<Vec<usize>>()
                .iter()
                .map(|i| (*i as f64) * bin_size + r_min)
                .collect::<Vec<f64>>(),
        );
        let bin_edges: Array<f64, Dim<[usize; 2]>> =
            ndarray::stack![Axis(0), bin_edges_arr, bin_edges_arr];
        let mut distances: Vec<[f64; 1]> = vec![];
        for site_1 in microstate.site_indices() {
            for site_2 in microstate.site_indices() {
                match site_1 {
                    Some(site_1_index) => match site_2 {
                        Some(site_2_index) => {
                            if site_1_index > site_2_index {
                                distances.push([microstate.sites()[*site_1_index]
                                    .properties
                                    .position()
                                    .distance(
                                        microstate.sites()[*site_2_index].properties.position(),
                                    )]);
                            }
                        }
                        // return blank if microstate is empty
                        None => return Err(Error::EmptyMicrostate),
                    },
                    None => return Err(Error::EmptyMicrostate),
                }
            }
        }
        Ok(SpatialHistogram::<1, f64>::histogram(
            &distances,
            bin_edges,
            [[r_min, r_max]; 1],
            [nbins],
        ))
    }
}

impl<B, S> CorrelationFunction<B, S, Periodic<Cuboid<2>>, Cartesian<2>> for SpatialHistogram<1, f64>
where
    S: Position<Metric = Cartesian<2>> + Copy + Default,
    B: Transform<S> + Position<Metric = Cartesian<2>> + Copy,
{
    /// Calculate the radial distribution function (RDF), g(r), for a given microstate with periodic boundary conditions
    #[inline]
    fn rdf(
        microstate: &Microstate<B, S, Periodic<Cuboid<2>>>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<Self, Error> {
        let boundary_max = MaximumAllowableInteractionRange::maximum_allowable_interaction_range(
            microstate.boundary().shape(),
        );
        if r_max > boundary_max {
            return Err(Error::RDFRangeTooLarge(r_max, boundary_max));
        }

        let bin_size: f64 = (r_max - r_min) / (nbins as f64);
        let bin_edges_arr = Array::from_vec(
            (0..=nbins)
                .collect::<Vec<usize>>()
                .iter()
                .map(|i| (*i as f64) * bin_size + r_min)
                .collect::<Vec<f64>>(),
        );
        let bin_edges: Array<f64, Dim<[usize; 2]>> =
            ndarray::stack![Axis(0), bin_edges_arr, bin_edges_arr];
        let mut distances: Vec<[f64; 1]> = vec![];

        let max_boundary = Periodic::new(boundary_max, *microstate.boundary().shape())
            .expect("copy of valid boundary");
        let new_microstate: Microstate<B, S, Periodic<Cuboid<2>>> =
            MicrostateBuilder::with_boundary(max_boundary)
                .bodies(microstate.bodies().iter().map(|b| b.clone().item))
                .try_build()
                .expect("copy of existing valid microstate");
        let mut all_ghosts: Vec<Cartesian<2>> = vec![];
        for site_b in new_microstate.sites() {
            let ghosts =
                GenerateGhosts::generate_ghosts(new_microstate.boundary(), &site_b.properties);
            for ghost in ghosts {
                all_ghosts.push(*ghost.position());
            }
        }
        let mut everyone_else: Vec<Cartesian<2>> = microstate
            .sites()
            .iter()
            .map(|s| *s.properties.position())
            .collect();
        everyone_else.append(&mut all_ghosts);
        for site_1 in microstate.sites() {
            for site_2 in &everyone_else {
                let loc = site_1.properties.position();
                let distance = loc.distance(site_2);
                distances.push([distance]);
            }
        }

        Ok(SpatialHistogram::<1, f64>::histogram(
            &distances,
            bin_edges,
            [[r_min, r_max]; 1],
            [nbins],
        ))
    }
}

impl<B, S> CorrelationFunction<B, S, Periodic<Cuboid<3>>, Cartesian<3>> for SpatialHistogram<1, f64>
where
    S: Position<Metric = Cartesian<3>> + Copy + Default,
    B: Transform<S> + Position<Metric = Cartesian<3>> + Copy,
{
    /// Calculate the radial distribution function (RDF), g(r), for a given microstate with periodic boundary conditions
    #[inline]
    fn rdf(
        microstate: &Microstate<B, S, Periodic<Cuboid<3>>>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<Self, Error> {
        let boundary_max = MaximumAllowableInteractionRange::maximum_allowable_interaction_range(
            microstate.boundary().shape(),
        );
        if r_max > boundary_max {
            return Err(Error::RDFRangeTooLarge(r_max, boundary_max));
        }

        let bin_size: f64 = (r_max - r_min) / (nbins as f64);
        let bin_edges_arr = Array::from_vec(
            (0..=nbins)
                .collect::<Vec<usize>>()
                .iter()
                .map(|i| (*i as f64) * bin_size + r_min)
                .collect::<Vec<f64>>(),
        );
        let bin_edges: Array<f64, Dim<[usize; 2]>> =
            ndarray::stack![Axis(0), bin_edges_arr, bin_edges_arr];
        let mut distances: Vec<[f64; 1]> = vec![];

        let max_boundary = Periodic::new(boundary_max, *microstate.boundary().shape())
            .expect("copy of valid boundary");
        let new_microstate: Microstate<B, S, Periodic<Cuboid<3>>> =
            MicrostateBuilder::with_boundary(max_boundary)
                .bodies(microstate.bodies().iter().map(|b| b.clone().item))
                .try_build()
                .expect("copy of existing valid microstate");
        let mut all_ghosts: Vec<Cartesian<3>> = vec![];
        for site_b in new_microstate.sites() {
            let ghosts =
                GenerateGhosts::generate_ghosts(new_microstate.boundary(), &site_b.properties);
            for ghost in ghosts {
                all_ghosts.push(*ghost.position());
            }
        }
        let mut everyone_else: Vec<Cartesian<3>> = microstate
            .sites()
            .iter()
            .map(|s| *s.properties.position())
            .collect();
        everyone_else.append(&mut all_ghosts);
        for site_1 in microstate.sites() {
            for site_2 in &everyone_else {
                let loc = site_1.properties.position();
                let distance = loc.distance(site_2);
                distances.push([distance]);
            }
        }

        Ok(SpatialHistogram::<1, f64>::histogram(
            &distances,
            bin_edges,
            [[r_min, r_max]; 1],
            [nbins],
        ))
    }
}

impl<B, S> CorrelationFunction<B, S, Periodic<EightEight>, Hyperboloid<3>>
    for SpatialHistogram<1, f64>
where
    S: Position<Metric = Hyperboloid<3>> + Copy + Default,
    B: Transform<S> + Position<Metric = Hyperboloid<3>> + Copy,
{
    /// Calculate the radial distribution function (RDF), g(r), for a given microstate with periodic boundary conditions
    #[inline]
    fn rdf(
        microstate: &Microstate<B, S, Periodic<EightEight>>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<Self, Error> {
        let boundary_max = MaximumAllowableInteractionRange::maximum_allowable_interaction_range(
            microstate.boundary().shape(),
        );
        if r_max > boundary_max {
            return Err(Error::RDFRangeTooLarge(r_max, boundary_max));
        }

        let bin_size: f64 = (r_max - r_min) / (nbins as f64);
        let bin_edges_arr = Array::from_vec(
            (0..=nbins)
                .collect::<Vec<usize>>()
                .iter()
                .map(|i| (*i as f64) * bin_size + r_min)
                .collect::<Vec<f64>>(),
        );
        let bin_edges: Array<f64, Dim<[usize; 2]>> =
            ndarray::stack![Axis(0), bin_edges_arr, bin_edges_arr];
        let mut distances: Vec<[f64; 1]> = vec![];

        let max_boundary = Periodic::new(
            1.0,
            EightEight {
                skirt: microstate.sites()[0].properties.position().skirt(),
            },
        )
        .expect("hard coded");
        let new_microstate: Microstate<B, S, Periodic<EightEight>> =
            MicrostateBuilder::with_boundary(max_boundary)
                .bodies(microstate.bodies().iter().map(|b| b.clone().item))
                .try_build()
                .expect("copy of existing valid microstate");
        let mut all_ghosts: Vec<Hyperboloid<3>> = vec![];
        for site_b in new_microstate.sites() {
            let ghosts =
                GenerateGhosts::generate_ghosts(new_microstate.boundary(), &site_b.properties);
            for ghost in ghosts {
                all_ghosts.push(*ghost.position());
                //println!("{:?}", ghost.position())
            }
        }
        let mut everyone_else: Vec<Hyperboloid<3>> = microstate
            .sites()
            .iter()
            .map(|s| *s.properties.position())
            .collect();
        everyone_else.append(&mut all_ghosts);
        for site_1 in microstate.sites() {
            for site_2 in &everyone_else {
                let loc = site_1.properties.position();
                let distance = loc.distance(site_2);
                distances.push([distance]);
            }
        }

        Ok(SpatialHistogram::<1, f64>::histogram(
            &distances,
            bin_edges,
            [[r_min, r_max]; 1],
            [nbins],
        ))
    }
}

impl<A> GenerateHistogram<2, A> for SpatialHistogram<2, A>
where
    A: std::ops::Add<Output = A> + std::cmp::PartialOrd,
{
    /// Create a two-dimensional histogram
    #[inline]
    fn histogram(
        data: &[[A; 2]],
        bin_edges: Array<A, Dim<[usize; 2]>>,
        bounds: [[A; 2]; 2],
        nbins: [usize; 2],
    ) -> Self {
        let mut counts = Array::zeros((nbins[0], nbins[1]));
        for i in 0..nbins[0] {
            for j in 0..nbins[1] {
                counts[[i, j]] = data
                    .iter()
                    .filter(|[a, b]| {
                        *a > bin_edges[[0, i]]
                            && *a <= bin_edges[[0, i + 1]]
                            && *b > bin_edges[[1, j]]
                            && *b <= bin_edges[[1, j + 1]]
                    })
                    .count();
            }
        }
        SpatialHistogram {
            bin_edges,
            bounds,
            bin_counts: counts,
            n_bins: nbins,
        }
    }
}

impl<A> GenerateHistogram<3, A> for SpatialHistogram<3, A>
where
    A: std::ops::Add<Output = A> + std::cmp::PartialOrd,
{
    /// Create a three-dimensional histogram
    #[inline]
    fn histogram(
        data: &[[A; 3]],
        bin_edges: Array<A, Dim<[usize; 2]>>,
        bounds: [[A; 2]; 3],
        nbins: [usize; 3],
    ) -> Self {
        let mut counts = Array::zeros((nbins[0], nbins[1], nbins[2]));
        for i in 0..nbins[0] {
            for j in 0..nbins[1] {
                for k in 0..nbins[2] {
                    counts[[i, j, k]] = data
                        .iter()
                        .filter(|[a, b, c]| {
                            *a > bin_edges[[0, i]]
                                && *a <= bin_edges[[0, i + 1]]
                                && *b > bin_edges[[1, j]]
                                && *b <= bin_edges[[1, j + 1]]
                                && *c > bin_edges[[2, k]]
                                && *c <= bin_edges[[2, k + 1]]
                        })
                        .count();
                }
            }
        }
        SpatialHistogram {
            bin_edges,
            bounds,
            bin_counts: counts,
            n_bins: nbins,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_manifold::{Hyperboloid, Minkowski};
    use hoomd_microstate::{Body, MicrostateBuilder, boundary::Open};
    use hoomd_vector::Cartesian;
    use std::f64::consts::PI;

    #[test]

    fn linear_1d_histogram_usize() {
        let numbers = vec![
            [1],
            [2],
            [4],
            [10],
            [11],
            [12],
            [14],
            [20],
            [21],
            [22],
            [23],
        ];
        let bin_edges = array![
            [0_usize, 10_usize, 20_usize, 30_usize],
            [0_usize, 0_usize, 0_usize, 0_usize]
        ];
        let bounds = [[0_usize, 30_usize]; 1];
        let hist = SpatialHistogram::<1, usize>::histogram(&numbers, bin_edges, bounds, [3_usize]);
        let ans = array![4_usize, 4_usize, 3_usize];
        assert_eq!(ans, hist.bin_counts);
    }
    #[test]
    fn linear_1d_histogram_f64() {
        let mut numbers: Vec<[f64; 1]> = vec![];
        for i in 1..=20 {
            numbers.push([f64::from(i) * 0.5]);
        }
        let bin_edges = array![
            [0.0_f64, 2.5_f64, 5.0_f64, 7.5_f64, 10.0_f64],
            [0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64]
        ];
        let bounds = [[0.0_f64, 10.0_f64]; 1];
        let hist = SpatialHistogram::<1, f64>::histogram(&numbers, bin_edges, bounds, [4_usize]);
        let ans = array![5_usize, 5_usize, 5_usize, 5_usize];
        assert_eq!(ans, hist.bin_counts);
    }
    #[test]
    fn linear_2d_histogram_usize() {
        let numbers: Vec<[usize; 2]> = vec![[1, 1], [3, 1], [1, 4], [1, 5]];
        let bin_edges = array![[0_usize, 2_usize, 4_usize], [0_usize, 3_usize, 6_usize]];
        let bounds = [[0_usize, 4_usize], [0_usize, 6_usize]];
        let hist = SpatialHistogram::<2, usize>::histogram(&numbers, bin_edges, bounds, [2, 2]);
        let ans = array![[1_usize, 2_usize], [1_usize, 0_usize]];
        assert_eq!(ans, hist.bin_counts);
    }
    #[test]
    fn linear_2d_histogram_float() {
        let numbers: Vec<[f64; 2]> = vec![
            [0.25, 0.25],
            [0.25, 0.5],
            [0.5, 0.5],
            [1.5, 0.5],
            [1.5, 1.5],
        ];
        let bin_edges = array![[0.0_f64, 1_f64, 2_f64], [0_f64, 1_f64, 2_f64]];
        let bounds = [[0.0_f64, 2.0_f64], [0.0_f64, 2.0_f64]];
        let hist = SpatialHistogram::<2, f64>::histogram(&numbers, bin_edges, bounds, [2, 2]);
        let ans = array![[3_usize, 0_usize], [1_usize, 1_usize]];
        assert_eq!(ans, hist.bin_counts);
    }
    #[test]
    fn linear_3d_histogram_usize() {
        let numbers: Vec<[usize; 3]> = vec![[1, 3, 1], [3, 1, 1], [1, 1, 3], [3, 3, 3]];
        let bin_edges = array![
            [0_usize, 2_usize, 4_usize],
            [0_usize, 2_usize, 4_usize],
            [0_usize, 2_usize, 4_usize]
        ];
        let bounds = [[0_usize, 4_usize], [0_usize, 4_usize], [0_usize, 4_usize]];
        let hist = SpatialHistogram::<3, usize>::histogram(&numbers, bin_edges, bounds, [2, 2, 2]);
        let ans = array![
            [[0_usize, 1_usize], [1_usize, 0_usize]],
            [[1_usize, 0_usize], [0_usize, 1_usize]]
        ];
        assert_eq!(ans, hist.bin_counts);
    }
    #[test]
    fn linear_3d_histogram_f64() {
        let numbers: Vec<[f64; 3]> = vec![[0.5, 0.5, 0.5], [0.5, 0.5, 1.5], [1.5, 1.5, 1.5]];
        let bin_edges = array![
            [0_f64, 1_f64, 2_f64],
            [0_f64, 1_f64, 2_f64],
            [0_f64, 1_f64, 2_f64]
        ];
        let bounds = [[0_f64, 2_f64], [0_f64, 2_f64], [0_f64, 2_f64]];
        let hist = SpatialHistogram::<3, f64>::histogram(&numbers, bin_edges, bounds, [2, 2, 2]);
        let ans = array![
            [[1_usize, 1_usize], [0_usize, 0_usize]],
            [[0_usize, 0_usize], [0_usize, 1_usize]]
        ];
        assert_eq!(ans, hist.bin_counts);
    }

    #[test]
    fn rdf_cartesian_square() -> Result<(), Box<dyn std::error::Error>> {
        const SIZE: usize = 2;
        let a: f64 = 1.0;
        let mut microstate = MicrostateBuilder::with_boundary(Open)
            .try_build()
            .expect("empty microstate");
        for i in 0..SIZE {
            for j in 0..SIZE {
                let new_point = Cartesian::from([(i as f64) * a, (j as f64) * a]);
                microstate
                    .add_body(Body::point(new_point))
                    .expect("hard coded distributions should be valid");
            }
        }
        let rdf_hist = SpatialHistogram::<1, f64>::rdf(&microstate, 0.0_f64, 2.0_f64, 2_usize)?;
        let ans = array![4_usize, 2_usize];
        assert_eq!(ans, rdf_hist.bin_counts);
        assert_eq!(rdf_hist.bin_edges.slice(s![0, ..]), array![0.0, 1.0, 2.0]);

        let rdf_hist_normalized = NormalizedHistogram::normalize(&rdf_hist);
        let ans_normed = array![2.0 / 3.0, 1.0 / 3.0];
        assert_eq!(ans_normed, rdf_hist_normalized.bin_counts);
        Ok(())
    }

    #[test]
    fn rdf_cartesian_square_periodic() -> Result<(), Box<dyn std::error::Error>> {
        const SIZE: usize = 2;
        let boundary = Periodic::new(
            1.0,
            Cuboid::<2>::with_equal_edges(2.0.try_into().expect("hard-coded positive number")),
        )
        .expect("no interactions");
        let microstate = MicrostateBuilder::with_boundary(boundary)
            .bodies([
                Body::point(Cartesian::from([-0.5, 0.8])),
                Body::point(Cartesian::from([0.75, 0.8])),
                Body::point(Cartesian::from([-0.5, -0.8])),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let rdf_hist = SpatialHistogram::<1, f64>::rdf(&microstate, 0.0_f64, 1.0_f64, 2_usize)?;
        let ans = array![2_usize, 4_usize];
        assert_eq!(ans, rdf_hist.bin_counts);
        assert_eq!(rdf_hist.bin_edges.slice(s![0, ..]), array![0.0, 0.5, 1.0]);

        let rdf_hist_normalized = NormalizedHistogram::normalize(&rdf_hist);
        let ans_normed = array![1.0 / 3.0, 2.0 / 3.0];
        assert_eq!(ans_normed, rdf_hist_normalized.bin_counts);
        Ok(())
    }

    #[test]
    fn rdf_hyperboloid() -> Result<(), Box<dyn std::error::Error>> {
        let microstate = MicrostateBuilder::with_boundary(Open)
            .bodies([
                Body::point(Hyperboloid::from(&Minkowski::from([
                    1.0,
                    0.0,
                    2.0_f64.sqrt(),
                ]))),
                Body::point(Hyperboloid::from(&Minkowski::from([
                    2.0,
                    0.0,
                    5.0_f64.sqrt(),
                ]))),
                Body::point(Hyperboloid::from(&Minkowski::from([
                    1.0,
                    1.0,
                    3.0_f64.sqrt(),
                ]))),
                Body::point(Hyperboloid::from(&Minkowski::from([
                    2.0,
                    1.0,
                    6.0_f64.sqrt(),
                ]))),
            ])
            .try_build()
            .expect("hard coded distribution should be valid");
        let rdf_hist = SpatialHistogram::<1, f64>::rdf(&microstate, 0.0_f64, 2.0_f64, 4_usize)?;
        let ans = array![0_usize, 5_usize, 1_usize, 0_usize];
        assert_eq!(ans, rdf_hist.bin_counts);
        assert_eq!(
            rdf_hist.bin_edges.slice(s![0, ..]),
            array![0.0, 0.5, 1.0, 1.5, 2.0]
        );

        let rdf_hist_normalized = NormalizedHistogram::normalize(&rdf_hist);
        let ans_normed = array![0.0, 5.0 / 6.0, 1.0 / 6.0, 0.0];
        assert_eq!(ans_normed, rdf_hist_normalized.bin_counts);
        Ok(())
    }

    #[test]
    fn rdf_hyperboloid_periodic() -> Result<(), Box<dyn std::error::Error>> {
        const EIGHTEIGHT: f64 = 2.448_452_447_678_076;
        let boundary = Periodic::new(1.0, EightEight { skirt: 1.0_f64 })?;
        let microstate = MicrostateBuilder::with_boundary(boundary)
            .bodies([
                Body::point(Hyperboloid::<3>::from_polar(EIGHTEIGHT - 0.2, 0.0, 1.0)),
                Body::point(Hyperboloid::<3>::from_polar(EIGHTEIGHT - 0.25, 0.0, 1.0)),
                Body::point(Hyperboloid::<3>::from_polar(
                    1.8,
                    0.01 + PI * 3.0 / 4.0,
                    1.0,
                )),
            ])
            .try_build()
            .expect("hard coded distribution should be valid");
        let rdf_hist = SpatialHistogram::<1, f64>::rdf(&microstate, 0.0_f64, 1.0_f64, 2_usize)?;
        let ans = array![4_usize, 2_usize];
        assert_eq!(ans, rdf_hist.bin_counts);
        assert_eq!(rdf_hist.bin_edges.slice(s![0, ..]), array![0.0, 0.5, 1.0]);

        let rdf_hist_normalized = NormalizedHistogram::normalize(&rdf_hist);
        let ans_normed = array![2.0 / 3.0, 1.0 / 3.0];
        assert_eq!(ans_normed, rdf_hist_normalized.bin_counts);
        Ok(())
    }
}

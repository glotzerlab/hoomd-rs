// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement order parameters relating to the density of the system.

use hoomd_geometry::shape::{EightEight, Hypercuboid};
use hoomd_manifold::{Hyperbolic, Spherical};
use hoomd_microstate::{
    Microstate, Transform,
    boundary::{GenerateGhosts, MaximumAllowableInteractionRange, Open, Periodic},
    property::{Point, Position},
};
use hoomd_vector::{Cartesian, Metric};
use ndarray::prelude::*;
use thiserror::Error;

///  Struct for creating and manipulating histograms.
///
/// `N` specifies the dimension of the histogram bins (may be 1, 2 or 3), `C`
/// is the boundary condition of the data (e.g. `Open`, `Periodic`), and `A`
/// is the type for the data itself. `A` must be able to implement `Add`
/// and `PartialOrd`.
// The default output `bin_counts` is an array of the frequencies for each of
/// the bins and is stored as the type `usize`.
///
/// ```
/// use hoomd_microstate::{Microstate, property::Position};
/// use hoomd_order::{GenerateHistogram, SpatialHistogram};
/// use hoomd_vector::Metric;
/// use ndarray::prelude::*;
///
/// let numbers = vec![
///     [1],
///     [2],
///     [4],
///     [10],
///     [11],
///     [12],
///     [14],
///     [20],
///     [21],
///     [22],
///     [23],
/// ];
/// let bin_edges = array![
///     [0_usize, 10_usize, 20_usize, 30_usize],
///     [0_usize, 0_usize, 0_usize, 0_usize]
/// ];
/// let bounds = [[0_usize, 30_usize]; 1];
/// let hist = SpatialHistogram::<1, usize>::histogram(
///     &numbers,
///     bin_edges,
///     bounds,
///     [3_usize],
/// );
/// let ans = array![4_usize, 4_usize, 3_usize];
/// assert_eq!(ans, hist.bin_counts);
/// ```
pub struct SpatialHistogram<const N: usize, A> {
    /// A vector containing the bin edges of the histogram.
    pub bin_edges: Array<A, Dim<[usize; 2]>>,
    /// An array containing the upper and lower bounds of the histogram.
    pub bounds: [[A; 2]; N],
    /// The bin counts in the histogram.
    pub bin_counts: Array<usize, Dim<[usize; N]>>,
    /// The number of bins in each dimension.
    pub n_bins: [usize; N],
}

/// A one-dimensional histogram storing data of type `f64`.
pub struct FloatHistogram {
    /// a vector containing the bin edges of the histogram
    pub bin_edges: Array<f64, Dim<[usize; 1]>>,
    /// an array containing the upper and lower bounds of the histogram
    pub bounds: [f64; 2],
    /// the bin counts in the histogram
    pub bin_counts: Array<f64, Dim<[usize; 1]>>,
    /// number of bins in each dimension
    pub n_bins: usize,
}

impl FloatHistogram {
    /// Normalize the 1D histogram.
    #[inline]
    fn normalize_rdf(
        histogram: &SpatialHistogram<1, f64>,
        weights: &Array<f64, Dim<[usize; 1]>>,
    ) -> FloatHistogram {
        //         let sum = histogram
        // .bin_counts
        // .iter()
        // .fold(0.0_f64, |sum, x| sum + *x as f64);
        let normed_counts: Vec<f64> = histogram
            .bin_counts
            .iter()
            .zip(weights.iter())
            .map(|(count, weight)| *count as f64 * *weight)
            .collect();
        let n_bins = histogram.n_bins[0];
        let bounds = histogram.bounds[0];
        let bin_edges: Array<f64, Dim<[usize; 1]>> = histogram.bin_edges.row(0).to_owned();
        FloatHistogram {
            bin_edges,
            bounds,
            bin_counts: Array::from_vec(normed_counts),
            n_bins,
        }
    }
}

/// Compute a histogram with `N` dimensional data of type `A` which implements `Add`
/// and `PartialOrd`
pub trait GenerateHistogram<const N: usize, A> {
    /// Generate a histogram from a given microstate.
    fn histogram(
        data: &[[A; N]],
        bin_edges: Array<A, Dim<[usize; 2]>>,
        bounds: [[A; 2]; N],
        nbins: [usize; N],
    ) -> SpatialHistogram<N, A>;
}

/// Correlation functions from a microstate.
pub trait CorrelationFunction<B, S, X, C, P>
where
    P: Metric + ShellMeasure,
{
    /// Computes the raw historgram of distances between sites in a given microstate
    /// TODO:
    /// # Errors
    fn radial_distance_histogram(
        microstate: &Microstate<B, S, X, C>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<SpatialHistogram<1, f64>, Error>;
    /// Get the radial distribution function g(r) from a given microstate.
    /// # Errors
    #[inline]
    fn rdf(
        microstate: &Microstate<B, S, X, C>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
        density: f64,
    ) -> Result<FloatHistogram, Error> {
        let unnormed_rdf = Self::radial_distance_histogram(microstate, r_min, r_max, nbins)?;
        let number_of_particles = microstate
            .sites()
            .iter()
            .fold(0.0, |sum, _site| sum + 1.0_f64);
        let weights: Array<f64, Dim<[usize; 1]>> = unnormed_rdf
            .bin_edges()
            .row(0)
            .windows(2)
            .into_iter()
            .map(|bin_window| {
                let (r0, r1) = (bin_window[0], bin_window[1]);
                let width = r1 - r0;
                let shell_measure = <P as ShellMeasure>::shell_measure(r0 + width / 2.0, width);
                2.0 / number_of_particles / density / shell_measure
            })
            .collect();
        Ok(FloatHistogram::normalize_rdf(&unnormed_rdf, &weights))
    }
}

/// TODO: documentation
pub trait ShellMeasure {
    /// Get the volume of a shell of radius `r`.
    fn shell_measure(r: f64, shell_width: f64) -> f64;
}

impl ShellMeasure for Cartesian<2> {
    fn shell_measure(r: f64, shell_width: f64) -> f64 {
        2.0 * std::f64::consts::PI * r * shell_width
    }
}

impl ShellMeasure for Cartesian<3> {
    fn shell_measure(r: f64, shell_width: f64) -> f64 {
        4.0 * std::f64::consts::PI * r * r * shell_width
    }
}

impl ShellMeasure for Hyperbolic<3> {
    fn shell_measure(r: f64, shell_width: f64) -> f64 {
        let sinh_r = r.sinh();
        2.0 * std::f64::consts::PI * sinh_r * shell_width
    }
}

impl ShellMeasure for Spherical<3> {
    fn shell_measure(r: f64, shell_width: f64) -> f64 {
        let sin_r = r.sin();
        2.0 * std::f64::consts::PI * sin_r * shell_width
    }
}

impl ShellMeasure for Spherical<4> {
    fn shell_measure(r: f64, shell_width: f64) -> f64 {
        let sin_r = r.sin();
        4.0 * std::f64::consts::PI * sin_r * sin_r * shell_width
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
    /// A 2D array with the bin edges of the histogram. Each row gives the edges along
    /// one of the axes.
    #[inline]
    pub fn bin_edges(&self) -> &Array<A, Dim<[usize; 2]>> {
        &self.bin_edges
    }
    /// The lower and upper bounds of the histogram for each of the axes.
    #[inline]
    pub fn bounds(&self) -> &[[A; 2]; N] {
        &self.bounds
    }
    /// The frequency counts for each of the bins.
    #[inline]
    pub fn bin_counts(&self) -> &Array<usize, Dim<[usize; N]>> {
        &self.bin_counts
    }
}

impl<A> GenerateHistogram<1, A> for SpatialHistogram<1, A>
where
    A: std::ops::Add<Output = A> + std::cmp::PartialOrd,
{
    /// Create a one-dimensional histogram.
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

impl<B, S, X, P> CorrelationFunction<B, S, X, Open, P> for SpatialHistogram<1, f64>
where
    S: Position<Position = P>,
    P: Metric + ShellMeasure,
{
    /// Calculate the histogram of distances between sites in a microstate with
    /// open boundary conditions.
    #[inline]
    fn radial_distance_histogram(
        microstate: &Microstate<B, S, X, Open>,
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

impl<B, S, X> CorrelationFunction<B, S, X, Periodic<Hypercuboid<2>>, Cartesian<2>>
    for SpatialHistogram<1, f64>
where
    S: Position<Position = Cartesian<2>> + Copy + Default,
    B: Transform<S> + Position<Position = Cartesian<2>> + Copy,
{
    /// Calculate the radial distribution function (RDF), g(r), for a given microstate
    /// with periodic boundary conditions.
    #[inline]
    fn radial_distance_histogram(
        microstate: &Microstate<B, S, X, Periodic<Hypercuboid<2>>>,
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

        let max_boundary = Periodic::new(boundary_max, microstate.boundary().shape().clone())
            .expect("copy of valid boundary");
        let new_microstate = Microstate::builder()
            .boundary(max_boundary)
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

impl<B, S, X> CorrelationFunction<B, S, X, Periodic<Hypercuboid<3>>, Cartesian<3>>
    for SpatialHistogram<1, f64>
where
    S: Position<Position = Cartesian<3>> + Copy + Default,
    B: Transform<S> + Position<Position = Cartesian<3>> + Copy,
{
    /// Calculate the radial distribution function (RDF), g(r), for a given microstate
    /// with periodic boundary conditions.
    #[inline]
    fn radial_distance_histogram(
        microstate: &Microstate<B, S, X, Periodic<Hypercuboid<3>>>,
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

        let max_boundary = Periodic::new(boundary_max, microstate.boundary().shape().clone())
            .expect("copy of valid boundary");
        let new_microstate = Microstate::builder()
            .boundary(max_boundary)
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

impl<X>
    CorrelationFunction<
        Point<Hyperbolic<3>>,
        Point<Hyperbolic<3>>,
        X,
        Periodic<EightEight>,
        Hyperbolic<3>,
    > for SpatialHistogram<1, f64>
{
    /// Calculate the radial distribution function (RDF), g(r), for a given microstate
    /// with periodic boundary conditions.
    #[inline]
    fn radial_distance_histogram(
        microstate: &Microstate<
            Point<Hyperbolic<3>>,
            Point<Hyperbolic<3>>,
            X,
            Periodic<EightEight>,
        >,
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

        let max_boundary = Periodic::new(boundary_max, EightEight {}).expect("hard coded");
        let new_microstate = Microstate::builder()
            .boundary(max_boundary)
            .bodies(microstate.bodies().iter().map(|b| b.clone().item))
            .try_build()
            .expect("copy of existing valid microstate");
        let mut all_ghosts: Vec<Hyperbolic<3>> = vec![];
        for site_b in new_microstate.sites() {
            let ghosts =
                GenerateGhosts::generate_ghosts(new_microstate.boundary(), &site_b.properties);
            for ghost in ghosts {
                all_ghosts.push(*ghost.position());
            }
        }
        let mut everyone_else: Vec<Hyperbolic<3>> = microstate
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
    /// Create a two-dimensional histogram.
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
    /// Create a three-dimensional histogram.
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
    use hoomd_manifold::{Hyperbolic, Minkowski};
    use hoomd_microstate::{Body, boundary::Open};
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
        let mut microstate = Microstate::builder()
            .boundary(Open)
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
        let rdf_hist = SpatialHistogram::<1, f64>::radial_distance_histogram(
            &microstate,
            0.0_f64,
            2.0_f64,
            2_usize,
        )?;
        let ans = array![4_usize, 2_usize];
        assert_eq!(ans, rdf_hist.bin_counts);
        assert_eq!(rdf_hist.bin_edges.slice(s![0, ..]), array![0.0, 1.0, 2.0]);

        // TODO: fix!!
        // let rdf_hist_normalized = FloatHistogram::normalize_rdf(&rdf_hist);
        // let ans_normed = array![2.0 / 3.0, 1.0 / 3.0];
        // assert_eq!(ans_normed, rdf_hist_normalized.bin_counts);
        Ok(())
    }

    #[test]
    fn rdf_cartesian_square_periodic() -> Result<(), Box<dyn std::error::Error>> {
        let boundary = Periodic::new(
            1.0,
            Hypercuboid::<2>::with_equal_edges(2.0.try_into().expect("hard-coded positive number")),
        )
        .expect("no interactions");
        let microstate = Microstate::builder()
            .boundary(boundary)
            .bodies([
                Body::point(Cartesian::from([-0.5, 0.8])),
                Body::point(Cartesian::from([0.75, 0.8])),
                Body::point(Cartesian::from([-0.5, -0.8])),
            ])
            .try_build()
            .expect("hard-coded distributions should be valid");

        let rdf_hist = SpatialHistogram::<1, f64>::radial_distance_histogram(
            &microstate,
            0.0_f64,
            1.0_f64,
            2_usize,
        )?;
        let ans = array![2_usize, 4_usize];
        assert_eq!(ans, rdf_hist.bin_counts);
        assert_eq!(rdf_hist.bin_edges.slice(s![0, ..]), array![0.0, 0.5, 1.0]);

        // TODO: Fix examples!!!!!
        // let rdf_hist_normalized = FloatHistogram::normalize(&rdf_hist);
        // let ans_normed = array![1.0 / 3.0, 2.0 / 3.0];
        // assert_eq!(ans_normed, rdf_hist_normalized.bin_counts);
        Ok(())
    }

    #[test]
    fn rdf_hyperbolic() -> Result<(), Box<dyn std::error::Error>> {
        let microstate = Microstate::builder()
            .boundary(Open)
            .bodies([
                Body::point(Hyperbolic::from_minkowski_coordinates(Minkowski::from([
                    1.0,
                    0.0,
                    2.0_f64.sqrt(),
                ]))),
                Body::point(Hyperbolic::from_minkowski_coordinates(Minkowski::from([
                    2.0,
                    0.0,
                    5.0_f64.sqrt(),
                ]))),
                Body::point(Hyperbolic::from_minkowski_coordinates(Minkowski::from([
                    1.0,
                    1.0,
                    3.0_f64.sqrt(),
                ]))),
                Body::point(Hyperbolic::from_minkowski_coordinates(Minkowski::from([
                    2.0,
                    1.0,
                    6.0_f64.sqrt(),
                ]))),
            ])
            .try_build()
            .expect("hard coded distribution should be valid");
        let rdf_hist = SpatialHistogram::<1, f64>::radial_distance_histogram(
            &microstate,
            0.0_f64,
            2.0_f64,
            4_usize,
        )?;
        let ans = array![0_usize, 5_usize, 1_usize, 0_usize];
        assert_eq!(ans, rdf_hist.bin_counts);
        assert_eq!(
            rdf_hist.bin_edges.slice(s![0, ..]),
            array![0.0, 0.5, 1.0, 1.5, 2.0]
        );

        // let rdf_hist_normalized = FloatHistogram::normalize(&rdf_hist);
        // let ans_normed = array![0.0, 5.0 / 6.0, 1.0 / 6.0, 0.0];
        // assert_eq!(ans_normed, rdf_hist_normalized.bin_counts);
        Ok(())
    }

    #[test]
    fn rdf_hyperbolic_periodic() -> Result<(), Box<dyn std::error::Error>> {
        const EIGHTEIGHT: f64 = 2.448_452_447_678_076;
        let boundary = Periodic::new(1.0, EightEight {})?;
        let microstate = Microstate::builder()
            .boundary(boundary)
            .bodies([
                Body::point(Hyperbolic::<3>::from_polar_coordinates(
                    EIGHTEIGHT - 0.2,
                    0.0,
                )),
                Body::point(Hyperbolic::<3>::from_polar_coordinates(
                    EIGHTEIGHT - 0.25,
                    0.0,
                )),
                Body::point(Hyperbolic::<3>::from_polar_coordinates(
                    1.8,
                    0.01 + PI * 3.0 / 4.0,
                )),
            ])
            .try_build()
            .expect("hard coded distribution should be valid");
        let rdf_hist = SpatialHistogram::<1, f64>::radial_distance_histogram(
            &microstate,
            0.01_f64,
            1.01_f64,
            2_usize,
        )?;
        let ans = array![4_usize, 2_usize];
        assert_eq!(ans, rdf_hist.bin_counts);
        assert_eq!(
            rdf_hist.bin_edges.slice(s![0, ..]),
            array![0.01, 0.51, 1.01]
        );

        // let rdf_hist_normalized = FloatHistogram::normalize(&rdf_hist);
        // let ans_normed = array![2.0 / 3.0, 1.0 / 3.0];
        // assert_eq!(ans_normed, rdf_hist_normalized.bin_counts);
        Ok(())
    }
}

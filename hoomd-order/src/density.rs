// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement various methods relating to the density of the system, namely radial distribution functions and correlation functions.
The methods are based on the struct `SpatialHistogram`.
*/

#![allow(dead_code)]
use hoomd_microstate::{Microstate, boundary::Open, property::Position};
use hoomd_vector::Metric;
use ndarray::prelude::*;

/** Struct for creating and manipulating histograms. `N` specifies the dimension of the histogram bins (may be 1, 2 or 3), `C`
is the boundary condition of the data (e.g. `Open`, periodic), and `A` is the type for the data itself. `A` must be able to implement `Add`
 and `PartialOrd`.

The default output `bin_counts` is an array of the frequencies for each of the bins and is stored as the type `usize`.

```
use hoomd_microstate::{Microstate, boundary::Open, property::Position};
use hoomd_order::SpatialHistogram;
use hoomd_vector::Metric;
use ndarray::prelude::*;

let numbers = (0..51).collect::<Vec<usize>>();
let bin_edges = array![
    [0_usize, 10_usize, 20_usize, 30_usize, 40_usize, 50_usize],
    [0_usize, 0_usize, 0_usize, 0_usize, 0_usize, 0_usize]
];
let bounds = [[0_usize, 50_usize]; 1];
let hist =
    SpatialHistogram::<1, Open, usize>::histogram_1d(&numbers, bin_edges, bounds, 5_usize);
let ans = array![10_usize, 10_usize, 10_usize, 10_usize, 10_usize];
assert_eq!(ans, hist.bin_counts);
```
*/
pub struct SpatialHistogram<const N: usize, C, A> {
    /// a vector containing the bin edges of the histogram
    pub bin_edges: Array<A, Dim<[usize; 2]>>,
    /// an array containing the upper and lower bounds of the histogram
    pub bounds: [[A; 2]; N],
    /// the simulation box
    pub boundary: C,
    /// the bin counts in the histogram
    pub bin_counts: Array<usize, Dim<[usize; N]>>,
    /// number of bins in each dimension
    pub n_bins: [usize; N],
}

impl<const N: usize, C, A> SpatialHistogram<N, C, A> {
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

impl<A> SpatialHistogram<1, Open, A>
where
    A: std::ops::Add<Output = A> + std::cmp::PartialOrd,
{
    /// Create a one-dimensional histogram
    #[inline]
    pub fn histogram_1d(
        data: &[A],
        bin_edges: Array<A, Dim<[usize; 2]>>,
        bounds: [[A; 2]; 1],
        nbins: usize,
    ) -> Self {
        let mut counts = Array::zeros(nbins);
        for n in 0..nbins {
            counts[n] = data
                .iter()
                .filter(|a| **a > bin_edges[[0, n]] && **a <= bin_edges[[0, n + 1]])
                .count();
        }
        SpatialHistogram {
            bin_edges,
            bounds,
            boundary: Open,
            bin_counts: counts,
            n_bins: [nbins; 1],
        }
    }
}

#[allow(clippy::panic)]
impl SpatialHistogram<1, Open, f64> {
    /// Calculate the radial distribution function (RDF), g(r), for a given microstate.
    /// # Panics
    /// Function panics when it is called on an empty microstate
    #[must_use]
    #[inline]
    pub fn rdf<B, S, M>(
        microstate: &Microstate<B, S, Open>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Self
    where
        S: Position<Metric = M>,
        M: Metric,
    {
        let bin_size: f64 = (r_max - r_min) / (nbins as f64);
        let bin_edges_arr = Array::from_vec(
            (0..=nbins)
                .collect::<Vec<usize>>()
                .iter()
                .map(|i| (*i as f64) * bin_size + r_min)
                .collect::<Vec<f64>>(),
        );
        let dummy_row = Array::from_vec(
            (0..=nbins)
                .collect::<Vec<usize>>()
                .iter()
                .map(|i| *i as f64)
                .collect::<Vec<f64>>(),
        );
        let bin_edges: Array<f64, Dim<[usize; 2]>> =
            ndarray::stack![Axis(0), bin_edges_arr, dummy_row];
        let mut distances: Vec<f64> = vec![];
        for site_1 in microstate.site_indices() {
            for site_2 in microstate.site_indices() {
                match site_1 {
                    Some(site_1_index) => match site_2 {
                        Some(site_2_index) => {
                            if site_1_index > site_2_index {
                                distances.push(
                                    microstate.sites()[*site_1_index]
                                        .properties
                                        .position()
                                        .distance(
                                            microstate.sites()[*site_2_index].properties.position(),
                                        ),
                                );
                            }
                        }
                        None => panic!("given microstate is empty"),
                    },
                    None => panic!("given microstate is empty"),
                }
            }
        }
        SpatialHistogram::<1, Open, f64>::histogram_1d(
            &distances,
            bin_edges,
            [[r_min, r_max]; 1],
            nbins,
        )
    }
//     Computes the pairwise correlation function between two sets of points p_1 and p_2 with associated values s_1 and s_2, respectively.
//    math
//    C(r) = \langle s_1(0)\cdot s_2(r)\rangle
//    
//    
//    #[must_use]
//    #[inline]
//    pub fn correlation_function<B, S, M>(
//        points: &Microstate<B, S, Open>,
//        query_points: &Microstate<B, S, Open>,
//        body_trait: B,
//        r_min: f64,
//        r_max: f64,
//        nbins: usize,
//    ) -> Self 
//    where
//        S: Position<Metric = M>,
//        M: Metric,
//    {
//        let bin_size: f64 = (r_max - r_min) / (nbins as f64);
//        let bin_edges_arr = Array::from_vec(
//            (0..=nbins)
//                .collect::<Vec<usize>>()
//                .iter()
//                .map(|i| (*i as f64) * bin_size + r_min)
//                .collect::<Vec<f64>>(),
//        );
//        let dummy_row = Array::from_vec(
//            (0..=nbins)
//                .collect::<Vec<usize>>()
//                .iter()
//                .map(|i| *i as f64)
//                .collect::<Vec<f64>>(),
//        );
//        let bin_edges: Array<f64, Dim<[usize; 2]>> =
//            ndarray::stack![Axis(0), bin_edges_arr, dummy_row];
//        let mut correlations: Vec<f64> = vec![];
//
//        SpatialHistogram::<1, Open, f64>::histogram_1d(
//            &correlations,
//            bin_edges,
//            [[r_min, r_max]; 1],
//            nbins,
//        )
//    }
}

impl<A> SpatialHistogram<2, Open, A>
where
    A: std::ops::Add<Output = A> + std::cmp::PartialOrd,
{
    /// Create a two-dimensional histogram
    #[inline]
    pub fn histogram_2d(
        data: &[[A; 2]],
        bin_edges: Array<A, Dim<[usize; 2]>>,
        bounds: [[A; 2]; 2],
        n_bins: [usize; 2],
    ) -> Self {
        let mut counts = Array::zeros((n_bins[0], n_bins[1]));
        for i in 0..n_bins[0] {
            for j in 0..n_bins[1] {
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
            boundary: Open,
            bin_counts: counts,
            n_bins,
        }
    }
}

impl<A> SpatialHistogram<3, Open, A>
where
    A: std::ops::Add<Output = A> + std::cmp::PartialOrd,
{
    /// Create a three-dimensional histogram
    #[inline]
    pub fn histogram_3d(
        data: &[[A; 3]],
        bin_edges: Array<A, Dim<[usize; 2]>>,
        bounds: [[A; 2]; 3],
        n_bins: [usize; 3],
    ) -> Self {
        let mut counts = Array::zeros((n_bins[0], n_bins[1], n_bins[2]));
        for i in 0..n_bins[0] {
            for j in 0..n_bins[1] {
                for k in 0..n_bins[2] {
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
            boundary: Open,
            bin_counts: counts,
            n_bins,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_manifold::{Hyperboloid, Minkowski};
    use hoomd_microstate::{Body, MicrostateBuilder, boundary::Open, property::Point};
    use hoomd_vector::Cartesian;
    use std::iter;

    #[test]

    fn linear_1d_histogram_usize() {
        let numbers = (0..51).collect::<Vec<usize>>();
        let bin_edges = array![
            [0_usize, 10_usize, 20_usize, 30_usize, 40_usize, 50_usize],
            [0_usize, 0_usize, 0_usize, 0_usize, 0_usize, 0_usize]
        ];
        let bounds = [[0_usize, 50_usize]; 1];
        let hist =
            SpatialHistogram::<1, Open, usize>::histogram_1d(&numbers, bin_edges, bounds, 5_usize);
        let ans = array![10_usize, 10_usize, 10_usize, 10_usize, 10_usize];
        assert_eq!(ans, hist.bin_counts);
    }
    #[test]
    fn linear_1d_histogram_f64() {
        let numbers: Vec<f64> = iter::successors(Some(0.0_f64), |&i| {
            let next = i + 0.5_f64;
            if next <= 10.0_f64 { Some(next) } else { None }
        })
        .collect();
        let bin_edges = array![
            [0.0_f64, 2.5_f64, 5.0_f64, 7.5_f64, 10.0_f64],
            [0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64]
        ];
        let bounds = [[0.0_f64, 10.0_f64]; 1];
        let hist =
            SpatialHistogram::<1, Open, f64>::histogram_1d(&numbers, bin_edges, bounds, 4_usize);
        let ans = array![5_usize, 5_usize, 5_usize, 5_usize];
        assert_eq!(ans, hist.bin_counts);
    }
    #[test]
    fn linear_2d_histogram_usize() {
        let numbers: Vec<[usize; 2]> = vec![[1, 1], [3, 1], [1, 4], [1, 5]];
        let bin_edges = array![[0_usize, 2_usize, 4_usize], [0_usize, 3_usize, 6_usize]];
        let bounds = [[0_usize, 4_usize], [0_usize, 6_usize]];
        let hist =
            SpatialHistogram::<2, Open, usize>::histogram_2d(&numbers, bin_edges, bounds, [2, 2]);
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
        let hist =
            SpatialHistogram::<2, Open, f64>::histogram_2d(&numbers, bin_edges, bounds, [2, 2]);
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
        let hist = SpatialHistogram::<3, Open, usize>::histogram_3d(
            &numbers,
            bin_edges,
            bounds,
            [2, 2, 2],
        );
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
        let hist =
            SpatialHistogram::<3, Open, f64>::histogram_3d(&numbers, bin_edges, bounds, [2, 2, 2]);
        let ans = array![
            [[1_usize, 1_usize], [0_usize, 0_usize]],
            [[0_usize, 0_usize], [0_usize, 1_usize]]
        ];
        assert_eq!(ans, hist.bin_counts);
    }

    #[test]
    fn rdf_cartesian_square() {
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
        let rdf_hist = SpatialHistogram::<1, Open, f64>::rdf::<
            Point<Cartesian<2>>,
            Point<Cartesian<2>>,
            Cartesian<2>,
        >(&microstate, 0.0_f64, 2.0_f64, 2_usize);
        let ans = array![4_usize, 2_usize];
        assert_eq!(ans, rdf_hist.bin_counts);
        assert_eq!(rdf_hist.bin_edges.slice(s![0, ..]), array![0.0, 1.0, 2.0]);
    }

    #[test]
    fn rdf_hyperboloid() {
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
        let rdf_hist = SpatialHistogram::<1, Open, f64>::rdf::<
            Point<Hyperboloid<3>>,
            Point<Hyperboloid<3>>,
            Hyperboloid<3>,
        >(&microstate, 0.0_f64, 2.0_f64, 4_usize);
        let ans = array![0_usize, 5_usize, 1_usize, 0_usize];
        assert_eq!(ans, rdf_hist.bin_counts);
        assert_eq!(
            rdf_hist.bin_edges.slice(s![0, ..]),
            array![0.0, 0.5, 1.0, 1.5, 2.0]
        );
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//TODO: documentation 

/*! Implement various methods relating to the density of the system
*/
use hoomd_microstate::{Microstate, boundary::Open, property::Position};
use ndarray::prelude::*;

pub struct SpatialHistogram<const N: usize, C, A> {
    /// a vector containing the bin edges of the histogram
    pub bin_edges: Array<A, Dim<[usize;2]>>,
    /// an array containing the upper and lower bounds of the histogram
    pub bounds: [[A; 2];N],
    /// the simulation box
    pub boundary: C,
    /// the bin counts in the histogram
    pub bin_counts: Array<usize, Dim<[usize;N]>>,
    /// number of bins in each dimension
    pub n_bins: [usize; N],
}

impl<const N: usize, C, A> SpatialHistogram<N, C, A> 
{
    pub fn bin_edges(&self) -> &Array<A, Dim<[usize;2]>> {
        &self.bin_edges
    }
    pub fn bounds(&self) -> &[[A;2];N] {
        &self.bounds
    }
    pub fn bin_counts(&self) -> &Array<usize, Dim<[usize;N]>> {
        &self.bin_counts
    }
}

impl<A> SpatialHistogram<1, Open, A> 
where A: std::ops::Add<Output = A> + std::cmp::PartialOrd
{
    pub fn histogram1D(data: &Vec<A>, bin_edges: Array<A, Dim<[usize;2]>>, bounds: [[A;2];1], nbins: usize) -> Self {
        let mut counts =  Array::zeros(nbins);
        for n in 0..nbins {
            let bin_num = data.iter()
                .filter(|a| {**a>bin_edges[[0,n]] && **a<= bin_edges[[0,n+1]]})
                .collect::<Vec<&A>>();
                //.fold(A::default(), |sum, x| sum+x.clone());
            counts[n] = bin_num.len();
        }
        SpatialHistogram{
            bin_edges : bin_edges,
            bounds: bounds,
            boundary: Open,
            bin_counts: counts,
            n_bins : [nbins;1],
        }
    }
}

//TODO: need to figure out some way to pass relevant distance to rdf
impl SpatialHistogram<1, Open, f64> {
    pub fn RDF<V>(microstate: &Microstate<>, r_min: f64, r_mas: f64, n_bins : usize) -> Self {
        let bin_size: f64 = (r_max - r_min)/(n_bins as f64);
        let bin_edges_arr: [f64; n_bins] = array::from_fn(|i| i*bin_size + r_min);
        let bin_edges: Array<f64, Dim<[usize;2]>> = array![[bin_edges_arr]];
        let pos_vec =  //Vec::from(microstate.sites().properties.position().coordinates);
        histogram1D(&distances, bin_edges, [[r_min,r_max];1], n_bins = n_bins)
    }
}

impl<A> SpatialHistogram<2, Open, A> 
where A: std::ops::Add<Output = A> + std::cmp::PartialOrd
{
    pub fn histogram2D(data: &Vec<[A;2]>, bin_edges: Array<A, Dim<[usize;2]>>, bounds: [[A;2];2], n_bins: [usize; 2] ) -> Self {
        let mut counts =  Array::zeros((n_bins[0],n_bins[1]));
        for i in 0..n_bins[0] {
            for j in 0..n_bins[1] {
                let bin_num = data.iter()
                .filter(|[a,b]| {*a > bin_edges[[0,i]] && 
                                       *a <= bin_edges[[0,i+1]] &&
                                       *b > bin_edges[[1, j]] && 
                                       *b <= bin_edges[[1,j+1]]})
                .collect::<Vec<&[A;2]>>();
                counts[[i,j]] = bin_num.len();
            }
        }
        SpatialHistogram{
            bin_edges : bin_edges,
            bounds: bounds,
            boundary: Open,
            bin_counts: counts,
            n_bins : n_bins,
        }
    }
}

impl<A> SpatialHistogram<3, Open, A> 
where A: std::ops::Add<Output = A> + std::cmp::PartialOrd
{
    pub fn histogram2D(data: &Vec<[A;3]>, bin_edges: Array<A, Dim<[usize;2]>>, bounds: [[A;2];3], n_bins: [usize; 3] ) -> Self {
        let mut counts =  Array::zeros((n_bins[0],n_bins[1], n_bins[2]));
        for i in 0..n_bins[0] {
            for j in 0..n_bins[1] {
                for k in 0..n_bins[2] {
                        let bin_num = data.iter()
                        .filter(|[a,b,c]| {*a > bin_edges[[0,i]] && 
                                            *a <= bin_edges[[0,i+1]] &&
                                            *b > bin_edges[[1, j]] && 
                                            *b <= bin_edges[[1,j+1]] &&
                                            *c > bin_edges[[2, k]] && 
                                            *c <= bin_edges[[2,k+1]]})
                        .collect::<Vec<&[A;3]>>();
                        counts[[i,j,k]] = bin_num.len();
                }
            }
        }
        SpatialHistogram{
            bin_edges : bin_edges,
            bounds: bounds,
            boundary: Open,
            bin_counts: counts,
            n_bins : n_bins,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_microstate::boundary::Open;
    use approx::assert_relative_eq;
    use paste::paste;
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::rstest;
    use std::iter;

    #[test]

    fn linear_1d_histogram_usize() {
        let numbers = (0..51).collect::<Vec<usize>>();
        let bin_edges = array![[0_usize, 10_usize, 20_usize, 30_usize, 40_usize, 50_usize],
                                                                [0_usize,0_usize,0_usize,0_usize,0_usize,0_usize]];
        let bounds = [[0_usize, 50_usize];1];
        let hist = SpatialHistogram::<1,Open,usize>::histogram1D(&numbers, bin_edges, bounds, 5 as usize);
        let ans = array![10_usize, 10_usize, 10_usize, 10_usize,10_usize];
        assert_eq!(ans, hist.bin_counts);
    }
    #[test]
    fn linear_1d_histogram_f64() {
        let numbers: Vec<f64> = iter::successors(Some(0.0_f64), |&i| {
        let next = i + 0.5_f64;
        if next <= 10.0_f64 {
            Some(next)
        } else {
            None
        }
        })
        .collect();
        let bin_edges = array![[0.0_f64, 2.5_f64, 5.0_f64, 7.5_f64, 10.0_f64],
                                                            [0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64]];
        let bounds = [[0.0_f64, 10.0_f64];1];
        let hist = SpatialHistogram::<1,Open,f64>::histogram1D(&numbers, bin_edges, bounds, 4 as usize);
        let ans = array![5_usize, 5_usize, 5_usize, 5_usize];
        assert_eq!(ans, hist.bin_counts);
    }
    #[test]
    fn linear_2d_histogram_usize() {
        let numbers : Vec<[usize;2]> = vec![[1,1],[3,1],[1,4],[1,5]];
        let bin_edges = array![[0_usize, 2_usize, 4_usize],
                                                                [0_usize, 3_usize, 6_usize]];
        let bounds = [[0_usize, 4_usize],[0_usize, 6_usize]];
        let hist = SpatialHistogram::<2, Open, usize>::histogram2D(&numbers, bin_edges, bounds, [2,2]);
        let ans = array![[1_usize,2_usize],[1_usize,0_usize]];
        assert_eq!(ans, hist.bin_counts);
    }
    #[test]
    fn linear_2d_histogram_float() {
        let numbers : Vec<[f64;2]> = vec![[0.25,0.25],[0.25,0.5],[0.5,0.5],[1.5,0.5],[1.5,1.5]];
        let bin_edges = array![[0.0_f64, 1_f64, 2_f64],
                                                                [0_f64, 1_f64, 2_f64]];
        let bounds = [[0.0_f64, 2.0_f64],[0.0_f64, 2.0_f64]];
        let hist = SpatialHistogram::<2, Open, f64>::histogram2D(&numbers, bin_edges, bounds, [2,2]);
        let ans = array![[3_usize,0_usize],[1_usize,1_usize]];
        assert_eq!(ans, hist.bin_counts);
    }
    #[test]
    fn linear_3d_histogram_usize() {
        let numbers : Vec<[usize;3]> = vec![[1,3,1],[3,1,1],[1,1,3],[3,3,3]];
        let bin_edges = array![[0_usize, 2_usize, 4_usize],
                                                               [0_usize, 2_usize, 4_usize],
                                                               [0_usize, 2_usize, 4_usize]];
        let bounds = [[0_usize, 4_usize],[0_usize, 4_usize], [0_usize, 4_usize]];
        let hist = SpatialHistogram::<3, Open, usize>::histogram2D(&numbers, bin_edges, bounds, [2,2,2]);
        let ans = array![[[0_usize,1_usize],[1_usize,0_usize]],[[1_usize,0_usize],[0_usize,1_usize]]];
        assert_eq!(ans, hist.bin_counts);
    }
    #[test]
    fn linear_3d_histogram_f64() {
        let numbers : Vec<[f64;3]> = vec![[0.5,0.5,0.5],[0.5,0.5,1.5],[1.5,1.5,1.5]];
        let bin_edges = array![[0_f64, 1_f64, 2_f64],
                                                               [0_f64, 1_f64, 2_f64],
                                                               [0_f64, 1_f64, 2_f64]];
        let bounds = [[0_f64, 2_f64],[0_f64, 2_f64], [0_f64, 2_f64]];
        let hist = SpatialHistogram::<3, Open, f64>::histogram2D(&numbers, bin_edges, bounds, [2,2,2]);
        let ans = array![[[1_usize,1_usize],[0_usize,0_usize]],[[0_usize,0_usize],[0_usize,1_usize]]];
        assert_eq!(ans, hist.bin_counts);
    }
}



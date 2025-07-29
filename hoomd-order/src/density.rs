// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//TODO: documentation 

/*! Implement various methods relating to the density of the system
*/

pub struct SpatialHistogram<const D: usize, C> {
    /// a vector containing the bin edges of the histogram
    pub bin_edges: Vec<[f64;D]>,
    /// an array containing the upper and lower bounds of the histogram
    pub bounds: [[f64,f64];D],
    /// the simulation box
    pub boundary: C,
}

impl<const D: usize, C> SpatialHistogram<D, C> {
    pub fn bin_edges(&self) -> &Vec<[f64;D]> {
        self.bin_edges
    }
    pub fn bounds(&self) -> &[[f64,f64];D] {
        self.bounds
    }
}

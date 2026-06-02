// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement neighbor-related order parameters.

use hoomd_manifold::Hyperbolic;
use hoomd_meshless_voronoi::NeighborList;
use hoomd_microstate::{Microstate, property::Position};
use hoomd_vector::{Cartesian, Metric};
use ndarray::prelude::*;
use num::complex::Complex;
use thiserror::Error;

/// Methods for comparing relative orientations of nearby bodies.
trait RelativeLocalOrientation<P> {
    /// Get the orientations of `points_i` relative to `point_j` in the local frame
    /// of `point_j`.
    fn relative_orientations(points_i: Vec<&P>, point_j: &P) -> Vec<f64>;
}

impl RelativeLocalOrientation<Cartesian<2>> for Cartesian<2> {
    /// Get the orientations of `points_i` relative to `point_j` in the local frame
    /// of `point_j` in two-dimensional Cartesian space.
    fn relative_orientations(points_i: Vec<&Cartesian<2>>, point_j: &Cartesian<2>) -> Vec<f64> {
        let points_i_transformed: Vec<Cartesian<2>> = points_i
            .iter()
            .map(|query_point| **query_point - *point_j)
            .collect();
        let angles: Vec<f64> = points_i_transformed
            .iter()
            .map(|pt_i| pt_i[1].atan2(pt_i[0]))
            .collect();
        angles
    }
}

impl RelativeLocalOrientation<Hyperbolic<3>> for Hyperbolic<3> {
    /// Get the orientations of `points_i` relative to `point_j` in the local frame
    /// of `point_j` in two-dimensional hyperbolic space.
    fn relative_orientations(points_i: Vec<&Hyperbolic<3>>, point_j: &Hyperbolic<3>) -> Vec<f64> {
        let boost = -(point_j.coordinates()[2].acosh());
        let rot = -point_j.coordinates()[1].atan2(point_j.coordinates()[0]);
        let points_i_transformed: Vec<Hyperbolic<3>> = points_i
            .iter()
            .map(|query_point| {
                let nn = query_point.coordinates();
                Hyperbolic::<3>::from_minkowski_coordinates(
                    [
                        nn[0] * (boost.cosh()) * (rot.cos()) - nn[1] * (boost.cosh()) * (rot.sin())
                            + nn[2] * (boost.sinh()),
                        nn[0] * (rot.sin()) + nn[1] * (rot.cos()),
                        nn[0] * (boost.sinh()) * (rot.cos()) - nn[1] * (boost.sinh()) * (rot.sin())
                            + nn[2] * (boost.cosh()),
                    ]
                    .into(),
                )
            })
            .collect();
        let angles: Vec<f64> = points_i_transformed
            .iter()
            .map(|pt_i| pt_i.coordinates()[1].atan2(pt_i.coordinates()[0]))
            .collect();
        angles
    }
}

/// Orientation-based local order parameters.
pub trait DirectorField<B, S, X, C, P> {
    /// Calculate the hexatic order
    /// ```math
    /// \psi_6(\vec{r}_i) = \frac{1}{N}\sum_{\text{neighbors }j}^{N}e^{i6\theta_{ij}}
    /// ```
    /// for a given site index belonging to a microstate.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSiteIndex`] when `site_index` is `None`
    ///
    /// [`Error::NoNearestNeighbors`] when
    /// `microstate.neighbors_of_site(site_index)` fails.
    fn hexatic_at_site(
        &self,
        microstate: &Microstate<B, S, X, C>,
        site_index: Option<usize>,
    ) -> Result<Complex<f64>, Error>;
    /// Compute the orientational correlation
    /// ```math
    /// G_6(|\vec{r}_i-\vec{r}_k|) = \langle \psi_6(\vec{r}_i\cdot\psi^*_6(\vec{r}_k))\rangle
    /// ```
    /// where $`\psi_6(\vec{r}_i)`$ is the complex-valued hexatic director at position $`\vec{r}_i`$
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSiteIndex`] when `site_index` is `None`
    ///
    /// [`Error::NoNearestNeighbors`] when
    /// `microstate.neighbors_of_site(site_index)` fails.
    fn orientational_correlation(
        &self,
        microstate: &Microstate<B, S, X, C>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<ComplexSpatialField, Error>;
}

/// Enumerate possible sources of error.
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

/// A one-dimensional complex-valued field. This struct is used for
/// complex-valued spatial data, e.g. orientational order.
pub struct ComplexSpatialField {
    /// An array containing the bin edges of the field.
    pub bin_edges: Array<f64, Dim<[usize; 1]>>,
    /// The lower and upper bounds of the field.
    pub bounds: [f64; 2],
    /// The value at each field point.
    pub field_value: Array<Complex<f64>, Dim<[usize; 1]>>,
    /// The number of field points.
    pub n_bins: usize,
}

impl<B, S, X, C, P> DirectorField<B, S, X, C, P> for NeighborList<'_, B, S, X, C>
where
    S: Position<Position = P>,
    P: RelativeLocalOrientation<P> + Metric,
{
    /// Compute the complex hexatic director field at a point from the microstate.
    #[inline]
    fn hexatic_at_site(
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
                // get the positions of neighbors in the query site frame
                let neighbor_coords: Vec<&P> = site_neighbors
                    .iter()
                    .map(|s| microstate.sites()[*s].properties.position())
                    .collect();
                let angles = <P as RelativeLocalOrientation<P>>::relative_orientations(
                    neighbor_coords,
                    point,
                );
                let hex: Complex<f64> = angles.iter().fold(Complex::new(0.0, 0.0), |sum, theta| {
                    sum + Complex::new(0.0, 6.0 * theta).exp()
                });
                Ok(hex.scale(1.0 / (site_neighbors.len() as f64)))
            }
            None => Err(Error::InvalidSiteIndex),
        }
    }
    /// Get a histogram of hexatic orders $`\psi_6`$ across all body sites in a
    /// given microstate.
    #[inline]
    fn orientational_correlation(
        &self,
        microstate: &Microstate<B, S, X, C>,
        r_min: f64,
        r_max: f64,
        nbins: usize,
    ) -> Result<ComplexSpatialField, Error> {
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
                            let dir1 = self.hexatic_at_site(microstate, *site_1)?;
                            let dir2 = self.hexatic_at_site(microstate, *site_2)?;
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
        Ok(ComplexSpatialField {
            bin_edges,
            bounds: [r_min, r_max],
            field_value: Array::from_vec(directors),
            n_bins: nbins,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use hoomd_meshless_voronoi::GenerateNeighborList;
    use hoomd_microstate::{Body, boundary::Open};
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
        let hexatic_0 = nlist.hexatic_at_site(&microstate, microstate.site_indices()[0])?;
        assert_relative_eq!(1.0, hexatic_0.re, epsilon = 1e-12);
        assert_relative_eq!(0.0, hexatic_0.im, epsilon = 1e-12);
        Ok(())
    }
}

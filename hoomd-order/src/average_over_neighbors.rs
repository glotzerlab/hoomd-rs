// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `average_over_neighbors`

use std::{
    collections::HashMap,
    ops::{AddAssign, Div},
};

use crate::Error;

/// Average a quantity over the neighbors around a site.
pub trait AverageOverNeighbors {
    /// Type of the generated container.
    type Output;

    /// Average a quantity over the neighbors around a site.
    ///
    /// The average quantity is:
    /// ```math
    /// \bar{x}_i = \frac{1}{n_i} \sum_j x_j
    /// ```
    /// where $` n_i `$ is the number of *j* tags produced by the iterator `neighbors(i)`.
    /// The value $` x_j `$ comes from the container `self`.
    ///
    /// When the container is a *slice*, the tag *j* is given by the index of the element in the
    /// slice: `x_j = self[j]`. When the container is a [`HashMap`], $` x_j `$ is the *value* with the
    /// key *j*: `x_j = self.get(&j)`. Use slices (e.g. [`Vec`]) for convenience only when you are
    /// certain that no sites will ever be deleted. Use [`HashMap`] when sites might be deleted,
    /// leading to a sparse value map.
    ///
    /// Use neighbor queries like [`SitesInBall::near_point`] to include the site itself in the
    /// average. Or use queries like [`SitesInBall::near_site`] to average over only the neighbors
    /// while excluding the *i* site.
    ///
    /// [`SitesInBall::near_point`]: crate::SitesInBall::near_point
    /// [`SitesInBall::near_site`]: crate::SitesInBall::near_site
    ///
    /// # Examples
    ///
    /// Open the collapsed sections below for examples.
    fn average_over_neighbors<F: Fn(usize) -> I, I: IntoIterator<Item = usize>>(
        &self,
        neighbor_tags: F,
    ) -> Self::Output;
}

impl<T, S> AverageOverNeighbors for HashMap<usize, T, S>
where
    T: AddAssign + Copy + Default + Div<f64, Output = T>,
    S: ::std::hash::BuildHasher + Default,
{
    type Output = Result<HashMap<usize, T, S>, Error>;

    /// Average a quantity over the neighbors around a site.
    ///
    /// Values are given in the [`HashMap`] `self` where the keys are the tags.
    ///
    /// # Errors
    ///
    /// [`Error::SiteTagMissing`] when any *j* produced by a `neighbors` iterator is not present
    /// as a key in `x`.
    ///
    /// # Example
    ///
    /// Estimate the position of a lattice site as the average of the nearest neighbor positions:
    /// ```
    /// use std::collections::HashMap;
    ///
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::{Body, Microstate, Replicate, boundary::Periodic};
    /// use hoomd_order::{AverageOverNeighbors, SitesInBall};
    /// use hoomd_spatial::VecCell;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model_maximum_interaction_range: f64 = 1.0;
    /// let order_maximum_neighbor_distance: f64 = 1.5;
    /// let maximum_interaction_range =
    ///     model_maximum_interaction_range.max(order_maximum_neighbor_distance);
    ///
    /// let unit_cell_square = Rectangle::with_equal_edges(1.0.try_into()?);
    /// let periodic_unit_cell = Periodic::new(0.0, unit_cell_square)?;
    /// let vec_cell = VecCell::builder()
    ///     .nominal_search_radius(model_maximum_interaction_range.try_into()?)
    ///     .maximum_search_radius(maximum_interaction_range)
    ///     .build();
    /// let microstate = Microstate::builder()
    ///     .boundary(periodic_unit_cell)
    ///     .spatial_data(vec_cell)
    ///     .bodies([Body::point(Cartesian::default())])
    ///     .try_build()?
    ///     .replicate_with_maximum_interaction_range(
    ///         [16; 2],
    ///         maximum_interaction_range,
    ///     )?;
    ///
    /// let mut r: HashMap<usize, Cartesian<2>> = microstate
    ///     .sites()
    ///     .iter()
    ///     .map(|s| (s.site_tag, s.properties.position))
    ///     .collect();
    /// let r_average = r.average_over_neighbors(|site_tag| {
    ///     SitesInBall::near_site(
    ///         &microstate,
    ///         microstate.site_indices()[site_tag]
    ///             .expect("site tag should be present"),
    ///         order_maximum_neighbor_distance,
    ///     )
    ///     .iter_site_tags()
    ///     .collect::<Vec<_>>()
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn average_over_neighbors<F: Fn(usize) -> I, I: IntoIterator<Item = usize>>(
        &self,
        neighbor_tags: F,
    ) -> Result<HashMap<usize, T, S>, Error> {
        let mut result = HashMap::with_capacity_and_hasher(self.len(), S::default());

        for site_tag_i in self.keys() {
            let mut total = T::default();
            let mut count = 0;

            for site_tag_j in neighbor_tags(*site_tag_i) {
                total += *self
                    .get(&site_tag_j)
                    .ok_or(Error::SiteTagMissing(site_tag_j, *site_tag_i))?;
                count += 1;
            }

            result.insert(*site_tag_i, total / f64::from(count));
        }

        Ok(result)
    }
}

impl<T> AverageOverNeighbors for &[T]
where
    T: AddAssign + Copy + Default + Div<f64, Output = T>,
{
    type Output = Vec<T>;

    /// Average a quantity over the neighbors around a site.
    ///
    /// Values are given in the slice `self` where the keys are the the indices.
    ///
    /// # Example
    ///
    /// Estimate the position of a lattice site as the average of the nearest neighbor positions
    /// (assumes that all site tags 0..N exist in the microstate):
    /// ```
    /// use std::collections::HashMap;
    ///
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::{Body, Microstate, Replicate, boundary::Periodic};
    /// use hoomd_order::{AverageOverNeighbors, SitesInBall};
    /// use hoomd_spatial::VecCell;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let model_maximum_interaction_range: f64 = 1.0;
    /// let order_maximum_neighbor_distance: f64 = 1.5;
    /// let maximum_interaction_range =
    ///     model_maximum_interaction_range.max(order_maximum_neighbor_distance);
    ///
    /// let unit_cell_square = Rectangle::with_equal_edges(1.0.try_into()?);
    /// let periodic_unit_cell = Periodic::new(0.0, unit_cell_square)?;
    /// let vec_cell = VecCell::builder()
    ///     .nominal_search_radius(model_maximum_interaction_range.try_into()?)
    ///     .maximum_search_radius(maximum_interaction_range)
    ///     .build();
    /// let microstate = Microstate::builder()
    ///     .boundary(periodic_unit_cell)
    ///     .spatial_data(vec_cell)
    ///     .bodies([Body::point(Cartesian::default())])
    ///     .try_build()?
    ///     .replicate_with_maximum_interaction_range(
    ///         [16; 2],
    ///         maximum_interaction_range,
    ///     )?;
    ///
    /// let mut r: Vec<Cartesian<2>> = microstate
    ///     .iter_sites_tag_order()
    ///     .map(|s| (s.properties.position))
    ///     .collect();
    /// let r_average = r.as_slice().average_over_neighbors(|site_tag| {
    ///     SitesInBall::near_site(
    ///         &microstate,
    ///         microstate.site_indices()[site_tag]
    ///             .expect("site tag should be present"),
    ///         order_maximum_neighbor_distance,
    ///     )
    ///     .iter_site_tags()
    ///     .collect::<Vec<_>>()
    /// });
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn average_over_neighbors<F: Fn(usize) -> I, I: IntoIterator<Item = usize>>(
        &self,
        neighbor_tags: F,
    ) -> Vec<T> {
        let mut result = Vec::with_capacity(self.len());

        for site_tag_i in 0..self.len() {
            let mut total = T::default();
            let mut count = 0;

            for site_tag_j in neighbor_tags(site_tag_i) {
                total += self[site_tag_j];
                count += 1;
            }

            result.push(total / f64::from(count));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;

    #[test]
    fn base_case_hash_map() -> anyhow::Result<()> {
        let neighbors: [Vec<usize>; 4] = [vec![1, 2, 3], vec![0], vec![0, 3], vec![0, 2]];
        let x: HashMap<usize, f64> = [(0, 1.0), (1, 3.0), (2, 2.0), (3, 4.0)]
            .into_iter()
            .collect();

        let x_average = x.average_over_neighbors(|tag| neighbors[tag].iter().copied())?;
        assert_eq!(x_average.len(), 4);
        assert_relative_eq!(x_average.get(&0).unwrap(), &3.0);
        assert_relative_eq!(x_average.get(&1).unwrap(), &1.0);
        assert_relative_eq!(x_average.get(&2).unwrap(), &2.5);
        assert_relative_eq!(x_average.get(&3).unwrap(), &1.5);

        Ok(())
    }

    #[test]
    fn base_case_vec() {
        let neighbors: [Vec<usize>; 4] = [vec![1, 2, 3], vec![0], vec![0, 3], vec![0, 2]];
        let x: Vec<f64> = vec![1.0, 3.0, 2.0, 4.0];

        let x_average = x
            .as_slice()
            .average_over_neighbors(|tag| neighbors[tag].iter().copied());
        assert_eq!(x_average.len(), 4);
        assert_relative_eq!(x_average[0], 3.0);
        assert_relative_eq!(x_average[1], 1.0);
        assert_relative_eq!(x_average[2], 2.5);
        assert_relative_eq!(x_average[3], 1.5);
    }

    #[test]
    fn missing_neighbor() {
        let neighbors: [Vec<usize>; 1] = [vec![1, 2, 3]];
        let x: HashMap<usize, f64> = [(0, 1.0)].into_iter().collect();

        assert!(matches!(
            x.average_over_neighbors(|tag| neighbors[tag].iter().copied()),
            Err(Error::SiteTagMissing(1, 0))
        ));
    }
}

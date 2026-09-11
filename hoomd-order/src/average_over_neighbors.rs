// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `average_over_neighbors`

use std::{
    collections::HashMap,
    ops::{AddAssign, Div},
};

use crate::Error;

/// Average a quantity over the neighbors around a site.
///
/// The average quantity is:
/// ```math
/// \bar{x}_i = \frac{1}{n_i} \sum_j x_j
/// ```
/// where $` n_i `$ is the number of *j* tags produced by the iterator `neighbors(i)`.
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
/// use hoomd_order::SitesInBall;
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
/// let r_average = hoomd_order::average_over_neighbors(&r, |site_tag| {
///     SitesInBall::near_site(
///         &microstate,
///         microstate.site_indices()[site_tag].expect("site tag should be present"),
///         order_maximum_neighbor_distance,
///     )
///     .iter_site_tags()
///     .collect::<Vec<_>>()
/// })?;
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn average_over_neighbors<T, F, I, S>(
    x: &HashMap<usize, T, S>,
    neighbor_tags: F,
) -> Result<HashMap<usize, T, S>, Error>
where
    T: AddAssign + Copy + Default + Div<f64, Output = T>,
    F: Fn(usize) -> I,
    I: IntoIterator<Item = usize>,
    S: ::std::hash::BuildHasher + Default,
{
    let mut result = HashMap::with_capacity_and_hasher(x.len(), S::default());

    for site_tag_i in x.keys() {
        let mut total = T::default();
        let mut count = 0;

        for site_tag_j in neighbor_tags(*site_tag_i) {
            total += *x
                .get(&site_tag_j)
                .ok_or(Error::SiteTagMissing(site_tag_j, *site_tag_i))?;
            count += 1;
        }

        result.insert(*site_tag_i, total / f64::from(count));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;

    #[test]
    fn base_case() -> anyhow::Result<()> {
        let neighbors: [Vec<usize>; 4] = [vec![1, 2, 3], vec![0], vec![0, 3], vec![0, 2]];
        let x: HashMap<usize, f64> = [(0, 1.0), (1, 3.0), (2, 2.0), (3, 4.0)]
            .into_iter()
            .collect();

        let x_average = average_over_neighbors(&x, |tag| neighbors[tag].iter().copied())?;
        assert_eq!(x_average.len(), 4);
        assert_relative_eq!(x_average.get(&0).unwrap(), &3.0);
        assert_relative_eq!(x_average.get(&1).unwrap(), &1.0);
        assert_relative_eq!(x_average.get(&2).unwrap(), &2.5);
        assert_relative_eq!(x_average.get(&3).unwrap(), &1.5);

        Ok(())
    }

    #[test]
    fn missing_neighbor() {
        let neighbors: [Vec<usize>; 1] = [vec![1, 2, 3]];
        let x: HashMap<usize, f64> = [(0, 1.0)].into_iter().collect();

        assert!(matches!(
            average_over_neighbors(&x, |tag| neighbors[tag].iter().copied()),
            Err(Error::SiteTagMissing(1, 0))
        ));
    }
}

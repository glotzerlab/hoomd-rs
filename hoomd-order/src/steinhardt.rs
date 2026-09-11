// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Steinhardt`

use std::f64::consts::PI;

use hoomd_vector::{Cartesian, InnerProduct};

use crate::{
    Error,
    math::{SphericalHarmonic, SphericalHarmonicOutputs},
};

/// Compute the 3D Steinhardt order parameters $` q_{Lm} `$ and $` q_L `$.
///
/// [Steinhardt, Nelson, and Ronchetti 1983](https://doi.org/10.1103/PhysRevB.28.784)
/// introduces the order parameter $` q_{Lm} `$ and its rotationally invariant counterpart,
/// $` q_L `$.
///
/// Let the `neighbors` iterator produce `n` points $` \vec{r}_j `$.
/// The order parameters are then:
/// ```math
/// \begin{align*}
/// q_{Lm} &= \frac{1}{n} \sum_j Y_L^m \left( \frac{\vec{r}_j - \vec{r}}{\lvert \vec{r}_j - \vec{r} \rvert} \right) \\
/// q_L &= \sqrt{\frac{4 \pi}{2L + 1} \sum_{m=-L}^L \lvert q_{Lm} \rvert^2} \\
/// \end{align*}
/// ```
///
/// Construct a [`Steinhardt`] with a given `L`, then call [`q_l`] or [`q_lm`] to evaluate
/// $` q_L `$ or $` q_{Lm} `$ respectively. Use the same [`Steinhardt`] for may calls to
/// [`q_l`] and/or [`q_lm`] as [`new`] is computationally expensive.
///
/// [`q_l`]: Self::q_l
/// [`q_lm`]: Self::q_lm
/// [`new`]: Self::new
///
/// # Examples
///
/// Compute $` q_4 `$ for all sites in a microstate:
/// ```
/// use std::collections::HashMap;
///
/// use hoomd_geometry::shape::Cuboid;
/// use hoomd_microstate::{Body, Microstate, Replicate, boundary::Periodic};
/// use hoomd_order::{SitesInBall, Steinhardt};
/// use hoomd_spatial::VecCell;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model_maximum_interaction_range: f64 = 1.0;
/// let order_maximum_neighbor_distance: f64 = 1.5;
/// let maximum_interaction_range =
///     model_maximum_interaction_range.max(order_maximum_neighbor_distance);
///
/// let unit_cell_cube = Cuboid::with_equal_edges(1.0.try_into()?);
/// let periodic_unit_cell = Periodic::new(0.0, unit_cell_cube)?;
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
///         [16; 3],
///         maximum_interaction_range,
///     )?;
///
/// let steinhardt = Steinhardt::<4>::new();
/// let mut q_4 = HashMap::new();
/// for (site_index, site) in microstate.sites().iter().enumerate() {
///     let neighbors = SitesInBall::near_site(
///         &microstate,
///         site_index,
///         order_maximum_neighbor_distance,
///     );
///     q_4.insert(
///         site.site_tag,
///         steinhardt.q_l(
///             &site.properties.position,
///             neighbors.iter_site_positions().copied(),
///         )?,
///     );
/// }
/// # Ok(())
/// # }
/// ```
///
/// Compute the averaged $` q_4 `$ for all sites in a microstate using
/// [`average_over_neighbors`]. First, compute the  $` q_{Lm} `$ values for each
/// site. Then average them over the site *and* all its neighbors (by using a
/// `near_point` query). Finally, compute $` q_4 `$ by applying the rotational
/// invariance operation to the averaged $` q_{Lm} `$:
/// ```
/// use std::collections::HashMap;
///
/// use hoomd_geometry::shape::Cuboid;
/// use hoomd_microstate::{Body, Microstate, Replicate, boundary::Periodic};
/// use hoomd_order::{SitesInBall, Steinhardt};
/// use hoomd_spatial::VecCell;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model_maximum_interaction_range: f64 = 1.0;
/// let order_maximum_neighbor_distance: f64 = 1.5;
/// let maximum_interaction_range =
///     model_maximum_interaction_range.max(order_maximum_neighbor_distance);
///
/// let unit_cell_cube = Cuboid::with_equal_edges(1.0.try_into()?);
/// let periodic_unit_cell = Periodic::new(0.0, unit_cell_cube)?;
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
///         [16; 3],
///         maximum_interaction_range,
///     )?;
///
/// let steinhardt = Steinhardt::<4>::new();
/// let mut q_lm = HashMap::new();
/// for (site_index, site) in microstate.sites().iter().enumerate() {
///     let neighbors = SitesInBall::near_site(
///         &microstate,
///         site_index,
///         order_maximum_neighbor_distance,
///     );
///     q_lm.insert(
///         site.site_tag,
///         steinhardt.q_lm(
///             &site.properties.position,
///             neighbors.iter_site_positions().copied(),
///         )?,
///     );
/// }
///
/// let q_average_lm = hoomd_order::average_over_neighbors(&q_lm, |site_tag| {
///    let site_index = microstate.site_indices()[site_tag].expect("site tag should be present");
///    SitesInBall::near_point(&microstate,
///    microstate.sites()[site_index].properties.position,
///    order_maximum_neighbor_distance,
///    )
///    .iter_site_tags()
///    .collect::<Vec<_>>()
/// })?;
///
/// let q_average_4: HashMap<usize, f64> = q_average_lm.iter().map(|(&site_tag, q_lm_j)|
///    (site_tag, Steinhardt::make_rotationally_invariant(q_lm_j))).collect();
/// # Ok(())
/// # }
/// ```
///
/// [`average_over_neighbors`]: crate::average_over_neighbors
#[derive(Copy, Clone, Debug, Default)]
pub struct Steinhardt<const L: usize> {
    /// Compute the spherical harmonics.
    spherical_harmonic: SphericalHarmonic<L>,
}

impl<const L: usize> Steinhardt<L> {
    /// Construct a new Steinhardt order parameter evaluator with degree `L`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_order::Steinhardt;
    ///
    /// let steinhardt = Steinhardt::<6>::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            spherical_harmonic: SphericalHarmonic::new(),
        }
    }

    /// Compute $` q_{Lm} `$.
    ///
    /// ```math
    /// q_{Lm} = \frac{1}{n} \sum_j Y_L^m \left( \frac{\vec{r}_j - \vec{r}}{\lvert \vec{r}_j - \vec{r} \rvert} \right)
    /// ```
    /// where the `neighbors` iterator produces `n` points $` \vec{r}_j `$.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidDeltaR3`] when any $` |\vec{r}_j - \vec{r}| = 0 `$.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_order::Steinhardt;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let steinhardt = Steinhardt::<6>::new();
    /// let fcc_bonds: Vec<Cartesian<3>> = [
    ///     [-1.0, -1.0, 0.0],
    ///     [-1.0, 1.0, 0.0],
    ///     [1.0, -1.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [-1.0, 0.0, -1.0],
    ///     [-1.0, 0.0, 1.0],
    ///     [1.0, 0.0, -1.0],
    ///     [1.0, 0.0, 1.0],
    ///     [0.0, -1.0, -1.0],
    ///     [0.0, -1.0, 1.0],
    ///     [0.0, 1.0, -1.0],
    ///     [0.0, 1.0, 1.0],
    /// ]
    /// .map(Cartesian::<3>::from)
    /// .to_vec();
    ///
    /// let q_lm = steinhardt.q_lm(&Cartesian::default(), fcc_bonds)?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn q_lm<I: IntoIterator<Item = Cartesian<3>>>(
        &self,
        r: &Cartesian<3>,
        neighbors: I,
    ) -> Result<SphericalHarmonicOutputs<L>, Error> {
        let mut total = SphericalHarmonicOutputs::default();
        let mut count: usize = 0;

        for r_j in neighbors {
            let (delta_r_unit, _) = (r_j - *r)
                .to_unit()
                .map_err(|e| Error::InvalidDeltaR3(r_j, *r, e))?;
            total += self.spherical_harmonic.evaluate(&delta_r_unit);
            count += 1;
        }

        Ok(total / (count as f64))
    }

    /// Compute $` q_L `$ given $` q_{Lm} `$.
    ///
    /// ```math
    /// q_L = \sqrt{\frac{4 \pi}{2L + 1} \sum_{m=-L}^L \lvert q_{Lm} \rvert^2}
    /// ```
    #[inline]
    pub fn make_rotationally_invariant(q_lm: &SphericalHarmonicOutputs<L>) -> f64 {
        let mut sum_squared = q_lm[0].norm_sqr();
        for i in 1..=L {
            sum_squared += 2.0 * q_lm[i].norm_sqr();
        }
        (4.0 * PI / (2 * L + 1) as f64 * sum_squared).sqrt()
    }

    /// Compute $` q_L `$ given $` \vec{r} `$ and a set of neighbors $` \vec{r}_j `$.
    ///
    /// ```math
    /// q_L = \sqrt{\frac{4 \pi}{2L + 1} \sum_{m=-L}^L \lvert q_{Lm} \rvert^2}
    /// ```
    /// where
    /// ```math
    /// q_{Lm} = \frac{1}{n} \sum_j Y_L^m \left( \frac{\vec{r}_j - \vec{r}}{\lvert \vec{r}_j - \vec{r} \rvert} \right)
    /// ```
    /// and the `neighbors` iterator produces `n` points $` \vec{r}_j `$.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidDeltaR3`] when any $` |\vec{r}_j - \vec{r}| = 0 `$.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    ///
    /// use hoomd_order::Steinhardt;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let steinhardt = Steinhardt::<6>::new();
    /// let fcc_bonds: Vec<Cartesian<3>> = [
    ///     [-1.0, -1.0, 0.0],
    ///     [-1.0, 1.0, 0.0],
    ///     [1.0, -1.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [-1.0, 0.0, -1.0],
    ///     [-1.0, 0.0, 1.0],
    ///     [1.0, 0.0, -1.0],
    ///     [1.0, 0.0, 1.0],
    ///     [0.0, -1.0, -1.0],
    ///     [0.0, -1.0, 1.0],
    ///     [0.0, 1.0, -1.0],
    ///     [0.0, 1.0, 1.0],
    /// ]
    /// .map(Cartesian::<3>::from)
    /// .to_vec();
    ///
    /// let q_l = steinhardt.q_l(&Cartesian::default(), fcc_bonds)?;
    /// assert_relative_eq!(q_l, 0.57452416, epsilon = 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn q_l<I: IntoIterator<Item = Cartesian<3>>>(
        &self,
        r: &Cartesian<3>,
        neighbors: I,
    ) -> Result<f64, Error> {
        let q_lm = self.q_lm(r, neighbors)?;
        Ok(Self::make_rotationally_invariant(&q_lm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_delta_r() {
        let steinhardt = Steinhardt::<6>::new();

        let neighbors = [[1.0, 1.0, 1.0].into()];

        assert!(matches!(
            steinhardt.q_lm(&[1.0, 1.0, 1.0].into(), neighbors),
            Err(Error::InvalidDeltaR3(_, _, _))
        ));
    }
}

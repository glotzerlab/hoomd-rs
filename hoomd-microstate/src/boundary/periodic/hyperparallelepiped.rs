// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for hyperparallelepipeds in cartesian space.

use std::array;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use arrayvec::ArrayVec;
use hoomd_geometry::{IsPointInside, shape::Hyperparallelepiped};
use hoomd_linear_algebra::{
    MatMul,
    matrix::{Matrix, qr},
};
use hoomd_vector::Cartesian;

impl<const N: usize> MaximumAllowableInteractionRange for Hyperparallelepiped<N> {
    /// The largest value that the maximum interaction range can take. While theoretically to avoid self-interaction the interaction distance may be as large as 1/2 the smallest box vector, we choose to take the maximum interaction range to be 1/2 the smallest perpendicular distance between pairs of parallel faces in order to avoid having to generating more than one ghost per particle.
    ///
    /// # Example
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        let plane_distances = self.get_nearest_plane_distance();
        plane_distances
            .iter()
            .map(|x| x.get() * 0.5)
            .fold(f64::INFINITY, f64::min)
    }
}

impl<P, const N: usize> Wrap<P> for Periodic<Hyperparallelepiped<N>>
where
    P: Position<Position = Cartesian<N>>,
{
    /// Wrap any cartesian vector to the inside of the given hyperparallepiped.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Parallelepiped;
    /// use hoomd_microstate::{
    ///     boundary::{Periodic, Wrap},
    ///     property::Point,
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let box_ = Parallelepiped {
    ///     edge_vectors: [
    ///         [1.0, 0.0, 0.0].into(),
    ///         [0.5, f64::sqrt(3.0) / 2.0, 0.0].into(),
    ///         [0.0, 0.0, 1.0].into(),
    ///     ],
    ///     _qr: None,
    /// };
    /// let periodic = Periodic::new(0.25, box_)?;
    /// let point = Point::new(Cartesian::from([1.0, f64::sqrt(3.0), 2.5]));
    /// let wrapped_point = periodic.wrap(point)?;
    /// assert_eq!(wrapped_point.position, [0.0, 0.0, -0.5].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn wrap(&self, properties: P) -> Result<P, Error> {
        let mut properties = properties;
        let r = properties.position_mut();

        let a = Matrix::<N, N> {
            rows: self.shape.edge_vectors.map(|v| v.coordinates),
        }
        .transpose();

        let fractional = qr::qr_solve(&a, r.to_column_matrix());

        let position_offset = a.matmul(&fractional.map_elements(f64::round));
        *r -= Cartesian::from_col_matrix(position_offset);

        Ok(properties)
    }
}

impl<S, const N: usize> GenerateGhosts<S> for Periodic<Hyperparallelepiped<N>>
where
    S: Position<Position = Cartesian<N>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    // /// Place periodic images of sites near the edge of the periodic boundary.
    #[inline]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        let mut result = ArrayVec::new();

        let r = site_properties.position();

        if !self.shape.is_point_inside(r) {
            return result;
        }

        // Determine fractional coordinates of "twighlight zones," where ghosts must be generated
        let plane_distances = self.shape.get_nearest_plane_distance();
        let fractional_cutoffs: [f64; N] =
            array::from_fn(|i| self.maximum_interaction_range() / plane_distances[i].get());
        let fractional_coordinate = self.shape.to_fractional(*r);

        for (i, fractional_cutoff) in fractional_cutoffs.iter().enumerate() {
            if fractional_coordinate[i] <= -0.5 + fractional_cutoff {
                let mut new_site = *site_properties;
                *new_site.position_mut() += self.shape.edge_vectors[i];
                result.push(new_site)
            } else if fractional_coordinate[i] > 0.5 - fractional_cutoff {
                let mut new_site = *site_properties;
                *new_site.position_mut() -= self.shape.edge_vectors[0];
                result.push(new_site)
            }
        }

        result
    }
}

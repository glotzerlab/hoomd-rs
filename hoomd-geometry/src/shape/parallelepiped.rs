// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

// use crate::{BoundingSphereRadius, SupportMapping, Volume};
use hoomd_vector::{Cartesian, InnerProduct};

use crate::SupportMapping;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hyperparallelepiped<const N: usize> {
    /// The lengths of each edge of the cuboid.
    pub edge_vectors: [Cartesian<N>; N],
}

pub type Parallelogram = Hyperparallelepiped<2>;
pub type Parallelepiped = Hyperparallelepiped<3>;

impl<const N: usize> Default for Hyperparallelepiped<N> {
    #[inline]
    fn default() -> Self {
        Self {
            edge_vectors: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { 1. } else { 0. }).into()
            }),
        }
    }
}

impl<const N: usize> Hyperparallelepiped<N> {
    #[inline]
    #[must_use]
    /// Determine the maximal extents of the cuboid along each Cartesian axis.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hypercuboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let unit_cube = Hypercuboid {
    ///     edge_lengths: [1.0.try_into()?; 3],
    /// };
    ///
    /// let max_extents = unit_cube.maximal_extents();
    /// assert_eq!(max_extents, [0.5; 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn maximal_extents(&self) -> [f64; N] {
        (0.5 * self
            .edge_vectors
            .iter()
            .fold(Cartesian::<N>::default(), |acc, v| v.map(f64::abs) + acc))
        .into()
    }

    #[inline]
    #[must_use]
    /// Determine the minimal extents of the cuboid along each Cartesian axis.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hypercuboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let unit_cube = Hypercuboid {
    ///     edge_lengths: [1.0.try_into()?; 3],
    /// };
    ///
    /// let min_extents = unit_cube.minimal_extents();
    /// assert_eq!(min_extents, [-0.5; 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn minimal_extents(&self) -> [f64; N] {
        self.maximal_extents().map(|x| -x)
    }
}

impl<const N: usize> SupportMapping<Cartesian<N>> for Hyperparallelepiped<N> {
    #[inline]
    fn support_mapping(&self, direction: &Cartesian<N>) -> Cartesian<N> {
        0.5 * self
            .edge_vectors
            .iter()
            .fold(Cartesian::<N>::default(), |acc, v| {
                v.dot(direction).signum() * *v + acc
            })
    }
}

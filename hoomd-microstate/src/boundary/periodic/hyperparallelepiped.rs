// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for cuboids in cartesian space.

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use arrayvec::ArrayVec;
use hoomd_geometry::{IsPointInside, shape::Hyperparallelepiped};
use hoomd_linear_algebra::{MatMul, matrix::Matrix, matrix::Matrix33, matrix::qr};
use hoomd_vector::{Cartesian, Cross, InnerProduct};

impl<const N: usize> MaximumAllowableInteractionRange for Hyperparallelepiped<N> {
    /// The largest value that the maximum interaction range can take.
    ///
    /// For a parallelepiped, the maximum is
    /// ```math
    /// \frac{L_\mathrm{min}}{2}
    /// ```
    /// where $`L_\mathrm{min}`$ is the smallest edge length.
    ///
    /// # Example
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        let minimum_l = self
            .edge_vectors
            .iter()
            .map(Cartesian::<N>::norm)
            .reduce(f64::min)
            .expect("parallelipiped should have dimension 1 or greater");
        minimum_l / 2.0
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

impl<S> GenerateGhosts<S> for Periodic<Hyperparallelepiped<3>>
where
    S: Position<Position = Cartesian<3>> + Copy + Default,
{
    // #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    // /// Place periodic images of sites near the edge of the periodic boundary.
    // #[inline]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        // let mut result = ArrayVec::new();

        // let r = site_properties.position();
        // let max = self.shape.maximal_extents();
        // let min = self.shape.minimal_extents();

        // if !self.shape.is_point_inside(r) {
        //     return result;
        // }

        // let new_site = |x, y, z| {
        //     let mut new_site = *site_properties;
        //     new_site.position_mut()[0] += x * self.shape.edge_vectors[0];
        //     new_site.position_mut()[1] += y * self.shape.edge_vectors[1];
        //     new_site.position_mut()[2] += z * self.shape.edge_vectors[2];
        //     new_site
        // };
        // Find which boundaries particle is near.
        todo!();
    }
}

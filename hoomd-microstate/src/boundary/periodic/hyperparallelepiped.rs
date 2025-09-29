// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for cuboids in cartesian space.

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use hoomd_geometry::{IsPointInside, shape::Hyperparallelepiped};
use hoomd_linalg::{MatMul, matrix::Matrix33};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::InnerProduct;
use hoomd_vector::{Cartesian, Cross};

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
    ///
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

impl<P> Wrap<P> for Periodic<Hyperparallelepiped<3>>
where
    P: Position<Vector = Cartesian<3>>,
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
    /// let box_ = Parallelepiped{edge_vectors: [[1.0,0.0,0.0].into(),[0.5,f64::sqrt(3.0)/2.0,0.0].into(), [0.0,0.0,1.0].into()]};
    /// let periodic =
    ///     Periodic::new(0.25, box_)?;
    /// let point = Point::new(Cartesian::from([6.0, -15.0, 2.5]));
    /// println!("{}",point.position);
    /// let wrapped_point = periodic.wrap(point)?;
    /// assert_eq!(wrapped_point.position, [0.0, 0.67949192, 0.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn wrap(&self, properties: P) -> Result<P, Error> {
        let mut properties = properties;
        let r = properties.position_mut();

        let from_fractional = Matrix33 {
            rows: self.shape.edge_vectors.map(|v| v.coordinates),
        }
        .transpose();

        let to_fractional = Matrix33 {
            rows: [
                self.shape.edge_vectors[1]
                    .cross(&self.shape.edge_vectors[2])
                    .coordinates,
                self.shape.edge_vectors[2]
                    .cross(&self.shape.edge_vectors[0])
                    .coordinates,
                self.shape.edge_vectors[0]
                    .cross(&self.shape.edge_vectors[1])
                    .coordinates,
            ],
        } * from_fractional.det().recip();

        let box_offset = to_fractional.matmul(&r.to_column_matrix()).map(f64::round);

        *r -= from_fractional.matmul(&box_offset).into();

        Ok(properties)
    }
}
